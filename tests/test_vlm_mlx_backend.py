"""Real subprocess/pipe protocol tests; fake worker never imports MLX or loads weights."""
import json
import math
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import MappingProxyType

import pytest

from ml.vlm.contracts import Frame, InferenceRequest
from ml.vlm import mlx_backend as module
from ml.vlm.preflight import ModelUnavailable


FAKE_WORKER = r'''
import json, os, sys, time
mode = sys.argv[1]
if mode == "startup_hang":
    time.sleep(10)
if mode == "startup_bad":
    print('{"status":"ready","extra":true}', flush=True)
else:
    print('{"status":"ready"}', flush=True)
if mode == "exit":
    raise SystemExit(0)
for line in sys.stdin:
    inference_request = json.loads(line)
    if mode == "hang":
        time.sleep(10)
    if mode == "slow":
        time.sleep(0.15)
    if mode == "eof":
        raise SystemExit(0)
    if mode == "deep":
        os.write(1, b'{"id":1,"status":"ok","text":' + b"[" * 2000 + b"0" + b"]" * 2000 + b"}\n")
        continue
    if mode == "overlong":
        print("x" * 65537, flush=True)
        continue
    if mode == "duplicate":
        print('{"id":1,"id":1,"status":"ok","text":"unsafe"}', flush=True)
        continue
    if mode == "nan":
        print('{"id":1,"status":"ok","text":"unsafe","extra":NaN}', flush=True)
        continue
    reply = {"id": inference_request["id"], "status": "ok", "text": "synthetic"}
    if mode == "bool_id":
        reply["id"] = True
    if mode == "wrong_id":
        reply["id"] += 1
    if mode == "unknown":
        reply["extra"] = "unsafe"
    if mode == "bad_text":
        reply["text"] = {"not": "text"}
    if mode == "wire":
        reply["text"] = json.dumps(inference_request)
    if mode == "offline":
        reply["text"] = json.dumps({k: os.environ.get(k) for k in [
            "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY", "DO_NOT_TRACK"]})
    payload = (json.dumps(reply) + "\n").encode()
    if mode == "chunked":
        for i in range(0, len(payload), 3):
            os.write(1, payload[i:i+3])
    else:
        os.write(1, payload)
'''


@pytest.fixture
def backend_factory(tmp_path, monkeypatch):
    model = tmp_path / 'model'
    model.mkdir()
    for name, value in [('config.json', {'model_type': 'qwen3_vl'}),
                        ('preprocessor_config.json', {}), ('tokenizer_config.json', {}),
                        ('tokenizer.json', {})]:
        (model / name).write_text(json.dumps(value))
    (model / 'model.safetensors').write_bytes(b'SYNTHETIC STRUCTURAL FIXTURE; NEVER LOADED')
    worker = tmp_path / 'worker.py'
    worker.write_text(FAKE_WORKER)
    resources = []

    def make(mode='echo'):
        backend = module.LocalMLXBackend(model, python_executable=sys.executable)
        monkeypatch.setattr(backend, '_command', lambda directory: [sys.executable, '-u', str(worker), mode])
        resources.append(backend)
        return backend

    yield make
    for backend in resources:
        backend.close()


@pytest.fixture
def inference_request(tmp_path):
    frame = tmp_path / 'frame.png'
    frame.write_bytes(b'SYNTHETIC FRAME PATH ONLY; NEVER DECODED')
    return InferenceRequest('synthetic prompt', (Frame(1.25, frame, 1.25),), 'candidate', 'cv-1', ('r1',))


@pytest.mark.parametrize('mode', ['echo', 'chunked', 'offline'])
def test_protocol_worker_is_warm_and_enforced_offline(backend_factory, inference_request, monkeypatch, mode):
    monkeypatch.setenv('HF_HUB_OFFLINE', '0')
    backend = backend_factory(mode)
    backend.validate(2)
    process = backend._process
    backend.validate(2)
    for _ in range(2):
        result = backend.infer(inference_request, 2)
        if mode == 'offline':
            assert set(json.loads(result).values()) == {'1'}
        else:
            assert result == 'synthetic'
        assert backend._process is process
    backend.close()
    assert process.poll() is not None
    assert process.stdin.closed and process.stdout.closed
    with pytest.raises(ModelUnavailable, match='runtime_not_prepared'):
        backend.validate(2)


@pytest.mark.parametrize('mode', ['bool_id', 'wrong_id', 'duplicate', 'nan', 'unknown', 'bad_text', 'overlong', 'eof', 'deep'])
def test_invalid_protocol_closes_worker_without_restart(backend_factory, inference_request, mode):
    backend = backend_factory(mode)
    backend.validate(2)
    process = backend._process
    with pytest.raises((ModelUnavailable, ValueError)):
        backend.infer(inference_request, 2)
    assert backend._closed and backend._process is None
    assert process.poll() is not None
    assert process.stdin.closed and process.stdout.closed
    with pytest.raises(ModelUnavailable, match='runtime_not_prepared'):
        backend.validate(2)


@pytest.mark.parametrize('mode', ['startup_bad', 'startup_hang'])
def test_failed_startup_closes_and_never_retries(backend_factory, mode):
    backend = backend_factory(mode)
    with pytest.raises((ModelUnavailable, TimeoutError)):
        backend.validate(0.2)
    assert backend._closed and backend._process is None
    with pytest.raises(ModelUnavailable, match='runtime_not_prepared'):
        backend.validate(1)


def test_generation_timeout_kills_worker_and_closes_pipes(backend_factory, inference_request):
    backend = backend_factory('hang')
    backend.validate(2)
    process = backend._process
    with pytest.raises(TimeoutError, match='inference_timeout'):
        backend.infer(inference_request, 0.05)
    assert process.poll() is not None
    assert process.stdin.closed and process.stdout.closed
    assert backend._closed


def test_busy_callers_do_not_touch_active_worker(backend_factory, inference_request, monkeypatch):
    backend = backend_factory()
    backend.validate(2)
    entered, release = threading.Event(), threading.Event()
    original = backend._write

    def write(*args):
        original(*args)
        entered.set()
        assert release.wait(2)

    monkeypatch.setattr(backend, '_write', write)
    process = backend._process
    with ThreadPoolExecutor(max_workers=1) as pool:
        active = pool.submit(backend.infer, inference_request, 2)
        assert entered.wait(1)
        for call in (lambda: backend.infer(inference_request, 1), lambda: backend.validate(1), backend.close):
            with pytest.raises(ModelUnavailable, match='runtime_busy'):
                call()
        assert backend._process is process and process.poll() is None
        release.set()
        assert active.result(2) == 'synthetic'
    assert backend.infer(inference_request, 2) == 'synthetic'


def test_dead_worker_is_reaped_without_model_reload(backend_factory, inference_request):
    backend = backend_factory('exit')
    backend.validate(2)
    process = backend._process
    process.wait(timeout=2)
    with pytest.raises(ModelUnavailable, match='runtime_not_prepared'):
        backend.infer(inference_request, 1)
    assert backend._closed and backend._process is None
    assert process.stdin.closed and process.stdout.closed


@pytest.mark.parametrize('timestamp', [True, math.inf, math.nan, -1.0, '1.0'])
def test_invalid_frame_timestamps_never_reach_worker(backend_factory, inference_request, monkeypatch, timestamp):
    backend = backend_factory()
    backend.validate(2)
    bad = InferenceRequest(inference_request.prompt, (Frame(timestamp, inference_request.frames[0].path, timestamp),),
                           'candidate', 'cv-1', ('r1',))
    monkeypatch.setattr(backend, '_write', lambda *args: pytest.fail('invalid inference_request sent'))
    with pytest.raises(ValueError, match='local frames required'):
        backend.infer(bad, 1)


def test_request_size_is_bounded_before_json_encoding(backend_factory, inference_request, monkeypatch):
    backend = backend_factory()
    backend.validate(2)
    bad = InferenceRequest('x' * (module.MAX_WIRE_BYTES + 1), inference_request.frames, 'candidate', 'cv-1', ('r1',))
    monkeypatch.setattr(module.json, 'dumps', lambda *args, **kwargs: pytest.fail('oversize prompt encoded'))
    with pytest.raises(ValueError, match='request too large'):
        backend.infer(bad, 1)


def test_source_file_size_is_bounded_before_worker_decode(backend_factory, inference_request):
    backend = backend_factory()
    backend.validate(2)
    with inference_request.frames[0].path.open('wb') as stream:
        stream.truncate(module.MAX_FRAME_FILE_BYTES + 1)
    with pytest.raises(ValueError, match='local frames required'):
        backend.infer(inference_request, 1)


def test_preparation_cannot_start_worker_after_deadline(tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(module.time, 'monotonic', lambda: clock[0])

    def slow_check(path):
        clock[0] = 2.0
        return tmp_path

    monkeypatch.setattr(module, '_validated_mlx_model_directory', slow_check)
    monkeypatch.setattr(module.subprocess, 'Popen', lambda *args, **kwargs: pytest.fail('worker started after deadline'))
    backend = module.LocalMLXBackend(tmp_path, python_executable=sys.executable)
    with pytest.raises(TimeoutError, match='inference_timeout'):
        backend.validate(1)
    assert backend._closed and backend._process is None


def test_process_start_failure_marks_backend_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(module, '_validated_mlx_model_directory', lambda path: tmp_path)

    def fail_start(*args, **kwargs):
        raise OSError('synthetic start failure')

    monkeypatch.setattr(module.subprocess, 'Popen', fail_start)
    backend = module.LocalMLXBackend(tmp_path, python_executable=sys.executable)
    with pytest.raises(OSError, match='synthetic start failure'):
        backend.validate(1)
    assert backend._closed and backend._process is None


def test_optional_response_schema_roundtrips_without_changing_legacy_wire(backend_factory, inference_request):
    backend = backend_factory('wire')
    backend.validate(2)
    schema = {'type': 'object', 'properties': {'answer': {'enum': ['synthetic']}}}
    structured = replace(inference_request, response_schema=MappingProxyType(schema))
    for item in (inference_request, structured, inference_request):
        payload = json.loads(backend.infer(item, 2))
        assert payload['prompt'] == inference_request.prompt
        assert payload['frames'][0]['timestamp_s'] == 1.25
        if item.response_schema is None:
            assert set(payload) == {'id', 'prompt', 'frames'}
        else:
            assert set(payload) == {'id', 'prompt', 'frames', 'response_schema'}
            assert payload['response_schema'] == schema


@pytest.mark.parametrize('schema', [[], 'schema', True, {'description': 'x' * module.MAX_WIRE_BYTES}])
def test_bad_or_oversized_schema_never_reaches_worker(backend_factory, inference_request, monkeypatch, schema):
    backend = backend_factory()
    backend.validate(2)
    monkeypatch.setattr(backend, '_write', lambda *args: pytest.fail('invalid schema sent'))
    with pytest.raises(ValueError):
        backend.infer(replace(inference_request, response_schema=schema), 2)
    assert backend._ready and not backend._closed


FAKE_RUNTIME = r'''
import json, os, sys, types
from pathlib import Path
from types import SimpleNamespace
model_path, mode, side = sys.argv[1:]
core = types.ModuleType("mlx.core")
core.synchronize = lambda: None
core.clear_cache = lambda: None
mlx = types.ModuleType("mlx")
mlx.core = core
sys.modules["mlx"] = mlx
sys.modules["mlx.core"] = core
runtime = types.ModuleType("mlx_vlm")
prompt_utils = types.ModuleType("mlx_vlm.prompt_utils")
processor = SimpleNamespace(image_processor=SimpleNamespace(max_pixels=0, min_pixels=65536), tokenizer=object())
if mode == "processor_tokenizer": del processor.tokenizer
structured = types.ModuleType("mlx_vlm.structured")
compiled = []
class SchemaProcessor:
    def __init__(self, schema): self.schema = schema
    def __call__(self, *args): return args[-1]
def build(tokenizer, schema):
    assert tokenizer is (processor.tokenizer if hasattr(processor, "tokenizer") else processor)
    if mode in {"no_schema", "compile_failure"}: raise ValueError("synthetic compiler failure")
    compiled.append(schema)
    return SchemaProcessor(schema)
structured.build_json_schema_logits_processor = build
sys.modules["mlx_vlm.structured"] = structured
def load(path, **kwargs):
    assert kwargs == {"trust_remote_code": False, "local_files_only": True, "strict": True}
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    Path(model_path, "load-called").write_text("synthetic")
    return object(), processor
runtime.load = load
def generate(model, processor, prompt, images, **kwargs):
    assert kwargs.get("max_tokens") == 384 and kwargs.get("temperature") == 0.0 and kwargs.get("verbose") is False
    schema = None
    if "logits_processors" in kwargs:
        assert len(kwargs["logits_processors"]) == 1 and isinstance(kwargs["logits_processors"][0], SchemaProcessor)
        schema = kwargs["logits_processors"][0].schema
    if mode == "no_schema": assert "logits_processors" not in kwargs
    return SimpleNamespace(text=json.dumps({"prompt":prompt,"images":len(images),
        "max_pixels":processor.image_processor.max_pixels,"min_pixels":processor.image_processor.min_pixels,
        "schema":schema,"compile_count":len(compiled)}))
runtime.generate = generate
def apply_chat_template(processor, config, prompt, **kwargs):
    assert config["model_type"] in {"qwen3_vl", "qwen3_5"}
    assert kwargs["num_images"] > 0 and kwargs["enable_thinking"] is False
    return prompt
prompt_utils.apply_chat_template = apply_chat_template
sys.modules["mlx_vlm"] = runtime
sys.modules["mlx_vlm.prompt_utils"] = prompt_utils
class FakeImage:
    width = 5000 if mode == "huge" else 64
    height = 5000 if mode == "huge" else 64
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def convert(self, color):
        if mode == "huge": Path(model_path, "decoded-too-early").write_text("unsafe")
        return self
    def thumbnail(self, size): pass
    def close(self): pass
pil = types.ModuleType("PIL")
pil.Image = SimpleNamespace(open=lambda path: FakeImage())
sys.modules["PIL"] = pil
from ml.vlm.mlx_worker import main
sys.argv = ["synthetic", model_path, "384", side]
main()
'''


@pytest.mark.parametrize('model_type', ['qwen3_vl', 'qwen3_5'])
@pytest.mark.parametrize('mode', ['valid', 'huge'])
@pytest.mark.parametrize('side', [224, 640])
def test_worker_load_flags_time_mapping_and_pixel_guard_without_mlx(
        backend_factory, inference_request, monkeypatch, tmp_path, mode, side, model_type):
    backend = backend_factory()
    (backend.model_path / "config.json").write_text(json.dumps({"model_type": model_type}))
    backend.max_image_side = side
    worker = tmp_path / 'fake_runtime.py'
    worker.write_text(FAKE_RUNTIME)
    monkeypatch.setattr(backend, '_command', lambda directory: [sys.executable, '-u', str(worker), str(directory), mode, str(side)])
    backend.validate(2)
    if mode == 'huge':
        with pytest.raises(ModelUnavailable):
            backend.infer(inference_request, 2)
        assert not (backend.model_path / 'decoded-too-early').exists()
    else:
        result = json.loads(backend.infer(inference_request, 2))
        assert 'Image 1: timestamp_s=1.250000000' in result['prompt']
        assert result['prompt'].endswith(inference_request.prompt)
        assert result['images'] == 1 and result['max_pixels'] == side * side
        assert result['min_pixels'] <= result['max_pixels']


@pytest.mark.parametrize('model_type', ['qwen3_vl', 'qwen3_5'])
@pytest.mark.parametrize('mode', ['valid', 'processor_tokenizer', 'no_schema', 'compile_failure'])
def test_worker_builds_fresh_structured_processor_only_when_requested(
        backend_factory, inference_request, monkeypatch, tmp_path, mode, model_type):
    backend = backend_factory()
    (backend.model_path / "config.json").write_text(json.dumps({"model_type": model_type}))
    worker = tmp_path / 'fake_runtime.py'
    worker.write_text(FAKE_RUNTIME)
    monkeypatch.setattr(backend, '_command', lambda directory: [sys.executable, '-u', str(worker), str(directory), mode, '640'])
    backend.validate(2)
    schema = {'type': 'object', 'additionalProperties': False,
              'properties': {'answer': {'enum': ['synthetic']}}, 'required': ['answer']}
    if mode == 'compile_failure':
        with pytest.raises(ModelUnavailable):
            backend.infer(replace(inference_request, response_schema=schema), 2)
        assert backend._closed
        return
    requests = [inference_request] if mode == 'no_schema' else [
        replace(inference_request, response_schema=schema), replace(inference_request, response_schema=schema),
        inference_request]
    for index, item in enumerate(requests, 1):
        output = json.loads(backend.infer(item, 2))
        assert output['schema'] == (schema if item.response_schema else None)
        assert output['compile_count'] == (0 if mode == 'no_schema' else min(index, 2))


@pytest.mark.parametrize('schema', [None, [], 'schema', True])
def test_worker_rejects_malformed_optional_schema_from_wire(
        backend_factory, inference_request, monkeypatch, tmp_path, schema):
    backend = backend_factory()
    worker = tmp_path / 'fake_runtime.py'
    worker.write_text(FAKE_RUNTIME)
    monkeypatch.setattr(backend, '_command', lambda directory: [sys.executable, '-u', str(worker), str(directory), 'valid', '640'])
    original = backend._write

    def malformed(data, deadline):
        payload = json.loads(data)
        payload['response_schema'] = schema
        original(json.dumps(payload).encode() + b'\n', deadline)

    monkeypatch.setattr(backend, '_write', malformed)
    backend.validate(2)
    with pytest.raises(ModelUnavailable):
        backend.infer(inference_request, 2)
    assert backend._closed


@pytest.mark.parametrize('model_type', ['qwen3_5_text', 'qwen3_5_moe', 'qwen2_vl', 'arbitrary_architecture'])
def test_mlx_preparation_rejects_other_model_types_before_start(backend_factory, monkeypatch, model_type):
    backend = backend_factory()
    (backend.model_path / 'config.json').write_text(json.dumps({'model_type': model_type}))
    # The generic artifact guard remains reusable by other optional backends.
    assert module._validated_model_directory(backend.model_path) == backend.model_path
    monkeypatch.setattr(module.subprocess, 'Popen', lambda *a, **kw: pytest.fail('unsupported worker started'))
    with pytest.raises(ModelUnavailable, match='unsupported_checkpoint'):
        backend.validate(2)
    assert backend._closed and backend._process is None


@pytest.mark.parametrize('model_type', ['qwen3_5_text', 'qwen3_5_moe', 'qwen2_vl', 'arbitrary_architecture'])
def test_worker_independently_rejects_unsupported_architectures_before_load(
        backend_factory, monkeypatch, tmp_path, model_type):
    backend = backend_factory()
    (backend.model_path / 'config.json').write_text(json.dumps({'model_type': model_type}))
    worker = tmp_path / 'fake_runtime.py'
    worker.write_text(FAKE_RUNTIME)
    monkeypatch.setattr(backend, '_command', lambda directory: [sys.executable, '-u', str(worker), str(directory), 'valid', '640'])
    # Deliberately bypass only the parent guard: the worker must independently fail closed.
    monkeypatch.setattr(module, '_validated_mlx_model_directory', lambda directory: backend.model_path)
    with pytest.raises(ModelUnavailable):
        backend.validate(2)
    assert not (backend.model_path / 'load-called').exists()
    assert backend._closed and backend._process is None
