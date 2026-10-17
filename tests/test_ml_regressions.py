"""Reviewer regressions: artificial files and loader boundaries only."""
import importlib
import builtins
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest


PATH_GUARDS = [
    ("ml.baseline.features", "ffmpeg_command", ValueError),
    ("ml.baseline.features", "local_weights", ValueError),
    ("ml.vlm.frames", "local_file", RuntimeError),
    ("ml.vlm.preflight", "model_directory", RuntimeError),
]


@pytest.mark.parametrize("module_name,function_name,error", PATH_GUARDS)
@pytest.mark.parametrize("raw", [r"\\fixture.invalid\share\artifact", "//fixture.invalid/share/artifact",
                                  r"\\?\UNC\fixture.invalid\share\artifact", "smb://fixture.invalid/share/artifact"])
def test_raw_network_paths_rejected_before_resolution_or_disk_probe(module_name, function_name, error, raw):
    guard = getattr(importlib.import_module(module_name), function_name)
    with patch.object(Path, "resolve", side_effect=AssertionError("network resolution forbidden")), \
         patch.object(Path, "is_file", side_effect=AssertionError("disk probe forbidden")), \
         patch.object(Path, "is_dir", side_effect=AssertionError("disk probe forbidden")):
        with pytest.raises(error):
            guard(raw)


@pytest.mark.parametrize("module_name,function_name,error", PATH_GUARDS)
def test_resolved_network_paths_rejected_before_disk_probe(module_name, function_name, error):
    guard = getattr(importlib.import_module(module_name), function_name)
    network = Path("//fixture.invalid/share/artifact")
    with patch.object(Path, "resolve", return_value=network), \
         patch.object(Path, "is_file", side_effect=AssertionError("disk probe forbidden")), \
         patch.object(Path, "is_dir", side_effect=AssertionError("disk probe forbidden")):
        with pytest.raises(error):
            guard("local-link-fixture")


@pytest.fixture
def snapshot(tmp_path):
    """Structural placeholders, never usable model weights."""
    (tmp_path / "config.json").write_text('{"model_type":"qwen2_5_vl"}', encoding="utf-8")
    for name in ("preprocessor_config.json", "tokenizer_config.json", "tokenizer.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"ARTIFICIAL STRUCTURAL FIXTURE ONLY")
    return tmp_path


@pytest.mark.parametrize("model_type", ["qwen2_5_vl", "fixture_generic"])
@pytest.mark.parametrize("diagnostic", ["missing_keys", "mismatched_keys", "error_msgs"])
def test_runtime_rejects_incomplete_checkpoint_diagnostics(snapshot, model_type, diagnostic):
    module = importlib.import_module("ml.vlm.transformers_backend")
    (snapshot / "config.json").write_text(json.dumps({"model_type": model_type}), encoding="utf-8")
    model = Mock()
    info = {key: [] for key in ("missing_keys", "mismatched_keys", "error_msgs")}
    info[diagnostic] = ["artificial checkpoint defect"]
    loader = Mock()
    loader.from_pretrained.side_effect = lambda *args, **kwargs: (
        (model, info) if kwargs.get("output_loading_info") else model)
    modules = {"torch": SimpleNamespace(), "PIL": SimpleNamespace(Image=Mock()),
               "transformers": SimpleNamespace(AutoProcessor=Mock(),
                   Qwen2_5_VLForConditionalGeneration=loader, AutoModelForImageTextToText=loader)}
    with patch.dict(sys.modules, modules):
        with pytest.raises(module.ModelUnavailable):
            module._LocalRuntime(snapshot, "cpu", 16)
    assert loader.from_pretrained.call_args.kwargs["output_loading_info"] is True
    model.eval.assert_not_called()


@pytest.mark.parametrize("origin", [0.0, 10.0])
def test_probe_uses_container_origin_not_first_video_frame(tmp_path, origin):
    module = importlib.import_module("ml.vlm.frames")
    video = tmp_path / "artificial.mkv"
    video.write_bytes(b"PROBE BOUNDARY FIXTURE")
    data = {"frames": [{"best_effort_timestamp_time": str(origin + t), "duration_time": "1"}
                       for t in (5, 6, 11)],
            "streams": [{"start_time": str(origin + 5), "duration": "7"}],
            "format": {"start_time": str(origin), "duration": "13"}}
    source = module.FFmpegFrames()
    with patch.object(source, "_run", return_value=json.dumps(data).encode()) as run:
        info = source.probe(video, 5)
    assert [f.timestamp_s for f in info.frames] == [5, 6, 11]
    assert [f.pts_s for f in info.frames] == [origin + t for t in (5, 6, 11)]
    assert info.duration_s == 12  # Video endpoint, not the container's trailing padding.
    assert "format=start_time,duration" in run.call_args.args[0][
        run.call_args.args[0].index("-show_entries") + 1]


@pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="ffmpeg required")
def test_delayed_video_matches_baseline_and_selects_candidate_container_seconds(tmp_path):
    api = importlib.import_module("ml.vlm")
    features = importlib.import_module("ml.baseline.features")
    video = tmp_path / "artificial-delayed-video.mkv"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
        "color=c=black:s=32x32:r=1:d=4", "-f", "lavfi", "-i",
        "anullsrc=r=8000:cl=mono:d=13", "-vf", r"setpts=(5+N+4*gte(N\,2))/TB",
        "-fps_mode", "vfr", "-c:v", "ffv1", "-c:a", "pcm_s16le", str(video)],
        check=True, capture_output=True, timeout=20)
    source = api.FFmpegFrames()
    info = source.probe(video, 10)
    baseline_times = [t for t, _ in features.decode_frames(video, fps=1, max_frames=4, timeout_s=10)]
    assert baseline_times == [5, 6, 11, 12]
    assert [f.timestamp_s for f in info.frames] == baseline_times
    assert info.duration_s == 13
    candidate = {"candidate_id": "cv-delayed", "rule_id": "r1", "violation_type": "fixture",
                 "start_s": 6, "end_s": 6}
    rule = {"rule_id": "r1", "violation_type": "fixture", "description": "Artificial fixture",
            "citation": "FIXTURE", "verified": True}
    backend = Mock()
    backend.infer.return_value = json.dumps({"candidate_id": "cv-delayed", "rule_id": "r1",
        "violation": True, "confidence": 0.9, "insufficient_evidence": False,
        "observations": [{"timestamp_s": 6, "text": "Artificial fixture only"}]})
    report = api.verify_candidates(video, [candidate], [rule], backend=backend, frame_source=source,
                                   budget_sec=10)
    assert report.results[0].status == "confirmed"
    request = backend.infer.call_args.args[0]
    assert [(frame.timestamp_s, frame.pts_s) for frame in request.frames] == [(6, 6)]


@pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="ffmpeg required")
def test_nonzero_container_origin_keeps_duration_relative_to_same_origin(tmp_path):
    module = importlib.import_module("ml.vlm.frames")
    video = tmp_path / "artificial-offset.mkv"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
        "color=c=black:s=32x32:r=1:d=3", "-vf", "setpts=PTS+5/TB", "-fps_mode", "vfr",
        "-c:v", "ffv1", str(video)], check=True, capture_output=True, timeout=20)
    info = module.FFmpegFrames().probe(video, 10)
    assert [f.timestamp_s for f in info.frames] == [0, 1, 2]
    assert [f.pts_s for f in info.frames] == [5, 6, 7]
    assert info.duration_s == 3


@pytest.mark.parametrize("weight_map", [[], None, "shard.safetensors", {},
    {"tensor": None}, {"tensor": []}, {"tensor": ""}, {"tensor": " "},
    {"": "shard.safetensors"}, {" ": "shard.safetensors"},
    {"tensor": "../outside.safetensors"}, {"tensor": "C:/outside.safetensors"},
    {"tensor": "//fixture.invalid/share/model.safetensors"}, {"tensor": "shard.bin"}])
def test_malformed_shard_maps_fail_closed_and_preserve_cv_candidate(snapshot, weight_map):
    module = importlib.import_module("ml.vlm.preflight")
    api = importlib.import_module("ml.vlm")
    (snapshot / "model.safetensors").unlink()
    (snapshot / "shard.safetensors").write_bytes(b"ARTIFICIAL SHARD")
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8")
    candidate = {"candidate_id": "cv-shards", "rule_id": "r1", "violation_type": "fixture",
                 "start_s": 0, "end_s": 1, "cv_metadata": {"score": 0.81}}
    rule = {"rule_id": "r1", "violation_type": "fixture", "description": "Artificial fixture",
            "citation": "FIXTURE", "verified": True}
    with patch.object(module.importlib.util, "find_spec", return_value=object()), \
         patch.object(module.shutil, "which", return_value="artificial-executable"):
        result = module.preflight(snapshot)
        assert not result.ready
        assert "weight_shards_invalid" in result.errors
        report = api.verify_candidates("unused-fixture.mp4", [candidate], [rule], model_path=snapshot,
                                       budget_sec=5)
    assert report.results[0].status == "uncertain"
    assert report.results[0].reason == "model_unavailable"
    assert report.retained_candidates == [candidate]
    assert report.calls_used == 0


def test_valid_nonempty_local_shard_map_passes_structural_preflight(snapshot):
    module = importlib.import_module("ml.vlm.preflight")
    (snapshot / "model.safetensors").unlink()
    (snapshot / "shard.safetensors").write_bytes(b"ARTIFICIAL SHARD")
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"tensor.weight": "shard.safetensors"}}), encoding="utf-8")
    with patch.object(module.importlib.util, "find_spec", return_value=object()), \
         patch.object(module.shutil, "which", return_value="artificial-executable"):
        assert module.preflight(snapshot).ready


@pytest.mark.parametrize("missing", ["config.json", "model.safetensors", "all"])
@pytest.mark.parametrize("entry", ["validate", "infer", "runtime"])
def test_invalid_artifacts_rejected_before_worker_start_or_heavy_imports(snapshot, missing, entry):
    module = importlib.import_module("ml.vlm.transformers_backend")
    for path in snapshot.iterdir() if missing == "all" else [snapshot / missing]:
        path.unlink()
    original_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "PIL", "transformers"}:
            raise AssertionError("heavy imports forbidden for invalid model")
        return original_import(name, *args, **kwargs)
    backend = module.LocalTransformersBackend(snapshot)
    with patch.object(builtins, "__import__", side_effect=guarded_import), \
         patch.object(module.multiprocessing, "get_context", side_effect=AssertionError("worker start forbidden")):
        with pytest.raises(module.ModelUnavailable):
            if entry == "runtime":
                module._LocalRuntime(snapshot, "cpu", 16)
            elif entry == "validate":
                backend.validate(timeout_s=5)
            else:
                backend.infer(None, timeout_s=5)
    assert backend._process is None


@pytest.mark.parametrize("pil_available", [True, False])
def test_siglip_uses_pil_processor_without_auto_torchvision_dependency(tmp_path, pil_available):
    module = importlib.import_module("ml.baseline.features")
    processor, legacy = Mock(), Mock()
    exports = {"SiglipVisionModel": Mock(), "SiglipImageProcessor": legacy,
               "AutoImageProcessor": Mock(side_effect=AssertionError("Auto processor forbidden"))}
    if pil_available:
        exports["SiglipImageProcessorPil"] = processor
        legacy.from_pretrained.side_effect = AssertionError("PIL processor must be preferred")
    else:
        processor = legacy
    model = Mock()
    exports["SiglipVisionModel"].from_pretrained.return_value = (
        model, {"missing_keys": [], "mismatched_keys": [], "error_msgs": []})
    modules = {"torch": Mock(), "transformers": SimpleNamespace(**exports),
               "PIL": SimpleNamespace(Image=Mock())}
    with patch.dict(sys.modules, modules), patch.object(module, "local_weights", return_value=tmp_path):
        encoder = module.SiglipEncoder(tmp_path)
    assert encoder.processor is processor.from_pretrained.return_value
    kwargs = processor.from_pretrained.call_args.kwargs
    assert kwargs["local_files_only"] is True
    assert kwargs["trust_remote_code"] is False
    model.eval.assert_called_once()
