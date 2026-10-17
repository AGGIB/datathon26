"""Synthetic-only contracts for the isolated local pair screening CLI."""
import importlib
import json
import subprocess
import sys

import pytest


def module():
    return importlib.import_module("ml.vlm.pair_eval")


def test_response_requires_exact_strict_boolean_json():
    parse = module().parse_response
    assert parse('{"violation":true,"insufficient_evidence":false}') == {
        "violation": True, "insufficient_evidence": False}
    invalid = [
        '{"violation":1,"insufficient_evidence":false}',
        '{"violation":false,"insufficient_evidence":"false"}',
        '{"violation":false}',
        '{"violation":false,"insufficient_evidence":false,"observations":[]}',
        '{"violation":false,"violation":true,"insufficient_evidence":false}',
        '```json\n{"violation":true,"insufficient_evidence":false}\n```',
        '{"violation":NaN,"insufficient_evidence":false}',
    ]
    for raw in invalid:
        with pytest.raises(ValueError):
            parse(raw)


PAIRS = (("X-2", "X-3"), ("X-13", "X-14"), ("X-16", "X-17"),
         ("X-34", "X-35"), ("X-43", "X-44"))


def inputs(tmp_path):
    videos = tmp_path / "synthetic-videos"
    videos.mkdir()
    labels, annotations = [], []
    for pair in PAIRS:
        for slot, key in enumerate(pair):
            literal = key.replace("X", "Х") + ".MOV"
            (videos / literal).write_bytes(("synthetic-" + key).encode())
            labels.append(dict(video=key, positive=slot == 0, timestamp=987 if slot == 0 else None))
            if slot == 0:
                annotations.append(dict(video=key, type_id="synthetic_type", t_peak=987,
                                        what_visible="DO_NOT_USE_TRUTH_SCENE"))
    catalog = [dict(id="synthetic_type", codex_code="Т99", name="Synthetic target condition")]
    return labels, videos, annotations, catalog


def test_prepare_preserves_literals_is_order_invariant_and_uses_one_blind_prompt(tmp_path):
    m = module()
    labels, videos, annotations, catalog = inputs(tmp_path)
    plan = m.prepare_inputs(labels, videos, annotations, catalog)
    reverse = m.prepare_inputs(list(reversed(labels)), videos, list(reversed(annotations)), catalog)
    assert plan == reverse
    assert len(plan) == 10
    assert plan[0]["path"].name == "Х-2.MOV"
    assert plan[0]["content_sha256"]
    for i in range(0, 10, 2):
        assert plan[i]["prompt"] == plan[i + 1]["prompt"]
        prompt = plan[i]["prompt"]
        assert "Synthetic target condition" in prompt
        for forbidden in ("X-", "Х-", "987", "DO_NOT_USE_TRUTH_SCENE", "positive", "negative", "observations"):
            assert forbidden not in prompt


@pytest.mark.parametrize("defect", ["missing_label", "same_truth", "label_collision", "annotation_collision",
                                    "file_collision", "malformed_label", "unknown_type", "ambiguous_type"])
def test_prepare_rejects_invalid_pair_inputs(tmp_path, defect):
    m = module()
    labels, videos, annotations, catalog = inputs(tmp_path)
    if defect == "missing_label":
        labels.pop()
    elif defect == "same_truth":
        labels[1]["positive"] = True
    elif defect == "label_collision":
        labels.append(dict(labels[0], video="Х-2"))
    elif defect == "annotation_collision":
        annotations.append(dict(annotations[0], video="Х-2"))
    elif defect == "file_collision":
        (videos / "X-2.mov").write_bytes(b"synthetic-collision")
    elif defect == "malformed_label":
        labels[0]["video"] += "."
    elif defect == "unknown_type":
        annotations[0]["type_id"] = "unknown"
    elif defect == "ambiguous_type":
        annotations.append(dict(annotations[0], type_id="other_type", t_peak=999))
        catalog.append(dict(id="other_type", name="Other synthetic condition"))
    with pytest.raises(ValueError):
        m.prepare_inputs(labels, videos, annotations, catalog)


def test_missing_file_is_planned_failure_not_removed_from_denominator(tmp_path):
    m = module()
    labels, videos, annotations, catalog = inputs(tmp_path)
    (videos / "Х-2.MOV").unlink()
    plan = m.prepare_inputs(labels, videos, annotations, catalog)
    assert len(plan) == 10
    assert plan[0]["path"] is None
    assert plan[0]["input_status"] == "missing"


class Clock:
    now = 100.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class Frames:
    timebase = "cv_index_fps"

    def __init__(self, clock):
        self.clock = clock
        self.calls = []
        self.closed = False

    def probe(self, path, timeout_s):
        from ml.vlm.contracts import FrameTimestamp, VideoInfo
        self.calls.append(("probe", timeout_s))
        self.clock.advance(2)
        return VideoInfo(100.0, tuple(FrameTimestamp(i, float(i), float(i)) for i in range(100)))

    def extract(self, path, selected, directory, timeout_s):
        import struct
        from ml.vlm.contracts import Frame
        self.calls.append(("extract", timeout_s, [frame.timestamp_s for frame in selected]))
        self.clock.advance(3)
        result = []
        for slot, frame in enumerate(selected):
            path = directory / f"frame-{slot}.png"
            path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + struct.pack(">II", 320, 240))
            result.append(Frame(frame.timestamp_s, path, frame.pts_s))
        return tuple(result)

    def close(self):
        self.closed = True


class Backend:
    def __init__(self, clock, responses):
        self.clock = clock
        self.responses = iter(responses)
        self.validations = []
        self.calls = []
        self.closed = False

    def validate(self, timeout_s):
        self.validations.append(timeout_s)
        self.clock.advance(7)

    def infer(self, request, timeout_s):
        self.calls.append((request, timeout_s))
        self.clock.advance(5)
        answer = next(self.responses)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    def close(self):
        self.closed = True


def response(value, uncertain=False):
    return json.dumps(dict(violation=value, insufficient_evidence=uncertain))


def run_synthetic(tmp_path, monkeypatch, responses=None, budget=20):
    m = module()
    labels, videos, annotations, catalog = inputs(tmp_path)
    plan = m.prepare_inputs(labels, videos, annotations, catalog)
    clock = Clock()
    backend = Backend(clock, responses if responses is not None else [response(i % 2 == 0) for i in range(10)])
    sources, constructors = [], []

    def factory(model_path, **kwargs):
        constructors.append((model_path, kwargs))
        return backend

    def frames():
        source = Frames(clock)
        sources.append(source)
        return source

    output = tmp_path / "artifacts" / "screen"
    kwargs = dict(model_path=tmp_path / "synthetic-model", budget_sec=budget,
                  backend_factory=factory, frame_factory=frames, clock=clock, artifact_root=tmp_path)
    return m, plan, backend, sources, constructors, output, kwargs


def test_screen_warms_once_samples_without_truth_and_measures_remaining_budget(tmp_path, monkeypatch):
    m, plan, backend, sources, constructors, output, kwargs = run_synthetic(tmp_path, monkeypatch)
    report = m.run_screening(plan, output, **kwargs)
    assert backend.validations == [120]
    assert constructors[0][1]["max_new_tokens"] == 128
    assert len(backend.calls) == 10
    assert all(timeout == 15 for _, timeout in backend.calls)
    assert len(sources) == 10 and all(source.closed for source in sources)
    assert all(source.calls == [("probe", 20), ("extract", 18, [10, 50, 90])] for source in sources)
    assert backend.closed
    assert report["summary"]["pairs_correct"] == 5
    assert report["summary"]["pair_denominator"] == 5
    assert report["summary"]["video_denominator"] == 10
    assert report["startup_sec"] == 7
    for row in report["records"]:
        assert row["probe_sec"] == 2 and row["extract_sec"] == 3
        assert row["inference_sec"] == 5 and row["end_to_end_sec"] == 10
        assert row["frame_count"] == 3
        assert [frame["timestamp_s"] for frame in row["frames"]] == [10, 50, 90]
        assert all(frame["width"] == 320 and frame["height"] == 240 for frame in row["frames"])
    assert json.loads((output / "report.json").read_text())["summary"] == report["summary"]


def test_invalid_uncertain_errors_and_missing_stay_in_fixed_denominators(tmp_path, monkeypatch):
    answers = [response(True), response(True), response(True, True), response(False),
               "PRIVATE_RAW_INVALID", response(False), TimeoutError("PRIVATE_TIMEOUT"), response(False),
               RuntimeError("PRIVATE_ERROR"), response(False)]
    m, plan, backend, sources, constructors, output, kwargs = run_synthetic(tmp_path, monkeypatch, answers)
    report = m.run_screening(plan, output, **kwargs)
    summary = report["summary"]
    assert summary["pairs_correct"] == 0 and summary["pair_denominator"] == 5
    assert summary["video_denominator"] == 10 and len(report["records"]) == 10
    assert summary["scored_videos"] == 6 and summary["coverage"] == 0.6
    assert summary["status_counts"] == {"ok": 6, "uncertain": 1, "invalid_response": 1, "timed_out": 1, "error": 1}
    assert m.summarize(list(reversed(report["records"]))) == summary
    assert "PRIVATE_RAW_INVALID" in (output / "report.json").read_text()
    assert "PRIVATE_ERROR" not in (output / "report.json").read_text()
    assert "PRIVATE" not in json.dumps(summary)


def test_missing_file_is_retained_without_any_frame_or_infer_call(tmp_path, monkeypatch):
    m, plan, backend, sources, constructors, output, kwargs = run_synthetic(tmp_path, monkeypatch)
    plan[0] = dict(plan[0], path=None, input_status="missing")
    backend.responses = iter([response(i % 2 == 0) for i in range(1, 10)])
    report = m.run_screening(plan, output, **kwargs)
    assert report["records"][0]["status"] == "missing"
    assert report["summary"]["pairs_correct"] == 4
    assert report["summary"]["video_denominator"] == 10
    assert len(sources) == len(backend.calls) == 9


def test_atomic_prefix_survives_interrupt_and_output_cannot_be_reused(tmp_path, monkeypatch):
    m, plan, backend, sources, constructors, output, kwargs = run_synthetic(
        tmp_path, monkeypatch, [response(True), KeyboardInterrupt()])
    with pytest.raises(KeyboardInterrupt):
        m.run_screening(plan, output, **kwargs)
    saved = json.loads((output / "report.json").read_text())
    assert len(saved["records"]) == 1
    assert saved["summary"]["pair_denominator"] == 5
    assert saved["summary"]["video_denominator"] == 10
    assert backend.closed
    with pytest.raises(FileExistsError):
        m.run_screening(plan, output, **kwargs)


def test_too_small_budget_never_renews_deadline_or_calls_inference(tmp_path, monkeypatch):
    m, plan, backend, sources, constructors, output, kwargs = run_synthetic(tmp_path, monkeypatch, budget=1)
    report = m.run_screening(plan, output, **kwargs)
    assert len(backend.calls) == 0
    assert report["summary"]["status_counts"] == {"timed_out": 10}
    assert all(source.calls == [("probe", 1)] for source in sources)


def test_startup_failure_keeps_ten_failures_without_retry_or_frame_loading(tmp_path, monkeypatch):
    m, plan, backend, sources, constructors, output, kwargs = run_synthetic(tmp_path, monkeypatch)
    validations = []

    def unavailable(timeout):
        validations.append(timeout)
        raise RuntimeError("PRIVATE_STARTUP")

    backend.validate = unavailable
    report = m.run_screening(plan, output, **kwargs)
    assert validations == [120] and len(constructors) == 1
    assert report["summary"]["status_counts"] == {"model_unavailable": 10}
    assert report["summary"]["pairs_correct"] == 0
    assert not sources and not backend.calls


def test_artifact_guard_precedes_backend_creation(tmp_path, monkeypatch):
    m, plan, backend, sources, constructors, output, kwargs = run_synthetic(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        m.run_screening(plan, tmp_path / "outside", **kwargs)
    assert not constructors


def test_cli_safe_stdout_offline_guard_and_explicit_model_provenance(tmp_path, monkeypatch, capsys):
    m, plan, backend, sources, constructors, output, kwargs = run_synthetic(
        tmp_path, monkeypatch, ["PRIVATE_RAW"] + [response(i % 2 == 0) for i in range(1, 10)])
    labels = tmp_path / "labels.csv"
    labels.write_text("video,timestamp,label\n" + "".join(
        f'{item["video_key"]},{987 if item["positive"] else ""},{"нарушение" if item["positive"] else "без нарушения"}\n'
        for item in plan), encoding="utf-8")
    annotations = tmp_path / "annotations.csv"
    annotations.write_text("video,type_id,t_peak\n" + "".join(
        f'{item["video_key"]},Т99,987\n' for item in plan if item["positive"]), encoding="utf-8")
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps([dict(id="synthetic_type", codex_code="Т99", name="Synthetic condition")]))
    model = tmp_path / "synthetic-model"
    model.mkdir()
    events = []
    monkeypatch.setattr(m, "_deny_network", lambda: events.append("network_denied"))
    factory = kwargs["backend_factory"]

    def noisy_factory(path, **options):
        assert events == ["network_denied"]
        print("PRIVATE_LIBRARY_NOISE")
        return factory(path, **options)

    args = ["--labels", str(labels), "--videos-dir", str(tmp_path / "synthetic-videos"),
            "--annotations", str(annotations), "--catalog", str(catalog), "--model-path", str(model),
            "--model-id", "synthetic-explicit-provenance", "--output", str(output), "--budget-sec", "20"]
    code = m.main(args, backend_factory=noisy_factory, frame_factory=kwargs["frame_factory"],
                  clock=kwargs["clock"], artifact_root=tmp_path)
    assert code == 2
    captured = capsys.readouterr()
    assert not captured.err
    safe = json.loads(captured.out)
    assert safe["pair_denominator"] == 5 and safe["video_denominator"] == 10
    assert "PRIVATE" not in captured.out and "X-" not in captured.out and "Т99" not in captured.out
    saved = json.loads((output / "report.json").read_text())
    assert saved["provenance"]["model_id"] == "synthetic-explicit-provenance"
    assert saved["records"][0]["raw_response"] == "PRIVATE_RAW"
    assert saved["input_provenance"]["labels_sha256"]
    assert saved["input_provenance"]["annotations_sha256"]
    assert all(row["prompt"] for row in saved["planned_inputs"])


def test_pair_orientation_is_not_assumed_from_the_order_of_ids(tmp_path):
    m = module()
    labels, videos, annotations, catalog = inputs(tmp_path)
    for row in labels:
        row["positive"] = not row["positive"]
        row["timestamp"] = 123 if row["positive"] else None
    annotations = [dict(video=row["video"], type_id="synthetic_type", t_peak=123)
                   for row in labels if row["positive"]]
    plan = m.prepare_inputs(labels, videos, annotations, catalog)
    assert all(plan[index]["positive"] is (index % 2 == 1) for index in range(10))
    assert all(plan[index]["prompt"] == plan[index + 1]["prompt"] for index in range(0, 10, 2))


def test_spawn_module_installs_network_guard_without_heavy_imports():
    # Audit an event directly: this test never opens a network connection.
    script = '''
import runpy
import sys
runpy.run_module("ml.vlm.pair_eval", run_name="__mp_main__")
assert not any(name in sys.modules for name in ("torch", "transformers", "cv2", "PIL"))
try:
    sys.audit("socket.connect", None, ("192.0.2.1", 443))
except PermissionError:
    pass
else:
    raise AssertionError("spawned worker network guard is absent")
print("guarded_and_lazy")
'''
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "guarded_and_lazy"


@pytest.mark.parametrize("budget", [0, -1, True, float("inf"), float("nan")])
def test_invalid_budgets_are_rejected_before_model_initialization(tmp_path, monkeypatch, budget):
    m, plan, backend, sources, constructors, output, kwargs = run_synthetic(tmp_path, monkeypatch, budget=budget)
    with pytest.raises(ValueError):
        m.run_screening(plan, output, **kwargs)
    assert not constructors and not output.exists()


def test_output_reserved_before_input_hashing_and_preparation(tmp_path, monkeypatch):
    m, plan, backend, sources, constructors, output, kwargs = run_synthetic(tmp_path, monkeypatch)
    called = []

    def prepare():
        assert output.is_dir()
        assert not constructors
        called.append(True)
        return plan

    report = m.run_screening(prepare, output, **kwargs)
    assert called == [True] and report["summary"]["pairs_correct"] == 5
