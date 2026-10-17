import base64
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.app.schemas import AnalysisResult
from ml import pipeline
from ml.candidates import Candidate
from ml.detect import Detection
from ml.perception import BATCH, Perception

REPO = Path(__file__).resolve().parents[2]
REAL_VIDEO = REPO / "data" / "open" / "videos" / "Х-26.mov"  # Cyrillic Х, as in the dataset
PERSON = Detection(box=(10.0, 5.0, 40.0, 45.0), score=0.9, label="person")


class FakeDetector:
    """Stands in for RT-DETR: one person per frame, optional delay, records concurrent use."""

    device = "cpu"
    created = 0

    def __init__(self, delay_sec: float = 0.0, first_call_delay_sec: float = 0.0) -> None:
        FakeDetector.created += 1
        self.delay_sec = delay_sec
        self.first_call_delay_sec = first_call_delay_sec
        self.calls = 0
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def detect(self, frames):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(self.first_call_delay_sec if self.calls == 0 else self.delay_sec)
        self.calls += 1
        with self._lock:
            self.active -= 1
        return [[PERSON] for _ in frames]


def make_video(path: Path, seconds: float, fps: int = 10) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for i in range(int(seconds * fps)):
        writer.write(np.full((48, 64, 3), (i * 3) % 255, dtype=np.uint8))
    writer.release()
    return path


@pytest.fixture
def clip(tmp_path):
    return make_video(tmp_path / "clip.mp4", seconds=3.0)


@pytest.fixture
def shared(monkeypatch):
    """Fresh shared model per test, backed by a fake detector."""
    holder = {}

    def factory():
        holder["detector"] = FakeDetector()
        return holder["detector"]

    model = pipeline.SharedDetector(factory=factory)
    monkeypatch.setattr(pipeline, "_SHARED", model)
    return holder


def one_candidate(type_id="cash_left_on_desk_when_leaving", t=1.0):
    return Candidate(
        type_id=type_id, t_start=max(0.0, t - 0.5), t_end=t + 0.5, t_peak=t, confidence=0.9,
        observation="кассир выходит из кадра, пачка остаётся на столе",
        evidence_times=(t,), highlight_boxes=((10.0, 10.0, 30.0, 30.0),),
    )


def test_plan_uses_full_rate_without_budget():
    assert pipeline.plan_sample_fps(600.0, None, sec_per_frame=1.0) == pipeline.MAX_SAMPLE_FPS


def test_plan_uses_full_rate_for_short_video():
    assert pipeline.plan_sample_fps(20.0, 240.0, sec_per_frame=0.15) == pipeline.MAX_SAMPLE_FPS


def test_plan_lowers_rate_for_long_video_to_fit_budget():
    fps = pipeline.plan_sample_fps(600.0, 240.0, sec_per_frame=0.15)
    assert pipeline.MIN_SAMPLE_FPS <= fps < pipeline.MAX_SAMPLE_FPS
    assert 600.0 * fps * 0.15 <= 240.0


def test_plan_never_goes_below_minimum():
    assert pipeline.plan_sample_fps(3600.0, 10.0, sec_per_frame=1.0) == pipeline.MIN_SAMPLE_FPS


def test_shared_detector_learns_speed_after_warmup():
    model = pipeline.SharedDetector(factory=lambda: FakeDetector(delay_sec=0.01))
    model.detect([np.zeros((4, 4, 3), np.uint8)] * 4)  # warm-up batch is ignored
    assert model.sec_per_frame() == pipeline.SEC_PER_FRAME_PRIOR["cpu"]
    model.detect([np.zeros((4, 4, 3), np.uint8)] * 4)
    assert model.sec_per_frame() < pipeline.SEC_PER_FRAME_PRIOR["cpu"]


def test_plan_accounts_for_jobs_running_in_parallel(tmp_path, shared, monkeypatch):
    video = make_video(tmp_path / "v.mp4", seconds=4.0)
    seen = []
    real_plan = pipeline.plan_sample_fps
    monkeypatch.setattr(pipeline, "plan_sample_fps", lambda d, b, spf: seen.append(spf) or real_plan(d, b, spf))
    monkeypatch.setattr(pipeline._SHARED, "sec_per_frame", lambda: 0.2)

    pipeline.analyze(str(video), time_budget_sec=240.0)
    with pipeline._SHARED.job():  # another worker is busy with its own video
        pipeline.analyze(str(video), time_budget_sec=240.0)

    assert seen == pytest.approx([0.2, 0.4])
    assert pipeline._SHARED.active_jobs() == 0


def test_analyze_with_stub_rules_is_no_violation(clip, shared):
    result = pipeline.analyze(str(clip))
    assert result == {"verdict": "no_violation", "violations": []}


def test_analyze_renders_candidates_into_valid_k1(clip, shared, monkeypatch):
    seen = {}

    def fake_rules(perception, scene=None):
        seen["perception"], seen["scene"] = perception, scene
        return [one_candidate()]

    monkeypatch.setattr("ml.rules.find_candidates", fake_rules)
    progress = []
    result = pipeline.analyze(str(clip), on_progress=lambda p, s: progress.append((p, s)))

    AnalysisResult.model_validate(result)
    assert result["verdict"] == "violation"
    [v] = result["violations"]
    assert v["type_id"] == "cash_left_on_desk_when_leaving"
    assert v["description"].endswith(".") and "Основание:" in v["description"]
    jpeg = base64.b64decode(v["evidence_frames"][0]["image_base64"])
    assert jpeg[:3] == b"\xff\xd8\xff"

    assert isinstance(seen["perception"], Perception) and seen["scene"] is None
    assert len(seen["perception"].frames) == pytest.approx(3.0 * pipeline.MAX_SAMPLE_FPS, abs=1)

    values = [p for p, _ in progress]
    assert values == sorted(values) and values[-1] == 1.0
    assert {s for _, s in progress} >= {"detect", "rules", "evidence"}
    assert max(p for p, s in progress if s == "detect") <= pipeline.PROGRESS_DETECT_END


def test_budget_overrun_returns_partial_result_instead_of_raising(tmp_path, monkeypatch):
    video = make_video(tmp_path / "long.mp4", seconds=20.0)
    slow = FakeDetector(first_call_delay_sec=1.1)
    monkeypatch.setattr(pipeline, "_SHARED", pipeline.SharedDetector(factory=lambda: slow))
    monkeypatch.setattr(pipeline, "FINISH_RESERVE_SEC", 0.0)
    seen = {}

    def fake_rules(perception, scene=None):
        seen["perception"] = perception
        return [one_candidate(t=1.0)]

    monkeypatch.setattr("ml.rules.find_candidates", fake_rules)
    result = pipeline.analyze(str(video), time_budget_sec=1.0)

    AnalysisResult.model_validate(result)
    assert result["verdict"] == "violation"  # what was seen before the deadline is still reported
    p = seen["perception"]
    assert slow.calls == 1  # the rest of the video was not sent to the model
    assert len(p.frames) == BATCH
    assert p.info.duration < 20.0
    assert all(e.t <= p.info.duration for e in p.events)


def test_budget_exhausted_before_any_detection_gives_no_violation(clip, shared, monkeypatch):
    monkeypatch.setattr("ml.rules.find_candidates", lambda perception, scene=None: [one_candidate()])
    monkeypatch.setattr(pipeline, "FINISH_RESERVE_SEC", 10.0)
    result = pipeline.analyze(str(clip), time_budget_sec=1.0)
    assert result == {"verdict": "no_violation", "violations": []}


def test_parallel_jobs_share_one_model_and_never_infer_concurrently(tmp_path, monkeypatch):
    FakeDetector.created = 0
    detector_box = {}

    def factory():
        time.sleep(0.2)  # slow init widens the race window
        detector_box["d"] = FakeDetector(delay_sec=0.02)
        return detector_box["d"]

    monkeypatch.setattr(pipeline, "_SHARED", pipeline.SharedDetector(factory=factory))
    videos = [make_video(tmp_path / f"v{i}.mp4", seconds=4.0) for i in range(2)]
    errors = []

    def job(path):
        try:
            AnalysisResult.model_validate(pipeline.analyze(str(path)))
        except Exception as exc:  # noqa: BLE001 — surfaced through the assertion below
            errors.append(exc)

    threads = [threading.Thread(target=job, args=(v,)) for v in videos]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors
    assert FakeDetector.created == 1
    assert detector_box["d"].max_active == 1


def test_unreadable_video_raises(tmp_path, shared):
    bad = tmp_path / "bad.mov"
    bad.write_bytes(b"not a video")
    with pytest.raises(ValueError):
        pipeline.analyze(str(bad))


@pytest.mark.slow
@pytest.mark.skipif(not REAL_VIDEO.exists(), reason="data/open link is missing")
def test_real_video_end_to_end(monkeypatch):
    monkeypatch.setattr("ml.rules.find_candidates", lambda perception, scene=None: [one_candidate(t=10.0)])
    started = time.monotonic()
    result = pipeline.analyze(str(REAL_VIDEO), time_budget_sec=240.0)
    elapsed = time.monotonic() - started

    parsed = AnalysisResult.model_validate(result)
    assert parsed.verdict == "violation"
    frame = parsed.violations[0].evidence_frames[0]
    assert len(base64.b64decode(frame.image_base64)) <= 150 * 1024
    assert elapsed < 240.0
