import base64
import dataclasses
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.app.schemas import AnalysisResult
from ml import pipeline
from ml.candidates import Candidate
from ml.detect import INPUT_SIZE, Detection
from ml.kb import KBError
from ml.perception import Perception
from ml.shared_detector import CALIBRATION_FRAMES, SharedDetector

REPO = Path(__file__).resolve().parents[2]
REAL_VIDEO = REPO / "data" / "open" / "videos" / "Х-26.mov"  # Cyrillic Х, as in the dataset
PERSON = Detection(box=(10.0, 5.0, 40.0, 45.0), score=0.9, label="person")
SMALL = pipeline.SMALL_INPUT_SIZE


class FakeDetector:
    """Stands in for RT-DETR: one person per frame, a fixed cost per frame, records concurrent use."""

    device = "cpu"

    def __init__(self, sec_per_frame: float = 0.0) -> None:
        self.input_size = INPUT_SIZE
        self.sec_per_frame = sec_per_frame
        self.calls = 0
        self.frames_seen = 0
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def detect(self, frames):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(self.sec_per_frame * len(frames))
        self.calls += 1
        self.frames_seen += len(frames)
        with self._lock:
            self.active -= 1
        return [[PERSON] for _ in frames]


def make_video(path: Path, seconds: float, fps: int = 10) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for i in range(int(seconds * fps)):
        writer.write(np.full((48, 64, 3), (i * 3) % 255, dtype=np.uint8))
    writer.release()
    return path


def use_detector(monkeypatch, detector: FakeDetector) -> FakeDetector:
    monkeypatch.setattr(pipeline, "_SHARED", SharedDetector(factory=lambda: detector))
    return detector


def capture_rules(monkeypatch, candidates=()):
    seen = {}

    def fake_rules(perception, scene=None):
        seen["perception"], seen["scene"] = perception, scene
        return list(candidates)

    monkeypatch.setattr("ml.rules.find_candidates", fake_rules)
    return seen


def one_candidate(type_id="cash_left_on_desk_when_leaving", t=1.0):
    return Candidate(
        type_id=type_id, t_start=max(0.0, t - 0.5), t_end=t + 0.5, t_peak=t, confidence=0.9,
        observation="кассир выходит из кадра, пачка остаётся на столе",
        evidence_times=(t,), highlight_boxes=((10.0, 10.0, 30.0, 30.0),),
    )


@pytest.fixture
def clip(tmp_path):
    return make_video(tmp_path / "clip.mp4", seconds=3.0)


@pytest.fixture
def fake(monkeypatch):
    return use_detector(monkeypatch, FakeDetector())


# --- planning -------------------------------------------------------------------------------------


def costs(full: float, small: float):
    calls = []

    def cost_at(size: int) -> float:
        calls.append(size)
        return {INPUT_SIZE: full, SMALL: small}[size]

    return cost_at, calls


def test_plan_without_budget_uses_full_rate_and_size_without_measuring():
    cost_at, calls = costs(1.0, 0.5)
    plan = pipeline.plan_sampling(600.0, None, cost_at)
    assert (plan.sample_fps, plan.input_size) == (pipeline.MAX_SAMPLE_FPS, INPUT_SIZE)
    assert calls == []


def test_plan_keeps_full_rate_when_it_fits():
    cost_at, calls = costs(0.05, 0.03)
    plan = pipeline.plan_sampling(142.5, 185.0, cost_at)
    assert (plan.sample_fps, plan.input_size) == (pipeline.MAX_SAMPLE_FPS, INPUT_SIZE)
    assert calls == [INPUT_SIZE]  # the smaller input is not even measured


def test_plan_lowers_rate_before_shrinking_input():
    cost_at, _ = costs(0.3, 0.17)
    plan = pipeline.plan_sampling(142.5, 185.0, cost_at)
    assert plan.input_size == INPUT_SIZE
    assert pipeline.SAFE_MIN_FPS <= plan.sample_fps < pipeline.MAX_SAMPLE_FPS
    assert 142.5 * plan.sample_fps * plan.sec_per_frame <= 185.0


def test_plan_shrinks_input_when_full_size_cannot_keep_tracking_rate():
    cost_at, calls = costs(1.0, 0.4)
    plan = pipeline.plan_sampling(142.5, 185.0, cost_at)
    assert plan.input_size == SMALL
    assert plan.sample_fps >= pipeline.SAFE_MIN_FPS
    assert plan.sec_per_frame == 0.4
    assert calls[0] == INPUT_SIZE and SMALL in calls


def test_plan_goes_below_safe_rate_only_at_small_input():
    cost_at, _ = costs(1.6, 0.7)  # Docker CPU, slow end
    plan = pipeline.plan_sampling(142.5, 185.0, cost_at)
    assert plan.input_size == SMALL
    assert pipeline.HARD_MIN_FPS <= plan.sample_fps < pipeline.SAFE_MIN_FPS


def test_plan_never_goes_below_the_tracker_limit():
    cost_at, _ = costs(5.0, 3.0)
    plan = pipeline.plan_sampling(3600.0, 100.0, cost_at)
    assert (plan.sample_fps, plan.input_size) == (pipeline.HARD_MIN_FPS, SMALL)


def test_tracker_limit_keeps_consecutive_samples_linked():
    # IoUTracker in perceive() retires a track after a 1.0 s gap; sampling must stay denser than that
    assert 1.0 / pipeline.HARD_MIN_FPS < pipeline.TRACKER_GAP_SEC
    assert 2.0 / pipeline.SAFE_MIN_FPS < pipeline.TRACKER_GAP_SEC  # survives one missed detection


def test_plan_counts_parallel_jobs(tmp_path, monkeypatch, fake):
    video = make_video(tmp_path / "v.mp4", seconds=4.0)
    seen = []
    real_plan = pipeline.plan_sampling
    monkeypatch.setattr(pipeline, "plan_sampling",
                        lambda d, w, cost_at, contention=1: seen.append(contention) or real_plan(d, w, cost_at, contention))

    pipeline.analyze(str(video), time_budget_sec=240.0)
    with pipeline._SHARED.job():  # another worker is busy with its own video
        pipeline.analyze(str(video), time_budget_sec=240.0)

    assert seen == [1, 2]
    assert pipeline._SHARED.active_jobs() == 0


# --- analyze ----------------------------------------------------------------------------------------


def test_analyze_with_stub_rules_is_no_violation(clip, fake):
    assert pipeline.analyze(str(clip)) == {"verdict": "no_violation", "violations": []}


def test_analyze_renders_candidates_into_valid_k1(clip, fake, monkeypatch):
    seen = capture_rules(monkeypatch, [one_candidate()])
    progress = []
    result = pipeline.analyze(str(clip), on_progress=lambda p, s: progress.append((p, s)))

    AnalysisResult.model_validate(result)
    assert result["verdict"] == "violation"
    [v] = result["violations"]
    assert v["type_id"] == "cash_left_on_desk_when_leaving"
    assert "Основание:" in v["description"]
    assert base64.b64decode(v["evidence_frames"][0]["image_base64"])[:3] == b"\xff\xd8\xff"

    assert isinstance(seen["perception"], Perception) and seen["scene"] is None
    assert len(seen["perception"].frames) == pytest.approx(3.0 * pipeline.MAX_SAMPLE_FPS, abs=1)

    values = [p for p, _ in progress]
    assert values == sorted(values) and values[-1] == 1.0
    assert {s for _, s in progress} >= {"detect", "rules", "evidence"}
    assert max(p for p, s in progress if s == "detect") <= pipeline.PROGRESS_DETECT_END


def test_slow_detector_is_planned_to_finish_within_budget(tmp_path, monkeypatch):
    video = make_video(tmp_path / "long.mp4", seconds=30.0)
    slow = use_detector(monkeypatch, FakeDetector(sec_per_frame=0.02))
    monkeypatch.setattr(pipeline, "FINISH_RESERVE_SEC", 0.0)
    seen = capture_rules(monkeypatch, [one_candidate()])

    started = time.monotonic()
    result = pipeline.analyze(str(video), time_budget_sec=2.0)

    AnalysisResult.model_validate(result)
    assert time.monotonic() - started < 2.5
    assert slow.frames_seen < 30 * pipeline.MAX_SAMPLE_FPS  # sampling was thinned out up front
    assert seen["perception"].frames  # and something was analysed


def test_budget_overrun_returns_partial_result_instead_of_raising(tmp_path, monkeypatch):
    video = make_video(tmp_path / "long.mp4", seconds=60.0)
    slow = use_detector(monkeypatch, FakeDetector(sec_per_frame=0.03))
    monkeypatch.setattr(pipeline, "FINISH_RESERVE_SEC", 0.0)
    seen = capture_rules(monkeypatch, [one_candidate(t=1.0)])

    started = time.monotonic()
    result = pipeline.analyze(str(video), time_budget_sec=1.5)  # even the sparsest plan does not fit

    assert time.monotonic() - started < 2.5
    AnalysisResult.model_validate(result)
    assert result["verdict"] == "violation"  # what was seen before the deadline is still reported
    p = seen["perception"]
    assert 0 < len(p.frames) < 60 * pipeline.HARD_MIN_FPS
    assert p.info.duration < 60.0
    assert p.frames[-1].t <= p.info.duration
    assert all(e.t <= p.info.duration for e in p.events)
    assert slow.frames_seen < 60 * pipeline.HARD_MIN_FPS + 2 * CALIBRATION_FRAMES


def test_budget_exhausted_before_any_detection_gives_no_violation(clip, fake, monkeypatch):
    seen = capture_rules(monkeypatch, [one_candidate()])
    monkeypatch.setattr(pipeline, "FINISH_RESERVE_SEC", 10.0)
    result = pipeline.analyze(str(clip), time_budget_sec=1.0)
    assert result == {"verdict": "no_violation", "violations": []}
    assert "perception" not in seen


def test_parallel_jobs_share_one_model_and_never_infer_concurrently(tmp_path, monkeypatch):
    created = []

    def factory():
        time.sleep(0.2)  # slow init widens the race window
        created.append(FakeDetector(sec_per_frame=0.002))
        return created[-1]

    monkeypatch.setattr(pipeline, "_SHARED", SharedDetector(factory=factory))
    videos = [make_video(tmp_path / f"v{i}.mp4", seconds=4.0) for i in range(2)]
    errors = []

    def job(path):
        try:
            AnalysisResult.model_validate(pipeline.analyze(str(path), time_budget_sec=60.0))
        except Exception as exc:  # noqa: BLE001 — surfaced through the assertion below
            errors.append(exc)

    threads = [threading.Thread(target=job, args=(v,)) for v in videos]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not any(t.is_alive() for t in threads)
    assert not errors
    assert len(created) == 1
    assert created[0].max_active == 1


# --- robustness -------------------------------------------------------------------------------------


def test_unreadable_video_raises(tmp_path, fake):
    bad = tmp_path / "bad.mov"
    bad.write_bytes(b"not a video")
    with pytest.raises(ValueError):
        pipeline.analyze(str(bad))


def test_broken_kb_fails_before_detection(clip, fake, monkeypatch):
    def broken():
        raise KBError("broken catalog")

    monkeypatch.setattr(pipeline, "catalog", broken)
    with pytest.raises(KBError):
        pipeline.analyze(str(clip))
    assert fake.calls == 0


def test_video_without_decodable_frames_fails_instead_of_clean_verdict(clip, fake, monkeypatch):
    real_perceive = pipeline.perceive
    monkeypatch.setattr(pipeline, "perceive",
                        lambda *a, **kw: dataclasses.replace(real_perceive(*a, **kw), frames=()))
    with pytest.raises(ValueError, match="frames"):
        pipeline.analyze(str(clip))


def test_undecodable_tail_is_not_read_as_an_empty_room(tmp_path, fake, monkeypatch):
    video = make_video(tmp_path / "v.mp4", seconds=10.0)
    real_perceive = pipeline.perceive

    def tail_lost(*args, **kwargs):
        p = real_perceive(*args, **kwargs)
        return dataclasses.replace(p, frames=tuple(f for f in p.frames if f.t < 4.0))

    monkeypatch.setattr(pipeline, "perceive", tail_lost)
    seen = capture_rules(monkeypatch)
    pipeline.analyze(str(video))
    assert seen["perception"].info.duration < 5.0


def test_failed_evidence_falls_back_to_interval_edges(clip, fake, monkeypatch):
    capture_rules(monkeypatch, [one_candidate(t=1.0)])
    real_render = pipeline.render_evidence
    calls = []

    def flaky(video_path, times, boxes=()):
        calls.append(tuple(times))
        if len(calls) == 1:
            raise cv2.error("seek failed")
        return real_render(video_path, times, boxes)

    monkeypatch.setattr(pipeline, "render_evidence", flaky)
    [v] = pipeline.analyze(str(clip))["violations"]
    assert calls == [(1.0,), (0.5, 1.5)]
    assert [f["timestamp_sec"] for f in v["evidence_frames"]] == [0.5, 1.5]


def test_one_unrenderable_candidate_does_not_sink_the_others(clip, fake, monkeypatch):
    capture_rules(monkeypatch, [one_candidate(t=0.5), one_candidate(type_id="door_left_open", t=2.0)])
    real_render = pipeline.render_evidence

    def broken_early(video_path, times, boxes=()):
        if min(times) < 1.0:
            raise ValueError("cannot encode")
        return real_render(video_path, times, boxes)

    monkeypatch.setattr(pipeline, "render_evidence", broken_early)
    result = pipeline.analyze(str(clip))
    AnalysisResult.model_validate(result)
    assert [v["type_id"] for v in result["violations"]] == ["door_left_open"]


# --- real model -------------------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(not REAL_VIDEO.exists(), reason="data/open link is missing")
def test_real_video_end_to_end(monkeypatch):
    capture_rules(monkeypatch, [one_candidate(t=10.0)])
    started = time.monotonic()
    result = pipeline.analyze(str(REAL_VIDEO), time_budget_sec=200.0)
    elapsed = time.monotonic() - started

    parsed = AnalysisResult.model_validate(result)
    assert parsed.verdict == "violation"
    frame = parsed.violations[0].evidence_frames[0]
    assert len(frame.image_base64) <= 150 * 1024
    assert elapsed < 200.0
