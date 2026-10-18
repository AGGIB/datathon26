import base64
import dataclasses
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from backend.app.schemas import AnalysisResult
import ml.evidence as evidence
from ml import pipeline
from ml.candidates import Candidate
from ml.detect import INPUT_SIZE, Detection
from ml.kb import KBError
from ml.perception import Perception
from ml.shared_detector import CALIBRATION_FRAMES, SharedDetector


@pytest.fixture(autouse=True)
def _scene_off_by_default(monkeypatch):
    """Unit tests run the pipeline without the OWLv2 scene stage unless a test enables it."""
    monkeypatch.setenv("ML_SCENE", "0")

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
    cost_at, _ = costs(0.35, 0.17)  # fits ~2.6 fps at full size: between SAFE_MIN_FPS and the 3 fps ceiling
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


def test_budget_exhausted_before_any_detection_raises(clip, fake, monkeypatch):
    seen = capture_rules(monkeypatch, [one_candidate()])
    monkeypatch.setattr(pipeline, "FINISH_RESERVE_SEC", 10.0)
    with pytest.raises(RuntimeError, match="no frames were analysed"):
        pipeline.analyze(str(clip), time_budget_sec=1.0)
    assert "perception" not in seen
    assert fake.frames_seen == 0
    assert pipeline._SHARED.active_jobs() == 0


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


class PersonUntil(FakeDetector):
    """A person is visible while the synthetic frame is darker than `brightness` (i.e. early in the clip)."""

    def __init__(self, brightness: float) -> None:
        super().__init__()
        self.brightness = brightness

    def detect(self, frames):
        self.calls += 1
        return [[PERSON] if f.mean() < self.brightness else [] for f in frames]


def lose_tail_after(monkeypatch, t_cut: float) -> None:
    real_perceive = pipeline.perceive

    def tail_lost(*args, **kwargs):
        p = real_perceive(*args, **kwargs)
        return dataclasses.replace(p, frames=tuple(f for f in p.frames if f.t < t_cut))

    monkeypatch.setattr(pipeline, "perceive", tail_lost)


def test_undecodable_tail_is_not_read_as_an_empty_room(tmp_path, fake, monkeypatch):
    video = make_video(tmp_path / "v.mp4", seconds=10.0)
    lose_tail_after(monkeypatch, 4.0)
    seen = capture_rules(monkeypatch)
    pipeline.analyze(str(video))
    p = seen["perception"]
    assert p.info.duration < 5.0
    assert not [e for e in p.events if e.kind in ("exit", "room_empty")]  # the person is there until the cut


def test_real_exit_before_lost_tail_is_kept_and_nothing_is_invented_at_the_cut(tmp_path, monkeypatch):
    video = make_video(tmp_path / "v.mp4", seconds=10.0)  # brightness = 3 * frame index
    use_detector(monkeypatch, PersonUntil(brightness=60))  # person visible for the first ~2 s
    lose_tail_after(monkeypatch, 6.0)
    seen = capture_rules(monkeypatch)
    pipeline.analyze(str(video))
    p = seen["perception"]
    exits = [e.t for e in p.events if e.kind == "exit"]
    assert exits and max(exits) < 2.5
    assert all(e.t <= p.info.duration and (e.t_end or 0.0) <= p.info.duration for e in p.events)


def test_reported_times_never_leave_the_analysed_part(tmp_path, fake, monkeypatch):
    video = make_video(tmp_path / "v.mp4", seconds=10.0)
    lose_tail_after(monkeypatch, 4.0)
    capture_rules(monkeypatch, [one_candidate(t=8.0)])  # rules pointing past what was analysed
    result = pipeline.analyze(str(video))
    [v] = result["violations"]
    assert v["timecode_sec"] <= 4.0 and v["t_end"] <= 4.0
    assert all(f["timestamp_sec"] <= 4.0 for f in v["evidence_frames"])


def test_evidence_for_all_violations_comes_from_one_decode_pass(clip, fake, monkeypatch):
    capture_rules(monkeypatch, [one_candidate(t=0.5), one_candidate(type_id="door_left_open", t=2.0)])
    calls = []
    real = evidence.frames_at
    monkeypatch.setattr(evidence, "frames_at", lambda path, times: calls.append(sorted(times)) or real(path, times))
    result = pipeline.analyze(str(clip))
    AnalysisResult.model_validate(result)
    assert len(result["violations"]) == 2
    assert len(calls) == 1 and {0.5, 2.0} <= set(calls[0])


@pytest.mark.parametrize('fault_site', ['frames_at', 'encode_jpeg_b64'])
def test_failed_evidence_raises_instead_of_clean_verdict(clip, fake, monkeypatch, caplog, fault_site):
    capture_rules(monkeypatch, [one_candidate(t=0.5)])

    def broken(*args):
        raise ValueError("synthetic evidence failure")

    monkeypatch.setattr(evidence, fault_site, broken)
    with caplog.at_level("ERROR"), pytest.raises(RuntimeError, match="lost their evidence"):
        pipeline.analyze(str(clip))
    assert {r.name for r in caplog.records if r.levelname == "ERROR"} >= {"ml.evidence", "ml.assemble"}
    assert pipeline._SHARED.active_jobs() == 0


@pytest.mark.parametrize('fault', ['zero_frames', 'decode', 'encode'])
def test_real_pipeline_fault_reaches_existing_failed_api(clip, fake, monkeypatch, tmp_path, fault):
    from backend.app.config import Settings
    from backend.app.main import create_app
    from fastapi.testclient import TestClient

    capture_rules(monkeypatch, [one_candidate(t=0.5)])
    settings = Settings(api_keys=frozenset({'test-key'}), data_dir=tmp_path / 'api',
                        workers=1, pipeline='mock', job_timeout_sec=30)

    def broken(*args):
        raise ValueError('synthetic evidence failure')

    with TestClient(create_app(settings, pipeline_fn=pipeline.analyze)) as client:
        def submit():
            response = client.post('/api/v1/analyze', headers={'X-API-Key': 'test-key'},
                                   files={'file': ('synthetic.mp4', clip.read_bytes(), 'video/mp4')})
            assert response.status_code == 200
            job_id = response.json()['job_id']
            client.app.state.runner.wait(job_id, timeout=10)
            result = client.get(f'/api/v1/jobs/{job_id}', headers={'X-API-Key': 'test-key'})
            assert result.status_code == 200
            return job_id, result.json()

        with monkeypatch.context() as fault_patch:
            if fault == 'zero_frames':
                fault_patch.setattr(pipeline, 'FINISH_RESERVE_SEC', 1e6)
            else:
                fault_patch.setattr(evidence, 'frames_at' if fault == 'decode' else 'encode_jpeg_b64', broken)
            job_id, failed = submit()
            assert failed['status'] == 'failed' and failed['error']
            assert {'verdict', 'violations', 'result'}.isdisjoint(failed)
            stored = client.app.state.store.get(job_id)
            assert stored['stage'] == 'failed' and stored.get('result') is None
        _, recovered = submit()
        assert recovered['status'] == 'done' and recovered['verdict'] == 'violation'
        assert client.app.state.runner.lost_slots == 0
        assert pipeline._SHARED.active_jobs() == 0


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


def test_truncated_perception_keeps_role_events_of_the_analysed_prefix():
    from ml.perception import FrameObjects, analyze_frames
    from ml.video import VideoInfo

    seat = (100.0, 300.0, 200.0, 500.0)
    frames = tuple(
        FrameObjects(round(i / 3, 3), (Detection(seat, 0.9, "person"),) if i / 3 <= 10 else (), ((0.5, 0.5),)
                     if i / 3 <= 10 else ())
        for i in range(3 * 30 + 1)
    )
    info = VideoInfo(fps=30.0, frame_count=901, width=300, height=600)
    scene = analyze_frames(frames, info.duration, 3.0)
    full = Perception(info, frames, scene.roles.in_room, scene.events, 1.0, scene.roles, scene.all_tracks, 3.0)
    cut = pipeline._truncate(full, n_frames=3 * 16 + 1)  # analysed up to 16 s
    assert [e.t for e in cut.events if e.kind == "cashier_left"] == [10.0]
    assert cut.roles is not None and cut.roles.cashier_ids and cut.info.duration <= 16.1
    assert all(p.t <= 16.0 for tr in cut.all_tracks for p in tr.points)


# --- scene stage wiring -------------------------------------------------------------------------

def test_scene_failure_falls_back_to_rules_without_scene(monkeypatch):
    monkeypatch.setenv("ML_SCENE", "1")
    seen = {}

    def boom(*a, **k):
        raise RuntimeError("owl exploded")

    def fake_rules(perception, scene=None):
        seen["scene"] = scene
        return []

    monkeypatch.setattr(pipeline, "_analyze_scene", boom)
    monkeypatch.setattr(pipeline.rules, "find_candidates", fake_rules)
    perception = SimpleNamespace(frames=(1,))
    assert pipeline._judge("x.mov", perception, scene_deadline=None) == []
    assert seen["scene"] is None


def test_scene_skipped_when_too_little_time_left(monkeypatch):
    monkeypatch.setenv("ML_SCENE", "1")
    called = {"scene": False}

    def scene(*a, **k):
        called["scene"] = True

    monkeypatch.setattr(pipeline, "_analyze_scene", scene)
    monkeypatch.setattr(pipeline.rules, "find_candidates", lambda perception, scene=None: [])
    perception = SimpleNamespace(frames=(1,))
    pipeline._judge("x.mov", perception, scene_deadline=time.monotonic() + 1.0)
    assert called["scene"] is False


def test_scene_result_is_passed_to_rules(monkeypatch):
    monkeypatch.setenv("ML_SCENE", "1")
    marker = object()
    seen = {}
    def analyze_scene(video_path, perception, deadline=None):
        seen["deadline"] = deadline
        return marker

    monkeypatch.setattr(pipeline, "_analyze_scene", analyze_scene)
    monkeypatch.setattr(pipeline.rules, "find_candidates",
                        lambda perception, scene=None: seen.setdefault("scene", scene) and [])
    deadline = time.monotonic() + 60.0
    pipeline._judge("x.mov", SimpleNamespace(frames=(1,)), scene_deadline=deadline)
    assert seen["scene"] is marker
    assert seen["deadline"] == deadline


def test_scene_can_be_disabled_by_env(monkeypatch):
    monkeypatch.setenv("ML_SCENE", "0")
    monkeypatch.setattr(pipeline, "_analyze_scene", lambda *a, **k: (_ for _ in ()).throw(AssertionError("called")))
    monkeypatch.setattr(pipeline.rules, "find_candidates", lambda perception, scene=None: [])
    assert pipeline._judge("x.mov", SimpleNamespace(frames=(1,)), scene_deadline=None) == []


def test_scene_receives_the_absolute_deadline(monkeypatch):
    import ml.scene

    marker, owl = object(), object()
    seen = {}
    lock = threading.Lock()
    monkeypatch.setattr(pipeline, "_OWL_LOCK", lock)
    monkeypatch.setattr(pipeline, "_owl", lambda: owl)

    def analyze_scene(video_path, perception, **kwargs):
        seen.update(kwargs)
        assert lock.locked()
        return marker

    monkeypatch.setattr(ml.scene, "analyze_scene", analyze_scene)
    deadline = time.monotonic() + 60.0
    assert pipeline._analyze_scene("x.mov", SimpleNamespace(), deadline=deadline) is marker
    assert seen == {"owl": owl, "deadline": deadline}
    assert not lock.locked()


def test_scene_lock_wait_counts_towards_the_deadline(monkeypatch):
    lock = threading.Lock()
    lock.acquire()  # another worker owns the shared model
    monkeypatch.setattr(pipeline, "_OWL_LOCK", lock)
    monkeypatch.setattr(pipeline, "_owl", lambda: pytest.fail("must not load a model after lock timeout"))
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError, match="waiting"):
            pipeline._analyze_scene("x.mov", SimpleNamespace(), deadline=started + 0.03)
        assert time.monotonic() - started < 0.5
        assert lock.locked()  # the waiting worker must not release the other worker's slot
    finally:
        lock.release()


def test_scene_rechecks_remaining_time_after_acquiring_the_lock(monkeypatch):
    now = [10.0]

    class BusyLock:
        released = False

        def acquire(self, timeout):
            assert timeout == 20.0
            now[0] = 20.0  # ten seconds were spent waiting for the other worker
            return True

        def release(self):
            self.released = True

    lock = BusyLock()
    monkeypatch.setattr(pipeline, "_OWL_LOCK", lock)
    monkeypatch.setattr(pipeline.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(pipeline, "_owl", lambda: pytest.fail("not enough time left to start scene inference"))
    with pytest.raises(TimeoutError, match="after waiting"):
        pipeline._analyze_scene("x.mov", SimpleNamespace(), deadline=30.0)
    assert lock.released


def test_scene_timeout_keeps_event_candidates_and_releases_the_model(monkeypatch, caplog):
    import ml.scene

    monkeypatch.setenv("ML_SCENE", "1")
    lock = threading.Lock()
    monkeypatch.setattr(pipeline, "_OWL_LOCK", lock)
    monkeypatch.setattr(pipeline, "_owl", object)

    def timeout(*args, **kwargs):
        raise TimeoutError("deadline reached between keyframes")

    monkeypatch.setattr(ml.scene, "analyze_scene", timeout)
    candidate = one_candidate()
    seen = capture_rules(monkeypatch, [candidate])
    perception = SimpleNamespace(frames=(1,))
    assert pipeline._judge("x.mov", perception, scene_deadline=time.monotonic() + 60.0) == [candidate]
    assert seen["scene"] is None
    assert not lock.locked()
    assert "scene budget exhausted" in caplog.text
    assert not [r for r in caplog.records if r.levelno >= 40]


def test_shared_owl_wrapper_leaves_cold_loading_to_deadline_aware_inference(monkeypatch):
    import ml.scene_owl

    class LazyOwl:
        def load(self):
            pytest.fail("model loading must happen in the deadline-aware scene code")

    monkeypatch.setattr(ml.scene_owl, "OwlDetector", LazyOwl)
    monkeypatch.setattr(pipeline, "_OWL", None)
    first = pipeline._owl()
    assert isinstance(first, LazyOwl)
    assert pipeline._owl() is first


def test_scene_deadline_wiring_uses_the_real_scene_api(clip, fake, monkeypatch):
    from ml.scene import SceneAnalysis

    monkeypatch.setenv("ML_SCENE", "1")
    monkeypatch.setattr(pipeline, "_owl", lambda: None)  # statistics only, no model or downloads
    monkeypatch.setattr(pipeline, "_OWL_LOCK", threading.Lock())
    seen = capture_rules(monkeypatch)
    result = pipeline.analyze(str(clip), time_budget_sec=200.0)
    assert result == {"verdict": "no_violation", "violations": []}
    assert isinstance(seen["scene"], SceneAnalysis)
    assert seen["scene"].states
