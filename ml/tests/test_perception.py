import numpy as np
import pytest

from ml.detect import Detection
from ml.perception import FrameObjects, analyze_frames, observe, tolerances

PINK = (0.5, 0.5, 0.0, 0.0)
DARK = (0.0, 0.0, 0.5, 0.5)
SEAT = (100.0, 300.0, 200.0, 500.0)
VISITOR = (100.0, 150.0, 170.0, 300.0)
GLASS = (0.0, 0.0, 40.0, 80.0)
LAPTOP = (120.0, 300.0, 180.0, 340.0)
STEP = 0.25


def person(box, score=0.9):
    return Detection(box=box, score=score, label="person")


def scene(duration, cashier_until, visitor_span):
    frames = []
    for i in range(int(duration / STEP) + 1):
        t = round(i * STEP, 3)
        dets, sigs = [Detection(LAPTOP, 0.8, "laptop")], [()]
        if t <= cashier_until:
            dets.append(person(SEAT))
            sigs.append(PINK)
        if visitor_span[0] <= t <= visitor_span[1]:
            dets.append(person(VISITOR, 0.88))
            sigs.append(DARK)
        dets.append(person(GLASS, 0.85))  # someone sitting behind the glass the whole time
        sigs.append(DARK)
        frames.append(FrameObjects(t, tuple(dets), tuple(sigs)))
    return frames


def kinds(events):
    return [(e.kind, e.t) for e in events]


def test_observe_signs_people_only():
    frame = np.full((600, 300, 3), 120, dtype=np.uint8)
    fo = observe(1.0, frame, [person(SEAT), Detection(LAPTOP, 0.8, "laptop")])
    assert len(fo.signatures) == 2 and fo.signatures[0] and fo.signatures[1] == ()


def test_scene_roles_and_events_end_to_end():
    out = analyze_frames(scene(30.0, cashier_until=20.0, visitor_span=(5.0, 15.0)), duration=30.0)
    assert len(out.roles.cashier) == 1 and len(out.roles.visitors) == 1
    assert out.roles.workplace.desk is not None
    ev = kinds(out.events)
    assert ("visitor_enter", 5.0) in ev and ("visitor_exit", 15.0) in ev
    assert ("cashier_left", 20.0) in ev
    assert not [k for k, _ in ev if k in ("crowd", "visitors_overlap", "person_behind_cashier")]
    # the person behind the glass never produces room events
    glass_id = next(tr.track_id for tr in out.all_tracks if tr.points[0].box == GLASS)
    assert all(glass_id not in e.track_ids and e.track_id != glass_id for e in out.events)


def test_empty_detections_give_an_empty_room():
    frames = [FrameObjects(round(i * STEP, 3), ()) for i in range(41)]
    out = analyze_frames(frames, duration=10.0)
    assert kinds(out.events) == [("room_empty_at_start", 0.0)]


def lone_cashier(fps, duration, missing):
    """One seated person sampled at `fps`; detections at the sample indexes in `missing` are lost."""
    frames = []
    for i in range(int(duration * fps) + 1):
        t = round(i / fps, 3)
        dets = () if i in missing else (person(SEAT),)
        frames.append(FrameObjects(t, dets, tuple(PINK for _ in dets)))
    return frames


@pytest.mark.parametrize("fps", [1.25, 2.2, 4.0])
@pytest.mark.parametrize("missed", [1, 2])
def test_missed_detections_at_any_sampling_rate_are_not_an_empty_room(fps, missed):
    middle = int(10 * fps)
    frames = lone_cashier(fps, 20.0, missing=set(range(middle, middle + missed)))
    out = analyze_frames(frames, duration=20.0, sample_fps=fps)
    assert not [e for e in out.events if e.kind in ("room_empty", "exit", "enter", "cashier_left")]


def test_perception_tolerances_follow_the_sampling_interval():
    slow, fast = tolerances(1.25), tolerances(4.0)
    assert slow.tracker_gap_sec == pytest.approx(2.0) and fast.tracker_gap_sec == pytest.approx(1.0)
    assert slow.min_empty_sec > slow.tracker_gap_sec + 1 / 1.25 - 1e-9
    assert fast.min_empty_sec == pytest.approx(2.0)
    assert slow.stitch_gap_sec > slow.tracker_gap_sec


class FakeDetector:
    input_size = 320

    def detect(self, frames):
        return [[person(SEAT)] for _ in frames]


@pytest.fixture
def tiny_video(tmp_path):
    import cv2

    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (300, 600))
    for _ in range(30):
        writer.write(np.full((600, 300, 3), 90, dtype=np.uint8))
    writer.release()
    return path


def test_perceive_records_sampling_rate_and_detector_resolution(tiny_video):
    from ml.perception import perceive

    out = perceive(tiny_video, FakeDetector(), sample_fps=2.0)
    assert out.sample_fps == 2.0 and out.input_size == 320
    assert len(out.frames) == 6 and out.roles.cashier_ids
    assert perceive(tiny_video, FakeDetector(), sample_fps=2.0, input_size=640).input_size == 640
