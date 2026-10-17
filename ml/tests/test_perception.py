import numpy as np

from ml.detect import Detection
from ml.perception import FrameObjects, analyze_frames, observe

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
