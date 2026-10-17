from ml.detect import Detection
from ml.track import IoUTracker, iou


def det(x0, y0, x1, y1, score=0.9, label="person"):
    return Detection(box=(x0, y0, x1, y1), score=score, label=label)


def test_iou_identical_and_disjoint():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_partial_overlap():
    assert abs(iou((0, 0, 10, 10), (5, 0, 15, 10)) - 50 / 150) < 1e-9


def test_same_object_keeps_id_across_frames():
    tr = IoUTracker()
    a = tr.update(0.0, [det(0, 0, 10, 20)])
    b = tr.update(0.5, [det(1, 0, 11, 20)])
    assert a[0][0] == b[0][0]


def test_new_object_gets_new_id():
    tr = IoUTracker()
    tr.update(0.0, [det(0, 0, 10, 20)])
    out = tr.update(0.5, [det(0, 0, 10, 20), det(100, 0, 110, 20)])
    assert len({tid for tid, _ in out}) == 2


def test_track_survives_short_gap_but_not_long_one():
    tr = IoUTracker(max_gap_sec=1.0)
    first = tr.update(0.0, [det(0, 0, 10, 20)])[0][0]
    tr.update(0.5, [])
    assert tr.update(1.0, [det(0, 0, 10, 20)])[0][0] == first
    tr.update(1.5, [])
    tr.update(2.0, [])
    tr.update(2.5, [])
    assert tr.update(3.0, [det(0, 0, 10, 20)])[0][0] != first


def test_finished_tracks_expose_full_history():
    tr = IoUTracker()
    for t in (0.0, 0.5, 1.0):
        tr.update(t, [det(0, 0, 10, 20)])
    tracks = tr.tracks()
    assert len(tracks) == 1
    assert tracks[0].t_start == 0.0 and tracks[0].t_end == 1.0
    assert len(tracks[0].points) == 3


def test_non_person_detections_are_ignored():
    tr = IoUTracker()
    assert tr.update(0.0, [det(0, 0, 10, 20, label="laptop")]) == []
