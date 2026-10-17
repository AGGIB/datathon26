from ml.track import Track, TrackPoint, drop_duplicate_tracks, stitch_tracks

SIG_A = (1.0, 0.0)
SIG_B = (0.0, 1.0)


def trk(tid, t0, t1, box=(0.0, 0.0, 10.0, 20.0), step=0.5, sig=SIG_A, dx_per_sec=0.0, score=0.9):
    n = int(round((t1 - t0) / step)) + 1
    pts = []
    for i in range(n):
        t = round(t0 + i * step, 3)
        dx = dx_per_sec * (t - t0)
        pts.append(TrackPoint(t=t, box=(box[0] + dx, box[1], box[2] + dx, box[3]), score=score, signature=sig))
    return Track(track_id=tid, points=tuple(pts))


def ids(tracks):
    return sorted(tr.track_id for tr in tracks)


def test_fragments_with_short_gap_at_same_place_are_joined():
    out = stitch_tracks([trk(1, 0.0, 3.0), trk(2, 4.0, 8.0)], max_gap_sec=2.0)
    assert ids(out) == [1]
    assert out[0].t_start == 0.0 and out[0].t_end == 8.0


def test_gap_longer_than_limit_is_not_joined():
    out = stitch_tracks([trk(1, 0.0, 3.0), trk(2, 5.5, 8.0)], max_gap_sec=2.0)
    assert ids(out) == [1, 2]


def test_fragments_far_apart_are_not_joined():
    far = (200.0, 0.0, 210.0, 20.0)
    out = stitch_tracks([trk(1, 0.0, 3.0), trk(2, 4.0, 8.0, box=far)], max_gap_sec=2.0)
    assert ids(out) == [1, 2]


def test_tracks_overlapping_in_time_are_never_joined():
    out = stitch_tracks([trk(1, 0.0, 5.0), trk(2, 4.0, 8.0)], max_gap_sec=2.0)
    assert ids(out) == [1, 2]


def test_different_appearance_blocks_joining():
    out = stitch_tracks([trk(1, 0.0, 3.0, sig=SIG_A), trk(2, 4.0, 8.0, sig=SIG_B)], max_gap_sec=2.0)
    assert ids(out) == [1, 2]


def test_chain_of_three_fragments_becomes_one_track():
    out = stitch_tracks([trk(1, 0.0, 2.0), trk(2, 3.0, 5.0), trk(3, 6.0, 9.0)], max_gap_sec=2.0)
    assert ids(out) == [1]
    assert [p.t for p in out[0].points] == sorted(p.t for p in out[0].points)


def test_each_fragment_continues_the_nearest_candidate():
    left = (0.0, 0.0, 10.0, 20.0)
    right = (14.0, 0.0, 24.0, 20.0)
    tracks = [trk(1, 0.0, 3.0, box=left), trk(2, 0.0, 3.0, box=right), trk(3, 3.5, 6.0, box=right)]
    out = {tr.track_id: tr for tr in stitch_tracks(tracks, max_gap_sec=2.0)}
    assert sorted(out) == [1, 2]
    assert out[2].t_end == 6.0 and out[1].t_end == 3.0


def test_moving_person_is_joined_along_motion():
    out = stitch_tracks([trk(1, 0.0, 3.0, dx_per_sec=5.0), trk(2, 4.0, 6.0, box=(20.0, 0.0, 30.0, 20.0))])
    assert ids(out) == [1]


def test_duplicate_box_on_same_person_is_dropped():
    big = trk(1, 0.0, 10.0, box=(0.0, 0.0, 10.0, 20.0))
    inner = trk(2, 2.0, 8.0, box=(1.0, 1.0, 9.0, 12.0))
    assert ids(drop_duplicate_tracks([big, inner])) == [1]


def test_two_people_side_by_side_are_both_kept():
    a = trk(1, 0.0, 10.0, box=(0.0, 0.0, 10.0, 20.0))
    b = trk(2, 0.0, 10.0, box=(8.0, 0.0, 18.0, 20.0))
    assert ids(drop_duplicate_tracks([a, b])) == [1, 2]
