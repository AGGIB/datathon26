import pytest

from ml.detect import Detection
from ml.roles import (
    Workplace,
    assign_roles,
    desk_point,
    is_handoff,
    is_mirror,
    near_span,
    pick_anchor,
    workplace_of,
)
from ml.track import Track, TrackPoint

PINK = (0.5, 0.5, 0.0, 0.0)
DARK = (0.0, 0.0, 0.5, 0.5)
GREY = (0.25, 0.25, 0.25, 0.25)
SEAT = (100.0, 300.0, 200.0, 500.0)  # cashier: foot (150, 500), width 100, height 200
STEP = 0.5


def trk(tid, t0, t1, box, score=0.9, sig=PINK):
    n = int(round((t1 - t0) / STEP)) + 1
    return Track(tid, tuple(TrackPoint(round(t0 + i * STEP, 3), box, score, sig) for i in range(n)))


def moving(tid, t0, t1, start_box, end_box, score=0.9, sig=PINK):
    n = int(round((t1 - t0) / STEP)) + 1
    pts = []
    for i in range(n):
        a = i / max(n - 1, 1)
        box = tuple(s + (e - s) * a for s, e in zip(start_box, end_box))
        pts.append(TrackPoint(round(t0 + i * STEP, 3), box, score, sig))
    return Track(tid, tuple(pts))


WP = Workplace(home=(150.0, 500.0), box=SEAT, width=100.0, height=200.0, signature=PINK)
DESK_FAR_SIDE = (100.0, 150.0, 170.0, 300.0)  # visitor across the desk: foot (135, 300) = 2 widths away
BEHIND_GLASS = (0.0, 0.0, 40.0, 80.0)  # small, far: foot (20, 80) = 4.4 widths away


def test_anchor_is_the_big_long_clearly_visible_track():
    cashier = trk(1, 0, 30, SEAT)
    through_glass = trk(2, 0, 30, BEHIND_GLASS, score=0.85)
    visitor = trk(3, 5, 15, DESK_FAR_SIDE)
    assert pick_anchor([through_glass, visitor, cashier]) is cashier
    assert pick_anchor([]) is None


def test_workplace_is_where_the_cashier_sits_most_of_the_time():
    track = Track(1, trk(1, 0, 20, SEAT).points + moving(1, 20.5, 25, SEAT, (300.0, 0.0, 360.0, 120.0)).points)
    wp = workplace_of(track)
    assert wp.home == pytest.approx((150.0, 500.0))
    assert wp.width == pytest.approx(100.0) and wp.height == pytest.approx(200.0)
    assert wp.signature == pytest.approx(PINK)


def test_workplace_distance_is_in_cashier_widths():
    assert WP.distance(DESK_FAR_SIDE) == pytest.approx(((15**2 + 200**2) ** 0.5) / 100)
    assert WP.occupied_by(SEAT) and not WP.occupied_by(DESK_FAR_SIDE) and not WP.occupied_by(SEAT, score=0.5)


def test_weaker_look_alike_living_alongside_is_a_mirror():
    real = trk(1, 0, 20, SEAT, score=0.9)
    reflection = trk(2, 2, 18, BEHIND_GLASS, score=0.7)
    assert is_mirror(reflection, real)
    assert not is_mirror(trk(3, 2, 18, BEHIND_GLASS, score=0.7, sig=DARK), real)
    assert not is_mirror(trk(4, 2, 18, BEHIND_GLASS, score=0.95), real)


def test_handoff_needs_same_place_and_moment():
    old = trk(1, 0, 5, SEAT)
    assert is_handoff(old, trk(2, 4.5, 10, SEAT), WP)
    assert not is_handoff(old, trk(3, 9.0, 10, SEAT), WP)
    assert not is_handoff(old, trk(4, 4.5, 10, BEHIND_GLASS), WP)


def test_near_span_trims_the_walk_outside_the_room():
    walk = Track(5, trk(5, 2, 5, BEHIND_GLASS).points + trk(5, 5.5, 12, DESK_FAR_SIDE).points)
    span = near_span(walk, WP)
    assert span.t_start == 5.5 and span.t_end == 12.0
    assert near_span(trk(6, 0, 3, BEHIND_GLASS), WP) is None


def test_roles_split_cashier_visitor_and_people_behind_the_glass():
    cashier = trk(1, 0, 30, SEAT)
    visitor = trk(2, 5, 20, DESK_FAR_SIDE, score=0.88, sig=DARK)
    corridor = trk(3, 0, 30, BEHIND_GLASS, score=0.85, sig=DARK)
    weak_near = trk(4, 8, 20, DESK_FAR_SIDE, score=0.6, sig=GREY)
    roles = assign_roles([cashier, visitor, corridor, weak_near])
    assert roles.cashier_ids == (1,)
    assert roles.visitor_ids == (2,)
    assert dict(roles.rejected) == {3: "outside", 4: "outside"}


def test_cashier_coming_back_on_a_new_track_is_recognised_by_look():
    roles = assign_roles([trk(1, 0, 10, SEAT), trk(2, 20, 30, SEAT, sig=PINK), trk(3, 12, 18, DESK_FAR_SIDE, sig=DARK)])
    assert roles.cashier_ids == (1, 2)
    assert roles.visitor_ids == (3,)


def test_cashier_reflection_is_rejected_as_mirror():
    roles = assign_roles([trk(1, 0, 30, SEAT), trk(2, 0, 30, DESK_FAR_SIDE, score=0.82)])
    assert roles.visitor_ids == ()
    assert dict(roles.rejected) == {2: "mirror"}


def test_cashier_walking_in_on_a_separate_track_is_absorbed():
    walk_in = moving(5, 0, 3, (300.0, 0.0, 360.0, 120.0), SEAT, score=0.8, sig=DARK)
    roles = assign_roles([walk_in, trk(1, 2.5, 30, SEAT)])
    assert roles.cashier_ids == (5,)
    assert roles.cashier[0].t_start == 0.0 and roles.cashier[0].t_end == 30.0
    assert roles.visitor_ids == ()


def test_visitor_re_acquired_as_new_track_stays_one_visitor():
    a = trk(2, 2, 10, DESK_FAR_SIDE, sig=DARK)
    b = trk(3, 9.5, 20, DESK_FAR_SIDE, sig=DARK)
    roles = assign_roles([trk(1, 0, 30, SEAT), a, b])
    assert roles.visitor_ids == (2,)
    assert roles.visitors[0].t_start == 2.0 and roles.visitors[0].t_end == 20.0


def test_short_tracks_are_rejected_and_empty_input_has_no_roles():
    roles = assign_roles([trk(1, 0, 30, SEAT), trk(2, 3, 3.5, DESK_FAR_SIDE)])
    assert dict(roles.rejected) == {2: "short"}
    assert assign_roles([]).workplace is None


def test_desk_is_where_objects_stay_put_near_the_seat():
    laptop = Detection(box=(140.0, 380.0, 180.0, 420.0), score=0.8, label="laptop")
    passing_bag = [Detection(box=(10.0 * i, 0.0, 10.0 * i + 20, 20.0), score=0.5, label="handbag") for i in range(10)]
    far_tv = Detection(box=(900.0, 900.0, 950.0, 950.0), score=0.9, label="tv")
    frames = [[laptop, passing_bag[i], far_tv] for i in range(10)]
    assert desk_point(frames, WP) == pytest.approx((160.0, 400.0))
    assert desk_point([[passing_bag[i]] for i in range(10)], WP) is None


def test_near_span_starts_when_the_visitor_is_clearly_inside():
    through_glass = trk(5, 3, 5, DESK_FAR_SIDE, score=0.62)  # behind the glass door, already close in the image
    inside = trk(5, 5.5, 13, DESK_FAR_SIDE, score=0.9)
    span = near_span(Track(5, through_glass.points + inside.points), WP)
    assert 5.0 <= span.t_start <= 6.0 and span.t_end == 13.0


def test_visitor_present_from_the_video_start_keeps_the_start():
    weak_start = Track(5, trk(5, 0, 2, DESK_FAR_SIDE, score=0.62).points + trk(5, 2.5, 10, DESK_FAR_SIDE).points)
    span = near_span(weak_start, WP, duration=10.0)
    assert span.t_start == 0.0 and span.t_end == 10.0
