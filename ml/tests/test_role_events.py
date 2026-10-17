from ml.events import extract_role_events
from ml.roles import Roles, Workplace
from ml.track import Track, TrackPoint

SEAT = (40.0, 100.0, 80.0, 160.0)  # cashier seated: foot (60, 160), width 40, height 60
WP = Workplace(home=(60.0, 160.0), box=SEAT, width=40.0, height=60.0, signature=())
STEP = 0.5


def times(duration):
    return [round(i * STEP, 3) for i in range(int(duration / STEP) + 1)]


def trk(tid, t0, t1, box, score=0.9):
    pts = tuple(TrackPoint(t=t, box=box, score=score) for t in times(t1) if t >= t0)
    return Track(tid, pts)


def path(tid, segments, score=0.9):
    """segments: [(t0, t1, box), ...] concatenated into one track."""
    pts = []
    for t0, t1, box in segments:
        pts += [TrackPoint(t=t, box=box, score=score) for t in times(t1) if t0 <= t <= t1 and (not pts or t > pts[-1].t)]
    return Track(tid, tuple(pts))


def roles(cashier=(), visitors=(), wp=WP):
    return Roles(workplace=wp, cashier=tuple(cashier), visitors=tuple(visitors), rejected=())


def kinds(events, kind):
    return [e for e in events if e.kind == kind]


FAR = (200.0, 0.0, 230.0, 50.0)  # away from the seat (by the door)
VISITOR_SPOT = (40.0, 20.0, 80.0, 70.0)  # across the desk: foot (60, 70) = 2.25 widths from the seat
BEHIND = (80.0, 70.0, 120.0, 150.0)  # standing next to the seat: foot (100, 150), tall box


def test_cashier_who_leaves_and_comes_back_gives_left_and_returned():
    cashier = path(1, [(0, 10, SEAT), (10.5, 20, FAR), (20.5, 30, SEAT)])
    ev = extract_role_events(roles([cashier]), times(30), duration=30.0)
    [left] = kinds(ev, "cashier_left")
    [back] = kinds(ev, "cashier_returned")
    assert left.t == 10.0 and left.t_end == 20.5 and left.track_ids == (1,)
    assert back.t == 20.5


def test_cashier_lost_by_the_tracker_counts_as_away():
    cashier = trk(1, 0, 8, SEAT)
    ev = extract_role_events(roles([cashier]), times(20), duration=20.0)
    [left] = kinds(ev, "cashier_left")
    assert left.t == 8.0 and left.t_end is None
    assert not kinds(ev, "cashier_returned")


def test_short_absence_from_the_seat_is_not_leaving():
    cashier = path(1, [(0, 10, SEAT), (10.5, 11.5, FAR), (12, 20, SEAT)])
    ev = extract_role_events(roles([cashier]), times(20), duration=20.0)
    assert not kinds(ev, "cashier_left")


def test_cashier_reidentified_on_a_new_track_returns():
    ev = extract_role_events(roles([trk(1, 0, 5, SEAT), trk(7, 12, 20, SEAT)]), times(20), duration=20.0)
    [left] = kinds(ev, "cashier_left")
    assert left.t == 5.0 and left.t_end == 12.0 and left.track_ids == (1, 7)
    assert [e.t for e in kinds(ev, "cashier_returned")] == [12.0]


def test_cashier_arriving_into_an_empty_room_is_not_a_return():
    ev = extract_role_events(roles([trk(1, 3, 20, SEAT)]), times(20), duration=20.0)
    assert not kinds(ev, "cashier_left") and not kinds(ev, "cashier_returned")


def test_visitor_enter_and_exit_inside_the_video():
    ev = extract_role_events(roles([trk(1, 0, 20, SEAT)], [trk(5, 4, 12, VISITOR_SPOT)]), times(20), 20.0)
    assert [(e.t, e.track_id) for e in kinds(ev, "visitor_enter")] == [(4.0, 5)]
    assert [(e.t, e.track_id) for e in kinds(ev, "visitor_exit")] == [(12.0, 5)]


def test_visitor_present_from_start_to_end_has_no_enter_or_exit():
    ev = extract_role_events(roles([trk(1, 0, 20, SEAT)], [trk(5, 0, 20, VISITOR_SPOT)]), times(20), 20.0)
    assert not kinds(ev, "visitor_enter") and not kinds(ev, "visitor_exit")


def test_two_visitors_at_once_is_an_overlap():
    visitors = [trk(5, 0, 15, VISITOR_SPOT), trk(6, 8, 20, (90.0, 20.0, 130.0, 70.0))]
    ev = extract_role_events(roles([trk(1, 0, 20, SEAT)], visitors), times(20), 20.0)
    [overlap] = kinds(ev, "visitors_overlap")
    assert overlap.t == 8.0 and overlap.t_end == 15.0 and overlap.track_ids == (5, 6) and overlap.count == 2


def test_next_visitor_after_the_first_left_is_not_an_overlap():
    visitors = [trk(5, 0, 10, VISITOR_SPOT), trk(6, 10.5, 20, VISITOR_SPOT)]
    ev = extract_role_events(roles([trk(1, 0, 20, SEAT)], visitors), times(20), 20.0)
    assert not kinds(ev, "visitors_overlap")


def test_brief_crossing_of_two_visitors_is_not_an_overlap():
    visitors = [trk(5, 0, 10, VISITOR_SPOT), trk(6, 9, 20, VISITOR_SPOT)]
    ev = extract_role_events(roles([trk(1, 0, 20, SEAT)], visitors), times(20), 20.0)
    assert not kinds(ev, "visitors_overlap")


def test_person_standing_next_to_the_seated_cashier_is_behind_cashier():
    intruder = path(9, [(6, 9, FAR), (9.5, 14, BEHIND)])
    ev = extract_role_events(roles([trk(1, 0, 20, SEAT)], [intruder]), times(20), 20.0)
    [behind] = kinds(ev, "person_behind_cashier")
    assert behind.t == 9.5 and behind.t_end == 14.0 and behind.track_id == 9


def test_visitor_across_the_desk_is_not_behind_cashier():
    ev = extract_role_events(roles([trk(1, 0, 20, SEAT)], [trk(5, 2, 18, VISITOR_SPOT)]), times(20), 20.0)
    assert not kinds(ev, "person_behind_cashier")


def test_person_on_the_desk_side_is_not_behind_even_when_close():
    wp = Workplace(home=(60.0, 160.0), box=SEAT, width=40.0, height=60.0, signature=(), desk=(110.0, 160.0))
    ev = extract_role_events(roles([trk(1, 0, 20, SEAT)], [trk(9, 2, 18, BEHIND)], wp=wp), times(20), 20.0)
    assert not kinds(ev, "person_behind_cashier")


def test_no_workplace_means_no_role_events():
    ev = extract_role_events(Roles(None, (), (), ()), times(10), 10.0)
    assert ev == []


STANDING = (40.0, 40.0, 80.0, 160.0)  # same place, twice as tall: stood up


def test_standing_at_the_seat_is_still_at_work_and_walking_away_is_leaving():
    cashier = path(1, [(0, 8, SEAT), (8.5, 10, STANDING), (10.5, 20, FAR)])
    ev = extract_role_events(roles([cashier]), times(20), duration=20.0)
    [left] = kinds(ev, "cashier_left")
    assert left.t == 10.0


def test_brief_or_weak_reappearance_at_the_seat_after_leaving_is_not_a_return():
    flicker = path(1, [(0, 8, SEAT), (8.5, 14, FAR), (14.5, 15, SEAT), (15.5, 20, FAR)])
    ghost = Track(2, tuple(TrackPoint(t, SEAT, 0.55) for t in times(20) if t >= 16))
    ev = extract_role_events(roles([flicker, ghost]), times(20), duration=20.0)
    [left] = kinds(ev, "cashier_left")
    assert left.t == 8.0 and left.t_end is None
    assert not kinds(ev, "cashier_returned")


def test_standing_near_the_seat_after_leaving_is_not_a_return():
    # seen through the glass door right behind the seat: same spot in the image, but standing
    cashier = path(1, [(0, 8, SEAT), (8.5, 12, FAR), (12.5, 20, STANDING)])
    ev = extract_role_events(roles([cashier]), times(20), duration=20.0)
    [left] = kinds(ev, "cashier_left")
    assert left.t == 8.0 and left.t_end is None


def test_unstable_weak_figure_at_the_seat_after_leaving_is_not_a_return():
    back = [TrackPoint(t, SEAT, 0.72 if i % 2 else 0.78) for i, t in enumerate(times(20)) if t >= 12.5]
    cashier = Track(1, path(1, [(0, 8, SEAT), (8.5, 12, FAR)]).points + tuple(back))
    ev = extract_role_events(roles([cashier]), times(20), duration=20.0)
    assert not kinds(ev, "cashier_returned")
