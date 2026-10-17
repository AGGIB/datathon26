from ml.events import Event
from ml.tools.eval_events import DOOR_CROSSINGS, ROLE_EXPECTATIONS, Absent, Expect, count_room_events, judge


def test_expected_event_within_tolerance_is_a_hit_and_others_are_extra():
    exp = Expect("X-0", "cashier_left", 10.0, 12.0, "")
    row = judge(exp, [Event("cashier_left", 13.5), Event("cashier_left", 30.0), Event("visitor_enter", 11.0)])
    assert row.verdict == "hit" and row.detected == (13.5, 30.0) and row.false_alarms == 1


def test_expected_event_outside_tolerance_is_a_miss():
    row = judge(Expect("X-0", "visitor_exit", 10.0, 10.0, ""), [Event("visitor_exit", 12.5)])
    assert row.verdict == "miss"


def test_absent_event_present_is_a_false_alarm():
    assert judge(Absent("X-0", "visitors_overlap", ""), []).verdict == "ok"
    assert judge(Absent("X-0", "visitors_overlap", ""), [Event("visitors_overlap", 3.0)]).verdict == "false alarm"


def test_room_events_are_matched_one_to_one_with_door_crossings():
    events = [Event("enter", 4.5), Event("enter", 5.0), Event("exit", 53.0), Event("exit", 30.0), Event("crowd", 1.0)]
    count = count_room_events("X-4", events)  # crossings: enter 4.5, exit 53, enter 57
    assert (count.detected, count.matched, count.spurious, count.missed) == (4, 2, 2, 1)


def test_tables_cover_the_task_videos_and_all_42_videos():
    assert len(DOOR_CROSSINGS) == 42
    assert {e.video for e in ROLE_EXPECTATIONS} >= {"X-11", "X-30", "X-31", "X-33", "X-18", "X-19", "X-4", "X-39"}
