import pytest

from ml.events import Event
from ml.rules import RuleInput, Sample, run_rules


def samples(t0, t1, step=0.5, **state):
    base = dict(safe_open=0.0, door_open=0.0, screen_work=0.0, key_in_lock=0.0, items=frozenset(), people=1,
                keyframe=False)
    out = []
    t = t0
    while t <= t1 + 1e-9:
        out.append(Sample(t=round(t, 3), **{**base, **state}))
        t += step
    return out


def run(events=(), smp=(), duration=30.0):
    return run_rules(RuleInput(duration=duration, events=tuple(events), samples=tuple(smp)))


def types(cands):
    return sorted({c.type_id for c in cands})


# --- leaving with something left behind -------------------------------------------------

def test_cash_left_on_desk_when_cashier_leaves():
    smp = samples(0, 10) + samples(10.5, 20, people=0, items=frozenset({"cash_bundle"}))
    cands = run([Event("cashier_left", 10.0, t_end=20.0)], smp)
    [c] = [c for c in cands if c.type_id == "cash_left_on_desk_when_leaving"]
    assert c.t_start <= 10.0 <= c.t_peak and c.evidence_times[0] == c.t_peak
    assert "пачка" in c.observation


def test_clean_leave_gives_nothing():
    smp = samples(0, 10) + samples(10.5, 20, people=0)
    assert run([Event("cashier_left", 10.0, t_end=20.0)], smp) == []


def test_room_empty_is_used_when_role_events_are_missing():
    smp = samples(0, 10) + samples(10.5, 20, people=0, items=frozenset({"book", "stamp"}))
    assert types(run([Event("room_empty", 10.0, t_end=20.0)], smp)) == ["stamps_and_book_left_on_desk"]


def test_screen_safe_key_door_left_when_leaving():
    after = samples(10.5, 20, people=0, screen_work=0.9, safe_open=0.9, key_in_lock=0.9, door_open=0.9)
    cands = run([Event("cashier_left", 10.0, t_end=20.0)], samples(0, 10) + after)
    assert {"computer_left_unlocked", "safe_left_open", "key_left_in_safe_lock", "door_left_open"} <= set(types(cands))


def test_item_seen_only_in_one_noisy_sample_is_ignored():
    smp = samples(0, 10) + samples(10.5, 20, people=0)
    smp[25] = Sample(t=smp[25].t, safe_open=0, door_open=0, screen_work=0, key_in_lock=0,
                     items=frozenset({"cash_bundle"}), people=0, keyframe=False)
    assert run([Event("cashier_left", 10.0, t_end=20.0)], smp) == []


# --- states that last while the cashier works -------------------------------------------

def test_safe_open_for_long_while_working():
    smp = samples(0, 3) + samples(3.5, 15, safe_open=0.9) + samples(15.5, 20)
    [c] = run([], smp)
    assert c.type_id == "safe_left_open" and 3.0 <= c.t_start <= 4.0 and 14.5 <= c.t_end <= 15.5


def test_safe_open_briefly_to_take_cash_is_fine():
    smp = samples(0, 3) + samples(3.5, 6, safe_open=0.9) + samples(6.5, 20)
    assert run([], smp) == []


def test_door_open_for_long_while_working():
    smp = samples(0, 2) + samples(2.5, 20, door_open=0.8)
    assert types(run([], smp)) == ["door_left_open"]


# --- people -------------------------------------------------------------------------------

def test_person_behind_cashier():
    [c] = run([Event("person_behind_cashier", 12.0, t_end=15.0)], samples(0, 20))
    assert c.type_id == "person_entered_cash_area" and c.t_peak == 12.0


def test_two_visitors_at_once():
    [c] = run([Event("visitors_overlap", 18.0, t_end=22.0, count=2)], samples(0, 30))
    assert c.type_id == "second_client_served_alongside"


def test_previous_cash_stays_when_next_visitor_comes():
    smp = samples(0, 30, items=frozenset({"cash_bundle"}))
    events = [Event("visitor_exit", 10.0), Event("visitor_enter", 16.0)]
    [c] = run(events, smp)
    assert c.type_id == "previous_cash_left_on_desk" and c.t_peak == 18.0  # handover ~2 s after arrival


def test_visitor_change_with_clean_desk_is_fine():
    assert run([Event("visitor_exit", 10.0), Event("visitor_enter", 16.0)], samples(0, 30)) == []


# --- things on the desk ---------------------------------------------------------------------

def test_personal_items_on_desk():
    smp = samples(0, 5) + samples(5.5, 20, items=frozenset({"bag", "clothes"}))
    assert types(run([], smp)) == ["personal_items_on_desk"]


def test_food_on_desk():
    smp = samples(0, 5) + samples(5.5, 12, items=frozenset({"food", "cup"}))
    assert types(run([], smp)) == ["food_or_drink_at_workplace"]


def test_cup_alone_is_not_flagged():
    smp = samples(0, 20, items=frozenset({"cup"}))
    assert run([], smp) == []


def test_all_candidates_are_valid_and_sorted():
    smp = samples(0, 10, items=frozenset({"bag"})) + samples(10.5, 25, people=0, items=frozenset({"bag", "cash_bundle"}))
    cands = run([Event("cashier_left", 10.0, t_end=25.0), Event("person_behind_cashier", 4.0)], smp)
    assert cands == sorted(cands, key=lambda c: c.t_peak)
    assert all(0.0 <= c.confidence <= 1.0 and c.observation for c in cands)


# --- adapter ------------------------------------------------------------------------------------

def _perception(frames, events=(), duration=20.0):
    from ml.perception import Perception
    from ml.video import VideoInfo

    info = VideoInfo(fps=10.0, frame_count=int(duration * 10), width=464, height=848)
    return Perception(info=info, frames=tuple(frames), tracks=(), events=tuple(events), elapsed_sec=0.0)


def test_adapter_without_scene_uses_coco_items():
    from ml.detect import Detection
    from ml.perception import FrameObjects
    from ml.rules import to_rule_input

    bag = Detection((0, 0, 10, 10), 0.9, "handbag")
    person = Detection((0, 0, 10, 30), 0.9, "person")
    frames = [FrameObjects(t=t / 2, detections=(bag, person)) for t in range(0, 20)]
    inp = to_rule_input(_perception(frames))
    assert inp.samples[0].items == frozenset({"bag"}) and inp.samples[0].people == 1
    assert inp.samples[0].safe_open is None


def test_adapter_with_scene_maps_states_and_trusts_scene_item_filter():
    from types import SimpleNamespace as NS

    from ml.rules import to_rule_input

    items = (NS(label="cash_bundle", novelty=0.8), NS(label="book", novelty=0.1))
    state = NS(t=1.0, safe_open=0.9, door_open=0.1, laptop_screen_work=None, key_in_lock=0.0,
               desk_items=items, people=0, keyframe=True)
    inp = to_rule_input(_perception([]), NS(states=(state,)))
    [s] = inp.samples
    assert s.items == frozenset({"cash_bundle", "book"}) and s.safe_open == 0.9 and s.screen_work is None
    assert s.keyframe is True


def test_find_candidates_end_to_end_on_adapter():
    from types import SimpleNamespace as NS

    from ml.rules import find_candidates

    def st(t, cash):
        items = (NS(label="cash_bundle", novelty=0.9),) if cash else ()
        return NS(t=t, safe_open=0.0, door_open=0.0, laptop_screen_work=0.0, key_in_lock=0.0,
                  desk_items=items, people=0 if cash else 1)

    states = [st(t / 2, t / 2 > 10) for t in range(0, 41)]
    perc = _perception([], events=[Event("cashier_left", 10.0, t_end=20.0)])
    assert [c.type_id for c in find_candidates(perc, NS(states=states))] == ["cash_left_on_desk_when_leaving"]


# --- review of PR #36 ---------------------------------------------------------------------------

def _kf(t, people=0, **kw):
    base = dict(safe_open=0.0, door_open=0.0, screen_work=0.0, key_in_lock=0.0, items=frozenset(), people=people,
                keyframe=True)
    return Sample(t=t, **{**base, **kw})


def test_cash_seen_only_on_sparse_keyframes_still_counts():
    dense = samples(0, 10) + samples(10.5, 20, people=0)
    kfs = [_kf(13.0, items=frozenset({"cash_bundle"})), _kf(18.0, items=frozenset({"cash_bundle"}))]
    smp = sorted(dense + kfs, key=lambda s: s.t)
    types_ = types(run([Event("cashier_left", 10.0, t_end=20.0)], smp))
    assert types_ == ["cash_left_on_desk_when_leaving"]


def test_tidy_up_before_leaving_is_not_left_behind():
    # stood up at 5.5 s, stamps on the desk while still in the room, room empty from 13 s with a clean desk
    tidy = samples(5.5, 12.5, people=1, items=frozenset({"stamp", "book"}))
    smp = samples(0, 5) + tidy + samples(13, 25, people=0)
    assert run([Event("cashier_left", 5.5, t_end=25.0)], smp) == []


def test_left_behind_peak_is_when_the_room_is_empty():
    smp = samples(0, 9.5) + samples(10, 14, people=1, screen_work=0.9) + samples(14.5, 25, people=0, screen_work=0.9)
    [c] = run([Event("cashier_left", 10.0, t_end=25.0)], smp)
    assert c.type_id == "computer_left_unlocked" and c.t_peak == 14.5


def test_courier_bag_is_not_personal_items():
    smp = samples(0, 6) + samples(6.5, 20, items=frozenset({"bag"}))
    events = [Event("visitor_enter", 5.0), Event("visitor_exit", 12.0)]
    assert run(events, smp) == []


def test_bag_brought_without_visitor_is_personal_items_when_roles_known():
    smp = samples(0, 6) + samples(6.5, 20, items=frozenset({"bag"}))
    events = [Event("visitor_enter", 25.0)]  # roles known, no visitor while the bag appears
    assert types(run(events, smp, duration=30.0)) == ["personal_items_on_desk"]


def test_bag_alone_without_role_events_is_ignored():
    smp = samples(0, 6) + samples(6.5, 20, items=frozenset({"bag"}))
    assert run([], smp) == []


def test_items_present_from_the_start_are_not_new():
    smp = samples(0, 20, items=frozenset({"clothes"}))
    assert run([], smp) == []


def test_single_missed_sample_does_not_split_an_open_door_run():
    smp = samples(0, 2) + samples(2.5, 6, door_open=0.8) + [_kf(6.5, people=1, keyframe=False)] + \
        samples(7, 10, door_open=0.8) + samples(10.5, 15)
    assert types(run([], smp)) == ["door_left_open"]


def test_open_door_is_not_reported_separately_when_someone_enters_through_it():
    smp = samples(0, 2) + samples(2.5, 20, door_open=0.8)
    cands = run([Event("person_behind_cashier", 12.0, t_end=15.0)], smp)
    assert types(cands) == ["person_entered_cash_area"]


def test_room_empty_fallback_ignored_when_it_overlaps_a_cashier_left():
    smp = samples(0, 10) + samples(10.5, 20, people=0, items=frozenset({"book"}))
    events = [Event("cashier_left", 10.0, t_end=20.0), Event("room_empty", 10.5, t_end=20.0)]
    cands = run(events, smp)
    assert [c.type_id for c in cands] == ["stamps_and_book_left_on_desk"]


def test_two_visitors_peak_is_the_middle_of_the_overlap():
    [c] = run([Event("visitors_overlap", 16.0, t_end=24.0, count=2)], samples(0, 30))
    assert c.t_peak == 20.0


def test_experimental_key_rule_stays_below_the_output_threshold():
    from ml.assemble import MIN_CONFIDENCE

    smp = samples(0, 10) + samples(10.5, 20, people=0, key_in_lock=0.9)
    [c] = run([Event("cashier_left", 10.0, t_end=20.0)], smp)
    assert c.type_id == "key_left_in_safe_lock" and c.confidence < MIN_CONFIDENCE


@pytest.mark.parametrize(
    "items, expected",
    [({"stamp"}, "предмет, похожий на штамп"), ({"book"}, "книга"), ({"stamp", "book"}, "предметы, похожие на штамп и книгу")],
)
def test_left_items_wording(items, expected):
    smp = samples(0, 10) + samples(10.5, 20, people=0, items=frozenset(items))
    [c] = run([Event("cashier_left", 10.0, t_end=20.0)], smp)
    assert expected in c.observation
