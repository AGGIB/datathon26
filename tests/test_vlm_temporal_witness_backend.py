"""Full-window temporal evidence with scripted replies; no media/model work."""
from dataclasses import replace
import json
import math
import subprocess
import sys
import threading
import time

import pytest

from ml.candidates import Candidate
from ml.kb import catalog
from ml.vlm.action_backend import _ACTIONS, _NORMAL
from ml.vlm.compact_verified_action_backend import _ATOMS
from ml.vlm.contracts import Frame, InferenceRequest
from ml.vlm.discovery import BudgetExhausted, DiscoveryBudget, DiscoveryWindow, discover_types
from ml.vlm.discovery_policy import TYPE_IDS
from ml.vlm.preflight import ModelUnavailable
from ml.vlm.temporal_witness_backend import TemporalWitnessBackend, _inventory_request, _schema
from ml.vlm.validation import InvalidResponse

COAT = "cashier_places_removed_outerwear_on_cashdesk"
FOOD = "food_or_drink_actually_consumed"
KEY = "key_remains_in_safe_lock_after_departure"
STAMP = "stamp_or_book_remains_after_departure"
COUNT = "banknotes_counted_after_visitor_departure"
PREVIOUS = "earlier_banknotes_remain_during_next_service"
RETURN = "processed_banknotes_handed_back"


def action(name=COAT, **changes):
    return {"action": name, "insufficient_evidence": False, "confidence": .95,
            "start_frame": 1, "peak_frame": 2, "end_frame": 3, **changes}


def atoms(name):
    return {
        COAT: [("cashier_holds_identifiable_outerwear",), ("cashier_places_outerwear_on_own_workdesk",),
               ("outerwear_resting_on_own_workdesk",)],
        FOOD: [("container_standing",), ("cashier_drinks_identifiable_cup_at_mouth",), ("papers_handled",)],
        KEY: [("resolved_key_inserted_identified_safe_lock", "cashier_visibly_leaves_safe"),
              ("resolved_key_inserted_identified_safe_lock", "cashier_away_from_safe"),
              ("resolved_key_inserted_identified_safe_lock", "cashier_away_from_safe")],
        STAMP: [("identifiable_stamp_or_book_resting_on_own_workdesk", "cashier_visibly_leaves_workdesk"),
                ("identifiable_stamp_or_book_resting_on_own_workdesk", "cashier_away_from_workdesk"),
                ("identifiable_stamp_or_book_resting_on_own_workdesk", "cashier_away_from_workdesk")],
        COUNT: [("visitor_visibly_leaves_service", "cashier_holds_identifiable_banknotes"),
                ("cashier_counts_identifiable_banknotes", "visitor_away_from_service"),
                ("cashier_counts_identifiable_banknotes", "visitor_away_from_service")],
        PREVIOUS: [("identifiable_banknotes_rest_on_own_workdesk", "previous_service_visibly_complete"),
                   ("identifiable_banknotes_rest_on_own_workdesk", "different_visitor_actually_being_served"),
                   ("identifiable_banknotes_rest_on_own_workdesk", "different_visitor_actually_being_served")],
        RETURN: [("cashier_processes_identifiable_banknotes",), ("cashier_hands_identifiable_banknotes_to_visitor",),
                 ("distinct_visitor_holds_identifiable_banknotes",)],
    }[name]


def inventory(name=COAT, *, count=12, indices=(4, 7, 10), actor="yes", obj="yes"):
    facts = {index: values for index, values in zip(indices, atoms(name), strict=True)}
    rows = [[index, *facts.get(index, ("ordinary_work",)), "unknown"] for index in range(1, count + 1)]
    rows = [row[:3] for row in rows]
    return {"episodes": [{"id": 1, "rows": rows, "same_actor": actor, "same_object": obj}]}


class Scripted:
    def __init__(self, proposals=(), evidence=None, *, first_raw=None, hook=None):
        self.responses = [first_raw if first_raw is not None else json.dumps({"actions": proposals}),
                          json.dumps(evidence if evidence is not None else inventory())]
        self.requests, self.hook, self.prepared, self.closed = [], hook, [], 0

    def infer(self, inference_request, timeout_s):
        self.requests.append((inference_request, timeout_s))
        if self.hook:
            self.hook(len(self.requests), inference_request, timeout_s)
        value = self.responses[(len(self.requests) - 1) % 2]
        if isinstance(value, Exception):
            raise value
        return value

    def validate(self, timeout_s):
        self.prepared.append(timeout_s)

    def close(self):
        self.closed += 1


@pytest.fixture
def inference_request(tmp_path):
    # Nonuniform CV source times intentionally differ from container-style PTS.
    times = (.2, 1.1, 2.8, 4.5, 6.2, 8.9, 11.7, 14.9, 18.1, 21.6, 25.3, 29.2)
    frames = tuple(Frame(t, tmp_path / f"source-{index}.png", t + 100) for index, t in enumerate(times))
    return InferenceRequest("PRIVATE_OWNER_CV_GOLD_FILENAME_LEGAL", frames, "checklist", "private-id", TYPE_IDS)


@pytest.mark.parametrize("name", list(_ACTIONS))
def test_finds_witnesses_outside_proposal_anchors_and_relocalizes_exact_source_times(inference_request, name):
    inner = Scripted([action(name)], inventory(name))
    backend = TemporalWitnessBackend(inner, catalog=catalog())
    [finding] = json.loads(backend.infer(inference_request, 2))["findings"]
    indices = (7, 7, 7) if name == FOOD else (4, 7, 10)
    assert (finding["t_start"], finding["t_peak"], finding["t_end"]) == tuple(inference_request.frames[i-1].timestamp_s for i in indices)
    assert finding["type_id"] == _ACTIONS[name].type_id and finding["confidence"] == .95
    assert not finding["insufficient_evidence"]
    assert set(row["timestamp_s"] for row in finding["observations"]) == {inference_request.frames[i-1].timestamp_s for i in indices}
    assert backend.last_verifications[0]["witness_frame_indices"] == list(indices)
    assert backend.calls_used == backend.last_inner_calls == len(inner.requests) == 2
    assert inner.requests[1][0].frames is inference_request.frames
    assert all(a is b for a, b in zip(inner.requests[1][0].frames, inference_request.frames, strict=True))
    assert all(frame.pts_s == frame.timestamp_s + 100 for frame in inference_request.frames)


def test_inventory_prompt_and_schema_blind_to_action_confidence_and_old_anchors(inference_request):
    requests = []
    for name, confidence, indices in ((COAT, .712345, (1, 2, 3)), (KEY, .981234, (8, 11, 12))):
        inner = Scripted([action(name, confidence=confidence, start_frame=indices[0], peak_frame=indices[1], end_frame=indices[2])], inventory(name))
        TemporalWitnessBackend(inner, catalog=catalog()).infer(inference_request, 2)
        translated = inner.requests[1][0]
        assert translated.candidate_id is None and translated.allowed_rule_ids == ()
        assert inference_request.prompt not in translated.prompt and str(confidence) not in translated.prompt
        assert all(name not in translated.prompt for name in _ACTIONS)
        assert all(type_id not in translated.prompt for type_id in TYPE_IDS)
        assert all(phase not in translated.prompt for spec in _ACTIONS.values() for phase in spec.phases)
        assert not any(word in translated.prompt for word in ("phase_anchors", "BEFORE", "PEAK", "AFTER", "unique_anchors"))
        assert len(translated.frames) == 12
        requests.append(translated)
    assert requests[0].prompt == requests[1].prompt
    assert requests[0].response_schema == requests[1].response_schema


@pytest.mark.parametrize("name", [COAT, KEY, STAMP, COUNT, PREVIOUS, RETURN])
@pytest.mark.parametrize("field,value", [("same_actor", "unknown"), ("same_actor", "no"), ("same_object", "unknown"), ("same_object", "no")])
def test_existing_identity_continuity_predicates_never_relaxed(inference_request, name, field, value):
    facts = inventory(name)
    facts["episodes"][0][field] = value
    backend = TemporalWitnessBackend(Scripted([action(name)], facts), catalog=catalog())
    assert json.loads(backend.infer(inference_request, 2)) == {"findings": []}
    assert backend.last_verifications[0]["status"] == "no_witness"


@pytest.mark.parametrize("name", [KEY, STAMP, COUNT])
def test_static_object_or_away_state_without_departure_cannot_become_sequence(inference_request, name):
    facts = inventory(name)
    facts["episodes"][0]["rows"][3] = [4, *atoms(name)[1]]
    backend = TemporalWitnessBackend(Scripted([action(name)], facts), catalog=catalog())
    assert json.loads(backend.infer(inference_request, 2)) == {"findings": []}


@pytest.mark.parametrize("name", [COAT, KEY, STAMP, COUNT, PREVIOUS, RETURN])
def test_reverse_order_does_not_satisfy_directional_action(inference_request, name):
    facts = inventory(name, indices=(10, 7, 4))
    backend = TemporalWitnessBackend(Scripted([action(name)], facts), catalog=catalog())
    assert json.loads(backend.infer(inference_request, 2)) == {"findings": []}


def test_food_earliest_actual_peak_stands_alone_without_invented_phases(inference_request):
    facts = inventory(FOOD, actor="unknown", obj="unknown")
    facts["episodes"][0]["rows"][4] = [5, "cashier_takes_visible_bite_of_identifiable_food", "unknown"]
    inner = Scripted([action(FOOD)], facts)
    backend = TemporalWitnessBackend(inner, catalog=catalog())
    [finding] = json.loads(backend.infer(inference_request, 2))["findings"]
    assert backend.last_verifications[0]["witness_frame_indices"] == [5, 5, 5]
    assert finding["t_start"] == finding["t_peak"] == finding["t_end"] == inference_request.frames[4].timestamp_s
    assert finding["observations"] == [{"timestamp_s": inference_request.frames[4].timestamp_s, "text": _ACTIONS[FOOD].phases[1]}]


def test_minimum_span_then_earliest_peak_tie_break_is_deterministic(inference_request):
    frames = tuple(Frame(float(index), frame.path, frame.pts_s) for index, frame in enumerate(inference_request.frames))
    inference_request = replace(inference_request, frames=frames)
    facts = inventory(COAT, indices=(2, 6, 10))
    for index, atom in ((4, "cashier_holds_identifiable_outerwear"), (8, "cashier_places_outerwear_on_own_workdesk")):
        facts["episodes"][0]["rows"][index-1] = [index, atom, "unknown"]
    backend = TemporalWitnessBackend(Scripted([action()], facts), catalog=catalog())
    backend.infer(inference_request, 2)
    # Both (4,6,10) and (4,8,10) span6; choose earlier peak6.
    assert backend.last_verifications[0]["witness_frame_indices"] == [4, 6, 10]


@pytest.mark.parametrize("normal", list(_NORMAL))
def test_normal_actions_take_only_one_raw_call(inference_request, normal):
    inner = Scripted([action(normal)])
    backend = TemporalWitnessBackend(inner, catalog=catalog())
    assert json.loads(backend.infer(inference_request, 2)) == {"findings": []}
    assert backend.calls_used == len(inner.requests) == 1 and backend.last_verifications == ()


@pytest.mark.parametrize("changes", [{"confidence": .6999}, {"insufficient_evidence": True}])
def test_parent_proposal_gate_stays_point_seven(inference_request, changes):
    inner = Scripted([action(**changes)])
    backend = TemporalWitnessBackend(inner, catalog=catalog())
    assert json.loads(backend.infer(inference_request, 2)) == {"findings": []}
    assert len(inner.requests) == backend.calls_used == 1


def test_multiple_eligible_proposals_scope_abstention_never_spends_verifier_call(inference_request):
    inner = Scripted([action(), action(FOOD)])
    backend = TemporalWitnessBackend(inner, catalog=catalog())
    assert json.loads(backend.infer(inference_request, 2)) == {"findings": []}
    assert backend.calls_used == len(inner.requests) == 1
    assert len(backend.last_verifications) == 2
    assert all(item["reason"] == "multiple_proposals_not_supported" and item["status"] == "scope_abstention"
               and not item["accepted"] and item["witness_frame_indices"] == [] for item in backend.last_verifications)


def test_low_confidence_second_proposal_does_not_suppress_single_eligible_proposal(inference_request):
    inner = Scripted([action(), action(FOOD, confidence=.6999)], inventory())
    backend = TemporalWitnessBackend(inner, catalog=catalog())
    assert len(json.loads(backend.infer(inference_request, 2))["findings"]) == 1
    assert backend.calls_used == 2


def test_shared_four_call_budget_across_two_windows_and_no_hidden_retry(inference_request):
    inner = Scripted([action()], inventory())
    backend = TemporalWitnessBackend(inner, catalog=catalog())
    budget = DiscoveryBudget(time.monotonic() + 2, max_calls=4)
    assert json.loads(backend.infer_with_budget(inference_request, budget))["findings"]
    assert json.loads(backend.infer_with_budget(inference_request, budget))["findings"]
    assert budget.calls_used == backend.calls_used == len(inner.requests) == 4
    with pytest.raises(BudgetExhausted):
        backend.infer_with_budget(inference_request, budget)
    assert backend.calls_used == 4 and backend.last_verifications == ()


def test_one_call_budget_cannot_emit_unverified_proposal(inference_request):
    inner = Scripted([action()], inventory())
    backend = TemporalWitnessBackend(inner, catalog=catalog())
    budget = DiscoveryBudget(time.monotonic() + 2, max_calls=1)
    with pytest.raises(BudgetExhausted):
        backend.infer_with_budget(inference_request, budget)
    assert budget.calls_used == backend.calls_used == len(inner.requests) == 1
    assert backend.last_verifications == ()


def test_expired_budget_after_inventory_returns_no_witness(inference_request):
    budget = DiscoveryBudget(time.monotonic() + 2, max_calls=4)
    def expire(index, inference_request, timeout_s):
        if index == 2:
            budget.deadline = time.monotonic() - 1
    backend = TemporalWitnessBackend(Scripted([action()], inventory(), hook=expire), catalog=catalog())
    with pytest.raises(TimeoutError):
        backend.infer_with_budget(inference_request, budget)
    assert backend.last_verifications == () and backend.calls_used == 2


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "reorder", "extra", "bool_frame", "bad_atom", "bool_flag", "extra_episode", "bool_id"])
def test_inventory_schema_strict_and_atomic(inference_request, mutation):
    facts = inventory()
    episode = facts["episodes"][0]
    if mutation == "missing":
        episode["rows"].pop()
    if mutation == "duplicate":
        episode["rows"][3][0] = 3
    if mutation == "reorder":
        episode["rows"][0], episode["rows"][1] = episode["rows"][1], episode["rows"][0]
    if mutation == "extra":
        episode["rows"][3].append("unknown")
    if mutation == "bool_frame":
        episode["rows"][0][0] = True
    if mutation == "bad_atom":
        episode["rows"][0][1] = "free description"
    if mutation == "bool_flag":
        episode["same_actor"] = True
    if mutation == "extra_episode":
        facts["episodes"].append(episode)
    if mutation == "bool_id":
        episode["id"] = True
    backend = TemporalWitnessBackend(Scripted([action()], facts), catalog=catalog())
    with pytest.raises(InvalidResponse):
        backend.infer(inference_request, 2)
    assert backend.calls_used == 2 and backend.last_verifications == ()


@pytest.mark.parametrize("count", [True, 0, 13])
def test_schema_bounded_count(count):
    with pytest.raises(ValueError):
        _schema(count)


def test_schema_exactly_twelve_rows_same_balanced_atoms_no_fabricated_anchors():
    episode = _schema(12)["properties"]["episodes"]["prefixItems"][0]
    rows = episode["properties"]["rows"]
    assert rows["minItems"] == rows["maxItems"] == 12 and rows["items"] is False
    assert [row["prefixItems"][0]["const"] for row in rows["prefixItems"]] == list(range(1, 13))
    assert all(field["enum"] == list(_ATOMS) for row in rows["prefixItems"] for field in row["prefixItems"][1:])
    assert {"unknown", "ordinary_work", "visitor_bag_handled", "papers_handled", "highlight_or_handle"} <= set(_ATOMS)


@pytest.mark.parametrize("times", [(1., 1.), (2., 1.), (math.nan,), (math.inf,), (-1.,), (True,)])
def test_inventory_rejects_nonfinite_unordered_duplicate_or_bool_times(inference_request, times):
    frames = tuple(Frame(t, inference_request.frames[0].path, 100) for t in times)
    with pytest.raises(InvalidResponse):
        _inventory_request(frames)


def test_inventory_requires_actual_frame_and_nonempty_bounded_sequence(inference_request):
    for frames in ((), (object(),), inference_request.frames + inference_request.frames[:1]):
        with pytest.raises(InvalidResponse):
            _inventory_request(frames)


def test_owned_worker_lifecycle_forwards_once_and_closed_backend_cannot_restart(inference_request):
    inner = Scripted([action()], inventory())
    backend = TemporalWitnessBackend(inner, catalog=catalog())
    backend.validate(2)
    backend.infer(inference_request, 2)
    backend.close()
    backend.close()
    assert inner.prepared == [2] and inner.closed == 1
    with pytest.raises(ModelUnavailable):
        backend.infer(inference_request, 2)
    assert len(inner.requests) == 2


def test_parallel_request_rejected_by_inherited_lock(inference_request):
    entered, release = threading.Event(), threading.Event()
    def wait(index, inference_request, timeout_s):
        if index == 1:
            entered.set()
            assert release.wait(2)
    backend = TemporalWitnessBackend(Scripted([action()], inventory(), hook=wait), catalog=catalog())
    output = []
    thread = threading.Thread(target=lambda: output.append(backend.infer(inference_request, 2)))
    thread.start()
    assert entered.wait(2)
    try:
        with pytest.raises(ModelUnavailable):
            backend.infer(inference_request, 2)
    finally:
        release.set()
        thread.join(2)
    assert output and backend.calls_used == 2


@pytest.mark.parametrize("confidence,expected_count", [(.949, 0), (.95, 1)])
def test_actual_discovery_relocalized_peak_passes_same_contract_and_cv_objects_retained(inference_request, tmp_path, confidence, expected_count):
    original = Candidate("safe_left_open", 1, 9, 5, .87, "CV immutable", (5,))
    inner = Scripted([action(confidence=confidence)], inventory())
    backend = TemporalWitnessBackend(inner, catalog=catalog())
    times = tuple(frame.timestamp_s for frame in inference_request.frames)
    class Source:
        calls = 0
        def extract(self, video, requested, directory, timeout_s):
            self.calls += 1
            assert requested == times
            self.created = []
            for index in range(len(times)):
                path = directory / f"{index}.png"
                path.write_bytes(b"scripted no pixels")
                self.created.append(path)
            return tuple(Frame(t, path, t) for t,path in zip(times,self.created,strict=True))
    source = Source()
    report = discover_types(tmp_path / "private-video.mp4", (original,), catalog(), duration_sec=30,
        windows=(DiscoveryWindow(times[0], times[-1], times),), allowed_type_ids=TYPE_IDS,
        budget=DiscoveryBudget(time.monotonic()+2,max_calls=4), backend=backend, frame_source=source, min_confidence=.95)
    assert source.calls == 1
    assert report.retained_candidates[0] is original and original.confidence == .87
    assert len(report.added_candidates) == expected_count
    if expected_count:
        [added] = report.added_candidates
        assert added.type_id == "personal_items_on_desk" and added.t_peak == inference_request.frames[6].timestamp_s
    assert report.calls_used == 2 and report.status == "completed"
    assert all(not path.exists() for path in source.created)


def test_import_loads_no_model_or_decode_libraries():
    code = ("import sys; import ml.vlm.temporal_witness_backend; "
            "assert not {'torch','mlx','mlx_vlm','transformers','cv2','PIL','numpy'} & set(sys.modules)")
    subprocess.run([sys.executable, "-c", code], check=True, timeout=5)
