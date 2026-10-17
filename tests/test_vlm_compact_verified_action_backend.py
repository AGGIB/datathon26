"""Coherent physical atoms tested with scripted responses; no model or media."""
from dataclasses import replace
import json
import time

import pytest

from ml.kb import catalog
from ml.vlm.action_backend import _ACTIONS
from ml.vlm.compact_verified_action_backend import CompactVerifiedActionBackend, _ATOMS, _parse_atoms, _schema
from ml.vlm.contracts import Frame, InferenceRequest
from ml.vlm.discovery import BudgetExhausted, DiscoveryBudget
from ml.vlm.discovery_policy import TYPE_IDS
from ml.vlm.validation import InvalidResponse

COAT = "cashier_places_removed_outerwear_on_cashdesk"
FOOD = "food_or_drink_actually_consumed"
KEY = "key_remains_in_safe_lock_after_departure"
STAMP = "stamp_or_book_remains_after_departure"
COUNT = "banknotes_counted_after_visitor_departure"
PREVIOUS = "earlier_banknotes_remain_during_next_service"
RETURN = "processed_banknotes_handed_back"


def action(name=COAT, **changes):
    return {"action": name, "insufficient_evidence": False, "confidence": .9,
            "start_frame": 1, "peak_frame": 2, "end_frame": 3, **changes}


def row(index, first, second="unknown"):
    return [index, first, second]


def facts(name):
    return {
        COAT: [row(1, "cashier_holds_identifiable_outerwear"), row(2, "cashier_places_outerwear_on_own_workdesk"),
               row(3, "outerwear_resting_on_own_workdesk")],
        FOOD: [row(1, "container_standing"), row(2, "cashier_drinks_identifiable_cup_at_mouth"), row(3, "papers_handled")],
        KEY: [row(1, "resolved_key_inserted_identified_safe_lock", "cashier_visibly_leaves_safe"),
              row(2, "resolved_key_inserted_identified_safe_lock", "cashier_away_from_safe"),
              row(3, "resolved_key_inserted_identified_safe_lock", "cashier_away_from_safe")],
        STAMP: [row(1, "identifiable_stamp_or_book_resting_on_own_workdesk", "cashier_visibly_leaves_workdesk"),
                row(2, "identifiable_stamp_or_book_resting_on_own_workdesk", "cashier_away_from_workdesk"),
                row(3, "identifiable_stamp_or_book_resting_on_own_workdesk", "cashier_away_from_workdesk")],
        COUNT: [row(1, "visitor_visibly_leaves_service", "cashier_holds_identifiable_banknotes"),
                row(2, "cashier_counts_identifiable_banknotes", "visitor_away_from_service"),
                row(3, "cashier_counts_identifiable_banknotes", "visitor_away_from_service")],
        PREVIOUS: [row(1, "identifiable_banknotes_rest_on_own_workdesk", "previous_service_visibly_complete"),
                   row(2, "identifiable_banknotes_rest_on_own_workdesk", "different_visitor_actually_being_served"),
                   row(3, "identifiable_banknotes_rest_on_own_workdesk", "different_visitor_actually_being_served")],
        RETURN: [row(1, "cashier_processes_identifiable_banknotes"), row(2, "cashier_hands_identifiable_banknotes_to_visitor"),
                 row(3, "distinct_visitor_holds_identifiable_banknotes")],
    }[name]


def episode(rows, *, identity=1, actor="yes", obj="yes"):
    return {"id": identity, "rows": rows, "same_actor": actor, "same_object": obj}


class Scripted:
    def __init__(self, proposals, episodes=None, raw=None, hook=None):
        self.responses = [json.dumps({"actions": proposals}), raw if raw is not None else json.dumps({"episodes": episodes})]
        self.requests, self.hook = [], hook

    def infer(self, request, timeout_s):
        self.requests.append((request, timeout_s))
        if self.hook:
            self.hook(len(self.requests))
        return self.responses[len(self.requests) - 1]


@pytest.fixture
def inference_request(tmp_path):
    frames = tuple(Frame(t, tmp_path / f"{index}.png", t) for index, t in enumerate((.4, 4.6, 9.7)))
    return InferenceRequest("PRIVATE_CV_LEGAL_GT_PROMPT", frames, "checklist", None, TYPE_IDS)


@pytest.mark.parametrize("name", list(_ACTIONS))
def test_each_supported_action_needs_whole_physical_atoms_and_keeps_source_times(inference_request, name):
    inner = Scripted([action(name)], [episode(facts(name))])
    backend = CompactVerifiedActionBackend(inner, catalog=catalog())
    [finding] = json.loads(backend.infer(inference_request, 2))["findings"]
    assert finding["type_id"] == _ACTIONS[name].type_id
    assert (finding["t_start"], finding["t_peak"], finding["t_end"], finding["confidence"]) == (.4, 4.6, 9.7, .9)
    assert len(inner.requests) == backend.calls_used == backend.last_inner_calls == 2


def test_blind_balanced_schema_and_prompt_never_reveal_chosen_action_or_confidence(inference_request):
    translated = []
    for name, confidence in ((COAT, .71234), (KEY, .91234)):
        inner = Scripted([action(name, confidence=confidence)], [episode(facts(name))])
        CompactVerifiedActionBackend(inner, catalog=catalog()).infer(inference_request, 2)
        request = inner.requests[1][0]
        assert request.allowed_rule_ids == () and request.candidate_id is None
        assert inference_request.prompt not in request.prompt and str(confidence) not in request.prompt
        assert all(kind not in request.prompt for kind in TYPE_IDS)
        assert all(name not in request.prompt for name in _ACTIONS)
        translated.append(request)
    assert translated[0].prompt == translated[1].prompt
    assert translated[0].response_schema == translated[1].response_schema


@pytest.mark.parametrize("name", [COAT, KEY, STAMP, COUNT, PREVIOUS, RETURN])
@pytest.mark.parametrize("field,value", [("same_actor", "unknown"), ("same_actor", "no"), ("same_object", "unknown"), ("same_object", "no")])
def test_ordered_actions_never_force_actor_or_object_continuity(inference_request, name, field, value):
    evidence = episode(facts(name))
    evidence[field] = value
    backend = CompactVerifiedActionBackend(Scripted([action(name)], [evidence]), catalog=catalog())
    assert json.loads(backend.infer(inference_request, 2)) == {"findings": []}


@pytest.mark.parametrize("continuity", ["unknown", "no"])
def test_actual_food_peak_stands_alone_despite_mundane_adjacent_objects(inference_request, continuity):
    backend = CompactVerifiedActionBackend(Scripted([action(FOOD)],
        [episode(facts(FOOD), actor=continuity, obj=continuity)]), catalog=catalog())
    assert len(json.loads(backend.infer(inference_request, 2))["findings"]) == 1


@pytest.mark.parametrize("atom", ["container_standing", "hand_near_face", "papers_handled", "ordinary_work", "unknown"])
def test_no_ordinary_atom_or_face_touch_can_replace_actual_food_consumption(inference_request, atom):
    rows = facts(FOOD)
    rows[1] = row(2, atom)
    backend = CompactVerifiedActionBackend(Scripted([action(FOOD)], [episode(rows)]), catalog=catalog())
    assert json.loads(backend.infer(inference_request, 2)) == {"findings": []}


@pytest.mark.parametrize("name,phase,atom", [
    (COAT, 0, "visitor_holds_identifiable_outerwear"), (COAT, 1, "visitor_bag_handled"), (COAT, 2, "papers_handled"),
    (KEY, 0, "highlight_or_handle"), (KEY, 1, "ordinary_work"), (KEY, 2, "unknown"),
    (STAMP, 0, "papers_handled"), (STAMP, 1, "unknown"),
    (COUNT, 0, "ordinary_work"), (COUNT, 1, "cashier_counts_banknotes_during_visible_service"),
    (RETURN, 1, "cashier_receives_banknotes_from_visitor"), (RETURN, 2, "ordinary_work"),
])
def test_missing_physical_prerequisite_or_wrong_role_object_direction_rejects(inference_request, name, phase, atom):
    rows = facts(name)
    rows[phase] = row(phase + 1, atom)
    backend = CompactVerifiedActionBackend(Scripted([action(name)], [episode(rows)]), catalog=catalog())
    assert json.loads(backend.infer(inference_request, 2)) == {"findings": []}


@pytest.mark.parametrize("mutation", ["duplicate_id", "bool_id", "bool_frame", "wrong_frame", "extra_field", "bad_atom", "flag_bool", "missing_anchor", "extra_atom"])
def test_malformed_batch_never_returns_valid_prefix(inference_request, mutation):
    payload = [episode(facts(COAT)), episode(facts(FOOD), identity=2)]
    target = payload[1]
    if mutation == "duplicate_id":
        target["id"] = 1
    if mutation == "bool_id":
        target["id"] = True
    if mutation == "bool_frame":
        target["rows"][0][0] = True
    if mutation == "wrong_frame":
        target["rows"][0][0] = 2
    if mutation == "extra_field":
        target["confidence"] = .9
    if mutation == "bad_atom":
        target["rows"][0][1] = "free explanation"
    if mutation == "flag_bool":
        target["same_actor"] = True
    if mutation == "missing_anchor":
        target["rows"].pop()
    if mutation == "extra_atom":
        target["rows"][0].append("unknown")
    backend = CompactVerifiedActionBackend(Scripted([action(), action(FOOD)], payload), catalog=catalog())
    with pytest.raises(InvalidResponse):
        backend.infer(inference_request, 2)
    assert backend.last_verifications == () and backend.calls_used == 2


def test_unique_late_source_anchors_reuse_frame_objects_and_expand_locally(inference_request):
    frames = tuple(Frame(float(index), inference_request.frames[0].path, float(index)) for index in range(1, 13))
    request = replace(inference_request, frames=frames)
    rows = [row(1, "cashier_holds_identifiable_outerwear"), row(2, "outerwear_resting_on_own_workdesk")]
    inner = Scripted([action(start_frame=8, peak_frame=11, end_frame=11)], [episode(rows)])
    backend = CompactVerifiedActionBackend(inner, catalog=catalog())
    [finding] = json.loads(backend.infer(request, 2))["findings"]
    assert (finding["t_start"], finding["t_peak"], finding["t_end"]) == (8., 11., 11.)
    assert inner.requests[1][0].frames[0] is frames[7] and inner.requests[1][0].frames[1] is frames[10]
    parsed = _parse_atoms(json.dumps({"episodes": [episode(rows)]}), [{"id": 1, "anchors": [1, 2, 2]}])
    assert parsed[1]["phases"][1] == parsed[1]["phases"][2]


def test_recount_needs_prior_banknote_evidence_not_only_departure_and_identity_flag(inference_request):
    rows = facts(COUNT)
    rows[0] = row(1, "visitor_visibly_leaves_service")
    backend = CompactVerifiedActionBackend(Scripted([action(COUNT)], [episode(rows)]), catalog=catalog())
    assert json.loads(backend.infer(inference_request, 2)) == {"findings": []}


def test_budget_and_deadline_are_shared_and_never_emit_unverified_proposal(inference_request):
    inner = Scripted([action()], [episode(facts(COAT))])
    backend = CompactVerifiedActionBackend(inner, catalog=catalog())
    budget = DiscoveryBudget(time.monotonic() + 2, max_calls=1)
    with pytest.raises(BudgetExhausted):
        backend.infer_with_budget(inference_request, budget)
    assert budget.calls_used == backend.calls_used == len(inner.requests) == 1
    assert not backend.last_verifications


def test_schema_const_anchors_and_all_atom_choices_are_identical_in_every_slot():
    schema = _schema([{"id": 1, "anchors": [1, 2, 2]}, {"id": 2, "anchors": [3, 4, 5]}], 5)
    choices = []
    for episode_schema in schema["properties"]["episodes"]["prefixItems"]:
        for row_schema in episode_schema["properties"]["rows"]["prefixItems"]:
            assert row_schema["minItems"] == row_schema["maxItems"] == 3 and row_schema["items"] is False
            assert "const" in row_schema["prefixItems"][0]
            choices.extend(field["enum"] for field in row_schema["prefixItems"][1:])
        assert episode_schema["properties"]["same_actor"]["enum"] == ["yes", "no", "unknown"]
    assert all(choice == list(_ATOMS) for choice in choices)
    assert {"unknown", "ordinary_work", "visitor_bag_handled", "papers_handled", "hand_near_face", "highlight_or_handle"} <= set(_ATOMS)
