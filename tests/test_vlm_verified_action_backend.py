"""Blind primitive gate contract tests with scripted model responses only."""
from dataclasses import replace
import json
import threading
import time

import pytest

from ml.candidates import Candidate
from ml.kb import catalog
from ml.vlm.action_backend import _ACTIONS, _NORMAL
from ml.vlm.contracts import Frame, InferenceRequest
from ml.vlm.discovery import BudgetExhausted, DiscoveryBudget, DiscoveryWindow, discover_types
from ml.vlm.discovery_policy import TYPE_IDS
from ml.vlm.preflight import ModelUnavailable
from ml.vlm.validation import InvalidResponse
from ml.vlm.verified_action_backend import VerifiedActionBackend, _primitives, _schema

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


def row(index, obj="outerwear", state="held", place="hand", direction="none", actor="a", role="cashier", object_id="x"):
    return [index, actor, role, object_id, obj, state, place, direction]


def facts(name=COAT):
    if name == COAT:
        return [row(1), row(2, state="placing", place="own_workdesk"), row(3, state="resting", place="own_workdesk")]
    if name == FOOD:
        return [row(1, "cup"), row(2, "cup", "drinking", "mouth"), row(3, "cup")]
    if name in (KEY, STAMP):
        obj, state, place = ("key", "inserted", "identified_safe_lock") if name == KEY else ("stamp", "resting", "own_workdesk")
        return [row(1, obj, state, place, "cashier_leaving"),
                *[row(i, obj, state, place, "cashier_absent", "not_visible", "not_visible") for i in (2, 3)]]
    if name == COUNT:
        return [row(1, "banknotes", direction="visitor_leaving"),
                *[row(i, "banknotes", "counting", "own_workdesk", "visitor_absent") for i in (2, 3)]]
    if name == PREVIOUS:
        return [row(1, "banknotes", "resting", "own_workdesk", "service_complete"),
                *[row(i, "banknotes", "resting", "own_workdesk", "next_visitor_served") for i in (2, 3)]]
    if name == RETURN:
        return [row(1, "banknotes", "processing", "device"), row(2, "banknotes", "handing", "hand", "to_visitor"),
                row(3, "banknotes", "held", "visitor_side", "from_cashier", "b", "visitor")]
    raise AssertionError(name)


def primitive_payload(*episodes):
    return json.dumps({"episodes": [{"id": index, "rows": rows} for index, rows in enumerate(episodes, 1)]})


class Scripted:
    def __init__(self, actions=(), primitive=None, *, first_raw=None, hook=None):
        self.responses = [first_raw if first_raw is not None else json.dumps({"actions": list(actions)}),
                          primitive if primitive is not None else primitive_payload(facts())]
        self.requests, self.hook, self.prepared, self.closed = [], hook, [], 0

    def infer(self, inference_request, timeout_s):
        self.requests.append((inference_request, timeout_s))
        if self.hook:
            self.hook(len(self.requests), inference_request, timeout_s)
        response = self.responses[len(self.requests) - 1]
        if isinstance(response, Exception):
            raise response
        return response

    def validate(self, timeout_s):
        self.prepared.append(timeout_s)

    def close(self):
        self.closed += 1


@pytest.fixture
def inference_request(tmp_path):
    frames = tuple(Frame(timestamp, tmp_path / f"{index}.png", timestamp)
                   for index, timestamp in enumerate((.4, 4.6, 9.7)))
    return InferenceRequest("PRIVATE_CV_PROMPT_FILENAME_LEGAL_GT_REASONING", frames, "checklist", None, TYPE_IDS)


@pytest.mark.parametrize("name", list(_ACTIONS))
def test_each_action_needs_second_visual_facts_and_keeps_original_times(inference_request, name):
    inner = Scripted([action(name)], primitive_payload(facts(name)))
    backend = VerifiedActionBackend(inner, catalog=catalog())
    [finding] = json.loads(backend.infer(inference_request, 2))["findings"]
    assert finding["type_id"] == _ACTIONS[name].type_id
    assert (finding["t_start"], finding["t_peak"], finding["t_end"]) == (.4, 4.6, 9.7)
    assert finding["confidence"] == .9
    assert backend.calls_used == backend.last_inner_calls == len(inner.requests) == 2
    assert backend.last_verifications == ({"episode_id": 1, "accepted": True, "reason": "visible_primitives_supported"},)


def test_verifier_is_blind_to_proposed_action_confidence_and_owner_input(inference_request):
    schemas, prompts = [], []
    for name, confidence in ((COAT, .7132431), (KEY, .9432191)):
        inner = Scripted([action(name, confidence=confidence)], primitive_payload(facts(name)))
        VerifiedActionBackend(inner, catalog=catalog()).infer(inference_request, 2)
        translated, timeout = inner.requests[1]
        assert translated.frames == inference_request.frames
        assert all(left is right for left, right in zip(translated.frames, inference_request.frames, strict=True))
        assert translated.candidate_id is None and translated.allowed_rule_ids == ()
        assert inference_request.prompt not in translated.prompt and str(confidence) not in translated.prompt
        assert all(kind not in translated.prompt for kind in TYPE_IDS)
        assert all(name not in translated.prompt for name in _ACTIONS)
        assert all(phase not in translated.prompt for spec in _ACTIONS.values() for phase in spec.phases)
        assert 0 < timeout <= inner.requests[0][1] <= 2
        schemas.append(translated.response_schema)
        prompts.append(translated.prompt)
    assert schemas[0] == schemas[1] and prompts[0] == prompts[1]


@pytest.mark.parametrize("normal", list(_NORMAL))
def test_normal_actions_skip_verifier(inference_request, normal):
    inner = Scripted([action(normal)])
    backend = VerifiedActionBackend(inner, catalog=catalog())
    assert json.loads(backend.infer(inference_request, 2)) == {"findings": []}
    assert backend.calls_used == 1 and backend.last_inner_calls == 1 and not backend.last_verifications


@pytest.mark.parametrize("changes", [{"confidence": .6999}, {"insufficient_evidence": True}])
def test_low_confidence_or_uncertain_proposals_never_spend_verification_call(inference_request, changes):
    inner = Scripted([action(**changes)])
    backend = VerifiedActionBackend(inner, catalog=catalog())
    assert json.loads(backend.infer(inference_request, 2)) == {"findings": []}
    assert len(inner.requests) == backend.calls_used == 1


@pytest.mark.parametrize("name,phase,column,replacement", [
    (COAT, 0, 2, "visitor"), (COAT, 1, 1, "b"), (COAT, 2, 3, "y"),
    (COAT, 1, 4, "bag"), (COAT, 2, 4, "papers"), (COAT, 2, 5, "held"),
    (COAT, 1, 6, "visitor_side"), (COAT, 0, 1, "unknown"),
    (FOOD, 1, 4, "other"), (FOOD, 1, 5, "held"), (FOOD, 1, 6, "hand"),
    (KEY, 1, 4, "highlight_handle"), (KEY, 1, 6, "unidentified_lock"),
    (KEY, 0, 7, "none"), (KEY, 1, 7, "unknown"),
    (STAMP, 2, 4, "papers"), (STAMP, 0, 7, "none"),
    (COUNT, 1, 4, "papers"), (COUNT, 0, 7, "visitor_absent"), (COUNT, 2, 7, "none"),
    (PREVIOUS, 0, 7, "none"), (PREVIOUS, 2, 3, "y"),
    (RETURN, 1, 7, "from_visitor"), (RETURN, 2, 1, "a"), (RETURN, 2, 2, "cashier"),
])
def test_plausible_contrasts_cannot_support_proposal(inference_request, name, phase, column, replacement):
    rows = facts(name)
    rows[phase][column] = replacement
    backend = VerifiedActionBackend(Scripted([action(name)], primitive_payload(rows)), catalog=catalog())
    assert json.loads(backend.infer(inference_request, 2)) == {"findings": []}
    assert backend.last_verifications[0]["reason"] == "primitive_evidence_not_supported"


def test_one_verifier_call_checks_two_episodes_and_does_not_transfer_support(inference_request):
    wrong = facts(COAT)
    for row in wrong:
        row[4] = "bag"
    inner = Scripted([action(COAT), action(FOOD)], primitive_payload(wrong, facts(FOOD)))
    backend = VerifiedActionBackend(inner, catalog=catalog())
    [finding] = json.loads(backend.infer(inference_request, 2))["findings"]
    assert finding["type_id"] == _ACTIONS[FOOD].type_id
    assert [item["accepted"] for item in backend.last_verifications] == [False, True]
    assert len(inner.requests) == 2


@pytest.mark.parametrize("name", [COAT, FOOD])
def test_unknown_unused_direction_does_not_negate_required_visible_facts(inference_request, name):
    rows = facts(name)
    for row in rows:
        row[7] = "unknown"
    backend = VerifiedActionBackend(Scripted([action(name)], primitive_payload(rows)), catalog=catalog())
    assert len(json.loads(backend.infer(inference_request, 2))["findings"]) == 1


def test_return_unused_before_direction_does_not_negate_known_processing_and_handover(inference_request):
    rows = facts(RETURN)
    rows[0][7] = "unknown"
    backend = VerifiedActionBackend(Scripted([action(RETURN)], primitive_payload(rows)), catalog=catalog())
    assert len(json.loads(backend.infer(inference_request, 2))["findings"]) == 1


def test_food_peak_is_sufficient_when_adjacent_role_object_and_context_are_unknown(inference_request):
    rows = [[1, *(["unknown"] * 7)], row(2, "food", "biting", "mouth", "unknown"),
            [3, "not_visible", "not_visible", "unknown", "unknown", "unknown", "unknown", "unknown"]]
    backend = VerifiedActionBackend(Scripted([action(FOOD)], primitive_payload(rows)), catalog=catalog())
    [finding] = json.loads(backend.infer(inference_request, 2))["findings"]
    assert finding["type_id"] == _ACTIONS[FOOD].type_id


@pytest.mark.parametrize("phase,column,value", [
    (0, 1, "b"), (2, 1, "c"), (0, 2, "visitor"), (2, 2, "other"),
    (0, 3, "y"), (2, 3, "z"), (0, 4, "papers"), (2, 4, "food"),
])
def test_food_known_adjacent_actor_or_object_contradiction_still_rejects(inference_request, phase, column, value):
    rows = facts(FOOD)
    rows[phase][column] = value
    backend = VerifiedActionBackend(Scripted([action(FOOD)], primitive_payload(rows)), catalog=catalog())
    assert json.loads(backend.infer(inference_request, 2)) == {"findings": []}


@pytest.mark.parametrize("column", [1, 2, 3, 4, 5, 6])
def test_food_unknown_required_peak_fact_still_rejects(inference_request, column):
    rows = facts(FOOD)
    rows[1][column] = "unknown"
    backend = VerifiedActionBackend(Scripted([action(FOOD)], primitive_payload(rows)), catalog=catalog())
    assert json.loads(backend.infer(inference_request, 2)) == {"findings": []}


def test_coat_can_persist_after_cashier_is_no_longer_visible(inference_request):
    rows = facts(COAT)
    rows[2][1:3] = ["not_visible", "not_visible"]
    rows[2][7] = "unknown"
    backend = VerifiedActionBackend(Scripted([action(COAT)], primitive_payload(rows)), catalog=catalog())
    assert len(json.loads(backend.infer(inference_request, 2))["findings"]) == 1


@pytest.mark.parametrize("phase,column,value", [
    (0, 1, "unknown"), (1, 2, "unknown"), (2, 1, "unknown"),
    (2, 2, "visitor"), (2, 1, "b"), (2, 3, "unknown"),
    (2, 4, "unknown"), (2, 5, "unknown"), (2, 6, "unknown"),
])
def test_coat_still_requires_known_placement_persistence_and_no_actor_substitution(inference_request, phase, column, value):
    rows = facts(COAT)
    rows[phase][column] = value
    backend = VerifiedActionBackend(Scripted([action(COAT)], primitive_payload(rows)), catalog=catalog())
    assert json.loads(backend.infer(inference_request, 2)) == {"findings": []}


@pytest.mark.parametrize("name", [KEY, STAMP])
@pytest.mark.parametrize("phase,column,value", [
    (0, 7, "unknown"), (1, 7, "unknown"), (2, 7, "unknown"),
    (0, 3, "unknown"), (1, 4, "unknown"), (2, 5, "unknown"), (2, 6, "unknown"),
])
def test_departure_unknowns_never_replace_required_observed_departure_and_persistence(inference_request, name, phase, column, value):
    rows = facts(name)
    rows[phase][column] = value
    backend = VerifiedActionBackend(Scripted([action(name)], primitive_payload(rows)), catalog=catalog())
    assert json.loads(backend.infer(inference_request, 2)) == {"findings": []}


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "extra", "wrong_id", "bool_id", "wrong_phase", "bool_index",
    "float_index", "unknown_enum", "short_row", "extra_key", "visibility", "missing_phase", "non_json"])
def test_malformed_verification_is_atomic_and_emits_no_valid_prefix(inference_request, mutation):
    payload = json.loads(primitive_payload(facts(), facts(FOOD)))
    if mutation == "missing":
        payload["episodes"].pop()
    elif mutation == "duplicate":
        payload["episodes"][1]["id"] = 1
    elif mutation == "extra":
        payload["episodes"].append(payload["episodes"][0])
    elif mutation == "wrong_id":
        payload["episodes"][1]["id"] = 3
    elif mutation == "bool_id":
        payload["episodes"][1]["id"] = True
    elif mutation == "wrong_phase":
        payload["episodes"][1]["rows"].reverse()
    elif mutation == "bool_index":
        payload["episodes"][1]["rows"][0][0] = True
    elif mutation == "float_index":
        payload["episodes"][1]["rows"][0][0] = 1.
    elif mutation == "unknown_enum":
        payload["episodes"][1]["rows"][0][4] = "imagined object"
    elif mutation == "short_row":
        payload["episodes"][1]["rows"][0].pop()
    elif mutation == "extra_key":
        payload["episodes"][1]["confidence"] = .9
    elif mutation == "visibility":
        payload["episodes"][1]["rows"][0][1] = "not_visible"
    elif mutation == "missing_phase":
        payload["episodes"][1]["rows"].pop()
    raw = "not JSON" if mutation == "non_json" else json.dumps(payload)
    backend = VerifiedActionBackend(Scripted([action(), action(FOOD)], raw), catalog=catalog())
    with pytest.raises(InvalidResponse):
        backend.infer(inference_request, 2)
    assert backend.last_verifications == () and backend.calls_used == 2


def test_repeated_phase_anchors_use_one_fact_expanded_locally_and_food_can_be_one_frame(inference_request):
    selected = action(FOOD, start_frame=2, peak_frame=2, end_frame=2)
    eating = row(1, "food", "biting", "mouth")
    inner = Scripted([selected], primitive_payload([eating]))
    backend = VerifiedActionBackend(inner, catalog=catalog())
    [finding] = json.loads(backend.infer(inference_request, 2))["findings"]
    assert finding["t_start"] == finding["t_peak"] == finding["t_end"] == 4.6
    assert inner.requests[1][0].frames == (inference_request.frames[1],)
    expanded = _primitives(primitive_payload([eating]), [{"id": 1, "anchors": [1, 1, 1]}])[1]
    assert expanded == [eating, eating, eating] and expanded[0] is not expanded[1]
    backend = VerifiedActionBackend(Scripted([selected], primitive_payload([eating] * 3)), catalog=catalog())
    with pytest.raises(InvalidResponse, match="primitive_phase_coverage"):
        backend.infer(inference_request, 2)


def test_shared_call_budget_is_claimed_for_each_raw_attempt_and_never_returns_proposal_alone(inference_request):
    inner = Scripted([action()])
    backend = VerifiedActionBackend(inner, catalog=catalog())
    budget = DiscoveryBudget(time.monotonic() + 2, max_calls=1)
    with pytest.raises(BudgetExhausted):
        backend.infer_with_budget(inference_request, budget)
    assert budget.calls_used == backend.calls_used == backend.last_inner_calls == len(inner.requests) == 1
    assert backend.last_verifications == ()
    empty = DiscoveryBudget(time.monotonic() + 2, max_calls=0)
    with pytest.raises(BudgetExhausted):
        backend.infer_with_budget(inference_request, empty)
    assert backend.last_inner_calls == 0 and len(inner.requests) == 1


def test_inner_failure_counts_attempt_and_deadline_never_renews(inference_request, monkeypatch):
    ticks = iter(index / 1000 for index in range(10000))
    monkeypatch.setattr(time, 'monotonic', lambda: next(ticks))
    inner = Scripted([action()], RuntimeError("private details"))
    backend = VerifiedActionBackend(inner, catalog=catalog())
    budget = DiscoveryBudget(time.monotonic() + 2, max_calls=2)
    with pytest.raises(RuntimeError):
        backend.infer_with_budget(inference_request, budget)
    assert budget.calls_used == backend.calls_used == backend.last_inner_calls == 2
    assert inner.requests[1][1] < inner.requests[0][1]
    def expire(call, *unused):
        if call == 1:
            late_budget.deadline = time.monotonic() - .001
    late_budget = DiscoveryBudget(time.monotonic() + 2, max_calls=2)
    late = VerifiedActionBackend(Scripted([action()], hook=expire), catalog=catalog())
    with pytest.raises(TimeoutError):
        late.infer_with_budget(inference_request, late_budget)
    assert late_budget.calls_used == late.calls_used == 1 and not late.last_verifications


def test_busy_transaction_cannot_reset_or_interleave_diagnostics(inference_request):
    entered, release = threading.Event(), threading.Event()
    def hold(call, *unused):
        if call == 1:
            entered.set()
            assert release.wait(2)
    inner = Scripted([action()], hook=hold)
    backend = VerifiedActionBackend(inner, catalog=catalog())
    outputs = []
    thread = threading.Thread(target=lambda: outputs.append(backend.infer(inference_request, 3)))
    thread.start()
    assert entered.wait(1)
    try:
        with pytest.raises(ModelUnavailable, match="runtime_busy"):
            backend.infer(inference_request, 1)
        assert len(inner.requests) == 1 and backend.calls_used == 1
    finally:
        release.set()
        thread.join(2)
    assert not thread.is_alive() and len(outputs) == 1 and backend.last_inner_calls == 2


def test_verified_discovery_decodes_once_counts_raw_calls_and_protects_cv_identity():
    class Frames:
        calls = 0
        def extract(self, path, timestamps, directory, timeout_s):
            self.calls += 1
            result = []
            for index, t in enumerate(timestamps):
                path = directory / f"{index}.png"
                path.write_bytes(b"SCRIPTED SYNTHETIC FRAME PLACEHOLDER")
                result.append(Frame(t, path, t))
            return tuple(result)
    source = Frames()
    inner = Scripted([action()], primitive_payload(facts()))
    backend = VerifiedActionBackend(inner, catalog=catalog())
    original = Candidate("safe_left_open", 0, 1, 0, .51, "CV remains", (0,))
    budget = DiscoveryBudget(time.monotonic() + 3, max_calls=2)
    report = discover_types("PRIVATE.mov", [original], catalog(), duration_sec=10.,
        windows=(DiscoveryWindow(0., 9., (0., 4., 9.)),), allowed_type_ids=TYPE_IDS,
        backend=backend, frame_source=source, budget=budget)
    assert report.status == "completed" and report.retained_candidates[0] is original
    assert source.calls == 1 and report.calls_used == report.budget_calls_used == backend.calls_used == 2
    assert len(report.added_candidates) == 1
    assert inner.requests[0][0].frames == inner.requests[1][0].frames
    assert all(left is right for left, right in zip(inner.requests[0][0].frames, inner.requests[1][0].frames, strict=True))


def test_lifecycle_and_zero_timeout_do_not_load_or_call_inner(inference_request):
    inner = Scripted()
    backend = VerifiedActionBackend(inner, catalog=catalog())
    backend.validate(12)
    assert inner.prepared == [12]
    with pytest.raises(TimeoutError):
        backend.infer(inference_request, 0)
    assert not inner.requests
    backend.close()
    backend.close()
    assert inner.closed == 1
    with pytest.raises(ModelUnavailable, match="runtime_not_prepared"):
        backend.infer(inference_request, 2)


def test_owner_allowlist_is_respected_before_any_second_call(inference_request):
    inference_request = replace(inference_request, allowed_rule_ids=("food_or_drink_at_workplace",))
    inner = Scripted([action(COAT)])
    backend = VerifiedActionBackend(inner, catalog=catalog())
    with pytest.raises(InvalidResponse, match="action_unknown"):
        backend.infer(inference_request, 2)
    assert len(inner.requests) == 1


def test_schema_has_bounded_uniform_eight_field_rows_and_no_proposal_conditioning():
    schema = _schema([{"id": 1, "anchors": [2, 7, 12]}, {"id": 2, "anchors": [4, 9, 9]}], 12)
    episodes = schema["properties"]["episodes"]
    assert episodes["minItems"] == episodes["maxItems"] == 2
    assert episodes["items"] is False and len(episodes["prefixItems"]) == 2
    choices = []
    for identity, expected in ((1, [2, 7, 12]), (2, [4, 9, 9])):
        episode = episodes["prefixItems"][identity - 1]
        assert episode["properties"]["id"] == {"type": "integer", "const": identity}
        rows = episode["properties"]["rows"]
        expected = list(dict.fromkeys(expected))
        assert rows["minItems"] == rows["maxItems"] == len(rows["prefixItems"]) == len(expected)
        assert rows["items"] is False
        for row_schema, anchor in zip(rows["prefixItems"], expected, strict=True):
            assert row_schema["minItems"] == row_schema["maxItems"] == len(row_schema["prefixItems"]) == 8
            assert row_schema["items"] is False
            assert row_schema["prefixItems"][0] == {"type": "integer", "const": anchor}
            choices.append(row_schema["prefixItems"][1:])
            assert all("unknown" in field["enum"] for field in choices[-1])
    assert all(choice == choices[0] for choice in choices)
    with pytest.raises(InvalidResponse):
        _primitives('{"episodes":[],"episodes":[]}', [])


def test_late_proposal_anchors_are_constrained_in_each_generated_phase(inference_request):
    frames = tuple(Frame(float(index), inference_request.frames[0].path, float(index)) for index in range(1, 13))
    inference_request = replace(inference_request, frames=frames)
    selected = action(start_frame=8, peak_frame=11, end_frame=12)
    rows = facts()
    inner = Scripted([selected], primitive_payload(rows))
    backend = VerifiedActionBackend(inner, catalog=catalog())
    [finding] = json.loads(backend.infer(inference_request, 2))["findings"]
    assert finding["t_peak"] == 11.
    assert inner.requests[1][0].frames == tuple(inference_request.frames[index - 1] for index in (8, 11, 12))
    assert all(inner.requests[1][0].frames[local] is inference_request.frames[source - 1]
               for local, source in enumerate((8, 11, 12)))
    schema = inner.requests[1][0].response_schema
    row_schemas = schema["properties"]["episodes"]["prefixItems"][0]["properties"]["rows"]["prefixItems"]
    assert [item["prefixItems"][0]["const"] for item in row_schemas] == [1, 2, 3]
    with pytest.raises(InvalidResponse, match="primitive_values"):
        _primitives(primitive_payload(facts()), [{"id": 1, "anchors": [8, 11, 12]}])


def test_schema_never_treats_boolean_episode_id_as_integer_anchor_identity():
    with pytest.raises(ValueError, match="primitive_schema_anchors"):
        _schema([{"id": True, "anchors": [1, 2, 3]}], 12)


def test_two_episodes_share_at_most_six_original_anchor_frames_and_preserve_source_times(inference_request):
    frames = tuple(Frame(float(index), inference_request.frames[0].path, float(index)) for index in range(1, 13))
    inference_request = replace(inference_request, frames=frames)
    second_rows = facts(FOOD)
    for item, local in zip(second_rows, (4, 5, 6), strict=True):
        item[0] = local
    inner = Scripted([action(start_frame=1, peak_frame=4, end_frame=6),
                      action(FOOD, start_frame=8, peak_frame=10, end_frame=12)],
                     primitive_payload(facts(), second_rows))
    backend = VerifiedActionBackend(inner, catalog=catalog())
    findings = json.loads(backend.infer(inference_request, 2))["findings"]
    selected = inner.requests[1][0].frames
    assert len(selected) == 6 and [frame.timestamp_s for frame in selected] == [1., 4., 6., 8., 10., 12.]
    assert all(frame is frames[source - 1] for frame, source in zip(selected, (1, 4, 6, 8, 10, 12), strict=True))
    assert [(item["t_start"], item["t_peak"], item["t_end"]) for item in findings] == [(1., 4., 6.), (8., 10., 12.)]
    assert backend.calls_used == 2


def test_peak_end_duplicate_anchor_cannot_generate_inconsistent_second_fact(inference_request):
    selected = action(start_frame=1, peak_frame=3, end_frame=3)
    rows = [row(1), row(2, state="resting", place="own_workdesk")]
    inner = Scripted([selected], primitive_payload(rows))
    backend = VerifiedActionBackend(inner, catalog=catalog())
    [finding] = json.loads(backend.infer(inference_request, 2))["findings"]
    assert finding["t_peak"] == finding["t_end"] == 9.7
    schema_rows = inner.requests[1][0].response_schema["properties"]["episodes"]["prefixItems"][0]["properties"]["rows"]
    assert schema_rows["minItems"] == schema_rows["maxItems"] == 2
    assert [item["prefixItems"][0]["const"] for item in schema_rows["prefixItems"]] == [1, 2]
    assert inner.requests[1][0].frames == (inference_request.frames[0], inference_request.frames[2])
