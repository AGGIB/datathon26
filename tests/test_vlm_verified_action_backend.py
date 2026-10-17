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
        assert translated.frames is inference_request.frames
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
    (COAT, 1, 6, "visitor_side"), (COAT, 0, 7, "unknown"),
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


def test_same_frame_must_have_identical_facts_and_food_can_be_one_frame(inference_request):
    selected = action(FOOD, start_frame=2, peak_frame=2, end_frame=2)
    eating = row(2, "food", "biting", "mouth")
    backend = VerifiedActionBackend(Scripted([selected], primitive_payload([eating] * 3)), catalog=catalog())
    assert len(json.loads(backend.infer(inference_request, 2))["findings"]) == 1
    inconsistent = [eating, eating, row(2, "food", "held", "hand")]
    backend = VerifiedActionBackend(Scripted([selected], primitive_payload(inconsistent)), catalog=catalog())
    with pytest.raises(InvalidResponse, match="primitive_same_frame_conflict"):
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


def test_inner_failure_counts_attempt_and_deadline_never_renews(inference_request):
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
    assert inner.requests[0][0].frames is inner.requests[1][0].frames


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
    schema = _schema(2, 12)
    episodes = schema["properties"]["episodes"]
    rows = episodes["items"]["properties"]["rows"]
    row_schema = rows["items"]
    assert episodes["minItems"] == episodes["maxItems"] == 2
    assert rows["minItems"] == rows["maxItems"] == 3
    assert row_schema["minItems"] == row_schema["maxItems"] == len(row_schema["prefixItems"]) == 8
    assert row_schema["items"] is False
    assert all("unknown" in field["enum"] for field in row_schema["prefixItems"][1:])
    with pytest.raises(InvalidResponse):
        _primitives('{"episodes":[],"episodes":[]}', [])
