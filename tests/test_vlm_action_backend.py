"""Scripted physical-action decoding, no model/media inference."""
from dataclasses import replace
import json
import time

import pytest

from ml.candidates import Candidate
from ml.kb import catalog
from ml.vlm.action_backend import ActionContrastBackend, _ACTIONS, _NORMAL
from ml.vlm.contracts import Frame, InferenceRequest
from ml.vlm.discovery import DiscoveryBudget, DiscoveryWindow, discover_types
from ml.vlm.discovery_policy import TYPE_IDS
from ml.vlm.validation import InvalidResponse

POSITIVE = "worn_coat_removed_then_placed_on_cashdesk"


def action(name=POSITIVE, **changes):
    return {"start_frame": 1, "peak_frame": 2, "end_frame": 3, "action": name,
            "confidence": .9, "insufficient_evidence": False, **changes}


class Scripted:
    def __init__(self, actions=(), raw=None, delay=0):
        self.raw = raw if raw is not None else json.dumps({"actions": list(actions)})
        self.requests = []
        self.prepared = []
        self.closed = 0
        self.delay = delay

    def validate(self, timeout_s):
        self.prepared.append(timeout_s)

    def close(self):
        self.closed += 1

    def infer(self, inference_request, timeout_s):
        self.requests.append((inference_request, timeout_s))
        time.sleep(self.delay)
        return self.raw


@pytest.fixture
def inference_request(tmp_path):
    frames = []
    for index, timestamp in enumerate((.4, 4.6, 9.7)):
        path = tmp_path / f"{index}.png"
        path.write_bytes(b"SYNTHETIC SCRIPTED EVIDENCE ONLY")
        frames.append(Frame(timestamp, path, timestamp))
    return InferenceRequest("PRIVATE_CV_OBSERVATION_FILE_NAME_LEGAL_QUOTE", tuple(frames), "checklist", None, TYPE_IDS)


@pytest.mark.parametrize("normal", list(_NORMAL))
def test_every_normal_or_unclear_action_returns_empty_findings(inference_request, normal):
    inner = Scripted([action(normal)])
    backend = ActionContrastBackend(inner, catalog=catalog())
    assert json.loads(backend.infer(inference_request, 2)) == {"findings": []}
    assert backend.calls_used == len(inner.requests) == 1
    assert backend.last_actions == (action(normal),)


@pytest.mark.parametrize("name", list(_ACTIONS))
def test_positive_indices_map_to_actual_source_times_with_fixed_phases(inference_request, name):
    backend = ActionContrastBackend(Scripted([action(name)]), catalog=catalog())
    [finding] = json.loads(backend.infer(inference_request, 2))["findings"]
    assert finding["type_id"] == _ACTIONS[name].type_id
    assert (finding["t_start"], finding["t_peak"], finding["t_end"]) == (.4, 4.6, 9.7)
    assert [item["text"] for item in finding["observations"]] == list(_ACTIONS[name].phases)
    assert not finding["insufficient_evidence"] and finding["confidence"] == .9


def test_model_input_is_balanced_physical_actions_without_owner_prompt_or_type_labels(inference_request):
    inner = Scripted([action("idle_or_other_action")])
    backend = ActionContrastBackend(inner, catalog=catalog())
    backend.infer(inference_request, 2)
    translated, timeout = inner.requests[0]
    assert translated.frames is inference_request.frames and 0 < timeout <= 2
    assert translated.candidate_id is None and translated.allowed_rule_ids == ()
    assert inference_request.prompt not in translated.prompt
    assert all(kind not in translated.prompt for kind in TYPE_IDS)
    schema = translated.response_schema["properties"]["actions"]
    properties = schema["items"]["properties"]
    assert schema["maxItems"] == 2
    assert set(properties["action"]["enum"]) == set(_ACTIONS) | set(_NORMAL)
    assert properties["start_frame"] == {"type": "integer", "enum": [1, 2, 3]}
    assert "most conclusive observed consequence" in translated.prompt


@pytest.mark.parametrize("changes", [{"insufficient_evidence": True}])
def test_uncertain_positive_cannot_add(inference_request, changes):
    backend = ActionContrastBackend(Scripted([action(**changes)]), catalog=catalog())
    assert json.loads(backend.infer(inference_request, 2)) == {"findings": []}


@pytest.mark.parametrize("gate", ["disabled", "draft", "not_allowed"])
def test_verified_owner_allowlist_gate_cannot_be_bypassed(inference_request, gate):
    types = dict(catalog())
    kind = _ACTIONS[POSITIVE].type_id
    if gate == "disabled":
        types[kind] = replace(types[kind], enabled=False)
    if gate == "draft":
        types[kind] = replace(types[kind], status="draft", rule=None)
    if gate == "not_allowed":
        inference_request = replace(inference_request, allowed_rule_ids=("food_or_drink_at_workplace",))
    inner = Scripted([action()])
    backend = ActionContrastBackend(inner, catalog=types)
    with pytest.raises(InvalidResponse, match="action_unknown"):
        backend.infer(inference_request, 2)
    choices = inner.requests[0][0].response_schema["properties"]["actions"]["items"]["properties"]["action"]["enum"]
    assert POSITIVE not in choices and set(_NORMAL) <= set(choices)


@pytest.mark.parametrize("raw", [
    'not JSON', '{"actions":[],"actions":[]}', '```json\n{"actions":[]}\n```',
    json.dumps({"actions": [action(extra="no free essay")]}),
    json.dumps({"actions": [action(action="unknown")]}),
    json.dumps({"actions": [action(start_frame=True)]}),
    json.dumps({"actions": [action(start_frame=1.0)]}),
    json.dumps({"actions": [action(peak_frame=4)]}),
    json.dumps({"actions": [action(peak_frame=3,end_frame=2)]}),
    json.dumps({"actions": [action(confidence=float("nan"))]}),
    json.dumps({"actions": [action(confidence=True)]}),
    json.dumps({"actions": [action(insufficient_evidence="false")]}),
    json.dumps({"actions": [action(), action(), action()]}),
    json.dumps({"actions": [action(), action(action="unknown")]}),
])
def test_bad_payload_is_atomic_and_never_emits_valid_prefix(inference_request, raw):
    backend = ActionContrastBackend(Scripted(raw=raw), catalog=catalog())
    with pytest.raises(InvalidResponse):
        backend.infer(inference_request, 2)
    assert backend.last_actions == ()


def test_sequence_requires_distinct_prerequisite_and_consequence_but_food_may_be_single_frame(inference_request):
    backend = ActionContrastBackend(Scripted([action(start_frame=2)]), catalog=catalog())
    with pytest.raises(InvalidResponse, match="action_order_unobserved"):
        backend.infer(inference_request, 2)
    backend = ActionContrastBackend(Scripted([action("food_or_drink_actually_consumed", start_frame=2, end_frame=2)]), catalog=catalog())
    [finding] = json.loads(backend.infer(inference_request, 2))["findings"]
    assert finding["observations"] == [{"timestamp_s": 4.6, "text": _ACTIONS["food_or_drink_actually_consumed"].phases[1]}]


def test_lifecycle_and_timeout_are_forwarded_without_loading_models(inference_request):
    inner = Scripted([action("unclear")])
    backend = ActionContrastBackend(inner, catalog=catalog())
    backend.validate(12)
    backend.close()
    assert inner.prepared == [12] and inner.closed == 1
    with pytest.raises(TimeoutError):
        backend.infer(inference_request, 0)
    assert not inner.requests
    late = ActionContrastBackend(Scripted([action()], delay=.02), catalog=catalog())
    with pytest.raises(TimeoutError):
        late.infer(inference_request, .001)
    assert late.last_actions == ()


def test_actual_discovery_accepts_only_physical_positive_and_preserves_cv_identity():
    class Frames:
        def extract(self, path, timestamps, directory, timeout_s):
            result = []
            for index,t in enumerate(timestamps):
                path = directory/f"{index}.png"
                path.write_bytes(b"SYNTHETIC SCRIPTED EVIDENCE ONLY")
                result.append(Frame(t,path,t))
            return tuple(result)
    inner = Scripted([action("visitor_or_courier_bag_handled"), action()])
    backend = ActionContrastBackend(inner, catalog=catalog())
    original = Candidate("safe_left_open",0,1,0,.51,"Исходный CV-кандидат",(0,))
    report = discover_types("PRIVATE.mov",[original],catalog(),duration_sec=10.,
        windows=(DiscoveryWindow(0.,9.,(0.,4.,9.)),),allowed_type_ids=TYPE_IDS,
        backend=backend,frame_source=Frames(),budget=DiscoveryBudget(time.monotonic()+2,max_calls=1))
    assert report.status == "completed" and report.retained_candidates[0] is original
    assert len(inner.requests) == report.calls_used == backend.calls_used == 1
    [added] = report.added_candidates
    assert added.type_id == "personal_items_on_desk" and added.t_peak == 4.
    assert "снятую верхнюю одежду" in added.observation
