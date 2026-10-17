"""Experimental blind visual primitive gate over V4 proposals, using the same frames.

The second model request contains neutral episode IDs and source indices only:
no proposed action, type, first confidence, reasoning, fixed phases or owner text.
Both visual calls share the owner's absolute deadline and raw-call budget. This
is correlated confirmation by the same model, not independent visual truth.
"""
from contextlib import contextmanager
import json
import threading
import time

from .action_backend import ActionContrastBackend, _ACTIONS
from .contracts import InferenceRequest
from .discovery import BudgetExhausted, DiscoveryBudget
from .preflight import ModelUnavailable
from .validation import InvalidResponse, number, strict_json

VERIFICATION_POLICY = "blind_episode_primitives_v3"
DECODER_POLICY = "physical_action_search_v4_blind_episode_primitives_v3"
MIN_PROPOSAL_CONFIDENCE = .7
_ACTORS = ("a", "b", "c", "not_visible", "unknown")
_ROLES = ("cashier", "visitor", "other", "not_visible", "unknown")
_OBJECT_IDS = ("x", "y", "z", "unknown")
_OBJECTS = ("banknotes", "papers", "bag", "outerwear", "food", "cup", "key",
            "highlight_handle", "stamp", "book", "other", "unknown")
_STATES = ("held", "placing", "resting", "processing", "handing", "counting",
           "biting", "drinking", "inserted", "loose", "other", "unknown")
_PLACES = ("hand", "own_workdesk", "visitor_side", "identified_safe_lock", "unidentified_lock",
           "mouth", "device", "elsewhere", "unknown")
_DIRECTIONS = ("to_visitor", "from_visitor", "to_cashier", "from_cashier", "visitor_leaving",
               "visitor_absent", "cashier_leaving", "cashier_absent", "next_visitor_served",
               "service_complete", "none", "unknown")
_ENUMS = (_ACTORS, _ROLES, _OBJECT_IDS, _OBJECTS, _STATES, _PLACES, _DIRECTIONS)
_INSTRUCTION = (
    "Inspect the supplied anchor images afresh. Describe only visible physical facts for each neutral episode. "
    "phase_anchors gives BEFORE, PEAK, AFTER; return one row for each unique anchor in unique_anchors order. "
    "Repeated phase anchors are the SAME image: describe it ONCE, never invent separate phase facts. "
    "Use the exact supplied local frame indices. "
    "No inference about intent, authenticity, rights, omissions or rules. Use unknown for unresolved facts; "
    "do not select an expected action. All options are alternatives, none is preferred. "
    "Return JSON {\"episodes\":[{\"id\":1,\"rows\":[[frame,actor,role,object_id,object,state,place,direction],...]},...]}. "
    "One result per episode. Look for one coherent person-object interaction or continuing object track "
    "bracketed by the episode anchors. Scan the whole anchor content, including hands and work areas. "
    "Do not automatically describe the largest or central object, or switch to an unrelated object between rows. "
    "When multiple interactions are equally plausible or the focal track cannot be resolved, use unknown. "
    "Actors a/b/c and objects x/y/z are local "
    "identity tracks within an episode, not screen positions. Reuse an ID only when visual continuity "
    "supports the SAME person/object; similar appearance alone or an occluded gap requires unknown. "
    "Do not force an identity track to stay constant when it cannot be established. "
    "An actor not visible uses not_visible for actor AND role; do not substitute a different visible person. "
    "cashier means visibly operating the workstation, visitor means being served; clothing alone cannot establish role. "
    "Distinguish banknotes from papers, bag from identifiable outerwear, stamp/book from a rectangle, "
    "and a visibly resolved key from a highlight/handle. inserted requires visible insertion in a lock; "
    "identified_safe_lock requires a visibly identified safe, not an inferred lock. "
    "biting means an actual visible bite of food; drinking means an identifiable cup at the mouth with a "
    "visible drinking act, not face touching or a nearby container. Place refers ONLY to the object's "
    "immediate contact/location, never the actor's position, room or overall workplace. An object held "
    "above a desk is at hand, not own_workdesk; mouth requires the object actually at the mouth. "
    "own_workdesk means the object is on the cashier's working desk surface. Use unknown if the object's "
    "immediate location cannot be resolved. Direction is the visible transfer "
    "direction or scene transition: leaving requires a visible departure, absent requires the earlier "
    "person has visibly left, not merely hidden. next_visitor_served requires a visibly completed prior "
    "service followed by actual service of a different visitor; several people alone is insufficient. "
    "service_complete must be visibly established. none means no such movement/transition, unknown means unclear. "
    "No free text or confidence. Return only the constrained JSON. "
)


def _schema(episodes, frame_count):
    # Every episode gets the SAME balanced enum sets. No action-conditioned choices.
    count = len(episodes)
    if (not 1 <= count <= 2 or type(frame_count) is not int or not 1 <= frame_count <= 12
            or any(type(episode["id"]) is not int or episode["id"] != index or len(episode["anchors"]) != 3
                or any(type(anchor) is not int or not 1 <= anchor <= frame_count for anchor in episode["anchors"])
                for index, episode in enumerate(episodes, 1))):
        raise ValueError("primitive_schema_anchors")
    schemas = []
    for episode in episodes:
        unique = tuple(dict.fromkeys(episode["anchors"]))
        rows = [{"type": "array", "minItems": 8, "maxItems": 8, "items": False,
            "prefixItems": [{"type": "integer", "const": anchor},
                *({"type": "string", "enum": list(values)} for values in _ENUMS)]}
            for anchor in unique]
        schemas.append({"type": "object", "additionalProperties": False,
            "properties": {"id": {"type": "integer", "const": episode["id"]},
                "rows": {"type": "array", "minItems": len(unique), "maxItems": len(unique),
                         "prefixItems": rows, "items": False}},
            "required": ["id", "rows"]})
    return {"type": "object", "additionalProperties": False,
        "properties": {"episodes": {"type": "array", "minItems": count, "maxItems": count,
                                    "prefixItems": schemas, "items": False}},
        "required": ["episodes"]}


def _primitives(raw, episodes):
    value = strict_json(raw)
    if not isinstance(value, dict) or set(value) != {"episodes"} or not isinstance(value["episodes"], list):
        raise InvalidResponse("primitive_schema")
    expected = {episode["id"]: episode["anchors"] for episode in episodes}
    if len(value["episodes"]) != len(expected):
        raise InvalidResponse("primitive_episode_coverage")
    result = {}
    for episode in value["episodes"]:
        if not isinstance(episode, dict) or set(episode) != {"id", "rows"}:
            raise InvalidResponse("primitive_schema")
        identity, rows = episode["id"], episode["rows"]
        if type(identity) is not int or identity not in expected or identity in result:
            raise InvalidResponse("primitive_episode_identity")
        unique = tuple(dict.fromkeys(expected[identity]))
        if not isinstance(rows, list) or len(rows) != len(unique):
            raise InvalidResponse("primitive_phase_coverage")
        seen = {}
        for row, index in zip(rows, unique, strict=True):
            if (not isinstance(row, list) or len(row) != 8 or type(row[0]) is not int or row[0] != index
                    or any(not isinstance(item, str) or item not in choices for item, choices in zip(row[1:], _ENUMS, strict=True))):
                raise InvalidResponse("primitive_values")
            if (row[1] == "not_visible") != (row[2] == "not_visible"):
                raise InvalidResponse("primitive_actor_visibility")
            seen[row[0]] = row
        # Duplicate phase anchors use exactly one observed row, never a second
        # model claim about the same pixels. Preserve the three-phase matcher.
        result[identity] = [list(seen[index]) for index in expected[identity]]
    return result


def _cashier(rows):
    return all(row[1] in ("a", "b", "c") and row[2] == "cashier" for row in rows) and len({row[1] for row in rows}) == 1


def _same_object(rows, kinds):
    return all(row[3] in ("x", "y", "z") and row[4] in kinds for row in rows) and len({(row[3], row[4]) for row in rows}) == 1


def _departure(rows, kinds, state, place):
    before, peak, after = rows
    return (_cashier([before]) and before[7] == "cashier_leaving"
        and all(row[1:3] == ["not_visible", "not_visible"] and row[7] == "cashier_absent" for row in (peak, after))
        and _same_object(rows, kinds) and all(row[5] == state and row[6] == place for row in rows))


def _consumption(rows):
    """A visible bite/drink is a peak fact; adjacent unknowns cannot negate it.

    Known adjacent role or identity/object contradictions still reject. This
    does not let a standing cup or face touch replace actual consumption.
    """
    peak = rows[1]
    if (not _cashier([peak]) or not _same_object([peak], {"food", "cup"}) or peak[6] != "mouth"
            or (peak[4], peak[5]) not in (("food", "biting"), ("cup", "drinking"))):
        return False
    for row in (rows[0], rows[2]):
        if row[2] in ("visitor", "other"):
            return False
        if row[1] in ("a", "b", "c") and row[1] != peak[1]:
            return False
        if row[3] in ("x", "y", "z") and row[3] != peak[3]:
            return False
        if row[4] != "unknown" and row[4] != peak[4]:
            return False
    return True


def _supports(action, rows):
    before, peak, after = rows
    if action == "cashier_places_removed_outerwear_on_cashdesk":
        post_actor = (after[1:3] == ["not_visible", "not_visible"]
                      or (after[1:3] == peak[1:3]))
        return (_cashier([before, peak]) and post_actor and _same_object(rows, {"outerwear"})
            and before[5:7] == ["held", "hand"] and peak[5] in ("placing", "resting")
            and peak[6] == "own_workdesk" and after[5:7] == ["resting", "own_workdesk"])
    if action == "food_or_drink_actually_consumed":
        return _consumption(rows)
    if action == "key_remains_in_safe_lock_after_departure":
        return _departure(rows, {"key"}, "inserted", "identified_safe_lock")
    if action == "stamp_or_book_remains_after_departure":
        return _departure(rows, {"stamp", "book"}, "resting", "own_workdesk")
    if action == "banknotes_counted_after_visitor_departure":
        return (_cashier(rows) and _same_object(rows, {"banknotes"}) and before[7] == "visitor_leaving"
            and before[5] in ("held", "resting", "counting") and before[6] in ("hand", "own_workdesk")
            and all(row[5] == "counting" and row[6] in ("hand", "own_workdesk") and row[7] == "visitor_absent" for row in (peak, after)))
    if action == "earlier_banknotes_remain_during_next_service":
        return (_cashier(rows) and _same_object(rows, {"banknotes"}) and before[7] == "service_complete"
            and all(row[5:7] == ["resting", "own_workdesk"] for row in rows)
            and peak[7] == after[7] == "next_visitor_served")
    if action == "processed_banknotes_handed_back":
        recipient = after[1] in ("a", "b", "c") and after[2] == "visitor" and after[1] != before[1]
        peak_transfer = ((peak[1:3] == before[1:3] and peak[5] == "handing" and peak[7] == "to_visitor")
                         or (peak[1:3] == after[1:3] and peak[5] == "held" and peak[7] == "from_cashier"))
        return (_cashier([before]) and _same_object(rows, {"banknotes"}) and before[5] == "processing"
            and before[6] in ("hand", "own_workdesk", "device") and peak[6] in ("hand", "visitor_side")
            and recipient and peak_transfer and after[5] == "held" and after[6] in ("hand", "visitor_side")
            and after[7] in ("from_cashier", "none"))
    return False


class _ClaimingInner:
    def __init__(self, owner):
        self.owner = owner

    def infer(self, request, timeout_s):
        owner, budget = self.owner, self.owner._budget
        if budget.remaining() <= 0:
            raise TimeoutError("verification_timeout")
        if not budget.claim():
            raise BudgetExhausted("verification_budget_exhausted")
        remaining = budget.remaining()
        if remaining <= 0:
            raise TimeoutError("verification_timeout")
        owner.calls_used += 1
        owner.last_inner_calls += 1
        return owner.inner.infer(request, timeout_s=remaining)


class VerifiedActionBackend:
    """One proposal call plus at most one blind primitive call per window.

    Default-off owner wiring chooses this profile. `calls_used` is cumulative
    raw attempts; the shared DiscoveryBudget is authoritative across windows.
    `last_verifications` retains only episode IDs, outcomes and fixed reasons.
    Worker loading, token cap and hard cancellation remain the inner's concern.
    """
    def __init__(self, inner, *, catalog):
        self.inner = inner
        self._proxy = _ClaimingInner(self)
        self._proposal = ActionContrastBackend(self._proxy, catalog=catalog)
        self._lock = threading.Lock()
        self._budget = None
        self._closed = False
        self.calls_used = 0
        self.last_inner_calls = 0
        self.last_actions = ()
        self.last_verifications = ()

    @contextmanager
    def _exclusive(self, *, allow_closed=False):
        if not self._lock.acquire(blocking=False):
            raise ModelUnavailable("runtime_busy")
        try:
            if self._closed and not allow_closed:
                raise ModelUnavailable("runtime_not_prepared")
            yield
        finally:
            self._lock.release()

    def validate(self, timeout_s):
        with self._exclusive():
            validate = getattr(self.inner, "validate", None)
            if validate is not None:
                return validate(timeout_s=timeout_s)

    def close(self):
        with self._exclusive(allow_closed=True):
            if self._closed:
                return
            self._closed = True
            close = getattr(self.inner, "close", None)
            if close is not None:
                return close()

    def infer(self, request, timeout_s):
        if not number(timeout_s) or timeout_s <= 0:
            raise TimeoutError("verification_timeout")
        return self.infer_with_budget(request, DiscoveryBudget(time.monotonic() + timeout_s, max_calls=2))

    def infer_with_budget(self, request, budget):
        with self._exclusive():
            if not isinstance(budget, DiscoveryBudget):
                raise ValueError("verification_budget_required")
            self._budget = budget
            self.last_inner_calls, self.last_actions, self.last_verifications = 0, (), ()
            try:
                return self._infer(request, budget)
            finally:
                self._budget = None

    def _infer(self, request, budget):
        raw = self._proposal.infer(request, timeout_s=budget.remaining())
        actions = self._proposal.last_actions
        proposal_findings = strict_json(raw)["findings"]
        positive = [action for action in actions if action["action"] in _ACTIONS and not action["insufficient_evidence"]]
        if len(positive) != len(proposal_findings):
            raise InvalidResponse("verification_proposal_mapping")
        proposals = [(action, finding) for action, finding in zip(positive, proposal_findings, strict=True)
                     if action["confidence"] >= MIN_PROPOSAL_CONFIDENCE]
        self.last_actions = actions
        if not proposals:
            return '{"findings":[]}'
        original = [{"id": index, "anchors": [action[key] for key in ("start_frame", "peak_frame", "end_frame")]}
                    for index, (action, _) in enumerate(proposals, 1)]
        source_indices = sorted({anchor for episode in original for anchor in episode["anchors"]})
        frames = tuple(request.frames[index - 1] for index in source_indices)
        local = {source: index for index, source in enumerate(source_indices, 1)}
        episodes = [{"id": episode["id"], "anchors": [local[index] for index in episode["anchors"]]}
                    for episode in original]
        translated = self._verification_request(episodes, frames)
        verified_raw = self._proxy.infer(translated, timeout_s=budget.remaining())
        if budget.remaining() <= 0:
            raise TimeoutError("verification_timeout")
        facts = self._verification_facts(verified_raw, episodes)
        findings, diagnostics = [], []
        for index, (action, finding) in enumerate(proposals, 1):
            supported = self._supports_proposal(action["action"], facts[index])
            diagnostics.append({"episode_id": index, "accepted": supported,
                                "reason": "visible_primitives_supported" if supported else "primitive_evidence_not_supported"})
            if supported:
                findings.append(finding)
        if budget.remaining() <= 0:
            raise TimeoutError("verification_timeout")
        self.last_verifications = tuple(diagnostics)
        return json.dumps({"findings": findings}, ensure_ascii=False, allow_nan=False)

    def _verification_request(self, episodes, frames):
        prompt_episodes = [{"id": episode["id"], "phase_anchors": episode["anchors"],
                            "unique_anchors": list(dict.fromkeys(episode["anchors"]))} for episode in episodes]
        prompt = _INSTRUCTION + json.dumps({"row_fields": ["frame", "actor", "role", "object_id", "object", "state", "place", "direction"],
            "enums": [list(values) for values in _ENUMS], "episodes": prompt_episodes,
            "frames_in_supplied_order": [{"frame": index, "timestamp_s": frame.timestamp_s}
                for index, frame in enumerate(frames, 1)]}, separators=(",", ":"))
        return InferenceRequest(prompt, frames, "checklist", None, (), _schema(episodes, len(frames)))

    def _verification_facts(self, raw, episodes):
        return _primitives(raw, episodes)

    def _supports_proposal(self, action, facts):
        return _supports(action, facts)
