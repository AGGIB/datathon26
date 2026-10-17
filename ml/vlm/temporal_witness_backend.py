"""Experimental full-window blind inventory, followed by local witness search.

The verifier inspects the SAME already-decoded source Frames, once each, without
proposed labels, confidence or temporal anchors. Existing compact predicates are
unchanged. Accepted findings are relocalized to observed CV index/FPS timestamps;
these are not claims about container PTS. No owner/default wiring is installed.
"""
from itertools import combinations
import json

from .action_backend import _ACTIONS
from .compact_verified_action_backend import _ATOMS, _CONTINUITY, _parse_atoms, _supports
from .contracts import Frame, InferenceRequest
from .validation import InvalidResponse, number, strict_json
from .verified_action_backend import MIN_PROPOSAL_CONFIDENCE, VerifiedActionBackend

DECODER_POLICY = "physical_action_search_v4_blind_full_window_temporal_witness_v1"
RECOMMENDED_MAX_NEW_TOKENS = 768  # Caller-owned experimental worker cap, never configured here.
_INSTRUCTION = (
    "Inspect ALL supplied images afresh, once each in source order. Report visible physical facts "
    "for one neutral continuing interaction/object track across the supplied window. Return one row "
    "for EVERY supplied frame, using its exact 1-based frame index. Choose up to two coherent atoms "
    "at each frame, using unknown for an unused slot or unresolved fact. Every selected atom asserts "
    "ALL words in its name are directly visible, not merely a nearby object or a plausible action. "
    "No atom is preferred; ordinary contrasts and unknown are valid. Do not force the closest "
    "positive atom. Scan hands and work areas as well as the rest of each image; do not automatically "
    "choose the largest/central object. If multiple tracks are equally plausible, use unknown. Do not "
    "switch to an unrelated person/object between rows. cashier means visibly operating the workstation, "
    "visitor means being served; appearance alone cannot establish role. Drinking requires an "
    "identifiable cup actually at the mouth with visible drinking; a held cup, face touch or standing "
    "container alone is insufficient. A visible bite requires identifiable food and an actual bite. "
    "Hand contact and mouth contact can coexist while drinking. Identifiable outerwear must be "
    "distinguished from a bag or papers. A resolved inserted key and an identified safe must both be "
    "visible; a highlight/handle is insufficient. Departure must be visibly FROM the named fixture/service; "
    "away means later away from that fixture, possibly still visible elsewhere, not hidden by occlusion. "
    "Banknotes, stamp or book must be identifiable. Completed service/different visitor need an actual "
    "visible transition, not several people alone. No absent checks, intent, authenticity or rules. "
    "same_actor tracks the SAME CASHIER across cashier-bearing facts; distinct visitors can legitimately "
    "appear, and a recipient visitor must be a different person from that cashier. same_object tracks "
    "the focal object across object-bearing facts. yes requires visible identity continuity; no means "
    "a known change, unknown means unresolved. Never force continuity or infer it solely from similarity "
    "across a gap. Select atoms before deciding continuity. Return only JSON episodes with id, rows of "
    "[exact_frame_index,atom1,atom2],same_actor,same_object. No explanations or confidence. "
)


def _schema(frame_count):
    if type(frame_count) is not int or not 1 <= frame_count <= 12:
        raise ValueError("inventory_frame_count")
    rows = [{"type": "array", "minItems": 3, "maxItems": 3, "items": False,
             "prefixItems": [{"type": "integer", "const": index},
                             {"type": "string", "enum": list(_ATOMS)},
                             {"type": "string", "enum": list(_ATOMS)}]}
            for index in range(1, frame_count + 1)]
    properties = {
        "id": {"type": "integer", "const": 1},
        "rows": {"type": "array", "minItems": frame_count, "maxItems": frame_count,
                 "prefixItems": rows, "items": False},
        "same_actor": {"type": "string", "enum": list(_CONTINUITY)},
        "same_object": {"type": "string", "enum": list(_CONTINUITY)},
    }
    return {"type": "object", "additionalProperties": False, "required": ["episodes"],
            "properties": {"episodes": {"type": "array", "minItems": 1, "maxItems": 1, "items": False,
                "prefixItems": [{"type": "object", "additionalProperties": False,
                                 "required": list(properties), "properties": properties}]}}}


def _inventory_request(frames):
    if (not isinstance(frames, (tuple, list)) or not 1 <= len(frames) <= 12
            or any(not isinstance(frame, Frame) or not number(frame.timestamp_s) or frame.timestamp_s < 0 for frame in frames)):
        raise InvalidResponse("inventory_frames")
    times = tuple(frame.timestamp_s for frame in frames)
    if times != tuple(sorted(set(times))):
        raise InvalidResponse("inventory_frames")
    schema = _schema(len(frames))
    prompt = _INSTRUCTION + json.dumps({
        "episodes": [{"id": 1}], "row_fields": ["frame", "atom1", "atom2"],
        "atom_choices": list(_ATOMS), "continuity_choices": list(_CONTINUITY),
        "frames_in_supplied_order": [{"frame": index, "timestamp_s": frame.timestamp_s}
                                     for index, frame in enumerate(frames, 1)],
    }, separators=(",", ":"))
    return InferenceRequest(prompt, frames, "checklist", None, (), schema)


def _witness(action, evidence, frames):
    """Choose only fully supported source witnesses, without temporal gold/hints."""
    facts = evidence["phases"]
    continuity = {key: evidence[key] for key in ("same_actor", "same_object")}
    if not _ACTIONS[action].ordered:
        # Consumption is a directly visible peak fact. Do not invent the approach
        # or finish merely because other window frames exist.
        return next(((index, index, index) for index, fact in enumerate(facts)
                     if _supports(action, {"phases": (fact, fact, fact), **continuity})), None)
    supported = [indices for indices in combinations(range(len(facts)), 3)
                 if _supports(action, {"phases": tuple(facts[index] for index in indices), **continuity})]
    return min(supported, key=lambda indices: (
        frames[indices[2]].timestamp_s - frames[indices[0]].timestamp_s,
        frames[indices[1]].timestamp_s, frames[indices[0]].timestamp_s, frames[indices[2]].timestamp_s,
    ), default=None)


def _relocalized(finding, action, witness, frames):
    start, peak, end = (frames[index].timestamp_s for index in witness)
    spec = _ACTIONS[action]
    # Consequence wins for repeated source timestamps (peak-only consumption).
    phases = {end: spec.phases[2], start: spec.phases[0], peak: spec.phases[1]}
    return {**finding, "t_start": start, "t_peak": peak, "t_end": end,
            "observations": [{"timestamp_s": time, "text": phases[time]} for time in sorted(phases)]}


class TemporalWitnessBackend(VerifiedActionBackend):
    """At most one proposal and one full-window blind inventory call per window.

    Parent owns absolute deadlines, raw-call accounting, lock and lifecycle.
    Multiple eligible proposals abstain without another model call: one neutral
    inventory cannot safely associate several different proposed interactions.
    ``last_verifications`` includes coded status and 1-based source witness indices.
    """
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
        if len(proposals) != 1:
            self.last_verifications = tuple({"episode_id": index, "accepted": False,
                "status": "scope_abstention", "reason": "multiple_proposals_not_supported",
                "witness_frame_indices": []} for index in range(1, len(proposals) + 1))
            return '{"findings":[]}'
        translated = _inventory_request(request.frames)
        inventory_raw = self._proxy.infer(translated, timeout_s=budget.remaining())
        if budget.remaining() <= 0:
            raise TimeoutError("verification_timeout")
        episodes = [{"id": 1, "anchors": list(range(1, len(request.frames) + 1))}]
        evidence = _parse_atoms(inventory_raw, episodes)[1]
        action, finding = proposals[0]
        witness = _witness(action["action"], evidence, request.frames)
        if budget.remaining() <= 0:
            raise TimeoutError("verification_timeout")
        self.last_verifications = ({"episode_id": 1, "accepted": witness is not None,
            "status": "supported" if witness is not None else "no_witness",
            "reason": "visible_temporal_witness_supported" if witness is not None else "primitive_evidence_not_supported",
            "witness_frame_indices": [index + 1 for index in witness] if witness is not None else []},)
        findings = [_relocalized(finding, action["action"], witness, request.frames)] if witness is not None else []
        return json.dumps({"findings": findings}, ensure_ascii=False, allow_nan=False)
