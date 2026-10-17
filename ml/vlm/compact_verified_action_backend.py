"""Optional blind coherent visual atoms; default legacy primitive profile stays available.

These atoms couple role, object and contact/action, avoiding a Cartesian product
of mutually exclusive locations. They are visual model claims, not truth. No
chosen first-pass action, type, confidence or owner prompt reaches verification.
"""
import json

from .contracts import InferenceRequest
from .validation import InvalidResponse, strict_json
from .verified_action_backend import VerifiedActionBackend, _schema as _primitive_schema

DECODER_POLICY = "physical_action_search_v4_blind_physical_atoms_v4"
_CONTINUITY = ("yes", "no", "unknown")
_ATOMS = (
    "cashier_holds_identifiable_outerwear",
    "cashier_places_outerwear_on_own_workdesk",
    "outerwear_resting_on_own_workdesk",
    "cashier_drinks_identifiable_cup_at_mouth",
    "cashier_takes_visible_bite_of_identifiable_food",
    "resolved_key_inserted_identified_safe_lock",
    "cashier_visibly_leaves_safe",
    "cashier_away_from_safe",
    "identifiable_stamp_or_book_resting_on_own_workdesk",
    "cashier_visibly_leaves_workdesk",
    "cashier_away_from_workdesk",
    "cashier_processes_identifiable_banknotes",
    "cashier_holds_identifiable_banknotes",
    "cashier_hands_identifiable_banknotes_to_visitor",
    "distinct_visitor_holds_identifiable_banknotes",
    "cashier_counts_identifiable_banknotes",
    "identifiable_banknotes_rest_on_own_workdesk",
    "visitor_visibly_leaves_service",
    "visitor_away_from_service",
    "previous_service_visibly_complete",
    "different_visitor_actually_being_served",
    "visitor_bag_handled",
    "visitor_holds_identifiable_outerwear",
    "papers_handled",
    "cashier_receives_banknotes_from_visitor",
    "cashier_counts_banknotes_during_visible_service",
    "container_standing",
    "hand_near_face",
    "highlight_or_handle",
    "ambiguous_object",
    "ordinary_work",
    "unknown",
)
_INSTRUCTION = (
    "Inspect only the supplied anchor images afresh. Report visible physical facts for each neutral episode. "
    "phase_anchors are BEFORE, PEAK, AFTER; return each unique anchor ONCE in unique_anchors order. "
    "Choose up to two coherent atoms about one focal interaction/continuing track at that anchor, using "
    "unknown for an unused slot or unresolved fact. Every selected atom asserts ALL words in its name "
    "are directly supported, not merely a nearby object or plausible action. No atom is preferred; ordinary "
    "contrasts and unknown are valid. Do not force the closest positive atom. Scan hands and work areas "
    "as well as the rest of each image; do not automatically choose the largest/central object. If multiple "
    "tracks are equally plausible, use unknown. Do not switch to an unrelated person/object between anchors. "
    "cashier means visibly operating the workstation, visitor means being served; appearance alone cannot "
    "establish role. Drinking requires an identifiable cup actually at the mouth with visible drinking; "
    "a held cup, face touch or standing container alone is insufficient. A visible bite requires identifiable "
    "food and an actual bite. Hand contact and mouth contact can coexist while drinking. Identifiable "
    "outerwear must be distinguished from a bag or papers. A resolved inserted key and an identified safe "
    "must both be visible; a highlight/handle is insufficient. Departure must be visibly FROM the named "
    "fixture/service; away means later away from that fixture, possibly still visible elsewhere, not hidden "
    "by occlusion. Banknotes, stamp or book must be identifiable. Completed service/different visitor need "
    "actual visible transition, not several people alone. No absent checks, intent, authenticity or rules. "
    "same_actor tracks the SAME CASHIER across cashier-bearing facts; distinct visitors can legitimately "
    "appear, and a recipient visitor must be a different person from that cashier. same_object tracks the "
    "focal object across object-bearing facts. yes requires visible identity continuity; no means "
    "a known change, unknown means unresolved. Never force continuity or infer it solely from similarity "
    "across a gap. Select atoms before deciding continuity. Return only JSON episodes with id, rows of "
    "[exact_local_frame,atom1,atom2],same_actor,same_object. No explanations or confidence. "
)


def _schema(episodes, frame_count):
    # Reuse the source-bound validation, without conditioning choices on an action.
    _primitive_schema(episodes, frame_count)
    episode_schemas = []
    for episode in episodes:
        unique = tuple(dict.fromkeys(episode["anchors"]))
        row_schemas = [{"type": "array", "minItems": 3, "maxItems": 3, "items": False,
            "prefixItems": [{"type": "integer", "const": anchor},
                            {"type": "string", "enum": list(_ATOMS)},
                            {"type": "string", "enum": list(_ATOMS)}]} for anchor in unique]
        properties = {"id": {"type": "integer", "const": episode["id"]},
            "rows": {"type": "array", "minItems": len(unique), "maxItems": len(unique),
                     "prefixItems": row_schemas, "items": False},
            "same_actor": {"type": "string", "enum": list(_CONTINUITY)},
            "same_object": {"type": "string", "enum": list(_CONTINUITY)}}
        episode_schemas.append({"type": "object", "additionalProperties": False,
                               "properties": properties, "required": list(properties)})
    count = len(episodes)
    return {"type": "object", "additionalProperties": False,
        "properties": {"episodes": {"type": "array", "minItems": count, "maxItems": count,
                                    "prefixItems": episode_schemas, "items": False}}, "required": ["episodes"]}


def _parse_atoms(raw, episodes):
    document = strict_json(raw)
    if (not isinstance(document, dict) or set(document) != {"episodes"}
            or not isinstance(document["episodes"], list) or len(document["episodes"]) != len(episodes)):
        raise InvalidResponse("atom_schema")
    expected = {episode["id"]: episode["anchors"] for episode in episodes}
    result = {}
    for episode in document["episodes"]:
        if not isinstance(episode, dict) or set(episode) != {"id", "rows", "same_actor", "same_object"}:
            raise InvalidResponse("atom_schema")
        identity = episode["id"]
        if type(identity) is not int or identity not in expected or identity in result:
            raise InvalidResponse("atom_episode_identity")
        if any(not isinstance(episode[key], str) or episode[key] not in _CONTINUITY for key in ("same_actor", "same_object")):
            raise InvalidResponse("atom_continuity")
        unique = tuple(dict.fromkeys(expected[identity]))
        rows = episode["rows"]
        if not isinstance(rows, list) or len(rows) != len(unique):
            raise InvalidResponse("atom_anchor_coverage")
        observed = {}
        for row, index in zip(rows, unique, strict=True):
            if (not isinstance(row, list) or len(row) != 3 or type(row[0]) is not int or row[0] != index
                    or any(not isinstance(atom, str) or atom not in _ATOMS for atom in row[1:])):
                raise InvalidResponse("atom_values")
            observed[index] = frozenset(row[1:]) - {"unknown"}
        result[identity] = {"phases": tuple(observed[index] for index in expected[identity]),
                            "same_actor": episode["same_actor"], "same_object": episode["same_object"]}
    return result


def _supports(action, evidence):
    before, peak, after = evidence["phases"]
    if action == "food_or_drink_actually_consumed":
        # A complete nonordered consumption fact stands alone. Other mundane
        # objects before/after do not negate a directly observed peak bite/drink.
        return bool(peak & {"cashier_drinks_identifiable_cup_at_mouth", "cashier_takes_visible_bite_of_identifiable_food"})
    if evidence["same_actor"] != "yes" or evidence["same_object"] != "yes":
        return False
    if action == "cashier_places_removed_outerwear_on_cashdesk":
        contrasts = {"visitor_bag_handled", "visitor_holds_identifiable_outerwear", "papers_handled", "ambiguous_object"}
        return (not any(facts & contrasts for facts in (before, peak, after))
            and "cashier_holds_identifiable_outerwear" in before
            and bool(peak & {"cashier_places_outerwear_on_own_workdesk", "outerwear_resting_on_own_workdesk"})
            and "outerwear_resting_on_own_workdesk" in after)
    if action == "key_remains_in_safe_lock_after_departure":
        return (all("resolved_key_inserted_identified_safe_lock" in facts for facts in (before, peak, after))
            and not any(facts & {"highlight_or_handle", "ambiguous_object"} for facts in (before, peak, after))
            and "cashier_visibly_leaves_safe" in before
            and all("cashier_away_from_safe" in facts for facts in (peak, after)))
    if action == "stamp_or_book_remains_after_departure":
        return (all("identifiable_stamp_or_book_resting_on_own_workdesk" in facts for facts in (before, peak, after))
            and not any(facts & {"papers_handled", "ambiguous_object"} for facts in (before, peak, after))
            and "cashier_visibly_leaves_workdesk" in before
            and all("cashier_away_from_workdesk" in facts for facts in (peak, after)))
    if action == "banknotes_counted_after_visitor_departure":
        return ("visitor_visibly_leaves_service" in before
            and bool(before & {"cashier_holds_identifiable_banknotes", "identifiable_banknotes_rest_on_own_workdesk",
                               "cashier_counts_identifiable_banknotes", "cashier_processes_identifiable_banknotes"})
            and all({"cashier_counts_identifiable_banknotes", "visitor_away_from_service"} <= facts for facts in (peak, after))
            and not any(facts & {"papers_handled", "cashier_counts_banknotes_during_visible_service"} for facts in (peak, after)))
    if action == "earlier_banknotes_remain_during_next_service":
        return ("previous_service_visibly_complete" in before
            and all("identifiable_banknotes_rest_on_own_workdesk" in facts for facts in (before, peak, after))
            and all("different_visitor_actually_being_served" in facts for facts in (peak, after)))
    if action == "processed_banknotes_handed_back":
        return ("cashier_processes_identifiable_banknotes" in before
            and "cashier_hands_identifiable_banknotes_to_visitor" in peak
            and "distinct_visitor_holds_identifiable_banknotes" in after
            and not any(facts & {"papers_handled", "cashier_receives_banknotes_from_visitor"} for facts in (peak, after)))
    return False


class CompactVerifiedActionBackend(VerifiedActionBackend):
    """Two raw calls maximum per window, preserving the inherited owner budget.

    Only representation/parser/matcher change; reuse, source mapping, proposal
    cutoff, worker lifecycle, lock, diagnostics and original finding data are
    inherited. Legacy V3 remains the default unless the owner explicitly opts in.
    """
    def _verification_request(self, episodes, frames):
        prompt_episodes = [{"id": episode["id"], "phase_anchors": episode["anchors"],
                           "unique_anchors": list(dict.fromkeys(episode["anchors"]))} for episode in episodes]
        prompt = _INSTRUCTION + json.dumps({"fact_choices": list(_ATOMS), "continuity_choices": list(_CONTINUITY),
            "episodes": prompt_episodes, "frames_in_supplied_order": [
                {"frame": index, "timestamp_s": frame.timestamp_s} for index, frame in enumerate(frames, 1)]}, separators=(",", ":"))
        return InferenceRequest(prompt, frames, "checklist", None, (), _schema(episodes, len(frames)))

    def _verification_facts(self, raw, episodes):
        return _parse_atoms(raw, episodes)

    def _supports_proposal(self, action, facts):
        return _supports(action, facts)
