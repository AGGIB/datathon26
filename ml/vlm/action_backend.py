"""Experimental physical-action contrast adapter; no model loading or media decode.

A normal/open-set choice is always available. The underlying model sees neutral
physical actions, never the owner's violation IDs, legal text or CV prompt. The
owner still decides whether an eligible action becomes an advisory candidate.
"""
from dataclasses import dataclass
import json
import time
from typing import Mapping

from .contracts import Backend, Frame, InferenceRequest
from .discovery import DiscoveryWindow, MAX_WINDOW_FRAMES, _catalog_types, _parse
from .validation import InvalidResponse, number, strict_json

DECODER_POLICY = "physical_action_search_v4"


@dataclass(frozen=True)
class _Action:
    type_id: str
    criterion: str
    phases: tuple[str, str, str]
    ordered: bool = True


_ACTIONS = {
    "processed_banknotes_handed_back": _Action("banknote_returned_after_check",
        "Banknotes are visibly checked/processed, then those banknotes are actually handed back to the visitor; cash receipt or ordinary handover alone does not qualify.",
        ("Кассир проверяет или обрабатывает банкноты", "Кассир передаёт эти банкноты обратно посетителю", "Передача банкнот посетителю завершается")),
    "earlier_banknotes_remain_during_next_service": _Action("previous_cash_left_on_desk",
        "The same banknotes from a visibly completed service remain on the desk when a next visitor is actually served; a stack or two people alone does not qualify.",
        ("Банкноты остаются на столе после обслуживания посетителя", "Начинается обслуживание следующего посетителя, прежние банкноты остаются на столе", "Обслуживание продолжается, прежние банкноты остаются на столе")),
    "banknotes_counted_after_visitor_departure": _Action("recount_after_client_left",
        "A visitor visibly departs; afterwards the cashier actually counts that service's banknotes. Counting during service or handling papers does not qualify.",
        ("Посетитель уходит после обслуживания", "После ухода посетителя кассир пересчитывает банкноты", "Пересчёт банкнот после ухода посетителя продолжается")),
    "cashier_places_removed_outerwear_on_cashdesk": _Action("personal_items_on_desk",
        "The person acting as cashier visibly holds an identifiable outer garment already off the body, then places that garment on their own cash desk. The act of taking it off need not be visible; do not claim hidden removal. A visitor/courier bag, bag handover or ambiguous fabric does not qualify.",
        ("Кассир держит снятую верхнюю одежду", "Кассир кладёт снятую верхнюю одежду на кассовый стол", "Снятая верхняя одежда остаётся на кассовом столе")),
    "food_or_drink_actually_consumed": _Action("food_or_drink_at_workplace",
        "The cashier visibly takes a bite or drinks with an identifiable cup at the mouth. A standing cup/container, face touching or ambiguous object does not qualify.",
        ("Кассир подносит еду или напиток ко рту", "Кассир ест или пьёт на рабочем месте", "Кассир завершает видимый приём еды или напитка"), False),
    "key_remains_in_safe_lock_after_departure": _Action("key_left_in_safe_lock",
        "A cashier visibly departs and a clearly resolved key remains inserted in an identified safe lock. A tiny highlight/handle, inferred key or safe use alone does not qualify.",
        ("Кассир отходит от сейфа", "После ухода кассира ключ остаётся в замке сейфа", "Ключ продолжает оставаться в замке сейфа")),
    "stamp_or_book_remains_after_departure": _Action("stamps_and_book_left_on_desk",
        "A clearly identifiable stamp or book remains on the cash desk after visible cashier departure. Loose papers, cash or an ambiguous rectangle do not qualify.",
        ("Кассир уходит от кассового стола", "После ухода кассира на столе остаётся видимая печать или книга", "Печать или книга продолжает оставаться на кассовом столе")),
}
_NORMAL = {
    "visitor_or_courier_bag_handled": "A visitor/courier bag is handed over, handled, or rests on the desk; this is not the cashier placing an identifiable outer garment on their desk.",
    "cash_received": "Banknotes are received from a visitor, without a clearly observed checking-and-return sequence.",
    "identity_document_presented": "A visitor presents an identity document or card; do not infer an omitted identity check.",
    "banknotes_counted_during_service": "Banknotes are counted while the visitor remains or the departure order is not established.",
    "papers_handled": "Forms or papers are handled, with no clearly identified banknote-counting action.",
    "cash_stored": "Cash is put away; do not infer an absent check or recount from sparse frames.",
    "idle_or_other_action": "No listed target action, ordinary work, or a different action is visible.",
    "unclear": "The action, object, ownership, actor or temporal order cannot be established from the supplied frames.",
}
_INSTRUCTION = (
    "Inspect ALL supplied frames, including the last frames, before selecting an action. Search the entire "
    "clip for a fully supported target action anywhere, then choose its most conclusive observed consequence. "
    "A target can occur after ordinary work or idle frames; do not choose only the earliest action. "
    "If no target is fully supported anywhere, prefer at most one best ordinary action or unclear; [] is valid. "
    "Do not fill two slots with trivial early idle actions. Never invent a target or force the closest match. "
    "Sparse frames cannot prove an omitted check or hidden step. No identity, intent, "
    "authenticity, access-rights or legal conclusions. Distinguish a visitor's bag from the cashier's worn coat, "
    "papers from banknotes, and a counting machine from an authenticity detector. "
    "Choose the action before anchoring its frames. Return JSON only in this field order: {\"actions\":[{"
    "\"action\":string,\"insufficient_evidence\":boolean,\"confidence\":number,"
    "\"start_frame\":integer,\"peak_frame\":integer,\"end_frame\":integer}]}. At most 2 actions. "
    "Use supplied 1-based frame indices, start<=peak<=end. A target sequence needs distinct prerequisite "
    "and consequence frames. peak_frame is the most conclusive observed consequence, not the earliest "
    "prerequisite. Mark insufficient_evidence=true if any required action/object/order is unclear. "
)


def _schema(actions, count):
    index = {"type": "integer", "enum": list(range(1, count + 1))}
    properties = {"action": {"type": "string", "enum": list(actions)},
                  "insufficient_evidence": {"type": "boolean"},
                  "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                  "start_frame": index, "peak_frame": index, "end_frame": index}
    return {"type": "object", "additionalProperties": False, "required": ["actions"],
            "properties": {"actions": {"type": "array", "maxItems": 2, "items": {
                "type": "object", "additionalProperties": False, "properties": properties,
                "required": list(properties)}}}}


def _validate(raw, allowed, count):
    value = strict_json(raw)
    if not isinstance(value, dict) or set(value) != {"actions"} or not isinstance(value["actions"], list) or len(value["actions"]) > 2:
        raise InvalidResponse("action_schema")
    keys = {"start_frame", "peak_frame", "end_frame", "action", "confidence", "insufficient_evidence"}
    for action in value["actions"]:
        if not isinstance(action, dict) or set(action) != keys:
            raise InvalidResponse("action_schema")
        indices = [action[key] for key in ("start_frame", "peak_frame", "end_frame")]
        if (any(type(index) is not int or not 1 <= index <= count for index in indices)
                or indices != sorted(indices)):
            raise InvalidResponse("action_indices")
        if not isinstance(action["action"], str) or action["action"] not in allowed:
            raise InvalidResponse("action_unknown")
        if (not number(action["confidence"]) or not 0 <= action["confidence"] <= 1
                or type(action["insufficient_evidence"]) is not bool):
            raise InvalidResponse("action_confidence")
        if (action["action"] in _ACTIONS and _ACTIONS[action["action"]].ordered
                and indices[0] >= indices[1] and not action["insufficient_evidence"]):
            raise InvalidResponse("action_order_unobserved")
    return value["actions"]


class ActionContrastBackend:
    """One bounded inner call, translating only supported positive physical acts.

    `catalog` is provided by the owner; only its verified/enabled allowlist can
    become a discovery finding. The inner backend owns hard timeout cancellation.
    `last_actions` contains sanitized enums/indices/numbers only, never model text.
    """
    def __init__(self, inner: Backend, *, catalog: Mapping):
        self.inner = inner
        eligible, _ = _catalog_types(catalog, tuple(action.type_id for action in _ACTIONS.values()))
        self._eligible = frozenset(eligible)
        self.last_actions = ()
        self.calls_used = 0

    def validate(self, timeout_s):
        validate = getattr(self.inner, "validate", None)
        if validate is not None:
            return validate(timeout_s=timeout_s)

    def close(self):
        close = getattr(self.inner, "close", None)
        if close is not None:
            return close()

    def infer(self, request: InferenceRequest, timeout_s: float) -> str:
        self.last_actions = ()
        if not number(timeout_s) or timeout_s <= 0:
            raise TimeoutError("action_timeout")
        deadline = time.monotonic() + timeout_s
        frames = request.frames
        if (not 1 <= len(frames) <= MAX_WINDOW_FRAMES or any(not isinstance(frame, Frame) or not number(frame.timestamp_s)
                or frame.timestamp_s < 0 for frame in frames)):
            raise InvalidResponse("action_frames")
        times = tuple(frame.timestamp_s for frame in frames)
        if times != tuple(sorted(set(times))):
            raise InvalidResponse("action_frames")
        positive = {name: action for name, action in _ACTIONS.items()
                    if action.type_id in self._eligible and action.type_id in request.allowed_rule_ids}
        choices = {**{name: action.criterion for name, action in positive.items()}, **_NORMAL}
        prompt = _INSTRUCTION + json.dumps({
            "action_choices": [{"action": name, "visible_criterion": criterion} for name, criterion in choices.items()],
            "frames_in_supplied_order": [{"frame_index": index, "timestamp_s": frame.timestamp_s}
                                         for index, frame in enumerate(frames, 1)]}, separators=(",", ":"))
        translated = InferenceRequest(prompt, frames, "checklist", None, (), _schema(choices, len(frames)))
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("action_timeout")
        self.calls_used += 1
        raw = self.inner.infer(translated, timeout_s=remaining)
        if time.monotonic() >= deadline:
            raise TimeoutError("action_timeout")
        actions = _validate(raw, choices, len(frames))
        self.last_actions = tuple(dict(action) for action in actions)
        findings = []
        for action in actions:
            specification = positive.get(action["action"])
            if specification is None or action["insufficient_evidence"]:
                continue
            start, peak, end = (times[action[key] - 1] for key in ("start_frame", "peak_frame", "end_frame"))
            # Consequence wins when an anchor serves multiple phases (e.g. peak=end).
            phases = {end: specification.phases[2], start: specification.phases[0], peak: specification.phases[1]}
            findings.append({"observations": [{"timestamp_s": t, "text": phases[t]} for t in sorted(phases)],
                             "type_id": specification.type_id, "t_start": start, "t_peak": peak, "t_end": end,
                             "confidence": action["confidence"], "insufficient_evidence": False})
        result = json.dumps({"findings": findings}, ensure_ascii=False, allow_nan=False)
        _parse(result, self._eligible & set(request.allowed_rule_ids), frames, DiscoveryWindow(times[0], times[-1], times))
        if time.monotonic() >= deadline:
            raise TimeoutError("action_timeout")
        return result
