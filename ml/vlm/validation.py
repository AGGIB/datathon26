"""Strict, fail-closed model output parser. No JSON repair or markdown extraction."""
import json
import math
import re
from typing import Any, Sequence
from .contracts import Frame


class InvalidResponse(ValueError):
    pass


def number(value: Any) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InvalidResponse("duplicate_key")
        result[key] = value
    return result


def _constant(value):
    raise InvalidResponse("nonfinite")


def strict_json(raw: str) -> Any:
    if not isinstance(raw, str) or len(raw) > 65536:
        raise InvalidResponse("invalid_json")
    try:
        return json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_constant)
    except (ValueError, TypeError, RecursionError) as exc:
        raise InvalidResponse("invalid_json") from exc


def validate_finding(value: Any, *, candidate_id: str | None, allowed_rule_ids: Sequence[str],
                     frames: Sequence[Frame], start_s: float, end_s: float) -> dict:
    keys = {"rule_id", "violation", "confidence", "insufficient_evidence", "observations"}
    if candidate_id is not None:
        keys.add("candidate_id")
    if not isinstance(value, dict) or set(value) != keys:
        raise InvalidResponse("invalid_schema")
    if candidate_id is not None and value["candidate_id"] != candidate_id:
        raise InvalidResponse("candidate_id")
    if not isinstance(value["rule_id"], str) or value["rule_id"] not in allowed_rule_ids:
        raise InvalidResponse("rule_id")
    if type(value["violation"]) is not bool or type(value["insufficient_evidence"]) is not bool:
        raise InvalidResponse("boolean")
    if not number(value["confidence"]) or not 0 <= value["confidence"] <= 1:
        raise InvalidResponse("confidence")
    observations = value["observations"]
    if not isinstance(observations, list) or not 1 <= len(observations) <= 12:
        raise InvalidResponse("observations")
    for observation in observations:
        if not isinstance(observation, dict) or set(observation) != {"timestamp_s", "text"}:
            raise InvalidResponse("observation_schema")
        timestamp = observation["timestamp_s"]
        text = observation["text"]
        if not number(timestamp) or not start_s <= timestamp <= end_s:
            raise InvalidResponse("timestamp_bounds")
        if not any(abs(timestamp - frame.timestamp_s) <= 1e-6 for frame in frames):
            raise InvalidResponse("timestamp_not_supplied")
        if not isinstance(text, str) or not text.strip() or len(text) > 1000:
            raise InvalidResponse("observation_text")
        # Conservative lexical guard, not semantic/legal validation. False positives
        # are uncertain, never rejection; authoritative citation is attached outside.
        if re.search(r"§|\b(?:article|law|statute|кодекс\w*|закон\w*|стать[яеию])\b|\b(?:art|ст)\.\s*\d", text, re.I):
            raise InvalidResponse("legal_prose_not_visible_action")
    return value
