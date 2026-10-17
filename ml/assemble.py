"""Stage 3: rule candidates -> К1 answer {verdict, violations[]}.

Pure rendering: which events are suspicious is decided by ml/rules.py; here they are filtered,
de-duplicated, ranked and turned into the jury-facing text with evidence frames and the norm.
The wording comes from the kb template of the type (see ml/kb.py); code only fills the placeholders.
"""

import dataclasses
import logging
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from .candidates import Candidate
from .kb import ViolationType, catalog

log = logging.getLogger(__name__)

MIN_CONFIDENCE = 0.5  # candidates below this never reach the answer
MAX_VIOLATIONS = 5  # most confident ones are kept, then shown in time order
MERGE_GAP_SEC = 1.0  # same-type candidates closer than this are one episode
MAX_EVIDENCE_FRAMES = 3
TIME_SCALE = 10  # times are reported with 0.1 s precision
# Types without a verified norm are reported without `rule`; their template says the basis is being
# clarified (lead's decision). Set to False to follow the stricter kb/violations.yaml note and drop them.
EMIT_UNVERIFIED_TYPES = True

# Gets all selected candidates at once (one decode pass for every evidence frame), returns their frames.
RenderFrames = Callable[[Sequence[Candidate]], Mapping[Candidate, list[dict]]]


def _known(candidates: Iterable[Candidate], types: Mapping[str, ViolationType]) -> list[Candidate]:
    known = []
    for c in candidates:
        vtype = types.get(c.type_id)
        if vtype is None:
            log.warning("assemble: unknown type_id %r from rules (t=%.1fs) dropped", c.type_id, c.t_peak)
        elif not vtype.enabled:
            log.info("assemble: %s is disabled in kb, dropped (t=%.1fs)", c.type_id, c.t_peak)
        elif vtype.rule is None and not EMIT_UNVERIFIED_TYPES:
            log.info("assemble: %s has no verified norm, dropped (t=%.1fs)", c.type_id, c.t_peak)
        else:
            known.append(c)
    return known


def _merge_pair(a: Candidate, b: Candidate) -> Candidate:
    stronger = a if a.confidence >= b.confidence else b
    return dataclasses.replace(stronger, t_start=min(a.t_start, b.t_start), t_end=max(a.t_end, b.t_end))


def merge_overlapping(candidates: Sequence[Candidate], gap_sec: float = MERGE_GAP_SEC) -> list[Candidate]:
    """One candidate per episode and type: the strongest one, stretched over the whole episode."""
    merged: list[Candidate] = []
    for type_id in sorted({c.type_id for c in candidates}):
        current: Candidate | None = None
        for c in sorted((c for c in candidates if c.type_id == type_id), key=lambda c: c.t_start):
            if current is not None and c.t_start <= current.t_end + gap_sec:
                current = _merge_pair(current, c)
                continue
            if current is not None:
                merged.append(current)
            current = c
        if current is not None:
            merged.append(current)
    return merged


def select(candidates: Iterable[Candidate], types: Mapping[str, ViolationType]) -> list[Candidate]:
    confident = [c for c in _known(candidates, types) if c.confidence >= MIN_CONFIDENCE]
    ranked = sorted(merge_overlapping(confident), key=lambda c: c.confidence, reverse=True)
    if len(ranked) > MAX_VIOLATIONS:
        log.info("assemble: %d candidates, keeping the %d most confident", len(ranked), MAX_VIOLATIONS)
    return sorted(ranked[:MAX_VIOLATIONS], key=lambda c: c.t_peak)


def clamp_time(t: float, duration: float | None = None) -> float:
    """Rounded to 0.1 s without leaving [0, duration] (plain rounding can step past the end)."""
    value = round(max(0.0, t), 1)
    if duration is not None and value > duration:
        value = math.floor(max(0.0, duration) * TIME_SCALE) / TIME_SCALE
    return value


def _ru_seconds(t: float) -> str:
    return f"{t:.1f}".replace(".", ",")


def _observation_for(template: str, observation: str) -> str:
    """Observation as a clause: capitalised at a sentence start, lower-case mid-sentence (keeps «ПК»)."""
    text = observation.strip().rstrip(".").strip()
    head = template.split("{observation}", 1)[0].rstrip()
    if not head or head.endswith((".", "!", "?")):
        return text[:1].upper() + text[1:]
    if len(text) > 1 and text[1].isupper():
        return text
    return text[:1].lower() + text[1:]


def describe(candidate: Candidate, vtype: ViolationType, duration: float | None = None) -> str:
    """kb template of the type with the time, the observation and the verified norm filled in."""
    rule = vtype.rule
    return vtype.template.format(
        t=_ru_seconds(clamp_time(candidate.t_peak, duration)),
        observation=_observation_for(vtype.template, candidate.observation),
        rule_document=rule.document if rule else "",
        rule_clause=(rule.clause or "") if rule else "",
        rule_number_date=(rule.number_date or "") if rule else "",
    )


def build_violation(
    candidate: Candidate, vtype: ViolationType, frames: Sequence[dict], duration: float | None = None
) -> dict[str, Any]:
    violation: dict[str, Any] = {
        "type": vtype.name,
        "type_id": vtype.id,
        "timecode_sec": clamp_time(candidate.t_peak, duration),
        "t_start": clamp_time(candidate.t_start, duration),
        "t_end": clamp_time(candidate.t_end, duration),
        "confidence": round(candidate.confidence, 3),
        "description": describe(candidate, vtype, duration),
        "evidence_frames": list(frames[:MAX_EVIDENCE_FRAMES]),
    }
    if vtype.rule is not None:
        violation["rule"] = vtype.rule.as_ref()
    return violation


def assemble(
    candidates: Iterable[Candidate],
    render_frames: RenderFrames,
    types: Mapping[str, ViolationType] | None = None,
    duration: float | None = None,
) -> dict[str, Any]:
    """К1 body without job_id/status. verdict is derived from the list, so the invariant always holds.

    duration — length of the analysed part of the video; all reported times are clamped to it.
    """
    types = catalog() if types is None else types
    selected = select(candidates, types)
    frames_by_candidate = render_frames(selected) if selected else {}
    violations = []
    for c in selected:
        frames = frames_by_candidate.get(c) or []
        if not frames:
            log.warning("assemble: no evidence frames for %s at %.1fs, dropped", c.type_id, c.t_peak)
            continue
        violations.append(build_violation(c, types[c.type_id], frames, duration))
    return {"verdict": "violation" if violations else "no_violation", "violations": violations}
