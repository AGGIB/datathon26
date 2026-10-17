"""Stage 3: rule candidates -> К1 answer {verdict, violations[]}.

Pure rendering: which events are suspicious is decided by ml/rules.py; here they are filtered,
de-duplicated, ranked and turned into the jury-facing text with evidence frames and the norm.
"""

import dataclasses
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from .candidates import Candidate
from .kb import ViolationType, catalog

log = logging.getLogger(__name__)

MIN_CONFIDENCE = 0.5  # candidates below this never reach the answer
MAX_VIOLATIONS = 5  # most confident ones are kept, then shown in time order
MERGE_GAP_SEC = 1.0  # same-type candidates closer than this are one episode
MAX_EVIDENCE_FRAMES = 3
DRAFT_BASIS = "нормативное основание уточняется"
# Types without a verified norm are reported without `rule` and with DRAFT_BASIS (lead's decision);
# set to False to follow the stricter kb/violations.yaml note and drop them instead.
EMIT_UNVERIFIED_TYPES = True

RenderFrames = Callable[[Candidate], list[dict]]


def _known(candidates: Iterable[Candidate], types: Mapping[str, ViolationType]) -> list[Candidate]:
    known = []
    for c in candidates:
        vtype = types.get(c.type_id)
        if vtype is None:
            log.warning("assemble: unknown type_id %r from rules (t=%.1fs) dropped", c.type_id, c.t_peak)
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


def _ru_seconds(t: float) -> str:
    return f"{max(0.0, t):.1f}".replace(".", ",")


def _lower_first(text: str) -> str:
    """Sentence start -> mid-sentence, but keep abbreviations like «ПК» intact."""
    if len(text) > 1 and text[1].isupper():
        return text
    return text[:1].lower() + text[1:]


def _clause(text: str) -> str:
    return _lower_first(text.strip().rstrip(".").strip())


def describe(candidate: Candidate, vtype: ViolationType) -> str:
    """Russian, observable facts only; the last sentence always names the basis."""
    observation = _clause(candidate.observation) or _clause(vtype.summary)
    basis = vtype.rule.basis if vtype.rule else DRAFT_BASIS
    return (
        f"На {_ru_seconds(candidate.t_peak)} с {observation}. "
        f"Это может указывать на нарушение: {_clause(vtype.summary)}. "
        f"Основание: {basis}."
    )


def build_violation(candidate: Candidate, vtype: ViolationType, frames: Sequence[dict]) -> dict[str, Any]:
    violation: dict[str, Any] = {
        "type": vtype.name,
        "type_id": vtype.id,
        "timecode_sec": round(max(0.0, candidate.t_peak), 1),
        "t_start": round(max(0.0, candidate.t_start), 1),
        "t_end": round(max(0.0, candidate.t_end), 1),
        "confidence": round(candidate.confidence, 3),
        "description": describe(candidate, vtype),
        "evidence_frames": list(frames[:MAX_EVIDENCE_FRAMES]),
    }
    if vtype.rule is not None:
        violation["rule"] = vtype.rule.as_ref()
    return violation


def assemble(
    candidates: Iterable[Candidate],
    render_frames: RenderFrames,
    types: Mapping[str, ViolationType] | None = None,
) -> dict[str, Any]:
    """К1 body without job_id/status. verdict is derived from the list, so the invariant always holds."""
    types = catalog() if types is None else types
    violations = []
    for c in select(candidates, types):
        frames = render_frames(c)
        if not frames:
            log.warning("assemble: no evidence frames for %s at %.1fs, dropped", c.type_id, c.t_peak)
            continue
        violations.append(build_violation(c, types[c.type_id], frames))
    return {"verdict": "violation" if violations else "no_violation", "violations": violations}
