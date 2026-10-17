"""Shared interface between rules (who decides) and assembly (who renders the К1 answer)."""

from dataclasses import dataclass

from .detect import Box


@dataclass(frozen=True)
class Candidate:
    """A suspected violation found by rules, before evidence frames and normative text are attached.

    type_id          — id from kb/violations.yaml
    t_start/t_end    — event interval, seconds from video start
    t_peak           — the most telling moment; becomes timecode_sec
    confidence       — 0..1, used for thresholding and ranking
    observation      — Russian, only what is visible ("кассир выходит, пачка остаётся на столе")
    evidence_times   — 1–3 timestamps to extract evidence frames from; the first one MUST be t_peak
    highlight_boxes  — optional boxes (source-frame pixels) valid at t_peak; drawn on the first evidence frame only
    """

    type_id: str
    t_start: float
    t_end: float
    t_peak: float
    confidence: float
    observation: str
    evidence_times: tuple[float, ...]
    highlight_boxes: tuple[Box, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be within 0..1")
        if not self.t_start <= self.t_peak <= self.t_end:
            raise ValueError("t_peak must lie within [t_start, t_end]")
        if not 1 <= len(self.evidence_times) <= 3:
            raise ValueError("evidence_times must have 1..3 timestamps")
        if self.evidence_times[0] != self.t_peak:
            raise ValueError("the first evidence time must be t_peak")
        if not any(ch.isalpha() for ch in self.observation):  # "", "...", "—" would render as «На 1,5 с .»
            raise ValueError("observation must describe what is visible in words")
