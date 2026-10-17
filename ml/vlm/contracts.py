"""Standard-library integration contracts; importing these never loads ML packages."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol, Sequence, TypedDict


class Candidate(TypedDict):
    candidate_id: str
    violation_type: str
    start_s: float
    end_s: float
    rule_id: str | None


class Rule(TypedDict):
    rule_id: str
    violation_type: str
    description: str
    citation: str
    verified: bool


@dataclass(frozen=True)
class FrameTimestamp:
    index: int
    timestamp_s: float
    pts_s: float


@dataclass(frozen=True)
class Frame:
    timestamp_s: float
    path: Path
    pts_s: float


@dataclass(frozen=True)
class VideoInfo:
    duration_s: float
    frames: tuple[FrameTimestamp, ...]


@dataclass(frozen=True)
class InferenceRequest:
    prompt: str
    frames: tuple[Frame, ...]
    mode: Literal["candidate", "checklist"]
    candidate_id: str | None
    allowed_rule_ids: tuple[str, ...]


class Backend(Protocol):
    def infer(self, request: InferenceRequest, timeout_s: float) -> str: ...


class FrameSource(Protocol):
    def probe(self, video_path: str | Path, timeout_s: float) -> VideoInfo: ...

    def extract(self, video_path: str | Path, selected: Sequence[FrameTimestamp],
                directory: Path, timeout_s: float) -> tuple[Frame, ...]: ...


@dataclass(frozen=True)
class VerificationResult:
    candidate: dict[str, Any]
    status: Literal["confirmed", "rejected", "uncertain"]
    reason: str
    confidence: float | None = None
    observations: tuple[Mapping[str, Any], ...] = ()
    rule_id: str | None = None
    citation: str | None = None
    additional: bool = False
    advisory_only: bool = True


@dataclass
class VerificationReport:
    results: list[VerificationResult] = field(default_factory=list)
    calls_used: int = 0
    human_review_required: bool = True
    checklist_status: str = "disabled"

    @property
    def retained_candidates(self) -> list[dict[str, Any]]:
        return [r.candidate for r in self.results if r.status != "rejected"]
