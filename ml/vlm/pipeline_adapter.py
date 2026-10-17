"""Optional bridge for the team's Candidate/ViolationType objects, without CV imports.

The pipeline owner calls this once for the complete candidate set of one video.
This module neither edits K1 results nor adds open-checklist CV candidates.
"""
from dataclasses import dataclass, replace
import time
from pathlib import Path
from typing import Mapping, Protocol, Sequence, TypeVar, Generic

from .contracts import Backend, FrameSource, VerificationReport
from .verifier import remaining_budget, verify_candidates


class PipelineCandidate(Protocol):
    @property
    def type_id(self) -> str: ...

    @property
    def t_start(self) -> float: ...

    @property
    def t_end(self) -> float: ...


class CatalogRule(Protocol):
    @property
    def document(self) -> str: ...


class CatalogType(Protocol):
    @property
    def id(self) -> str: ...

    @property
    def summary(self) -> str: ...

    @property
    def status(self) -> str: ...

    @property
    def rule(self) -> CatalogRule | None: ...


T = TypeVar("T", bound=PipelineCandidate)


@dataclass(frozen=True)
class PipelineVerificationReport(Generic[T]):
    verification: VerificationReport
    retained_candidates: tuple[T, ...]


def verify_pipeline_candidates(
    video_path: str | Path,
    candidates: Sequence[T],
    catalog: Mapping[str, CatalogType],
    *,
    model_path: str | Path | None = None,
    backend: Backend | None = None,
    frame_source: FrameSource | None = None,
    max_calls: int = 6,
    timeout_s: float = 60,
    budget_sec: float | None = None,
    video_timeout_s: float | None = None,
    device: str = "cpu",
    timebase_verified: bool = False,
    allow_rejection: bool = False,
    _deadline: float | None = None,
) -> PipelineVerificationReport[T]:
    """Shadow verification, retaining original CV objects by default.

    `catalog` must come from the owner's validated `ml.kb` loader. Its status is
    citation verification only, not proof of a visible violation. The type ID is
    also the VLM routing rule_id (not an invented legal clause). IDs are literal;
    per-call cv-N identifiers disambiguate multiple candidates of the same type.
    Missing norms, including a draft T04, stay uncertain and remain in assembly.
    Set timebase_verified only after checking CV times against container PTS:
    frame-index / FPS timestamps are not generally equivalent (especially VFR).
    Otherwise no model is loaded/called and eligible candidates are uncertain.
    Shadow verification always retains the original CV objects, including a
    validated advisory rejection. Legacy allow_rejection never permits filtering
    or changing CV confidence; independent calibration has not authorized it.
    """
    budget = remaining_budget(budget_sec, video_timeout_s)
    deadline = time.monotonic() + budget
    if _deadline is not None:
        deadline = min(deadline, _deadline)
    if type(timebase_verified) is not bool or type(allow_rejection) is not bool:
        raise ValueError("timebase_verified and allow_rejection must be booleans")
    originals = tuple(candidates)
    rules = []
    for type_id, entry in catalog.items():
        if (entry.id == type_id and entry.status == "verified" and entry.rule is not None
                and isinstance(entry.rule.document, str) and entry.rule.document.strip()):
            # New owner RuleText has source fields, not legacy basis. An explicitly
            # empty legacy basis stays invalid; only absence enables the fallback.
            citation = getattr(entry.rule, "basis", ", ".join(
                field for field in (entry.rule.document, getattr(entry.rule, "number_date", None),
                                    getattr(entry.rule, "clause", None))
                if isinstance(field, str) and field.strip()))
            rules.append({"rule_id": type_id, "violation_type": type_id,
                          "description": entry.summary, "citation": citation,
                          "verified": True})
    converted = [
        {"candidate_id": f"cv-{index}", "violation_type": candidate.type_id,
         "start_s": candidate.t_start, "end_s": candidate.t_end,
         "rule_id": candidate.type_id if isinstance(candidate.type_id, str)
         and candidate.type_id in catalog else None}
        for index, candidate in enumerate(originals, 1)
    ]
    verification = verify_candidates(
        video_path, converted, rules, model_path=model_path if timebase_verified else None,
        backend=backend if timebase_verified else None,
        frame_source=frame_source, max_calls=max_calls, enable_open_checklist=False,
        timeout_s=timeout_s, budget_sec=budget_sec, video_timeout_s=video_timeout_s, device=device,
        _deadline=deadline,
    )
    if not timebase_verified:
        verification.results = [replace(result, reason="unverified_timebase")
                                if result.reason == "model_unavailable" else result
                                for result in verification.results]
    # Legacy allow_rejection is accepted but never grants deletion authority.
    return PipelineVerificationReport(verification, originals)
