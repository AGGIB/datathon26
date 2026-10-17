"""К1 — response contract. Required fields follow the ТЗ schema exactly;
fields marked "extension" are ours and optional (UI uses them, the jury may ignore them)."""

import base64
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Verdict = Literal["violation", "no_violation"]
JobStatus = Literal["processing", "done", "failed"]


class EvidenceFrame(BaseModel):
    model_config = ConfigDict(frozen=True)

    timestamp_sec: float = Field(ge=0)
    image_base64: str = Field(min_length=1, description="JPEG, base64, no data: prefix")

    @field_validator("image_base64")
    @classmethod
    def _validate_jpeg_base64(cls, v: str) -> str:
        if v.startswith("data:"):
            raise ValueError("image_base64 must not have data: URI prefix")
        try:
            raw = base64.b64decode(v, validate=True)
        except Exception as exc:
            raise ValueError("image_base64 must be valid base64 data") from exc
        if not raw.startswith(b"\xff\xd8"):
            raise ValueError("image_base64 must be JPEG format (missing 0xFFD8 magic header)")
        return v


class RuleRef(BaseModel):
    """Extension: normative source. Only verified quotes go here (owner: Олжас)."""

    model_config = ConfigDict(frozen=True)

    document: str
    clause: str | None = None
    quote: str | None = None
    url: str | None = None


class Violation(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: str = Field(min_length=1)
    timecode_sec: float = Field(ge=0)
    description: str = Field(min_length=1)
    evidence_frames: list[EvidenceFrame] = Field(min_length=1, max_length=3)

    # extensions
    type_id: str | None = None
    t_start: float | None = Field(default=None, ge=0)
    t_end: float | None = Field(default=None, ge=0)
    confidence: float | None = Field(default=None, ge=0, le=1)
    rule: RuleRef | None = None


class AnalysisResult(BaseModel):
    """What the pipeline returns (К2) and what the jury reads when status == done."""

    model_config = ConfigDict(frozen=True)

    verdict: Verdict
    violations: list[Violation]

    @model_validator(mode="after")
    def _verdict_matches_violations(self) -> "AnalysisResult":
        if self.verdict == "violation" and not self.violations:
            raise ValueError("verdict=violation requires at least one violation")
        if self.verdict == "no_violation" and self.violations:
            raise ValueError("verdict=no_violation requires an empty violations list")
        return self


class JobAccepted(BaseModel):
    job_id: str
    status: JobStatus


class JobView(BaseModel):
    job_id: str
    status: JobStatus
    verdict: Verdict | None = None
    violations: list[Violation] | None = None
    # extensions for the UI while processing / on failure
    progress: float | None = None
    stage: str | None = None
    error: str | None = None
