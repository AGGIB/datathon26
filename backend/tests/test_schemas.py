import pytest
from pydantic import ValidationError

from backend.app.schemas import AnalysisResult

FRAME = {"timestamp_sec": 1.0, "image_base64": "AAAA"}


def _violation(**overrides):
    base = {
        "type": "Тип",
        "timecode_sec": 1.0,
        "description": "Что видно",
        "evidence_frames": [FRAME],
    }
    return {**base, **overrides}


def test_violation_verdict_requires_violations():
    with pytest.raises(ValidationError):
        AnalysisResult.model_validate({"verdict": "violation", "violations": []})


def test_no_violation_verdict_must_have_empty_list():
    with pytest.raises(ValidationError):
        AnalysisResult.model_validate({"verdict": "no_violation", "violations": [_violation()]})


def test_evidence_frames_limited_to_three():
    with pytest.raises(ValidationError):
        AnalysisResult.model_validate(
            {"verdict": "violation", "violations": [_violation(evidence_frames=[FRAME] * 4)]}
        )


def test_evidence_frames_required():
    with pytest.raises(ValidationError):
        AnalysisResult.model_validate(
            {"verdict": "violation", "violations": [_violation(evidence_frames=[])]}
        )


def test_negative_timecode_rejected():
    with pytest.raises(ValidationError):
        AnalysisResult.model_validate(
            {"verdict": "violation", "violations": [_violation(timecode_sec=-1)]}
        )


def test_extension_fields_are_optional():
    result = AnalysisResult.model_validate({"verdict": "violation", "violations": [_violation()]})
    assert result.violations[0].rule is None
