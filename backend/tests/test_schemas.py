import pytest
from pydantic import ValidationError

from backend.app.schemas import AnalysisResult

DUMMY_JPEG_B64 = (
    "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP/////////////////////////////////////////////////////////////////"
    "/////////////////////wgALCAABAAEBAREA/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPxA="
)
FRAME = {"timestamp_sec": 1.0, "image_base64": DUMMY_JPEG_B64}


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


def test_evidence_frames_rejects_data_uri_prefix():
    with pytest.raises(ValidationError):
        AnalysisResult.model_validate(
            {"verdict": "violation", "violations": [_violation(evidence_frames=[{"timestamp_sec": 1.0, "image_base64": f"data:image/jpeg;base64,{DUMMY_JPEG_B64}"}])]}
        )


def test_evidence_frames_rejects_non_base64():
    with pytest.raises(ValidationError):
        AnalysisResult.model_validate(
            {"verdict": "violation", "violations": [_violation(evidence_frames=[{"timestamp_sec": 1.0, "image_base64": "not a base64 string!"}])]}
        )


def test_evidence_frames_rejects_non_jpeg():
    # "AAAA" is valid base64 (decodes to b'\x00\x00\x00'), but not JPEG
    with pytest.raises(ValidationError):
        AnalysisResult.model_validate(
            {"verdict": "violation", "violations": [_violation(evidence_frames=[{"timestamp_sec": 1.0, "image_base64": "AAAA"}])]}
        )
