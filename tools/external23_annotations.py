"""Private annotation drafts and validation; no media reads or console output."""

from datetime import date
import math


PROVENANCE = {"source_human", "local_model", "reviewed_human"}
ANNOTATION_STATUSES = PROVENANCE | {"pending_review"}


def _require(condition, code):
    if not condition:
        raise ValueError(code)


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _number(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _iso_date(value):
    if not isinstance(value, str):
        return False
    try:
        return date.fromisoformat(value).isoformat() == value
    except ValueError:
        return False


def _record_map(records):
    _require(isinstance(records, (list, tuple)), "annotation_records_invalid")
    result = {}
    for record in records:
        _require(isinstance(record, dict), "annotation_records_invalid")
        clip_id = record.get("clip_id")
        _require(_text(clip_id), "annotation_record_id_invalid")
        _require(clip_id not in result, "annotation_record_id_duplicate")
        probe = record.get("ffprobe")
        _require(isinstance(probe, dict), "annotation_duration_invalid")
        duration = probe.get("duration_seconds")
        _require(_number(duration) and duration > 0, "annotation_duration_invalid")
        result[clip_id] = record
    return result


def draft_annotations(records):
    """Create empty pending drafts without inferring observations or a verdict."""
    clips = []
    for clip_id in _record_map(records):
        clips.append({
            "clip_id": clip_id,
            "annotation_status": "pending_review",
            "events": [],
            "legal_gold": None,
            "applicability": {
                "jurisdiction": None,
                "scene_type": None,
                "roles_established": False,
                "normative_conditions_established": False,
                "evidence_references": [],
            },
        })
    return {"schema_version": 1, "status": "pending_review", "clips": clips}


def _validate_event(event, record):
    duration = record["ffprobe"]["duration_seconds"]
    _require(isinstance(event, dict), "annotation_event_invalid")
    provenance = event.get("provenance")
    _require(isinstance(provenance, str) and provenance in PROVENANCE,
             "annotation_provenance_invalid")
    start, end = event.get("t_start"), event.get("t_end")
    _require(_number(start) and _number(end) and 0 <= start <= end <= duration,
             "annotation_clip_interval_invalid")
    _require(_text(event.get("action_label")), "annotation_action_invalid")
    _require(isinstance(event.get("roles"), list), "annotation_roles_invalid")
    for role in event["roles"]:
        _require(isinstance(role, dict) and _text(role.get("role"))
                 and role.get("status") in ("unknown", "source_asserted", "human_reviewed"),
                 "annotation_roles_invalid")
    if "observations" in event:
        _require(isinstance(event["observations"], str), "annotation_observations_invalid")
    origin = event.get("origin")
    _require(isinstance(origin, dict) and _text(origin.get("reference")),
             "annotation_origin_invalid")
    if provenance == "source_human":
        interval = origin.get("original_source_interval")
        _require(isinstance(interval, dict), "annotation_source_interval_invalid")
        source_start, source_end = interval.get("start"), interval.get("end")
        unit = interval.get("unit")
        _require(_number(source_start) and _number(source_end)
                 and 0 <= source_start <= source_end,
                 "annotation_source_interval_invalid")
        _require(unit in ("seconds", "frames"), "annotation_source_unit_invalid")
        fps = interval.get("fps")
        if unit == "frames" or fps is not None:
            _require(_number(fps) and fps > 0, "annotation_source_fps_invalid")
        if unit == "frames":
            source_start, source_end = source_start / fps, source_end / fps
        offsets = record.get("source_offsets")
        _require(isinstance(offsets, dict) and _number(offsets.get("start_sec"))
                 and offsets["start_sec"] >= 0, "annotation_source_offsets_invalid")
        expected_start = max(0, source_start - offsets["start_sec"])
        expected_end = min(duration, source_end - offsets["start_sec"])
        tolerance = max(0.1, 1 / fps) if fps is not None else 0.1
        _require(expected_start < expected_end and abs(start - expected_start) <= tolerance
                 and abs(end - expected_end) <= tolerance, "annotation_source_alignment_invalid")
    elif provenance == "local_model":
        _require(_text(origin.get("model_id")) and _text(origin.get("version")),
                 "annotation_model_origin_invalid")
    else:
        _require(_text(origin.get("reviewer")) and _iso_date(origin.get("reviewed_on")),
                 "annotation_review_origin_invalid")


def validate_annotations(document, records):
    """Raise ValueError with a fixed safe code; never print private values."""
    record_map = _record_map(records)
    _require(isinstance(document, dict), "annotation_document_invalid")
    _require(type(document.get("schema_version")) is int
             and document["schema_version"] == 1, "annotation_schema_invalid")
    status = document.get("status")
    _require(isinstance(status, str) and status in ANNOTATION_STATUSES,
             "annotation_status_invalid")
    clips = document.get("clips")
    _require(isinstance(clips, list), "annotation_clips_invalid")
    seen = set()
    for clip in clips:
        _require(isinstance(clip, dict), "annotation_clip_invalid")
        clip_id = clip.get("clip_id")
        _require(_text(clip_id) and clip_id in record_map, "annotation_clip_id_invalid")
        _require(clip_id not in seen, "annotation_clip_id_duplicate")
        seen.add(clip_id)
        status = clip.get("annotation_status")
        _require(isinstance(status, str) and status in ANNOTATION_STATUSES,
                 "annotation_status_invalid")
        if status == "reviewed_human":
            _require(_text(clip.get("reviewer")) and _iso_date(clip.get("reviewed_on")),
                     "annotation_review_origin_invalid")
        events = clip.get("events")
        _require(isinstance(events, list), "annotation_events_invalid")
        for event in events:
            _validate_event(event, record_map[clip_id])
            _require(status not in ("source_human", "local_model")
                     or event["provenance"] == status, "annotation_status_provenance_mismatch")
        applicability = clip.get("applicability")
        _require(isinstance(applicability, dict), "annotation_applicability_invalid")
        for key in ("jurisdiction", "scene_type"):
            _require(key in applicability and (applicability[key] is None
                     or _text(applicability[key])), "annotation_applicability_invalid")
        for key in ("roles_established", "normative_conditions_established"):
            _require(type(applicability.get(key)) is bool, "annotation_applicability_invalid")
        _require("legal_gold" in clip, "annotation_gold_missing")
        gold = clip["legal_gold"]
        if gold is not None:
            context = record_map[clip_id].get("source_context")
            refs = applicability.get("evidence_references")
            observed = (isinstance(context, dict) and context.get('scene_type') == 'bank_cashdesk')
            _require(status == "reviewed_human" and applicability["jurisdiction"] == "KZ"
                     and applicability["scene_type"] == "bank_cashdesk"
                     and applicability["roles_established"]
                     and applicability["normative_conditions_established"]
                     and observed
                     and isinstance(refs, list) and bool(refs) and all(_text(ref) for ref in refs),
                     "annotation_gold_review_required")
            _require(isinstance(gold, dict), "annotation_gold_invalid")
            verdict, types = gold.get("verdict"), gold.get("type_ids")
            _require(verdict in ("violation", "no_violation") and isinstance(types, list)
                     and all(_text(value) for value in types), "annotation_gold_invalid")
            _require((verdict == "violation") == bool(types), "annotation_gold_inconsistent")
    _require(seen == set(record_map), "annotation_clip_set_mismatch")
