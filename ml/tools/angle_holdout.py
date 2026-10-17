"""Prepare a local angle holdout without treating development footage as unseen.

No media decoding or inference. Manifests and detailed reports are private local artifacts.
"""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
from urllib.parse import urlsplit

from ml import eval as evaluation
from ml.evaluation_labels import file_sha256, load_labels, source_index, validate_content_sha256, video_key

UNKNOWN = {"", "unknown", "unassigned", "неизвестно"}
MEDIA_SUFFIXES = {".mov", ".mp4", ".m4v", ".mkv", ".avi", ".webm"}
ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_LICENSES = {
    "CC0-1.0": "https://creativecommons.org/publicdomain/zero/1.0/",
    "CC-BY-4.0": "https://creativecommons.org/licenses/by/4.0/",
    "CC-BY-3.0": "https://creativecommons.org/licenses/by/3.0/",
    "PDM-1.0": "https://creativecommons.org/publicdomain/mark/1.0/",
}


def _known(value):
    return isinstance(value, str) and value == value.strip() and value.casefold() not in UNKNOWN


def _local_media(value):
    if not isinstance(value, str) or not value or "://" in value or value.startswith(("//", "\\\\")):
        raise ValueError("local_media_path_required")
    supplied = Path(value).expanduser()
    path = supplied.resolve()
    if {part.casefold() for part in (*supplied.parts, *path.parts)} & {"excluded", "quarantine"}:
        raise ValueError("excluded_media_not_allowed")
    if not path.is_file() or path.suffix.casefold() not in MEDIA_SUFFIXES:
        raise ValueError("local_media_file_required")
    return path


def _review_date(value):
    try:
        datetime.fromisoformat(value)
        return True
    except (ValueError, TypeError):
        return False


def _approved_origin(row):
    if row.get("rights_confirmed") is not True:
        return False
    if row.get("origin") == "locally_recorded":
        return True
    if row.get("origin") != "external_licensed":
        return False
    license_url = EXTERNAL_LICENSES.get(row.get("license_id"))
    if not license_url or row.get("license_url") != license_url:
        return False
    source = urlsplit(row.get("source_url", ""))
    context = row.get("source_context", {})
    if not (source.scheme == "https" and source.hostname and not source.username and not source.password
            and _known(row.get("attribution")) and _known(row.get("rights_reviewed_by"))
            and _review_date(row.get("rights_reviewed_at")) and isinstance(context, dict)
            and context.get("country") == "KZ"
            and context.get("scene_type") in {"bank_cashdesk", "bank_vault", "bank_incassation"}
            and row.get("bank_scene_confirmed") is True):
        return False
    proof_value = row.get("rights_proof_path")
    if not isinstance(proof_value, str) or "://" in proof_value:
        return False
    supplied = Path(proof_value).expanduser()
    proof = supplied.resolve()
    if {part.casefold() for part in (*supplied.parts, *proof.parts)} & {"excluded", "quarantine"}:
        return False
    return (proof.is_file() and proof.stat().st_size > 0
            and file_sha256(proof) == validate_content_sha256(row.get("rights_proof_sha256")))


def freeze_development(videos_dir, labels_path, annotations_path=None):
    """Hash the full previously used cohort; missing angle/source IDs stay unknown."""
    labels = load_labels(labels_path)
    directory = Path(videos_dir).resolve()
    paths = {}
    for path in directory.iterdir():
        if path.is_file() and path.suffix.casefold() in MEDIA_SUFFIXES:
            key = video_key(path.name)
            if key in paths:
                raise ValueError("canonical_media_collision")
            paths[key] = _local_media(str(path))
    metadata = defaultdict(dict)
    if annotations_path is not None:
        with Path(annotations_path).open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                key = video_key(row["video"])
                for field in ("angle_id", "source_group"):
                    value = row.get(field, "")
                    if _known(value):
                        if field in metadata[key] and metadata[key][field] != value:
                            raise ValueError("conflicting_development_metadata")
                        metadata[key][field] = value
    if set(paths) != {video_key(row["video"]) for row in labels}:
        raise ValueError("development_labels_media_mismatch")
    records = []
    for label in labels:
        key = video_key(label["video"])
        path = paths[key]
        records.append({"video_id": label["video"], "path": str(path), "origin": "organizer_open",
                        "label": int(label["positive"]), "timestamp": label["timestamp"],
                        "content_sha256": file_sha256(path),
                        "angle_id": metadata[key].get("angle_id", "unknown"),
                        "source_group": metadata[key].get("source_group", "unknown")})
    return {"schema_version": 1, "development_used_for_tuning": True,
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "records": records}


def _records(document, side):
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError("invalid_manifest_schema")
    rows = document.get("records")
    if not isinstance(rows, list) or not rows:
        raise ValueError("nonempty_manifest_required")
    source_index(rows, side)
    return rows


def prepare(development, holdout=None):
    """Return blocked readiness or verified inputs for ml.eval; never infer camera IDs or truth."""
    development_rows = _records(development, "development")
    if development.get("development_used_for_tuning") is not True:
        raise ValueError("development_history_declaration_required")
    if holdout is None:
        return {"status": "blocked", "new_angle_available": False,
                "blocking_codes": ["new_angle_manifest_missing"], "new_angle_holdout_validated": False}
    holdout_rows = _records(holdout, "holdout")
    problems = set()
    if holdout.get("annotation_origin") != "human" or not _known(holdout.get("reviewed_by")):
        problems.add("human_review_required")
    if not _review_date(holdout.get("reviewed_at")):
        problems.add("human_review_date_required")
    if holdout.get("events_exhaustive") is not True:
        problems.add("exhaustive_event_inventory_required")
    if any(not _approved_origin(row) for row in holdout_rows):
        problems.add("approved_origin_and_rights_required")
    # Do not read excluded or unapproved external bytes as a side effect of readiness inspection.
    if "approved_origin_and_rights_required" in problems:
        return {"status": "blocked", "new_angle_available": False,
                "blocking_codes": sorted(problems), "new_angle_holdout_validated": False}
    labels, annotations, hashes, media_paths, split = [], [], {}, {}, {"train": [], "test": []}
    seen_ids, hash_sides, source_sides, angle_sides = set(), defaultdict(set), defaultdict(set), defaultdict(set)
    contents = defaultdict(list)
    for side, rows in (("train", development_rows), ("test", holdout_rows)):
        for row in rows:
            literal = row.get("video_id", row.get("video"))
            key = video_key(literal)
            if key in seen_ids:
                raise ValueError("canonical_identity_crosses_partitions")
            seen_ids.add(key)
            path = _local_media(row.get("path"))
            digest = file_sha256(path)
            if digest != validate_content_sha256(row.get("content_sha256")):
                raise ValueError("declared_hash_disagrees_with_actual_bytes")
            hashes[literal] = digest
            media_paths[literal] = str(path)
            label = row.get("label")
            if type(label) is not int or label not in (0, 1):
                raise ValueError("explicit_binary_label_required")
            angle, source = row.get("angle_id"), row.get("source_group")
            if not _known(angle):
                problems.add("explicit_development_and_test_angles_required")
            if not _known(source):
                problems.add("source_recording_groups_required")
            else:
                source_sides[source.casefold()].add(side)
            if _known(angle):
                angle_sides[angle.casefold()].add(side)
            hash_sides[digest].add(side)
            events = row.get("events", [])
            if not isinstance(events, list):
                raise ValueError("event_inventory_must_be_array")
            if side == "test" and label and not events:
                problems.add("positive_event_annotations_required")
            labels.append({"video": literal, "positive": bool(label), "timestamp": row.get("timestamp")})
            normalized_events = []
            if side == "test":
                for event in events:
                    if not isinstance(event, dict) or not _known(event.get("type_id")) or event.get("type_id") == "none":
                        raise ValueError("explicit_event_type_required")
                    peak = evaluation.parse_timestamp(event.get("t_peak"))
                    normalized_events.append({"type_id": event["type_id"], "t_peak": peak})
                    annotations.append({"video": literal, "angle_id": angle or "unknown",
                                        "type_id": event["type_id"], "t_peak": peak})
                if not label and events:
                    raise ValueError("negative_label_contradicts_events")
            if not events or side != "test":
                annotations.append({"video": literal, "angle_id": angle or "unknown"})
            split[side].append({"video_id": literal, "angle_id": angle, "content_sha256": digest})
            contents[digest].append({"side": side, "label": label, "angle": angle,
                                     "events": sorted(normalized_events, key=lambda event: (event["t_peak"], event["type_id"]))})
    if any(len(sides) > 1 for sides in hash_sides.values()):
        problems.add("development_content_reused_as_test")
    if any(len(sides) > 1 for sides in source_sides.values()):
        problems.add("source_recording_crosses_partitions")
    if any(len(sides) > 1 for sides in angle_sides.values()):
        problems.add("camera_angle_crosses_partitions")
    for members in contents.values():
        if len({row["label"] for row in members}) > 1:
            raise ValueError("duplicate_content_label_conflict")
        if len({row["angle"] for row in members}) > 1:
            problems.add("duplicate_content_angle_conflict")
        test_events = [json.dumps(row["events"], sort_keys=True) for row in members if row["side"] == "test"]
        if len(set(test_events)) > 1:
            raise ValueError("duplicate_content_event_inventory_conflict")
    if problems:
        return {"status": "blocked", "new_angle_available": True,
                "blocking_codes": sorted(problems), "new_angle_holdout_validated": False}
    evaluation._validate_split(labels, split, evaluation._annotation_index(labels, annotations),
                               evaluation._content_hash_index(labels, hashes))
    return {"status": "ready", "new_angle_available": True, "blocking_codes": [],
            "new_angle_holdout_validated": True, "labels": labels, "annotations": annotations,
            "split": split, "content_hashes": hashes, "media_paths": media_paths,
            "review": {field: holdout[field] for field in ("annotation_origin", "reviewed_by", "reviewed_at", "events_exhaustive")},
            "test_source_provenance": {
                row.get("video_id", row.get("video")): {field: row[field] for field in (
                    "origin", "rights_confirmed", "source_url", "license_id", "license_url", "attribution",
                    "rights_reviewed_by", "rights_reviewed_at", "rights_proof_path", "rights_proof_sha256",
                    "source_context", "bank_scene_confirmed", "angle_id", "source_group") if field in row}
                for row in holdout_rows},
            "verification_scope": "actual_exact_hashes_and_provided_human_camera_recording_labels",
            "limitations": ["No visual or rights verification is performed by this tool.",
                            "Hash isolation does not detect undeclared re-encodes or related recordings.",
                            "Keep model, prompt and thresholds frozen before viewing test outcomes."]}


def score(prepared, predictions, tolerance_sec=2.0):
    if prepared.get("status") != "ready" or prepared.get("new_angle_holdout_validated") is not True:
        raise ValueError("holdout_not_ready")
    for literal, digest in prepared["content_hashes"].items():
        if file_sha256(_local_media(prepared["media_paths"][literal])) != digest:
            raise ValueError("media_changed_after_preparation")
    for provenance in prepared["test_source_provenance"].values():
        if not _approved_origin(provenance):
            raise ValueError("rights_evidence_changed_after_preparation")
    report = evaluation.score(prepared["labels"], predictions, annotations=prepared["annotations"],
                              split=prepared["split"], subset="test", tolerance_sec=tolerance_sec,
                              content_hashes=prepared["content_hashes"])
    by_angle = defaultdict(list)
    for row in report["per_content_group"]:
        by_angle[row["angle_id"]].append(row)
    independent = {}
    for angle, rows in by_angle.items():
        confusion = Counter(row["outcome"] for row in rows)
        truth_count = predicted_count = joint = timed = type_only = 0
        for row in rows:
            truth, predicted = row["truth_events"], row["predicted_events"]
            truth_count += len(truth)
            predicted_count += len(predicted)
            timed += len(evaluation._match_events(truth, predicted, tolerance_sec))
            type_only += sum((Counter(e["type"] for e in truth) & Counter(e["type"] for e in predicted)).values())
            for kind in {event["type"] for event in truth}:
                joint += len(evaluation._match_events([e for e in truth if e["type"] == kind],
                                                       [e for e in predicted if e["type"] == kind], tolerance_sec))
        independent[angle] = {"unique_contents": len(rows), "truth_events": truth_count,
                              "predicted_events": predicted_count, "type_only_matches": type_only,
                              "time_only_matches": timed, "joint_type_time_matches": joint,
                              "joint_precision": joint / predicted_count if predicted_count else None,
                              "joint_recall": joint / truth_count if truth_count else None,
                              "failed_or_missing": sum(row["status"] != "ok" for row in rows),
                              "video_confusion": dict(confusion),
                              "end_to_end_accuracy": (confusion["tp"] + confusion["tn"]) / len(rows),
                              "end_to_end_binary": evaluation._metrics(
                                  confusion["tp"], confusion["tn"], confusion["fp"],
                                  confusion["fn"] + confusion["abstain_positive"], len(rows))}
    report["independent_angle_events"] = independent
    totals = {field: sum(row[field] for row in independent.values()) for field in
              ("unique_contents", "truth_events", "predicted_events", "type_only_matches",
               "time_only_matches", "joint_type_time_matches", "failed_or_missing")}
    joint, expected, predicted = (totals[field] for field in
                                   ("joint_type_time_matches", "truth_events", "predicted_events"))
    totals.update(joint_precision=joint / predicted if predicted else None,
                  joint_recall=joint / expected if expected else None)
    report["independent_overall_events"] = totals
    report["holdout_provenance"] = {key: prepared[key] for key in
                                    ("verification_scope", "limitations", "review", "test_source_provenance")}
    return report


def _write(path, document, inputs=()):
    path = Path(path).absolute()
    allowed = (ROOT / "var", ROOT / "artifacts")
    if ".." in path.parts or not any(path.is_relative_to(folder) for folder in allowed):
        raise ValueError("private_output_directory_required")
    if any(ancestor.is_symlink() for ancestor in (path, *path.parents) if ancestor != ROOT.parent):
        raise ValueError("output_symlink_not_allowed")
    if any(path.resolve() == Path(value).resolve() for value in inputs if value):
        raise ValueError("output_overlaps_input")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        # Exclusive creation avoids following a pre-existing predictable .part
        # symlink, including one aimed at an input manifest.
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix=f".{path.name}.",
                                         suffix=".part", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze-development")
    freeze.add_argument("--videos-dir", required=True)
    freeze.add_argument("--labels", required=True)
    freeze.add_argument("--annotations")
    freeze.add_argument("--output", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--development", required=True)
    prepare_parser.add_argument("--holdout")
    prepare_parser.add_argument("--output", required=True)
    scoring = commands.add_parser("score")
    scoring.add_argument("--prepared", required=True)
    scoring.add_argument("--predictions", required=True)
    scoring.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    def read(path):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    try:
        if args.command == "freeze-development":
            result = freeze_development(args.videos_dir, args.labels, args.annotations)
            inputs = (args.labels, args.annotations)
        elif args.command == "prepare":
            result = prepare(read(args.development), read(args.holdout) if args.holdout else None)
            inputs = (args.development, args.holdout)
        else:
            result = score(read(args.prepared), read(args.predictions))
            inputs = (args.prepared, args.predictions)
        _write(args.output, result, inputs)
    except (ValueError, OSError, TypeError, KeyError):
        print(json.dumps({"status": "blocked", "blocking_codes": ["invalid_input_or_output"]}))
        return 2
    print(json.dumps({"status": result.get("status", "complete"),
                      "blocking_codes": result.get("blocking_codes", [])}))
    return 1 if result.get("status") == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
