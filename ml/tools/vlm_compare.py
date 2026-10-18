"""Local numeric A/B comparison; no model imports, decoding or private text output.

The baseline is a reference arm, not necessarily CV. Only --cv-baseline permits
an exact-CV-retention claim. All quality math/duplicate aggregation delegates to
ml.eval.score; primary-only annotations never establish exhaustive precision.
"""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
from xml.etree.ElementTree import ParseError
from zipfile import BadZipFile

from ml.eval import _audit_directory, _load_records, score
from ml.evaluation_labels import file_sha256, load_labels, parse_timestamp, validate_content_sha256, video_key

ROOT = Path(__file__).resolve().parents[2]
TYPE = re.compile(r"[a-z][a-z0-9_]{0,79}")
SUCCESS = {"done", "ok", "success", "completed"}
REPORT_STATUSES = {"completed", "video_timeout", "budget_exhausted", "frames_unavailable",
                   "backend_error", "startup_failed", "prompt_too_long"}


def file_identity(path):
    value = Path(path).lstat()
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def sha256(path):
    before = file_identity(path)
    result = file_sha256(path)
    if file_identity(path) != before:
        raise ValueError("input_changed_during_hash")
    return result


def read_json(path):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("duplicate_json_key")
            value[key] = item
        return value

    def nonfinite(value):
        raise ValueError("nonfinite_json")

    if Path(path).stat().st_size > 64 * 1024 * 1024:
        raise ValueError("report_too_large")
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=nonfinite)


def _number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _one(value, names, default=None):
    items = [value[name] for name in names if value.get(name) is not None]
    if items and any(item != items[0] for item in items):
        raise ValueError("conflicting_aliases")
    return items[0] if items else default


def _event(value):
    if not isinstance(value, dict):
        raise ValueError("invalid_event")
    kind = value.get("type_id", value.get("type", "unknown"))
    peak = _one(value, ("t", "timecode_sec", "t_peak"))
    confidence = _one(value, ("conf", "confidence"))
    if (not isinstance(kind, str) or TYPE.fullmatch(kind) is None
            or not _number(peak) or peak < 0
            or confidence is not None and (not _number(confidence) or not 0 <= confidence <= 1)):
        raise ValueError("invalid_event")
    result = {"type_id": kind, "t_peak": float(peak), "confidence": float(confidence) if confidence is not None else None}
    for name in ("t_start", "t_end"):
        if value.get(name) is not None:
            if not _number(value[name]) or value[name] < 0:
                raise ValueError("invalid_interval")
            result[name] = float(value[name])
    if not result.get("t_start", peak) <= peak <= result.get("t_end", peak):
        raise ValueError("invalid_interval")
    return result


def predictions(document):
    """Discard all descriptions, observations, evidence and exception messages."""
    if not isinstance(document, dict):
        raise ValueError("prediction_map_required")
    result = {}
    for literal, raw in document.items():
        key = video_key(literal)
        if key in result:
            raise ValueError("canonical_collision")
        value = {"video": key, "status": "invalid", "enhancement_error": False}
        result[key] = value
        if not isinstance(raw, dict):
            continue
        value["enhancement_error"] = "error" in raw or "vlm_error" in raw
        status = raw.get("status", "done")
        if not isinstance(status, str) or status not in SUCCESS:
            value["status"] = status if isinstance(status, str) and status in {"failed", "error", "timed_out", "timeout", "timedout"} else "invalid"
            continue
        payload = raw.get("result", raw)
        try:
            if not isinstance(payload, dict):
                raise ValueError("invalid_answer")
            if any(name in raw and name in payload and raw[name] != payload[name]
                   for name in ("verdict", "violations")):
                raise ValueError("conflicting_envelope")
            events = payload["violations"]
            verdict = payload["verdict"]
            if (verdict not in {"violation", "no_violation"} or not isinstance(events, list)
                    or len(events) > 5 or bool(events) != (verdict == "violation")):
                raise ValueError("invalid_answer")
            parsed = [_event(item) for item in events]
        except (KeyError, TypeError, ValueError):
            continue
        value.update(status="done", verdict=verdict, violations=parsed)
    return result


def primary_annotations(labels, annotations):
    indexed = {video_key(row["video"]): row for row in labels}
    result = {}
    for raw in annotations:
        key = video_key(raw["video"])
        if key in result or key not in indexed:
            raise ValueError("primary_annotation_identity")
        kind, peak = raw.get("type_id"), raw.get("t_peak")
        peak = None if peak in (None, "") else parse_timestamp(peak)
        if not isinstance(kind, str) or TYPE.fullmatch(kind) is None:
            raise ValueError("primary_annotation_type")
        label = indexed[key]
        if ((label["positive"] and (kind == "none" or peak is None))
                or (not label["positive"] and (kind != "none" or peak is not None))
                or label["timestamp"] != peak):
            raise ValueError("primary_annotation_conflict")
        result[key] = {"video": key, "type_id": kind, "t_peak": peak}
    if set(result) != set(indexed):
        raise ValueError("primary_annotation_coverage")
    return result


def actual_hashes(document, known):
    if not isinstance(document, dict):
        raise ValueError("actual_hash_report_required")
    metadata = document.get("content_aggregation", {})
    if not isinstance(metadata, dict):
        raise ValueError("actual_hash_report_required")
    if metadata.get("hash_source") != "actual" or metadata.get("hashes_authoritatively_verified") is not True:
        raise ValueError("actual_hash_report_required")
    result, seen = {}, set()
    for group in document.get("per_content_group", []):
        if not isinstance(group, dict):
            raise ValueError("invalid_hash_group")
        digest = validate_content_sha256(group.get("content_sha256"))
        members = group.get("videos")
        if digest in seen or not isinstance(members, list) or not members:
            raise ValueError("invalid_hash_group")
        seen.add(digest)
        for literal in members:
            key = video_key(literal)
            if key not in known or key in result:
                raise ValueError("hash_identity_conflict")
            result[key] = digest
    if set(result) != set(known):
        raise ValueError("actual_hash_coverage")
    return result


def _scored(labels, rows, annotations, hashes, tolerance, primary_only=False):
    records = []
    for key, raw in rows.items():
        value = dict(raw)
        if primary_only and value["status"] == "done":
            value["violations"] = [e for e in value["violations"] if e["type_id"] == annotations[key]["type_id"]]
            value["verdict"] = "violation" if value["violations"] else "no_violation"
        records.append(value)
    return score(labels, records, annotations=list(annotations.values()),
                 content_hashes=hashes, tolerance_sec=tolerance)


def _inventory(events):
    # Shared evaluator already unions duplicates with maximum multiplicity.
    return Counter(json.dumps(event, sort_keys=True) for event in events)


def _exact_retention(before, after, selected):
    changed, checked = [], 0
    for key in sorted(selected):
        original = before[key]
        events = original.get("violations", [])
        checked += len(events)
        missing = _inventory(events) - _inventory(after.get(key, {}).get("violations", []))
        if original["status"] != "done" or after.get(key, {}).get("status") != "done" or missing:
            changed.append({"video_id": key, "missing_exact_events": sum(missing.values())})
    return {"passed": not changed, "checked_files": len(selected), "checked_events": checked,
            "nonempty_check_exercised": checked > 0, "changed_files": changed}


def audit_summary(audit, rows, hashes, cv_sha=None):
    errors = sum(value.get("enhancement_error", False) for value in rows.values())
    if audit is None:
        return {"available": False, "source_stable": None, "status_counts": {},
                "unknown_metadata_files": len(rows), "actual_model_calls": None,
                "semantic_rejections_recorded": 0, "malformed_windows": 0,
                "uncertain_decisions": 0, "enhancement_error_files": errors}
    if not isinstance(audit, dict):
        raise ValueError("audit_object_required")
    config = audit.get("config", {})
    if not isinstance(config, dict) or not isinstance(audit.get("videos", {}), dict):
        raise ValueError("audit_object_required")
    if cv_sha is not None and config.get("baseline_sha256") not in (None, cv_sha):
        raise ValueError("audit_cv_hash_mismatch")
    videos = {}
    for literal, item in audit.get("videos", {}).items():
        key = video_key(literal)
        if key in videos or key not in rows or not isinstance(item, dict):
            raise ValueError("audit_identity")
        if item.get("sha256") not in (None, hashes[key]):
            raise ValueError("audit_content_hash")
        videos[key] = item
    statuses, calls = Counter(), []
    missing = rejected = malformed = uncertain = reused = 0
    for key in rows:
        item = videos.get(key, {})
        report = item.get("report", {})
        if not isinstance(report, dict):
            raise ValueError("audit_report_required")
        status = report.get("status")
        observed = isinstance(status, str) and status in REPORT_STATUSES
        statuses[status if observed else "unknown"] += 1
        missing += not observed
        actual = item.get("actual_calls_used")
        if type(actual) is int and actual >= 0:
            calls.append(actual)
        was_reused = item.get("inference_reused") is True
        if was_reused and actual not in (None, 0):
            raise ValueError("reused_inference_calls_must_be_zero")
        cap = config.get("max_model_calls_per_video")
        if (type(actual) is int and type(cap) is int and actual > cap
                or not was_reused and type(actual) is int and type(report.get("calls_used")) is int
                and actual != report["calls_used"]):
            raise ValueError("audit_call_accounting")
        if was_reused and item.get("inference_source_video") is not None:
            owner = video_key(item["inference_source_video"])
            if owner not in hashes or hashes[owner] != hashes[key] or owner == key:
                raise ValueError("audit_reuse_source")
        reused += was_reused
        decisions, verifications = report.get("decisions", []), item.get("verifier_last_call_decisions", [])
        if not isinstance(decisions, list) or not isinstance(verifications, list):
            raise ValueError("audit_decisions_required")
        for decision in decisions:
            if isinstance(decision, dict):
                malformed += decision.get("reason") == "invalid_response"
                uncertain += decision.get("status") == "uncertain"
        for decision in verifications:
            if isinstance(decision, dict):
                rejected += (decision.get("accepted") is False
                             and decision.get("reason") == "primitive_evidence_not_supported")
    return {"available": True, "source_stable": audit.get("code_inputs_stable")
            if type(audit.get("code_inputs_stable")) is bool else None,
            "status_counts": dict(statuses), "unknown_metadata_files": missing,
            "actual_model_calls": sum(calls) if len(calls) == len(rows) else None,
            "call_metadata_files": len(calls), "duplicate_inference_reused": reused,
            "semantic_rejections_recorded": rejected, "malformed_windows": malformed,
            "uncertain_decisions": uncertain, "enhancement_error_files": errors,
            "diagnostic_unit": "recorded_file_decisions_including_reused_and_last_call_only",
            "warm_runtime_is_not_full_pipeline_sla": True}


def compare(baseline, probe, labels, annotations, hashes, *, cv=None, audit=None, cv_sha=None, tolerance=2.0):
    """Compare already numeric maps; return no private text or evidence content."""
    if not _number(tolerance) or tolerance < 0:
        raise ValueError("invalid_tolerance")
    known = {video_key(row["video"]) for row in labels}
    if set(baseline) != known or not probe or set(probe) - known or set(hashes) != known:
        raise ValueError("comparison_identity")
    if cv is not None and set(cv) != known:
        raise ValueError("cv_identity")
    selected_hashes = {hashes[key] for key in probe}
    selected = {key for key in known if hashes[key] in selected_hashes}
    cohort = [row for row in labels if video_key(row["video"]) in selected]
    ann = {key: value for key, value in annotations.items() if key in selected}
    content = {key: value for key, value in hashes.items() if key in selected}
    arms, details = {}, {}
    for name, values in (("baseline", baseline), ("probe", probe)):
        rows = {key: value for key, value in values.items() if key in selected}
        ordinary = _scored(cohort, rows, ann, content, tolerance)
        # Filtering by the evaluated primary type BEFORE matching means an
        # unrelated closer event cannot steal the correct same-event match.
        typed = _scored(cohort, rows, ann, content, tolerance, primary_only=True)
        typed_groups = {g["content_sha256"]: g for g in typed["per_content_group"]}
        arms[name] = {"confusion": ordinary["overall"]["confusion"],
                      "contents": ordinary["overall"]["total"],
                      "scored_contents": ordinary["overall"]["scored"],
                      "unknown_type_events": ordinary["overall"]["types"]["unknown"]["predicted_events"],
                      "primary_type_hits": 0, "primary_joint_hits": 0}
        for group in ordinary["per_content_group"]:
            digest = group["content_sha256"]
            typed_group = typed_groups[digest]
            valid = typed_group["status"] == "ok" and group["truth_positive"]
            item = {"status": group["status"], "predicted_positive": group["predicted_positive"],
                    "primary_type_hit": bool(valid and typed_group["predicted_positive"]),
                    "primary_joint_hit": bool(valid and typed_group["matched_events"])}
            arms[name]["primary_type_hits"] += item["primary_type_hit"]
            arms[name]["primary_joint_hits"] += item["primary_joint_hit"]
            target = details.setdefault(digest, {"content_sha256": digest, "members": group["videos"],
                "truth_positive": group["truth_positive"]})
            target[name] = item
            target[name + "_events"] = group["predicted_events"]
    cv_groups = {} if cv is None else {g["content_sha256"]: g for g in _scored(cohort,
        {key: value for key, value in cv.items() if key in selected}, ann, content, tolerance)["per_content_group"]}
    loss_type = loss_joint = gain_type = gain_joint = clean_adds = positive_extras = corroborating = 0
    cv_clean_adds = None if cv is None else 0
    for item in details.values():
        before, after = item["baseline"], item["probe"]
        loss_type += before["primary_type_hit"] and not after["primary_type_hit"]
        loss_joint += before["primary_joint_hit"] and not after["primary_joint_hit"]
        gain_type += after["primary_type_hit"] and not before["primary_type_hit"]
        gain_joint += after["primary_joint_hit"] and not before["primary_joint_hit"]
        candidate_events = item.pop("probe_events")
        new = _inventory(candidate_events) - _inventory(item.pop("baseline_events"))
        added = sum(new.values())
        item["new_events"] = added
        if item["truth_positive"]:
            # Primary-only labels cannot adjudicate the remaining extra events.
            supported = int(added > 0 and after["primary_joint_hit"] and not before["primary_joint_hit"])
            corroborating += supported
            positive_extras += added - supported
        else:
            clean_adds += added
            if cv is not None:
                cv_clean_adds += sum((_inventory(candidate_events)
                    - _inventory(cv_groups[item["content_sha256"]]["predicted_events"])).values())
    retention = None if cv is None else {name: _exact_retention(cv, values, selected)
                                        for name, values in (("baseline", baseline), ("probe", probe))}
    trace = audit_summary(audit, probe, hashes, cv_sha)
    complete = set(probe) == known
    responses_valid = all(arm["scored_contents"] == arm["contents"] for arm in arms.values())
    cv_ok = retention is not None and all(value["passed"] for value in retention.values())
    metric_gate = complete and responses_valid and cv_ok and not loss_type and not loss_joint and cv_clean_adds == 0 and bool(gain_type or gain_joint)
    audit_complete = (trace["source_stable"] is True and trace["unknown_metadata_files"] == 0
                      and trace["actual_model_calls"] is not None and not trace["enhancement_error_files"]
                      and set(trace["status_counts"]) <= {"completed"})
    return {"format_version": 1, "status": "complete" if complete else "partial",
            "scope": {"development_only": True, "new_camera_evaluation": False,
                      "annotations_primary_only": True, "full_event_precision_available": False,
                      "runtime_does_not_measure_full_pipeline_sla": True,
                      "hashes_from_supplied_report_not_rehashed_media": True,
                      "aggregation_unit": "actual_sha256_content_group", "tolerance_sec": tolerance},
            "cohort": {"expected_files": len(known), "compared_files": len(selected),
                       "probe_files": len(probe), "compared_contents": len(details),
                       "missing_probe_files": len(known - set(probe))},
            "arms": arms, "paired": {"type_gains": gain_type, "type_losses": loss_type,
                "joint_gains": gain_joint, "joint_losses": loss_joint,
                "new_events_on_known_clean": cv_clean_adds,
                "new_reference_events_on_known_clean": clean_adds,
                "new_primary_corroborating_events": corroborating,
                "new_events_on_positive_not_exhaustively_adjudicated": positive_extras},
            "exact_cv_retention": {"available": cv is not None, "arms": retention},
            "exact_reference_retention": _exact_retention(baseline, probe, selected),
            "audit": trace, "metric_gate_passed": metric_gate,
            "eligible_for_fresh_full_run": metric_gate and audit_complete,
            "per_content": list(details.values())}


def _local_file(value):
    path = Path(value).absolute()
    parent = _audit_directory(path.parent)
    path = parent / path.name
    if path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("local_regular_file_required")
    return path


def _save(path, document, inputs):
    path = Path(path).absolute()
    if (".." in path.parts or not path.is_relative_to(ROOT / "var") or path.suffix != ".json"
            or any(p.is_symlink() for p in (path, *path.parents)) or path.exists()
            or any(path.resolve() == p.resolve() for p in inputs)):
        raise ValueError("fresh_private_var_output_required")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(document, stream, ensure_ascii=True, allow_nan=False, indent=2)
            stream.write("\n")
        os.link(temporary, path)  # Never overwrite a concurrent human result.
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class SafeParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError("invalid_arguments")


def main(argv=None):
    parser = SafeParser(description=__doc__)
    for name in ("baseline", "probe", "labels", "annotations", "hash-report", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--cv-baseline", type=Path)
    parser.add_argument("--tolerance", type=float, default=2.0)
    parser.add_argument("--expect-complete", action="store_true")
    try:
        args = parser.parse_args(argv)
        names = ("baseline", "probe", "labels", "annotations", "hash_report", "audit", "cv_baseline")
        paths = {name: _local_file(getattr(args, name)) for name in names if getattr(args, name) is not None}
        identities = {name: file_identity(path) for name, path in paths.items()}
        digests = {name: sha256(path) for name, path in paths.items()}
        labels = load_labels(paths["labels"])
        annotations = primary_annotations(labels, _load_records(paths["annotations"]))
        hashes = actual_hashes(read_json(paths["hash_report"]), set(annotations))
        report = compare(predictions(read_json(paths["baseline"])), predictions(read_json(paths["probe"])),
            labels, annotations, hashes, tolerance=args.tolerance,
            cv=predictions(read_json(paths["cv_baseline"])) if "cv_baseline" in paths else None,
            audit=read_json(paths["audit"]) if "audit" in paths else None, cv_sha=digests.get("cv_baseline"))
        if any(file_identity(paths[name]) != identity or sha256(paths[name]) != digests[name]
               for name, identity in identities.items()):
            raise ValueError("inputs_changed")
        report["input_sha256"] = digests
        report["scoring_engine"] = "ml.eval.score"
        report["scoring_source_sha256"] = {
            "comparator": sha256(Path(__file__)), "evaluator": sha256(ROOT / "ml/eval.py")}
        _save(args.out, report, paths.values())
        print(json.dumps({"status": report["status"], "metric_gate_passed": report["metric_gate_passed"],
                          "eligible_for_fresh_full_run": report["eligible_for_fresh_full_run"],
                          "paired": report["paired"]}, sort_keys=True))
        return 2 if args.expect_complete and report["status"] != "complete" else 0
    except (OSError, ValueError, TypeError, KeyError, OverflowError, BadZipFile, ParseError):
        print('{"status":"blocked","reason":"invalid_or_missing_local_input"}')
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
