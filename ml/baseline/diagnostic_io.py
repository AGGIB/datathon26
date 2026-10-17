"""Local audit for development diagnostics. Never extracts, downloads or exports weights."""
import csv
import hashlib
import math
import re
import stat
import zlib
from collections import Counter
from pathlib import Path
from xml.etree.ElementTree import ParseError
from zipfile import BadZipFile

import numpy as np

from ml import eval as evaluation
from ml.evaluation_labels import file_sha256, load_labels, source_index, video_key
from . import dataset, features
from .core import FEATURE_SPEC, content_sha256, read_json


FLAGS = {"development_only": True, "camera_isolation_verified": False,
         "new_angle_holdout_validated": False, "deployable_model_created": False}


def local_file(value):
    """Reject network roots/ancestor links before opening a regular local file."""
    path = Path(value)
    # Validate the raw spelling first, before Path can normalize a UNC prefix.
    if str(value).startswith(("//", "\\\\")) or "://" in str(value):
        raise ValueError("Only local files are allowed")
    evaluation._audit_directory(path.parent)
    metadata = path.lstat()
    evaluation._reject_link(metadata)
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("Input must be a regular local file")
    return path


def audit_inputs(index_path, videos_dir, labels_path, manifest_path, weights):
    """Fresh source hashes + safe cache loader, then label-free mean pooling once/content."""
    paths = {"index": local_file(index_path), "labels": local_file(labels_path),
             "manifest": local_file(manifest_path)}
    input_hashes = {key: file_sha256(path) for key, path in paths.items()}
    try:
        labels = load_labels(paths["labels"])
    except (csv.Error, BadZipFile, ParseError, zlib.error,
            AttributeError, IndexError, KeyError, TypeError) as exc:
        # Malformed CSV/XLSX bytes, missing XML members/cells or invalid shared-string references.
        raise ValueError("Invalid label table serialization or schema") from exc
    index, manifest = read_json(paths["index"]), read_json(paths["manifest"])
    if not isinstance(index, dict) or not isinstance(manifest, list):
        raise ValueError("Extraction index and manifest have invalid schemas")
    records = index.get("videos")
    if not isinstance(records, list):
        raise ValueError("Extraction index requires video records")
    by_label = source_index(labels, "labels")
    by_manifest = source_index(manifest, "manifest")
    by_cache = source_index(records, "cache index")
    if not (set(by_label) == set(by_manifest) == set(by_cache)):
        raise ValueError("Exactly one manifest/cache/label per video is required")
    n = len(labels)
    for key, expected in {"expected_videos": n, "successful_videos": n,
                          "failed_videos": 0, "capped_videos": 0}.items():
        if type(index.get(key)) is not int or index[key] != expected:
            raise ValueError("Extraction totals must be complete, successful and uncapped")
    snapshot = content_sha256(index.get("snapshot_sha256"))
    # Enumerate no-follow entries before the existing fingerprint helper follows files.
    evaluation._audit_files(evaluation._audit_directory(weights))
    try:
        actual_snapshot = dataset.snapshot_fingerprint(weights)
    except AttributeError as exc:
        # The public loader assumes config and vision_config are JSON objects.
        raise ValueError("Invalid local snapshot configuration") from exc
    if actual_snapshot != snapshot:
        raise ValueError("Actual local encoder/processor/weights fingerprint changed")
    config = index.get("extraction_config")
    if not isinstance(config, dict):
        raise ValueError("Extraction configuration missing")
    groups, caches, dimensions, audit_records = {}, {}, set(), []
    for key in sorted(by_label):
        label, row, record = by_label[key], by_manifest[key], by_cache[key]
        if type(row.get("label")) is not int or row["label"] != int(label["positive"]):
            raise ValueError("Manifest binary label differs from organiser label")
        digest = content_sha256(row.get("content_sha256"))
        if content_sha256(record.get("content_sha256")) != digest:
            raise ValueError("Manifest/cache index content mismatch")
        if "content_sha256" in label and content_sha256(label["content_sha256"]) != digest:
            raise ValueError("Label content SHA256 differs from manifest/cache")
        if record.get("status") != "ok" or record.get("cap_reached") is not False:
            raise ValueError("Failed/capped cache cannot be used in diagnostic")
        name = record["video_id"]
        expected_file = hashlib.sha256(name.encode("utf-8")).hexdigest() + ".npz"
        if record.get("cache_file") != expected_file:
            raise ValueError("Cache file must be its literal-ID digest, without traversal")
        path = local_file(paths["index"].parent / expected_file)
        cache_hash = file_sha256(path)
        try:
            cache = features.load_cache(path)
        except (BadZipFile, zlib.error) as exc:
            raise ValueError("Invalid numeric cache archive") from exc
        meta, extraction = cache["metadata"], cache["metadata"].get("extraction", {})
        if not isinstance(extraction, dict):
            raise ValueError("Cache extraction provenance must be an object")
        if (cache["video_id"] != name or meta.get("source_filename") != name
                or meta.get("content_sha256") != digest or meta.get("snapshot_sha256") != snapshot
                or extraction.get("cap_reached") is not False or extraction.get("autorotate") is not True
                or extraction.get("time_origin") != "container-relative PTS: copyts + start_at_zero"):
            raise ValueError("Cache identity, fingerprint or extraction provenance mismatch")
        for field, cache_field in (("fps", "fps_requested"), ("max_frames", "max_frames"),
                                   ("batch_size", "batch_size"), ("device", "device"), ("seed", "seed")):
            if field not in config or extraction.get(cache_field) != config[field]:
                raise ValueError("Caches require a common extraction specification")
        frames, dimension = cache["embeddings"].shape
        if (type(record.get("n_frames")) is not int or record["n_frames"] != frames
                or extraction.get("n_frames") != frames or frames >= config["max_frames"]
                or type(record.get("embedding_dim")) is not int or record["embedding_dim"] != dimension):
            raise ValueError("Cache numeric dimensions disagree with the index")
        if file_sha256(path) != cache_hash:
            raise ValueError("Cache changed during audit")
        dimensions.add(dimension)
        if digest in caches and any(not np.array_equal(caches[digest][field], cache[field])
                                    for field in ("timestamps_s", "embeddings")):
            raise ValueError("Identical content has inconsistent numeric caches")
        caches[digest] = cache
        group = groups.setdefault(digest, {"content_sha256": digest, "label": row["label"],
                                          "members": [], "source_groups": [], "anchors": []})
        if group["label"] != row["label"]:
            raise ValueError("Identical content has contradictory labels")
        group["members"].append(name)
        if "source_group" in row and "source_groups" in row:
            raise ValueError("Use either source_group or source_groups, not both")
        sources = row.get("source_groups", [row.get("source_group")])
        if not isinstance(sources, list):
            raise ValueError("source_groups must be a list of explicit related-source tokens")
        for source in sources:
            if source is None or source == "":
                continue
            if not isinstance(source, str) or source != source.strip():
                raise ValueError("source_group must be a literal human metadata token")
            if source.lower() in {"unknown", "unassigned", "неизвестно"}:
                continue
            group["source_groups"].append(source)
        anchors = row.get("positive_times_s", [])
        if (not isinstance(anchors, list) or any(type(t) not in (int, float)
                or not math.isfinite(t) or t < 0 for t in anchors) or (not row["label"] and anchors)):
            raise ValueError("Invalid human anchors in manifest")
        group["anchors"].extend(anchors)
        if label["timestamp"] is not None and label["positive"]:
            group["anchors"].append(label["timestamp"])
        audit_records.append({"video_id": name, "content_sha256": digest,
                              "cache_sha256": cache_hash, "n_frames": frames,
                              "embedding_dim": dimension, "hash_status": "pending_source_audit"})
    # Reuse the team's source audit: raw filenames survive canonical joining; no decoder/model call.
    actual = evaluation.audit(labels, videos_dir, use_ffprobe=False)
    actual_by_id = {r["video_key"]: r for r in actual["per_video"]}
    for record in audit_records:
        row = actual_by_id[video_key(record["video_id"])]
        if row["file_status"] != "matched" or row["content_sha256"] != record["content_sha256"]:
            raise ValueError("Actual source bytes do not match cache/manifest SHA256")
        record["hash_status"] = "matched_actual_source_cache_manifest"
    if len(dimensions) != 1 or index.get("unique_contents") != len(groups) or index.get("encoded_contents") != len(groups):
        raise ValueError("Unique-content/dimension totals disagree")
    if any(file_sha256(path) != input_hashes[key] for key, path in paths.items()):
        raise ValueError("Inputs changed during audit")
    samples = []
    for digest, group in sorted(groups.items()):
        samples.append({k: group[k] for k in ("content_sha256", "label")})
        samples[-1].update(members=sorted(group["members"]), source_groups=sorted(set(group["source_groups"])),
                           vector=caches[digest]["embeddings"].mean(axis=0, dtype=np.float64))
    audit = {"report_kind": "development_cache_audit", **FLAGS,
             "videos_verified": n, "unique_contents": len(samples), "class_counts": [
                 sum(s["label"] == c for s in samples) for c in (0, 1)],
             "total_cached_frames": sum(r["n_frames"] for r in audit_records),
             "unique_content_frames": sum(len(c["timestamps_s"]) for c in caches.values()),
             "embedding_dim": next(iter(dimensions)), "feature_spec": dict(FEATURE_SPEC),
             "snapshot_sha256": snapshot, "extraction_config": config,
             "input_sha256": input_hashes, "cache_records": audit_records,
             "known_source_groups": len({g for s in samples for g in s["source_groups"]}),
             "unknown_camera_metadata": True,
             "alternative_anchors_by_content": {h: sorted(set(g["anchors"])) for h, g in groups.items()},
             "duplicate_groups": [s["members"] for s in samples if len(s["members"]) > 1]}
    return samples, audit


def load_frozen_cv(path, audit):
    """Accept real hash-bound CV responses; no truth reconstruction or missing=negative."""
    path = local_file(path)
    digest = file_sha256(path)
    document = read_json(path)
    if (not isinstance(document, dict) or document.get("schema_version") != 1
            or not isinstance(document.get("producer_revision"), str)
            or not re.fullmatch("[0-9a-f]{40}", document["producer_revision"])
            or type(document.get("development_tuned")) is not bool
            or not isinstance(document.get("records"), list)):
        raise ValueError("Frozen CV requires explicit revision, tuning provenance and records")
    records = source_index(document["records"], "frozen CV")
    expected = {video_key(r["video_id"]): r for r in audit["cache_records"]}
    if set(records) - set(expected):
        raise ValueError("Unknown video in frozen CV")
    grouped, statuses = {}, Counter()
    for key, source in expected.items():
        record = records.get(key)
        if record is not None and content_sha256(record.get("content_sha256")) != source["content_sha256"]:
            raise ValueError("Frozen CV content hash differs from actual source")
        parsed = evaluation._prediction(record)
        statuses[parsed["status"]] += 1
        grouped.setdefault(source["content_sha256"], []).append(parsed)
    rank = {"ok": 0, "missing": 1, "failed": 2, "timed_out": 3, "invalid": 4}
    result = {}
    for content, members in grouped.items():
        status = max((p["status"] for p in members), key=rank.__getitem__)
        result[content] = {"status": status, "decision": any(p["positive"] for p in members) if status == "ok" else None}
    if file_sha256(path) != digest:
        raise ValueError("Frozen CV changed during audit")
    return result, {"status": "provided", "file_sha256": digest,
                    "producer_revision": document["producer_revision"],
                    "development_tuned": document["development_tuned"],
                    "member_status_counts": dict(statuses),
                    "no_symmetric_blind_holdout_claim": True}
