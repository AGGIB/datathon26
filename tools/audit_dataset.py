#!/usr/bin/env python3
"""Read-only intake: labels, hashes, taxonomy and structural IDs, never video content.

Requires openpyxl; python-docx only when --taxonomy is requested. No network, media decoding or package-script imports.
Only the JSON report is written, strictly under this repository's data/audit/.
"""

import argparse
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import importlib
import os
import stat
import tempfile
from pathlib import Path
import re
import sys
import unicodedata


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# One team implementation for names, timestamps and actual XLSX loading.
from ml.labels import load_labels, normalize_video_name, parse_timestamp  # noqa: E402 (script ROOT bootstrap)

INPUT_KEYS = ("labels", "videos", "manifest", "taxonomy", "report_data", "visual_first", "visual_second")
OPTIONAL_KEYS = INPUT_KEYS[2:]


class AuditFailure(ValueError):
    """Only fixed, non-content diagnostics may cross the CLI boundary."""
    def __init__(self, code, input_key="inputs", module=None):
        self.code = code
        self.input_key = input_key if input_key in {*INPUT_KEYS, "arguments", "output", "inputs"} else "inputs"
        self.module = module if module in {"openpyxl", "docx"} else None
        super().__init__(code)


class InputChanged(AuditFailure):
    pass
HEADERS = ("наименование файла", "таймстемп", "нарушение / без нарушения")
LABELS = {"нарушение": "violation", "без нарушения": "no_violation"}
VIDEO_NAME = re.compile(r"[XХxх]-\d+(?:\.mov)?", re.IGNORECASE)
TYPE_HEADING = re.compile(r"([ТД]\d{2})\s+(.+)")
REPORT_WIDTHS = {"sources": 4, "types": 8, "additional_checks": 6}
REGISTRY_FIELDS = {"id", "name", "original_label", "timestamp", "summary", "assessment", "observation"}
VISUAL_FIELDS = {"id", "name", "evidence", "hypothesis", "limitations", "observation", "observed_interval", "visual_confidence"}


def nfc(value):
    return unicodedata.normalize("NFC", value)


def whitespace(value):
    return " ".join(nfc(value).split())


def name_key(value):
    """Validate a technical identifier, then use the shared team name normalizer."""
    if not isinstance(value, str) or not VIDEO_NAME.fullmatch(nfc(value.strip())):
        raise ValueError("invalid_technical_video_name")
    return normalize_video_name(nfc(value)).casefold()


def stat_identity(value):
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


def file_identity(path):
    """Include the directory entry as well as a possible symlink target."""
    return stat_identity(path.lstat()), stat_identity(path.stat())


def sha256(path):
    """Streaming hash; reject replacement, truncation or metadata changes while read."""
    path = Path(path)
    before = file_identity(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        if not stat.S_ISREG(opened.st_mode) or stat_identity(opened) != before[1]:
            raise InputChanged("file_changed_during_hash")
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        if stat_identity(os.fstat(stream.fileno())) != before[1]:
            raise InputChanged("file_changed_during_hash")
    if file_identity(path) != before:
        raise InputChanged("file_changed_during_hash")
    return digest.hexdigest()


def json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def read_json(path):
    def reject_constant(_):
        raise ValueError("nonfinite_json_number")
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=json_object,
                      parse_constant=reject_constant)


def input_path(value):
    path = Path(value)
    path = path if path.is_absolute() else ROOT / path
    if path.exists():
        return path.absolute()
    # macOS filenames may be stored in NFD; never choose between NFC collisions.
    matches = [p for p in path.parent.iterdir() if nfc(p.name) == nfc(path.name)] if path.parent.is_dir() else []
    if len(matches) != 1:
        raise AuditFailure("input_missing_or_nfc_ambiguous")
    return matches[0].absolute()


def path_label(path):
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


class Audit:
    def __init__(self):
        self.issues = []
        self.name_checks = {}

    def issue(self, code, where, severity="error", **details):
        self.issues.append(dict(code=code, where=where, severity=severity, **details))

    def index_names(self, rows, where):
        groups = defaultdict(list)
        for row in rows:
            try:
                key = name_key(row["name"])
            except (KeyError, ValueError):
                self.issue("invalid_video_name", where, position=row.get("position"))
                continue
            groups[key].append(row)
        collisions = []
        for key, group in groups.items():
            if len(group) > 1:
                raw = [r["name"] for r in group]
                kind = "exact_duplicate" if len(set(raw)) == 1 else (
                    "nfc_collision" if len({nfc(s) for s in raw}) == 1 else "normalized_name_collision")
                collisions.append(key)
                self.issue(kind, where, key=key, names=raw)
        self.name_checks[where] = {"rows": len(rows), "unique_keys": len(groups),
                                  "collision_groups": len(collisions)}
        # Ambiguous groups are deliberately absent from the join index.
        return {key: group[0] for key, group in groups.items() if len(group) == 1}

    def compare_keys(self, left, right, where):
        only_left = sorted(set(left) - set(right))
        only_right = sorted(set(right) - set(left))
        if only_left or only_right:
            self.issue("id_set_mismatch", where, only_left=only_left, only_right=only_right)
        return {"matched": len(set(left) & set(right)), "only_left": only_left, "only_right": only_right}

    def count(self, actual, expected, where):
        if actual != expected:
            self.issue("unexpected_count", where, actual=actual, expected=expected)


def blank_cell(value):
    return value is None or (isinstance(value, str) and not value.strip())


def read_labels(path, sheet, audit):
    """Thin shape/flag checks; all record loading and time parsing belong to ml.labels."""
    from openpyxl import load_workbook
    workbook = load_workbook(path, read_only=True, data_only=False)
    try:
        active = workbook.active
        if sheet is not None and sheet != active.title:
            raise AuditFailure("unsupported_sheet", "labels")
        if active.max_column > 3:
            raise AuditFailure("invalid_labels_width", "labels")
        rows = list(active.iter_rows(max_col=3, values_only=True))
    finally:
        workbook.close()
    if not rows or tuple(whitespace(str(x or "")).casefold() for x in rows[0]) != HEADERS:
        raise AuditFailure("invalid_labels_header", "labels")
    metadata = []
    for position, values in enumerate(rows[1:], start=2):
        blank = all(blank_cell(x) for x in values)
        if blank:
            audit.issue("blank_label_row", "labels", severity="warning", position=position)
        else:
            if blank_cell(values[0]) or blank_cell(values[2]):
                audit.issue("partial_label_row", "labels", position=position)
            if not blank_cell(values[0]) and not isinstance(values[0], str):
                audit.issue("invalid_video_name", "labels", position=position)
            if LABELS.get(whitespace(str(values[2] or "")).casefold()) is None:
                audit.issue("unknown_label", "labels", position=position)
        # Mirror only the shared loader's row inclusion, never its timestamp logic.
        if values[0]:
            metadata.append((position, values, blank))
    try:
        loaded = load_labels(path)
    except Exception as error:
        raise AuditFailure("shared_labels_loader_failed", "labels") from error
    if len(loaded) != len(metadata):
        raise AuditFailure("loader_disagreement", "labels")
    result = []
    for loaded_row, (position, values, blank) in zip(loaded, metadata):
        if blank:
            continue
        label = LABELS.get(whitespace(str(values[2] or "")).casefold())
        if loaded_row.video != normalize_video_name(str(values[0])):
            raise AuditFailure("loader_disagreement", "labels")
        if label is not None and loaded_row.has_violation != (label == "violation"):
            raise AuditFailure("loader_disagreement", "labels")
        seconds = loaded_row.t_sec
        if label == "violation" and seconds is None:
            audit.issue("missing_violation_timestamp", "labels", position=position)
        if label == "no_violation" and seconds is not None:
            audit.issue("timestamp_for_negative_label", "labels", position=position)
        result.append({"position": position, "name": values[0], "label": label,
                       "timestamp_original": str(values[1]) if values[1] is not None else None,
                       "timestamp_seconds": seconds})
    return result


def manifest_rows(raw, audit):
    if not isinstance(raw, list):
        raise ValueError("manifest_must_be_array")
    result = []
    ids = []
    for position, item in enumerate(raw, start=1):
        if not isinstance(item, dict) or not isinstance(item.get("file"), str):
            audit.issue("invalid_manifest_row", "manifest", position=position)
            continue
        # The author's absolute path is not followed. Only its technical basename is used.
        filename = item["file"].replace("\\", "/").rsplit("/", 1)[-1]
        row = {"position": position, "name": filename, "id": item.get("id"),
               "sha256": item.get("sha256"), "duration": item.get("duration"),
               "label": item.get("label"), "timestamp": item.get("timestamp")}
        for key in ("size_bytes", "size", "bytes"):
            if key in item:
                row["size_bytes"] = item[key]
                break
        try:
            key = name_key(filename)
            if name_key(item.get("name")) != key:
                audit.issue("manifest_name_mismatch", "manifest", position=position)
            if type(row["id"]) is not int or row["id"] != int(key.split("-")[1]):
                audit.issue("manifest_id_mismatch", "manifest", position=position)
        except ValueError:
            audit.issue("invalid_manifest_name", "manifest", position=position)
        if type(row["id"]) is int:
            ids.append(row["id"])
        result.append(row)
    if len(ids) != len(set(ids)):
        audit.issue("duplicate_numeric_id", "manifest")
    return result


def audit_video_files(videos, labels, manifest, audit):
    records = []
    for key, video in sorted(videos.items()):
        path = video["path"]
        before = file_identity(path)
        try:
            digest = sha256(path)
            after = path.stat()
            stable = file_identity(path) == before
        except (InputChanged, OSError):
            digest, stable = None, False
            after = type("UnavailableStat", (), {"st_size": before[1][3]})()
        if not stable:
            digest = None
            audit.issue("file_changed_during_hash", "videos", key=key)
        source = manifest.get(key)
        label = labels.get(key)
        hash_status, size_status = "not_provided", "not_provided"
        duration, duration_status = None, "unverified_no_manifest"
        if source:
            expected = source.get("sha256")
            if expected is None:
                audit.issue("missing_manifest_hash", "manifest", severity="warning", key=key)
            elif not isinstance(expected, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
                hash_status = "invalid"
                audit.issue("invalid_manifest_hash", "manifest", key=key)
            else:
                hash_status = "matched" if stable and expected.lower() == digest else "mismatch"
                if hash_status == "mismatch":
                    audit.issue("hash_mismatch", "videos", key=key)
            if "size_bytes" in source:
                size = source["size_bytes"]
                size_status = "matched" if type(size) is int and size == after.st_size else "mismatch"
                if size_status == "mismatch":
                    audit.issue("size_mismatch", "videos", key=key)
            value = source.get("duration")
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0:
                duration = float(value)
                duration_status = "manifest_hash_matched" if hash_status == "matched" else "unverified_hash_" + hash_status
            else:
                duration_status = "unverified_invalid_duration"
                audit.issue("invalid_manifest_duration", "manifest", key=key)
            if label:
                source_label = LABELS.get(whitespace(str(source.get("label"))).casefold())
                if source_label != label["label"]:
                    audit.issue("manifest_label_mismatch", "manifest", key=key)
                try:
                    source_time = parse_timestamp(source.get("timestamp"))
                except ValueError:
                    source_time = "invalid"
                if source_time != label["timestamp_seconds"]:
                    audit.issue("manifest_timestamp_mismatch", "manifest", key=key)
        bounds = "not_applicable" if label and label["timestamp_seconds"] is None else "unverified"
        if label and label["timestamp_seconds"] is not None and duration_status == "manifest_hash_matched":
            bounds = "within_manifest_duration" if 0 <= label["timestamp_seconds"] < duration else "out_of_bounds"
            if bounds == "out_of_bounds":
                audit.issue("timestamp_out_of_bounds", "labels", key=key)
        records.append({"key": key, "filename": path.name, "size_bytes": after.st_size,
                        "sha256": digest, "manifest_hash_check": hash_status,
                        "manifest_size_check": size_status,
                        "duration_seconds": duration, "duration_provenance": "author_manifest" if source else "not_provided",
                        "duration_verification": duration_status, "duration_measured_by_audit": False,
                        "label": label["label"] if label else None,
                        "timestamp_seconds": label["timestamp_seconds"] if label else None,
                        "timestamp_original": label.get("timestamp_original") if label else None,
                        "timestamp_bounds": bounds,
                        "name_spellings": {"video": video["name"], "labels": label["name"] if label else None,
                                           "manifest": source["name"] if source else None}})
    hashes = defaultdict(list)
    for record in records:
        if record["sha256"] is not None:
            hashes[record["sha256"]].append(record["key"])
    by_key = {record["key"]: record for record in records}
    for keys in hashes.values():
        if len(keys) > 1:
            audit.issue("duplicate_file_content", "videos", severity="warning", keys=keys)
            annotations = [{"key": key, "label": by_key[key]["label"],
                            "timestamp_seconds": by_key[key]["timestamp_seconds"],
                            "timestamp_original": by_key[key]["timestamp_original"]} for key in keys]
            if {"violation", "no_violation"} <= {row["label"] for row in annotations}:
                audit.issue("duplicate_content_label_conflict", "videos", annotations=annotations)
            if len({row["timestamp_seconds"] for row in annotations}) > 1:
                audit.issue("duplicate_content_multiple_timestamps", "videos", severity="warning", annotations=annotations)
    return records


def read_docx(path, audit):
    from docx import Document
    doc = Document(path)
    types, blocks = [], {}
    current = None
    for paragraph in doc.paragraphs:
        text = paragraph.text
        match = TYPE_HEADING.fullmatch(whitespace(text)) if paragraph.style.name == "Heading 2" else None
        if match:
            current = match[1]
            types.append({"id": current, "name": match[2]})
            if current in blocks:
                audit.issue("duplicate_taxonomy_id", "docx", id=current)
            blocks[current] = []
        elif paragraph.style.name.startswith("Heading"):
            current = None
        elif current:
            # Used locally to compare generic references. Never serialized or printed.
            blocks[current].append(text)
    registry = []
    matching_tables = []
    for table in doc.tables:
        if table.rows and whitespace(table.rows[0].cells[0].text) == "Видео и метка":
            matching_tables.append(table)
    if len(matching_tables) != 1:
        raise ValueError("docx_registry_table_missing_or_ambiguous")
    table = matching_tables[0]
    if len(table.columns) != 4:
        audit.issue("docx_registry_column_count", "docx")
    for position, row in enumerate(table.rows[1:], start=2):
        # No reads of observation/assessment columns or per-video paragraphs.
        match = re.match(r"^([XХxх]-\d+(?:\.mov)?)(?=\s|$)", row.cells[0].text.strip(), re.I)
        if match:
            registry.append({"position": position, "name": match[1]})
        else:
            audit.issue("docx_registry_invalid_id", "docx", position=position)
    return types, blocks, registry


def taxonomy_audit(raw, doc_types, blocks, doc_registry, labels, expected, audit):
    if not isinstance(raw, dict) or set(raw) != {*REPORT_WIDTHS, "registry"}:
        raise ValueError("invalid_report_sections")
    for section, width in REPORT_WIDTHS.items():
        rows = raw[section]
        if not isinstance(rows, list):
            raise ValueError("report_section_must_be_array")
        for row in rows:
            if not isinstance(row, list) or len(row) != width:
                raise ValueError("report_row_array_width_mismatch")
            if not all(isinstance(x, str) for x in row[:width - (section != "sources")]):
                raise ValueError("report_row_field_type_mismatch")
            if section != "sources" and (not isinstance(row[-1], list) or not all(isinstance(x, str) for x in row[-1])):
                raise ValueError("report_source_ids_must_be_array")
    sources = [{"id": r[0], "title": r[1], "url": r[3], "legal_review": "not_reviewed"} for r in raw["sources"]]
    source_ids = [r["id"] for r in sources]
    if len(source_ids) != len(set(source_ids)):
        audit.issue("duplicate_source_id", "report_data")
    has_docx = doc_types is not None
    doc_index = {r["id"]: r for r in (doc_types or [])}
    blocks = blocks or {}
    generic = []
    comparisons = {}
    for section, prefix, count, legal_index in (("types", "Т", 17, 4), ("additional_checks", "Д", 15, 2)):
        rows = raw[section]
        ids = [r[0] for r in rows]
        expected_ids = {f"{prefix}{i:02}" for i in range(1, count + 1)}
        audit.count(len(rows), count, section)
        if len(ids) != len(set(ids)):
            audit.issue("duplicate_taxonomy_id", section)
        comparisons[section] = audit.compare_keys(set(ids), expected_ids, section + "/expected_ids")
        if has_docx:
            audit.compare_keys(set(ids), {i for i in doc_index if i.startswith(prefix)}, section + "/docx_ids")
        for row in rows:
            tid, name, refs = row[0], row[1], row[-1]
            block = whitespace("\n".join(blocks.get(tid, [])))
            doc_refs = sorted(set(re.findall(r"\bИ\d+\b", block)))
            name_matches = tid in doc_index and whitespace(name) == whitespace(doc_index[tid]["name"]) if has_docx else None
            reference_matches = whitespace(row[legal_index]) in block if has_docx else None
            if has_docx and not name_matches:
                audit.issue("taxonomy_name_mismatch", section, id=tid)
            if has_docx and (not reference_matches or set(refs) != set(doc_refs)):
                audit.issue("taxonomy_reference_mismatch", section, id=tid)
            unknown = sorted(set(refs) - set(source_ids))
            if unknown:
                audit.issue("unknown_source_reference", section, id=tid, source_ids=unknown)
            generic.append({"id": tid, "name": name, "source_ids": refs,
                            "docx_source_ids": doc_refs, "name_matches_docx": name_matches,
                            "legal_reference_matches_docx": reference_matches, "legal_review": "not_reviewed"})
    registry = raw["registry"]
    if not isinstance(registry, list):
        raise ValueError("report_registry_must_be_array")
    technical_registry = []
    numeric_ids = []
    for position, row in enumerate(registry, start=1):
        if not isinstance(row, dict) or set(row) != REGISTRY_FIELDS:
            raise ValueError("invalid_report_registry_schema")
        # Only structural types and technical identifiers are inspected here.
        if not (type(row["id"]) is int and isinstance(row["name"], str)
                and isinstance(row["original_label"], str) and isinstance(row["summary"], str)
                and (row["timestamp"] is None or type(row["timestamp"]) in (int, float))
                and (row["assessment"] is None or isinstance(row["assessment"], list))
                and isinstance(row["observation"], dict)):
            audit.issue("invalid_registry_field_type", "report_registry", position=position)
        technical_registry.append({"position": position, "name": row["name"]})
        numeric_ids.append(row["id"])
        try:
            if row["id"] != int(name_key(row["name"]).split("-")[1]):
                audit.issue("registry_numeric_id_mismatch", "report_registry", position=position)
        except ValueError:
            audit.issue("registry_invalid_name", "report_registry", position=position)
    if len(numeric_ids) != len(set(numeric_ids)):
        audit.issue("duplicate_numeric_id", "report_registry")
    audit.count(len(registry), expected, "report_registry")
    report_index = audit.index_names(technical_registry, "report_registry")
    if has_docx:
        audit.count(len(doc_registry), expected, "docx_registry")
        doc_registry_index = audit.index_names(doc_registry, "docx_registry")
        registry_comparison = audit.compare_keys(report_index, doc_registry_index, "registry/docx_vs_json")
    else:
        registry_comparison = {"status": "skipped", "reason": "taxonomy_not_requested"}
    audit.compare_keys(report_index, labels, "registry/labels")
    return {"types_count": len(raw["types"]), "additional_checks_count": len(raw["additional_checks"]),
            "sources": sources, "generic_entries": generic, "id_comparisons": comparisons,
            "registry_comparison": registry_comparison, "registry_ids": sorted(report_index),
            "registry_narrative_exported": False}


def visual_ids(paths, labels, expected, audit):
    combined, counts = [], {}
    for number, item in enumerate(paths, 1):
        where, path = item if isinstance(item, tuple) else ("visual_first" if number == 1 else "visual_second", item)
        rows = read_json(path)
        if not isinstance(rows, list):
            raise ValueError("visual_file_must_be_array")
        counts[where] = len(rows)
        for position, row in enumerate(rows, start=1):
            if not isinstance(row, dict) or set(row) != VISUAL_FIELDS:
                raise ValueError("invalid_visual_schema")
            try:
                if type(row["id"]) is not int or row["id"] != int(name_key(row["name"]).split("-")[1]):
                    audit.issue("visual_numeric_id_mismatch", where, position=position)
            except ValueError:
                audit.issue("invalid_visual_name", where, position=position)
            combined.append({"position": position, "name": row["name"]})
    audit.count(len(combined), expected, "visual_ids")
    index = audit.index_names(combined, "visual_ids")
    comparison = audit.compare_keys(index, labels, "visual_ids/labels")
    return {"counts": counts, "comparison": comparison, "narrative_exported": False}


def static_package_inventory(folder, audit):
    scripts = []
    for path in sorted(folder.glob("*.py")):
        try:
            before = file_identity(path)
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeError):
            audit.issue("package_script_ast_error", path.name, severity="warning")
            continue
        digest = sha256(path)
        if file_identity(path) != before:
            raise InputChanged("input_changed_during_audit", "manifest")
        constants = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
        scripts.append({"file": path.name, "sha256": digest,
                        "absolute_user_path_literals": sum(s.startswith(("/Users/", "/home/")) for s in constants),
                        "parsed_only": True, "executed": False})
    return {"extension_counts": dict(Counter(p.suffix.lower() for p in folder.rglob("*") if p.is_file())),
            "scripts": scripts, "media_opened": False, "scripts_executed": False}


def preload_dependencies(args):
    """Import requested document libraries before hashing even the first input."""
    for module, key in [("openpyxl", "labels"), *([("docx", "taxonomy")] if getattr(args, "taxonomy", None) else [])]:
        try:
            importlib.import_module(module)
        except ImportError as error:
            raise AuditFailure("missing_dependency", key, module) from error


def checked_input(key, operation):
    try:
        return operation()
    except AuditFailure as error:
        if error.input_key == "inputs":
            error.input_key = key
        raise
    except Exception as error:
        raise AuditFailure("invalid_input", key) from error


def directory_snapshot(path):
    return {"directory": file_identity(path),
            "entries": {entry.name: file_identity(entry) for entry in path.iterdir()}}


def skipped(reason="not_requested"):
    return {"status": "skipped", "reason": reason}


def run(args):
    preload_dependencies(args)
    audit = Audit()
    paths = {}
    for name in INPUT_KEYS:
        value = getattr(args, name, None)
        if value is not None:
            paths[name] = checked_input(name, lambda value=value: input_path(value))
    for key in ("labels", "videos"):
        if key not in paths or not (paths[key].is_dir() if key == "videos" else paths[key].is_file()):
            raise AuditFailure("input_missing_or_wrong_kind", key)
    for key in OPTIONAL_KEYS:
        if key in paths and not paths[key].is_file():
            raise AuditFailure("input_missing_or_wrong_kind", key)
    identities = {name: checked_input(name, lambda path=path: file_identity(path))
                  for name, path in paths.items() if path.is_file()}
    initial_videos = checked_input("videos", lambda: directory_snapshot(paths["videos"]))
    initial_package = checked_input("manifest", lambda: directory_snapshot(paths["manifest"].parent)) if "manifest" in paths else None
    input_hashes = {name: checked_input(name, lambda path=path: sha256(path))
                    for name, path in paths.items() if path.is_file()}
    labels = checked_input("labels", lambda: read_labels(paths["labels"], getattr(args, "sheet", None), audit))
    video_rows = []
    unexpected = Counter()
    for entry in sorted(paths["videos"].iterdir()):
        if entry.is_file() and entry.suffix.casefold() == ".mov":
            video_rows.append({"name": entry.name, "path": entry})
        else:
            unexpected[entry.suffix.casefold() if entry.is_file() else "<directory>"] += 1
    for extension, count in sorted(unexpected.items()):
        audit.issue("unexpected_file_in_videos", "videos", severity="warning", extension=extension, count=count)
    manifest = (checked_input("manifest", lambda: manifest_rows(read_json(paths["manifest"]), audit))
                if "manifest" in paths else [])
    for where, rows in (("labels", labels), ("videos", video_rows)):
        audit.count(len(rows), args.expected_count, where)
    if "manifest" in paths:
        audit.count(len(manifest), args.expected_count, "manifest")
    label_index = audit.index_names(labels, "labels")
    video_index = audit.index_names(video_rows, "videos")
    manifest_index = audit.index_names(manifest, "manifest") if "manifest" in paths else {}
    joins = {"labels_videos": audit.compare_keys(label_index, video_index, "labels/videos"),
             "videos_manifest": audit.compare_keys(video_index, manifest_index, "videos/manifest") if "manifest" in paths else skipped()}
    videos = checked_input("videos", lambda: audit_video_files(video_index, label_index, manifest_index, audit))
    sections = {key: {"status": "checked"} if key in paths else skipped() for key in OPTIONAL_KEYS}
    doc_types = blocks = doc_registry = None
    if "taxonomy" in paths:
        doc_types, blocks, doc_registry = checked_input("taxonomy", lambda: read_docx(paths["taxonomy"], audit))
    taxonomy = skipped("report_data_not_requested")
    if "report_data" in paths:
        taxonomy = checked_input("report_data", lambda: taxonomy_audit(read_json(paths["report_data"]), doc_types, blocks,
                                                                      doc_registry, label_index, args.expected_count, audit))
        taxonomy["status"] = "checked"
    elif doc_types is not None:
        for section, prefix, count in (("types", "Т", 17), ("additional_checks", "Д", 15)):
            ids = [row["id"] for row in doc_types if row["id"].startswith(prefix)]
            audit.count(len(ids), count, "docx/" + section)
            audit.compare_keys(set(ids), {f"{prefix}{number:02}" for number in range(1, count + 1)},
                               "docx/" + section + "/expected_ids")
        audit.count(len(doc_registry), args.expected_count, "docx_registry")
        doc_index = audit.index_names(doc_registry, "docx_registry")
        taxonomy = {"status": "checked", "types_count": sum(r["id"].startswith("Т") for r in doc_types),
                    "additional_checks_count": sum(r["id"].startswith("Д") for r in doc_types),
                    "registry_comparison": audit.compare_keys(doc_index, label_index, "docx_registry/labels"),
                    "generic_entries": [], "json_comparison": skipped("report_data_not_requested")}
    visual_paths = [(key, paths[key]) for key in ("visual_first", "visual_second") if key in paths]
    visual = checked_input("visual_first" if "visual_first" in paths else "visual_second",
                           lambda: visual_ids(visual_paths, label_index, args.expected_count, audit)) if visual_paths else skipped()
    if visual_paths:
        visual["status"] = "checked"
    package = checked_input("manifest", lambda: static_package_inventory(paths["manifest"].parent, audit)) if "manifest" in paths else skipped()
    for name, original_hash in input_hashes.items():
        try:
            changed = file_identity(paths[name]) != identities[name] or sha256(paths[name]) != original_hash
        except (OSError, InputChanged):
            changed = True
        if changed:
            audit.issue("input_changed_during_audit", name)
    try:
        videos_changed = directory_snapshot(paths["videos"]) != initial_videos
    except OSError:
        videos_changed = True
    if videos_changed:
        audit.issue("input_changed_during_audit", "videos")
    if initial_package is not None:
        try:
            package_changed = directory_snapshot(paths["manifest"].parent) != initial_package
        except OSError:
            package_changed = True
        if package_changed:
            audit.issue("input_changed_during_audit", "manifest")
    summary = {"label_rows": len(labels), "video_files": len(video_rows),
               "manifest_rows": len(manifest) if "manifest" in paths else None,
               "unique_video_contents": len({r["sha256"] for r in videos if r["sha256"] is not None}),
               "labels": dict(Counter(r["label"] or "unknown" for r in labels)),
               "hash_checks": dict(Counter(r["manifest_hash_check"] for r in videos)),
               "size_checks": dict(Counter(r["manifest_size_check"] for r in videos)),
               "duration_verification": dict(Counter(r["duration_verification"] for r in videos)),
               "timestamp_bounds": dict(Counter(r["timestamp_bounds"] for r in videos)),
               "video_filename_alphabets": dict(Counter("Cyrillic_Х" if "Х" in nfc(r["name"]) else "Latin_X" for r in video_rows)),
               "unexpected_video_extensions": dict(unexpected),
               "types": taxonomy.get("types_count"), "additional_checks": taxonomy.get("additional_checks_count"),
               "registry_ids_matched": taxonomy.get("registry_comparison", {}).get("matched"),
               "generic_names_matched": sum(r["name_matches_docx"] is True for r in taxonomy.get("generic_entries", [])),
               "generic_legal_refs_matched": sum(r["legal_reference_matches_docx"] is True for r in taxonomy.get("generic_entries", [])),
               "errors": sum(i["severity"] == "error" for i in audit.issues),
               "warnings": sum(i["severity"] == "warning" for i in audit.issues),
               "issue_codes": dict(Counter(i["code"] for i in audit.issues))}
    return {"schema_version": 1, "audited_at_utc": datetime.now(timezone.utc).isoformat(),
            "inputs": {key: {"path": path_label(paths[key]), "sha256": input_hashes.get(key), "status": "checked"}
                       if key in paths else skipped() for key in INPUT_KEYS},
            "sections": sections, "summary": summary, "name_checks": audit.name_checks, "joins": joins,
            "label_rows": labels, "videos": videos, "taxonomy": taxonomy,
            "visual_files_structure": visual, "package_static": package, "issues": audit.issues,
            "limitations": ["No video, audio or frames decoded; no visual validation.",
                            "Duration is optional author manifest metadata, never measured by this audit.",
                            "Author taxonomy and legal references are not reviewed ground truth.",
                            "Approximate timestamps use the shared ml.labels loader, not visual refinement."]}


def validate_output(value, input_paths=()):
    root = ROOT.resolve()
    candidate = Path(value)
    if ".." in candidate.parts:
        raise AuditFailure("invalid_output", "output")
    output = candidate if candidate.is_absolute() else root / candidate
    allowed = root / "data" / "audit"
    if not output.is_relative_to(allowed) or output.suffix != ".json":
        raise AuditFailure("invalid_output", "output")
    current = root
    for part in output.relative_to(root).parts:
        current = current / part
        if current.is_symlink():
            raise AuditFailure("output_symlink", "output")
    if output.exists() and not output.is_file():
        raise AuditFailure("invalid_output", "output")
    for source in input_paths:
        if output.resolve() == source.resolve() or (source.is_dir() and output.resolve().is_relative_to(source.resolve())):
            raise AuditFailure("output_overlaps_input", "output")
        if output.exists() and source.is_file() and os.path.samestat(output.stat(), source.stat()):
            raise AuditFailure("output_overlaps_input", "output")
    return output


def write_report(output, result, input_paths):
    """A unique same-directory temporary file; never follow a report/parent symlink."""
    output = validate_output(output, input_paths)
    output.parent.mkdir(parents=True, exist_ok=True)
    output = validate_output(output, input_paths)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                         prefix=".audit-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        validate_output(output, input_paths)
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise AuditFailure("invalid_arguments", "arguments")


def main(argv=None):
    parser = SafeArgumentParser(prog="audit_dataset.py", description=__doc__)
    defaults = {"labels": "data/open/open_labels.xlsx", "videos": "data/open/videos",
                **{key: None for key in OPTIONAL_KEYS}}
    for name, default in defaults.items():
        parser.add_argument("--" + name.replace("_", "-"), default=default)
    parser.add_argument("--sheet", default=None, help="Only the shared loader's active worksheet is supported")
    parser.add_argument("--expected-count", type=int, default=42)
    parser.add_argument("--output", default="data/audit/intake.json")
    try:
        args = parser.parse_args(argv)
        if args.expected_count <= 0:
            raise AuditFailure("invalid_expected_count", "arguments")
        output = validate_output(args.output)
        # Preflight before hashes: dependency failures and input aliases are cheap.
        preload_dependencies(args)
        requested = [checked_input(name, lambda name=name: input_path(getattr(args, name)))
                     for name in INPUT_KEYS if getattr(args, name) is not None]
        validate_output(output, requested)
        result = run(args)
        write_report(output, result, requested)
    except SystemExit as error:
        return int(error.code)
    except AuditFailure as error:
        module = f" module={error.module}" if error.module else ""
        print(f"AUDIT FAILED: input={error.input_key} code={error.code}{module}", file=sys.stderr)
        return 2
    except Exception:
        print("AUDIT FAILED: input=inputs code=audit_failed", file=sys.stderr)
        return 2
    print(json.dumps({"summary": result["summary"],
                      "sections": {key: value["status"] for key, value in result["sections"].items()}},
                     ensure_ascii=False, indent=2))
    return 1 if result["summary"]["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
