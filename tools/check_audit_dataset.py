#!/usr/bin/env python3
"""Small synthetic checks. No real dataset or media decoder required."""

import hashlib
from pathlib import Path
import tempfile

from audit_dataset import Audit, ROOT, audit_video_files, name_key, nfc, parse_timestamp, static_package_inventory, taxonomy_audit


def main():
    examples = [(None, None), ("", None), (0, 0), ("8 секунда", 8), ("8 секунд", 8),
                ("1минута 8 секунда", 68), ("2 минуты 3 секунды", 123),
                ("1мин.8сек.", 68), ("2 минуты", 120), ("01:08", 68),
                ("1,5 секунды", 1.5), ("68", 68), ("8\u00a0секунд", 8)]
    for raw, expected in examples:
        assert parse_timestamp(raw) == expected
    for raw in (True, -1, float("nan"), float("inf"), "-1 секунда", "1:60", "1 минута 60 секунд", "около 8 секунд", "8 секунд текст"):
        try:
            parse_timestamp(raw)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid timestamp accepted")

    assert nfc("И\u0306") == "Й"
    assert name_key("Х-7.mov") == name_key("X-7")
    audit = Audit()
    result = audit.index_names([{"name": "X-7.mov"}, {"name": "Х-7.mov"}], "synthetic")
    assert result == {} and audit.issues[0]["code"] == "normalized_name_collision"
    duplicate = Audit()
    assert duplicate.index_names([{"name": "Х-7"}, {"name": "Х-7"}], "synthetic") == {}
    assert duplicate.issues[0]["code"] == "exact_duplicate"
    unique = Audit()
    assert set(unique.index_names([{"name": "X-7"}, {"name": "Х-8"}], "synthetic")) == {"x-7", "x-8"}

    parent = ROOT / "data/audit"
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="synthetic-", dir=parent) as temp:
        folder = Path(temp)
        path = folder / "X-7.mov"
        payload = b"synthetic bytes, deliberately not a video"
        path.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        videos = {"x-7": {"name": path.name, "path": path}}
        labels = {"x-7": {"name": "Х-7", "label": "violation", "timestamp_seconds": 8}}
        source = {"name": "X-7", "sha256": digest, "duration": 10,
                  "size_bytes": len(payload), "label": "нарушение", "timestamp": 8}
        ok = audit_video_files(videos, labels, {"x-7": source}, Audit())[0]
        assert ok["timestamp_bounds"] == "within_manifest_duration"
        assert not ok["duration_measured_by_audit"]
        bad_hash = dict(source, sha256="0" * 64, duration=2)
        bad = audit_video_files(videos, labels, {"x-7": bad_hash}, Audit())[0]
        assert bad["duration_verification"] == "unverified_hash_mismatch"
        assert bad["timestamp_bounds"] == "unverified"
        out_of_bounds = Audit()
        audit_video_files(videos, labels, {"x-7": dict(source, duration=8)}, out_of_bounds)
        assert any(i["code"] == "timestamp_out_of_bounds" for i in out_of_bounds.issues)
        bad_size = Audit()
        audit_video_files(videos, labels, {"x-7": dict(source, size_bytes=0)}, bad_size)
        assert any(i["code"] == "size_mismatch" for i in bad_size.issues)
        assert path.read_bytes() == payload

        second = folder / "Х-8.mov"
        second.write_bytes(payload)
        duplicates = dict(videos, **{"x-8": {"name": second.name, "path": second}})
        conflicting_labels = dict(labels, **{"x-8": {"name": "Х-8", "label": "no_violation", "timestamp_seconds": None}})
        conflicts = Audit()
        audit_video_files(duplicates, conflicting_labels, {"x-7": source}, conflicts)
        assert {"duplicate_content_multiple_timestamps", "duplicate_content_label_conflict"} <= {i["code"] for i in conflicts.issues}
        conflicting_times = dict(labels, **{"x-8": {"name": "Х-8", "label": "violation", "timestamp_seconds": 9,
                                                     "timestamp_original": "9 секунд"}})
        timestamp_conflict = Audit()
        audit_video_files(duplicates, conflicting_times, {"x-7": source}, timestamp_conflict)
        issue = next(i for i in timestamp_conflict.issues if i["code"] == "duplicate_content_multiple_timestamps")
        assert {a["timestamp_seconds"] for a in issue["annotations"]} == {8, 9}
        assert not any(i["code"] == "duplicate_content_label_conflict" for i in timestamp_conflict.issues)

        # Parsing must not execute even a script whose top level writes a file.
        (folder / "foreign.py").write_text("from pathlib import Path\nPath('DO_NOT_CREATE').write_text('bad')\n")
        inventory = static_package_inventory(folder, Audit())
        assert not inventory["scripts_executed"]
        assert not (folder / "DO_NOT_CREATE").exists() and not (Path.cwd() / "DO_NOT_CREATE").exists()

    invalid_report = {"sources": [], "types": [{}], "additional_checks": [], "registry": []}
    try:
        taxonomy_audit(invalid_report, [], {}, [], {}, 42, Audit())
    except ValueError as error:
        assert str(error) == "report_row_array_width_mismatch"
    else:
        raise AssertionError("dict accepted where row array is required")
    print("OK: synthetic timestamp parsing, collision refusal, hash-gated bounds, sizes, strict arrays, static-only parsing.")


if __name__ == "__main__":
    main()
