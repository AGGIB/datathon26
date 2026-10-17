"""Synthetic bytes/labels exercise gates; no fake benchmark or media decoding."""

import copy
import hashlib

import pytest

from ml.tools import angle_holdout as holdout


@pytest.fixture
def manifests(tmp_path):
    def row(name, payload, angle, source, positive=1):
        path = tmp_path / (name + ".mp4")
        path.write_bytes(payload)
        return {"video_id": name, "path": str(path), "content_sha256": hashlib.sha256(payload).hexdigest(),
                "angle_id": angle, "source_group": source, "label": positive,
                "origin": "locally_recorded", "rights_confirmed": True,
                "events": [{"type_id": "safe_left_open", "t_peak": 10.0}] if positive else []}

    development = {"schema_version": 1, "development_used_for_tuning": True,
                   "records": [row("old", b"development fake bytes", "camera_A", "recording_old")]}
    test = {"schema_version": 1, "annotation_origin": "human", "reviewed_by": "Synthetic reviewer",
            "reviewed_at": "2026-10-17", "events_exhaustive": True,
            "records": [row("new", b"new fake bytes", "camera_B", "recording_new")]}
    return development, test


def test_missing_new_footage_returns_blocked_without_reading_development(manifests, monkeypatch):
    monkeypatch.setattr(holdout, "file_sha256", lambda _: pytest.fail("no footage supplied"))
    result = holdout.prepare(manifests[0])
    assert result["status"] == "blocked" and result["new_angle_available"] is False
    assert result["blocking_codes"] == ["new_angle_manifest_missing"]
    with pytest.raises(ValueError, match="not_ready"):
        holdout.score(result, [])


def test_unknown_development_angles_cannot_become_a_validated_holdout(manifests):
    development, test = manifests
    development["records"][0]["angle_id"] = "unknown"
    result = holdout.prepare(development, test)
    assert result["status"] == "blocked"
    assert "explicit_development_and_test_angles_required" in result["blocking_codes"]


def test_renamed_old_video_cannot_be_reported_as_unseen(manifests):
    development, test = manifests
    old, new = development["records"][0], test["records"][0]
    new["path"] = old["path"]
    new["content_sha256"] = old["content_sha256"]
    result = holdout.prepare(development, test)
    assert result["status"] == "blocked"
    assert "development_content_reused_as_test" in result["blocking_codes"]


@pytest.mark.parametrize("field,code", [("source_group", "source_recording_crosses_partitions"),
                                        ("angle_id", "camera_angle_crosses_partitions")])
def test_related_recordings_and_angle_case_aliases_are_isolated(manifests, field, code):
    development, test = manifests
    test["records"][0][field] = development["records"][0][field].swapcase()
    assert code in holdout.prepare(development, test)["blocking_codes"]


@pytest.mark.parametrize("origin", ["external", "synthetic_perspective", "quarantine"])
def test_external_or_synthetic_data_never_becomes_locally_shot_gold(manifests, monkeypatch, origin):
    development, test = manifests
    test["records"][0]["origin"] = origin
    monkeypatch.setattr(holdout, "file_sha256", lambda _: pytest.fail("unapproved media read"))
    result = holdout.prepare(development, test)
    assert result["status"] == "blocked"
    assert "local_recording_and_rights_required" in result["blocking_codes"]


def test_declared_hash_is_checked_against_real_source_bytes(manifests):
    development, test = manifests
    test["records"][0]["content_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="actual_bytes"):
        holdout.prepare(development, test)


@pytest.mark.parametrize("field,value,code", [("annotation_origin", "local_model", "human_review_required"),
                                            ("events_exhaustive", False, "exhaustive_event_inventory_required")])
def test_machine_or_primary_only_annotations_do_not_enable_event_metrics(manifests, field, value, code):
    development, test = manifests
    test[field] = value
    assert code in holdout.prepare(development, test)["blocking_codes"]


def test_score_rechecks_source_hashes_after_preparation(manifests):
    prepared = holdout.prepare(*manifests)
    from pathlib import Path

    Path(prepared["media_paths"]["new"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="media_changed"):
        holdout.score(prepared, [])


def test_joint_type_and_time_metrics_do_not_credit_different_predictions(manifests):
    prepared = holdout.prepare(*manifests)
    predictions = [{"video_id": "new", "status": "done", "verdict": "violation", "violations": [
        {"type_id": "door_left_open", "timecode_sec": 10.0},
        {"type_id": "safe_left_open", "timecode_sec": 30.0}]}]
    row = holdout.score(prepared, predictions)["independent_angle_events"]["camera_B"]
    assert row["type_only_matches"] == row["time_only_matches"] == 1
    assert row["joint_type_time_matches"] == 0
    assert row["joint_precision"] == row["joint_recall"] == 0


def test_joint_matching_uses_the_correct_type_even_with_a_closer_wrong_candidate(manifests):
    prepared = holdout.prepare(*manifests)
    prediction = [{"video_id": "new", "status": "done", "verdict": "violation", "violations": [
        {"type_id": "door_left_open", "timecode_sec": 10.0},
        {"type_id": "safe_left_open", "timecode_sec": 11.0}]}]
    row = holdout.score(prepared, prediction)["independent_angle_events"]["camera_B"]
    assert row["joint_type_time_matches"] == 1 and row["predicted_events"] == 2
    assert row["joint_precision"] == 0.5


def test_duplicate_contents_are_one_unit_and_missing_predictions_are_not_clean(manifests):
    development, test = manifests
    duplicate = copy.deepcopy(test["records"][0])
    duplicate["video_id"] = "new_alias"
    test["records"].append(duplicate)
    prepared = holdout.prepare(development, test)
    report = holdout.score(prepared, [])
    row = report["independent_angle_events"]["camera_B"]
    assert row["unique_contents"] == 1 and row["truth_events"] == 1
    assert row["failed_or_missing"] == 1 and row["joint_recall"] == 0
    assert row["end_to_end_accuracy"] == 0


def test_quarantine_path_is_rejected_before_hashing(manifests, monkeypatch):
    development, test = manifests
    test["records"][0]["path"] = "/private/tmp/QuArAnTiNe/clip.mp4"
    real = holdout.file_sha256
    monkeypatch.setattr(holdout, "file_sha256", lambda path: real(path) if str(path).endswith("old.mp4")
                        else pytest.fail("quarantined media hashed"))
    with pytest.raises(ValueError, match="excluded_media"):
        holdout.prepare(development, test)


def test_freeze_preserves_unknown_angles_and_records_actual_hashes(tmp_path):
    video = tmp_path / "old.mp4"
    video.write_bytes(b"synthetic development fixture")
    labels = tmp_path / "labels.csv"
    labels.write_text("video,timestamp,label\nold,5,нарушение\n", encoding="utf-8")
    annotations = tmp_path / "annotations.csv"
    annotations.write_text("video,angle_id\nold,\n", encoding="utf-8")
    frozen = holdout.freeze_development(tmp_path, labels, annotations)
    [record] = frozen["records"]
    assert frozen["development_used_for_tuning"] is True
    assert record["content_sha256"] == hashlib.sha256(video.read_bytes()).hexdigest()
    assert record["angle_id"] == record["source_group"] == "unknown"


def test_cli_missing_new_angle_writes_blocked_status_without_metrics(manifests, tmp_path, monkeypatch, capsys):
    import json

    monkeypatch.setattr(holdout, "ROOT", tmp_path)
    development = tmp_path / "development.json"
    development.write_text(json.dumps(manifests[0]))
    output = tmp_path / "var/readiness.json"
    assert holdout.main(["prepare", "--development", str(development), "--output", str(output)]) == 1
    assert json.loads(output.read_text())["status"] == "blocked"
    assert "new_angle_manifest_missing" in capsys.readouterr().out


def test_output_cannot_overwrite_a_manifest(manifests, tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(holdout, "ROOT", tmp_path)
    development = tmp_path / "var/development.json"
    development.parent.mkdir()
    original = json.dumps(manifests[0])
    development.write_text(original)
    assert holdout.main(["prepare", "--development", str(development), "--output", str(development)]) == 2
    assert development.read_text() == original
