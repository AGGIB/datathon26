"""Synthetic audit/CLI fixtures only: no real models, media or observations."""
import csv
import hashlib
import importlib
import importlib.util
import json
import stat
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest

from ml.baseline.core import write_json
from ml.baseline.dataset import snapshot_fingerprint
from ml.baseline.features import save_cache


def io():
    assert importlib.util.find_spec("ml.baseline.diagnostic_io"), "diagnostic audit module missing"
    return importlib.import_module("ml.baseline.diagnostic_io")


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    videos = tmp_path / "videos"
    videos.mkdir()
    caches = tmp_path / "artifacts" / "caches"
    caches.mkdir(parents=True)
    labels = tmp_path / "labels.csv"
    labels.write_text("video,timestamp,label\nA.mov,1,нарушение\nB.mov,2,нарушение\nC.mov,,без нарушения\n", encoding="utf-8")
    config = {"fps": 1.0, "max_frames": 100, "batch_size": 1,
              "device": "cpu", "cpu_threads": 2, "seed": 0}
    records, manifest = [], []
    for name, data, label in [("A.mov", b"SYNTHETIC-A", 1), ("B.mov", b"SYNTHETIC-A", 1),
                              ("C.mov", b"SYNTHETIC-C", 0)]:
        (videos / name).write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        cache_file = hashlib.sha256(name.encode()).hexdigest() + ".npz"
        extraction = {"fps_requested": 1.0, "max_frames": 100, "batch_size": 1,
                      "n_frames": 2, "cap_reached": False, "autorotate": True,
                      "time_origin": "container-relative PTS: copyts + start_at_zero",
                      "device": "cpu", "seed": 0}
        save_cache(caches / cache_file, name, [0.0, 2.0], np.array([[1, 3], [3, 5]]) + label,
                   {"content_sha256": digest, "snapshot_sha256": "a" * 64,
                    "extraction": extraction, "source_filename": name,
                    "source_size_bytes": len(data)})
        records.append({"video_id": name, "status": "ok", "content_sha256": digest,
                        "cache_file": cache_file, "n_frames": 2, "embedding_dim": 2,
                        "cap_reached": False})
        manifest.append({"video_id": name, "label": label, "angle_id": "unknown",
                         "content_sha256": digest, "positive_times_s": [1.0] if label else []})
    write_json(caches / "index.json", {"schema_version": 1,
               "expected_videos": 3, "successful_videos": 3, "failed_videos": 0,
               "capped_videos": 0, "unique_contents": 2, "encoded_contents": 2,
               "snapshot_sha256": "a" * 64, "extraction_config": config, "videos": records})
    manifest_path = tmp_path / "manifest.json"
    write_json(manifest_path, manifest)
    (tmp_path / "synthetic-weights").mkdir()
    # Public snapshot boundary is already independently tested; this fixture has no weights.
    from ml.baseline import dataset
    monkeypatch.setattr(dataset, "snapshot_fingerprint", lambda _: "a" * 64)
    return {"index_path": caches / "index.json", "videos_dir": videos, "labels_path": labels,
            "manifest_path": manifest_path, "weights": tmp_path / "synthetic-weights"}


def test_actual_hash_audit_deduplicates_and_pools_every_frame_without_anchors(inputs):
    rows, audit = io().audit_inputs(**inputs)
    assert len(rows) == 2
    assert audit["videos_verified"] == 3 and audit["unique_contents"] == 2
    assert audit["class_counts"] == [1, 1]
    assert audit["total_cached_frames"] == 6
    positive = next(r for r in rows if r["label"] == 1)
    assert positive["members"] == ["A.mov", "B.mov"]
    np.testing.assert_array_equal(positive["vector"], [3, 5])
    assert positive["source_groups"] == []
    assert all("positive_times_s" not in r and "timestamp" not in r for r in rows)
    assert audit["alternative_anchors_by_content"][positive["content_sha256"]] == [1.0, 2.0]
    json.dumps(audit, allow_nan=False)


@pytest.mark.parametrize("damage", ["source", "manifest_label", "manifest_hash", "snapshot", "cap",
                                   "totals", "cache_path", "dimensions", "duplicate_features"])
def test_audit_rejects_untrusted_or_inconsistent_inputs(inputs, damage):
    module = io()
    index = json.loads(inputs["index_path"].read_text())
    manifest = json.loads(inputs["manifest_path"].read_text())
    if damage == "source":
        (inputs["videos_dir"] / "A.mov").write_bytes(b"CHANGED SYNTHETIC SOURCE")
    elif damage.startswith("manifest"):
        manifest[0]["label" if damage == "manifest_label" else "content_sha256"] = 0 if damage == "manifest_label" else "b" * 64
        write_json(inputs["manifest_path"], manifest)
    elif damage == "snapshot":
        index["snapshot_sha256"] = "b" * 64
    elif damage == "cap":
        index["videos"][0]["cap_reached"] = True
    elif damage == "totals":
        index["successful_videos"] = 2
    elif damage == "cache_path":
        index["videos"][0]["cache_file"] = "../private.npz"
    elif damage == "dimensions":
        index["videos"][0]["embedding_dim"] = 42
    else:
        from ml.baseline.features import load_cache
        path = inputs["index_path"].parent / index["videos"][1]["cache_file"]
        cache = load_cache(path)
        save_cache(path, cache["video_id"], cache["timestamps_s"], cache["embeddings"] + 1, cache["metadata"])
    write_json(inputs["index_path"], index)
    with pytest.raises(ValueError):
        module.audit_inputs(**inputs)


@pytest.mark.parametrize("damage", ["all_wrong", "contradictory_duplicate", "other_content"])
def test_declared_label_hash_must_match_its_cache_and_manifest(inputs, damage):
    manifest = json.loads(inputs["manifest_path"].read_text())
    declared = [row["content_sha256"] for row in manifest]
    if damage == "all_wrong":
        declared = ["f" * 64] * len(manifest)
    elif damage == "contradictory_duplicate":
        declared[1] = "f" * 64  # A and B have identical actual bytes, not different provenance.
    else:
        declared[0], declared[2] = declared[2], declared[0]
    lines = inputs["labels_path"].read_text(encoding="utf-8").splitlines()
    inputs["labels_path"].write_text(lines[0] + ",content_sha256\n" + "\n".join(
        line + "," + digest for line, digest in zip(lines[1:], declared)) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="content|SHA256|hash"):
        io().audit_inputs(**inputs)


@pytest.mark.parametrize("declared", [False, True])
def test_correct_optional_label_hashes_preserve_literal_source_joins(inputs, declared):
    manifest = json.loads(inputs["manifest_path"].read_text())
    lines = inputs["labels_path"].read_text(encoding="utf-8").splitlines()
    # Canonical joining must not change the literal IDs used to locate caches/source files.
    body = [line.replace(row["video_id"], row["video_id"].lower())
            for line, row in zip(lines[1:], manifest)]
    if declared:
        body = [line + "," + row["content_sha256"] for line, row in zip(body, manifest)]
    inputs["labels_path"].write_text(lines[0] + (",content_sha256" if declared else "") + "\n"
                                    + "\n".join(body) + "\n", encoding="utf-8")
    rows, audit = io().audit_inputs(**inputs)
    assert audit["videos_verified"] == 3 and len(rows) == 2
    assert next(row for row in rows if row["label"])["members"] == ["A.mov", "B.mov"]
    assert {row["video_id"] for row in audit["cache_records"]} == {"A.mov", "B.mov", "C.mov"}
    for row in audit["cache_records"]:
        assert row["content_sha256"] == hashlib.sha256(
            (inputs["videos_dir"] / row["video_id"]).read_bytes()).hexdigest()
        assert row["hash_status"] == "matched_actual_source_cache_manifest"


def test_frozen_cv_requires_actual_content_and_cannot_choose_successful_duplicate(inputs):
    module = io()
    _, audit = module.audit_inputs(**inputs)
    records = [{"video": r["video_id"], "content_sha256": r["content_sha256"],
                "status": "ok", "verdict": "violation", "violations": []}
               for r in audit["cache_records"]]
    records[1] = dict(records[1], status="failed")
    path = inputs["index_path"].parent / "frozen.json"
    write_json(path, {"schema_version": 1, "producer_revision": "b" * 40,
                      "development_tuned": True, "records": records})
    cv, provenance = module.load_frozen_cv(path, audit)
    duplicate = next(r["content_sha256"] for r in audit["cache_records"] if r["video_id"] == "A.mov")
    assert cv[duplicate] == {"status": "failed", "decision": None}
    assert provenance["member_status_counts"] == {"ok": 2, "failed": 1}
    records[0]["content_sha256"] = "c" * 64
    write_json(path, {"schema_version": 1, "producer_revision": "b" * 40,
                      "development_tuned": True, "records": records})
    with pytest.raises(ValueError, match="content"):
        module.load_frozen_cv(path, audit)


def test_runner_freezes_files_before_fit_reserves_output_and_reports_missing_cv(inputs, monkeypatch):
    assert importlib.util.find_spec("ml.baseline.development"), "development CLI missing"
    cli = importlib.import_module("ml.baseline.development")
    from ml.baseline import diagnostic
    output = inputs["index_path"].parents[1] / "diagnostic"
    real_run = diagnostic.run_diagnostic
    called = []

    def inspect(rows, plan, **kwargs):
        assert json.loads((output / "folds.json").read_text())["manifest_sha256"] == plan["manifest_sha256"]
        assert (output / "cache-audit.json").is_file()
        assert (output / "input-manifest.json").is_file()
        called.append(True)
        return real_run(rows, plan, **kwargs)

    monkeypatch.setattr(diagnostic, "run_diagnostic", inspect)
    report = cli.run_experiment(**inputs, output=output, artifact_root=output.parents[1])
    assert called == [True]
    assert report["comparison_provenance"]["status"] == "missing"
    assert report["individual"]["frozen_cv"]["coverage"]["evaluated"] == 0
    before = (output / "report.json").read_bytes()
    with pytest.raises(FileExistsError):
        cli.run_experiment(**inputs, output=output, artifact_root=output.parents[1])
    assert (output / "report.json").read_bytes() == before


@pytest.mark.parametrize("kind", ["dangling_symlink", "symlink", "reparse", "file", "directory"])
def test_existing_output_is_rejected_without_resolution_audit_or_writes(tmp_path, monkeypatch, kind):
    from ml.baseline import development

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    output, destination = artifacts / "existing-output", artifacts / "fresh-target"
    if kind in {"symlink", "directory"}:
        (destination if kind == "symlink" else output).mkdir()
    if "symlink" in kind:
        try:
            output.symlink_to(destination, target_is_directory=True)
        except OSError as exc:
            if getattr(exc, "winerror", None) == 1314:
                pytest.skip("Windows symlink creation privilege unavailable")
            raise
    elif kind == "file":
        output.write_bytes(b"SYNTHETIC EXISTING OUTPUT")
    elif kind == "reparse":
        original_lstat = Path.lstat

        def lstat(path, *args, **kwargs):
            if path == output:
                return SimpleNamespace(st_mode=stat.S_IFDIR,
                                       st_file_attributes=stat.FILE_ATTRIBUTE_REPARSE_POINT)
            return original_lstat(path, *args, **kwargs)

        monkeypatch.setattr(Path, "lstat", lstat)
    calls = []
    original_artifact_path = development.artifact_path

    def resolve(*args, **kwargs):
        calls.append("resolve")
        return original_artifact_path(*args, **kwargs)

    def audit(*args, **kwargs):
        calls.append("audit")
        raise ValueError("SYNTHETIC AUDIT MUST NOT BE REACHED")

    monkeypatch.setattr(development, "artifact_path", resolve)
    monkeypatch.setattr(io(), "audit_inputs", audit)
    monkeypatch.setattr(development, "_atomic_json", lambda *args: calls.append("write"))
    with pytest.raises((ValueError, FileExistsError)):
        development.run_experiment(None, None, None, None, None, output, artifact_root=tmp_path)
    assert calls == []
    if kind == "symlink":
        assert list(destination.iterdir()) == []
    else:
        assert not destination.exists()
    if kind == "file":
        assert output.read_bytes() == b"SYNTHETIC EXISTING OUTPUT"
    elif kind == "directory":
        assert list(output.iterdir()) == []


def test_runner_failure_is_class_only_and_not_a_negative_result(inputs):
    assert importlib.util.find_spec("ml.baseline.development"), "development CLI missing"
    cli = importlib.import_module("ml.baseline.development")
    output = inputs["index_path"].parents[1] / "failed-diagnostic"
    (inputs["videos_dir"] / "A.mov").write_bytes(b"CHANGED")
    with pytest.raises(ValueError):
        cli.run_experiment(**inputs, output=output, artifact_root=output.parents[1])
    failure = json.loads((output / "failure.json").read_text())
    assert failure["status"] == "failed" and failure["error_class"] == "ValueError"
    assert not (output / "report.json").exists()
    assert "A.mov" not in json.dumps(failure)


@pytest.mark.parametrize("extraction", [None, [], "SYNTHETIC PRIVATE METADATA"])
def test_cli_nonobject_extraction_is_a_class_only_failure(inputs, monkeypatch, capsys, extraction):
    from ml import eval_runner
    from ml.baseline import development, diagnostic
    from ml.baseline.features import load_cache

    index = json.loads(inputs["index_path"].read_text())
    path = inputs["index_path"].parent / index["videos"][0]["cache_file"]
    cache = load_cache(path)
    save_cache(path, cache["video_id"], cache["timestamps_s"], cache["embeddings"],
               dict(cache["metadata"], extraction=extraction))
    output = inputs["index_path"].parents[1] / "malformed-diagnostic"
    original = development.run_experiment

    def run(**kwargs):
        return original(**kwargs, artifact_root=output.parents[1])

    def forbidden(*args, **kwargs):
        pytest.fail("Malformed extraction must fail before fitting or emitting negative results")

    monkeypatch.setattr(development, "run_experiment", run)
    monkeypatch.setattr(diagnostic, "run_diagnostic", forbidden)
    monkeypatch.setattr(eval_runner, "_deny_network", lambda: None)
    argv = [part for key, value in dict(inputs, output=output).items()
            for part in ("--" + key.replace("_", "-"), str(value))]
    assert development.main(argv) == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"status": "failed", "error_class": "ValueError"}
    assert captured.err == ""
    failure = json.loads((output / "failure.json").read_text())
    assert failure == {"report_kind": "development_diagnostic_failure", **io().FLAGS,
                       "status": "failed", "error_class": "ValueError"}
    assert {p.name for p in output.iterdir()} == {"failure.json"}


def _assert_cli_loader_failure(inputs, monkeypatch, capsys):
    from ml import eval_runner
    from ml.baseline import development, diagnostic

    output = inputs["index_path"].parents[1] / "malformed-loader-diagnostic"
    original = development.run_experiment

    def run(**kwargs):
        return original(**kwargs, artifact_root=output.parents[1])

    def forbidden(*args, **kwargs):
        pytest.fail("Malformed loader input must fail before planning, fitting or loading a model")

    monkeypatch.setattr(development, "run_experiment", run)
    monkeypatch.setattr(diagnostic, "make_plan", forbidden)
    monkeypatch.setattr(diagnostic, "run_diagnostic", forbidden)
    monkeypatch.setattr(io().features, "SiglipEncoder", forbidden)
    monkeypatch.setattr(eval_runner, "_deny_network", lambda: None)
    argv = [part for key, value in dict(inputs, output=output).items()
            for part in ("--" + key.replace("_", "-"), str(value))]
    assert development.main(argv) == 1
    captured = capsys.readouterr()
    assert captured.out == '{"status": "failed", "error_class": "ValueError"}\n'
    assert captured.err == ""
    assert json.loads((output / "failure.json").read_text()) == {
        "report_kind": "development_diagnostic_failure", **io().FLAGS,
        "status": "failed", "error_class": "ValueError"}
    assert {path.name for path in output.iterdir()} == {"failure.json"}


@pytest.mark.parametrize("config", [[], {"model_type": "siglip", "vision_config": None}],
                         ids=["nonobject-config", "null-vision-config"])
def test_cli_real_snapshot_malformed_config_is_a_class_only_failure(inputs, monkeypatch, capsys, config):
    # Restore the real public loader: the shared fixture's fingerprint stub hides this failure.
    monkeypatch.setattr(io().dataset, "snapshot_fingerprint", snapshot_fingerprint)
    write_json(inputs["weights"] / "config.json", config)
    write_json(inputs["weights"] / "preprocessor_config.json", {})
    (inputs["weights"] / "model.safetensors").write_bytes(b"SYNTHETIC CHECKPOINT, NOT A MODEL")
    _assert_cli_loader_failure(inputs, monkeypatch, capsys)


def test_cli_real_nonzip_xlsx_is_a_class_only_failure(inputs, monkeypatch, capsys):
    labels = inputs["labels_path"].with_suffix(".xlsx")
    labels.write_bytes(b"SYNTHETIC NON-ZIP LABEL TABLE")
    _assert_cli_loader_failure(dict(inputs, labels_path=labels), monkeypatch, capsys)


@pytest.mark.parametrize("damage", ["csv-field-too-large", "malformed-xml", "missing-member",
                                   "missing-target", "shared-index", "shared-value",
                                   "missing-inline-string", "corrupt-deflate"])
def test_cli_real_label_parser_errors_are_class_only_failures(inputs, monkeypatch, capsys, damage):
    if damage == "csv-field-too-large":
        labels = inputs["labels_path"]
        labels.write_text("video,timestamp,label\n" + "x" * (csv.field_size_limit() + 1)
                          + ",,без нарушения\n", encoding="utf-8")
    else:
        labels = inputs["labels_path"].with_suffix(".xlsx")
        workbook = ('<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
                    'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
                    '<sheets><sheet name="labels" sheetId="1" r:id="r1"/></sheets></workbook>')
        target = '' if damage == "missing-target" else ' Target="worksheets/sheet1.xml"'
        cell = {"shared-index": '<c r="A1" t="s"><v>0</v></c>',
                "shared-value": '<c r="A1" t="s"><v/></c>',
                "missing-inline-string": '<c r="A1" t="inlineStr"/>'}.get(damage, '')
        with ZipFile(labels, "w", compression=ZIP_DEFLATED) as archive:
            if damage != "missing-member":
                archive.writestr("xl/workbook.xml", "<" if damage == "malformed-xml" else workbook)
                archive.writestr("xl/_rels/workbook.xml.rels",
                                 '<Relationships><Relationship Id="r1"' + target + '/></Relationships>')
                archive.writestr("xl/worksheets/sheet1.xml",
                                 '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                                 '<sheetData><row>' + cell + '</row></sheetData></worksheet>')
        if damage == "corrupt-deflate":
            payload = bytearray(labels.read_bytes())
            # First member has no extra fields; reserved DEFLATE block type is invalid.
            payload[30 + len("xl/workbook.xml")] = 0x07
            labels.write_bytes(payload)
    _assert_cli_loader_failure(dict(inputs, labels_path=labels), monkeypatch, capsys)


@pytest.mark.parametrize("damage", ["crc", "deflate"])
def test_cli_real_cache_archive_errors_are_class_only_failures(inputs, monkeypatch, capsys, damage):
    index = json.loads(inputs["index_path"].read_text())
    path = inputs["index_path"].parent / index["videos"][0]["cache_file"]
    payload = bytearray(path.read_bytes())
    if damage == "crc":
        payload[payload.index(b"PK\x01\x02") + 16] ^= 1
    else:
        # NumPy writes a ZIP64 local header; use its actual filename/extra lengths.
        start = 30 + int.from_bytes(payload[26:28], "little") + int.from_bytes(payload[28:30], "little")
        payload[start] = 0x07
    path.write_bytes(payload)
    _assert_cli_loader_failure(inputs, monkeypatch, capsys)


@pytest.mark.parametrize("index", [[], {"weight_map": None}, {"weight_map": []},
                                  {"weight_map": {"vision.weight": None}},
                                  {"weight_map": {"vision.weight": "../outside.safetensors"}}])
def test_cli_real_malformed_sharded_snapshot_is_a_class_only_failure(inputs, monkeypatch, capsys, index):
    monkeypatch.setattr(io().dataset, "snapshot_fingerprint", snapshot_fingerprint)
    write_json(inputs["weights"] / "config.json", {"model_type": "siglip"})
    write_json(inputs["weights"] / "preprocessor_config.json", {})
    write_json(inputs["weights"] / "model.safetensors.index.json", index)
    _assert_cli_loader_failure(inputs, monkeypatch, capsys)


@pytest.mark.parametrize("seam", ["labels", "snapshot", "cache"])
@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit, AssertionError])
def test_loader_guards_preserve_control_flow_and_assertions(inputs, monkeypatch, seam, error_type):
    module = io()
    owner, name = {"labels": (module, "load_labels"),
                   "snapshot": (module.dataset, "snapshot_fingerprint"),
                   "cache": (module.features, "load_cache")}[seam]
    error = error_type("SYNTHETIC CONTROL FLOW")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(owner, name, fail)
    with pytest.raises(error_type) as caught:
        module.audit_inputs(**inputs)
    assert caught.value is error


def test_loader_guards_do_not_hide_attribute_errors_in_diagnostic_logic(inputs, monkeypatch):
    error = AttributeError("SYNTHETIC PROGRAMMING BUG OUTSIDE LOADERS")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(io(), "source_index", fail)
    with pytest.raises(AttributeError) as caught:
        io().audit_inputs(**inputs)
    assert caught.value is error


def test_known_related_manifest_sources_are_preserved_but_unknown_sentinels_are_not_groups(inputs):
    manifest = json.loads(inputs["manifest_path"].read_text())
    manifest[0]["source_groups"] = ["filming-session-A", "Unknown"]
    manifest[1]["source_group"] = "filming-session-B"
    manifest[2]["source_groups"] = ["filming-session-A", "UNASSIGNED"]
    write_json(inputs["manifest_path"], manifest)
    rows, audit = io().audit_inputs(**inputs)
    assert audit["known_source_groups"] == 2
    assert next(r for r in rows if r["label"])["source_groups"] == ["filming-session-A", "filming-session-B"]
    from ml.baseline.diagnostic import make_plan
    assert len(make_plan(rows)["components"]) == 1


@pytest.mark.parametrize("token", ["../escape.mov", "//server/share/file", "https://example.org/input"])
def test_local_file_rejects_network_paths_and_unsafe_inputs(token):
    with pytest.raises((ValueError, FileNotFoundError)):
        io().local_file(token)
