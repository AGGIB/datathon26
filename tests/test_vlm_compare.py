"""Only synthetic identifiers, byte hashes and numeric records; no media/models."""
import copy
import csv
import hashlib
import json
import subprocess
import sys

import pytest

from ml.tools import vlm_compare as tool
from ml.eval import score


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def answer(*events, **extra):
    return {"verdict": "violation" if events else "no_violation", "violations": list(events), **extra}


def event(kind="fixture", peak=10, confidence=.8, **extra):
    return {"type_id": kind, "t": peak, "conf": confidence, **extra}


def inputs(duplicate=False):
    labels = [{"video": "positive", "positive": True, "timestamp": 10.0},
              {"video": "clean", "positive": False, "timestamp": None}]
    if duplicate:
        labels.append({"video": "duplicate", "positive": True, "timestamp": 20.0})
    annotations = tool.primary_annotations(labels, [
        {"video": row["video"], "type_id": "fixture" if row["positive"] else "none",
         "t_peak": row["timestamp"]} for row in labels])
    hashes = {tool.video_key(row["video"]): digest("same" if row["positive"] else "other") for row in labels}
    baseline = {row["video"]: answer() for row in labels}
    return labels, annotations, hashes, baseline


def compare(baseline, probe, *, duplicate=False, cv=None, audit=None):
    labels, annotations, hashes, _ = inputs(duplicate)
    return tool.compare(tool.predictions(baseline), tool.predictions(probe), labels, annotations,
                        hashes, cv=tool.predictions(cv) if cv is not None else None, audit=audit)


def test_correct_type_at_boundary_is_not_stolen_by_closer_wrong_type():
    baseline = {"positive": answer(event("wrong", 10)), "clean": answer()}
    probe = {"positive": answer(event("wrong", 10), event("fixture", 12)), "clean": answer()}
    report = compare(baseline, probe, cv=baseline)
    assert report["arms"]["probe"]["primary_type_hits"] == 1
    assert report["arms"]["probe"]["primary_joint_hits"] == 1
    assert report["paired"]["joint_gains"] == 1
    assert report["paired"]["new_primary_corroborating_events"] == 1
    assert report["paired"]["new_events_on_positive_not_exhaustively_adjudicated"] == 0


def test_type_and_time_on_different_events_cannot_form_joint_hit():
    baseline = {"positive": answer(), "clean": answer()}
    probe = {"positive": answer(event("wrong", 10), event("fixture", 30)), "clean": answer()}
    report = compare(baseline, probe)
    assert report["arms"]["probe"]["primary_type_hits"] == 1
    assert report["arms"]["probe"]["primary_joint_hits"] == 0
    assert report["paired"]["new_events_on_positive_not_exhaustively_adjudicated"] == 2
    assert report["exact_cv_retention"]["available"] is False
    assert report["metric_gate_passed"] is False


def test_duplicate_contents_accept_alternative_anchor_and_do_not_multiply_events():
    _, _, _, baseline = inputs(duplicate=True)
    probe = copy.deepcopy(baseline)
    probe["positive"] = probe["duplicate"] = answer(event(peak=22))
    report = compare(baseline, probe, duplicate=True)
    assert report["cohort"]["compared_files"] == 3
    assert report["cohort"]["compared_contents"] == 2
    assert report["arms"]["probe"]["confusion"]["tp"] == 1
    assert report["arms"]["probe"]["primary_joint_hits"] == 1
    assert report["paired"]["new_primary_corroborating_events"] == 1


@pytest.mark.parametrize("change", ["confidence", "interval", "multiplicity"])
def test_cv_retention_checks_confidence_interval_and_event_multiplicity(change):
    original = event(t_start=9, t_end=11)
    baseline = {"positive": answer(original, original), "clean": answer()}
    probe = copy.deepcopy(baseline)
    if change == "confidence":
        probe["positive"]["violations"][0]["conf"] = .800001
    elif change == "interval":
        probe["positive"]["violations"][0]["t_start"] = 8
    else:
        probe["positive"]["violations"].pop()
    report = compare(baseline, probe, cv=baseline)
    assert report["exact_cv_retention"]["arms"]["baseline"]["passed"]
    assert not report["exact_cv_retention"]["arms"]["probe"]["passed"]
    assert not report["metric_gate_passed"]


def test_changed_vlm_reference_event_is_not_falsely_called_lost_cv():
    cv = {"positive": answer(event("cv", 8, .6)), "clean": answer()}
    baseline = {"positive": answer(event("cv", 8, .6), event("wrong", 10)), "clean": answer()}
    probe = {"positive": answer(event("cv", 8, .6), event("fixture", 10)), "clean": answer()}
    report = compare(baseline, probe, cv=cv)
    assert not report["exact_reference_retention"]["passed"]
    assert all(value["passed"] for value in report["exact_cv_retention"]["arms"].values())
    assert report["metric_gate_passed"]
    assert not report["eligible_for_fresh_full_run"]  # No recorded audit provenance.


def test_clean_additions_block_gate_and_positive_extras_remain_unadjudicated():
    baseline = {"positive": answer(), "clean": answer()}
    probe = {"positive": answer(event(), event("another", 15)), "clean": answer(event())}
    report = compare(baseline, probe, cv=baseline)
    assert report["paired"]["new_events_on_known_clean"] == 1
    assert report["paired"]["new_events_on_positive_not_exhaustively_adjudicated"] == 1
    assert not report["metric_gate_passed"]
    assert not report["scope"]["full_event_precision_available"]


def test_inherited_reference_false_addition_is_still_a_cv_clean_addition():
    cv = {"positive": answer(), "clean": answer()}
    baseline = {"positive": answer(), "clean": answer(event("wrong", 2))}
    probe = {"positive": answer(event()), "clean": answer(event("wrong", 2))}
    report = compare(baseline, probe, cv=cv)
    assert report["paired"]["new_reference_events_on_known_clean"] == 0
    assert report["paired"]["new_events_on_known_clean"] == 1
    assert not report["metric_gate_passed"]


def test_failed_duplicate_or_missing_alias_cannot_cherry_pick_successful_member():
    _, _, _, baseline = inputs(duplicate=True)
    probe = {"positive": answer(event()), "clean": answer(), "duplicate": {"status": "failed"}}
    report = compare(baseline, probe, duplicate=True)
    assert report["arms"]["probe"]["confusion"]["abstain_positive"] == 1
    assert report["arms"]["probe"]["primary_type_hits"] == 0
    probe.pop("duplicate")
    partial = compare(baseline, probe, duplicate=True)
    assert partial["status"] == "partial"
    assert partial["cohort"]["compared_contents"] == 2
    assert partial["arms"]["probe"]["confusion"]["abstain_positive"] == 1
    assert not partial["metric_gate_passed"]


def test_unknown_type_invalid_response_and_semantic_rejection_are_distinct():
    baseline = {"positive": answer(), "clean": answer()}
    probe = {"positive": answer(event("unknown")), "clean": answer()}
    audit = {"videos": {"positive": {"report": {"status": "completed", "decisions": [
        {"status": "uncertain", "reason": "invalid_response", "observations": ["PRIVATE_SENTINEL"]}]},
        "verifier_last_call_decisions": [{"accepted": False},
            {"accepted": False, "reason": "primitive_evidence_not_supported"}]}},
        "PRIVATE_SENTINEL": "PRIVATE_SENTINEL"}
    report = compare(baseline, probe, audit=audit)
    assert report["arms"]["probe"]["unknown_type_events"] == 1
    assert report["audit"]["malformed_windows"] == 1
    assert report["audit"]["semantic_rejections_recorded"] == 1
    assert report["audit"]["unknown_metadata_files"] == 1
    assert "PRIVATE_SENTINEL" not in json.dumps(report)
    probe["positive"] = answer(event(peak=True))
    malformed = compare(baseline, probe)
    assert malformed["arms"]["probe"]["confusion"]["abstain_positive"] == 1


def test_custom_screen_metadata_and_fallback_do_not_claim_complete_visual_run():
    baseline = {"positive": answer(), "clean": answer()}
    probe = {"positive": answer(event()), "clean": answer(vlm_error="PRIVATE_SENTINEL")}
    report = compare(baseline, probe, cv=baseline, audit={"scope": "custom_screen", "code_inputs_stable": True})
    assert report["metric_gate_passed"]
    assert report["audit"]["enhancement_error_files"] == 1
    assert report["audit"]["actual_model_calls"] is None
    assert report["audit"]["unknown_metadata_files"] == 2
    assert not report["eligible_for_fresh_full_run"]
    assert report["arms"]["probe"]["confusion"]["tn"] == 1  # Usable CV fallback.


def test_audit_hash_identity_must_match_numeric_hash_join():
    baseline = {"positive": answer(), "clean": answer()}
    with pytest.raises(ValueError, match="hash"):
        compare(baseline, baseline, audit={"videos": {"positive": {"sha256": digest("different")}}})


def test_complete_audit_call_budget_and_failure_accounting():
    labels, annotations, hashes, baseline = inputs()
    probe = {"positive": answer(event()), "clean": answer()}
    audit = {"code_inputs_stable": True, "config": {"max_model_calls_per_video": 4},
             "videos": {key: {"actual_calls_used": 2, "sha256": hashes[tool.video_key(key)],
                "report": {"status": "completed", "calls_used": 2}} for key in probe}}
    report = compare(baseline, probe, cv=baseline, audit=audit)
    assert report["eligible_for_fresh_full_run"]
    assert report["audit"]["actual_model_calls"] == 4
    expected = score(labels, list(tool.predictions(probe).values()), annotations=list(annotations.values()),
                     content_hashes=hashes)
    assert report["arms"]["probe"]["confusion"] == expected["overall"]["confusion"]
    audit["videos"]["positive"]["actual_calls_used"] = 5
    with pytest.raises(ValueError, match="accounting"):
        compare(baseline, probe, cv=baseline, audit=audit)


def test_duplicate_reuse_must_not_claim_extra_calls_or_different_content():
    _, _, hashes, baseline = inputs(duplicate=True)
    audit = {"videos": {"duplicate": {"sha256": hashes["DUPLICATE"], "inference_reused": True,
            "actual_calls_used": 1}}}
    with pytest.raises(ValueError, match="zero"):
        compare(baseline, baseline, duplicate=True, audit=audit)
    audit["videos"]["duplicate"].update(actual_calls_used=0, inference_source_video="clean")
    with pytest.raises(ValueError, match="reuse_source"):
        compare(baseline, baseline, duplicate=True, audit=audit)


def test_actual_hash_groups_and_canonical_collisions_rejected():
    labels, _, _, _ = inputs()
    with pytest.raises(ValueError):
        tool.actual_hashes({"content_aggregation": {"hash_source": "labels"}}, {r["video"] for r in labels})
    with pytest.raises(ValueError, match="collision"):
        tool.predictions({"Х-8.mov": answer(), "X-8": answer()})


@pytest.fixture
def cli_case(tmp_path, monkeypatch):
    monkeypatch.setattr(tool, "ROOT", tmp_path)
    (tmp_path / "ml").mkdir()
    (tmp_path / "ml/eval.py").write_text("# Synthetic evaluator provenance only\n")
    labels, annotations, hashes, baseline = inputs()
    paths = {}
    for name, document in {"baseline": baseline, "probe": {"positive": answer(event()), "clean": answer()},
        "hash-report": {"content_aggregation": {"hash_source": "actual", "hashes_authoritatively_verified": True},
            "per_content_group": [{"content_sha256": sha, "videos": [key]} for key, sha in hashes.items()]}}.items():
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps(document))
        paths[name] = path
    for name, fields, rows in [("labels", ["video", "timestamp", "label"], [
        {"video": r["video"], "timestamp": r["timestamp"], "label": "нарушение" if r["positive"] else "без нарушения"} for r in labels]),
        ("annotations", ["video", "type_id", "t_peak"], list(annotations.values()))]:
        path = tmp_path / (name + ".csv")
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        paths[name] = path
    out = tmp_path / "var" / "comparison.json"
    args = [part for name, path in paths.items() for part in ("--" + name, str(path))] + ["--out", str(out)]
    return args, out, paths


def test_cli_portable_numeric_report_no_overwrite_or_private_echo(cli_case, capsys):
    args, out, _ = cli_case
    assert tool.main(args) == 0
    report = json.loads(out.read_text())
    assert len(report["input_sha256"]) == 5
    before = out.read_bytes()
    assert tool.main(args) == 2
    assert out.read_bytes() == before
    assert tool.main(["--PRIVATE_SENTINEL"]) == 2
    assert "PRIVATE_SENTINEL" not in capsys.readouterr().out


@pytest.mark.parametrize("payload", ['{"x":1,"x":2}', '{"x":NaN}'])
def test_cli_rejects_nonfinite_duplicate_json_without_traceback(cli_case, payload, capsys):
    args, out, paths = cli_case
    paths["probe"].write_text(payload)
    assert tool.main(args) == 2
    assert not out.exists()
    assert "Traceback" not in capsys.readouterr().out


def test_import_has_no_model_or_media_module_side_effects():
    root = str(tool.ROOT)
    code = ("import sys;sys.path.insert(0," + repr(root) + ");import ml.tools.vlm_compare;"
            "assert not any(n in sys.modules for n in ('torch','mlx','transformers','cv2','PIL'))")
    subprocess.run([sys.executable, "-I", "-c", code], check=True, timeout=5)
