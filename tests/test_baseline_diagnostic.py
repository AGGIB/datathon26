"""Synthetic-only tests for development content-group OOF; no private I/O."""
import copy
import hashlib
import importlib
import importlib.util
import itertools
import json

from collections import UserDict

import numpy as np
import pytest


def samples(n=40):
    return [{"content_sha256": hashlib.sha256(f"synthetic-{i}".encode()).hexdigest(),
             "label": i % 2, "vector": np.array([2 * (i % 2) - 1, i / n, 1.0]),
             "members": [f"literal-{i}.mp4"], "source_groups": []} for i in range(n)]


def engine():
    assert importlib.util.find_spec("ml.baseline.diagnostic") is not None, "diagnostic API is missing"
    return importlib.import_module("ml.baseline.diagnostic")


def test_plan_is_deterministic_json_snapshot_with_exhaustive_group_splits():
    d = engine()
    rows = samples()
    plan = d.make_plan(rows)
    assert json.loads(json.dumps(plan, allow_nan=False)) == plan
    assert plan == d.make_plan(list(reversed(rows)))
    assert plan["policy"]["l2"] == [0.0001, 0.001, 0.01]
    assert plan["policy"]["threshold_grid"] == [i / 20 for i in range(1, 20)]
    assert plan["n_folds"] == 5
    hashes = {s["content_sha256"] for s in rows}
    seen = []
    for fold in plan["folds"]:
        sides = [set(fold[k]) for k in ("train", "validation", "test")]
        assert set.union(*sides) == hashes
        assert all(not sides[i] & sides[j] for i in range(3) for j in range(i))
        for side in sides:
            assert {s["label"] for s in rows if s["content_sha256"] in side} == {0, 1}
        seen += fold["test"]
    assert len(seen) == len(set(seen)) == 40


def test_run_emits_one_numeric_oof_per_content_without_estimator_or_production_calls(monkeypatch):
    d = engine()
    from ml.baseline import classifier

    def forbidden(*args, **kwargs):
        pytest.fail("Production fitting/export must never be called")

    monkeypatch.setattr(classifier, "train_classifier", forbidden)
    monkeypatch.setattr(classifier, "save_model", forbidden)
    rows = samples()
    plan = d.make_plan(rows)
    original = copy.deepcopy(plan)
    report = d.run_diagnostic(rows, plan)
    assert plan == original
    assert report["development_only"] is True
    for flag in ("camera_isolation_verified", "new_angle_holdout_validated", "deployable_model_created"):
        assert report[flag] is False
    assert report["non_applicable"] == ["violation_type", "event_timing"]
    assert len(report["oof"]) == len({p["content_sha256"] for p in report["oof"]}) == 40
    assert all(p["status"] == "ok" and 0 <= p["score"] <= 1 for p in report["oof"])
    assert all(type(p["decision"]) is bool for p in report["oof"])
    assert report["individual"]["siglip"]["coverage"]["evaluated"] == 40
    assert report["individual"]["siglip"]["balanced_accuracy"] == 1
    assert report["joint"]["coverage"]["evaluated"] == 0
    assert report["timings_s"]["total"] >= 0
    forbidden_keys = {"weights", "intercept", "mean", "scale", "timestamps_s", "anchors", "positive_times_s"}

    def walk(value):
        if isinstance(value, dict):
            assert not forbidden_keys.intersection(value)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(report)
    json.dumps(report, allow_nan=False)


def test_fit_failure_keeps_prior_and_cv_individual_coverage_with_fixed_denominators(monkeypatch):
    d = engine()
    rows = samples()
    plan = d.make_plan(rows)
    original_fit = d._fit
    calls = 0

    def fail_first_fold(x, y, l2, policy):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise FloatingPointError("synthetic failure")
        return original_fit(x, y, l2, policy)

    monkeypatch.setattr(d, "_fit", fail_first_fold)
    cv = {s["content_sha256"]: {"status": "ok", "decision": bool(s["label"])} for s in rows}
    report = d.run_diagnostic(rows, plan, frozen_cv=cv)
    assert report["individual"]["train_prior"]["coverage"]["evaluated"] == 40
    assert report["individual"]["frozen_cv"]["coverage"]["evaluated"] == 40
    assert report["individual"]["siglip"]["coverage"]["evaluated"] == 32
    failed = [p for p in report["oof"] if p["status"] != "ok"]
    assert len(failed) == 8
    assert all(p["decision"] is None and p["score"] is None and p["error"] is None for p in failed)
    assert report["joint"]["coverage"] == {"evaluated": 32, "total": 40}
    assert all(m["coverage"]["evaluated"] == 32 for m in report["joint"]["metrics"].values())
    metrics = report["individual"]["siglip"]
    assert metrics["total_denominator_recall"] == {"numerator": 16, "denominator": 20, "value": 0.8}
    assert metrics["recall"]["value"] == 1
    assert report["fold_sensitivity"]["balanced_accuracy"]["defined_folds"] == 4
    assert report["fold_sensitivity"]["balanced_accuracy"]["total_folds"] == 5


def test_threshold_ties_choose_nearest_half_then_lower():
    d = engine()
    y = np.array([1, 1, 0, 0])
    scores = np.array([0.6, 0.47, 0.52, 0.4])
    assert d._select_threshold(y, scores, [i / 20 for i in range(1, 20)]) == (0.75, 0.45)
    assert d._select_threshold(y, np.full(4, 0.7), [i / 20 for i in range(1, 20)]) == (0.5, 0.5)


def test_fit_and_prediction_cpu_cost_are_measured_separately_from_total_job():
    d = engine()
    rows = samples(12)
    report = d.run_diagnostic(rows, d.make_plan(rows, max_folds=3))
    for phase in ("fit", "validation_predict", "outer_predict"):
        for clock in ("cpu", "wall"):
            key = phase + "_" + clock
            assert report["timings_s"][key] >= 0
            assert report["timings_s"][key] == sum(f["timings_s"][key] for f in report["folds"])
    assert report["timings_s"]["fit_wall"] <= report["timings_s"]["total"]


def test_configuration_uses_validation_accuracy_then_threshold_proximity_then_order(monkeypatch):
    d = engine()
    rows = samples()
    plan = d.make_plan(rows)

    def controlled_fit(x, y, l2, policy):
        # These are synthetic feature-driven probabilities, never truth/ID-driven production inference.
        return lambda values: (np.full(len(values), 0.5) if l2 == 0.0001
                               else np.where(values[:, 0] > 0, 0.7, 0.6) if l2 == 0.001
                               else np.where(values[:, 0] > 0, 0.4, 0.3))

    monkeypatch.setattr(d, "_fit", controlled_fit)
    report = d.run_diagnostic(rows, plan)
    assert all(f["chosen_l2"] == 0.01 and f["threshold"] == 0.4 for f in report["folds"])
    monkeypatch.setattr(d, "_fit", lambda x, y, l2, policy: lambda v: np.where(v[:, 0] > 0, 0.7, 0.3))
    report = d.run_diagnostic(rows, plan)
    assert all(f["chosen_l2"] == 0.0001 and f["threshold"] == 0.5 for f in report["folds"])


def test_outer_labels_and_values_cannot_change_frozen_fold_selection_or_train_inputs(monkeypatch):
    d = engine()
    rows = samples()
    plan = d.make_plan(rows)
    original_fit, seen = d._fit, []

    def spy(x, y, l2, policy):
        seen.append((x.copy(), y.copy()))
        return original_fit(x, y, l2, policy)

    monkeypatch.setattr(d, "_fit", spy)
    before = d.run_diagnostic(rows, plan)
    for i, fold in enumerate(plan["folds"]):
        expected = [s for h in fold["train"] for s in rows if s["content_sha256"] == h]
        for x, y in seen[i * 3:(i + 1) * 3]:
            np.testing.assert_array_equal(x, np.stack([s["vector"] for s in expected]))
            np.testing.assert_array_equal(y, [s["label"] for s in expected])
    assert len(seen) == 15  # No post-selection train+validation refit.
    changed = copy.deepcopy(rows)
    held = set(plan["folds"][0]["test"])
    for row in changed:
        if row["content_sha256"] in held:
            row["label"] = 1 - row["label"]
    after = d.run_diagnostic(changed, plan)
    assert [(p["score"], p["decision"]) for p in before["oof"] if p["fold"] == 0] == [
        (p["score"], p["decision"]) for p in after["oof"] if p["fold"] == 0]
    for row in changed:
        if row["content_sha256"] in held:
            row["vector"] = np.array([10000.0, -10000.0, 700.0])
    after = d.run_diagnostic(changed, plan)
    for key in ("chosen_l2", "threshold", "selection_candidates", "normalization_fit_hashes"):
        assert before["folds"][0][key] == after["folds"][0][key]


def test_numeric_fit_matches_train_only_baseline_math_with_imbalance_and_constant_column():
    d = engine()
    x = np.array([[-3, 1], [-1, 1], [0, 1], [2, 1], [9, 1]], dtype=float)
    y = np.array([0, 0, 0, 0, 1])
    query = np.array([[1000, 1], [-7, 1], [2, 1]], dtype=float)
    policy = d.make_plan(samples())["policy"]
    z = (x - x.mean(0)) / np.where(x.std(0) < 1e-12, 1, x.std(0))
    w, bias = np.random.default_rng(0).normal(0, 0.001, size=2), 0.0
    balance = len(y) / (2 * np.bincount(y)[y])
    for _ in range(1000):
        error = (1 / (1 + np.exp(-np.clip(z @ w + bias, -60, 60))) - y) * balance
        w -= 0.1 * (z.T @ error / len(y) + 0.001 * w)
        bias -= 0.1 * error.mean()
    expected = 1 / (1 + np.exp(-np.clip(((query - x.mean(0)) /
                       np.where(x.std(0) < 1e-12, 1, x.std(0))) @ w + bias, -60, 60)))
    np.testing.assert_allclose(d._fit(x, y, 0.001, policy)(query), expected, rtol=0, atol=1e-14)


def test_exact_duplicates_count_once_and_preserve_literal_members():
    d = engine()
    rows = samples()
    extra = copy.deepcopy(rows[0])
    extra["members"] = ["Cyrillic-С.mp4", "Latin-C.mp4"]
    duplicate_rows = rows + [extra, extra]
    assert d.make_plan(rows) == d.make_plan(duplicate_rows)
    a = d.run_diagnostic(rows, d.make_plan(rows))
    b = d.run_diagnostic(duplicate_rows, d.make_plan(duplicate_rows))
    assert b["n_unique_contents"] == 40 and b["n_input_records"] == 42
    assert b["individual"] == a["individual"]
    assert [p["score"] for p in a["oof"]] == [p["score"] for p in b["oof"]]
    record = next(p for p in b["oof"] if p["content_sha256"] == extra["content_sha256"])
    assert record["members"] == sorted(rows[0]["members"] + extra["members"])
    for key, value in (("label", 1), ("vector", extra["vector"] + 1)):
        bad = copy.deepcopy(extra)
        bad[key] = value
        with pytest.raises(ValueError, match="inconsistent"):
            d.make_plan(rows + [bad])


def test_source_groups_union_transitively_including_duplicate_group_merges():
    d = engine()
    rows = samples()
    rows[0]["source_groups"] = ["a"]
    rows[1]["source_groups"] = ["a", "b"]
    rows[2]["source_groups"] = ["b"]
    extra = copy.deepcopy(rows[2])
    extra["source_groups"] = ["c"]
    rows[3]["source_groups"] = ["c"]
    rows += [extra]
    plan = d.make_plan(rows)
    related = {s["content_sha256"] for s in rows[:4]}
    assert any(set(group) == related for group in plan["components"])
    for fold in plan["folds"]:
        assert sum(bool(set(fold[side]) & related) for side in ("train", "validation", "test")) == 1
    assert plan == d.make_plan(rows[::-1])
    d.run_diagnostic(rows, plan)


def test_infeasible_class_bearing_components_keep_all_failed_oof_records():
    d = engine()
    for n in (1, 2, 4):
        rows = samples(n)
        report = d.run_diagnostic(rows, d.make_plan(rows))
        assert len(report["oof"]) == n
        assert all(p["status"] == "failed" and p["decision"] is None for p in report["oof"])
        assert all(f["reason"].startswith("infeasible_") for f in report["folds"])
        assert report["individual"]["siglip"]["coverage"]["total"] == n
        assert report["individual"]["siglip"]["coverage"]["evaluated"] == 0
        json.dumps(report, allow_nan=False)
    rows = samples()
    for row in rows:
        row["source_groups"] = ["one-related-component"]
    plan = d.make_plan(rows)
    assert plan["class_bearing_components"] == [1, 1] and plan["n_folds"] == 1
    assert len(d.run_diagnostic(rows, plan)["oof"]) == 40
    # Three mixed components suffice for an outer test, inner validation and inner train.
    rows = samples(6)
    for i, row in enumerate(rows):
        row["source_groups"] = [f"pair-{i // 2}"]
    plan = d.make_plan(rows)
    assert plan["n_folds"] == 3
    assert all(f["status"] == "ready" for f in plan["folds"])
    assert all(p["status"] == "ok" for p in d.run_diagnostic(rows, plan)["oof"])


def test_tampered_plan_is_rejected_even_when_accident_seal_is_recomputed():
    d = engine()
    rows = samples()
    rows[0]["source_groups"] = rows[1]["source_groups"] = ["related"]
    plan = d.make_plan(rows)
    corrupt = copy.deepcopy(plan)
    corrupt["policy"]["l2"] = [7.0]
    with pytest.raises(ValueError, match="modified"):
        d.run_diagnostic(rows, corrupt)

    def reseal(p):
        p["manifest_sha256"] = d._seal({k: v for k, v in p.items() if k != "manifest_sha256"})
        return p

    mutations = []
    p = copy.deepcopy(plan)
    p["folds"][0]["train"].append(p["folds"][0]["test"][0])
    mutations.append(p)
    p = copy.deepcopy(plan)
    p["folds"][0]["train"].pop()
    mutations.append(p)
    p = copy.deepcopy(plan)
    p["folds"][0]["test"][0] = "f" * 64
    mutations.append(p)
    p = copy.deepcopy(plan)
    p["folds"][1] = copy.deepcopy(p["folds"][0])
    p["folds"][1]["fold"] = 1
    mutations.append(p)
    p = copy.deepcopy(plan)
    p["policy"]["l2"] = [7.0]
    mutations.append(p)
    p = copy.deepcopy(plan)
    fold = p["folds"][0]
    h = rows[0]["content_sha256"]
    side = next(s for s in ("train", "validation", "test") if h in fold[s])
    dest = next(s for s in ("train", "validation", "test") if s != side)
    fold[side].remove(h)
    fold[dest].append(h)
    mutations.append(p)
    for p in mutations:
        with pytest.raises(ValueError):
            d.run_diagnostic(rows, reseal(p))


@pytest.mark.parametrize("key,value", [
    ("content_sha256", "A" * 64), ("content_sha256", "0" * 63), ("label", True),
    ("label", 1.0), ("vector", [1, 2, 3]), ("vector", np.array([np.nan, 1, 2])),
    ("vector", np.array([[1, 2, 3]])), ("vector", np.array([True, False, True])),
    ("vector", np.array([1 + 1j])), ("vector", np.array(["1"])),
    ("members", []), ("source_groups", [" padded "]),
])
def test_invalid_samples_fail_closed(key, value):
    d = engine()
    rows = samples()
    rows[0][key] = value
    with pytest.raises(ValueError):
        d.make_plan(rows)


def test_tied_auc_and_ap_are_order_invariant_and_boolean_cv_has_no_invented_auc():
    d = engine()
    rows = {s["content_sha256"]: s for s in samples(4)}
    predictions = {h: {"status": "ok", "decision": True, "score": 0.5} for h in rows}
    metrics = d._metrics(rows, predictions)
    assert {k: metrics[k] for k in ("TP", "FP", "FN", "TN")} == {"TP": 2, "FP": 2, "FN": 0, "TN": 0}
    assert metrics["roc_auc"] == metrics["pr_auc_average_precision"] == 0.5
    assert metrics["precision"] == {"numerator": 2, "denominator": 4, "value": 0.5}
    assert metrics == d._metrics(rows, dict(reversed(list(predictions.items()))))
    for p in predictions.values():
        p["score"] = None
    assert d._metrics(rows, predictions)["roc_auc"] is None
    assert d._metrics(rows, predictions)["pr_auc_average_precision"] is None
    # Nontrivial tied ranking: positive and negative share the high-score block.
    for p, score in zip(predictions.values(), [0.8, 0.8, 0.1, 0.5]):
        p["score"] = score
    metrics = d._metrics(rows, predictions)
    assert metrics["roc_auc"] == 0.625
    assert metrics["pr_auc_average_precision"] == pytest.approx(7 / 12)


def test_cv_comparisons_use_common_units_and_report_lost_tps_new_fps(monkeypatch):
    d = engine()
    rows = samples()
    plan = d.make_plan(rows)
    # Deliberately inverted synthetic features; no threshold can repair this ranking.
    monkeypatch.setattr(d, "_fit", lambda x, y, l2, policy: lambda v: np.where(v[:, 0] > 0, 0.2, 0.8))
    cv = {s["content_sha256"]: {"status": "ok", "decision": bool(s["label"])} for s in rows[:38]}
    cv[rows[38]["content_sha256"]] = {"status": "missing", "decision": False}
    cv[rows[39]["content_sha256"]] = {"status": "failed", "decision": False}
    report = d.run_diagnostic(rows, plan, frozen_cv=cv)
    assert report["individual"]["siglip"]["coverage"]["evaluated"] == 40
    assert report["individual"]["frozen_cv"]["coverage"]["evaluated"] == 38
    assert report["joint"]["coverage"] == {"evaluated": 38, "total": 40}
    assert all(m["coverage"]["evaluated"] == 38 for m in report["joint"]["metrics"].values())
    # The all-negative/all-positive BA=.5 solutions beat an inverted BA=0 classifier.
    assert report["individual"]["frozen_cv"]["roc_auc"] is None
    assert report["individual"]["cv_or_siglip"]["roc_auc"] is None
    for name in ("siglip", "cv_and_siglip", "cv_or_siglip"):
        evaluated = [p for p in report["oof"] if p["content_sha256"] in report["joint"]["hashes"]]
        expected_lost = expected_new = 0
        for p in evaluated:
            base = cv[p["content_sha256"]]["decision"]
            decision = p["decision"] if name == "siglip" else (base and p["decision"]) if name == "cv_and_siglip" else (base or p["decision"])
            expected_lost += p["reference_label"] == 1 and base and not decision
            expected_new += p["reference_label"] == 0 and not base and decision
        assert report["joint"]["changes_vs_frozen_cv"][name] == {
            "lost_cv_true_positives": expected_lost, "new_false_positives": expected_new}
    assert report["frozen_cv"][rows[39]["content_sha256"]]["decision"] is None


def test_repeat_runs_and_input_permutations_are_deterministic_except_elapsed_time():
    d = engine()
    rows = samples()
    plan = d.make_plan(rows, seed=17, max_folds=4)

    def without_timing(value):
        if isinstance(value, dict):
            return {k: without_timing(v) for k, v in value.items() if k not in {"elapsed_s", "timings_s"}}
        if isinstance(value, list):
            return [without_timing(v) for v in value]
        return value

    assert without_timing(d.run_diagnostic(rows, plan)) == without_timing(d.run_diagnostic(rows[::-1], plan))


def test_extreme_finite_vectors_fail_explicitly_instead_of_silent_overflow():
    d = engine()
    rows = samples()
    for row in rows:
        row["vector"] = np.array([1e308 if row["label"] else -1e308, 1.0])
    with np.errstate(all="warn"):
        report = d.run_diagnostic(rows, d.make_plan(rows))
    assert all(p["status"] == "failed" and p["decision"] is None for p in report["oof"])
    assert report["individual"]["train_prior"]["coverage"]["evaluated"] == 40
    json.dumps(report, allow_nan=False)


def test_fold_sensitivity_includes_recall_specificity_and_precision():
    d = engine()
    rows = samples()
    report = d.run_diagnostic(rows, d.make_plan(rows))
    for metric in ("recall", "specificity", "precision"):
        assert report["fold_sensitivity"][metric] == {
            "macro": 1.0, "min": 1.0, "max": 1.0, "defined_folds": 5, "total_folds": 5}


def test_resealed_malformed_plan_is_rejected_before_any_fit(monkeypatch):
    d = engine()
    rows = samples()
    original = d.make_plan(rows)
    mutations = []
    p = copy.deepcopy(original)
    p["policy"] = None
    mutations.append(p)
    p = copy.deepcopy(original)
    del p["folds"][0]["reason"]
    mutations.append(p)
    p = copy.deepcopy(original)
    p["folds"][0]["status"] = "infeasible"
    mutations.append(p)
    p = copy.deepcopy(original)
    p["max_folds"] = 2
    mutations.append(p)
    p = copy.deepcopy(original)
    p["kind"] = "deployable_model"
    mutations.append(p)
    monkeypatch.setattr(d, "_fit", lambda *a: pytest.fail("Validate the whole plan before fitting"))
    for p in mutations:
        p["manifest_sha256"] = d._seal({k: v for k, v in p.items() if k != "manifest_sha256"})
        with pytest.raises(ValueError):
            d.run_diagnostic(rows, p)


def test_report_does_not_alias_the_frozen_manifest_policy():
    d = engine()
    rows = samples()
    plan = d.make_plan(rows)
    original = copy.deepcopy(plan)
    report = d.run_diagnostic(rows, plan)
    report["policy"]["l2"].append(123)
    assert plan == original


@pytest.mark.parametrize("record", [
    {"status": "ok", "decision": 1}, {"status": "ok", "decision": None},
    {"status": "ok", "decision": True, "score": True},
    {"status": "ok", "decision": True, "score": float("nan")},
    {"status": "ok", "decision": True, "score": 2.0}, {}, None,
])
def test_malformed_ok_cv_is_rejected_without_guessing(record):
    d = engine()
    rows = samples()
    with pytest.raises(ValueError):
        d.run_diagnostic(rows, d.make_plan(rows), frozen_cv={rows[0]["content_sha256"]: record})


def test_split_planning_is_label_stratified_but_never_uses_vector_values():
    d = engine()
    rows = samples()
    plan = d.make_plan(rows)
    altered = copy.deepcopy(rows)
    for i, row in enumerate(altered):
        row["vector"] = np.array([i * 10000.0, -17.0, 23.0])
    assert d.make_plan(altered) == plan


def test_inner_both_class_feasibility_matches_exhaustive_subset_oracle():
    d = engine()
    for types in itertools.combinations_with_replacement(((0,), (1,), (0, 1)), 6):
        rows, groups = {}, []
        for i, labels in enumerate(types):
            group = []
            for j, label in enumerate(labels):
                h = hashlib.sha256(f"component-{i}-{j}".encode()).hexdigest()
                rows[h] = {"label": label}
                group.append(h)
            groups.append(group)
        train, validation, reason = d._inner(groups, rows)
        possible = False
        for mask in itertools.product((False, True), repeat=len(groups)):
            sides = [{rows[h]["label"] for i, g in enumerate(groups) if mask[i] == side for h in g}
                     for side in (False, True)]
            possible |= all(labels == {0, 1} for labels in sides)
        assert (reason is None) == possible
        assert set(train) | set(validation) == set(rows)
        assert not set(train) & set(validation)
        if possible:
            assert {rows[h]["label"] for h in train} == {rows[h]["label"] for h in validation} == {0, 1}


def test_40_content_768_dimension_numpy_diagnostic_is_exercised():
    d = engine()
    rows = samples()
    rng = np.random.default_rng(23)
    for row in rows:
        vector = rng.normal(0, 0.1, 768)
        vector[:20] += 2 * row["label"] - 1
        row["vector"] = vector
    report = d.run_diagnostic(rows, d.make_plan(rows))
    assert len(report["oof"]) == 40
    assert report["individual"]["siglip"]["coverage"]["evaluated"] == 40
    assert all(np.isfinite(p["score"]) for p in report["oof"])
    json.dumps(report, allow_nan=False)


def test_cv_accepts_mapping_and_real_scores_but_rejects_unknown_content():
    d = engine()
    rows = samples()
    plan = d.make_plan(rows)
    cv = UserDict({s["content_sha256"]: {"status": "ok", "decision": bool(s["label"]),
                                         "score": 0.9 if s["label"] else 0.1} for s in rows})
    report = d.run_diagnostic(rows, plan, frozen_cv=cv)
    assert report["individual"]["frozen_cv"]["roc_auc"] == 1.0
    assert report["individual"]["frozen_cv"]["pr_auc_average_precision"] == 1.0
    cv["f" * 64] = {"status": "missing"}
    with pytest.raises(ValueError, match="known content"):
        d.run_diagnostic(rows, plan, frozen_cv=cv)


def test_production_still_rejects_unknown_angle_without_training_or_export():
    from ml.baseline.classifier import train_classifier
    rows = [{"video_id": "synthetic.mp4", "angle_id": "unknown", "label": 0,
             "content_sha256": "a" * 64, "positive_times_s": []}]
    with pytest.raises(ValueError, match="explicitly assigned angle_id"):
        train_classifier([], rows, ["train"], ["test"], validation_angles=["validation"])
