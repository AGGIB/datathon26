"""In-memory, development-only content-group OOF. No production training or export.

Inputs are already label-free mean-pooled SigLIP vectors, never frame/anchor data.
Plans are JSON snapshots sealed against accidental mutation (not authentication).
Unknown source groups remain isolated contents, NOT verified camera isolation.
Class-count arrays are ordered [negative, positive].
"""
import hashlib
import itertools
import json
import math
import re
import time
from collections.abc import Mapping

import numpy as np


_FLAGS = {"development_only": True, "camera_isolation_verified": False,
          "new_angle_holdout_validated": False, "deployable_model_created": False}


def _policy(seed):
    return {"seed": seed, "l2": [0.0001, 0.001, 0.01], "learning_rate": 0.1,
            "epochs": 1000, "loss": "train_only_balanced", "pooling": "label_free_mean",
            "normalization": "train_only_mean_std", "refit_validation": False,
            "threshold_grid": [i / 20 for i in range(1, 20)],
            "selection": "validation_balanced_accuracy_then_threshold_nearest_0.5_then_config_order",
            "threshold_tie": "nearest_0.5_then_lower", "inner_split": "group_holdout_quarter",
            "comparisons": ["siglip", "train_prior", "frozen_cv", "cv_and_siglip", "cv_or_siglip"],
            "positive_radius": None, "window": None, "time_head": False}


def _seal(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _samples(samples):
    if not isinstance(samples, list) or not samples:
        raise ValueError("samples must be a nonempty list")
    result, dimensions = {}, set()
    for row in samples:
        if not isinstance(row, dict):
            raise ValueError("Every sample must be an object")
        digest, label, vector = row.get("content_sha256"), row.get("label"), row.get("vector")
        if not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest):
            raise ValueError("content_sha256 must be literal lowercase SHA256")
        if type(label) is not int or label not in (0, 1):
            raise ValueError("label must be integer 0 or 1")
        if (not isinstance(vector, np.ndarray) or vector.ndim != 1 or not vector.size
                or vector.dtype.kind not in "fiu" or not np.isfinite(vector).all()):
            raise ValueError("vector must be finite nonempty 1D numeric ndarray")
        for key in ("members", "source_groups"):
            values = row.get(key)
            if (not isinstance(values, list) or (key == "members" and not values)
                    or any(not isinstance(v, str) or not v or v != v.strip() for v in values)):
                raise ValueError(key + " must contain literal nonempty strings")
        if digest in result:
            old = result[digest]
            if old["label"] != label or not np.array_equal(old["vector"], vector):
                raise ValueError("Duplicate content has inconsistent label/vector")
            for key in ("members", "source_groups"):
                old[key] = sorted(set(old[key]) | set(row[key]))
        else:
            result[digest] = dict(content_sha256=digest, label=label, vector=vector.astype(np.float64),
                                  members=sorted(set(row["members"])),
                                  source_groups=sorted(set(row["source_groups"])))
        dimensions.add(len(vector))
    if len(dimensions) != 1:
        raise ValueError("All vectors must have equal dimension")
    return dict(sorted(result.items()))


def _components(rows):
    parent, owners = {h: h for h in rows}, {}

    def root(h):
        while parent[h] != h:
            parent[h] = parent[parent[h]]
            h = parent[h]
        return h

    for h, row in rows.items():
        for group in row["source_groups"]:
            if group in owners:
                a, b = root(h), root(owners[group])
                parent[max(a, b)] = min(a, b)
            owners[group] = h
    components = {}
    for h in rows:
        components.setdefault(root(h), []).append(h)
    return sorted(components.values())


def _counts(hashes, rows):
    return [sum(rows[h]["label"] == c for h in hashes) for c in (0, 1)]


def _inner(pool, rows):
    """One separate group validation subset, with both classes on both sides if possible."""
    all_hashes = [h for group in pool for h in group]
    total = _counts(all_hashes, rows)
    candidates = itertools.chain(((i,) for i in range(len(pool))),
                                 itertools.combinations(range(len(pool)), 2))
    valid = []
    for indices in candidates:
        counts = _counts([h for i in indices for h in pool[i]], rows)
        if all(0 < counts[c] < total[c] for c in (0, 1)):
            valid.append(indices)
    if not valid:
        return all_hashes, [], "infeasible_inner_both_class_group_split"
    target = len(all_hashes) / 4
    chosen = set(min(valid, key=lambda ids: (abs(sum(len(pool[i]) for i in ids) - target), ids)))
    for i in range(len(pool)):
        if i in chosen:
            continue
        trial = chosen | {i}
        counts = _counts([h for j in trial for h in pool[j]], rows)
        size = sum(len(pool[j]) for j in trial)
        if (all(counts[c] < total[c] for c in (0, 1))
                and abs(size - target) < abs(sum(len(pool[j]) for j in chosen) - target)):
            chosen = trial
    return ([h for i, g in enumerate(pool) if i not in chosen for h in g],
            [h for i, g in enumerate(pool) if i in chosen for h in g], None)


def make_plan(samples, *, seed=0, max_folds=5):
    """Freeze label-stratified component partitions/configuration before any fitting.

    Labels alone (not vectors) determine splits. K is capped by the number of
    class-bearing components. Infeasible partitions remain explicit, never dropped.
    """
    if type(seed) is not int or seed < 0 or type(max_folds) is not int or not 1 <= max_folds <= 5:
        raise ValueError("seed must be nonnegative integer and max_folds in 1..5")
    rows = _samples(samples)
    components = _components(rows)
    bearing = [sum(any(rows[h]["label"] == c for h in g) for g in components) for c in (0, 1)]
    k = min(max_folds, *bearing)
    feasible = k >= 2
    k = k if feasible else 1
    rng = np.random.default_rng(seed)
    ordered = [components[i] for i in rng.permutation(len(components))]
    bins = [[] for _ in range(k)]
    if feasible:
        mixed = [g for g in ordered if set(rows[h]["label"] for h in g) == {0, 1}]
        pure = [[g for g in ordered if all(rows[h]["label"] == c for h in g)] for c in (0, 1)]
        for i in range(k):
            bins[i].extend([mixed.pop()] if mixed else [pure[0].pop(), pure[1].pop()])
        for group in mixed + pure[0] + pure[1]:
            counts = _counts(group, rows)
            side = min(range(k), key=lambda i: (sum(
                (_counts([h for g in bins[i] for h in g], rows)[c] + counts[c]) ** 2
                for c in (0, 1)), sum(map(len, bins[i])), i))
            bins[side].append(group)
    else:
        bins[0] = ordered
    folds = []
    for i, test_groups in enumerate(bins):
        pool = [g for j, groups in enumerate(bins) if j != i for g in groups]
        train, validation, reason = _inner(pool, rows)
        folds.append({"fold": i, "train": sorted(train), "validation": sorted(validation),
                      "test": sorted(h for g in test_groups for h in g),
                      "status": "infeasible" if reason else "ready",
                      "reason": reason if feasible else "infeasible_outer_class_bearing_components"})
    plan = {"manifest_version": 1, "kind": "development_content_group_oof", **_FLAGS,
            "policy": _policy(seed), "max_folds": max_folds, "n_folds": k,
            "hashes": list(rows), "components": components,
            "class_bearing_components": bearing, "folds": folds}
    plan["manifest_sha256"] = _seal(plan)
    return plan


def _validate_plan(plan, rows):
    if not isinstance(plan, dict):
        raise ValueError("plan must be a frozen JSON manifest")
    body = {k: v for k, v in plan.items() if k != "manifest_sha256"}
    if plan.get("manifest_sha256") != _seal(body):
        raise ValueError("Frozen plan was modified")
    policy = plan.get("policy")
    if (not isinstance(policy, dict) or plan.get("manifest_version") != 1
            or plan.get("kind") != "development_content_group_oof"
            or type(policy.get("seed")) is not int or policy["seed"] < 0
            or policy != _policy(policy["seed"]) or any(plan.get(k) != v for k, v in _FLAGS.items())
            or plan.get("hashes") != list(rows) or plan.get("components") != _components(rows)):
        raise ValueError("Plan policy, content coverage or source components mismatch")
    folds, cap = plan.get("folds"), plan.get("max_folds")
    if (type(cap) is not int or not 1 <= cap <= 5 or not isinstance(folds, list)
            or not 1 <= len(folds) <= cap or plan.get("n_folds") != len(folds)):
        raise ValueError("Invalid fold count")
    seen = []
    for i, fold in enumerate(folds):
        if (not isinstance(fold, dict) or set(fold) != {"fold", "train", "validation", "test", "status", "reason"}
                or type(fold["fold"]) is not int or fold["fold"] != i
                or fold["status"] not in ("ready", "infeasible")
                or (fold["status"] == "ready" and fold["reason"] is not None)
                or (fold["status"] == "infeasible" and (not isinstance(fold["reason"], str) or not fold["reason"]))):
            raise ValueError("Invalid fold identity/status/reason")
        assigned = {}
        for side in ("train", "validation", "test"):
            hashes = fold.get(side)
            if not isinstance(hashes, list) or (side == "test" and not hashes):
                raise ValueError("Invalid partition")
            for h in hashes:
                if not isinstance(h, str) or h not in rows or h in assigned:
                    raise ValueError("Partitions must be disjoint known content hashes")
                assigned[h] = side
        if set(assigned) != set(rows):
            raise ValueError("Each fold must exhaustively partition contents")
        if any(len({assigned[h] for h in g}) != 1 for g in plan["components"]):
            raise ValueError("Source component overlaps partitions")
        if fold.get("status") not in ("ready", "infeasible"):
            raise ValueError("Invalid fold status")
        if fold["status"] == "ready" and (not fold["train"] or not fold["validation"]):
            raise ValueError("Ready folds require separate train and validation")
        seen.extend(fold["test"])
    if len(seen) != len(rows) or set(seen) != set(rows):
        raise ValueError("Exactly one outer prediction per content is required")


@np.errstate(over="raise", invalid="raise", divide="raise")
def _fit(x, y, l2, policy):
    """Baseline logistic math; normalization and balancing only see this train matrix."""
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    normalized = (x - mean) / scale
    weights = np.random.default_rng(policy["seed"]).normal(0, 0.001, size=x.shape[1])
    intercept = 0.0
    counts = np.bincount(y.astype(int), minlength=2)
    balance = len(y) / (2 * counts[y.astype(int)])
    for _ in range(policy["epochs"]):
        logits = np.clip(normalized @ weights + intercept, -60, 60)
        error = (1 / (1 + np.exp(-logits)) - y) * balance
        weights -= policy["learning_rate"] * (normalized.T @ error / len(y) + l2 * weights)
        intercept -= policy["learning_rate"] * error.mean()
    if not np.isfinite(weights).all() or not math.isfinite(intercept):
        raise ValueError("Nonfinite fit")

    @np.errstate(over="raise", invalid="raise", divide="raise")
    def predict(values):
        logits = np.clip(((values - mean) / scale) @ weights + intercept, -60, 60)
        scores = 1 / (1 + np.exp(-logits))
        if not np.isfinite(scores).all():
            raise ValueError("Nonfinite prediction")
        return scores

    return predict


def _rate(numerator, denominator):
    return {"numerator": int(numerator), "denominator": int(denominator),
            "value": numerator / denominator if denominator else None}


def _metrics(rows, predictions):
    """Conditional quality plus fixed-population denominators; missing is never negative."""
    valid = [(h, p) for h, p in predictions.items() if p["status"] == "ok"]
    counts = {name: 0 for name in ("TP", "FP", "FN", "TN")}
    for h, p in valid:
        counts[("TP" if p["decision"] else "FN") if rows[h]["label"]
               else ("FP" if p["decision"] else "TN")] += 1
    tp, fp, fn, tn = (counts[k] for k in ("TP", "FP", "FN", "TN"))
    positives, negatives = _counts(list(rows), rows)[::-1]
    recall, specificity = _rate(tp, tp + fn), _rate(tn, tn + fp)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    scored = bool(valid) and all(p.get("score") is not None for _, p in valid)
    roc, ap = None, None
    if scored:
        groups = {}
        for h, p in valid:
            groups.setdefault(p["score"], []).append(rows[h]["label"])
        cum_pos, cum_neg, ap_sum, wins = 0, 0, 0.0, 0.0
        for score in sorted(groups, reverse=True):
            labels = groups[score]
            pos, neg = sum(labels), len(labels) - sum(labels)
            cum_pos += pos
            cum_neg += neg
            ap_sum += pos * cum_pos / (cum_pos + cum_neg)
            wins += neg * (cum_pos - pos + 0.5 * pos)
        roc = wins / ((tp + fn) * (tn + fp)) if (tp + fn) and (tn + fp) else None
        ap = ap_sum / (tp + fn) if tp + fn else None
    return {**counts, "coverage": {"evaluated": len(valid), "total": len(rows),
                                   "fraction": len(valid) / len(rows) if rows else None,
                                   "positive_evaluated": tp + fn, "positive_total": positives,
                                   "negative_evaluated": tn + fp, "negative_total": negatives},
            "recall": recall, "specificity": specificity, "precision": _rate(tp, tp + fp),
            "accuracy": _rate(tp + tn, len(valid)),
            "total_denominator_recall": _rate(tp, positives),
            "total_denominator_specificity": _rate(tn, negatives),
            "total_denominator_accuracy": _rate(tp + tn, len(rows)),
            "balanced_accuracy": (recall["value"] + specificity["value"]) / 2
            if recall["value"] is not None and specificity["value"] is not None else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "mcc": (tp * tn - fp * fn) / denominator if denominator else None,
            "roc_auc": roc, "pr_auc_average_precision": ap,
            "auc_note": "real scores only; AP uses tied-score blocks"}


def _select_threshold(y, scores, grid):
    candidates = []
    for threshold in grid:
        decisions = scores >= threshold
        ba = float((decisions[y == 1].mean() + (~decisions[y == 0]).mean()) / 2)
        candidates.append((ba, -abs(round(threshold - 0.5, 10)), -threshold, threshold))
    ba, _, _, threshold = max(candidates)
    return ba, threshold


def _measure(timings, phase, function, *args):
    cpu, wall = time.process_time(), time.perf_counter()
    try:
        return function(*args)
    finally:
        timings[phase + "_cpu"] += time.process_time() - cpu
        timings[phase + "_wall"] += time.perf_counter() - wall


def _run_fold(rows, fold, policy):
    started = time.perf_counter()
    details = {"fold": fold["fold"], "class_counts": {
        side: _counts(fold[side], rows) for side in ("train", "validation", "test")},
        "partitions": {side: list(fold[side]) for side in ("train", "validation", "test")},
        "normalization_fit_hashes": list(fold["train"]), "refit_validation": False,
        "status": "failed", "reason": fold["reason"], "chosen_l2": None, "threshold": None}
    timings = {phase + "_" + clock: 0.0 for phase in ("fit", "validation_predict", "outer_predict")
               for clock in ("cpu", "wall")}
    details["timings_s"] = timings
    prior = (sum(rows[h]["label"] for h in fold["train"]) / len(fold["train"])) if fold["train"] else None
    records = {h: {"status": "failed", "score": None, "decision": None,
                   "prior_score": prior, "prior_decision": bool(prior >= 0.5) if prior is not None else None}
               for h in fold["test"]}
    if fold["status"] == "ready":
        try:
            if any(0 in details["class_counts"][side] for side in ("train", "validation")):
                raise ValueError("Both classes required in train and validation")
            x = np.stack([rows[h]["vector"] for h in fold["train"]])
            y = np.array([rows[h]["label"] for h in fold["train"]], dtype=np.float64)
            vx = np.stack([rows[h]["vector"] for h in fold["validation"]])
            vy = np.array([rows[h]["label"] for h in fold["validation"]])
            best, chosen, choices = None, None, []
            for index, l2 in enumerate(policy["l2"]):
                predict = _measure(timings, "fit", _fit, x, y, l2, policy)
                scores = _measure(timings, "validation_predict", predict, vx)
                ba, threshold = _select_threshold(vy, scores, policy["threshold_grid"])
                choices.append({"l2": l2, "threshold": threshold, "validation_balanced_accuracy": ba})
                key = (ba, -abs(round(threshold - 0.5, 10)), -index)
                if best is None or key > best:
                    best, chosen = key, (predict, l2, threshold, ba)
            assert chosen is not None  # The validated manifest fixes three configurations.
            predict, l2, threshold, ba = chosen
            scores = _measure(timings, "outer_predict", predict,
                              np.stack([rows[h]["vector"] for h in fold["test"]]))
            prior = float(y.mean())
            for h, score in zip(fold["test"], scores):
                records[h] = {"status": "ok", "score": float(score), "decision": bool(score >= threshold),
                              "prior_score": prior, "prior_decision": bool(prior >= 0.5)}
            details.update(status="ok", reason=None, chosen_l2=l2, threshold=threshold,
                           validation_balanced_accuracy=ba, selection_candidates=choices,
                           train_class_balance_counts=details["class_counts"]["train"])
        except (ValueError, FloatingPointError, ArithmeticError, np.linalg.LinAlgError) as exc:
            details["reason"] = "numeric_or_class_failure:" + type(exc).__name__
    details["elapsed_s"] = time.perf_counter() - started
    return details, records


def run_diagnostic(samples, plan, *, frozen_cv=None):
    """Evaluate a frozen development plan. Returns JSON diagnostics, never estimator arrays.

    frozen_cv maps known content hashes to status and a strict bool decision when ok;
    an optional finite score in [0,1] is used for AUC, never fabricated from bools.
    A single inner group holdout selects L2/threshold; train-only estimator retained.
    """
    started = time.perf_counter()
    rows = _samples(samples)
    _validate_plan(plan, rows)
    if frozen_cv is not None and (not isinstance(frozen_cv, Mapping) or set(frozen_cv) - set(rows)):
        raise ValueError("frozen_cv must map known content hashes")
    cv = {}
    for h in rows:
        p = (frozen_cv or {}).get(h, {"status": "missing"})
        if not isinstance(p, dict) or not isinstance(p.get("status"), str) or not p["status"]:
            raise ValueError("CV records require explicit status")
        if p["status"] == "ok":
            score = p.get("score")
            if type(p.get("decision")) is not bool or (score is not None and (
                    type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1)):
                raise ValueError("OK CV requires bool decision and optional real finite score")
            cv[h] = {"status": "ok", "decision": p["decision"], "score": score}
        else:
            cv[h] = {"status": p["status"], "decision": None, "score": None}
    oof, folds = {}, []
    for fold in plan["folds"]:
        details, records = _run_fold(rows, fold, plan["policy"])
        for h, p in records.items():
            p.update(content_sha256=h, fold=fold["fold"], members=rows[h]["members"],
                     source_groups=rows[h]["source_groups"], reference_label=rows[h]["label"],
                     error=("FN" if rows[h]["label"] else "FP")
                     if p["status"] == "ok" and p["decision"] != bool(rows[h]["label"]) else None,
                     failure_reason=details["reason"])
        oof.update(records)
        folds.append(details)
    systems = {"siglip": oof, "train_prior": {
        h: {"status": "ok" if p["prior_score"] is not None else "missing_train",
            "score": p["prior_score"], "decision": p["prior_decision"]}
        for h, p in oof.items()}, "frozen_cv": cv}
    for name, operator in (("cv_and_siglip", all), ("cv_or_siglip", any)):
        systems[name] = {
            h: {"status": "ok", "score": None, "decision": operator([oof[h]["decision"], cv[h]["decision"]])}
            if oof[h]["status"] == cv[h]["status"] == "ok"
            else {"status": "missing_dependency", "score": None, "decision": None} for h in rows}
    common = [h for h in rows if all(p[h]["status"] == "ok" for p in systems.values())]
    common_rows = {h: rows[h] for h in common}
    joint = {name: _metrics(common_rows, {h: p[h] for h in common}) for name, p in systems.items()}
    deltas = {}
    for name in ("siglip", "cv_and_siglip", "cv_or_siglip"):
        deltas[name] = {
            "lost_cv_true_positives": sum(rows[h]["label"] == 1 and cv[h]["decision"]
                                          and not systems[name][h]["decision"] for h in common),
            "new_false_positives": sum(rows[h]["label"] == 0 and not cv[h]["decision"]
                                       and systems[name][h]["decision"] for h in common)}
    for fold in folds:
        subset = {h: rows[h] for h in fold["partitions"]["test"]}
        fold["metrics"] = _metrics(subset, {h: oof[h] for h in subset})
    sensitivity = {}
    for metric in ("balanced_accuracy", "f1", "mcc", "roc_auc", "pr_auc_average_precision",
                   "recall", "specificity", "precision", "accuracy"):
        values = [f["metrics"][metric] for f in folds]
        values = [v["value"] if isinstance(v, dict) else v for v in values]
        values = [v for v in values if v is not None]
        sensitivity[metric] = {"macro": sum(values) / len(values) if values else None,
                               "min": min(values) if values else None, "max": max(values) if values else None,
                               "defined_folds": len(values), "total_folds": len(folds)}
    return {"report_kind": "development_content_group_oof", **_FLAGS,
            "manifest_sha256": plan["manifest_sha256"], "policy": _policy(plan["policy"]["seed"]),
            "non_applicable": ["violation_type", "event_timing"],
            "n_unique_contents": len(rows), "n_input_records": len(samples),
            "oof": [oof[h] for h in rows], "frozen_cv": cv, "folds": folds,
            "individual": {name: _metrics(rows, p) for name, p in systems.items()},
            "joint": {"hashes": common, "coverage": {"evaluated": len(common), "total": len(rows)},
                      "metrics": joint, "changes_vs_frozen_cv": deltas},
            "fold_sensitivity": sensitivity, "timings_s": {"total": time.perf_counter() - started,
                **{key: sum(f["timings_s"][key] for f in folds) for key in folds[0]["timings_s"]}}}
