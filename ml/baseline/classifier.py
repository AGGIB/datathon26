"""NumPy-only logistic backup, fit solely on explicitly grouped train samples."""
import math
import json
from pathlib import Path

from .core import (FEATURE_SPEC, aggregate_predictions, canonical_video_id, content_sha256,
                   segment_labels, split_videos)
from .features import numpy_module, validate_cache


def _manifest_caches(caches, rows):
    """Canonical one-to-one join; compare source digests only when metadata has them."""
    by_id, dimensions, verification = {}, set(), {}
    expected = {canonical_video_id(row["video_id"]): row for row in rows}
    for cache in caches:
        key = canonical_video_id(cache["video_id"])
        cache = validate_cache(cache["video_id"], cache["timestamps_s"], cache["embeddings"], cache["metadata"])
        if key in by_id:
            raise ValueError("Duplicate feature cache canonical video_id: " + key)
        if key not in expected:
            raise ValueError("Cache coverage must be complete with no omissions/extras")
        verification[key] = "unavailable"
        if "content_sha256" in cache["metadata"]:
            digest = content_sha256(cache["metadata"]["content_sha256"])
            if digest != expected[key]["content_sha256"]:
                raise ValueError("Manifest content hash differs from extraction cache content hash")
            verification[key] = "matched_extraction_metadata"
        by_id[key] = cache
        dimensions.add(cache["embeddings"].shape[1])
    if set(by_id) != set(expected) or len(dimensions) != 1:
        raise ValueError("Exactly one compatible cache is required for every manifest video; no omissions/extras")
    np = numpy_module()
    by_content = {}
    for key, cache in by_id.items():
        digest = expected[key]["content_sha256"]
        if digest in by_content:
            original = by_content[digest]
            if any(not np.array_equal(cache[field], original[field])
                   for field in ("timestamps_s", "embeddings")):
                raise ValueError("Identical content has inconsistent feature caches")
        else:
            by_content[digest] = cache
    return by_id, verification


def train_classifier(caches, rows, train_angles, holdout_angles, positive_radius_s=2.0,
                     seed=0, epochs=1000, learning_rate=0.1, l2=0.001,
                     threshold=0.5, window_s=2.0, *, validation_angles=None):
    if validation_angles is None:
        raise ValueError("An explicit separate validation angle group is required for fitting")
    rows = list(rows)
    split = split_videos(rows, train_angles, holdout_angles, validation_angles=validation_angles)
    np = numpy_module()
    if (type(epochs) is not int or epochs <= 0 or not math.isfinite(learning_rate)
            or learning_rate <= 0 or not math.isfinite(l2) or l2 < 0
            or not math.isfinite(threshold) or not 0 <= threshold <= 1
            or not math.isfinite(window_s) or window_s <= 0):
        raise ValueError("Invalid optimizer, threshold or window configuration")
    by_id, verification = _manifest_caches(caches, rows)
    values, labels = [], []
    train_content = {}
    for row in split["train"]:
        digest = row["content_sha256"]
        if digest not in train_content:
            train_content[digest] = dict(row, positive_times_s=[])
        train_content[digest]["positive_times_s"] = sorted(set(
            train_content[digest]["positive_times_s"] + row.get("positive_times_s", [])))
    for row in train_content.values():
        cache = by_id[canonical_video_id(row["video_id"])]
        segment = segment_labels(row, cache["timestamps_s"], radius_s=positive_radius_s)
        mask = np.asarray([label is not None for label in segment])
        values.append(cache["embeddings"][mask])
        labels.extend(label for label in segment if label is not None)
    x, y = np.concatenate(values).astype(np.float64), np.asarray(labels, dtype=np.float64)
    if set(y.tolist()) != {0.0, 1.0}:
        raise ValueError("Training requires both positive segments and negative examples in train angles")
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    normalized = (x - mean) / scale
    rng = np.random.default_rng(seed)
    weights, intercept = rng.normal(0, 0.001, size=x.shape[1]), 0.0
    # Balanced loss prevents frame count alone from suppressing the minority class.
    counts = np.bincount(y.astype(int), minlength=2)
    sample_weights = len(y) / (2 * counts[y.astype(int)])
    for _ in range(epochs):
        logits = np.clip(normalized @ weights + intercept, -60, 60)
        error = (1 / (1 + np.exp(-logits)) - y) * sample_weights
        weights -= learning_rate * (normalized.T @ error / len(y) + l2 * weights)
        intercept -= learning_rate * error.mean()
    if not np.isfinite(weights).all() or not math.isfinite(intercept):
        raise ValueError("Training diverged; no model produced")
    return {"mean": mean, "scale": scale, "weights": weights, "intercept": np.asarray(intercept),
            "metadata": {"schema_version": 2, "feature_spec": dict(FEATURE_SPEC),
                         "content_sha256_by_video": {canonical_video_id(r["video_id"]): r["content_sha256"]
                                                     for r in rows},
                         "cache_video_ids": {key: c["video_id"] for key, c in by_id.items()},
                         "cache_content_hash_verification": verification,
                         "validation_video_ids": [r["video_id"] for r in split["validation"]],
                         "validation_angles": sorted({r["angle_id"] for r in split["validation"]}),
                         "train_video_ids": [r["video_id"] for r in split["train"]],
                         "holdout_video_ids": [r["video_id"] for r in split["holdout"]],
                         "train_angles": sorted(set(train_angles)), "holdout_angles": sorted(set(holdout_angles)),
                         "positive_radius_s": positive_radius_s, "seed": seed, "epochs": epochs,
                         "learning_rate": learning_rate, "l2": l2, "threshold": threshold,
                         "window_s": window_s, "n_supervised_train_frames": len(y),
                         "n_unique_train_contents": len(train_content)}}


def validate_model(model):
    np = numpy_module()
    meta = model.get("metadata", {})
    if meta.get("schema_version") != 2 or meta.get("feature_spec") != FEATURE_SPEC:
        raise ValueError("Invalid model schema or feature specification; a three-way split is required")
    for suffix in ("video_ids", "angles"):
        seen = set()
        for side in ("train", "validation", "holdout"):
            values = meta.get(side + "_" + suffix)
            if (not isinstance(values, list) or not values
                    or any(not isinstance(v, str) or not v or v != v.strip() for v in values)
                    or len(set(values)) != len(values) or set(values) & seen):
                raise ValueError("Invalid model explicit train/validation/holdout split")
            seen.update(values)
    hashes = meta.get("content_sha256_by_video")
    if not isinstance(hashes, dict):
        raise ValueError("Model requires complete content hashes for its split")
    seen_ids, content_sides = set(), {}
    for side in ("train", "validation", "holdout"):
        for value in meta[side + "_video_ids"]:
            key = canonical_video_id(value)
            if key in seen_ids:
                raise ValueError("Invalid canonical video collision in model split")
            seen_ids.add(key)
            digest = content_sha256(hashes.get(key))
            if digest in content_sides and content_sides[digest] != side:
                raise ValueError("Identical content cannot cross model split partitions")
            content_sides[digest] = side
    if set(hashes) != seen_ids:
        raise ValueError("Model content hashes must exactly cover its canonical split")
    for key in ("threshold", "window_s"):
        if not isinstance(meta.get(key), (int, float)) or not math.isfinite(meta[key]):
            raise ValueError("Invalid model prediction configuration")
    if not 0 <= meta["threshold"] <= 1 or meta["window_s"] <= 0:
        raise ValueError("Invalid model threshold/window")
    arrays = {}
    for key in ("mean", "scale", "weights", "intercept"):
        value = np.asarray(model[key])
        if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
            raise ValueError("Model arrays must be finite numeric arrays")
        arrays[key] = value.astype(np.float64)
    dimensions = arrays["weights"].shape
    if (len(dimensions) != 1 or not dimensions[0] or arrays["mean"].shape != dimensions
            or arrays["scale"].shape != dimensions or arrays["intercept"].shape != ()
            or (arrays["scale"] <= 0).any()):
        raise ValueError("Invalid model array dimensions or nonpositive scale")
    arrays["metadata"] = meta
    return arrays


def save_model(path, model):
    np = numpy_module()
    model = validate_model(model)
    serialized = json.dumps(model["metadata"], ensure_ascii=False, allow_nan=False)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        np.savez_compressed(handle, **{k: model[k] for k in ("mean", "scale", "weights", "intercept")},
                            metadata_json=np.asarray(serialized))


def load_model(path):
    np = numpy_module()
    try:
        with np.load(path, allow_pickle=False) as data:
            model = {key: data[key] for key in ("mean", "scale", "weights", "intercept")}
            model["metadata"] = json.loads(str(data["metadata_json"].item()))
        return validate_model(model)
    except (KeyError, TypeError, OSError, EOFError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid safe baseline model: " + str(path)) from exc


def predict_probabilities(model, embeddings):
    np = numpy_module()
    x = np.asarray(embeddings, dtype=np.float64)
    if (x.ndim != 2 or len(x) == 0 or x.shape[1] != len(model["weights"])
            or not np.isfinite(x).all()):
        raise ValueError("Prediction embeddings must be nonempty, finite and match model dimensions")
    logits = np.clip(((x - model["mean"]) / model["scale"]) @ model["weights"]
                     + model["intercept"], -60, 60)
    return 1 / (1 + np.exp(-logits))


def predict_cache(model, cache, threshold=None, window_s=None):
    model = validate_model(model)
    cache = validate_cache(cache["video_id"], cache["timestamps_s"], cache["embeddings"], cache["metadata"])
    scores = predict_probabilities(model, cache["embeddings"])
    result = aggregate_predictions(cache["video_id"], cache["timestamps_s"], scores,
                                   threshold=model["metadata"]["threshold"] if threshold is None else threshold,
                                   window_s=model["metadata"]["window_s"] if window_s is None else window_s)
    result["sampled_coverage"] = {"first_pts_s": float(cache["timestamps_s"][0]),
                                  "last_pts_s": float(cache["timestamps_s"][-1]),
                                  "n_frames": len(scores),
                                  "extraction": cache["metadata"].get("extraction")}
    result["seen_in_training"] = canonical_video_id(cache["video_id"]) in {
        canonical_video_id(value) for value in model["metadata"]["train_video_ids"]}
    return result


def evaluate_holdout(model, caches, rows, train_angles, holdout_angles, *, validation_angles=None):
    """Diagnostic candidate predictions only; official metrics belong to the evaluator."""
    model = validate_model(model)
    meta = model["metadata"]
    rows = list(rows)
    validation_angles = meta["validation_angles"] if validation_angles is None else validation_angles
    split = split_videos(rows, train_angles, holdout_angles, validation_angles=validation_angles)
    for side, supplied in (("train", train_angles), ("validation", validation_angles), ("holdout", holdout_angles)):
        if (set(supplied) != set(meta[side + "_angles"])
                or {canonical_video_id(r["video_id"]) for r in split[side]}
                != {canonical_video_id(value) for value in meta[side + "_video_ids"]}):
            raise ValueError("Evaluation must repeat the model's original explicit video/angle partition")
    if {canonical_video_id(r["video_id"]): r["content_sha256"] for r in rows} != meta["content_sha256_by_video"]:
        raise ValueError("Evaluation content hashes differ from the original model partition")
    by_id, verification = _manifest_caches(caches, split["holdout"])
    predictions = []
    for row in split["holdout"]:
        result = predict_cache(model, by_id[canonical_video_id(row["video_id"])])
        result["manifest_video_id"] = row["video_id"]
        result["reference_label"] = row["label"]
        result["angle_id"] = row["angle_id"]
        predictions.append(result)
    return {"schema_version": 1, "report_kind": "explicit_angle_holdout_candidates_not_legal_verdicts",
            "train_angles": list(meta["train_angles"]), "holdout_angles": list(meta["holdout_angles"]),
            "validation_angles": list(meta["validation_angles"]),
            "cache_content_hash_verification": verification,
            "n_videos": len(predictions), "predictions": predictions,
            "note": "No missing predictions imputed; threshold is fixed before holdout evaluation"}
