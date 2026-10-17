"""NumPy-only logistic backup, fit solely on explicitly grouped train samples."""
import math
import json
from pathlib import Path

from .core import FEATURE_SPEC, aggregate_predictions, segment_labels, split_videos
from .features import numpy_module, validate_cache


def train_classifier(caches, rows, train_angles, holdout_angles, positive_radius_s=2.0,
                     seed=0, epochs=1000, learning_rate=0.1, l2=0.001,
                     threshold=0.5, window_s=2.0):
    np = numpy_module()
    split = split_videos(rows, train_angles, holdout_angles)
    if (type(epochs) is not int or epochs <= 0 or not math.isfinite(learning_rate)
            or learning_rate <= 0 or not math.isfinite(l2) or l2 < 0
            or not math.isfinite(threshold) or not 0 <= threshold <= 1
            or not math.isfinite(window_s) or window_s <= 0):
        raise ValueError("Invalid optimizer, threshold or window configuration")
    by_id = {}
    dimensions = set()
    for cache in caches:
        cache = validate_cache(cache["video_id"], cache["timestamps_s"], cache["embeddings"], cache["metadata"])
        if cache["video_id"] in by_id:
            raise ValueError("Duplicate feature cache video_id: " + cache["video_id"])
        by_id[cache["video_id"]] = cache
        dimensions.add(cache["embeddings"].shape[1])
    if set(by_id) != {row["video_id"] for row in rows} or len(dimensions) != 1:
        raise ValueError("Exactly one compatible cache is required for every manifest video; no omissions/extras")
    values, labels = [], []
    for row in split["train"]:
        cache = by_id[row["video_id"]]
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
            "metadata": {"schema_version": 1, "feature_spec": dict(FEATURE_SPEC),
                         "train_video_ids": [r["video_id"] for r in split["train"]],
                         "holdout_video_ids": [r["video_id"] for r in split["holdout"]],
                         "train_angles": sorted(set(train_angles)), "holdout_angles": sorted(set(holdout_angles)),
                         "positive_radius_s": positive_radius_s, "seed": seed, "epochs": epochs,
                         "learning_rate": learning_rate, "l2": l2, "threshold": threshold,
                         "window_s": window_s, "n_supervised_train_frames": len(y)}}


def validate_model(model):
    np = numpy_module()
    meta = model.get("metadata", {})
    if (meta.get("schema_version") != 1 or meta.get("feature_spec") != FEATURE_SPEC
            or not meta.get("train_video_ids") or not meta.get("holdout_video_ids")
            or set(meta["train_video_ids"]) & set(meta["holdout_video_ids"])
            or not meta.get("train_angles") or not meta.get("holdout_angles")
            or set(meta["train_angles"]) & set(meta["holdout_angles"])):
        raise ValueError("Invalid model schema, feature specification or explicit grouped split")
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
    result["seen_in_training"] = cache["video_id"] in model["metadata"]["train_video_ids"]
    return result


def evaluate_holdout(model, caches, rows, train_angles, holdout_angles):
    """Diagnostic candidate predictions only; official metrics belong to the evaluator."""
    model = validate_model(model)
    split = split_videos(rows, train_angles, holdout_angles)
    meta = model["metadata"]
    if (set(train_angles) != set(meta["train_angles"])
            or set(holdout_angles) != set(meta["holdout_angles"])
            or {r["video_id"] for r in split["train"]} != set(meta["train_video_ids"])
            or {r["video_id"] for r in split["holdout"]} != set(meta["holdout_video_ids"])):
        raise ValueError("Evaluation must repeat the model's original explicit video/angle partition")
    by_id = {}
    for cache in caches:
        if cache["video_id"] in by_id:
            raise ValueError("Duplicate holdout prediction cache")
        by_id[cache["video_id"]] = cache
    if set(by_id) != {row["video_id"] for row in split["holdout"]}:
        raise ValueError("Holdout coverage must be complete and contain no train videos")
    predictions = []
    for row in split["holdout"]:
        result = predict_cache(model, by_id[row["video_id"]])
        result["reference_label"] = row["label"]
        result["angle_id"] = row["angle_id"]
        predictions.append(result)
    return {"schema_version": 1, "report_kind": "explicit_angle_holdout_candidates_not_legal_verdicts",
            "train_angles": list(meta["train_angles"]), "holdout_angles": list(meta["holdout_angles"]),
            "n_videos": len(predictions), "predictions": predictions,
            "note": "No missing predictions imputed; threshold is fixed before holdout evaluation"}
