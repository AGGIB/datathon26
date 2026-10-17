"""Dependency-free baseline contracts: literal IDs and explicit grouping."""
import math
import json
import re
from pathlib import Path, PureWindowsPath

from ml.labels import normalize_video_name

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "google/siglip-base-patch16-224"
FEATURE_SPEC = {"model_id": MODEL_ID, "feature": "vision.pooler_output",
                "preprocess": "ffmpeg-autorotate-rgb224+AutoImageProcessor", "version": 1}


def artifact_path(path, root=None):
    """Reject caches/models/reports outside <root>/artifacts, including symlink escapes."""
    base = (Path(root) if root is not None else REPO_ROOT).resolve() / "artifacts"
    target = Path(path).resolve()
    try:
        relative = target.relative_to(base)
    except ValueError as exc:
        raise ValueError("Output must be inside artifacts/: " + str(target)) from exc
    if not relative.parts:
        raise ValueError("Output must name a file within artifacts/")
    return target


def write_json(path, payload):
    target = Path(path)
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def read_json(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def sample_timestamps(timestamps, fps=1.0, max_frames=3600):
    """Select real PTS, never synthesize times from frame number / nominal FPS."""
    if not math.isfinite(fps) or fps <= 0 or not isinstance(max_frames, int) or max_frames <= 0:
        raise ValueError("fps must be finite/positive; max_frames must be a positive integer")
    selected = []
    previous = -math.inf
    for value in timestamps:
        value = float(value)
        if not math.isfinite(value) or value < 0 or value <= previous:
            raise ValueError("Frame timestamps must be finite, nonnegative and strictly increasing")
        previous = value
        if not selected or value - selected[-1] >= 1.0 / fps - 1e-9:
            selected.append(value)
            if len(selected) == max_frames:
                break
    return selected


def canonical_video_id(value):
    """Validate a literal filename/token BEFORE using the shared canonical join."""
    if (not isinstance(value, str) or not value or value != value.strip() or value.endswith(".")
            or any(c in value for c in '/\\:<>"|?*') or any(c.isspace() for c in value)
            or any(ord(c) < 32 or ord(c) == 127 for c in value)
            or value in {".", ".."} or PureWindowsPath(value).is_reserved()):
        raise ValueError("video_id must be a well-formed literal filename or ID, not a path")
    stem = Path(value).stem
    if not stem or stem != stem.strip() or not stem.strip(". "):
        raise ValueError("video_id must have a nonempty, unpadded stem")
    return normalize_video_name(value)


def content_sha256(value):
    """Validate a digest token; this does not hash or authenticate source bytes."""
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Every video requires a literal lowercase content SHA256")
    return value


def split_videos(rows, train_angles, holdout_angles, validation_angles=None):
    """Explicit content-isolated partition; two-way mode is diagnostic ONLY."""
    rows = list(rows)
    groups = {"train": list(train_angles), "holdout": list(holdout_angles)}
    if validation_angles is not None:
        groups["validation"] = list(validation_angles)
    if any(not isinstance(a, str) or not a or a != a.strip()
           or a.lower() in {"unknown", "неизвестно", "unassigned"}
           for values in groups.values() for a in values):
        raise ValueError("Unknown/empty angle IDs cannot define a holdout")
    groups = {side: set(values) for side, values in groups.items()}
    assigned = set()
    for values in groups.values():
        if not values or values & assigned:
            raise ValueError("Explicit nonempty disjoint train/validation/holdout angle IDs are required")
        assigned.update(values)
    result = {side: [] for side in groups}
    seen, angles = set(), set()
    content_partitions, content_labels = {}, {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Every manifest row must be a video object")
        video_id, angle = row.get("video_id"), row.get("angle_id")
        key = canonical_video_id(video_id)
        if key in seen:
            raise ValueError("Every video_id must be unique under canonical normalization")
        seen.add(key)
        if not isinstance(angle, str) or angle not in assigned:
            raise ValueError("Every video needs an explicitly assigned angle_id: " + video_id)
        if type(row.get("label")) is not int or row["label"] not in (0, 1):
            raise ValueError("label must be integer 0 or 1: " + video_id)
        times = row.get("positive_times_s", [])
        if not isinstance(times, list) or (row["label"] == 1 and not times):
            raise ValueError("Positive videos require explicit positive_times_s: " + video_id)
        if any(type(t) not in (int, float) or not math.isfinite(t) or t < 0 for t in times):
            raise ValueError("Positive timecodes must be finite nonnegative seconds")
        if row["label"] == 0 and times:
            raise ValueError("Negative videos cannot contain positive timecodes")
        angles.add(angle)
        side = next(side for side, values in groups.items() if angle in values)
        digest = content_sha256(row.get("content_sha256"))
        if digest in content_partitions and content_partitions[digest] != side:
            raise ValueError("Identical video content cannot cross train/validation/holdout")
        if digest in content_labels and content_labels[digest] != row["label"]:
            raise ValueError("Identical video content has conflicting binary labels")
        content_partitions[digest] = side
        content_labels[digest] = row["label"]
        result[side].append(dict(row))
    if angles != assigned or not all(result.values()):
        raise ValueError("Every requested angle must have videos; all split partitions must be nonempty")
    return result


def segment_labels(row, timestamps, radius_s=2.0):
    """Unannotated time in a positive video is UNKNOWN, not negative."""
    if not math.isfinite(radius_s) or radius_s < 0:
        raise ValueError("positive radius must be finite and nonnegative")
    if row["label"] == 0:
        return [0 for _ in timestamps]
    labels = [1 if any(abs(float(t) - event) <= radius_s + 1e-9
                       for event in row["positive_times_s"]) else None for t in timestamps]
    if 1 not in labels:
        raise ValueError("No sampled frame near the annotated event: " + row["video_id"])
    return labels


def aggregate_predictions(video_id, timestamps, probabilities, threshold=0.5, window_s=2.0):
    """Mean within fixed time windows, max across windows; earliest tied peak PTS."""
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    if not math.isfinite(window_s) or window_s <= 0:
        raise ValueError("window_s must be finite and positive")
    times, scores = list(timestamps), list(probabilities)
    if not times or len(times) != len(scores):
        raise ValueError("Missing/unaligned predictions are errors, never no_violation")
    buckets, previous = {}, -math.inf
    for time, score in zip(times, scores):
        time, score = float(time), float(score)
        if not math.isfinite(time) or time < 0 or time <= previous:
            raise ValueError("Prediction timestamps must be strictly increasing real PTS")
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Prediction probabilities must be finite in [0, 1]")
        previous = time
        buckets.setdefault(math.floor(time / window_s), []).append((time, score))
    windows = []
    for index, entries in sorted(buckets.items()):
        peak_time, _ = max(entries, key=lambda item: item[1])
        windows.append({"start_s": index * window_s, "end_s": (index + 1) * window_s,
                        "score": sum(p for _, p in entries) / len(entries),
                        "peak_time_s": peak_time, "n_frames": len(entries)})
    best = max(windows, key=lambda window: window["score"])
    candidate = best["score"] >= threshold
    time = best["peak_time_s"] if candidate else None
    timecode = None
    if time is not None:
        milliseconds = round(time * 1000)
        seconds, millis = divmod(milliseconds, 1000)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        timecode = f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"
    return {"schema_version": 1, "video_id": video_id, "source": "siglip_logistic_baseline",
            "verdict": "candidate" if candidate else "no_candidate",
            "video_score": best["score"], "threshold": threshold,
            "candidate_time_s": time, "candidate_timecode": timecode,
            "description": "Кандидат baseline; требуется проверка" if candidate
            else "Baseline не выделил кандидата; требуется проверка",
            "final_legal_label": None, "windows": windows}
