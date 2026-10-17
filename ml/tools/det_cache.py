"""Dev tool: cache Detector output per video under var/cache/ so event logic can be iterated without the GPU.

    python -m ml.tools.det_cache [--fps 6] [--only X-30,X-33]

Each cache file keeps, per sampled frame, every detection plus a small BGR crop of each person box
(ml.appearance.crop_person), so appearance features can change without re-running the detector.
Caches never leave var/ (gitignored).
"""

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ml.appearance import CROP_H, CROP_W, crop_person
from ml.detect import NMS_IOU, Detection, _nms
from ml.labels import normalize_video_name
from ml.video import VideoInfo, probe, sample_frames

CACHE_DIR = Path("var/cache/detections")
VIDEOS = Path("data/open/videos")
BATCH = 8
FORMAT_VERSION = 2

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CachedFrame:
    t: float
    detections: tuple[Detection, ...]
    crops: tuple[np.ndarray | None, ...]  # aligned with detections; None for non-person labels


@dataclass(frozen=True)
class CachedVideo:
    name: str
    info: VideoInfo
    sample_fps: float
    frames: tuple[CachedFrame, ...]
    detect_sec: float  # wall time the detector pass took when the cache was built
    thumb: np.ndarray  # one full-size frame for local debugging


def cache_path(name: str, fps: float, cache_dir: Path = CACHE_DIR) -> Path:
    return cache_dir / f"{name}_{fps:g}fps_v{FORMAT_VERSION}.npz"


def list_videos(videos_dir: Path = VIDEOS) -> list[Path]:
    return sorted(videos_dir.glob("*.mov"), key=lambda p: int(normalize_video_name(p.name).split("-")[1]))


def build(video_path: Path, detector, fps: float, cache_dir: Path = CACHE_DIR) -> Path:
    name = normalize_video_name(video_path.name)
    info = probe(video_path)
    rows: list[tuple[int, Detection]] = []
    crops: list[np.ndarray] = []
    crop_rows: list[int] = []
    times: list[float] = []
    thumb = None
    batch: list[tuple[float, np.ndarray]] = []
    started = time.monotonic()

    def flush() -> None:
        for (t, frame), dets in zip(batch, detector.detect([f for _, f in batch])):
            fi = len(times)
            times.append(t)
            for d in dets:
                if d.label == "person":
                    crop_rows.append(len(rows))
                    crops.append(crop_person(frame, d.box))
                rows.append((fi, d))
        batch.clear()

    for t, frame in sample_frames(video_path, fps):
        if thumb is None and t >= 0.5:
            thumb = frame.copy()
        batch.append((t, frame))
        if len(batch) == BATCH:
            flush()
    if batch:
        flush()
    elapsed = time.monotonic() - started

    labels = sorted({d.label for _, d in rows}) or ["person"]
    out = cache_path(name, fps, cache_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        meta=np.array([info.fps, info.frame_count, info.width, info.height, fps, elapsed], dtype=np.float64),
        times=np.array(times, dtype=np.float64),
        det_frame=np.array([fi for fi, _ in rows], dtype=np.int32),
        det_box=np.array([d.box for _, d in rows], dtype=np.float32).reshape(-1, 4),
        det_score=np.array([d.score for _, d in rows], dtype=np.float32),
        det_label=np.array([labels.index(d.label) for _, d in rows], dtype=np.int16),
        labels=np.array(labels),
        crop_row=np.array(crop_rows, dtype=np.int32),
        crops=np.array(crops, dtype=np.uint8).reshape(-1, CROP_H, CROP_W, 3),
        thumb=thumb if thumb is not None else np.zeros((1, 1, 3), np.uint8),
    )
    log.info("%s: %d frames, %d detections, %.1fs -> %s", name, len(times), len(rows), elapsed, out)
    return out


def load(path: Path, name: str = "") -> CachedVideo:
    z = np.load(path)
    fps, count, width, height, sample_fps, elapsed = z["meta"].tolist()
    labels = [str(x) for x in z["labels"]]
    crop_of = {int(r): z["crops"][i] for i, r in enumerate(z["crop_row"])}
    per_frame: list[list[tuple[Detection, np.ndarray | None]]] = [[] for _ in z["times"]]
    for row, (fi, box, score, li) in enumerate(zip(z["det_frame"], z["det_box"], z["det_score"], z["det_label"])):
        det = Detection(box=tuple(float(v) for v in box), score=float(score), label=labels[int(li)])
        per_frame[int(fi)].append((det, crop_of.get(row)))
    frames = tuple(_frame(float(t), items) for t, items in zip(z["times"], per_frame))
    info = VideoInfo(fps=fps, frame_count=int(count), width=int(width), height=int(height))
    return CachedVideo(name or path.stem.split("_")[0], info, sample_fps, frames, elapsed, z["thumb"])


def _frame(t: float, items: list[tuple[Detection, np.ndarray | None]]) -> CachedFrame:
    """Re-apply the detector's current NMS so caches built by an older detect.py match today's output."""
    kept = {id(d) for d in _nms([d for d, _ in items], NMS_IOU)}
    items = [(d, c) for d, c in items if id(d) in kept]
    return CachedFrame(t, tuple(d for d, _ in items), tuple(c for _, c in items))


def load_or_build(video_path: Path, fps: float, detector_factory=None, cache_dir: Path = CACHE_DIR) -> CachedVideo:
    name = normalize_video_name(video_path.name)
    path = cache_path(name, fps, cache_dir)
    if not path.exists():
        if detector_factory is None:
            raise FileNotFoundError(f"no detection cache for {name} at {fps:g} fps: {path}")
        build(video_path, detector_factory(), fps, cache_dir)
    return load(path, name)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=float, default=6.0)
    ap.add_argument("--only", default="")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    wanted = {v.strip().upper() for v in args.only.split(",") if v.strip()}

    from ml.detect import Detector

    detector = Detector()
    for path in list_videos():
        name = normalize_video_name(path.name)
        if wanted and name not in wanted:
            continue
        if cache_path(name, args.fps).exists() and not args.force:
            continue
        build(path, detector, args.fps)


if __name__ == "__main__":
    main()
