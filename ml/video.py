"""Video probing and fixed-rate frame sampling (OpenCV)."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps


def _open(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {path}")
    return cap


MAX_PIXELS = 3840 * 2160  # reject absurd resolutions before decoding (memory safety)
# Top-level ISO-BMFF/QuickTime atoms. Same rule as backend/app/media.py, repeated here because OpenCV's
# FFmpeg backend ignores protocol whitelists: a playlist renamed to .mov must never reach VideoCapture.
_CONTAINER_ATOMS = frozenset({b"ftyp", b"moov", b"mdat", b"wide", b"free", b"skip"})


def _check_container(path: Path) -> None:
    with path.open("rb") as fh:
        head = fh.read(8)
    if len(head) < 8 or head[4:8] not in _CONTAINER_ATOMS:
        raise ValueError(f"not a mov/mp4 container: {path.name}")


def probe(path: Path | str, max_pixels: int = MAX_PIXELS) -> VideoInfo:
    _check_container(Path(path))
    cap = _open(Path(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    if fps <= 0 or count <= 0 or width <= 0:
        raise ValueError(f"not a readable video: {path}")
    if width * height > max_pixels:
        raise ValueError(f"video resolution {width}x{height} exceeds the supported maximum")
    return VideoInfo(fps=fps, frame_count=count, width=width, height=height)


def sample_frames(path: Path | str, fps_out: float) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (timestamp_sec, BGR frame) at roughly fps_out, decoding sequentially."""
    info = probe(path)
    step = info.fps / fps_out
    cap = _open(Path(path))
    try:
        next_idx = 0.0
        idx = 0
        while cap.grab():
            if idx >= round(next_idx):
                ok, frame = cap.retrieve()
                if ok:
                    yield idx / info.fps, frame
                next_idx += step
            idx += 1
    finally:
        cap.release()


def frames_at(path: Path | str, times: list[float]) -> dict[float, np.ndarray]:
    """Exact frames for the given timestamps in one sequential pass.

    Seeking (CAP_PROP_POS_MSEC / POS_FRAMES) returns wrong or missing frames on some of these .mov files,
    so we decode sequentially and use the same idx / fps timebase as sample_frames. Times outside the video
    are clamped to the first/last frame. A time whose frame cannot be decoded is absent from the result.
    """
    info = probe(path)
    wanted: dict[int, list[float]] = {}
    for t in times:
        idx = min(info.frame_count - 1, max(0, round(t * info.fps)))
        wanted.setdefault(idx, []).append(t)
    if not wanted:
        return {}

    found: dict[float, np.ndarray] = {}
    cap = _open(Path(path))
    try:
        idx = 0
        last_needed = max(wanted)
        while idx <= last_needed and cap.grab():
            if idx in wanted:
                ok, frame = cap.retrieve()
                if ok:
                    for t in wanted[idx]:
                        found[t] = frame
            idx += 1
    finally:
        cap.release()
    unreachable = [t for target, ts in wanted.items() if target >= idx for t in ts]
    if unreachable and idx > 0:
        # frame_count overstated the decodable frames: those requests get the real last frame (second pass, rare)
        tail = _last_decodable_frame(Path(path), idx - 1)
        if tail is not None:
            for t in unreachable:
                found[t] = tail
    return found


def _last_decodable_frame(path: Path, last_idx: int) -> np.ndarray | None:
    cap = _open(path)
    try:
        for _ in range(last_idx):
            if not cap.grab():
                return None
        ok, frame = cap.read()
    finally:
        cap.release()
    return frame if ok else None


def frame_at(path: Path | str, t_sec: float) -> np.ndarray | None:
    """Single exact frame by timestamp (see frames_at)."""
    return frames_at(path, [t_sec]).get(t_sec)
