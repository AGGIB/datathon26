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


def probe(path: Path | str) -> VideoInfo:
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


def frame_at(path: Path | str, t_sec: float) -> np.ndarray | None:
    """Single frame by timestamp (for evidence images)."""
    cap = _open(Path(path))
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t_sec) * 1000)
        ok, frame = cap.read()
    finally:
        cap.release()
    return frame if ok else None
