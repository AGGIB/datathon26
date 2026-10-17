"""Evidence frames for К1: video frames at given times, with highlight boxes, as base64 JPEG.

No face processing of any kind happens here (no detection, blurring or recognition):
frames are passed through as recorded, only boxes from the rules stage are drawn on top.
"""

import base64
import logging
import math
from collections.abc import Iterable, Sequence
from pathlib import Path

import cv2
import numpy as np

from .candidates import Candidate
from .detect import Box
from .video import frame_at, probe

log = logging.getLogger(__name__)

MAX_FRAMES = 3
MAX_FRAME_BYTES = 150 * 1024
MAX_SIDE_PX = 1280
JPEG_QUALITIES = (85, 70, 55, 40)
DOWNSCALE_STEPS = (1.0, 0.75, 0.5, 0.35, 0.25)
TIME_DECIMALS = 1

BOX_COLOR = (0, 215, 255)  # BGR amber: readable on both dark floors and white desks
OUTLINE_COLOR = (0, 0, 0)
LINE_PX_PER_SIDE = 1 / 160  # line thickness relative to the shorter frame side


def _clip_box(box: Box, width: int, height: int) -> tuple[int, int, int, int] | None:
    if not all(math.isfinite(v) for v in box):
        return None
    x0, x1 = sorted((box[0], box[2]))
    y0, y1 = sorted((box[1], box[3]))
    x0, x1 = max(0, round(x0)), min(width - 1, round(x1))
    y0, y1 = max(0, round(y0)), min(height - 1, round(y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def draw_boxes(frame: np.ndarray, boxes: Iterable[Box]) -> np.ndarray:
    """Copy of `frame` with each box outlined; boxes outside the frame are clipped or skipped."""
    out = frame.copy()
    height, width = out.shape[:2]
    thickness = max(2, round(min(height, width) * LINE_PX_PER_SIDE))
    for box in boxes:
        clipped = _clip_box(box, width, height)
        if clipped is None:
            log.debug("evidence: box %s is outside a %dx%d frame, skipped", box, width, height)
            continue
        x0, y0, x1, y1 = clipped
        cv2.rectangle(out, (x0, y0), (x1, y1), OUTLINE_COLOR, thickness + 2, cv2.LINE_AA)
        cv2.rectangle(out, (x0, y0), (x1, y1), BOX_COLOR, thickness, cv2.LINE_AA)
    return out


def _resize_to(frame: np.ndarray, scale: float) -> np.ndarray:
    if scale >= 1.0:
        return frame
    height, width = frame.shape[:2]
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def encode_jpeg_b64(frame: np.ndarray, max_bytes: int = MAX_FRAME_BYTES) -> str:
    """Base64 JPEG (no data: prefix) not larger than max_bytes, lowering quality, then resolution."""
    if frame.size == 0:
        raise ValueError("cannot encode an empty frame")
    base_scale = min(1.0, MAX_SIDE_PX / max(frame.shape[:2]))
    for step in DOWNSCALE_STEPS:
        resized = _resize_to(frame, base_scale * step)
        for quality in JPEG_QUALITIES:
            ok, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if ok and buf.size <= max_bytes:
                return base64.b64encode(buf.tobytes()).decode("ascii")
    raise ValueError(f"frame {frame.shape} does not fit into {max_bytes} bytes as JPEG")


def _clamp_times(times: Sequence[float], duration: float, fps: float) -> list[float]:
    last = max(0.0, duration - 1.0 / fps)
    clamped: list[float] = []
    for t in times:
        value = round(min(max(0.0, float(t)), last), TIME_DECIMALS)
        if value not in clamped:
            clamped.append(value)
    return clamped[:MAX_FRAMES]


def render_evidence(video_path: Path | str, times: Sequence[float], boxes: Sequence[Box] = ()) -> list[dict]:
    """К1 evidence_frames: [{timestamp_sec, image_base64}], up to 3, in the order of `times`."""
    info = probe(video_path)
    frames = []
    for t in _clamp_times(times, info.duration, info.fps):
        image = frame_at(video_path, t)
        if image is None:
            log.warning("evidence: no frame at %.1fs in %s", t, Path(video_path).name)
            continue
        frames.append({"timestamp_sec": t, "image_base64": encode_jpeg_b64(draw_boxes(image, boxes))})
    return frames


def render_candidate(video_path: Path | str, candidate: Candidate) -> list[dict]:
    return render_evidence(video_path, candidate.evidence_times, candidate.highlight_boxes)
