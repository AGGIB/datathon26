"""Evidence frames for К1: video frames at the candidates' times, with highlight boxes, as base64 JPEG.

All frames of a job come from ONE sequential decode pass (ml.video.frames_at): seeking returns wrong or
missing frames on part of the dataset, and sequential indices match what the detector saw.
No face processing of any kind happens here (no detection, blurring or recognition):
frames are passed through as recorded, only boxes from the rules stage are drawn on top.
"""

import base64
import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import cv2
import numpy as np

from .assemble import clamp_time
from .candidates import Candidate
from .detect import Box
from .video import frames_at

log = logging.getLogger(__name__)

MAX_FRAMES = 3
MAX_FRAME_BYTES = 150 * 1024  # of the base64 text, i.e. ~112 KB of JPEG
MAX_SIDE_PX = 1280
JPEG_QUALITIES = (85, 70, 55, 40)
DOWNSCALE_STEPS = (1.0, 0.75, 0.5, 0.35, 0.25)
PEAK_FALLBACK_SEC = 0.5  # if the t_peak frame is unusable, the nearest of t_peak ∓ this is shown instead

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
    """Base64 JPEG (no data: prefix) whose text fits max_bytes, lowering quality, then resolution."""
    max_jpeg_bytes = max_bytes * 3 // 4  # base64 turns every 3 bytes into 4 characters
    if frame.size == 0:
        raise ValueError("cannot encode an empty frame")
    base_scale = min(1.0, MAX_SIDE_PX / max(frame.shape[:2]))
    for step in DOWNSCALE_STEPS:
        resized = _resize_to(frame, base_scale * step)
        for quality in JPEG_QUALITIES:
            ok, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if ok and buf.size <= max_jpeg_bytes:
                return base64.b64encode(buf.tobytes()).decode("ascii")
    raise ValueError(f"frame {frame.shape} does not fit into {max_bytes} bytes as JPEG")


def _peak_fallbacks(candidate: Candidate, duration: float) -> tuple[float, float]:
    return (clamp_time(candidate.t_peak - PEAK_FALLBACK_SEC, duration),
            clamp_time(candidate.t_peak + PEAK_FALLBACK_SEC, duration))


def wanted_times(candidates: Sequence[Candidate], duration: float) -> list[float]:
    """Every timestamp the evidence of these candidates may need (reported precision, inside the video)."""
    times: set[float] = set()
    for c in candidates:
        times.update(clamp_time(t, duration) for t in c.evidence_times)
        times.update(_peak_fallbacks(c, duration))
    return sorted(times)


def _encode(image: np.ndarray, boxes: Sequence[Box], t: float) -> dict | None:
    try:
        return {"timestamp_sec": t, "image_base64": encode_jpeg_b64(draw_boxes(image, boxes))}
    except (ValueError, cv2.error):
        log.exception("evidence: frame at %.1fs could not be encoded", t)
        return None


def _peak_frame(candidate: Candidate, frames: Mapping[float, np.ndarray], duration: float) -> dict | None:
    """t_peak with the rule's boxes (valid only there); else the nearest neighbour without boxes."""
    t = clamp_time(candidate.t_peak, duration)
    if t in frames and (rendered := _encode(frames[t], candidate.highlight_boxes, t)):
        return rendered
    for alt in _peak_fallbacks(candidate, duration):
        if alt in frames and (rendered := _encode(frames[alt], (), alt)):
            log.warning("evidence: %s frame at %.1fs unusable, showing %.1fs", candidate.type_id, t, alt)
            return rendered
    return None


def frames_for(candidate: Candidate, frames: Mapping[float, np.ndarray], duration: float) -> list[dict]:
    """К1 evidence_frames for one candidate: t_peak first, then its other evidence times (no boxes)."""
    out = [f for f in [_peak_frame(candidate, frames, duration)] if f]
    for t in (clamp_time(t, duration) for t in candidate.evidence_times[1:]):
        if t in frames and all(f["timestamp_sec"] != t for f in out) and (rendered := _encode(frames[t], (), t)):
            out.append(rendered)
    return out[:MAX_FRAMES]


def render_candidates(
    video_path: Path | str, candidates: Sequence[Candidate], duration: float
) -> dict[Candidate, list[dict]]:
    """Evidence for all candidates of a job from a single decode pass; times are clamped to `duration`."""
    if not candidates:
        return {}
    frames = frames_at(video_path, wanted_times(candidates, duration))
    return {c: frames_for(c, frames, duration) for c in candidates}
