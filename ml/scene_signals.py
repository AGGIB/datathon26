"""Pure image statistics behind the scene states: backgrounds, appearance change, laptop screen, smoothing.

Everything here works on numpy arrays only (no video IO, no models), so it is unit-tested with synthetic images.
"""

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median

import cv2
import numpy as np

from .detect import Box
from .scene_geometry import expand_box, to_int_box

# change_score: a cell counts as changed when its structure (ZNCC) or its brightness moved noticeably
CELL = 8
MIN_VALID_SHARE = 0.6
FLAT_STD = 4.0  # below this std a cell is "flat" (wall, cabinet face) and compared by mean only
FLAT_DIFF = 18.0
TEXTURE_JUMP = 6.0
ZNCC_MIN = 0.5
MEAN_DIFF = 30.0

# laptop screen
SCREEN_RATIO_OFF, SCREEN_RATIO_ON = 1.0, 1.15  # p95 of box luminance / p90 of context luminance
COLOR_FRAC_OFF, COLOR_FRAC_ON = 0.05, 0.25  # share of bright saturated pixels (wallpaper, lock screen)
WORK_FRAC_OFF, WORK_FRAC_ON = 0.04, 0.2  # share of white low-saturation pixels brighter than the surroundings
COLOR_MIN_SAT, COLOR_MIN_VAL = 110, 90
WORK_MAX_SAT, WORK_MARGIN = 70, 8.0


def ramp(x: float, lo: float, hi: float) -> float:
    """Linear 0..1 score: 0 at `lo`, 1 at `hi` (works in both directions), clipped."""
    if hi == lo:
        return float(x >= hi)
    return float(min(1.0, max(0.0, (x - lo) / (hi - lo))))


def person_mask(shape: tuple[int, int], boxes: Sequence[Box], pad_ratio: float = 0.08) -> np.ndarray:
    """Boolean HxW mask, True where a (padded) person box covers the frame."""
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    for b in boxes:
        x0, y0, x1, y1 = to_int_box(expand_box(b, pad_ratio, width=w, height=h))
        mask[y0:y1, x0:x1] = True
    return mask


def masked_median(frames: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Per-pixel median over frames where the pixel is not masked; plain median where it is always masked.

    frames: (N, H, W, C) uint8, masks: (N, H, W) bool (True = hidden by a person).
    """
    data = frames.astype(np.float32)
    data[np.broadcast_to(masks[..., None], data.shape)] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns are handled below
        med = np.nanmedian(data, axis=0)
    fallback = np.median(frames, axis=0)
    out = np.where(np.isnan(med), fallback, med)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def initial_background(
    frames: Sequence[np.ndarray], masks: Sequence[np.ndarray], fallback: np.ndarray | None = None
) -> np.ndarray:
    """The room as it was at the start: each pixel from the first frame where no person hides it.

    frames are in time order; pixels never visible take `fallback` (default: median of the frames).
    """
    out = np.zeros_like(frames[0])
    filled = np.zeros(masks[0].shape, dtype=bool)
    for frame, mask in zip(frames, masks):
        take = ~mask & ~filled
        if take.any():
            out[take] = frame[take]
            filled |= take
        if filled.all():
            return out
    if fallback is None:
        step = max(1, len(frames) // 15)
        fallback = np.median(np.stack(frames[::step]), axis=0).astype(frames[0].dtype)
    out[~filled] = fallback[~filled]
    return out


def _cells(arr: np.ndarray, cell: int) -> np.ndarray:
    h, w = (arr.shape[0] // cell) * cell, (arr.shape[1] // cell) * cell
    a = arr[:h, :w]
    return a.reshape(h // cell, cell, w // cell, cell).transpose(0, 2, 1, 3).reshape(h // cell, w // cell, -1)


def global_gain(cur: np.ndarray, ref: np.ndarray, valid: np.ndarray) -> float:
    """Brightness ratio cur/ref over visible pixels (lights switched off, auto exposure)."""
    if not valid.any():
        return 1.0
    ref_med = float(np.median(ref[valid]))
    return float(np.median(cur[valid])) / ref_med if ref_med > 1 else 1.0


def change_score(
    cur: np.ndarray, ref: np.ndarray, valid: np.ndarray, gain: float | None = None, cell: int = CELL
) -> float | None:
    """Share of visible cells whose appearance differs from the reference (grayscale uint8 arrays).

    Returns None when too little of the region is visible to say anything.
    """
    if cur.shape[0] < cell or cur.shape[1] < cell:
        return None
    if gain is None:
        gain = global_gain(cur, ref, valid)
    a = _cells(cur.astype(np.float32), cell)
    b = _cells(ref.astype(np.float32) * gain, cell)
    v = _cells(valid.astype(np.float32), cell)
    n = v.sum(-1)
    usable = n >= MIN_VALID_SHARE * cell * cell
    if not usable.any():
        return None
    n_safe = np.maximum(n, 1.0)
    ma = (a * v).sum(-1) / n_safe
    mb = (b * v).sum(-1) / n_safe
    da, db = a - ma[..., None], b - mb[..., None]
    sa = np.sqrt((da * da * v).sum(-1) / n_safe)
    sb = np.sqrt((db * db * v).sum(-1) / n_safe)
    zncc = (da * db * v).sum(-1) / n_safe / (sa * sb + 1e-6)
    mean_diff = np.abs(ma - mb)
    flat = np.maximum(sa, sb) < FLAT_STD
    half_flat = ~flat & (np.minimum(sa, sb) < FLAT_STD)
    textured = ~flat & ~half_flat
    changed = (
        (flat & (mean_diff > FLAT_DIFF))
        | (half_flat & ((np.abs(sa - sb) > TEXTURE_JUMP) | (mean_diff > FLAT_DIFF)))
        | (textured & ((zncc < ZNCC_MIN) | (mean_diff > MEAN_DIFF)))
    )
    return float((changed & usable).sum() / usable.sum())


@dataclass(frozen=True)
class ScreenFeatures:
    on: float  # 0..1: the screen emits light (any content, including a lock screen)
    work: float  # 0..1: bright white document-like content (spreadsheet, form) rather than wallpaper
    brightness_ratio: float
    color_frac: float
    white_frac: float


def screen_features(box_bgr: np.ndarray, context_bgr: np.ndarray) -> ScreenFeatures:
    """Score a laptop crop against its surroundings (desk and wall around the laptop)."""
    lum = cv2.cvtColor(box_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ctx = cv2.cvtColor(context_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(box_bgr, cv2.COLOR_BGR2HSV)
    sat, val = hsv[..., 1], hsv[..., 2]
    ctx_hi = float(np.percentile(ctx, 90))
    ratio = float(np.percentile(lum, 95)) / max(ctx_hi, 10.0)
    color_frac = float(((sat >= COLOR_MIN_SAT) & (val >= COLOR_MIN_VAL)).mean())
    white_frac = float(((lum > ctx_hi + WORK_MARGIN) & (sat <= WORK_MAX_SAT)).mean())
    on = max(ramp(ratio, SCREEN_RATIO_OFF, SCREEN_RATIO_ON), ramp(color_frac, COLOR_FRAC_OFF, COLOR_FRAC_ON))
    work = ramp(white_frac, WORK_FRAC_OFF, WORK_FRAC_ON)
    return ScreenFeatures(on=on, work=min(work, max(on, work)), brightness_ratio=ratio,
                          color_frac=color_frac, white_frac=white_frac)


def hold_fill(values: Sequence[float | None]) -> list[float | None]:
    """Carry the last seen value forward over gaps (occlusion), then back-fill the leading gap."""
    out: list[float | None] = []
    last: float | None = None
    for v in values:
        last = v if v is not None else last
        out.append(last)
    first = next((v for v in out if v is not None), None)
    return [first if v is None else v for v in out]


def median_smooth(values: Sequence[float | None], k: int) -> list[float | None]:
    """Running median over the non-missing neighbours (window k); missing entries stay missing."""
    half = k // 2
    out: list[float | None] = []
    for i, v in enumerate(values):
        if v is None:
            out.append(None)
            continue
        window = [x for x in values[max(0, i - half): i + half + 1] if x is not None]
        out.append(float(median(window)))
    return out
