"""Cheap appearance signatures for person re-identification inside one video (colour histograms of a box crop)."""

import cv2
import numpy as np

from .detect import Box

CROP_W, CROP_H = 16, 32
# Central part of the box: from above, the head/shoulders/torso sit there; the edges are mostly background.
CORE_X = (0.2, 0.8)
CORE_Y = (0.15, 0.75)
HUE_BINS, SAT_BINS, VAL_BINS = 8, 4, 4
MIN_SAT_FOR_HUE = 40  # below this saturation hue is noise, so such pixels only vote in the value histogram

Signature = tuple[float, ...]


def crop_person(frame: np.ndarray, box: Box) -> np.ndarray:
    """Box crop resized to a fixed CROP_H x CROP_W BGR patch (what the detection cache stores)."""
    h, w = frame.shape[:2]
    x0, y0 = int(max(0, min(w - 1, box[0]))), int(max(0, min(h - 1, box[1])))
    x1, y1 = int(max(x0 + 1, min(w, box[2]))), int(max(y0 + 1, min(h, box[3])))
    return cv2.resize(frame[y0:y1, x0:x1], (CROP_W, CROP_H), interpolation=cv2.INTER_AREA)


def color_signature(crop: np.ndarray) -> Signature:
    """L1-normalised [hue x saturation | value] histogram of the crop core; empty-safe."""
    ch, cw = crop.shape[:2]
    core = crop[int(ch * CORE_Y[0]):int(ch * CORE_Y[1]), int(cw * CORE_X[0]):int(cw * CORE_X[1])]
    hsv = cv2.cvtColor(core, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.int32)
    colored = hsv[hsv[:, 1] >= MIN_SAT_FOR_HUE]
    hs = np.zeros((HUE_BINS, SAT_BINS), dtype=np.float64)
    if len(colored):
        hue_idx = np.minimum(colored[:, 0] * HUE_BINS // 180, HUE_BINS - 1)
        sat_idx = np.minimum((colored[:, 1] - MIN_SAT_FOR_HUE) * SAT_BINS // (256 - MIN_SAT_FOR_HUE), SAT_BINS - 1)
        np.add.at(hs, (hue_idx, sat_idx), 1.0)
    val = np.bincount(np.minimum(hsv[:, 2] * VAL_BINS // 256, VAL_BINS - 1), minlength=VAL_BINS).astype(np.float64)
    hist = np.concatenate([hs.ravel(), val])
    total = hist.sum()
    return tuple(float(v) for v in (hist / total if total > 0 else hist))


def mean_signature(signatures: list[Signature]) -> Signature:
    if not signatures:
        return ()
    mean = np.mean(np.asarray(signatures, dtype=np.float64), axis=0)
    total = mean.sum()
    return tuple(float(v) for v in (mean / total if total > 0 else mean))


def signature_distance(a: Signature, b: Signature) -> float:
    """Hellinger distance in 0..1; 1.0 when either side has no signature."""
    if not a or not b or len(a) != len(b):
        return 1.0
    bc = float(np.sum(np.sqrt(np.asarray(a) * np.asarray(b))))
    return float(np.sqrt(max(0.0, 1.0 - bc)))
