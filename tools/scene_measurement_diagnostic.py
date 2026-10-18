"""Opt-in door observability sidecar for already prepared numerical scene inputs.

No decoding, file/network I/O, model calls or production wiring. The caller must
supply exactly the grayscale frames, visible masks, gains, lighting rejects and
ROI used by scene._region_series. This cannot reconstruct lost provenance from
cached final SceneState values. It adds no door state policy or threshold.

Raw/smoothed changes and production finished scores are separate quantities.
A missing current measurement remains unknown even when the finished score is
carried. Median smoothing may select/mix neighboring and future measurements;
raw_measurement_age is never a claim about final-score ancestry. Reference mode
says how an appearance was chosen, never that the door was physically closed.
Source indices below are zero-based positions in supplied arrays, not container
frame indices or verified PTS. The serialized sidecar contains no image arrays.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
from typing import Sequence

import numpy as np

from ml.scene import DOOR_CHANGE, MIN_ROI_VISIBLE, SMOOTH_K, TRUST_START_SEC, _finish, _region_series
from ml.scene_signals import CELL, MIN_VALID_SHARE, _cells, median_smooth, region_reference


@dataclass(frozen=True)
class DoorSampleDiagnostic:
    supplied_sample_index: int
    timestamp_s: float
    raw_change: float | None
    smoothed_change: float | None
    finished_score: float | None
    current_raw_valid: bool
    current_smoothed_available: bool
    current_measurement_status: str
    rejection_reason: str | None
    current_visible_fraction: float | None
    reference_visible_fraction: float | None
    common_visible_fraction: float | None
    usable_cells: int
    total_cells: int
    lighting_rejected: bool
    global_gain: float
    last_raw_measurement_index: int | None
    last_raw_measurement_timestamp_s: float | None
    raw_measurement_age_s: float | None
    smoothing_window_indices: tuple[int, ...]
    smoothing_uses_future: bool
    held_from_smoothed: bool
    final_from_smoothed_index: int | None
    final_smoothing_window_indices: tuple[int, ...]
    final_ancestry_kind: str


@dataclass(frozen=True)
class DoorMeasurementDiagnostic:
    reference_mode: str
    reference_source_indices: tuple[int, ...]
    reference_source_timestamps_s: tuple[float, ...]
    reference_identity_sha256: str | None
    reference_sources_lighting_rejected: tuple[int, ...]
    reference_source_global_gains: tuple[float, ...]
    reference_semantic_state: str
    roi: tuple[int, int, int, int] | None
    samples: tuple[DoorSampleDiagnostic, ...]
    timebase: str = 'caller_sample_timestamps_not_verified_container_PTS'
    production_decisions_modified: bool = False


def _finite(value):
    return isinstance(value, Real) and not isinstance(value, (bool, np.bool_)) and math.isfinite(value)


def _inputs(times, grays, visible, roi, gains, rejected):
    count = len(times)
    if (count == 0 or any(len(values) != count for values in (grays, visible, gains, rejected))
            or any(not _finite(t) or t < 0 for t in times) or list(times) != sorted(set(times))
            or any(not _finite(gain) or gain < 0 for gain in gains)
            or any(not isinstance(flag, (bool, np.bool_)) for flag in rejected)):
        raise ValueError('invalid_door_diagnostic_inputs')
    first = grays[0]
    if (not isinstance(first, np.ndarray) or first.ndim != 2 or not first.size
            or any(not isinstance(gray, np.ndarray) or gray.shape != first.shape or gray.dtype != np.uint8
                   for gray in grays)
            or any(not isinstance(mask, np.ndarray) or mask.shape != first.shape or mask.dtype != np.bool_
                   for mask in visible)):
        raise ValueError('invalid_door_diagnostic_arrays')
    if roi is not None:
        if (not isinstance(roi, tuple) or len(roi) != 4
                or any(not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) for value in roi)):
            raise ValueError('invalid_door_diagnostic_roi')
        x0, y0, x1, y1 = roi
        if not (0 <= x0 < x1 <= first.shape[1] and 0 <= y0 < y1 <= first.shape[0]):
            raise ValueError('invalid_door_diagnostic_roi')


def _reference(times, grays, visible, roi):
    if roi is None:
        return None, None, 'none', (), None
    ref, valid, mode = region_reference(grays, visible, times, roi, MIN_ROI_VISIBLE, TRUST_START_SEC)
    if ref is None:
        return None, None, mode, (), None
    x0, y0, x1, y1 = roi
    clean = tuple(i for i, mask in enumerate(visible) if mask[y0:y1, x0:x1].mean() >= MIN_ROI_VISIBLE)
    indices = clean[:1] if mode == 'start' else clean
    digest = hashlib.sha256()
    digest.update(json.dumps({'mode': mode, 'indices': indices, 'times': [float(times[i]) for i in indices],
                              'roi': [int(value) for value in roi]}, sort_keys=True).encode())
    digest.update(ref.tobytes(order='C'))
    digest.update(valid.tobytes(order='C'))
    return ref, valid, mode, indices, digest.hexdigest()


def diagnose_door(times: Sequence[float], grays: Sequence[np.ndarray], visible_masks: Sequence[np.ndarray],
                  roi: tuple[int, int, int, int] | None, global_gains: Sequence[float],
                  lighting_reject: Sequence[bool]) -> DoorMeasurementDiagnostic:
    """Describe exact existing measurement/smoothing/holding, without changing it.

    `visible_masks` has True for visible pixels, as in scene._region_series.
    Lighting rejection is supplied from the owner; it is not guessed here.
    A median reference is a composite with multiple source indices and no single
    fabricated timestamp. Fingerprints identify numerical reference inputs, not
    verified physical states or semantic truth.
    """
    _inputs(times, grays, visible_masks, roi, global_gains, lighting_reject)
    ref, ref_valid, mode, reference_indices, identity = _reference(times, grays, visible_masks, roi)
    raw, actual_mode = _region_series(grays, visible_masks, times, roi, global_gains, lighting_reject)
    if actual_mode != mode:
        raise RuntimeError('door_reference_reconstruction_mismatch')
    smoothed = median_smooth(raw, SMOOTH_K)
    finished = _finish(raw, DOOR_CHANGE)
    samples, last_raw, last_smoothed = [], None, None
    half = SMOOTH_K // 2
    windows = [tuple(j for j in range(max(0, i - half), min(len(raw), i + half + 1)) if raw[j] is not None)
               if raw[i] is not None else () for i in range(len(raw))]
    reference_visible = float(ref_valid.mean()) if ref_valid is not None else None
    for i, t in enumerate(times):
        current_visible, common_visible, usable, total = None, None, 0, 0
        if roi is not None:
            x0, y0, x1, y1 = roi
            current = visible_masks[i][y0:y1, x0:x1]
            current_visible = float(current.mean())
            if ref_valid is not None:
                common = current & ref_valid
                common_visible = float(common.mean())
                if min(common.shape) >= CELL:
                    cells = _cells(common.astype(np.float32), CELL)
                    total = cells.shape[0] * cells.shape[1]
                    usable = int((cells.sum(-1) >= MIN_VALID_SHARE * CELL * CELL).sum())
        if raw[i] is not None:
            last_raw = i
        if smoothed[i] is not None:
            last_smoothed = i
        if roi is None:
            reason = 'no_roi'
        elif ref is None:
            reason = 'no_reference'
        elif lighting_reject[i]:
            reason = 'lighting_rejected'
        elif current_visible < MIN_ROI_VISIBLE:
            reason = 'current_roi_occluded'
        elif raw[i] is None:
            reason = 'no_usable_cells'
        else:
            reason = None
        final_window = windows[last_smoothed] if last_smoothed is not None else ()
        ancestry = ('unknown' if not final_window else 'single_raw_window'
                    if len(final_window) == 1 else 'mixed_or_median_selected')
        samples.append(DoorSampleDiagnostic(
            supplied_sample_index=i, timestamp_s=float(t), raw_change=raw[i], smoothed_change=smoothed[i],
            finished_score=finished[i], current_raw_valid=raw[i] is not None,
            current_smoothed_available=smoothed[i] is not None,
            current_measurement_status='measured_change' if raw[i] is not None else 'unknown',
            rejection_reason=reason, current_visible_fraction=current_visible,
            reference_visible_fraction=reference_visible, common_visible_fraction=common_visible,
            usable_cells=usable, total_cells=total, lighting_rejected=bool(lighting_reject[i]),
            global_gain=float(global_gains[i]),
            last_raw_measurement_index=last_raw,
            last_raw_measurement_timestamp_s=float(times[last_raw]) if last_raw is not None else None,
            raw_measurement_age_s=float(t - times[last_raw]) if last_raw is not None else None,
            smoothing_window_indices=windows[i], smoothing_uses_future=any(j > i for j in windows[i]),
            held_from_smoothed=smoothed[i] is None and last_smoothed is not None,
            final_from_smoothed_index=last_smoothed, final_smoothing_window_indices=final_window,
            final_ancestry_kind=ancestry))
    return DoorMeasurementDiagnostic(mode, reference_indices, tuple(float(times[i]) for i in reference_indices),
        identity, tuple(i for i in reference_indices if lighting_reject[i]),
        tuple(float(global_gains[i]) for i in reference_indices), 'unknown',
        tuple(int(value) for value in roi) if roi is not None else None, tuple(samples))
