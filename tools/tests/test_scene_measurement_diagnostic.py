"""Opt-in observability tests use generated arrays only, with no files or models."""
from dataclasses import asdict
import json
import subprocess
import sys

import numpy as np
import pytest

from ml.scene import DOOR_CHANGE, _finish, _region_series
from tools.scene_measurement_diagnostic import diagnose_door


def inputs():
    times = list(map(float, range(8)))
    grays = [np.full((16, 16), 100 if i < 2 else 200, dtype=np.uint8) for i in range(8)]
    visible = [np.ones((16, 16), dtype=bool) for _ in times]
    return times, grays, visible, (0, 0, 16, 16), [1.] * 8, [False] * 8


def canonical_scores(args):
    times, grays, visible, roi, gains, skipped = args
    raw, mode = _region_series(grays, visible, times, roi, gains, skipped)
    return raw, mode, _finish(raw, DOOR_CHANGE)


def assert_production_exact(args, diagnostic):
    raw, mode, finished = canonical_scores(args)
    assert [row.raw_change for row in diagnostic.samples] == raw
    assert [row.finished_score for row in diagnostic.samples] == finished
    assert diagnostic.reference_mode == mode
    assert not diagnostic.production_decisions_modified


def test_sidecar_uses_existing_measurement_and_finish_bitexact():
    args = inputs()
    before = [(gray.tobytes(), mask.tobytes()) for gray, mask in zip(args[1], args[2], strict=True)]
    diagnostic = diagnose_door(*args)
    assert_production_exact(args, diagnostic)
    assert [(gray.tobytes(), mask.tobytes()) for gray, mask in zip(args[1], args[2], strict=True)] == before
    assert diagnostic.reference_source_indices == (0,)
    assert diagnostic.reference_source_timestamps_s == (0.,)
    assert diagnostic.reference_source_global_gains == (1.,)
    assert diagnostic.reference_semantic_state == 'unknown'  # Start is not a closed-door label.
    assert len(diagnostic.reference_identity_sha256) == 64
    assert all(row.current_raw_valid and row.current_smoothed_available and not row.held_from_smoothed
               for row in diagnostic.samples)
    json.dumps(asdict(diagnostic), allow_nan=False)  # Dataclasses contain no image arrays/bytes.


def test_occluded_high_is_carried_from_smoothing_not_new_measurement():
    args = inputs()
    for mask in args[2][3:]:
        mask[:] = False
    diagnostic = diagnose_door(*args)
    assert_production_exact(args, diagnostic)
    last = diagnostic.samples[-1]
    assert last.finished_score == 1 and last.raw_change is last.smoothed_change is None
    assert not last.current_raw_valid and not last.current_smoothed_available
    assert last.current_measurement_status == 'unknown' and last.rejection_reason == 'current_roi_occluded'
    assert last.held_from_smoothed and last.final_from_smoothed_index == 2
    assert last.last_raw_measurement_index == 2 and last.last_raw_measurement_timestamp_s == 2
    assert last.raw_measurement_age_s == 5
    assert last.final_smoothing_window_indices == (1, 2)
    assert last.final_ancestry_kind == 'mixed_or_median_selected'


def test_lighting_reject_and_occlusion_are_separate_unknown_reasons():
    args = inputs()
    args[5][3] = True
    args[2][4][:] = False
    diagnostic = diagnose_door(*args)
    assert_production_exact(args, diagnostic)
    light, occluded = diagnostic.samples[3:5]
    assert light.lighting_rejected and light.rejection_reason == 'lighting_rejected'
    assert not occluded.lighting_rejected and occluded.rejection_reason == 'current_roi_occluded'
    for sample in (light, occluded):
        assert sample.current_measurement_status == 'unknown'
        assert sample.raw_change is None and sample.held_from_smoothed
        assert sample.finished_score == 1


def test_missing_before_any_measurement_stays_unknown_not_closed():
    args = inputs()
    for i in (0, 1, 2):
        args[5][i] = True
    diagnostic = diagnose_door(*args)
    assert_production_exact(args, diagnostic)
    for sample in diagnostic.samples[:3]:
        assert sample.finished_score is None and sample.raw_change is None
        assert sample.raw_measurement_age_s is None and sample.last_raw_measurement_index is None
        assert sample.final_from_smoothed_index is None and sample.final_ancestry_kind == 'unknown'
        assert not sample.held_from_smoothed
    # Reference selection in existing production ignores lighting rejects; expose that, do not change it.
    assert diagnostic.reference_sources_lighting_rejected == (0,)


def test_smoothing_lookahead_is_not_misattributed_to_last_raw_time():
    args = inputs()
    args[1][1][:] = 200
    diagnostic = diagnose_door(*args)
    first = diagnostic.samples[0]
    assert_production_exact(args, diagnostic)
    assert first.raw_change == 0 and first.smoothed_change == .5 and first.finished_score == 1
    assert first.current_raw_valid and first.raw_measurement_age_s == 0
    assert first.smoothing_window_indices == (0, 1) and first.smoothing_uses_future
    assert first.final_ancestry_kind == 'mixed_or_median_selected'
    assert not first.held_from_smoothed  # Fresh raw exists, but final score is already smoothed.


@pytest.mark.parametrize('kind', ['missing_roi', 'never_visible', 'too_small'])
def test_unavailable_references_and_comparison_cells_never_invent_closed(kind):
    args = list(inputs())
    if kind == 'missing_roi':
        args[3] = None
        reason = 'no_roi'
    elif kind == 'never_visible':
        for mask in args[2]:
            mask[:] = False
        reason = 'no_reference'
    else:
        args[3] = (0, 0, 4, 4)
        reason = 'no_usable_cells'
    diagnostic = diagnose_door(*args)
    assert_production_exact(args, diagnostic)
    assert all(row.finished_score is None and row.raw_change is None and not row.current_raw_valid
               and row.rejection_reason == reason and row.final_ancestry_kind == 'unknown'
               for row in diagnostic.samples)


def test_median_reference_is_composite_not_fabricated_single_timestamp():
    args = list(inputs())
    args[0] = [4. + i for i in range(8)]  # First clear frame beyond TRUST_START_SEC.
    diagnostic = diagnose_door(*args)
    assert_production_exact(args, diagnostic)
    assert diagnostic.reference_mode == 'median'
    assert diagnostic.reference_source_indices == tuple(range(8))
    assert diagnostic.reference_source_timestamps_s == tuple(args[0])
    assert diagnostic.reference_semantic_state == 'unknown'
    assert diagnostic.reference_identity_sha256 == diagnose_door(*args).reference_identity_sha256


def test_reference_mask_intersection_visibility_and_usable_grid_match_production():
    args = inputs()
    args[2][0][:2] = False
    args[2][2][-2:] = False
    diagnostic = diagnose_door(*args)
    assert_production_exact(args, diagnostic)
    row = diagnostic.samples[2]
    assert row.current_visible_fraction == row.reference_visible_fraction == .875
    assert row.common_visible_fraction == .75 and row.usable_cells == row.total_cells == 4


def test_single_raw_smoothing_window_has_explicit_limited_ancestry():
    args = list(inputs())
    args[5] = [False] + [True] * 7
    diagnostic = diagnose_door(*args)
    assert_production_exact(args, diagnostic)
    last = diagnostic.samples[-1]
    assert last.finished_score == 0  # Exact held production score, not a new closed-door observation.
    assert not last.current_raw_valid and last.raw_change is None
    assert last.final_smoothing_window_indices == (0,) and last.final_ancestry_kind == 'single_raw_window'
    assert last.raw_measurement_age_s == 7


@pytest.mark.parametrize('mutation', ['duplicate_time', 'nan_time', 'bool_time', 'short_gains', 'nan_gain',
    'negative_gain', 'string_skip', 'roi_list', 'roi_outside', 'roi_bool', 'rgb_frame', 'float_frame', 'uint8_mask'])
def test_invalid_inputs_fail_instead_of_fabricating_observations(mutation):
    args = list(inputs())
    if mutation == 'duplicate_time':
        args[0][1] = args[0][0]
    elif mutation == 'nan_time':
        args[0][1] = float('nan')
    elif mutation == 'bool_time':
        args[0][0] = False
    elif mutation == 'short_gains':
        args[4] = []
    elif mutation == 'nan_gain':
        args[4][1] = float('nan')
    elif mutation == 'negative_gain':
        args[4][1] = -1
    elif mutation == 'string_skip':
        args[5][1] = 'false'
    elif mutation == 'roi_list':
        args[3] = list(args[3])
    elif mutation == 'roi_outside':
        args[3] = (0, 0, 17, 17)
    elif mutation == 'roi_bool':
        args[3] = (False, 0, 16, 16)
    elif mutation == 'rgb_frame':
        args[1][0] = np.zeros((16, 16, 3), dtype=np.uint8)
    elif mutation == 'float_frame':
        args[1][0] = args[1][0].astype(float)
    else:
        args[2][0] = args[2][0].astype(np.uint8)
    with pytest.raises(ValueError):
        diagnose_door(*args)


def test_module_import_does_not_load_model_runtime():
    code = "import sys; import tools.scene_measurement_diagnostic; assert not {'torch','transformers','mlx','mlx_vlm'} & set(sys.modules)"
    subprocess.run([sys.executable, '-c', code], check=True, timeout=5)
