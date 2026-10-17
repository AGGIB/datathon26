import numpy as np
import pytest

from ml.scene_signals import (
    change_score,
    hold_fill,
    initial_background,
    masked_median,
    median_smooth,
    person_mask,
    ramp,
    screen_features,
)


def test_ramp_is_linear_and_clipped():
    assert ramp(0.0, 1.0, 3.0) == 0.0
    assert ramp(2.0, 1.0, 3.0) == pytest.approx(0.5)
    assert ramp(9.0, 1.0, 3.0) == 1.0


def test_ramp_supports_decreasing_direction():
    assert ramp(1.0, 3.0, 1.0) == 1.0
    assert ramp(3.0, 3.0, 1.0) == 0.0


def test_person_mask_marks_padded_boxes():
    mask = person_mask((10, 10), [(2, 2, 4, 4)], pad_ratio=0.0)
    assert mask.sum() == 4
    assert mask[2:4, 2:4].all()
    assert person_mask((10, 10), [], pad_ratio=0.1).sum() == 0


def _frames(values):
    return np.stack([np.full((4, 4, 3), v, dtype=np.uint8) for v in values])


def test_masked_median_ignores_masked_pixels():
    frames = _frames([10, 200, 200, 10, 10])
    masks = np.zeros((5, 4, 4), dtype=bool)
    masks[3:] = True  # last two frames hidden -> median of 10, 200, 200
    assert masked_median(frames, masks)[0, 0, 0] == 200


def test_masked_median_falls_back_to_plain_median_when_always_masked():
    frames = _frames([10, 20, 30])
    masks = np.ones((3, 4, 4), dtype=bool)
    assert masked_median(frames, masks)[0, 0, 0] == 20


def test_initial_background_uses_earliest_unmasked_observations():
    frames = _frames([250, 10, 12, 14, 200, 200, 200])
    masks = np.zeros((7, 4, 4), dtype=bool)
    masks[0] = True  # a person covers everything in the first frame
    bg = initial_background(frames, masks, first_k=3)
    assert bg[0, 0, 0] == 12


def test_change_score_zero_for_identical_and_high_for_new_structure():
    rng = np.random.default_rng(0)
    ref = rng.integers(0, 255, size=(64, 64), dtype=np.uint8)
    valid = np.ones_like(ref, dtype=bool)
    assert change_score(ref, ref, valid) == 0.0
    cur = ref.copy()
    cur[:, :32] = rng.integers(0, 255, size=(64, 32), dtype=np.uint8)
    assert change_score(cur, ref, valid) == pytest.approx(0.5, abs=0.1)


def test_change_score_tolerates_global_brightness_change():
    rng = np.random.default_rng(1)
    ref = rng.integers(40, 160, size=(64, 64)).astype(np.uint8)
    darker = (ref.astype(float) * 0.6).astype(np.uint8)
    assert change_score(darker, ref, np.ones_like(ref, dtype=bool)) < 0.1


def test_change_score_detects_flat_region_turning_bright():
    ref = np.full((64, 64), 80, dtype=np.uint8)
    cur = ref.copy()
    cur[:32] = 200
    assert change_score(cur, ref, np.ones_like(ref, dtype=bool), gain=1.0) == pytest.approx(0.5)


def test_change_score_none_when_nothing_visible():
    ref = np.zeros((32, 32), dtype=np.uint8)
    assert change_score(ref, ref, np.zeros_like(ref, dtype=bool)) is None


def _screen(color, size=(20, 30)):
    return np.full((*size, 3), color, dtype=np.uint8)


def test_screen_features_white_document_is_on_and_work():
    box = _screen((235, 235, 235))
    context = _screen((170, 170, 170), size=(60, 60))
    feats = screen_features(box, context)
    assert feats.on > 0.8
    assert feats.work > 0.8


def test_screen_features_blue_lock_screen_is_on_but_not_work():
    box = _screen((160, 80, 20))  # BGR: saturated blue wallpaper
    context = _screen((60, 60, 60), size=(60, 60))
    feats = screen_features(box, context)
    assert feats.on > 0.5
    assert feats.work < 0.2


def test_screen_features_closed_dark_lid_is_off():
    box = _screen((50, 50, 50))
    context = _screen((170, 170, 170), size=(60, 60))
    feats = screen_features(box, context)
    assert feats.on < 0.2
    assert feats.work < 0.2


def test_hold_fill_forward_then_backward():
    assert hold_fill([None, 0.2, None, 0.8, None]) == [0.2, 0.2, 0.2, 0.8, 0.8]
    assert hold_fill([None, None]) == [None, None]


def test_median_smooth_removes_single_spikes_and_keeps_none():
    assert median_smooth([0.0, 0.0, 1.0, 0.0, 0.0], 3) == [0.0, 0.0, 0.0, 0.0, 0.0]
    assert median_smooth([0.0, None, 1.0, 1.0], 3) == [0.0, None, 1.0, 1.0]
