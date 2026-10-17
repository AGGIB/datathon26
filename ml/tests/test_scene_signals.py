import numpy as np
import pytest

from ml.scene_signals import (
    change_score,
    hold_forward,
    region_reference,
    key_protrusion,
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


def test_initial_background_uses_first_unmasked_observation():
    frames = list(_frames([250, 10, 12, 14, 200]))
    masks = [np.zeros((4, 4), dtype=bool) for _ in frames]
    masks[0] = np.ones((4, 4), dtype=bool)  # a person covers everything in the first frame
    masks[1][0, 0] = True  # ...and one pixel also in the second
    bg = initial_background(frames, masks)
    assert bg[1, 1, 0] == 10
    assert bg[0, 0, 0] == 12


def test_initial_background_falls_back_for_never_visible_pixels():
    frames = list(_frames([10, 20, 30]))
    masks = [np.ones((4, 4), dtype=bool) for _ in frames]
    assert initial_background(frames, masks)[0, 0, 0] == 20
    fallback = np.full((4, 4, 3), 99, dtype=np.uint8)
    assert initial_background(frames, masks, fallback)[0, 0, 0] == 99


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


def test_structural_change_ignores_brightness_shift_on_flat_face():
    ref = np.full((64, 64), 150, dtype=np.uint8)
    cur = ref.copy()
    cur[:32] = 120  # shadow of a person sitting next to the cabinet
    valid = np.ones_like(ref, dtype=bool)
    assert change_score(cur, ref, valid, gain=1.0) == pytest.approx(0.5)
    assert change_score(cur, ref, valid, gain=1.0, structural_only=True) == 0.0


def test_structural_change_sees_new_edges():
    ref = np.full((64, 64), 150, dtype=np.uint8)
    cur = ref.copy()
    cur[:, ::4] = 60  # door edge / interior structure appears
    valid = np.ones_like(ref, dtype=bool)
    assert change_score(cur, ref, valid, gain=1.0, structural_only=True) == 1.0


def test_change_score_none_for_regions_smaller_than_a_cell():
    tiny = np.zeros((3, 40), dtype=np.uint8)
    assert change_score(tiny, tiny, np.ones_like(tiny, dtype=bool)) is None


def test_change_score_none_when_nothing_visible():
    ref = np.zeros((32, 32), dtype=np.uint8)
    assert change_score(ref, ref, np.zeros_like(ref, dtype=bool)) is None


def _screen(color, size=(24, 32), lines=None):
    img = np.full((*size, 3), color, dtype=np.uint8)
    if lines is not None:
        img[::3] = lines  # rows of text / table grid
    return img


CONTEXT_LIT = np.full((60, 60, 3), 190, dtype=np.uint8)
CONTEXT_DARK = np.full((60, 60, 3), 30, dtype=np.uint8)


def test_screen_features_spreadsheet_is_on_and_work():
    feats = screen_features(_screen((200, 190, 185), lines=(90, 90, 90)), CONTEXT_LIT)
    assert feats.on > 0.8
    assert feats.work > 0.8


def test_screen_features_lock_screen_is_on_but_not_work():
    feats = screen_features(_screen((120, 60, 20), lines=(160, 120, 40)), CONTEXT_LIT)
    assert feats.on > 0.5
    assert feats.work < 0.2


def test_screen_features_closed_flat_lid_is_off():
    feats = screen_features(_screen((120, 120, 120)), CONTEXT_LIT)
    assert feats.on < 0.2
    assert feats.work < 0.2


def test_screen_features_glowing_screen_in_dark_room_is_on():
    feats = screen_features(_screen((210, 210, 210), lines=(120, 120, 120)), CONTEXT_DARK)
    assert feats.on > 0.8
    assert feats.work > 0.8


def test_screen_features_dark_screen_in_dark_room_is_off():
    feats = screen_features(_screen((25, 25, 25)), CONTEXT_DARK)
    assert feats.on < 0.2


def test_hold_fill_forward_then_backward():
    assert hold_fill([None, 0.2, None, 0.8, None]) == [0.2, 0.2, 0.2, 0.8, 0.8]
    assert hold_fill([None, None]) == [None, None]


def test_median_smooth_removes_single_spikes_and_keeps_none():
    assert median_smooth([0.0, 0.0, 1.0, 0.0, 0.0], 3) == [0.0, 0.0, 0.0, 0.0, 0.0]
    assert median_smooth([0.0, None, 1.0, 1.0], 3) == [0.0, None, 1.0, 1.0]


def _cabinet_face(with_key: bool) -> np.ndarray:
    face = np.full((40, 40), 157, dtype=np.uint8)
    face[18:21, 20:23] = 70  # lock cylinder
    if with_key:
        face[22:27, 17:22] = 135  # key head hanging below the cylinder
    return face


def test_key_protrusion_high_with_key_low_without():
    lock = (19.0, 18.0, 23.0, 21.0)
    with_key = key_protrusion(_cabinet_face(True), lock, scale=5)
    without = key_protrusion(_cabinet_face(False), lock, scale=5)
    assert with_key is not None and without is not None
    assert with_key > 15
    assert without < 5


def test_key_protrusion_none_near_image_border():
    assert key_protrusion(np.full((6, 6), 150, dtype=np.uint8), (0.0, 0.0, 3.0, 3.0), scale=5) is None


def _gray(v):
    return np.full((8, 8), v, dtype=np.uint8)


def test_region_reference_trusts_a_clean_start():
    grays = [_gray(10), _gray(200), _gray(200)]
    visible = [np.ones((8, 8), bool)] * 3
    ref, how = region_reference(grays, visible, [0.0, 1.0, 2.0], (0, 0, 8, 8), 0.8, 2.0)
    assert how == "start" and ref[0, 0] == 10


def test_region_reference_uses_the_usual_state_when_hidden_at_start():
    hidden = np.zeros((8, 8), bool)
    shown = np.ones((8, 8), bool)
    grays = [_gray(0), _gray(250), _gray(10), _gray(10), _gray(10)]  # opened at 3 s, closed after
    visible = [hidden, shown, shown, shown, shown]
    ref, how = region_reference(grays, visible, [0.0, 3.0, 4.0, 5.0, 6.0], (0, 0, 8, 8), 0.8, 2.0)
    assert how == "median" and ref[0, 0] == 10


def test_region_reference_none_when_never_clean():
    ref, how = region_reference([_gray(1)], [np.zeros((8, 8), bool)], [0.0], (0, 0, 8, 8), 0.8, 2.0)
    assert ref is None and how == "none"


def test_hold_forward_keeps_leading_gap_unknown():
    assert hold_forward([None, 0.2, None, 0.8]) == [None, 0.2, 0.2, 0.8]
