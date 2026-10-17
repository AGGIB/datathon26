import numpy as np
import pytest

from ml.appearance import (
    CROP_H,
    CROP_W,
    color_signature,
    crop_person,
    mean_signature,
    signature_distance,
)


def solid(bgr, h=CROP_H, w=CROP_W):
    return np.full((h, w, 3), bgr, dtype=np.uint8)


def test_crop_is_resized_to_fixed_patch_and_clipped_to_frame():
    frame = solid((10, 20, 30), h=100, w=50)
    crop = crop_person(frame, (-5.0, 90.0, 60.0, 140.0))
    assert crop.shape == (CROP_H, CROP_W, 3)


def test_signature_is_normalised():
    sig = color_signature(solid((0, 0, 200)))
    assert sum(sig) == pytest.approx(1.0)


def test_same_colour_is_close_and_different_colour_is_far():
    red, red2, blue = (color_signature(solid(c)) for c in [(0, 0, 200), (10, 10, 210), (200, 0, 0)])
    assert signature_distance(red, red2) < 0.2
    assert signature_distance(red, blue) > 0.6


def test_dark_and_light_grey_differ_by_value_histogram():
    dark, light = color_signature(solid((30, 30, 30))), color_signature(solid((220, 220, 220)))
    assert signature_distance(dark, light) > 0.6


def test_missing_signature_counts_as_maximally_different():
    assert signature_distance((), (1.0,)) == 1.0


def test_mean_signature_stays_normalised():
    a, b = color_signature(solid((0, 0, 200))), color_signature(solid((200, 0, 0)))
    assert sum(mean_signature([a, b])) == pytest.approx(1.0)
    assert mean_signature([]) == ()
