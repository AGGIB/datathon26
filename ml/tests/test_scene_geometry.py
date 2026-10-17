import pytest

from ml.scene_geometry import (
    box_area,
    clip_box,
    expand_box,
    intersection,
    median_box,
    overlap_fraction,
    sub_box,
    to_int_box,
    union_box,
)


def test_box_area_is_zero_for_degenerate_boxes():
    assert box_area((0, 0, 10, 5)) == 50
    assert box_area((5, 5, 5, 10)) == 0
    assert box_area((10, 10, 0, 0)) == 0


def test_intersection_returns_none_when_disjoint():
    assert intersection((0, 0, 10, 10), (5, 5, 20, 20)) == (5, 5, 10, 10)
    assert intersection((0, 0, 10, 10), (10, 0, 20, 10)) is None


def test_overlap_fraction_is_relative_to_first_box():
    inner = (0, 0, 10, 10)
    outer = (5, 0, 100, 100)
    assert overlap_fraction(inner, outer) == pytest.approx(0.5)
    assert overlap_fraction(outer, inner) == pytest.approx(50 / (95 * 100))
    assert overlap_fraction((0, 0, 0, 0), outer) == 0.0


def test_clip_box_keeps_box_inside_frame():
    assert clip_box((-5, -5, 50, 500), width=40, height=100) == (0, 0, 40, 100)


def test_expand_box_grows_symmetrically_and_clips():
    assert expand_box((10, 10, 20, 30), 0.5, width=100, height=100) == (5, 0, 25, 40)
    assert expand_box((0, 0, 10, 10), 1.0, width=12, height=100) == (0, 0, 12, 20)


def test_expand_box_supports_separate_vertical_ratio():
    assert expand_box((10, 10, 20, 20), 0.0, 1.0, width=100, height=100) == (10, 0, 20, 30)


def test_union_box_covers_all():
    assert union_box([(0, 5, 10, 10), (5, 0, 20, 8)]) == (0, 0, 20, 10)
    with pytest.raises(ValueError):
        union_box([])


def test_median_box_is_robust_to_outlier():
    boxes = [(10, 10, 20, 20), (11, 10, 21, 20), (10, 11, 20, 21), (90, 90, 99, 99)]
    assert median_box(boxes) == (10.5, 10.5, 20.5, 20.5)


def test_sub_box_uses_relative_coordinates():
    assert sub_box((100, 200, 200, 400), 0.0, 0.5, 0.5, 1.0) == (100, 300, 150, 400)


def test_to_int_box_rounds_outwards_and_guarantees_one_pixel():
    assert to_int_box((1.6, 2.2, 3.1, 4.9)) == (1, 2, 4, 5)
    assert to_int_box((5.0, 5.0, 5.0, 5.0)) == (5, 5, 6, 6)
