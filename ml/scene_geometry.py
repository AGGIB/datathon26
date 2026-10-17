"""Pure box geometry for the scene stage. Boxes are (x0, y0, x1, y1) in source-frame pixels."""

import math
from collections.abc import Sequence
from statistics import median

from .detect import Box

IntBox = tuple[int, int, int, int]


def box_area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def intersection(a: Box, b: Box) -> Box | None:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def overlap_fraction(inner: Box, outer: Box) -> float:
    """Share of `inner` covered by `outer` (0 when inner is degenerate)."""
    area = box_area(inner)
    inter = intersection(inner, outer)
    if area <= 0 or inter is None:
        return 0.0
    return box_area(inter) / area


def clip_box(b: Box, width: float, height: float) -> Box:
    return (
        min(max(b[0], 0.0), width),
        min(max(b[1], 0.0), height),
        min(max(b[2], 0.0), width),
        min(max(b[3], 0.0), height),
    )


def expand_box(b: Box, ratio_x: float, ratio_y: float | None = None, *, width: float, height: float) -> Box:
    """Grow by ratio of the box size on each side (0.5 adds half the width left and right), then clip."""
    ry = ratio_x if ratio_y is None else ratio_y
    dx = (b[2] - b[0]) * ratio_x
    dy = (b[3] - b[1]) * ry
    return clip_box((b[0] - dx, b[1] - dy, b[2] + dx, b[3] + dy), width, height)


def union_box(boxes: Sequence[Box]) -> Box:
    if not boxes:
        raise ValueError("union of no boxes")
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def median_box(boxes: Sequence[Box]) -> Box:
    if not boxes:
        raise ValueError("median of no boxes")
    return tuple(float(median(b[i] for b in boxes)) for i in range(4))  # type: ignore[return-value]


def sub_box(b: Box, fx0: float, fy0: float, fx1: float, fy1: float) -> Box:
    """Part of a box given in fractions of its width/height."""
    w, h = b[2] - b[0], b[3] - b[1]
    return (b[0] + fx0 * w, b[1] + fy0 * h, b[0] + fx1 * w, b[1] + fy1 * h)


def to_int_box(b: Box) -> IntBox:
    """Integer pixel box that contains `b` and is at least one pixel wide and tall."""
    x0, y0 = math.floor(b[0]), math.floor(b[1])
    x1, y1 = max(math.ceil(b[2]), x0 + 1), max(math.ceil(b[3]), y0 + 1)
    return (x0, y0, x1, y1)


def center(b: Box) -> tuple[float, float]:
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
