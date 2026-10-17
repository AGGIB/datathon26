"""Pick desk / safe / door / laptop / lock boxes from zero-shot hits and COCO detections (pure logic).

No pixel coordinates are hard-coded: everything is chosen per video from detections on the empty-room
background, using only relations between objects (laptop stands on the desk, safe is an upright cabinet
next to the desk, the entrance door is a glass door with a handle).
"""

from collections.abc import Sequence

from .detect import Box
from .scene_geometry import box_area, center, expand_box, median_box, overlap_fraction, sub_box
from .scene_owl import OwlHit

DESK_QUERIES = ("a desk", "an office desk")
SAFE_QUERIES = ("a grey metal cabinet", "a metal safe", "a metal cabinet")
DOOR_QUERIES = ("a glass door",)
HANDLE_QUERIES = ("a door handle",)
LAPTOP_QUERIES = ("a laptop",)
LOCK_QUERIES = ("a lock", "a keyhole")
SCREEN_QUERIES = ("a computer screen", "a keyboard")
MIN_SCREEN_SCORE = 0.2
SCREEN_FALLBACK_SHARE = 0.6  # without a screen hit: upper part of the laptop box
LAYOUT_QUERIES = DESK_QUERIES + SAFE_QUERIES + DOOR_QUERIES + HANDLE_QUERIES + LAPTOP_QUERIES

MIN_COCO_LAPTOPS = 3
MIN_LAPTOP_SCORE = 0.2
MIN_DESK_SCORE = 0.15
MAX_DESK_FRAME_SHARE = 0.5
MIN_SAFE_SCORE = 0.12
MAX_SAFE_FRAME_SHARE = 0.12
MIN_SAFE_ASPECT = 0.8  # height / width: cabinets and safes stand upright
MAX_SAFE_IN_DESK = 0.6
MIN_DOOR_SCORE = 0.15
MIN_HANDLE_SCORE = 0.2
NO_HANDLE_PENALTY = 0.6
HANDLE_REL_Y = (0.3, 0.85)
MIN_HANDLE_INSIDE = 0.3
MAX_LOCK_SHARE_OF_SAFE = 0.05


def _of(hits: Sequence[OwlHit], queries: Sequence[str], min_score: float) -> list[OwlHit]:
    return [h for h in hits if h.query in queries and h.score >= min_score]


def pick_laptop(coco_laptops: Sequence[Box], hits: Sequence[OwlHit]) -> tuple[Box | None, float]:
    """Median COCO laptop box over the video (stable), else the best OWLv2 laptop."""
    if len(coco_laptops) >= MIN_COCO_LAPTOPS:
        return median_box(coco_laptops), min(1.0, len(coco_laptops) / 10)
    owl = sorted(_of(hits, LAPTOP_QUERIES, MIN_LAPTOP_SCORE), key=lambda h: h.score, reverse=True)
    return (owl[0].box, owl[0].score) if owl else (None, 0.0)


def _supports(desk: Box, laptop: Box, width: float, height: float) -> bool:
    lx, _ = center(laptop)
    zone = expand_box(desk, 0.05, 0.15, width=width, height=height)
    return zone[0] <= lx <= zone[2] and zone[1] <= laptop[3] <= zone[3]


def pick_desk(hits: Sequence[OwlHit], laptop: Box | None, width: float, height: float) -> tuple[Box | None, float]:
    """The desk that carries the laptop; otherwise the best-scoring desk that is not the whole floor."""
    frame = width * height
    cands = [h for h in _of(hits, DESK_QUERIES, MIN_DESK_SCORE) if box_area(h.box) <= MAX_DESK_FRAME_SHARE * frame]
    if laptop is not None:
        under = [h for h in cands if _supports(h.box, laptop, width, height)]
        cands = under or cands
    if not cands:
        return None, 0.0
    best = max(cands, key=lambda h: h.score)
    return best.box, best.score


def _gap(a: Box, b: Box) -> float:
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return (dx * dx + dy * dy) ** 0.5


def pick_safe(
    hits: Sequence[OwlHit], desk: Box | None, laptop: Box | None, width: float, height: float
) -> tuple[Box | None, float]:
    """Upright grey cabinet, not the desk itself, preferably touching the desk."""
    frame = width * height
    cands = []
    for h in _of(hits, SAFE_QUERIES, MIN_SAFE_SCORE):
        bw, bh = h.box[2] - h.box[0], h.box[3] - h.box[1]
        if bw <= 0 or bh / bw < MIN_SAFE_ASPECT or box_area(h.box) > MAX_SAFE_FRAME_SHARE * frame:
            continue
        if desk is not None and overlap_fraction(h.box, desk) > MAX_SAFE_IN_DESK:
            continue
        if laptop is not None and overlap_fraction(laptop, h.box) > 0.5:
            continue
        near = 1.0
        if desk is not None:
            near = 1.0 if _gap(h.box, desk) <= 0.15 * (desk[2] - desk[0]) else 0.5
            near *= 1.0 if h.box[1] < desk[1] + 0.25 * (desk[3] - desk[1]) else 0.5  # stands at desk-top level
        cands.append((h.score * near, h))
    if not cands:
        return None, 0.0
    score, best = max(cands, key=lambda c: c[0])
    return best.box, score


def pick_door(hits: Sequence[OwlHit], width: float, height: float) -> tuple[Box | None, float]:
    """Glass door that contains a door handle; else the best glass door with reduced confidence."""
    doors = sorted(_of(hits, DOOR_QUERIES, MIN_DOOR_SCORE), key=lambda h: h.score, reverse=True)
    handles = _of(hits, HANDLE_QUERIES, MIN_HANDLE_SCORE)
    fits = []
    for d in doors:
        for hd in handles:
            inside = overlap_fraction(hd.box, d.box)  # the handle belongs to the panel that contains it
            if inside >= MIN_HANDLE_INSIDE:
                fits.append((d.score * inside * _handle_height_fit(d.box, center(hd.box)[1]), d, hd))
    if fits:
        _, d, hd = max(fits, key=lambda f: f[0])
        return d.box, max(d.score, hd.score)
    if doors:
        return doors[0].box, doors[0].score * NO_HANDLE_PENALTY
    return None, 0.0


def _handle_height_fit(door: Box, handle_y: float) -> float:
    """Handles sit around the middle of a door's height, not at its top or bottom edge."""
    rel = (handle_y - door[1]) / max(door[3] - door[1], 1.0)
    return 1.0 if HANDLE_REL_Y[0] <= rel <= HANDLE_REL_Y[1] else 0.3


def pick_lock(hits: Sequence[OwlHit], safe: Box | None) -> tuple[Box | None, float]:
    """Small lock/keyhole hit inside the safe box."""
    if safe is None:
        return None, 0.0
    limit = MAX_LOCK_SHARE_OF_SAFE * box_area(safe)
    cands = [
        h for h in hits
        if h.query in LOCK_QUERIES and overlap_fraction(h.box, safe) > 0.8 and box_area(h.box) <= limit
    ]
    if not cands:
        return None, 0.0
    best = max(cands, key=lambda h: h.score)
    return best.box, best.score


def pick_screen(hits: Sequence[OwlHit], laptop: Box | None) -> tuple[Box | None, float]:
    """Screen of the laptop: best 'computer screen' hit centred on the laptop, else its upper part."""
    if laptop is None:
        return None, 0.0
    lx0, ly0, lx1, ly1 = laptop
    pad_x, pad_y = 0.1 * (lx1 - lx0), 0.1 * (ly1 - ly0)
    cands = []
    for h in hits:
        cx, cy = center(h.box)
        if h.query == SCREEN_QUERIES[0] and h.score >= MIN_SCREEN_SCORE \
                and lx0 - pad_x <= cx <= lx1 + pad_x and ly0 - pad_y <= cy <= ly1 + pad_y:
            cands.append(h)
    if cands:
        best = max(cands, key=lambda h: h.score)
        return best.box, best.score
    return sub_box(laptop, 0.0, 0.0, 1.0, SCREEN_FALLBACK_SHARE), 0.0
