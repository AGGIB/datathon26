"""Objects lying on the cash desk: map COCO detections and OWLv2 hits to a small item vocabulary (pure logic).

Item labels: cash_bundle, book, stamp, bag, clothes, cup, food, phone.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from .detect import Box, Detection
from .scene_geometry import box_area, center, expand_box, intersection, overlap_fraction
from .scene_owl import OwlHit

# query -> item label. Bundles of banknotes seen from above look like a small stack of paper to OWLv2.
ITEM_QUERIES: dict[str, str] = {
    "a stack of banknotes": "cash_bundle",
    "money": "cash_bundle",
    "papers": "cash_bundle",
    "a book": "book",
    "a notebook": "book",
    "a rubber stamp": "stamp",
    "a handbag": "bag",
    "a backpack": "bag",
    "a jacket": "clothes",
    "a coat": "clothes",
    "a cup": "cup",
    "a mug": "cup",
    "a banana": "food",
    "food": "food",
    "a mobile phone": "phone",
}
# Fixed desk equipment: a hit of these that explains an item box better removes the item.
STATIC_QUERIES = ("a laptop", "a telephone", "a banknote counting machine", "a computer keyboard")
ITEM_SCAN_QUERIES = tuple(ITEM_QUERIES) + STATIC_QUERIES

OWL_MIN_SCORE: dict[str, float] = {
    "cash_bundle": 0.2, "book": 0.25, "stamp": 0.15, "bag": 0.2, "clothes": 0.2, "cup": 0.2, "food": 0.15,
    "phone": 0.2,
}
QUERY_MIN_SCORE: dict[str, float] = {"papers": 0.3}  # generic query, also fires on the counting machine
COCO_TO_ITEM: dict[str, str] = {
    "book": "book", "cup": "cup", "banana": "food", "handbag": "bag", "backpack": "bag", "suitcase": "bag",
    "cell phone": "phone",
}
COCO_MIN_SCORE = 0.4
MAX_PERSON_COVER = 0.5
PERSON_BODY_SHARE = 0.6  # upper part of a person box = the person; below = desk top in front
STATIC_COVER = 0.6  # item box lies mostly inside a stronger fixed-equipment box
DESK_ZONE_UP = 0.35  # objects on the desk top project above the desk box in a top-down oblique view
MERGE_IOU = 0.3
# novelty = change vs. the room at the start inside the item box; fixed equipment stays near 0
MIN_NOVELTY: dict[str, float] = {"owlv2": 0.15, "coco": 0.2}
STRONG_OWL_SCORE = 0.5
HANDHELD_LABELS = frozenset({"cup", "food", "phone"})


@dataclass(frozen=True)
class DeskItem:
    label: str  # cash_bundle | book | stamp | bag | clothes | cup | food | phone
    box: Box
    score: float  # detector confidence (not calibrated across sources)
    source: str  # "owlv2" | "coco"
    query: str  # OWLv2 text query or COCO class that produced the item
    novelty: float | None = None  # 0..1 change vs. the room at the start (None if not measured)


def keep_item(item: DeskItem) -> bool:
    """Drop fixed desk equipment present since the start (COCO calls counting machines 'handbag' or 'book')."""
    if item.source == "owlv2" and item.score >= STRONG_OWL_SCORE:
        return True
    if item.novelty is None:  # region hidden by a person
        # cups and snacks sit right in front of the cashier; bags/clothes there are usually her own or equipment
        return item.label in HANDHELD_LABELS
    return item.novelty >= MIN_NOVELTY.get(item.source, max(MIN_NOVELTY.values()))


def desk_zone(desk: Box, width: float, height: float) -> Box:
    """Area where an object counts as lying on the desk."""
    zone = expand_box(desk, 0.02, 0.0, width=width, height=height)
    up = DESK_ZONE_UP * (desk[3] - desk[1])
    return (zone[0], max(0.0, zone[1] - up), zone[2], zone[3])


def _on_desk(box: Box, zone: Box) -> bool:
    cx, cy = center(box)
    return zone[0] <= cx <= zone[2] and zone[1] <= cy <= zone[3]


def _covered_by_person(box: Box, person_boxes: Sequence[Box]) -> bool:
    """Item looks like part of a person (worn coat, held bag): inside the person box, above desk level.

    A seated cashier's box also covers the desk top, so objects in its lower part are kept.
    """
    _, cy = center(box)
    return any(
        overlap_fraction(box, p) > MAX_PERSON_COVER and cy < p[1] + PERSON_BODY_SHARE * (p[3] - p[1])
        for p in person_boxes
    )


def _explained_by_static(hit_box: Box, score: float, statics: Sequence[tuple[Box, float]]) -> bool:
    return any(overlap_fraction(hit_box, b) > STATIC_COVER and s >= score * 0.7 for b, s in statics)


def items_from_owl(hits: Sequence[OwlHit], zone: Box, person_boxes: Sequence[Box]) -> list[DeskItem]:
    statics = [(h.box, h.score) for h in hits if h.query in STATIC_QUERIES]
    items = []
    for h in hits:
        label = ITEM_QUERIES.get(h.query)
        if label is None or h.score < max(OWL_MIN_SCORE[label], QUERY_MIN_SCORE.get(h.query, 0.0)):
            continue
        if not _on_desk(h.box, zone) or _covered_by_person(h.box, person_boxes):
            continue
        if _explained_by_static(h.box, h.score, statics):
            continue
        items.append(DeskItem(label, h.box, h.score, "owlv2", h.query))
    return merge_items(items)


def items_from_coco(dets: Sequence[Detection], zone: Box, person_boxes: Sequence[Box]) -> list[DeskItem]:
    items = []
    for d in dets:
        label = COCO_TO_ITEM.get(d.label)
        if label is None or d.score < COCO_MIN_SCORE:
            continue
        if _on_desk(d.box, zone) and not _covered_by_person(d.box, person_boxes):
            items.append(DeskItem(label, d.box, d.score, "coco", d.label))
    return merge_items(items)


def _iou(a: Box, b: Box) -> float:
    inter = intersection(a, b)
    if inter is None:
        return 0.0
    i = box_area(inter)
    return i / (box_area(a) + box_area(b) - i)


def merge_items(items: Sequence[DeskItem]) -> list[DeskItem]:
    """One item per physical object and label: keep the strongest of overlapping same-label items."""
    kept: list[DeskItem] = []
    for it in sorted(items, key=lambda i: i.score, reverse=True):
        if all(k.label != it.label or _iou(k.box, it.box) < MERGE_IOU for k in kept):
            kept.append(it)
    return sorted(kept, key=lambda i: (i.label, -i.score))
