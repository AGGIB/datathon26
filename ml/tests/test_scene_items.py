from ml.detect import Detection
from ml.scene_items import DeskItem, desk_zone, items_from_coco, items_from_owl, keep_item, merge_items
from ml.scene_owl import OwlHit

DESK = (60, 350, 330, 540)
ZONE = desk_zone(DESK, 464, 848)


def test_desk_zone_extends_above_the_desk_top():
    assert ZONE[1] < DESK[1]
    assert ZONE[3] == DESK[3]


def test_owl_hits_map_to_item_labels_and_respect_thresholds():
    hits = [
        OwlHit("papers", (208, 397, 234, 408), 0.49),
        OwlHit("a notebook", (179, 382, 231, 411), 0.37),
        OwlHit("a banana", (100, 400, 120, 410), 0.05),  # too weak
        OwlHit("a jacket", (0, 0, 40, 40), 0.6),  # not on the desk
    ]
    items = items_from_owl(hits, ZONE, person_boxes=[])
    assert sorted((i.label, i.query) for i in items) == [("book", "a notebook"), ("cash_bundle", "papers")]
    assert all(i.source == "owlv2" for i in items)


def test_owl_item_explained_by_fixed_equipment_is_dropped():
    hits = [
        OwlHit("papers", (79, 371, 103, 388), 0.3),
        OwlHit("a banknote counting machine", (75, 345, 145, 430), 0.35),
    ]
    assert items_from_owl(hits, ZONE, person_boxes=[]) == []


def test_items_mostly_covered_by_a_person_are_dropped():
    hits = [OwlHit("a jacket", (150, 380, 200, 450), 0.5)]
    assert items_from_owl(hits, ZONE, person_boxes=[(140, 300, 260, 520)]) == []


def test_coco_detections_map_to_item_labels():
    dets = [
        Detection((200, 400, 220, 420), 0.6, "cup"),
        Detection((100, 400, 140, 420), 0.5, "banana"),
        Detection((210, 380, 280, 450), 0.7, "handbag"),
        Detection((240, 320, 310, 390), 0.9, "laptop"),
        Detection((100, 300, 200, 500), 0.9, "person"),
    ]
    items = items_from_coco(dets, ZONE, person_boxes=[])
    assert sorted(i.label for i in items) == ["bag", "cup", "food"]


def test_merge_items_keeps_best_per_overlapping_label():
    items = [
        DeskItem("book", (0, 0, 10, 10), 0.4, "owlv2", "a book"),
        DeskItem("book", (1, 1, 10, 10), 0.6, "coco", "book"),
        DeskItem("cup", (1, 1, 10, 10), 0.3, "coco", "cup"),
    ]
    merged = merge_items(items)
    assert [(i.label, i.score) for i in merged] == [("book", 0.6), ("cup", 0.3)]


def _item(source, score, novelty, label="book"):
    return DeskItem(label, (0, 0, 10, 10), score, source, label, novelty)


def test_keep_item_drops_fixed_equipment_seen_since_the_start():
    assert not keep_item(_item("coco", 0.6, 0.05, "bag"))  # counting machine called "handbag"
    assert keep_item(_item("coco", 0.45, 0.4, "cup"))
    assert keep_item(_item("coco", 0.45, None, "cup"))


def test_keep_item_trusts_strong_or_new_owl_hits():
    assert keep_item(_item("owlv2", 0.39, 0.15))  # notebook partly in the start reference
    assert not keep_item(_item("owlv2", 0.3, 0.02))
    assert keep_item(_item("owlv2", 0.6, 0.0))
