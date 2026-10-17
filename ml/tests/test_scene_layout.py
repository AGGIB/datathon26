from ml.scene_layout import pick_desk, pick_door, pick_laptop, pick_lock, pick_safe, pick_screen
from ml.scene_owl import OwlHit

W, H = 464, 848


def hit(query, box, score):
    return OwlHit(query=query, box=box, score=score)


def test_pick_laptop_prefers_median_of_coco_boxes():
    coco = [(220, 300, 300, 370), (222, 302, 298, 372), (221, 301, 299, 371)]
    box, conf = pick_laptop(coco, [hit("a laptop", (0, 0, 10, 10), 0.9)])
    assert box == (221, 301, 299, 371)
    assert conf > 0


def test_pick_laptop_falls_back_to_owl_then_none():
    box, _ = pick_laptop([], [hit("a laptop", (5, 5, 50, 50), 0.6)])
    assert box == (5, 5, 50, 50)
    assert pick_laptop([], [hit("a laptop", (5, 5, 50, 50), 0.05)])[0] is None


def test_pick_desk_prefers_the_desk_under_the_laptop():
    laptop = (226, 306, 297, 371)
    hits = [
        hit("a desk", (0, 519, 345, 840), 0.55),  # foreground meeting table, higher score
        hit("a desk", (66, 350, 328, 541), 0.51),  # cash desk with the laptop on it
    ]
    box, _ = pick_desk(hits, laptop, W, H)
    assert box == (66, 350, 328, 541)


def test_pick_desk_rejects_boxes_covering_most_of_the_frame():
    box, _ = pick_desk([hit("a desk", (0, 0, W, H), 0.9)], None, W, H)
    assert box is None


def test_pick_safe_takes_upright_cabinet_next_to_desk():
    desk = (66, 350, 328, 541)
    hits = [
        hit("a grey metal cabinet", (187, 250, 260, 355), 0.32),
        hit("a metal safe", (0, 430, 114, 549), 0.33),  # white drawer unit, wide and far below the desk top
        hit("a metal cabinet", (70, 360, 320, 530), 0.4),  # the desk itself
    ]
    box, _ = pick_safe(hits, desk, None, W, H)
    assert box == (187, 250, 260, 355)


def test_pick_door_requires_a_handle_inside_the_glass_door():
    hits = [
        hit("a glass door", (101, 1, 225, 276), 0.40),
        hit("a glass door", (0, 3, 113, 416), 0.37),
        hit("a door handle", (96, 262, 117, 274), 0.47),
    ]
    box, conf = pick_door(hits, W, H)
    assert box == (0, 3, 113, 416)
    assert conf > 0.3


def test_pick_door_prefers_the_panel_that_contains_the_handle():
    hits = [
        hit("a glass door", (133, 21, 252, 355), 0.445),  # fixed panel next to the door, handle on its edge
        hit("a glass door", (2, 21, 144, 427), 0.404),
        hit("a door handle", (121, 286, 142, 298), 0.46),
    ]
    box, _ = pick_door(hits, W, H)
    assert box == (2, 21, 144, 427)


def test_pick_door_falls_back_to_best_glass_door_without_handle():
    hits = [hit("a glass door", (0, 3, 113, 416), 0.37)]
    box, conf = pick_door(hits, W, H)
    assert box == (0, 3, 113, 416)
    assert conf < 0.37


def test_pick_lock_inside_safe_only():
    safe = (187, 250, 260, 355)
    hits = [hit("a lock", (23, 105, 31, 113), 0.3), hit("a keyhole", (191, 300, 199, 306), 0.14)]
    box, _ = pick_lock(hits, safe)
    assert box == (191, 300, 199, 306)
    assert pick_lock(hits, None)[0] is None


def test_pick_screen_takes_screen_hit_on_the_laptop():
    laptop = (245, 326, 317, 393)
    hits = [
        hit("a computer screen", (276, 327, 316, 374), 0.56),
        hit("a computer screen", (10, 10, 60, 60), 0.9),  # monitor elsewhere in the room
        hit("a keyboard", (259, 363, 294, 384), 0.64),
    ]
    box, conf = pick_screen(hits, laptop)
    assert box == (276, 327, 316, 374)
    assert conf == 0.56


def test_pick_screen_falls_back_to_upper_part_of_laptop():
    box, conf = pick_screen([], (0, 0, 100, 100))
    assert box == (0, 0, 100, 60)
    assert conf == 0.0
    assert pick_screen([], None) == (None, 0.0)
