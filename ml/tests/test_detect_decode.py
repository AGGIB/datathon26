import numpy as np

from ml.detect import decode_queries

ID2LABEL = {0: "person", 1: "chair", 2: "laptop"}
SHAPE = (100, 200, 3)  # h, w
BOX = np.array([[0.5, 0.5, 0.2, 0.4]])  # cx, cy, w, h normalized


def test_query_whose_best_class_is_excluded_is_dropped():
    # chair wins (0.9) even though person is above the person threshold (0.51)
    probs = np.array([[0.51, 0.9, 0.0]])
    assert decode_queries(probs, BOX, SHAPE, ID2LABEL) == []


def test_person_query_is_decoded_to_pixel_box():
    probs = np.array([[0.8, 0.1, 0.0]])
    [det] = decode_queries(probs, BOX, SHAPE, ID2LABEL)
    assert det.label == "person"
    assert det.box == (80.0, 30.0, 120.0, 70.0)
    assert abs(det.score - 0.8) < 1e-6


def test_object_below_its_threshold_is_dropped():
    probs = np.array([[0.0, 0.0, 0.3]])
    assert decode_queries(probs, BOX, SHAPE, ID2LABEL) == []


def test_boxes_are_clipped_to_the_frame():
    probs = np.array([[0.9, 0.0, 0.0]])
    box = np.array([[0.05, 0.5, 0.3, 0.4]])  # x0 would be -20 px
    [det] = decode_queries(probs, box, SHAPE, ID2LABEL)
    assert det.box[0] == 0.0 and det.box[2] == 40.0


def test_class_index_missing_from_label_map_is_skipped():
    probs = np.array([[0.0, 0.0, 0.0, 0.95]])  # index 3 not in ID2LABEL
    assert decode_queries(probs, BOX, SHAPE, ID2LABEL) == []


def test_non_person_object_above_object_threshold_is_kept():
    probs = np.array([[0.0, 0.0, 0.45]])
    [det] = decode_queries(probs, BOX, SHAPE, ID2LABEL)
    assert det.label == "laptop"


def test_person_between_object_and_person_threshold_is_dropped():
    probs = np.array([[0.45, 0.0, 0.0]])
    assert decode_queries(probs, BOX, SHAPE, ID2LABEL) == []
