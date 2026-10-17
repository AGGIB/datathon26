import numpy as np
import pytest

import threading

from ml.scene_owl import OWL_REVISION, OwlDetector, OwlHit, decode_hits, nms_per_query, shared_owl


def test_decode_hits_maps_normalised_boxes_to_source_pixels_with_offset():
    probs = np.array([[0.9, 0.01], [0.02, 0.3]])
    boxes = np.array([[0.5, 0.5, 0.2, 0.4], [0.25, 0.25, 0.1, 0.1]])
    hits = decode_hits(probs, boxes, ["desk", "safe"], side=100.0, offset=(10.0, 20.0), threshold=0.1)
    by_query = {h.query: h for h in hits}
    assert set(by_query) == {"desk", "safe"}
    assert by_query["desk"].box == pytest.approx((50.0, 50.0, 70.0, 90.0))
    assert by_query["safe"].box == pytest.approx((30.0, 40.0, 40.0, 50.0))
    assert by_query["desk"].score == pytest.approx(0.9)


def test_nms_per_query_suppresses_duplicates_only_within_a_query():
    hits = [
        OwlHit("a", (0, 0, 10, 10), 0.9),
        OwlHit("a", (1, 1, 10, 10), 0.8),
        OwlHit("b", (1, 1, 10, 10), 0.7),
    ]
    kept = nms_per_query(hits)
    assert [(h.query, h.score) for h in kept] == [("a", 0.9), ("b", 0.7)]


def test_nms_per_query_caps_hits_per_query():
    hits = [OwlHit("a", (i * 20, 0, i * 20 + 10, 10), 0.5) for i in range(5)]
    assert len(nms_per_query(hits, max_per_query=2)) == 2


def test_shared_owl_is_one_instance_across_threads():
    seen = []
    threads = [threading.Thread(target=lambda: seen.append(shared_owl())) for _ in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert len({id(o) for o in seen}) == 1


def test_owl_detector_is_pinned_and_lazy():
    owl = OwlDetector()
    assert owl.revision == OWL_REVISION and len(OWL_REVISION) == 40
    assert owl.calls == 0 and owl._model is None  # nothing loaded until the first forward
