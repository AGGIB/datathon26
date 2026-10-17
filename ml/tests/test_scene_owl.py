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


def _fake_backend(owl):
    """Exercise real locking/counters without loading weights or using a GPU."""
    from contextlib import nullcontext
    from types import SimpleNamespace

    class Tensor:
        def __init__(self, value):
            self.value = value

        def sigmoid(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.value

    class Inputs(dict):
        def to(self, device):
            return self

    def processor(*, text, images, return_tensors):
        return Inputs(batch=len(images))

    def model(*, batch):
        return SimpleNamespace(logits=Tensor(np.full((batch, 1, 1), 0.9)),
                               pred_boxes=Tensor(np.tile([[[0.5, 0.5, 0.2, 0.2]]], (batch, 1, 1))))

    owl._processor = processor
    owl._model = model
    owl._torch = SimpleNamespace(no_grad=nullcontext)


def test_concurrent_analyses_attribute_only_their_own_forwards():
    from concurrent.futures import ThreadPoolExecutor

    owl = OwlDetector(device="cpu")
    _fake_backend(owl)
    barrier = threading.Barrier(2)
    frame = np.zeros((16, 16, 3), dtype=np.uint8)

    def analyze():
        with owl.measure_calls() as stats:
            barrier.wait(timeout=3)
            owl.detect(frame, ["desk"])
            barrier.wait(timeout=3)
            return stats.calls, stats.images

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = pool.submit(analyze), pool.submit(analyze)
        assert first.result(timeout=5) == (1, 1)
        assert second.result(timeout=5) == (1, 1)
    assert (owl.calls, owl.images) == (2, 2)


def test_load_is_once_pinned_and_safetensors_under_concurrency(monkeypatch):
    import sys
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace

    entered, release = threading.Event(), threading.Event()
    processor_calls, model_calls = [], []
    model = SimpleNamespace()
    model.to = lambda device: model
    model.eval = lambda: model

    def processor_loader(model_id, **kwargs):
        processor_calls.append((model_id, kwargs))
        entered.set()
        assert release.wait(timeout=3)
        return object()

    def model_loader(model_id, **kwargs):
        model_calls.append((model_id, kwargs))
        return model

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        Owlv2Processor=SimpleNamespace(from_pretrained=processor_loader),
        Owlv2ForObjectDetection=SimpleNamespace(from_pretrained=model_loader),
    ))
    owl = OwlDetector(device="cpu")
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(owl.load)
        assert entered.wait(timeout=3)
        second = pool.submit(owl.load)
        release.set()
        first.result(timeout=3)
        second.result(timeout=3)
    assert processor_calls == [(owl.model_id, {"revision": OWL_REVISION, "local_files_only": True})]
    assert model_calls == [(owl.model_id, {"revision": OWL_REVISION, "use_safetensors": True, "local_files_only": True})]


def test_expired_deadline_does_not_load_model():
    import time

    owl = OwlDetector()
    with pytest.raises(TimeoutError, match="deadline"):
        owl.detect(np.zeros((16, 16, 3), dtype=np.uint8), ["desk"], deadline=time.monotonic() - 1)
    assert owl._model is None and owl.calls == 0


def test_owl_lock_wait_is_bounded_by_deadline():
    import time
    from concurrent.futures import ThreadPoolExecutor

    owl = OwlDetector()
    with owl._lock, ThreadPoolExecutor(max_workers=1) as pool:
        waiting = pool.submit(owl.load, deadline=time.monotonic() + 0.03)
        with pytest.raises(TimeoutError, match="waiting for OWLv2"):
            waiting.result(timeout=2)
    assert owl._model is None


def test_cache_is_revision_specific_and_writes_atomically(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    owl = OwlDetector(cache_dir=tmp_path)
    other = OwlDetector(cache_dir=tmp_path, revision="different")
    args = ("frame", ["desk"], (0, 0, 20, 20), 0.1)
    assert owl._cache_file(*args) != other._cache_file(*args)
    first = [OwlHit("desk", (0, 0, 10, 10), 0.9)]
    second = [OwlHit("desk", (1, 1, 11, 11), 0.8)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        writes = [pool.submit(owl._cache_put, *args, hits) for hits in (first, second)]
        for write in writes:
            write.result(timeout=3)
    assert owl._cache_get(*args) in (first, second)
    assert len(list(tmp_path.iterdir())) == 1  # no abandoned temporary files


def test_detect_batch_rejects_mismatched_cache_keys():
    with pytest.raises(ValueError, match="cache_keys"):
        OwlDetector().detect_batch([(np.zeros((8, 8, 3), np.uint8), None)], ["desk"], cache_keys=[])


def test_forward_deadline_overrun_does_not_start_another_batch(monkeypatch):
    import ml.scene_owl as owl_module

    owl = OwlDetector(device="cpu")
    _fake_backend(owl)
    clock = [0.0]
    monkeypatch.setattr(owl_module.time, "monotonic", lambda: clock[0])
    backend = owl._model

    def slow_native_forward(**inputs):
        clock[0] = 11.0
        return backend(**inputs)

    owl._model = slow_native_forward
    views = [(np.zeros((16, 16, 3), dtype=np.uint8), None)] * 3
    with pytest.raises(TimeoutError, match="deadline"):
        owl.detect_batch(views, ["desk"], deadline=10.0)
    assert (owl.calls, owl.images) == (1, 2)
