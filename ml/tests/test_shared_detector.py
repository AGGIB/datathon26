import threading
import time

import numpy as np
import pytest

from ml.detect import INPUT_SIZE
from ml.shared_detector import CALIBRATION_FRAMES, SharedDetector

FRAME = np.zeros((48, 64, 3), np.uint8)
SAMPLE = [FRAME] * CALIBRATION_FRAMES


class TimedDetector:
    """Fake model whose cost per frame depends on its input size, like the real one."""

    def __init__(self, device="cpu", sec_per_frame=None, first_call_extra_sec=0.0):
        self.device = device
        self.input_size = INPUT_SIZE
        self.sec_per_frame = sec_per_frame or {INPUT_SIZE: 0.02, 480: 0.01}
        self.first_call_extra_sec = first_call_extra_sec
        self.sizes_used = []

    def detect(self, frames):
        extra = self.first_call_extra_sec if not self.sizes_used else 0.0
        self.sizes_used.append(self.input_size)
        time.sleep(extra + self.sec_per_frame[self.input_size] * len(frames))
        return [[] for _ in frames]


def test_model_is_created_once_and_lazily():
    created = []
    shared = SharedDetector(factory=lambda: created.append(1) or TimedDetector())
    assert created == []
    assert shared.model() is shared.model()
    assert created == [1]


def test_detect_uses_requested_input_size_and_restores_it():
    model = TimedDetector()
    shared = SharedDetector(factory=lambda: model)
    shared.detect([FRAME], input_size=480)
    shared.detect([FRAME])
    assert model.sizes_used == [480, INPUT_SIZE]
    assert model.input_size == INPUT_SIZE


def test_speed_is_unknown_until_measured():
    shared = SharedDetector(factory=TimedDetector)
    assert shared.sec_per_frame() is None
    assert shared.sec_per_frame(480) is None


def test_calibration_measures_cost_per_input_size():
    shared = SharedDetector(factory=TimedDetector)
    full = shared.calibrate(SAMPLE, INPUT_SIZE)
    small = shared.calibrate(SAMPLE, 480)
    assert full == pytest.approx(0.02, abs=0.01)
    assert small < full


def test_calibration_reuses_known_speed():
    model = TimedDetector()
    shared = SharedDetector(factory=lambda: model)
    shared.calibrate(SAMPLE, INPUT_SIZE)
    calls = len(model.sizes_used)
    shared.calibrate(SAMPLE, INPUT_SIZE)
    assert len(model.sizes_used) == calls


def test_accelerator_ignores_first_call_of_each_shape():
    model = TimedDetector(device="mps", first_call_extra_sec=0.3)  # kernel compilation on first use
    shared = SharedDetector(factory=lambda: model)
    cost = shared.calibrate(SAMPLE, INPUT_SIZE)
    assert cost < 0.05
    assert len(model.sizes_used) == 2  # warm-up call + measured call


def test_cpu_measures_from_the_first_call():
    model = TimedDetector(device="cpu")
    shared = SharedDetector(factory=lambda: model)
    shared.calibrate(SAMPLE, INPUT_SIZE)
    assert len(model.sizes_used) == 1
    assert CALIBRATION_FRAMES >= 1


def test_detect_before_runs_when_batch_fits():
    model = TimedDetector()
    shared = SharedDetector(factory=lambda: model)
    out = shared.detect_before([FRAME], INPUT_SIZE, deadline=time.monotonic() + 5.0, sec_per_frame=0.02)
    assert out == [[]]
    assert model.sizes_used == [INPUT_SIZE]


def test_detect_before_refuses_batch_that_would_overrun():
    model = TimedDetector()
    shared = SharedDetector(factory=lambda: model)
    out = shared.detect_before([FRAME] * 8, INPUT_SIZE, deadline=time.monotonic() + 1.0, sec_per_frame=0.5)
    assert out is None
    assert model.sizes_used == []


def test_detect_before_gives_up_waiting_for_a_busy_model():
    model = TimedDetector(sec_per_frame={INPUT_SIZE: 1.0})
    shared = SharedDetector(factory=lambda: model)
    shared.model()
    busy = threading.Thread(target=shared.detect, args=([FRAME],))
    busy.start()
    time.sleep(0.1)  # the other job now holds the model for ~1 s
    started = time.monotonic()
    out = shared.detect_before([FRAME], INPUT_SIZE, deadline=started + 0.3, sec_per_frame=0.01)
    waited = time.monotonic() - started
    busy.join(timeout=5)
    assert not busy.is_alive()
    assert out is None
    assert waited < 0.8


def test_job_counter_tracks_running_jobs():
    shared = SharedDetector(factory=TimedDetector)
    assert shared.active_jobs() == 0
    with shared.job():
        with shared.job():
            assert shared.active_jobs() == 2
        assert shared.active_jobs() == 1
    assert shared.active_jobs() == 0


def test_job_counter_recovers_after_errors():
    shared = SharedDetector(factory=TimedDetector)
    with pytest.raises(RuntimeError):
        with shared.job():
            raise RuntimeError("boom")
    assert shared.active_jobs() == 0


def test_inference_is_serialised_across_threads():
    active, peak, lock = [0], [0], threading.Lock()

    class Probe(TimedDetector):
        def detect(self, frames):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(0.01)
            with lock:
                active[0] -= 1
            return [[] for _ in frames]

    shared = SharedDetector(factory=Probe)
    threads = [threading.Thread(target=shared.detect, args=([FRAME],)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert peak[0] == 1
