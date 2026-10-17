"""One detector for all backend worker threads, with measured inference speed per input size.

The backend runs analyses in parallel threads. Loading the model per job would waste seconds and
memory, and concurrent use of one MPS/CUDA model is not safe, so the model is created once and
every inference (including calibration) runs under one lock.
"""

import logging
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import numpy as np

from .detect import INPUT_SIZE, Detection, Detector

log = logging.getLogger(__name__)

SPEED_EMA_ALPHA = 0.3
CALIBRATION_FRAMES = 2  # what callers should pass to calibrate(): cheap even at 1.6 s/frame
CALIBRATION_CALLS = 2  # accelerators need one warm-up call per input shape before a fair measurement
ACCELERATORS = frozenset({"mps", "cuda"})  # compile kernels per input shape on first use


def _device_of(model: Detector) -> str:
    return str(getattr(model, "device", "cpu"))


class SharedDetector:
    def __init__(self, factory: Callable[[], Detector] = Detector) -> None:
        self._factory = factory
        self._model: Detector | None = None
        self._init_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._jobs_lock = threading.Lock()
        self._active_jobs = 0
        self._seen_shapes: set[tuple[int, int]] = set()
        self._speed: dict[int, float] = {}  # input size -> seconds per frame (EMA)

    def model(self) -> Detector:
        with self._init_lock:
            if self._model is None:
                started = time.monotonic()
                self._model = self._factory()
                log.info("detector ready on %s in %.1fs", _device_of(self._model), time.monotonic() - started)
            return self._model

    def device(self) -> str:
        return _device_of(self.model())

    @contextmanager
    def job(self) -> Iterator[None]:
        """Marks a running analysis, so planners know how many jobs compete for the model."""
        with self._jobs_lock:
            self._active_jobs += 1
        try:
            yield
        finally:
            with self._jobs_lock:
                self._active_jobs -= 1

    def active_jobs(self) -> int:
        return self._active_jobs

    def detect(self, frames: list[np.ndarray], input_size: int = INPUT_SIZE) -> list[list[Detection]]:
        model = self.model()
        with self._infer_lock:
            return self._run(model, frames, input_size)

    def detect_before(
        self, frames: list[np.ndarray], input_size: int, deadline: float, sec_per_frame: float
    ) -> list[list[Detection]] | None:
        """detect(), or None if the batch would end after `deadline` (time spent waiting for the model counts)."""
        model = self.model()
        timeout = deadline - time.monotonic()
        if timeout <= 0 or not self._infer_lock.acquire(timeout=timeout):
            return None
        try:
            if time.monotonic() + sec_per_frame * len(frames) > deadline:
                return None
            return self._run(model, frames, input_size)
        finally:
            self._infer_lock.release()

    def _run(self, model: Detector, frames: list[np.ndarray], input_size: int) -> list[list[Detection]]:
        # Detector reads input_size per call; switching it is safe only while the inference lock is held.
        previous = model.input_size
        model.input_size = input_size
        try:
            started = time.monotonic()
            out = model.detect(frames)
            self._record_speed(model, input_size, len(frames), time.monotonic() - started)
        finally:
            model.input_size = previous
        return out

    def _record_speed(self, model: Detector, input_size: int, n_frames: int, elapsed: float) -> None:
        if n_frames == 0:
            return
        shape = (input_size, n_frames)
        first_use = shape not in self._seen_shapes
        self._seen_shapes.add(shape)
        if first_use and _device_of(model) in ACCELERATORS:
            return
        per_frame = elapsed / n_frames
        prev = self._speed.get(input_size)
        self._speed[input_size] = per_frame if prev is None else prev + SPEED_EMA_ALPHA * (per_frame - prev)

    def sec_per_frame(self, input_size: int = INPUT_SIZE) -> float | None:
        """Measured inference time per frame (lock waiting excluded), None until measured."""
        return self._speed.get(input_size)

    def calibrate(self, frames: list[np.ndarray], input_size: int = INPUT_SIZE) -> float:
        """Per-frame cost at input_size on real frames of the video; measured once, then kept up to date."""
        known = self.sec_per_frame(input_size)
        if known is not None:
            return known
        for _ in range(CALIBRATION_CALLS):
            self.detect(frames, input_size)
            known = self.sec_per_frame(input_size)
            if known is not None:
                log.info("detector speed at %d px on %s: %.3f s/frame", input_size, self.device(), known)
                return known
        raise RuntimeError(f"detector speed at {input_size} px could not be measured")
