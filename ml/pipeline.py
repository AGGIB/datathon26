"""К2 — pipeline entry point used by the backend when PIPELINE=real.

Contract (do not change without the team lead):
    analyze(video_path, on_progress=None, time_budget_sec=None) -> dict
returns a dict valid against backend.app.schemas.AnalysisResult:
    {"verdict": "violation" | "no_violation", "violations": [...]}
on_progress(progress: float 0..1, stage: str) may be called any number of times.
time_budget_sec: soft deadline; on overrun return the best result so far instead of failing.

Stages: perception (detect, 0..0.7) -> rules (0.7..0.85) -> evidence + assembly (0.85..1).
The backend runs several worker threads; they share one detector, initialised once and
used by one thread at a time.
"""

import dataclasses
import logging
import math
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from . import rules
from .assemble import assemble
from .candidates import Candidate
from .detect import Detection, Detector
from .events import extract_events
from .evidence import render_candidate
from .perception import SAMPLE_FPS, Perception, perceive
from .video import VideoInfo, probe

log = logging.getLogger(__name__)

ProgressFn = Callable[[float, str], None]

PROGRESS_DETECT_END = 0.7
PROGRESS_RULES_END = 0.85

MAX_SAMPLE_FPS = SAMPLE_FPS
MIN_SAMPLE_FPS = 1.0  # below this, short hand-overs between frames are lost
PLAN_SHARE = 0.7  # share of the remaining budget the detection plan may use; the rest absorbs contention
FIXED_OVERHEAD_SEC = 20.0  # first-run warm-up, rules, evidence
FINISH_RESERVE_SEC = 15.0  # detection stops this long before the deadline so rules and evidence still run
# Seconds per sampled frame for RT-DETRv2-R18 at 640 px, batch 8 (MPS measured ~0.09, CPU ~0.46 on M-series).
SEC_PER_FRAME_PRIOR = {"cuda": 0.05, "mps": 0.15, "cpu": 0.6}
SPEED_EMA_ALPHA = 0.3


class SharedDetector:
    """One model for all worker threads: lazy init and inference under locks, plus measured speed."""

    def __init__(self, factory: Callable[[], Detector] = Detector) -> None:
        self._factory = factory
        self._model: Detector | None = None
        self._init_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._jobs_lock = threading.Lock()
        self._active_jobs = 0
        self._warmed_up = False
        self._sec_per_frame: float | None = None

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

    def model(self) -> Detector:
        with self._init_lock:
            if self._model is None:
                started = time.monotonic()
                self._model = self._factory()
                log.info("pipeline: detector ready on %s in %.1fs", self.device_of(self._model),
                         time.monotonic() - started)
            return self._model

    @staticmethod
    def device_of(model: Detector) -> str:
        return str(getattr(model, "device", "cpu"))

    def detect(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        model = self.model()
        with self._infer_lock:
            started = time.monotonic()
            out = model.detect(frames)
            self._record_speed(time.monotonic() - started, len(frames))
        return out

    def _record_speed(self, elapsed: float, n_frames: int) -> None:
        if n_frames == 0:
            return
        if not self._warmed_up:  # the first batch pays for kernel compilation / cache warm-up
            self._warmed_up = True
            return
        per_frame = elapsed / n_frames
        prev = self._sec_per_frame
        self._sec_per_frame = per_frame if prev is None else prev + SPEED_EMA_ALPHA * (per_frame - prev)

    def sec_per_frame(self) -> float:
        if self._sec_per_frame is not None:
            return self._sec_per_frame
        return SEC_PER_FRAME_PRIOR.get(self.device_of(self.model()), SEC_PER_FRAME_PRIOR["cpu"])


class _DeadlineDetector:
    """Per-job view of the shared model: after the deadline frames are skipped, not inferred."""

    def __init__(self, shared: SharedDetector, deadline: float | None) -> None:
        self._shared = shared
        self._deadline = deadline
        self.processed = 0
        self.skipped = 0

    def detect(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        if self._deadline is not None and time.monotonic() >= self._deadline:
            self.skipped += len(frames)
            return [[] for _ in frames]
        out = self._shared.detect(frames)
        self.processed += len(frames)
        return out


_SHARED = SharedDetector()


def plan_sample_fps(duration_sec: float, budget_sec: float | None, sec_per_frame: float) -> float:
    """Highest sampling rate (MIN..MAX) whose detection cost fits the remaining budget."""
    if budget_sec is None or duration_sec <= 0:
        return MAX_SAMPLE_FPS
    available = budget_sec * PLAN_SHARE - FIXED_OVERHEAD_SEC
    if available <= 0:
        return MIN_SAMPLE_FPS
    fps = available / (duration_sec * max(sec_per_frame, 1e-6))
    return float(min(MAX_SAMPLE_FPS, max(MIN_SAMPLE_FPS, fps)))


def _truncate(perception: Perception, n_frames: int) -> Perception:
    """Perception of the analysed prefix only, so rules do not read skipped frames as an empty room."""
    frames = perception.frames[:n_frames]
    info = perception.info
    t_cut = frames[-1].t if frames else 0.0
    cut_info = dataclasses.replace(info, frame_count=min(info.frame_count, math.floor(t_cut * info.fps) + 1))
    tracks = tuple(tr for tr in perception.tracks if tr.t_end <= t_cut)
    events = tuple(extract_events(tracks, duration=cut_info.duration))
    return dataclasses.replace(perception, info=cut_info, frames=frames, tracks=tracks, events=events)


def _reporter(on_progress: ProgressFn | None) -> ProgressFn:
    def report(progress: float, stage: str) -> None:
        if on_progress is not None:
            on_progress(progress, stage)

    return report


def _perceive_within(
    video_path: str, info: VideoInfo, deadline: float | None, budget_left: float | None, report: ProgressFn
) -> Perception:
    # inference is serialised, so every job running in parallel slows this one down proportionally
    cost = _SHARED.sec_per_frame() * max(1, _SHARED.active_jobs())
    fps = plan_sample_fps(info.duration, budget_left, cost)
    guard = _DeadlineDetector(_SHARED, deadline)
    perception = perceive(
        video_path, guard, sample_fps=fps,
        on_progress=lambda p, _stage: report(PROGRESS_DETECT_END * p, "detect"),
    )
    log.info("pipeline: %s %.1fs video sampled at %.2f fps, %d frames in %.1fs",
             Path(video_path).name, info.duration, fps, len(perception.frames), perception.elapsed_sec)
    if guard.skipped:
        log.warning("pipeline: time budget reached, %d of %d frames skipped — analysing the first %d",
                    guard.skipped, guard.skipped + guard.processed, guard.processed)
        return _truncate(perception, guard.processed)
    return perception


def _find_candidates(perception: Perception) -> Sequence[Candidate]:
    if not perception.frames:
        log.warning("pipeline: no frames were analysed, nothing to judge")
        return []
    return rules.find_candidates(perception, scene=None)


def analyze(
    video_path: str,
    on_progress: ProgressFn | None = None,
    time_budget_sec: float | None = None,
) -> dict[str, Any]:
    with _SHARED.job():
        return _analyze(video_path, _reporter(on_progress), time_budget_sec)


def _analyze(video_path: str, report: ProgressFn, time_budget_sec: float | None) -> dict[str, Any]:
    started = time.monotonic()
    report(0.0, "detect")
    info = probe(video_path)  # unreadable input fails here, before the model is touched
    _SHARED.model()

    deadline = None if time_budget_sec is None else started + time_budget_sec - FINISH_RESERVE_SEC
    budget_left = None if time_budget_sec is None else time_budget_sec - (time.monotonic() - started)
    perception = _perceive_within(video_path, info, deadline, budget_left, report)

    report(PROGRESS_DETECT_END, "rules")
    candidates = _find_candidates(perception)

    report(PROGRESS_RULES_END, "evidence")
    result = assemble(candidates, partial(render_candidate, video_path))
    report(1.0, "evidence")
    log.info("pipeline: %s -> %s, %d candidates, %d violations, %.1fs total", Path(video_path).name,
             result["verdict"], len(candidates), len(result["violations"]), time.monotonic() - started)
    return result
