"""К2 — pipeline entry point used by the backend when PIPELINE=real.

Contract (do not change without the team lead):
    analyze(video_path, on_progress=None, time_budget_sec=None) -> dict
returns a dict valid against backend.app.schemas.AnalysisResult:
    {"verdict": "violation" | "no_violation", "violations": [...]}
on_progress(progress: float 0..1, stage: str) may be called any number of times.
time_budget_sec: soft deadline; on overrun return the best result so far instead of failing.

Stages: perception (detect, 0..0.7) -> rules (0.7..0.85) -> evidence + assembly (0.85..1).
Speed differs ~30x between MPS and a CPU container, so the detector's cost per frame is measured
(calibrate), and the sampling rate / input size are chosen so that detection ends before the deadline.
"""

import dataclasses
import logging
import math
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import rules
from .assemble import assemble
from .candidates import Candidate
from .detect import INPUT_SIZE, Detection
from .events import extract_events
from .evidence import render_candidates
from .kb import ViolationType, catalog
from .perception import SAMPLE_FPS, Perception, perceive
from .shared_detector import CALIBRATION_FRAMES, SharedDetector
from .video import VideoInfo, frames_at, probe

log = logging.getLogger(__name__)

ProgressFn = Callable[[float, str], None]
CostFn = Callable[[int], float]

PROGRESS_DETECT_END = 0.7
PROGRESS_RULES_END = 0.85

TRACKER_GAP_SEC = 1.0  # IoUTracker(max_gap_sec=1.0) inside perceive(): a longer gap ends a track
MAX_SAMPLE_FPS = SAMPLE_FPS
SAFE_MIN_FPS = 2.2  # one missed detection still fits into the tracker gap (2 / 2.2 = 0.91 s)
HARD_MIN_FPS = 1.25  # consecutive samples still link (0.8 s); any miss splits the track
SMALL_INPUT_SIZE = 480  # ~0.55x the CPU cost of 640; on MPS it saves nothing
# (input size, lowest acceptable fps) from best to cheapest
PLAN_LADDER = ((INPUT_SIZE, SAFE_MIN_FPS), (SMALL_INPUT_SIZE, SAFE_MIN_FPS), (SMALL_INPUT_SIZE, HARD_MIN_FPS))
PLAN_SHARE = 0.7  # share of the detection window the plan may fill; calibration on 2 frames runs ~30% optimistic
FINISH_RESERVE_SEC = 15.0  # detection ends this long before the deadline so rules and evidence still run
TAIL_TOLERANCE_SEC = 1.0  # missing video end beyond one sampling step + this is treated as undecodable
CALIBRATION_AT_SEC = 1.0  # calibration frames come from the first seconds: decoding them is cheap


@dataclasses.dataclass(frozen=True)
class SamplingPlan:
    sample_fps: float
    input_size: int
    sec_per_frame: float  # expected inference cost per frame (0.0 when no budget, nothing measured)


def plan_sampling(duration_sec: float, window_sec: float | None, cost_at: CostFn, contention: int = 1) -> SamplingPlan:
    """Best (input size, fps) whose detection fits `window_sec`; cost_at(size) is called only when needed."""
    if window_sec is None or duration_sec <= 0:
        return SamplingPlan(MAX_SAMPLE_FPS, INPUT_SIZE, 0.0)
    if window_sec <= 0:
        return SamplingPlan(HARD_MIN_FPS, SMALL_INPUT_SIZE, 0.0)
    for size, min_fps in PLAN_LADDER:
        cost = cost_at(size)
        fps = window_sec * PLAN_SHARE / (duration_sec * max(cost * contention, 1e-6))
        if fps >= min_fps:
            return SamplingPlan(min(MAX_SAMPLE_FPS, fps), size, cost)
    # Even the sparsest plan does not fit: run it anyway, the deadline guard cuts the tail.
    return SamplingPlan(HARD_MIN_FPS, SMALL_INPUT_SIZE, cost_at(SMALL_INPUT_SIZE))


class _BudgetedDetector:
    """Per-job view of the shared model: once a batch would end after the deadline, the rest is skipped."""

    def __init__(self, shared: SharedDetector, plan: SamplingPlan, deadline: float | None) -> None:
        self._shared = shared
        self._plan = plan
        self._deadline = deadline
        self._stopped = False
        self.processed = 0
        self.skipped = 0

    def detect(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        if not self._stopped:
            out = self._infer(frames)
            if out is not None:
                self.processed += len(frames)
                return out
            self._stopped = True  # never resume: the analysed part must stay a prefix of the video
        self.skipped += len(frames)
        return [[] for _ in frames]

    def _infer(self, frames: list[np.ndarray]) -> list[list[Detection]] | None:
        if self._deadline is None:
            return self._shared.detect(frames, self._plan.input_size)
        return self._shared.detect_before(frames, self._plan.input_size, self._deadline, self._plan.sec_per_frame)


_SHARED = SharedDetector()


def _truncate(perception: Perception, n_frames: int) -> Perception:
    """Perception of the analysed prefix only, so rules do not read the missing part as an empty room."""
    frames = perception.frames[:n_frames]
    info = perception.info
    t_cut = frames[-1].t if frames else 0.0
    cut_info = dataclasses.replace(info, frame_count=min(info.frame_count, math.floor(t_cut * info.fps) + 1))
    tracks = tuple(tr for tr in perception.tracks if tr.t_end <= t_cut)
    events = tuple(extract_events(tracks, duration=cut_info.duration))
    return dataclasses.replace(perception, info=cut_info, frames=frames, tracks=tracks, events=events)


def _finalize(perception: Perception, guard: _BudgetedDetector, plan: SamplingPlan) -> Perception:
    if guard.skipped:
        log.warning("pipeline: time budget reached, %d of %d frames skipped — analysing the first %d",
                    guard.skipped, guard.skipped + guard.processed, guard.processed)
        return _truncate(perception, guard.processed)
    if not perception.frames:
        raise ValueError("no frames could be decoded from the video")
    missing = perception.info.duration - perception.frames[-1].t
    if missing > 1.0 / plan.sample_fps + TAIL_TOLERANCE_SEC:
        log.warning("pipeline: last %.1fs of the video could not be decoded — analysing the decoded part", missing)
        return _truncate(perception, len(perception.frames))
    return perception


def _perceive(video_path: str, info: VideoInfo, time_budget_sec: float | None, started: float,
              report: ProgressFn) -> Perception:
    deadline = None if time_budget_sec is None else started + time_budget_sec - FINISH_RESERVE_SEC
    window = None if deadline is None else deadline - time.monotonic()
    sample: list[np.ndarray] = []

    def cost_at(size: int) -> float:
        if not sample:
            sample.extend(_calibration_frames(video_path, info))
        return _SHARED.calibrate(sample, size)

    plan = plan_sampling(info.duration, window, cost_at,
                         contention=max(1, _SHARED.active_jobs()))
    log.info("pipeline: %s %.1fs video, plan %.2f fps at %d px (%.3f s/frame expected, %d jobs)",
             Path(video_path).name, info.duration, plan.sample_fps, plan.input_size, plan.sec_per_frame,
             _SHARED.active_jobs())
    guard = _BudgetedDetector(_SHARED, plan, deadline)
    perception = perceive(
        video_path, guard, sample_fps=plan.sample_fps,
        on_progress=lambda p, _stage: report(PROGRESS_DETECT_END * p, "detect"),
    )
    log.info("pipeline: %d frames detected in %.1fs", guard.processed, perception.elapsed_sec)
    return _finalize(perception, guard, plan)


def _calibration_frames(video_path: str, info: VideoInfo) -> list[np.ndarray]:
    """Real frames of the video (detection cost depends on content a little); blank if unreadable."""
    times = [min(info.duration, CALIBRATION_AT_SEC * (i + 1)) for i in range(CALIBRATION_FRAMES)]
    frames = list(frames_at(video_path, times).values())
    return frames or [np.zeros((info.height, info.width, 3), np.uint8)] * CALIBRATION_FRAMES


def _judge(perception: Perception) -> Sequence[Candidate]:
    if not perception.frames:
        log.warning("pipeline: no frames were analysed in time, nothing to judge")
        return []
    return rules.find_candidates(perception, scene=None)


def preload() -> None:
    """Load and validate what every job depends on (the kb catalog). The backend calls it at startup."""
    types = catalog()
    log.info("pipeline: kb ready, %d violation types", len(types))


def analyze(
    video_path: str,
    on_progress: ProgressFn | None = None,
    time_budget_sec: float | None = None,
) -> dict[str, Any]:
    with _SHARED.job():
        return _analyze(video_path, _reporter(on_progress), time_budget_sec)


def _reporter(on_progress: ProgressFn | None) -> ProgressFn:
    def report(progress: float, stage: str) -> None:
        if on_progress is not None:
            on_progress(progress, stage)

    return report


def _analyze(video_path: str, report: ProgressFn, time_budget_sec: float | None) -> dict[str, Any]:
    started = time.monotonic()
    report(0.0, "detect")
    types: Mapping[str, ViolationType] = catalog()  # a broken kb fails now, not after detection
    info = probe(video_path)  # so does an unreadable video
    _SHARED.model()
    perception = _perceive(video_path, info, time_budget_sec, started, report)

    report(PROGRESS_DETECT_END, "rules")
    candidates = _judge(perception)

    report(PROGRESS_RULES_END, "evidence")
    analysed = perception.info.duration  # shorter than the video after a budget stop or a lost tail
    result = assemble(candidates, lambda selected: render_candidates(video_path, selected, analysed),
                      types=types, duration=analysed)
    report(1.0, "evidence")
    log.info("pipeline: %s -> %s, %d candidates, %d violations, %.1fs total", Path(video_path).name,
             result["verdict"], len(candidates), len(result["violations"]), time.monotonic() - started)
    return result
