"""Stage 1 of the pipeline: frames -> detections -> person tracks -> roles -> room events.

perceive()        runs the detector over a video (GPU part) and then analyze_frames().
analyze_frames()  is the pure post-detection part: tracking, cleanup, roles, events. Dev tools feed it cached
                  detections (ml/tools/det_cache.py) so the logic can be tuned without the GPU.
"""

import dataclasses
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .appearance import Signature, color_signature, crop_person
from .detect import INPUT_SIZE, Detection, Detector
from .events import MIN_EMPTY_SEC, Event, extract_events, extract_role_events
from .roles import Roles, assign_roles, desk_point
from .track import STITCH_GAP_SEC, IoUTracker, Track, drop_duplicate_tracks, stitch_tracks
from .video import VideoInfo, probe, sample_frames

SAMPLE_FPS = 4.0
BATCH = 8
TRACKER_GAP_SEC = 1.0
TRACKER_GAP_SAMPLES = 2.5  # a track survives this many missed samples (at low sampling rates)

ProgressFn = Callable[[float, str], None]
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrameObjects:
    t: float
    detections: tuple[Detection, ...]
    signatures: tuple[Signature, ...] = ()  # aligned with detections; () for non-person boxes


@dataclass(frozen=True)
class Tolerances:
    """Dropout tolerances scaled to the sampling interval: at 1.25 fps one missed detection is already 1.6 s."""

    tracker_gap_sec: float
    stitch_gap_sec: float
    min_empty_sec: float  # an empty room / exit must last clearly longer than any bridged dropout


def tolerances(sample_fps: float) -> Tolerances:
    step = 1.0 / sample_fps
    tracker_gap = max(TRACKER_GAP_SEC, TRACKER_GAP_SAMPLES * step)
    longer = tracker_gap + step
    return Tolerances(tracker_gap, max(STITCH_GAP_SEC, longer), max(MIN_EMPTY_SEC, longer))


@dataclass(frozen=True)
class Scene:
    """Everything derived from detections: cleaned tracks, roles and events."""

    all_tracks: tuple[Track, ...]  # after stitching and duplicate removal, before role filtering
    roles: Roles
    events: tuple[Event, ...]


@dataclass(frozen=True)
class Perception:
    info: VideoInfo
    frames: tuple[FrameObjects, ...]
    tracks: tuple[Track, ...]  # people inside the room: cashier + visitors (see roles)
    events: tuple[Event, ...]  # role-agnostic and role-aware events, sorted by time
    elapsed_sec: float
    roles: Roles | None = None
    all_tracks: tuple[Track, ...] = ()
    sample_fps: float = SAMPLE_FPS  # frames analysed per second of video
    input_size: int = INPUT_SIZE  # detector input side in px: the resolution the evidence was seen at

    def objects_at(self, t: float, label: str) -> tuple[Detection, ...]:
        nearest = min(self.frames, key=lambda f: abs(f.t - t))
        return tuple(d for d in nearest.detections if d.label == label)


def observe(t: float, frame: np.ndarray, detections: Sequence[Detection]) -> FrameObjects:
    """Attach an appearance signature to every person box of one frame."""
    sigs = tuple(color_signature(crop_person(frame, d.box)) if d.label == "person" else () for d in detections)
    return FrameObjects(t, tuple(detections), sigs)


def track_people(frames: Sequence[FrameObjects], tol: Tolerances) -> list[Track]:
    tracker = IoUTracker(max_gap_sec=tol.tracker_gap_sec)
    for f in frames:
        tracker.update(f.t, list(f.detections), f.signatures or None)
    return drop_duplicate_tracks(stitch_tracks(tracker.tracks(), max_gap_sec=tol.stitch_gap_sec))


def analyze_frames(frames: Sequence[FrameObjects], duration: float, sample_fps: float = SAMPLE_FPS) -> Scene:
    tol = tolerances(sample_fps)
    tracks = track_people(frames, tol)
    roles = assign_roles(tracks)
    if roles.workplace is not None:
        desk = desk_point([f.detections for f in frames], roles.workplace)
        roles = dataclasses.replace(roles, workplace=dataclasses.replace(roles.workplace, desk=desk))
    times = [f.t for f in frames]
    events = extract_events(
        roles.in_room, duration=duration, min_empty_sec=tol.min_empty_sec, stitch_gap_sec=tol.stitch_gap_sec
    ) + extract_role_events(roles, times, duration)
    log.info("scene: %d tracks, cashier=%s visitors=%s, %d events", len(tracks), roles.cashier_ids,
             roles.visitor_ids, len(events))
    return Scene(tuple(tracks), roles, tuple(sorted(events, key=lambda e: (e.t, e.kind))))


def perceive(
    video_path: Path | str,
    detector: Detector,
    sample_fps: float = SAMPLE_FPS,
    on_progress: ProgressFn | None = None,
    input_size: int | None = None,
) -> Perception:
    """`input_size` is recorded on the result; by default it is read from the detector (640 if unknown)."""
    started = time.monotonic()
    info = probe(video_path)
    frames: list[FrameObjects] = []
    batch: list[tuple[float, np.ndarray]] = []
    expected = max(1, int(info.duration * sample_fps))

    def flush() -> None:
        for (t, frame), dets in zip(batch, detector.detect([f for _, f in batch])):
            frames.append(observe(t, frame, dets))
        batch.clear()
        if on_progress:
            on_progress(min(1.0, len(frames) / expected), "detect")

    for t, frame in sample_frames(video_path, sample_fps):
        batch.append((t, frame))
        if len(batch) == BATCH:
            flush()
    if batch:
        flush()

    scene = analyze_frames(frames, info.duration, sample_fps)
    return Perception(
        info=info,
        frames=tuple(frames),
        tracks=scene.roles.in_room,
        events=scene.events,
        elapsed_sec=time.monotonic() - started,
        roles=scene.roles,
        all_tracks=scene.all_tracks,
        sample_fps=sample_fps,
        input_size=input_size or getattr(detector, "input_size", INPUT_SIZE),
    )
