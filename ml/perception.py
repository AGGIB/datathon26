"""Stage 1 of the pipeline: frames -> detections -> person tracks -> room events."""

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .detect import Detection, Detector
from .events import Event, extract_events
from .track import IoUTracker, Track
from .video import VideoInfo, probe, sample_frames

SAMPLE_FPS = 4.0
BATCH = 8

ProgressFn = Callable[[float, str], None]


@dataclass(frozen=True)
class FrameObjects:
    t: float
    detections: tuple[Detection, ...]


@dataclass(frozen=True)
class Perception:
    info: VideoInfo
    frames: tuple[FrameObjects, ...]
    tracks: tuple[Track, ...]
    events: tuple[Event, ...]
    elapsed_sec: float

    def objects_at(self, t: float, label: str) -> tuple[Detection, ...]:
        nearest = min(self.frames, key=lambda f: abs(f.t - t))
        return tuple(d for d in nearest.detections if d.label == label)


def perceive(
    video_path: Path | str,
    detector: Detector,
    sample_fps: float = SAMPLE_FPS,
    on_progress: ProgressFn | None = None,
) -> Perception:
    started = time.monotonic()
    info = probe(video_path)
    tracker = IoUTracker(max_gap_sec=1.0)
    frames: list[FrameObjects] = []
    batch: list[tuple[float, object]] = []
    expected = max(1, int(info.duration * sample_fps))

    def flush() -> None:
        for (t, _), dets in zip(batch, detector.detect([f for _, f in batch])):
            tracker.update(t, dets)
            frames.append(FrameObjects(t, tuple(dets)))
        batch.clear()
        if on_progress:
            on_progress(min(1.0, len(frames) / expected), "detect")

    for t, frame in sample_frames(video_path, sample_fps):
        batch.append((t, frame))
        if len(batch) == BATCH:
            flush()
    if batch:
        flush()

    tracks = tuple(tracker.tracks())
    events = tuple(extract_events(tracks, duration=info.duration))
    return Perception(info, tuple(frames), tracks, events, time.monotonic() - started)
