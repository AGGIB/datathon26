"""Room-level events from person tracks: who entered, who left, when the room was empty or crowded."""

from collections.abc import Sequence
from dataclasses import dataclass

from .track import Track

MIN_TRACK_SEC = 1.0
MIN_TRACK_SCORE = 0.6
MIN_EMPTY_SEC = 1.0
EDGE_SEC = 1.0  # tracks starting/ending this close to the video edges are not enter/exit events
CROWD_COUNT = 3


@dataclass(frozen=True)
class Event:
    kind: str  # enter | exit | room_empty | room_empty_at_start | crowd
    t: float
    t_end: float | None = None
    track_id: int | None = None
    count: int | None = None


def occupancy(tracks: Sequence[Track], times: Sequence[float]) -> list[int]:
    return [sum(tr.alive_at(t) for tr in tracks) for t in times]


def reliable_tracks(tracks: Sequence[Track], min_track_sec: float, min_track_score: float) -> list[Track]:
    return [tr for tr in tracks if tr.duration >= min_track_sec and tr.mean_score >= min_track_score]


def _empty_intervals(tracks: Sequence[Track], duration: float) -> list[tuple[float, float]]:
    """Spans with nobody present, including the leading one (people walk into an empty room)."""
    gaps = []
    covered_until = 0.0
    for start, end in sorted((tr.t_start, tr.t_end) for tr in tracks):
        if start > covered_until:
            gaps.append((covered_until, start))
        covered_until = max(covered_until, end)
    if covered_until < duration:
        gaps.append((covered_until, duration))
    return gaps


def _count_after(tracks: Sequence[Track], t: float) -> int:
    """People present right after t (a track ending exactly at t no longer counts)."""
    return sum(tr.t_start <= t < tr.t_end for tr in tracks)


def _crowd_events(tracks: Sequence[Track], crowd_count: int) -> list[Event]:
    events = []
    start: float | None = None
    peak = 0
    for t in sorted({tr.t_start for tr in tracks} | {tr.t_end for tr in tracks}):
        count = _count_after(tracks, t)
        if count >= crowd_count:
            start = t if start is None else start
            peak = max(peak, count)
        elif start is not None:
            events.append(Event("crowd", start, t_end=t, count=peak))
            start, peak = None, 0
    return events


def extract_events(
    tracks: Sequence[Track],
    duration: float,
    min_track_sec: float = MIN_TRACK_SEC,
    min_track_score: float = MIN_TRACK_SCORE,
    min_empty_sec: float = MIN_EMPTY_SEC,
    edge_sec: float = EDGE_SEC,
    crowd_count: int = CROWD_COUNT,
) -> list[Event]:
    good = reliable_tracks(tracks, min_track_sec, min_track_score)
    events: list[Event] = []
    for tr in good:
        if tr.t_start > edge_sec:
            events.append(Event("enter", tr.t_start, track_id=tr.track_id))
        if tr.t_end < duration - edge_sec:
            events.append(Event("exit", tr.t_end, track_id=tr.track_id))
    for start, end in _empty_intervals(good, duration):
        if end - start >= min_empty_sec:
            events.append(Event("room_empty_at_start" if start == 0.0 else "room_empty", start, t_end=end))
    events.extend(_crowd_events(good, crowd_count))
    return sorted(events, key=lambda e: (e.t, e.kind))
