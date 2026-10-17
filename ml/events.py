"""Room-level events from person tracks.

extract_events      — role-agnostic: enter / exit / room_empty / room_empty_at_start / crowd
extract_role_events — role-aware:   cashier_left / cashier_returned / visitor_enter / visitor_exit /
                                    visitors_overlap / person_behind_cashier
"""

from collections.abc import Sequence
from dataclasses import dataclass

from .roles import Roles, Workplace
from .track import STITCH_GAP_SEC, Track, TrackPoint, stitch_tracks

MIN_TRACK_SEC = 1.0
MIN_TRACK_SCORE = 0.6
MIN_EMPTY_SEC = 2.0  # longer than the tracker's and the stitcher's dropout tolerance
EDGE_SEC = 1.0  # tracks starting/ending this close to the video edges are not enter/exit events
CROWD_COUNT = 3


@dataclass(frozen=True)
class Event:
    """kind: see the module docstring. t — when it happens; t_end — end of the interval (None if open/none);
    track_id — the person concerned; track_ids — all tracks involved; count — people in a crowd/overlap."""

    kind: str
    t: float
    t_end: float | None = None
    track_id: int | None = None
    count: int | None = None
    track_ids: tuple[int, ...] = ()


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
    stitch_gap_sec: float = STITCH_GAP_SEC,
) -> list[Event]:
    """Room events from person tracks; fragments split by a dropout <= stitch_gap_sec count as one person."""
    joined = stitch_tracks(tracks, max_gap_sec=stitch_gap_sec) if stitch_gap_sec > 0 else list(tracks)
    good = reliable_tracks(joined, min_track_sec, min_track_score)
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


# --- role-aware events -----------------------------------------------------------------------------------------

CASHIER_MIN_AWAY_SEC = 3.0  # away from the seat at least this long (or until the video ends)
CASHIER_MIN_AWAY_AT_END_SEC = 1.5
MIN_OVERLAP_SEC = 2.0
BEHIND_DIST = 1.3  # cashier widths from the seat
BEHIND_MIN_HEIGHT = 0.9  # standing person next to a seated cashier is at least about as tall
BEHIND_MIN_SCORE = 0.8
BEHIND_MAX_TOWARD_DESK = 0.8  # further than this toward the desk is the visitor side
MIN_BEHIND_SEC = 1.0
RUN_GAP_SEC = 1.0


def _point_at(track: Track, t: float, tol: float) -> TrackPoint | None:
    if not track.t_start - tol <= t <= track.t_end + tol:
        return None
    best = min(track.points, key=lambda p: abs(p.t - t))
    return best if abs(best.t - t) <= tol else None


def _seated_cashier(cashier: Sequence[Track], wp: Workplace, t: float, tol: float) -> int | None:
    for tr in cashier:
        p = _point_at(tr, t, tol)
        if p is not None and wp.occupied_by(p.box):
            return tr.track_id
    return None


def _runs(flags: Sequence[bool]) -> list[tuple[int, int]]:
    """Index ranges [i, j) of consecutive False values."""
    runs, start = [], None
    for i, flag in enumerate(list(flags) + [True]):
        if not flag and start is None:
            start = i
        elif flag and start is not None:
            runs.append((start, i))
            start = None
    return runs


def _cashier_events(roles: Roles, times: Sequence[float], duration: float) -> list[Event]:
    tol = (times[1] - times[0]) / 2 + 1e-6 if len(times) > 1 else 0.5
    seated = [_seated_cashier(roles.cashier, roles.workplace, t, tol) for t in times]
    events = []
    for i, j in _runs([s is not None for s in seated]):
        if i == 0:
            continue  # nobody at the desk yet: arrival, not a return
        left_t, left_id = times[i - 1], seated[i - 1]
        back = j < len(times)
        away = (times[j] if back else duration) - left_t
        if away < (CASHIER_MIN_AWAY_SEC if back else CASHIER_MIN_AWAY_AT_END_SEC):
            continue
        ids = tuple(dict.fromkeys([left_id] + ([seated[j]] if back else [])))
        events.append(Event("cashier_left", left_t, t_end=times[j] if back else None, track_id=left_id, track_ids=ids))
        if back:
            events.append(Event("cashier_returned", times[j], track_id=seated[j], track_ids=ids))
    return events


def _visitor_events(visitors: Sequence[Track], duration: float, edge_sec: float) -> list[Event]:
    events = []
    for v in visitors:
        if v.t_start > edge_sec:
            events.append(Event("visitor_enter", v.t_start, track_id=v.track_id, track_ids=(v.track_id,)))
        if v.t_end < duration - edge_sec:
            events.append(Event("visitor_exit", v.t_end, track_id=v.track_id, track_ids=(v.track_id,)))
    return events


def _overlap_events(visitors: Sequence[Track]) -> list[Event]:
    events = []
    start: float | None = None
    present: set[int] = set()
    for t in sorted({v.t_start for v in visitors} | {v.t_end for v in visitors}):
        now = {v.track_id for v in visitors if v.t_start <= t < v.t_end}
        if len(now) >= 2:
            start = t if start is None else start
            present |= now
        elif start is not None:
            if t - start >= MIN_OVERLAP_SEC:
                ids = tuple(sorted(present))
                events.append(Event("visitors_overlap", start, t_end=t, count=len(ids), track_ids=ids))
            start, present = None, set()
    return events


def _is_behind(p: TrackPoint, wp: Workplace) -> bool:
    if p.score < BEHIND_MIN_SCORE or (p.box[3] - p.box[1]) < BEHIND_MIN_HEIGHT * wp.height:
        return False
    if wp.distance(p.box) > BEHIND_DIST:
        return False
    toward = wp.toward_desk(p.box)
    return toward is None or toward <= BEHIND_MAX_TOWARD_DESK


def _behind_events(visitors: Sequence[Track], wp: Workplace) -> list[Event]:
    events = []
    for v in visitors:
        hits = [p.t for p in v.points if _is_behind(p, wp)]
        runs: list[list[float]] = []
        for t in hits:
            if runs and t - runs[-1][-1] <= RUN_GAP_SEC:
                runs[-1].append(t)
            else:
                runs.append([t])
        for run in runs:
            if run[-1] - run[0] >= MIN_BEHIND_SEC:
                events.append(Event("person_behind_cashier", run[0], t_end=run[-1], track_id=v.track_id,
                                    track_ids=(v.track_id,)))
    return events


def extract_role_events(
    roles: Roles, times: Sequence[float], duration: float, edge_sec: float = EDGE_SEC
) -> list[Event]:
    """Role-aware events; `times` are the sampled frame timestamps (presence is checked at each of them)."""
    if roles.workplace is None or not times:
        return []
    events = (
        _cashier_events(roles, times, duration)
        + _visitor_events(roles.visitors, duration, edge_sec)
        + _overlap_events(roles.visitors)
        + _behind_events(roles.visitors, roles.workplace)
    )
    return sorted(events, key=lambda e: (e.t, e.kind))
