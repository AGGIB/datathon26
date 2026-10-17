"""Who is who in the cash room: the cashier and their workplace, visitors, and tracks that are not people
inside the room (seen through the glass wall, glass reflections, fragments).

Everything is relative to the cashier of the same video, so nothing depends on pixel zones of a camera angle:
distances are measured in cashier box widths from the cashier's seat ("home").
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .appearance import Signature, mean_signature, signature_distance
from .detect import Box, Detection
from .track import Track, TrackPoint, foot

log = logging.getLogger(__name__)

MIN_TRACK_SEC = 1.0
ANCHOR_MIN_SCORE = 0.7
ANCHOR_STILL_RADIUS = 0.5  # own box widths: foot points this close count as sitting still at one spot
ANCHOR_START_SEC = 2.0  # the cashier is usually already at work when the clip starts ...
ANCHOR_START_BONUS = 1.2  # ... which breaks near-ties with a visitor who also sits still for long
ANCHOR_MAX_POINTS = 400  # subsample long tracks for the O(n^2) stillness count
HOME_RADIUS = 0.25  # in box heights: foot points this close are "the same place" when finding the seat
AT_WORK_DIST = 0.6  # cashier widths from the seat: standing up, reaching the safe next to it is still "at work"
AT_WORK_MIN_SCORE = 0.7  # seat occupancy at the reference clarity; scaled down with cashier confidence

# in-room test for non-cashier tracks
NEAR_DIST = 2.5  # cashier widths from the seat
MIN_NEAR_SEC = 1.5  # recall first: a short entry at the end of a clip must still count
# Clarity needed to count as inside: relative to how clearly this camera sees the cashier (people behind the glass
# are seen less clearly than people in the room), never above INROOM_SCORE_CAP nor below MIN_INROOM_SCORE.
MIN_INROOM_SCORE = 0.6
INROOM_SCORE_CAP = 0.8
INROOM_SCORE_MARGIN = 0.1

# a visit (visitor inside the room) starts/ends where the visitor is near and clearly seen, not through the glass
VISIT_SCORE_WINDOW_SEC = 1.0
VISIT_EDGE_SEC = 1.0  # a track touching the video start/end was already/still inside: keep that end untrimmed

# re-identification of the cashier after a break in tracking
REID_APPEARANCE = 0.35
CONCURRENT_TOL_SEC = 0.5

# desk: static non-person objects (laptop, counting machine, phone...) near the seat
DESK_MIN_PRESENCE = 0.3  # share of frames an object cell must be occupied in
DESK_MAX_DIST = 3.0  # cashier widths from the seat

# hand-off: the tracker opened a new track for the same person just before/after the old one ended
HANDOFF_SEC = 1.5
HANDOFF_DIST = 1.0  # cashier widths between the old track at the moment the new one starts
HANDOFF_APPEARANCE = 0.4  # only checked when merging two visitors


@dataclass(frozen=True)
class Workplace:
    """The cashier's seat: where the anchor track spends most time."""

    home: tuple[float, float]  # foot point
    box: Box  # typical box while seated
    width: float  # distance unit: cashier box width at the seat (stable, unlike height)
    height: float
    signature: Signature
    desk: tuple[float, float] | None = None  # centre of static desk objects, when they are detected
    clarity: float = INROOM_SCORE_CAP  # same cashier-relative confidence for roles and role events

    def distance(self, box: Box) -> float:
        fx, fy = foot(box)
        return float(np.hypot(fx - self.home[0], fy - self.home[1]) / max(self.width, 1.0))

    def occupied_by(self, box: Box, score: float = 1.0) -> bool:
        """Someone is at the cashier's working place (seat and its immediate surroundings)."""
        # Keep the seat's weaker confidence allowance proportional to the cashier's confidence.
        # Subtracting another fixed margin would let dimmer footage join previously separate stays.
        relative_score = (self.clarity + INROOM_SCORE_MARGIN) / (INROOM_SCORE_CAP + INROOM_SCORE_MARGIN)
        min_score = max(MIN_INROOM_SCORE, AT_WORK_MIN_SCORE * relative_score)
        return score >= min_score and self.distance(box) <= AT_WORK_DIST

    def toward_desk(self, box: Box) -> float | None:
        """How far the foot point lies in the direction of the desk, in cashier widths (None: desk unknown)."""
        if self.desk is None:
            return None
        dx, dy = self.desk[0] - self.home[0], self.desk[1] - self.home[1]
        norm = float(np.hypot(dx, dy))
        if norm < 1e-6:
            return None
        fx, fy = foot(box)
        return ((fx - self.home[0]) * dx + (fy - self.home[1]) * dy) / norm / max(self.width, 1.0)


@dataclass(frozen=True)
class Roles:
    workplace: Workplace | None
    cashier: tuple[Track, ...]
    visitors: tuple[Track, ...]
    rejected: tuple[tuple[int, str], ...]  # (track_id, reason)

    @property
    def in_room(self) -> tuple[Track, ...]:
        return tuple(sorted(self.cashier + self.visitors, key=lambda tr: tr.track_id))

    @property
    def cashier_ids(self) -> tuple[int, ...]:
        return tuple(tr.track_id for tr in self.cashier)

    @property
    def visitor_ids(self) -> tuple[int, ...]:
        return tuple(tr.track_id for tr in self.visitors)


def _height(b: Box) -> float:
    return b[3] - b[1]


def still_seconds(track: Track) -> float:
    """Longest time the person spends at one spot (the seat), independent of how big the box is."""
    pts = track.points
    idx = np.linspace(0, len(pts) - 1, min(len(pts), ANCHOR_MAX_POINTS)).round().astype(int)
    feet = np.array([foot(pts[i].box) for i in idx])
    radius = ANCHOR_STILL_RADIUS * float(np.median([pts[i].box[2] - pts[i].box[0] for i in idx]))
    dists = np.linalg.norm(feet[:, None, :] - feet[None, :, :], axis=2)
    share = float((dists <= radius).sum(axis=1).max()) / len(idx)
    return share * track.duration


def _median_height(track: Track) -> float:
    return float(np.median([_height(p.box) for p in track.points]))


def pick_anchor(tracks: Sequence[Track]) -> Track | None:
    """The cashier: the clearly visible track that sits still at one spot the longest.

    Presence at the start breaks near-ties. Box size only enters as a square root, so a visitor with a 1.4x bigger
    box (another camera angle) needs to sit still ~1.2x longer to win, while a small, far person sitting still
    behind the glass the whole clip (a 0.35x box) loses clearly.
    """
    if not tracks:
        return None
    clear = [tr for tr in tracks if tr.mean_score >= ANCHOR_MIN_SCORE] or list(tracks)
    tallest = max(_median_height(tr) for tr in clear) or 1.0

    def score(tr: Track) -> float:
        bonus = ANCHOR_START_BONUS if tr.t_start <= ANCHOR_START_SEC else 1.0
        return still_seconds(tr) * bonus * (_median_height(tr) / tallest) ** 0.5

    return max(clear, key=score)


def workplace_of(anchor: Track) -> Workplace:
    feet = np.array([foot(p.box) for p in anchor.points])
    heights = np.array([_height(p.box) for p in anchor.points])
    radius = HOME_RADIUS * float(np.median(heights))
    dists = np.linalg.norm(feet[:, None, :] - feet[None, :, :], axis=2)
    seat = dists[int(np.argmax((dists < radius).sum(axis=1)))] < radius
    seated = [p for p, s in zip(anchor.points, seat) if s]
    boxes = np.array([p.box for p in seated])
    box = tuple(float(v) for v in np.median(boxes, axis=0))
    home = tuple(float(v) for v in np.median(feet[seat], axis=0))
    return Workplace(
        home=(home[0], home[1]),
        box=(box[0], box[1], box[2], box[3]),
        width=float(np.median(boxes[:, 2] - boxes[:, 0])),
        height=float(np.median(boxes[:, 3] - boxes[:, 1])),
        signature=mean_signature([p.signature for p in seated if p.signature]),
        clarity=inroom_score(anchor),
    )


def _overlap_sec(a: Track, b: Track) -> float:
    return max(0.0, min(a.t_end, b.t_end) - max(a.t_start, b.t_start))


def _is_cashier_again(track: Track, cashier: Sequence[Track], wp: Workplace) -> bool:
    if any(_overlap_sec(track, c) > CONCURRENT_TOL_SEC for c in cashier):
        return False
    return signature_distance(track.signature(), wp.signature) <= REID_APPEARANCE


def _sample_step(track: Track) -> float:
    if len(track.points) < 2:
        return 0.0
    return float(np.median(np.diff([p.t for p in track.points])))


def near_seconds(track: Track, wp: Workplace) -> tuple[float, float]:
    """Time spent within NEAR_DIST of the seat and the mean detection score over that time."""
    near = [p for p in track.points if wp.distance(p.box) <= NEAR_DIST]
    if not near:
        return 0.0, 0.0
    return len(near) * _sample_step(track), float(np.mean([p.score for p in near]))


def inroom_score(anchor: Track) -> float:
    return max(MIN_INROOM_SCORE, min(INROOM_SCORE_CAP, anchor.mean_score - INROOM_SCORE_MARGIN))


def _is_visitor(track: Track, wp: Workplace, min_score: float = MIN_INROOM_SCORE) -> bool:
    secs, score = near_seconds(track, wp)
    return secs >= MIN_NEAR_SEC and score >= min_score


def is_handoff(old: Track, new: Track, wp: Workplace) -> bool:
    """`new` takes over `old`: it starts around the moment `old` ends, at the place where `old` is then."""
    if new.t_start < old.t_start or abs(new.t_start - old.t_end) > HANDOFF_SEC:
        return False
    (ax, ay), (bx, by) = foot(old.box_at(new.t_start)), foot(new.points[0].box)
    return float(np.hypot(ax - bx, ay - by)) / max(wp.width, 1.0) <= HANDOFF_DIST


def _joined(old: Track, new: Track) -> Track:
    return Track(old.track_id, tuple(p for p in old.points if p.t < new.t_start) + new.points)


def merge_visitor_handoffs(visitors: Sequence[Track], wp: Workplace) -> list[Track]:
    """One visitor whose box was re-acquired as a new track (sat down, bent over) stays one visitor."""
    merged: list[Track] = []
    for tr in sorted(visitors, key=lambda v: v.t_start):
        prev = next(
            (m for m in merged if is_handoff(m, tr, wp)
             and signature_distance(m.signature(last=4), tr.signature(first=4)) <= HANDOFF_APPEARANCE),
            None,
        )
        if prev is None:
            merged.append(tr)
        else:
            merged[merged.index(prev)] = _joined(prev, tr)
    return merged


def _window_score(points: Sequence[TrackPoint], i: int) -> float:
    half = VISIT_SCORE_WINDOW_SEC / 2
    window = [p.score for p in points if abs(p.t - points[i].t) <= half]
    return sum(window) / len(window)


def near_span(
    track: Track, wp: Workplace, duration: float | None = None, min_score: float = MIN_INROOM_SCORE
) -> Track | None:
    """The part of a visitor track between its first and last moment clearly inside, near the workplace."""
    pts = track.points
    near = [i for i, p in enumerate(pts) if wp.distance(p.box) <= NEAR_DIST and _window_score(pts, i) >= min_score]
    if not near:
        return None
    first = 0 if track.t_start <= VISIT_EDGE_SEC else near[0]
    last = len(pts) - 1 if duration is not None and track.t_end >= duration - VISIT_EDGE_SEC else near[-1]
    return Track(track.track_id, pts[first:last + 1])


def _classify(track: Track, cashier: list[Track], wp: Workplace, min_score: float) -> str:
    if _is_cashier_again(track, cashier, wp):
        return "cashier"
    return "visitor" if _is_visitor(track, wp, min_score) else "outside"


def _absorb_cashier_handoffs(cashier: list[Track], others: list[Track], wp: Workplace) -> list[Track]:
    """Fragments that hand over to/from a cashier track (walking in, standing up) are the cashier too."""
    changed = True
    while changed:
        changed = False
        for tr in list(others):
            if any(is_handoff(tr, c, wp) or is_handoff(c, tr, wp) for c in cashier):
                cashier.append(tr)
                others.remove(tr)
                changed = True
    return others


def desk_point(frames: Sequence[Sequence[Detection]], wp: Workplace) -> tuple[float, float] | None:
    """Centre of objects that stay put near the seat for a good part of the video (the desk equipment)."""
    cell = max(wp.width / 2, 1.0)
    seen: dict[tuple[int, int], set[int]] = {}
    centers: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for fi, dets in enumerate(frames):
        for d in dets:
            if d.label == "person":
                continue
            c = ((d.box[0] + d.box[2]) / 2, (d.box[1] + d.box[3]) / 2)
            key = (int(c[0] // cell), int(c[1] // cell))
            seen.setdefault(key, set()).add(fi)
            centers.setdefault(key, []).append(c)
    static = [
        pt for key, frames_in in seen.items() if len(frames_in) >= DESK_MIN_PRESENCE * len(frames)
        for pt in centers[key]
        if np.hypot(pt[0] - wp.home[0], pt[1] - wp.home[1]) / max(wp.width, 1.0) <= DESK_MAX_DIST
    ]
    if not static:
        return None
    x, y = np.median(np.asarray(static), axis=0)
    return float(x), float(y)


def _chain_cashier(cashier: Sequence[Track], wp: Workplace) -> list[Track]:
    """Join cashier fragments that hand over to each other into one track (keeps the first id)."""
    chained: list[Track] = []
    for tr in sorted(cashier, key=lambda c: c.t_start):
        if chained and is_handoff(chained[-1], tr, wp):
            chained[-1] = _joined(chained[-1], tr)
        else:
            chained.append(tr)
    return chained


def assign_roles(tracks: Sequence[Track], duration: float | None = None) -> Roles:
    long_enough = [tr for tr in tracks if tr.duration >= MIN_TRACK_SEC]
    short = [(tr.track_id, "short") for tr in tracks if tr.duration < MIN_TRACK_SEC]
    anchor = pick_anchor(long_enough)
    if anchor is None:
        return Roles(None, (), (), tuple(short))
    wp = workplace_of(anchor)
    clarity = wp.clarity
    cashier, visitors, kinds = [anchor], [], {}
    for tr in sorted((t for t in long_enough if t is not anchor), key=lambda t: t.mean_score, reverse=True):
        kinds[tr.track_id] = _classify(tr, cashier, wp, clarity)
        {"cashier": cashier, "visitor": visitors}.get(kinds[tr.track_id], []).append(tr)
    undecided = [t for t in long_enough if kinds.get(t.track_id) in ("visitor", "outside")]
    remaining = {t.track_id for t in _absorb_cashier_handoffs(cashier, undecided, wp)}
    visitors = [v for v in visitors if v.track_id in remaining]
    rejected = short + [(tid, k) for tid, k in kinds.items() if k == "outside" and tid in remaining]
    spans = [near_span(v, wp, duration, clarity) for v in merge_visitor_handoffs(visitors, wp)]
    roles = Roles(
        workplace=wp,
        cashier=tuple(_chain_cashier(cashier, wp)),
        visitors=tuple(sorted((v for v in spans if v is not None), key=lambda t: t.t_start)),
        rejected=tuple(sorted(rejected)),
    )
    log.debug("roles: cashier=%s visitors=%s rejected=%s", roles.cashier_ids, roles.visitor_ids, roles.rejected)
    return roles
