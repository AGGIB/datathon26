"""Lightweight IoU tracker for a handful of people in a fixed-camera room, plus post-hoc track cleanup."""

from collections.abc import Sequence
from dataclasses import dataclass, field

from .appearance import Signature, mean_signature, signature_distance
from .detect import Box, Detection

MIN_IOU = 0.1
MAX_CENTER_SHIFT = 0.6  # fraction of the previous box diagonal


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _center_shift(a: Box, b: Box) -> float:
    ax, ay = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    diag = ((a[2] - a[0]) ** 2 + (a[3] - a[1]) ** 2) ** 0.5 or 1.0
    return (((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5) / diag


def box_diag(b: Box) -> float:
    return ((b[2] - b[0]) ** 2 + (b[3] - b[1]) ** 2) ** 0.5


def foot(b: Box) -> tuple[float, float]:
    """Bottom-centre of a box: where a standing person touches the floor."""
    return (b[0] + b[2]) / 2, b[3]


@dataclass(frozen=True)
class TrackPoint:
    t: float
    box: Box
    score: float
    signature: Signature = ()


@dataclass(frozen=True)
class Track:
    track_id: int
    points: tuple[TrackPoint, ...]

    @property
    def t_start(self) -> float:
        return self.points[0].t

    @property
    def t_end(self) -> float:
        return self.points[-1].t

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start

    @property
    def mean_score(self) -> float:
        return sum(p.score for p in self.points) / len(self.points)

    def alive_at(self, t: float) -> bool:
        return self.t_start <= t <= self.t_end

    def box_at(self, t: float) -> Box:
        return min(self.points, key=lambda p: abs(p.t - t)).box

    def foot_at(self, t: float) -> tuple[float, float]:
        return foot(self.box_at(t))

    def signature(self, first: int | None = None, last: int | None = None) -> Signature:
        """Mean appearance over all points, or only over the first/last N points."""
        pts = self.points[:first] if first else self.points[-last:] if last else self.points
        return mean_signature([p.signature for p in pts if p.signature])


@dataclass
class _Live:
    track_id: int
    points: list[TrackPoint] = field(default_factory=list)

    @property
    def last(self) -> TrackPoint:
        return self.points[-1]


class IoUTracker:
    def __init__(self, max_gap_sec: float = 1.0, label: str = "person") -> None:
        self._max_gap = max_gap_sec
        self._label = label
        self._live: list[_Live] = []
        self._done: list[_Live] = []
        self._next_id = 1

    def update(
        self, t: float, detections: list[Detection], signatures: Sequence[Signature] | None = None
    ) -> list[tuple[int, Detection]]:
        """Assign this frame's detections to tracks; `signatures` (optional) is aligned with `detections`."""
        self._retire(t)
        sigs_all = list(signatures) if signatures is not None else [()] * len(detections)
        picked = [(d, s) for d, s in zip(detections, sigs_all) if d.label == self._label]
        dets = [d for d, _ in picked]
        sigs = [s for _, s in picked]
        pairs = sorted(
            (
                (-iou(trk.last.box, d.box), _center_shift(trk.last.box, d.box), ti, di)
                for ti, trk in enumerate(self._live)
                for di, d in enumerate(dets)
            )
        )
        used_t: set[int] = set()
        used_d: set[int] = set()
        assigned: list[tuple[int, Detection]] = []
        for neg_iou, shift, ti, di in pairs:
            if ti in used_t or di in used_d or (-neg_iou < MIN_IOU and shift > MAX_CENTER_SHIFT):
                continue
            used_t.add(ti)
            used_d.add(di)
            self._live[ti].points.append(TrackPoint(t, dets[di].box, dets[di].score, sigs[di]))
            assigned.append((self._live[ti].track_id, dets[di]))
        for di, d in enumerate(dets):
            if di not in used_d:
                live = _Live(self._next_id, [TrackPoint(t, d.box, d.score, sigs[di])])
                self._next_id += 1
                self._live.append(live)
                assigned.append((live.track_id, d))
        return assigned

    def tracks(self) -> list[Track]:
        everything = self._done + self._live
        return sorted((Track(x.track_id, tuple(x.points)) for x in everything), key=lambda tr: tr.track_id)

    def _retire(self, t: float) -> None:
        still = []
        for trk in self._live:
            (still if t - trk.last.t <= self._max_gap else self._done).append(trk)
        self._live = still


# --- post-hoc cleanup -------------------------------------------------------------------------------------------

STITCH_GAP_SEC = 2.0
STITCH_BASE_SHIFT = 0.6  # allowed centre jump across a gap, in box diagonals ...
STITCH_SHIFT_PER_SEC = 0.5  # ... growing with the gap length
STITCH_MAX_SIZE_RATIO = 1.8
STITCH_MAX_APPEARANCE = 0.5  # Hellinger distance between end/start colour signatures
STITCH_SIGNATURE_POINTS = 4
DUP_CONTAINMENT = 0.8
DUP_SHARE = 0.7


def _center(b: Box) -> tuple[float, float]:
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def _link_cost(a: Track, b: Track, max_gap_sec: float, max_appearance: float) -> float | None:
    """Cost of `b` continuing `a`, or None when the pair is not plausible."""
    gap = b.t_start - a.t_end
    if not 0 < gap <= max_gap_sec:
        return None
    end, start = a.points[-1].box, b.points[0].box
    diag = max(box_diag(end), box_diag(start), 1.0)
    (ax, ay), (bx, by) = _center(end), _center(start)
    shift = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 / diag
    if shift > STITCH_BASE_SHIFT + STITCH_SHIFT_PER_SEC * gap:
        return None
    hs = sorted((end[3] - end[1], start[3] - start[1]))
    if hs[0] <= 0 or hs[1] / hs[0] > STITCH_MAX_SIZE_RATIO:
        return None
    look = signature_distance(a.signature(last=STITCH_SIGNATURE_POINTS), b.signature(first=STITCH_SIGNATURE_POINTS))
    has_sigs = bool(a.points[-1].signature and b.points[0].signature)
    if has_sigs and look > max_appearance:
        return None
    return shift + gap / max_gap_sec + (look if has_sigs else 0.0)


def stitch_tracks(
    tracks: Sequence[Track],
    max_gap_sec: float = STITCH_GAP_SEC,
    max_appearance: float = STITCH_MAX_APPEARANCE,
) -> list[Track]:
    """Join fragments of one person broken by a short detection dropout (globally cheapest links first)."""
    links = sorted(
        (cost, i, j)
        for i, a in enumerate(tracks)
        for j, b in enumerate(tracks)
        if i != j and (cost := _link_cost(a, b, max_gap_sec, max_appearance)) is not None
    )
    nxt: dict[int, int] = {}
    prv: dict[int, int] = {}
    for _, i, j in links:
        if i not in nxt and j not in prv:
            nxt[i], prv[j] = j, i
    out = []
    for i in range(len(tracks)):
        if i in prv:
            continue
        chain = [i]
        while chain[-1] in nxt:
            chain.append(nxt[chain[-1]])
        points = tuple(p for k in chain for p in tracks[k].points)
        out.append(Track(tracks[i].track_id, points))
    return sorted(out, key=lambda tr: tr.track_id)


def _containment(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return ix * iy / smaller if smaller > 0 else 0.0


def _is_duplicate_of(small: Track, big: Track) -> bool:
    boxes = {round(p.t, 3): p.box for p in big.points}
    common = [(p.box, boxes[round(p.t, 3)]) for p in small.points if round(p.t, 3) in boxes]
    if len(common) < DUP_SHARE * len(small.points):
        return False
    inside = sum(_containment(a, b) >= DUP_CONTAINMENT for a, b in common)
    return inside >= DUP_SHARE * len(common)


def drop_duplicate_tracks(tracks: Sequence[Track]) -> list[Track]:
    """Remove tracks that are a second box on someone already tracked (e.g. upper body inside full body)."""
    ordered = sorted(tracks, key=lambda tr: len(tr.points), reverse=True)
    kept: list[Track] = []
    for tr in ordered:
        if not any(_is_duplicate_of(tr, k) for k in kept):
            kept.append(tr)
    return sorted(kept, key=lambda tr: tr.track_id)
