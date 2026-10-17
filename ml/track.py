"""Lightweight IoU tracker for a handful of people in a fixed-camera room."""

from dataclasses import dataclass, field

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


@dataclass(frozen=True)
class TrackPoint:
    t: float
    box: Box
    score: float


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

    def update(self, t: float, detections: list[Detection]) -> list[tuple[int, Detection]]:
        self._retire(t)
        dets = [d for d in detections if d.label == self._label]
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
            self._live[ti].points.append(TrackPoint(t, dets[di].box, dets[di].score))
            assigned.append((self._live[ti].track_id, dets[di]))
        for di, d in enumerate(dets):
            if di not in used_d:
                live = _Live(self._next_id, [TrackPoint(t, d.box, d.score)])
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
