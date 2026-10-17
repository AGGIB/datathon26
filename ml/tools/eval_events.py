"""Dev tool: compare perception events with expectations derived from kb/annotations.csv.

    python -m ml.tools.eval_events [--fps 6] [--only X-18,X-19] [--build]

Runs the production post-detection stage (ml.perception.analyze_frames) on cached detections
(ml/tools/det_cache.py; --build runs the detector for videos without a cache) and prints:
  1. role events vs expectations for the videos named in the task (hit = detected within TOL_SEC of the window);
  2. room enter/exit events vs the door crossings seen in each video (spurious = matches no crossing).

Expectation windows: kb/annotations.csv gives the interval and peak of each scene (t_start/t_peak/t_end); the
moment of the event inside that interval was read off the frames. A window [lo, hi] spans the event itself, e.g.
for a cashier leaving: from standing up / starting to leave to walking away from the workplace.
"""

import argparse
import functools
import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from ml.appearance import color_signature
from ml.events import Event
from ml.labels import normalize_video_name
from ml.perception import FrameObjects, analyze_frames
from ml.tools import det_cache

TOL_SEC = 2.0
CACHE_FPS = 6.0
ROOM_KINDS = ("enter", "exit")

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Expect:
    """An event of `kind` must happen within [lo - TOL_SEC, hi + TOL_SEC]."""

    video: str
    kind: str
    lo: float
    hi: float
    note: str


@dataclass(frozen=True)
class Absent:
    """No event of `kind` may happen in the video."""

    video: str
    kind: str
    note: str


def _leave(video: str, lo: float, hi: float, note: str) -> Expect:
    return Expect(video, "cashier_left", lo, hi, note)


ROLE_EXPECTATIONS: tuple[Expect | Absent, ...] = (
    # cashier leaves: annotation interval (peak) -> stands up / starts leaving .. walks away from the workplace
    _leave("X-11", 11.0, 16.5, "annot 10.0-25.5 (18) laptop left open; stands 11, at safe, walks to door 16.5"),
    _leave("X-12", 19.5, 34.0, "annot 10.0-42.2; stands 19.5, works at desk/cabinet, walks to door 34"),
    _leave("X-15", 13.0, 15.0, "annot 11.0-20.8 (15) door left open; at safe, walks off 13-15"),
    _leave("X-22", 14.0, 16.0, "annot 13.0-21.0 (14) key left in safe; at safe, walks off 14-16"),
    _leave("X-23", 12.0, 13.5, "annot 7.0-18.0; at safe 6-12, walks off 12-13.5"),
    _leave("X-24", 5.0, 9.0, "annot 4.0-15.0 (8) stamps left; stands 5, at desk, walks off 9"),
    _leave("X-25", 24.5, 26.0, "annot 5.0-31.0; at cabinet, walks off 24.5-26"),
    _leave("X-26", 9.5, 13.5, "annot 8.0-20.0 (9) cash left; stands 9.5, at safe, walks off 13.5"),
    _leave("X-27", 18.5, 20.0, "annot 8.0-26.0; at safe, walks off 18.5-20"),
    _leave("X-28", 6.5, 9.0, "annot 6.0-17.0 (7) screen unlocked; stands 6.5, walks off 9"),
    _leave("X-29", 7.5, 9.5, "annot 5.0-17.0; stands 7.5, walks off 9.5"),
    # someone behind the cashier
    Expect("X-30", "person_behind_cashier", 12.5, 13.5, "annot 4.0-19.0 (13): second person comes in behind"),
    Expect("X-33", "person_behind_cashier", 10.5, 12.0, "annot 4.0-14.0: second person appears ~10.5"),
    Absent("X-31", "person_behind_cashier", "annot 4.0-24.0: nobody else comes in"),
    # two visitors at once
    Expect("X-18", "visitors_overlap", 16.0, 17.0, "annot 16.5-22.5 (20): 2nd visitor while 1st stays"),
    Absent("X-19", "visitors_overlap", "annot 35.0-57.3: 2nd visitor comes after the 1st left"),
    # visitor change
    Expect("X-4", "visitor_exit", 52.5, 53.5, "annot 49.0-60.0 (57): 1st visitor leaves"),
    Expect("X-4", "visitor_enter", 56.5, 57.5, "annot 49.0-60.0 (57): next visitor comes in"),
    Expect("X-5", "visitor_exit", 59.5, 60.5, "annot 45.0-71.5: 1st visitor leaves"),
    Expect("X-5", "visitor_enter", 64.0, 65.0, "annot 45.0-71.5: next visitor comes in"),
    Expect("X-38", "visitor_exit", 26.5, 27.5, "annot 24.0-59.0 (33): 1st visitor leaves"),
    Expect("X-38", "visitor_enter", 32.5, 33.5, "annot 24.0-59.0 (33): next visitor comes in"),
    Expect("X-39", "visitor_exit", 36.5, 37.5, "annot 26.0-79.0: 1st visitor leaves"),
    Expect("X-39", "visitor_enter", 46.5, 47.5, "annot 26.0-79.0: next visitor comes in"),
)

# Door crossings (anyone) seen in the videos, as (kind, lo, hi). Crossings within 1 s of the video edges are left
# out (extract_events does not report them). A cashier who leaves through the glass door stays visible behind it,
# so her exit window runs from the door to the moment she disappears.
DOOR_CROSSINGS: dict[str, tuple[tuple[str, float, float], ...]] = {
    "X-1": (), "X-3": (), "X-8": (), "X-10": (), "X-14": (), "X-20": (), "X-21": (), "X-40": (),
    "X-43": (), "X-44": (), "X-45": (), "X-46": (),
    "X-2": (("enter", 4.0, 4.0),),
    "X-4": (("enter", 4.5, 4.5), ("exit", 53.0, 53.0), ("enter", 57.0, 57.0)),
    "X-47": (("enter", 4.5, 4.5), ("exit", 53.0, 53.0), ("enter", 57.0, 57.0)),
    "X-5": (("exit", 60.0, 60.0), ("enter", 64.5, 64.5)),
    "X-48": (("exit", 60.0, 60.0), ("enter", 64.5, 64.5)),
    "X-11": (("exit", 21.0, 24.2),),
    "X-12": (("exit", 37.0, 41.0),),
    "X-13": (("exit", 60.5, 63.8),),
    "X-15": (("exit", 17.0, 20.0),),
    "X-16": (("enter", 4.0, 4.0),),
    "X-17": (("enter", 3.7, 3.7), ("exit", 11.0, 11.0)),
    "X-18": (("enter", 2.7, 2.7), ("enter", 16.0, 16.0)),
    "X-19": (("enter", 2.7, 2.7), ("exit", 46.5, 46.5), ("enter", 51.0, 51.0)),
    "X-22": (("exit", 18.0, 19.7),),
    "X-23": (("exit", 15.0, 17.0),),
    "X-24": (("exit", 10.5, 13.2),),
    "X-25": (("exit", 27.5, 30.2),),
    "X-26": (("exit", 15.0, 17.7),),
    "X-27": (("exit", 22.0, 25.2),),
    "X-28": (("exit", 12.5, 15.5),),
    "X-29": (("exit", 12.5, 15.8),),
    "X-30": (("enter", 12.5, 12.5),),
    "X-31": (("enter", 2.0, 2.0),),
    "X-32": (("enter", 1.2, 1.2),),
    "X-33": (("enter", 10.5, 10.5),),
    "X-34": (("exit", 14.0, 14.0),),
    "X-35": (("exit", 53.8, 53.8),),
    "X-37": (("enter", 4.0, 4.0),),
    "X-38": (("exit", 27.0, 27.0), ("enter", 33.0, 33.0)),
    "X-39": (("exit", 37.0, 37.0), ("enter", 47.0, 47.0), ("exit", 76.0, 76.0)),
}

EventSource = Callable[[str], Sequence[Event]]


@dataclass(frozen=True)
class Row:
    video: str
    kind: str
    expected: str
    detected: tuple[float, ...]
    verdict: str  # hit | miss | ok | false alarm
    false_alarms: int
    note: str


def _within(t: float, lo: float, hi: float, tol: float = TOL_SEC) -> bool:
    return lo - tol <= t <= hi + tol


def judge(exp: Expect | Absent, events: Iterable[Event]) -> Row:
    times = tuple(round(e.t, 1) for e in events if e.kind == exp.kind)
    if isinstance(exp, Absent):
        verdict = "ok" if not times else "false alarm"
        return Row(exp.video, exp.kind, "none", times, verdict, len(times), exp.note)
    inside = [t for t in times if _within(t, exp.lo, exp.hi)]
    expected = f"{exp.lo:g}" if exp.lo == exp.hi else f"{exp.lo:g}-{exp.hi:g}"
    return Row(exp.video, exp.kind, expected, times, "hit" if inside else "miss", len(times) - len(inside), exp.note)


def judge_all(source: EventSource, expectations: Sequence[Expect | Absent] = ROLE_EXPECTATIONS) -> list[Row]:
    cache: dict[str, Sequence[Event]] = {}
    rows = []
    for exp in expectations:
        if exp.video not in cache:
            cache[exp.video] = source(exp.video)
        rows.append(judge(exp, cache[exp.video]))
    return rows


@dataclass(frozen=True)
class RoomCount:
    video: str
    detected: int
    matched: int
    spurious: int
    missed: int


def count_room_events(video: str, events: Iterable[Event]) -> RoomCount:
    """Greedy one-to-one matching of enter/exit events with the door crossings of the video."""
    truth = list(DOOR_CROSSINGS.get(video, ()))
    detected = sorted((e for e in events if e.kind in ROOM_KINDS), key=lambda e: e.t)
    used: set[int] = set()
    matched = 0
    for e in detected:
        hit = next((i for i, (k, lo, hi) in enumerate(truth) if i not in used and k == e.kind
                    and _within(e.t, lo, hi)), None)
        if hit is not None:
            used.add(hit)
            matched += 1
    return RoomCount(video, len(detected), matched, len(detected) - matched, len(truth) - len(used))


def cached_source(fps: float, build: bool) -> EventSource:
    every = max(1, round(CACHE_FPS / fps))
    videos = {normalize_video_name(p.name): p for p in det_cache.list_videos()}

    def factory():
        from ml.detect import Detector

        return Detector()

    @functools.cache
    def source(name: str) -> Sequence[Event]:
        cached = det_cache.load_or_build(videos[name], CACHE_FPS, factory if build else None)
        frames = [
            FrameObjects(f.t, f.detections, tuple(color_signature(c) if c is not None else () for c in f.crops))
            for f in cached.frames[::every]
        ]
        return analyze_frames(frames, cached.info.duration, CACHE_FPS / every).events

    return source


def _fmt_times(times: Sequence[float]) -> str:
    return ", ".join(f"{t:g}" for t in times) or "-"


def print_role_table(rows: Sequence[Row]) -> None:
    print("| video | event | expected | detected | verdict | note |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        extra = f" (+{r.false_alarms} extra)" if r.verdict == "hit" and r.false_alarms else ""
        print(f"| {r.video} | {r.kind} | {r.expected} | {_fmt_times(r.detected)} | {r.verdict}{extra} | {r.note} |")
    hits = sum(r.verdict in ("hit", "ok") for r in rows)
    print(f"\n{hits}/{len(rows)} expectations met (tolerance ±{TOL_SEC:g} s)")


def print_room_table(counts: Sequence[RoomCount]) -> None:
    print("| video | enter/exit events | matched | spurious | missed |")
    print("|---|---|---|---|---|")
    for c in counts:
        print(f"| {c.video} | {c.detected} | {c.matched} | {c.spurious} | {c.missed} |")
    total = {k: sum(getattr(c, k) for c in counts) for k in ("detected", "matched", "spurious", "missed")}
    print(f"| total | {total['detected']} | {total['matched']} | {total['spurious']} | {total['missed']} |")


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=float, default=CACHE_FPS, help="analysis rate; the 6 fps cache is subsampled")
    ap.add_argument("--only", default="")
    ap.add_argument("--build", action="store_true", help="run the detector for videos without a cache")
    args = ap.parse_args()
    wanted = {v.strip().upper() for v in args.only.split(",") if v.strip()}
    source = cached_source(args.fps, args.build)

    expectations = [e for e in ROLE_EXPECTATIONS if not wanted or e.video in wanted]
    print_role_table(judge_all(source, expectations))
    print()
    videos = [v for v in DOOR_CROSSINGS if not wanted or v in wanted]
    print_room_table([count_room_events(v, source(v)) for v in sorted(videos, key=lambda v: int(v.split("-")[1]))])


if __name__ == "__main__":
    main()
