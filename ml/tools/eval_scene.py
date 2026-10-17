"""Dev tool: validate scene states on the designed violation/correct pairs and print a table.

    python -m ml.tools.eval_scene                 # validation table
    python -m ml.tools.eval_scene --series X-20   # per-time signals of one or more videos
    python -m ml.tools.eval_scene --no-owl        # without the zero-shot model (COCO + statistics only)

Perception runs are cached in var/cache/perception, OWLv2 hits in var/cache/scene_owl.
Expected states come from kb/annotations.csv (time windows are the annotated intervals).
"""

import argparse
import logging
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from ml.scene import SceneAnalysis, SceneState, analyze_scene
from ml.scene_owl import OwlDetector
from ml.tools.scene_cache import cached_perception, video_paths

OWL_CACHE = Path("var/cache/scene_owl")
PASS_THRESHOLD = 0.5
ITEM_PRESENT = 0.01


@dataclass(frozen=True)
class Check:
    video: str
    what: str
    t0: float
    t1: float
    expected: bool  # True = state present (open / on / item on desk)
    measure: Callable[[SceneAnalysis, float, float], float | None]
    note: str = ""
    threshold: float = PASS_THRESHOLD  # items: any kept item counts (scores are raw detector confidences)


def _mean_of(field: str) -> Callable[[SceneAnalysis, float, float], float | None]:
    def measure(scene: SceneAnalysis, t0: float, t1: float) -> float | None:
        vals = [getattr(s, field) for s in scene.states if t0 <= s.t <= t1 and getattr(s, field) is not None]
        return round(mean(vals), 2) if vals else None

    return measure


def _items(labels: tuple[str, ...]) -> Callable[[SceneAnalysis, float, float], float | None]:
    """Best score of any of `labels` on the desk over keyframes (and COCO samples) inside the window."""

    def measure(scene: SceneAnalysis, t0: float, t1: float) -> float | None:
        states: list[SceneState] = [s for s in scene.states if t0 <= s.t <= t1]
        scores = [i.score for s in states for i in s.desk_items if i.label in labels]
        return round(max(scores), 2) if scores else 0.0

    return measure


ALL_ITEMS = ("cash_bundle", "book", "stamp", "bag", "clothes", "cup", "food")

CHECKS = (
    Check("X-20", "safe_open", 5, 11.5, True, _mean_of("safe_open"), "works at desk, safe wide open"),
    Check("X-21", "safe_open", 12, 19.5, False, _mean_of("safe_open"), "safe closed before counting"),
    Check("X-22", "key_in_lock", 14, 21, True, _mean_of("key_in_lock"), "key left in lock"),
    Check("X-23", "key_in_lock", 14, 18, False, _mean_of("key_in_lock"), "key removed"),
    Check("X-32", "door_open", 6, 21, True, _mean_of("door_open"), "works with door open"),
    Check("X-30", "door_open", 6, 18, True, _mean_of("door_open"), "door open, visitor enters"),
    Check("X-33", "door_open", 5, 14, True, _mean_of("door_open"), "door open, visitor enters"),
    Check("X-31", "door_open", 14, 23.5, False, _mean_of("door_open"), "door closed behind cashier"),
    Check("X-26", "cash_bundle", 10, 19.5, True, _items(("cash_bundle",)), "bundle left on desk", ITEM_PRESENT),
    Check("X-27", "cash_bundle", 20, 26, False, _items(("cash_bundle",)), "bundle moved to safe", ITEM_PRESENT),
    Check("X-24", "book/stamp", 6, 14.5, True, _items(("book", "stamp")), "book + stamps left", ITEM_PRESENT),
    Check("X-25", "book/stamp", 25, 31.5, False, _items(("book", "stamp")), "put away into cabinet", ITEM_PRESENT),
    Check("X-11", "screen_on", 18, 25.5, True, _mean_of("laptop_screen_on"), "lit laptop, room dark"),
    Check("X-11", "screen_work", 18, 25.5, True, _mean_of("laptop_screen_work"), "spreadsheet visible"),
    Check("X-12", "screen_on", 32, 42, False, _mean_of("laptop_screen_on"), "lid closed"),
    Check("X-12", "screen_work", 32, 42, False, _mean_of("laptop_screen_work"), "lid closed"),
    Check("X-28", "screen_work", 8, 16.5, True, _mean_of("laptop_screen_work"), "spreadsheet stays"),
    Check("X-29", "screen_work", 10, 17, False, _mean_of("laptop_screen_work"), "lock screen"),
    Check("X-40", "bag/clothes", 10, 25, True, _items(("bag", "clothes")), "coat and bag on desk", ITEM_PRESENT),
    Check("X-46", "food/cup", 6, 19, True, _items(("food", "cup")), "banana, mug", ITEM_PRESENT),
    Check("X-21", "bag/clothes", 0, 19.5, False, _items(("bag", "clothes")), "control: clean video", ITEM_PRESENT),
    Check("X-21", "food/cup", 0, 19.5, False, _items(("food", "cup")), "control: clean video", ITEM_PRESENT),
    Check("X-27", "bag/clothes", 0, 26, False, _items(("bag", "clothes")), "control: clean video", ITEM_PRESENT),
    Check("X-23", "any item", 0, 18.3, False, _items(ALL_ITEMS), "control: clean video", ITEM_PRESENT),
    Check("X-29", "any item", 0, 17.3, False, _items(ALL_ITEMS), "control: clean video", ITEM_PRESENT),
    Check("X-31", "any item", 0, 23.8, False, _items(ALL_ITEMS), "control: clean video", ITEM_PRESENT),
)


def _factory():
    from ml.detect import Detector

    return Detector()


def run_scene(name: str, path: Path, owl: OwlDetector | None) -> SceneAnalysis:
    perception = cached_perception(name, path, _factory)
    return analyze_scene(path, perception, owl=owl, cache_key=name)


def print_series(name: str, scene: SceneAnalysis) -> None:
    lay = scene.layout
    print(f"\n{name} layout: desk={lay.desk} safe={lay.safe} door={lay.door} laptop={lay.laptop} "
          f"lock={lay.lock} conf={lay.confidence}")
    print(f"   keyframes={scene.keyframes} took={scene.elapsed_sec:.1f}s owl_calls={scene.owl_calls}")
    print("       t  ppl  safe  door  on    work  key   items")
    for s in scene.states:
        items = ",".join(f"{i.label}:{i.score:.2f}/{i.source[0]}" for i in s.desk_items)
        mark = "*" if s.keyframe else " "
        print(f"   {s.t:5.1f}{mark} {s.people:2d}  {_f(s.safe_open)} {_f(s.door_open)} {_f(s.laptop_screen_on)} "
              f"{_f(s.laptop_screen_work)} {_f(s.key_in_lock)} {items}")


def _f(v: float | None) -> str:
    return "  -  " if v is None else f"{v:5.2f}"


def print_table(results: list[tuple[Check, float | None]], timings: dict[str, float]) -> None:
    print("\n| video | signal | window, s | expected | measured | ok | note |")
    print("|---|---|---|---|---|---|---|")
    correct = 0
    for c, value in results:
        ok = value is not None and (value >= c.threshold) == c.expected
        correct += ok
        exp = "yes" if c.expected else "no"
        shown = "n/a" if value is None else f"{value:.2f}"
        print(f"| {c.video} | {c.what} | {c.t0:g}–{c.t1:g} | {exp} | {shown} | {'✅' if ok else '❌'} | {c.note} |")
    print(f"\n{correct}/{len(results)} checks correct (states: mean score >= {PASS_THRESHOLD}; "
          "items: best score of a kept item, present if any)")
    if timings:
        print(f"scene time per video: mean {mean(timings.values()):.1f}s, max {max(timings.values()):.1f}s")


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default="", help="comma-separated videos to dump per-time signals for")
    ap.add_argument("--no-owl", action="store_true")
    ap.add_argument("--no-cache", action="store_true", help="re-run OWLv2 (honest timing)")
    args = ap.parse_args()
    paths = video_paths()
    owl = None if args.no_owl else OwlDetector(cache_dir=None if args.no_cache else OWL_CACHE)
    if owl is not None:
        started = time.monotonic()
        owl.load()  # one-off model load is not part of the per-video time
        print(f"OWLv2 loaded in {time.monotonic() - started:.1f}s")
    if args.series:
        for name in (v.strip().upper() for v in args.series.split(",")):
            print_series(name, run_scene(name, paths[name], owl))
        return
    scenes: dict[str, SceneAnalysis] = {}
    results = []
    for c in CHECKS:
        if c.video not in scenes:
            scenes[c.video] = run_scene(c.video, paths[c.video], owl)
        results.append((c, c.measure(scenes[c.video], c.t0, c.t1)))
    print_table(results, {k: v.elapsed_sec for k, v in scenes.items()})
    sys.stdout.flush()


if __name__ == "__main__":
    main()
