"""Stage 2: rules turn perception (+ scene state) into violation candidates. Owner: Агыбай.

Contract used by ml/pipeline.py:
    find_candidates(perception, scene=None) -> list[Candidate]

The rule logic runs on a small neutral input (RuleInput: room events + scene samples over time), so it does not
depend on how perception/scene are implemented; `to_rule_input` is the only adapter. Thresholds are constants to be
calibrated on the dataset pairs (violation vs correct variant).

Desk items come in two densities: COCO items on every sample (dense, noisy) and OWLv2 items only on a few scene
keyframes (sparse, chosen after exits / while the room is empty). One keyframe sighting counts; dense items need a
share of the window. Observations are Russian, only what is visible; legal wording comes from kb/violations.yaml.
"""

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from .candidates import Candidate
from .perception import Perception

HIGH = 0.6  # a 0..1 state score at or above this counts as "on" (open door, lit screen, ...)
LOW = 0.3  # ... and below this as clearly "off"; open/key states only count after an observed "off" period
MIN_SHARE = 0.5  # share of dense samples in a window that must show the condition
MIN_SAMPLES = 2
LEAVE_LOOK_SEC = 6.0  # how long after the room becomes empty we look at what was left behind
SAFE_OPEN_MIN_SEC = 7.0  # open-dataset duration sweep: removes one extra safe alarm without losing matched events
DOOR_OPEN_MIN_SEC = 5.0
RUN_MAX_MISSES = 1  # one missed sample does not break an "open" run
PERSONAL_MIN_SEC = 4.0
FOOD_MIN_SEC = 3.0
BASELINE_SEC = 3.0  # items already there in the first seconds are furniture/equipment, not "new" items
VISITOR_CHANGE_MAX_GAP_SEC = 30.0
HANDOVER_AFTER_ARRIVAL_SEC = 2.0
VISITOR_MARGIN_SEC = 2.0
PEAK_AFTER_START_SEC = 2.0
DOOR_ENTRY_OVERLAP_SEC = 3.0

CONF = {
    "cash_left_on_desk_when_leaving": 0.75,
    "stamps_and_book_left_on_desk": 0.65,
    "computer_left_unlocked": 0.6,
    "key_left_in_safe_lock": 0.45,  # experimental signal: below the output threshold until validated
    "safe_left_open": 0.7,
    "door_left_open": 0.65,
    "person_entered_cash_area": 0.75,
    "second_client_served_alongside": 0.6,
    "previous_cash_left_on_desk": 0.6,
    "personal_items_on_desk": 0.65,
    "food_or_drink_at_workplace": 0.55,
}

ROLE_KINDS = frozenset({"cashier_left", "cashier_returned", "visitor_enter", "visitor_exit", "visitors_overlap",
                        "person_behind_cashier"})
LEFT_PREFIX = {True: "кассир выходит из кассового помещения, ", False: "в помещении никого не остаётся, "}
COCO_TO_ITEM = {"handbag": "bag", "backpack": "bag", "suitcase": "bag", "cup": "cup", "banana": "food",
                "book": "book", "cell phone": "phone"}  # same mapping as ml/scene_items.py (dedupe after merge)


@dataclass(frozen=True)
class Sample:
    """Scene snapshot. Scores are 0..1 or None when unknown (None never satisfies a condition)."""

    t: float
    safe_open: float | None
    door_open: float | None
    screen_work: float | None
    key_in_lock: float | None
    items: frozenset[str]  # desk item labels: cash_bundle | book | stamp | bag | clothes | cup | food | phone
    people: int
    keyframe: bool = False  # items here come from the zoomed OWLv2 pass (sparse, reliable)


@dataclass(frozen=True)
class RuleInput:
    duration: float
    events: tuple[Any, ...]  # anything with .kind, .t, .t_end (ml.events.Event)
    samples: tuple[Sample, ...]


Cond = Callable[[Sample], bool]


def _on(value: float | None) -> bool:
    return value is not None and value >= HIGH


def _has(*labels: str) -> Cond:
    return lambda s: any(label in s.items for label in labels)


def _window(samples: Sequence[Sample], t0: float, t1: float) -> list[Sample]:
    return [s for s in samples if t0 <= s.t <= t1]


def _state_hit(window: Sequence[Sample], cond: Cond) -> Sample | None:
    hits = [s for s in window if cond(s)]
    if len(hits) < MIN_SAMPLES or len(hits) < MIN_SHARE * len(window):
        return None
    return hits[0]


def _item_hits(window: Sequence[Sample], labels: Sequence[str]) -> list[Sample]:
    """Samples that show any of the labels: one keyframe sighting is enough, dense items need a share."""
    cond = _has(*labels)
    keyframe_hits = [s for s in window if s.keyframe and cond(s)]
    if keyframe_hits:
        return keyframe_hits
    dense = [s for s in window if not s.keyframe]
    hits = [s for s in dense if cond(s)]
    return hits if len(hits) >= MIN_SAMPLES and len(hits) >= MIN_SHARE * len(dense) else []


def _runs(samples: Sequence[Sample], cond: Cond, min_sec: float) -> list[tuple[float, float]]:
    """Intervals where cond holds, tolerating RUN_MAX_MISSES isolated misses, at least min_sec long."""
    out, start, last, misses = [], None, None, 0
    for s in samples:
        if cond(s):
            start = s.t if start is None else start
            last, misses = s.t, 0
        elif start is not None:
            misses += 1
            if misses > RUN_MAX_MISSES:
                out.append((start, last))
                start, misses = None, 0
    if start is not None:
        out.append((start, last))
    return [(a, b) for a, b in out if b - a >= min_sec]


def _cand(type_id: str, t0: float, t1: float, peak: float, observation: str, extra: Iterable[float],
          duration: float) -> Candidate:
    def clamp(t: float) -> float:
        return min(max(t, 0.0), duration)

    peak = clamp(peak)
    t0, t1 = min(clamp(t0), peak), max(clamp(t1), peak)
    times = (peak, *sorted({clamp(t) for t in extra} - {peak}))[:3]
    return Candidate(type_id, t0, t1, peak, CONF[type_id], observation, times)


def _was_off_before(samples: Sequence[Sample], attr: str, t: float) -> bool:
    """Scene states are measured against the clip start: "on" from the first frame is an artifact, not an event."""
    off = [s for s in samples if s.t < t and getattr(s, attr) is not None and getattr(s, attr) < LOW]
    return len(off) >= MIN_SAMPLES


def _roles_known(events: Sequence[Any]) -> bool:
    return any(e.kind in ROLE_KINDS for e in events)


# --- left behind when the room becomes empty ---------------------------------------------------------

def _leaving_events(events: Sequence[Any]) -> list[tuple[Any, bool]]:
    """(event, is_cashier_left). room_empty is a fallback only where no cashier_left covers it."""
    left = [e for e in events if e.kind == "cashier_left"]
    spans = [(e.t, e.t_end if e.t_end is not None else math.inf) for e in left]
    empties = [e for e in events if e.kind == "room_empty" and not any(a <= e.t <= b for a, b in spans)]
    return [(e, True) for e in left] + [(e, False) for e in empties]


def _left_items_text(found: set[str]) -> str:
    if found == {"book"}:
        return "на столе остаётся книга"
    if found == {"stamp"}:
        return "на столе остаётся предмет, похожий на штамп"
    return "на столе остаются предметы, похожие на штамп и книгу"


def _empty_window(inp: RuleInput, ev: Any) -> list[Sample]:
    end = ev.t_end if ev.t_end is not None else inp.duration
    empty = [s for s in _window(inp.samples, ev.t, end) if s.people == 0]
    return [s for s in empty if s.t <= empty[0].t + LEAVE_LOOK_SEC] if empty else []


def _left_behind_findings(window: list[Sample], history: Sequence[Sample], t_leave: float
                          ) -> list[tuple[str, Sample, str]]:
    found: list[tuple[str, Sample, str]] = []
    if hits := _item_hits(window, ["cash_bundle"]):
        found.append(("cash_left_on_desk_when_leaving", hits[0], "на столе остаётся пачка, похожая на банкноты"))
    if hits := _item_hits(window, ["stamp", "book"]):
        labels = {i for s in hits for i in s.items if i in ("stamp", "book")}
        found.append(("stamps_and_book_left_on_desk", hits[0], _left_items_text(labels)))
    states = [
        ("computer_left_unlocked", None, lambda s: _on(s.screen_work),
         "экран ноутбука остаётся включённым с рабочим содержимым"),
        ("key_left_in_safe_lock", "key_in_lock", lambda s: _on(s.key_in_lock),
         "в замке сейфа остаётся предмет, похожий на ключ"),
        ("safe_left_open", "safe_open", lambda s: _on(s.safe_open), "дверца сейфа остаётся открытой"),
        ("door_left_open", "door_open", lambda s: _on(s.door_open), "дверь кассового помещения остаётся открытой"),
    ]
    for type_id, attr, cond, text in states:
        if attr is not None and not _was_off_before(history, attr, t_leave):
            continue
        if hit := _state_hit(window, cond):
            found.append((type_id, hit, text))
    return found


def rule_left_behind(inp: RuleInput) -> list[Candidate]:
    out = []
    for ev, by_cashier in _leaving_events(inp.events):
        window = _empty_window(inp, ev)
        if not window:
            continue  # nobody left the room (e.g. tidying up next to the desk)
        for type_id, hit, text in _left_behind_findings(window, inp.samples, ev.t):
            out.append(_cand(type_id, ev.t, window[-1].t, hit.t, LEFT_PREFIX[by_cashier] + text, [ev.t],
                             inp.duration))
    return out


# --- states that last -------------------------------------------------------------------------------------

def rule_open_too_long(inp: RuleInput) -> list[Candidate]:
    specs = [
        ("safe_left_open", "safe_open", lambda s: _on(s.safe_open), SAFE_OPEN_MIN_SEC,
         "дверца сейфа остаётся открытой около {d} с"),
        ("door_left_open", "door_open", lambda s: _on(s.door_open), DOOR_OPEN_MIN_SEC,
         "дверь кассового помещения остаётся открытой около {d} с"),
    ]
    out = []
    for type_id, attr, cond, min_sec, text in specs:
        for a, b in _runs(inp.samples, cond, min_sec):
            if not _was_off_before(inp.samples, attr, a):
                continue
            peak = min(a + PEAK_AFTER_START_SEC, b)
            out.append(_cand(type_id, a, b, peak, text.format(d=round(b - a)), [b], inp.duration))
    return out


# --- people -------------------------------------------------------------------------------------------------

def rule_people(inp: RuleInput) -> list[Candidate]:
    out = []
    for ev in inp.events:
        end = ev.t_end if ev.t_end is not None else ev.t
        if ev.kind == "person_behind_cashier":
            text = "в кассовое помещение проходит другой человек и оказывается на стороне кассира"
            out.append(_cand("person_entered_cash_area", ev.t, end, ev.t, text, [end], inp.duration))
        elif ev.kind == "visitors_overlap":
            text = "у стола одновременно находятся два посетителя"
            out.append(_cand("second_client_served_alongside", ev.t, end, (ev.t + end) / 2, text, [ev.t],
                             inp.duration))
    return out


def rule_previous_cash(inp: RuleInput) -> list[Candidate]:
    exits = sorted((e for e in inp.events if e.kind == "visitor_exit"), key=lambda e: e.t)
    enters = sorted((e for e in inp.events if e.kind == "visitor_enter"), key=lambda e: e.t)
    out = []
    for ex in exits:
        nxt = next((en for en in enters if ex.t < en.t <= ex.t + VISITOR_CHANGE_MAX_GAP_SEC), None)
        if nxt is None:
            continue
        handover = nxt.t + HANDOVER_AFTER_ARRIVAL_SEC
        if _item_hits(_window(inp.samples, ex.t, handover), ["cash_bundle"]):
            text = ("к столу подходит следующий посетитель, а пачка, похожая на банкноты, "
                    "от предыдущей операции остаётся на столе")
            out.append(_cand("previous_cash_left_on_desk", ex.t, handover, handover, text, [ex.t], inp.duration))
    return out


# --- things on the desk -----------------------------------------------------------------------------------

def _baseline(samples: Sequence[Sample]) -> frozenset[str]:
    first = [s for s in samples if s.t <= BASELINE_SEC]
    counts: dict[str, int] = {}
    for s in first:
        for label in s.items:
            counts[label] = counts.get(label, 0) + 1
    return frozenset(label for label, n in counts.items() if n >= max(1, MIN_SHARE * len(first)))


def _visitor_spans(events: Sequence[Any], duration: float) -> list[tuple[float, float]]:
    spans, open_at = [], []
    for e in sorted(events, key=lambda e: e.t):
        if e.kind == "visitor_enter":
            open_at.append(e.t)
        elif e.kind == "visitor_exit":
            spans.append((open_at.pop(0) if open_at else 0.0, e.t))
    spans.extend((t, duration) for t in open_at)
    return [(a - VISITOR_MARGIN_SEC, b + VISITOR_MARGIN_SEC) for a, b in spans]


def _first_new(samples: Sequence[Sample], label: str, min_sec: float, baseline: frozenset[str]) -> float | None:
    if label in baseline:
        return None
    if keyframe := next((s for s in samples if s.keyframe and label in s.items), None):
        return keyframe.t
    runs = _runs([s for s in samples if not s.keyframe], _has(label), min_sec)
    return runs[0][0] if runs else None


def rule_desk_items(inp: RuleInput) -> list[Candidate]:
    base = _baseline(inp.samples)
    out = []
    personal: dict[str, float] = {}
    if (t := _first_new(inp.samples, "clothes", PERSONAL_MIN_SEC, base)) is not None:
        personal["clothes"] = t
    t_bag = _first_new(inp.samples, "bag", PERSONAL_MIN_SEC, base)
    if t_bag is not None and _roles_known(inp.events):
        if not any(a <= t_bag <= b for a, b in _visitor_spans(inp.events, inp.duration)):
            personal["bag"] = t_bag  # brought by the cashier, not a visitor's / collector's bag
    if personal:
        words = [w for k, w in (("bag", "сумка"), ("clothes", "верхняя одежда")) if k in personal]
        verb = "лежат" if len(words) > 1 else "лежит"
        t0 = min(personal.values())
        out.append(_cand("personal_items_on_desk", t0, t0 + PEAK_AFTER_START_SEC, t0 + PEAK_AFTER_START_SEC,
                         f"на рабочем столе кассира {verb} {' и '.join(words)}", [t0], inp.duration))
    if (t := _first_new(inp.samples, "food", FOOD_MIN_SEC, base)) is not None:
        cup = any("cup" in s.items for s in _window(inp.samples, t, t + FOOD_MIN_SEC + 2.0))
        text = "на рабочем столе находятся предметы, похожие на продукты питания" + (", и кружка" if cup else "")
        out.append(_cand("food_or_drink_at_workplace", t, t + PEAK_AFTER_START_SEC, t + PEAK_AFTER_START_SEC, text,
                         [t], inp.duration))
    return out


# --- assembly of rule outputs ----------------------------------------------------------------------------

RULES = (rule_left_behind, rule_open_too_long, rule_people, rule_previous_cash, rule_desk_items)


def _drop_door_when_someone_entered(cands: list[Candidate]) -> list[Candidate]:
    """An open door through which a person walks in is reported as the entry (Т13), not separately (Т06)."""
    entries = [c for c in cands if c.type_id == "person_entered_cash_area"]

    def covered(c: Candidate) -> bool:
        return any(c.t_start - DOOR_ENTRY_OVERLAP_SEC <= e.t_peak <= c.t_end + DOOR_ENTRY_OVERLAP_SEC for e in entries)

    return [c for c in cands if not (c.type_id == "door_left_open" and covered(c))]


def run_rules(inp: RuleInput) -> list[Candidate]:
    cands = _drop_door_when_someone_entered([c for rule in RULES for c in rule(inp)])
    return sorted(cands, key=lambda c: (c.t_peak, c.type_id))


# --- adapter ---------------------------------------------------------------------------------------------

def _people_at(perception: Perception, t: float, fallback: int) -> int:
    """In-room people from the cleaned tracks (no reflections / corridor) when perception has them."""
    if not perception.tracks:
        return fallback
    return sum(tr.alive_at(t) for tr in perception.tracks)


def _samples_from_scene(perception: Perception, scene: Any) -> list[Sample]:
    out = []
    for st in scene.states:
        items = frozenset(it.label for it in st.desk_items)  # the scene already filters items by novelty
        people = _people_at(perception, st.t, getattr(st, "people", 0))
        out.append(Sample(st.t, st.safe_open, st.door_open, st.laptop_screen_work, st.key_in_lock, items, people,
                          bool(getattr(st, "keyframe", False))))
    return out


def _samples_from_perception(perception: Perception) -> list[Sample]:
    out = []
    for f in perception.frames:
        items = frozenset(COCO_TO_ITEM[d.label] for d in f.detections if d.label in COCO_TO_ITEM)
        people = _people_at(perception, f.t, sum(d.label == "person" for d in f.detections))
        out.append(Sample(f.t, None, None, None, None, items, people))
    return out


def to_rule_input(perception: Perception, scene: Any | None = None) -> RuleInput:
    samples = _samples_from_scene(perception, scene) if scene is not None else _samples_from_perception(perception)
    return RuleInput(duration=perception.info.duration, events=tuple(perception.events),
                     samples=tuple(sorted(samples, key=lambda s: s.t)))


def find_candidates(perception: Perception, scene: Any | None = None) -> list[Candidate]:
    return run_rules(to_rule_input(perception, scene))

