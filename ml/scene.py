"""Stage 1b: scene state. Where the desk, safe, entrance door and laptop are, and what state they are in over time.

Usage by the rules stage (field meanings are documented on the dataclasses below):

    from ml.scene import analyze_scene
    from ml.scene_owl import OwlDetector

    scene = analyze_scene(video_path, perception, owl=OwlDetector())
    scene.layout.safe                    # box or None
    st = scene.state_at(12.0)
    st.safe_open, st.door_open           # 0..1 or None
    st.laptop_screen_on, st.laptop_screen_work, st.key_in_lock
    scene.items_near(12.0)               # DeskItem tuple from the closest keyframe

How it works (no hard-coded pixel coordinates; everything is located per video):
* frames are sampled at SAMPLE_FPS; person boxes from perception mask people out;
* the start background (each pixel from the first frame where no person hides it) is the room as it was when
  the clip began. OWLv2 finds desk, safe (+ lock), glass door with handle and laptop screen on it; the COCO
  laptop box from perception takes precedence over OWLv2 for the laptop;
* safe_open / door_open = share of the safe / door region that differs from that region's reference, measured
  only while people hide < 20 % of it and the whole frame did not change (lights). Reference per region
  (SceneAnalysis.references): "start" = the first clean view if it comes within TRUST_START_SEC (state before
  anyone touched it), else "median" = the state the region is in most of the clip. The safe counts only
  structural change (edges, local gain), so shadows of a seated cashier do not open it.
  KNOWN LIMITATION: a door or safe that is already open when the clip starts becomes the reference, and the
  signal is inverted (closing it reads as "open"). Rules must not treat "0 all clip" as a confident "closed".
* laptop_screen_on / _work = brightness, saturation and texture of the inner screen area vs. its surroundings;
* key_in_lock (experimental) = darkening right under the lock cylinder; None unless the lock was found with
  confidence >= LOCK_MIN_CONF and the darkening changes on camera (key inserted / removed);
* desk items = COCO detections at every sample + OWLv2 on a zoomed desk crop at up to MAX_KEYFRAMES keyframes
  (one inside every cashier_left / room_empty span, after exits latest first, then start/middle/end), filtered
  by novelty against the start background; a mostly yellow "cash bundle" is relabelled food (banana).
Occluded samples hold the last measured value; before the first measurement a signal stays None (unknown).
"""

import logging
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .detect import Box, Detection
from .events import Event
from .perception import Perception
from .scene_geometry import clip_box, expand_box, overlap_fraction, to_int_box
from .scene_items import (
    ITEM_SCAN_QUERIES,
    DeskItem,
    desk_zone,
    items_from_coco,
    items_from_owl,
    keep_item,
    merge_items,
    relabel_by_color,
)
from .scene_layout import (
    LAYOUT_QUERIES,
    LOCK_QUERIES,
    SCREEN_QUERIES,
    pick_desk,
    pick_door,
    pick_laptop,
    pick_lock,
    pick_safe,
    pick_screen,
)
from .scene_owl import OwlDetector, OwlHit
from .scene_signals import (
    change_score,
    global_gain,
    hold_forward,
    initial_background,
    key_protrusion,
    masked_median,
    median_smooth,
    person_mask,
    ramp,
    region_reference,
    screen_features,
)
from .video import frames_at

SAMPLE_FPS = 2.0
MAX_BACKGROUND_FRAMES = 30
MAX_KEYFRAMES = 4
MAX_SAMPLES = 120  # long videos are sampled sparser to stay within the time budget
KEYFRAME_MIN_GAP = 1.5
PERSON_MATCH_SEC = 0.4
SMOOTH_K = 3

SAFE_ROI_GROW = (0.15, 0.05)  # a little room for the door swinging out
SAFE_CHANGE = (0.07, 0.17)  # share of structurally changed cells vs. the reference -> safe_open 0..1
DOOR_CHANGE = (0.10, 0.30)
MIN_ROI_VISIBLE = 0.8
TRUST_START_SEC = 2.0  # a region seen unhidden this early shows its state before anyone touched it
WHOLE_FRAME_CHANGE_MAX = 0.35  # lights switched off / exposure jump: no region comparison on that frame
ABSENCE_KINDS = ("cashier_left", "room_empty")  # perception events with t..t_end when the desk is unattended
ABSENCE_OFFSET = 1.5  # keyframe this long after an absence starts (inside it)
LAPTOP_CONTEXT_GROW = 0.6
SCREEN_CORE_SHRINK = (0.2, 0.15)  # skip bezel and the wall seen around a tilted screen
MAX_LAPTOP_PERSON_COVER = 0.35
KEY_SCALE = 0.05  # expected key size as share of the safe height
KEY_DARKENING = (8.0, 12.0)  # grey levels the area under the lock is darker than the door face
LOCK_MIN_CONF = 0.12  # key_in_lock only with a located lock ...
KEY_MIN_SWING = 6.0  # ... and when the darkening changes on camera (key inserted / removed)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Layout:
    """Per-video static layout; boxes are (x0, y0, x1, y1) in source pixels, None when not found."""

    width: int
    height: int
    desk: Box | None
    safe: Box | None
    door: Box | None
    laptop: Box | None
    lock: Box | None
    screen: Box | None = None  # laptop screen (inner part is measured)
    confidence: dict[str, float] = field(default_factory=dict)  # 0..1 per element


@dataclass(frozen=True)
class SceneState:
    """Scene at time t. Scores are 0..1 (higher = more likely); None = element not located."""

    t: float
    safe_open: float | None
    door_open: float | None
    laptop_screen_on: float | None  # screen emits light (any content, lock screen included)
    laptop_screen_work: float | None  # bright document-like content (spreadsheet), not a lock screen
    key_in_lock: float | None  # experimental: darkening under the lock cylinder, tuned on one angle
    desk_items: tuple[DeskItem, ...]  # COCO items at t; plus OWLv2 items when t is a keyframe
    keyframe: bool = False
    people: int = 0  # person detections at t (from perception)

    def has_item(self, label: str, min_score: float = 0.0) -> bool:
        return any(i.label == label and i.score >= min_score for i in self.desk_items)

    @property
    def item_labels(self) -> tuple[str, ...]:
        return tuple(sorted({i.label for i in self.desk_items}))


@dataclass(frozen=True)
class SceneAnalysis:
    layout: Layout
    states: tuple[SceneState, ...]
    keyframes: tuple[float, ...]
    elapsed_sec: float
    owl_calls: int = 0
    references: dict[str, str] = field(default_factory=dict)  # safe/door -> "start" | "median" | "none"

    def state_at(self, t: float) -> SceneState:
        return min(self.states, key=lambda s: abs(s.t - t))

    def items_near(self, t: float, max_dt: float = 3.0) -> tuple[DeskItem, ...]:
        """Desk items of the keyframe closest to t (within max_dt), else COCO items at t."""
        keyed = [s for s in self.states if s.keyframe and abs(s.t - t) <= max_dt]
        if keyed:
            return min(keyed, key=lambda s: abs(s.t - t)).desk_items
        return self.state_at(t).desk_items

    def series(self, name: str) -> list[tuple[float, float | None]]:
        return [(s.t, getattr(s, name)) for s in self.states]


# --------------------------------------------------------------------------------------------------- frames


def read_frames(video_path: Path | str, times: Sequence[float]) -> list[np.ndarray]:
    """Frames for the requested times (exact, one sequential pass via ml.video.frames_at)."""
    found = frames_at(video_path, list(times))
    if not found:
        raise ValueError(f"no frames decoded from {video_path}")
    missing = [t for t in times if t not in found]
    if missing:
        log.warning("%s: %d sample(s) not decodable, using the nearest decoded frame", video_path, len(missing))
    have = sorted(found)
    return [found[t] if t in found else found[min(have, key=lambda h: abs(h - t))] for t in times]


def detections_at(perception: Perception, t: float, max_dt: float = PERSON_MATCH_SEC) -> list[Detection]:
    """Detections of the perception frame nearest to t (empty if none within max_dt)."""
    if not perception.frames:
        return []
    nearest = min(perception.frames, key=lambda f: abs(f.t - t))
    return list(nearest.detections) if abs(nearest.t - t) <= max_dt else []


def persons_at(perception: Perception, t: float, max_dt: float = PERSON_MATCH_SEC) -> list[Detection]:
    return [d for d in detections_at(perception, t, max_dt) if d.label == "person"]


# --------------------------------------------------------------------------------------------------- layout


def _even_subset(n: int, k: int) -> list[int]:
    if n <= k:
        return list(range(n))
    return sorted({int(round(i)) for i in np.linspace(0, n - 1, k)})


def locate_layout(
    frames: Sequence[np.ndarray],
    detections: Sequence[Sequence[Detection]],
    owl: OwlDetector | None = None,
    cache_key: str | None = None,
) -> tuple[Layout, np.ndarray]:
    """Find the static layout from time-ordered sampled frames and their detections.

    Returns (layout, start_background): the room as it was at the start of the video with people removed.
    The start is used (not the median) because doors and the safe are closed before the cashier acts.
    """
    h, w = frames[0].shape[:2]
    masks = [person_mask((h, w), [d.box for d in dets if d.label == "person"]) for dets in detections]
    idx = _even_subset(len(frames), MAX_BACKGROUND_FRAMES)
    fallback = masked_median(np.stack([frames[i] for i in idx]), np.stack([masks[i] for i in idx]))
    background = initial_background(frames, masks, fallback)
    coco_laptops = [d.box for dets in detections for d in dets if d.label == "laptop"]
    laptop, c_laptop = pick_laptop(coco_laptops, [])
    hits, screen_hits = _layout_hits(background, laptop, owl, cache_key)
    if laptop is None:
        laptop, c_laptop = pick_laptop(coco_laptops, hits)
    desk, c_desk = pick_desk(hits, laptop, w, h)
    safe, c_safe = pick_safe(hits, desk, laptop, w, h)
    door, c_door = pick_door(hits, w, h)
    lock, c_lock = None, 0.0
    if owl is not None and safe is not None:
        region = expand_box(safe, 0.3, 0.1, width=w, height=h)
        lock_hits = owl.detect(background, LOCK_QUERIES, region=region,
                               cache_key=f"{cache_key}_lock" if cache_key else None)
        lock, c_lock = pick_lock(lock_hits, safe)
    screen, c_screen = pick_screen(screen_hits, laptop)
    conf = {"desk": c_desk, "safe": c_safe, "door": c_door, "laptop": c_laptop, "lock": c_lock, "screen": c_screen}
    layout = Layout(w, h, desk, safe, door, laptop, lock, screen, {k: round(v, 3) for k, v in conf.items()})
    log.info("layout %s", layout)
    return layout, background


def _layout_hits(
    background: np.ndarray, laptop: Box | None, owl: OwlDetector | None, cache_key: str | None
) -> tuple[list[OwlHit], list[OwlHit]]:
    """Whole-room layout queries and the zoomed laptop crop (screen) in one OWLv2 forward pass."""
    if owl is None:
        return [], []
    h, w = background.shape[:2]
    views: list[tuple[np.ndarray, Box | None]] = [(background, None)]
    if laptop is not None:
        views.append((background, expand_box(laptop, 0.4, width=w, height=h)))
    keys = [f"{cache_key}_layout", f"{cache_key}_screen"][: len(views)] if cache_key else None
    results = owl.detect_batch(views, LAYOUT_QUERIES + SCREEN_QUERIES, cache_keys=keys)
    hits = [x for x in results[0] if x.query in LAYOUT_QUERIES]
    screen_hits = results[1] if len(results) > 1 else []
    return hits, screen_hits


# --------------------------------------------------------------------------------------------------- signals


def _region_series(
    grays: Sequence[np.ndarray],
    visible: Sequence[np.ndarray],
    times: Sequence[float],
    roi: tuple[int, int, int, int] | None,
    gains: Sequence[float],
    skip: Sequence[bool],
    ignore: np.ndarray | None = None,
    structural: bool = False,
) -> tuple[list[float | None], str]:
    """Change of one region vs. its reference at every sample, and how the reference was chosen.

    None where people hide >= 20 % of the region (a half-hidden door looks unchanged) or the whole frame
    changed (lights). structural=True: local brightness gain and only edge/structure changes count (safe face).
    """
    if roi is None:
        return [None] * len(grays), "none"
    ref, how = region_reference(grays, visible, times, roi, MIN_ROI_VISIBLE, TRUST_START_SEC)
    if ref is None:
        return [None] * len(grays), how
    x0, y0, x1, y1 = roi
    out: list[float | None] = []
    for gray, vis, gain, skipped in zip(grays, visible, gains, skip):
        v = vis[y0:y1, x0:x1]
        if skipped or v.mean() < MIN_ROI_VISIBLE:
            out.append(None)
            continue
        valid = v if ignore is None else v & ~ignore[y0:y1, x0:x1]
        cur = gray[y0:y1, x0:x1]
        local = global_gain(cur, ref, valid) if structural else gain
        out.append(change_score(cur, ref, valid, gain=local, structural_only=structural))
    return out, how


def _int_roi(box: Box | None, w: int, h: int) -> tuple[int, int, int, int] | None:
    if box is None:
        return None
    roi = to_int_box(clip_box(box, w, h))
    return roi if roi[2] - roi[0] >= 8 and roi[3] - roi[1] >= 8 else None


def _screen(frame: np.ndarray, layout: Layout, persons: Sequence[Box]) -> tuple[float | None, float | None]:
    screen = layout.screen
    if screen is None or any(overlap_fraction(screen, p) > MAX_LAPTOP_PERSON_COVER for p in persons):
        return None, None
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = to_int_box(expand_box(screen, -SCREEN_CORE_SHRINK[0], -SCREEN_CORE_SHRINK[1], width=w, height=h))
    cx0, cy0, cx1, cy1 = to_int_box(expand_box(layout.laptop or screen, LAPTOP_CONTEXT_GROW, width=w, height=h))
    feats = screen_features(frame[y0:y1, x0:x1], frame[cy0:cy1, cx0:cx1])
    return feats.on, feats.work


def _key_raw(gray: np.ndarray, layout: Layout, persons: Sequence[Box]) -> float | None:
    if layout.lock is None or layout.safe is None:
        return None
    scale = KEY_SCALE * (layout.safe[3] - layout.safe[1])
    guard = expand_box(layout.lock, 3.0, width=layout.width, height=layout.height)
    if any(overlap_fraction(guard, p) > 0.0 for p in persons):
        return None
    return key_protrusion(gray, layout.lock, scale=round(scale))


def _novelty(gray: np.ndarray, ref: np.ndarray, valid: np.ndarray, box: Box, gain: float) -> float | None:
    x0, y0, x1, y1 = to_int_box(clip_box(box, gray.shape[1], gray.shape[0]))
    return change_score(gray[y0:y1, x0:x1], ref[y0:y1, x0:x1], valid[y0:y1, x0:x1], gain=gain, cell=4)


def _with_novelty(items: Sequence[DeskItem], frame, gray, ref, valid, gain) -> tuple[DeskItem, ...]:
    out = []
    for it in items:
        nov = _novelty(gray, ref, valid, it.box, gain)
        x0, y0, x1, y1 = to_int_box(clip_box(it.box, gray.shape[1], gray.shape[0]))
        label = relabel_by_color(it.label, cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV))
        cand = DeskItem(label, it.box, it.score, it.source, it.query, None if nov is None else round(nov, 3))
        if keep_item(cand):
            out.append(cand)
    return tuple(out)


# --------------------------------------------------------------------------------------------------- keyframes


def pick_keyframes(events: Sequence[Event], duration: float, max_n: int = MAX_KEYFRAMES) -> list[float]:
    """Moments worth an OWLv2 desk scan, by priority.

    1) one keyframe inside every absence (cashier_left / room_empty, t..t_end), ABSENCE_OFFSET after it starts;
    2) right after exits, latest first (what is left on the desk when the cashier goes);
    3) start / middle / end.
    """
    absences = sorted((e.t, e.t_end if e.t_end is not None else duration) for e in events if e.kind in ABSENCE_KINDS)
    inside = [min(t + ABSENCE_OFFSET, (t + t_end) / 2) for t, t_end in absences]
    exits = [e.t + 1.0 for e in sorted(events, key=lambda e: e.t, reverse=True) if e.kind == "exit"]
    fallback = [0.15 * duration, 0.5 * duration, 0.92 * duration]
    chosen: list[float] = []
    for t in inside + exits + fallback:
        t = min(max(t, 0.2), max(duration - 0.2, 0.2))
        if len(chosen) < max_n and all(abs(t - c) >= KEYFRAME_MIN_GAP for c in chosen):
            chosen.append(round(t, 2))
    return sorted(chosen)


# --------------------------------------------------------------------------------------------------- states


def _finish(raw: Sequence[float | None], lo_hi: tuple[float, float]) -> list[float | None]:
    smoothed = hold_forward(median_smooth(raw, SMOOTH_K))
    return [None if v is None else round(ramp(v, *lo_hi), 3) for v in smoothed]


def _finish_unit(raw: Sequence[float | None]) -> list[float | None]:
    smoothed = hold_forward(median_smooth(raw, SMOOTH_K))
    return [None if v is None else round(v, 3) for v in smoothed]


def key_trusted(raw: Sequence[float | None], lock_confidence: float) -> bool:
    """The key signal is reported only for a confidently located lock whose look changes on camera."""
    vals = [v for v in median_smooth(raw, SMOOTH_K) if v is not None]
    return lock_confidence >= LOCK_MIN_CONF and len(vals) >= 3 and max(vals) - min(vals) >= KEY_MIN_SWING


@dataclass
class _Raw:
    safe: list[float | None] = field(default_factory=list)
    door: list[float | None] = field(default_factory=list)
    screen_on: list[float | None] = field(default_factory=list)
    screen_work: list[float | None] = field(default_factory=list)
    key: list[float | None] = field(default_factory=list)
    items: list[tuple[DeskItem, ...]] = field(default_factory=list)
    people: list[int] = field(default_factory=list)
    references: dict[str, str] = field(default_factory=dict)


def _measure(
    times: Sequence[float],
    frames: Sequence[np.ndarray],
    perception: Perception,
    layout: Layout,
    keyframe_items: dict[float, list[DeskItem]],
    background: np.ndarray,
) -> _Raw:
    h, w = layout.height, layout.width
    dets_all = [detections_at(perception, t) for t in times]
    visible = [~person_mask((h, w), [d.box for d in dets if d.label == "person"]) for dets in dets_all]
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    ref = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    gains = [global_gain(g, ref, v) for g, v in zip(grays, visible)]
    whole = [change_score(g[::2, ::2], ref[::2, ::2], v[::2, ::2], gain=k) for g, v, k in zip(grays, visible, gains)]
    skip = [c is not None and c > WHOLE_FRAME_CHANGE_MAX for c in whole]
    laptop_mask = np.zeros((h, w), dtype=bool)
    if layout.laptop is not None:
        x0, y0, x1, y1 = to_int_box(expand_box(layout.laptop, 0.1, width=w, height=h))
        laptop_mask[y0:y1, x0:x1] = True
    safe_roi = _int_roi(expand_box(layout.safe, *SAFE_ROI_GROW, width=w, height=h) if layout.safe else None, w, h)
    raw = _Raw()
    raw.safe, safe_ref = _region_series(grays, visible, times, safe_roi, gains, skip, laptop_mask, structural=True)
    raw.door, door_ref = _region_series(grays, visible, times, _int_roi(layout.door, w, h), gains, skip)
    raw.references = {"safe": safe_ref, "door": door_ref}
    zone = desk_zone(layout.desk, w, h) if layout.desk else None
    for t, frame, gray, dets, valid, gain in zip(times, frames, grays, dets_all, visible, gains):
        persons = [d.box for d in dets if d.label == "person"]
        on, work = _screen(frame, layout, persons)
        raw.screen_on.append(on)
        raw.screen_work.append(work)
        raw.key.append(_key_raw(gray, layout, persons))
        items: list[DeskItem] = []
        if zone is not None:
            items = items_from_coco(dets, zone, persons) + keyframe_items.get(t, [])
        raw.items.append(_with_novelty(merge_items(items), frame, gray, ref, valid, gain))
        raw.people.append(len(persons))
    return raw


def _scan_keyframes(
    times: Sequence[float],
    frames: Sequence[np.ndarray],
    perception: Perception,
    layout: Layout,
    owl: OwlDetector | None,
    keyframes: Sequence[float],
    cache_key: str | None,
) -> dict[float, list[DeskItem]]:
    """OWLv2 desk scan at keyframes, keyed by the sampled time each keyframe snaps to."""
    if owl is None or layout.desk is None:
        return {}
    zone = desk_zone(layout.desk, layout.width, layout.height)
    region = expand_box(zone, 0.05, width=layout.width, height=layout.height)
    snapped = sorted({int(np.argmin([abs(t - kt) for t in times])) for kt in keyframes})
    views = [(frames[i], region) for i in snapped]
    keys = [f"{cache_key}_desk_{times[i]:.2f}" for i in snapped] if cache_key else None
    batches = owl.detect_batch(views, ITEM_SCAN_QUERIES, cache_keys=keys) if views else []
    out: dict[float, list[DeskItem]] = {}
    for i, hits in zip(snapped, batches):
        persons = [d.box for d in persons_at(perception, times[i])]
        out[times[i]] = items_from_owl(hits, zone, persons)
    return out


def _build_states(
    times: Sequence[float],
    frames: Sequence[np.ndarray],
    perception: Perception,
    layout: Layout,
    owl: OwlDetector | None,
    keyframes: Sequence[float],
    cache_key: str | None,
    background: np.ndarray,
) -> tuple[list[SceneState], tuple[float, ...], dict[str, str]]:
    keyframe_items = _scan_keyframes(times, frames, perception, layout, owl, keyframes, cache_key)
    raw = _measure(times, frames, perception, layout, keyframe_items, background)
    safe, door = _finish(raw.safe, SAFE_CHANGE), _finish(raw.door, DOOR_CHANGE)
    on, work = _finish_unit(raw.screen_on), _finish_unit(raw.screen_work)
    trusted = key_trusted(raw.key, layout.confidence.get("lock", 0.0))
    key = _finish(raw.key, KEY_DARKENING) if trusted else [None] * len(times)
    states = [
        SceneState(t=round(t, 3), safe_open=safe[i], door_open=door[i], laptop_screen_on=on[i],
                   laptop_screen_work=work[i], key_in_lock=key[i], desk_items=raw.items[i],
                   keyframe=t in keyframe_items, people=raw.people[i])
        for i, t in enumerate(times)
    ]
    return states, tuple(sorted(keyframe_items)), raw.references


def scene_states(
    video_path: Path | str,
    perception: Perception,
    times: Sequence[float],
    layout: Layout | None = None,
    owl: OwlDetector | None = None,
    keyframes: Sequence[float] | None = None,
    cache_key: str | None = None,
) -> list[SceneState]:
    """Scene state at each requested time (returned sorted by t). Locates the layout when not given.

    `times` should be reasonably dense (>= 1 per second): the start-of-video reference and smoothing
    are computed from the requested samples.
    """
    times = sorted(times)
    frames = read_frames(video_path, times)
    found, background = locate_layout(frames, [detections_at(perception, t) for t in times],
                                      owl if layout is None else None, cache_key)
    layout = layout or found
    if keyframes is None:
        keyframes = pick_keyframes(perception.events, perception.info.duration)
    states, _, _ = _build_states(times, frames, perception, layout, owl, keyframes, cache_key, background)
    return states


def sample_times(duration: float, fps: float = SAMPLE_FPS) -> list[float]:
    """Evenly spaced times in [0, duration), always at least one."""
    n = max(1, math.floor(duration * fps - 1e-9) + 1)
    return [round(i / fps, 3) for i in range(n)]


def analyze_scene(
    video_path: Path | str,
    perception: Perception,
    owl: OwlDetector | None = None,
    sample_fps: float = SAMPLE_FPS,
    cache_key: str | None = None,
) -> SceneAnalysis:
    """One-call entry for the pipeline: layout + states at `sample_fps` (at most MAX_SAMPLES) over the video."""
    started = time.monotonic()
    calls_before = owl.calls if owl else 0
    duration = perception.info.duration
    times = sample_times(duration, min(sample_fps, MAX_SAMPLES / max(duration, 1e-6)))
    frames = read_frames(video_path, times)
    layout, background = locate_layout(frames, [detections_at(perception, t) for t in times], owl, cache_key)
    keyframes = pick_keyframes(perception.events, perception.info.duration)
    states, used, refs = _build_states(times, frames, perception, layout, owl, keyframes, cache_key, background)
    calls = (owl.calls - calls_before) if owl else 0
    log.info("scene %s: layout confidence %s, references %s", video_path, layout.confidence, refs)
    return SceneAnalysis(layout, tuple(states), used, time.monotonic() - started, calls, refs)


__all__ = [
    "DeskItem", "Layout", "SceneAnalysis", "SceneState", "analyze_scene", "locate_layout", "pick_keyframes",
    "read_frames", "sample_times", "scene_states",
]
