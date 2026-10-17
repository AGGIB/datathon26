"""Zero-shot open-vocabulary detection with OWLv2 (google/owlv2-base-patch16-ensemble, Apache-2.0).

Used only on a handful of keyframes per video (layout on the empty-room background, desk crops when the
cashier leaves), never per frame. Results can be cached on disk because the same keyframes are re-scored
while tuning.
"""

import hashlib
import json
import logging
import threading
import time
from collections.abc import Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import numpy as np

from .detect import Box
from .scene_geometry import clip_box, to_int_box
from .track import iou

MODEL_ID = "google/owlv2-base-patch16-ensemble"
OWL_REVISION = "cfd3195ba4ea9592eec887ded089f4c08eff231d"  # pinned snapshot (Apache-2.0)
DEFAULT_THRESHOLD = 0.08
NMS_IOU = 0.5
MAX_HITS_PER_QUERY = 8
MAX_BATCH = 2  # views per forward pass; larger batches swap on a shared unified-memory Mac

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class OwlHit:
    query: str
    box: Box  # source-frame pixels
    score: float


def nms_per_query(hits: Sequence[OwlHit], iou_thr: float = NMS_IOU, max_per_query: int = MAX_HITS_PER_QUERY) -> list[OwlHit]:
    kept: list[OwlHit] = []
    for h in sorted(hits, key=lambda h: h.score, reverse=True):
        same = [k for k in kept if k.query == h.query]
        if len(same) < max_per_query and all(iou(h.box, k.box) < iou_thr for k in same):
            kept.append(h)
    return kept


def decode_hits(
    probs: np.ndarray, boxes: np.ndarray, queries: Sequence[str], side: float, offset: tuple[float, float],
    threshold: float,
) -> list[OwlHit]:
    """probs: (P, Q) sigmoid scores; boxes: (P, 4) cx, cy, w, h normalised to the padded square of `side` px."""
    hits = []
    for p, q in zip(*np.nonzero(probs >= threshold)):
        cx, cy, w, h = boxes[p] * side
        box = (cx - w / 2 + offset[0], cy - h / 2 + offset[1], cx + w / 2 + offset[0], cy + h / 2 + offset[1])
        hits.append(OwlHit(query=queries[q], box=tuple(float(v) for v in box), score=float(probs[p, q])))
    return nms_per_query(hits)


def check_deadline(deadline: float | None) -> None:
    """Raise when a caller's absolute monotonic deadline has expired."""
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("scene analysis deadline exceeded")


@dataclass
class OwlStats:
    """Forward passes attributed to one analysis, even when analyses share the detector."""

    calls: int = 0
    images: int = 0


class OwlDetector:
    """Lazy OWLv2 wrapper. `detect` / `detect_batch` take BGR frames and optional regions to zoom into.

    Thread-safe: the backend runs analyses in parallel threads and one MPS/CUDA model must not run two
    forwards at once, so loading and every forward pass happen under one lock. Use `shared_owl()` to get
    the single per-process instance.
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str | None = None,
        cache_dir: Path | None = None,
        revision: str | None = OWL_REVISION,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self._device = device
        self._torch = None
        self._model = None
        self._processor = None
        self._lock = threading.RLock()
        self.cache_dir = cache_dir
        self.calls = 0  # forward passes (updated under the lock)
        self.images = 0  # images scored
        self._stats: ContextVar[OwlStats | None] = ContextVar("owl_analysis_stats", default=None)

    @contextmanager
    def measure_calls(self):
        """Count only forwards in this thread/context, excluding concurrent video analyses."""
        stats = OwlStats()
        token = self._stats.set(stats)
        try:
            yield stats
        finally:
            self._stats.reset(token)

    @contextmanager
    def _locked(self, deadline: float | None):
        check_deadline(deadline)
        acquired = self._lock.acquire() if deadline is None else self._lock.acquire(
            timeout=max(0.0, deadline - time.monotonic())
        )
        if not acquired:
            raise TimeoutError("scene deadline exceeded waiting for OWLv2")
        try:
            check_deadline(deadline)
            yield
        finally:
            self._lock.release()

    def load(self, deadline: float | None = None) -> None:
        """Load the pinned local snapshot; no network requests. Safe to call from several threads.

        Populate the model cache during setup and prewarm outside timed jobs. A native model load already
        in progress cannot be interrupted, so `deadline` is cooperative and checked after each load step.
        """
        with self._locked(deadline):
            if self._model is not None:
                return
            import torch
            from transformers import Owlv2ForObjectDetection, Owlv2Processor

            from .detect import _pick_device

            self._torch = torch
            self._device = self._device or _pick_device()
            check_deadline(deadline)
            self._processor = Owlv2Processor.from_pretrained(self.model_id, revision=self.revision,
                                                                  local_files_only=True)
            check_deadline(deadline)
            model = Owlv2ForObjectDetection.from_pretrained(self.model_id, revision=self.revision,
                                                            use_safetensors=True, local_files_only=True)
            self._model = model.to(self._device).eval()
            check_deadline(deadline)

    def detect(
        self,
        frame_bgr: np.ndarray,
        queries: Sequence[str],
        region: Box | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        cache_key: str | None = None,
        deadline: float | None = None,
    ) -> list[OwlHit]:
        return self.detect_batch([(frame_bgr, region)], queries, threshold, [cache_key], deadline=deadline)[0]

    def detect_batch(
        self,
        views: Sequence[tuple[np.ndarray, Box | None]],
        queries: Sequence[str],
        threshold: float = DEFAULT_THRESHOLD,
        cache_keys: Sequence[str | None] | None = None,
        deadline: float | None = None,
    ) -> list[list[OwlHit]]:
        """Run several (frame, region) views with the same queries in one forward pass."""
        check_deadline(deadline)
        keys = list(cache_keys) if cache_keys is not None else [None] * len(views)
        if len(keys) != len(views):
            raise ValueError("cache_keys must match views")
        crops, todo, results = [], [], [None] * len(views)
        for i, ((frame, region), key) in enumerate(zip(views, keys)):
            check_deadline(deadline)
            h, w = frame.shape[:2]
            rect = to_int_box(clip_box(region, w, h)) if region else (0, 0, w, h)
            crop = frame[rect[1]:rect[3], rect[0]:rect[2]]
            if key is not None:  # the key names the call; the content hash keeps stale hits out
                key = f"{key}_{hashlib.sha1(crop.tobytes()).hexdigest()[:12]}"
            keys[i] = key
            results[i] = self._cache_get(key, queries, rect, threshold)
            if results[i] is None:
                todo.append((i, rect))
                crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        for start in range(0, len(todo), MAX_BATCH):  # 960x960 ViT attention: ~0.6 GB per image in flight
            chunk = todo[start:start + MAX_BATCH]
            hits_list = self._forward(crops[start:start + MAX_BATCH], queries, [r for _, r in chunk], threshold,
                                      deadline=deadline)
            for (i, rect), hits in zip(chunk, hits_list):
                results[i] = hits
                self._cache_put(keys[i], queries, rect, threshold, hits)
        check_deadline(deadline)
        return results

    def _forward(self, crops: list[np.ndarray], queries: Sequence[str], rects: list[tuple], threshold: float,
                 deadline: float | None = None):
        with self._locked(deadline):
            self.load(deadline=deadline)
            texts = [list(queries)] * len(crops)
            inputs = self._processor(text=texts, images=crops, return_tensors="pt").to(self._device)
            check_deadline(deadline)
            with self._torch.no_grad():
                out = self._model(**inputs)
            self.calls += 1
            self.images += len(crops)
            stats = self._stats.get()
            if stats is not None:
                stats.calls += 1
                stats.images += len(crops)
            probs = out.logits.sigmoid().cpu().numpy()  # (B, P, Q)
            boxes = out.pred_boxes.cpu().numpy()  # (B, P, 4)
            check_deadline(deadline)
        return [
            decode_hits(probs[j], boxes[j], queries, float(max(r[2] - r[0], r[3] - r[1])), (r[0], r[1]), threshold)
            for j, r in enumerate(rects)
        ]

    # --- tiny JSON cache: one file per (key, queries, region, threshold) -------------------------------
    def _cache_file(self, key: str, queries: Sequence[str], region: tuple, threshold: float) -> Path | None:
        if self.cache_dir is None or key is None:
            return None
        digest = hashlib.sha1(json.dumps([self.model_id, self.revision, list(queries), region, threshold]).encode()).hexdigest()
        return self.cache_dir / f"{key}_{digest[:16]}.json"

    def _cache_get(self, key, queries, region, threshold) -> list[OwlHit] | None:
        file = self._cache_file(key, queries, region, threshold)
        if file is None or not file.exists():
            return None
        try:
            return [OwlHit(query=d["query"], box=tuple(d["box"]), score=d["score"]) for d in json.loads(file.read_text())]
        except (ValueError, KeyError) as exc:
            log.warning("ignoring broken OWLv2 cache file %s: %s", file, exc)
            return None

    def _cache_put(self, key, queries, region, threshold, hits: Sequence[OwlHit]) -> None:
        file = self._cache_file(key, queries, region, threshold)
        if file is None:
            return
        file.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with NamedTemporaryFile(mode="w", encoding="utf-8", dir=file.parent, prefix=f".{file.name}.",
                                    suffix=".tmp", delete=False) as fh:
                temporary = Path(fh.name)
                fh.write(json.dumps([asdict(h) for h in hits]))
            temporary.replace(file)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


_SHARED: OwlDetector | None = None
_SHARED_LOCK = threading.Lock()


def shared_owl() -> OwlDetector:
    """The one OWLv2 instance of this process (created lazily, weights load on first use)."""
    global _SHARED
    with _SHARED_LOCK:
        if _SHARED is None:
            _SHARED = OwlDetector()
        return _SHARED
