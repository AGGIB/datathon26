"""Zero-shot open-vocabulary detection with OWLv2 (google/owlv2-base-patch16-ensemble, Apache-2.0).

Used only on a handful of keyframes per video (layout on the empty-room background, desk crops when the
cashier leaves), never per frame. Results can be cached on disk because the same keyframes are re-scored
while tuning.
"""

import hashlib
import json
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from .detect import Box
from .scene_geometry import box_area, clip_box, intersection, to_int_box

MODEL_ID = "google/owlv2-base-patch16-ensemble"
DEFAULT_THRESHOLD = 0.08
NMS_IOU = 0.5
MAX_HITS_PER_QUERY = 8

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class OwlHit:
    query: str
    box: Box  # source-frame pixels
    score: float


def _iou(a: Box, b: Box) -> float:
    inter = intersection(a, b)
    if inter is None:
        return 0.0
    i = box_area(inter)
    return i / (box_area(a) + box_area(b) - i)


def nms_per_query(hits: Sequence[OwlHit], iou_thr: float = NMS_IOU, max_per_query: int = MAX_HITS_PER_QUERY) -> list[OwlHit]:
    kept: list[OwlHit] = []
    for h in sorted(hits, key=lambda h: h.score, reverse=True):
        same = [k for k in kept if k.query == h.query]
        if len(same) < max_per_query and all(_iou(h.box, k.box) < iou_thr for k in same):
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


class OwlDetector:
    """Lazy OWLv2 wrapper; `detect` accepts a BGR frame and an optional region to zoom into."""

    def __init__(self, model_id: str = MODEL_ID, device: str | None = None, cache_dir: Path | None = None) -> None:
        self.model_id = model_id
        self._device = device
        self._model = None
        self._processor = None
        self.cache_dir = cache_dir
        self.calls = 0

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        from .detect import _pick_device

        self._torch = torch
        self._device = self._device or _pick_device()
        self._processor = Owlv2Processor.from_pretrained(self.model_id)
        self._model = Owlv2ForObjectDetection.from_pretrained(self.model_id).to(self._device).eval()

    def detect(
        self,
        frame_bgr: np.ndarray,
        queries: Sequence[str],
        region: Box | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        cache_key: str | None = None,
    ) -> list[OwlHit]:
        h, w = frame_bgr.shape[:2]
        x0, y0, x1, y1 = to_int_box(clip_box(region, w, h)) if region else (0, 0, w, h)
        if cache_key is not None:  # the key names the call; the content hash keeps stale hits out
            cache_key = f"{cache_key}_{hashlib.sha1(frame_bgr[y0:y1, x0:x1].tobytes()).hexdigest()[:12]}"
        cached = self._cache_get(cache_key, queries, (x0, y0, x1, y1), threshold)
        if cached is not None:
            return cached
        self._load()
        crop = cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
        inputs = self._processor(text=[list(queries)], images=crop, return_tensors="pt").to(self._device)
        with self._torch.no_grad():
            out = self._model(**inputs)
        self.calls += 1
        probs = out.logits[0].sigmoid().cpu().numpy()  # (P, Q)
        boxes = out.pred_boxes[0].cpu().numpy()  # (P, 4)
        hits = decode_hits(probs, boxes, queries, float(max(x1 - x0, y1 - y0)), (x0, y0), threshold)
        self._cache_put(cache_key, queries, (x0, y0, x1, y1), threshold, hits)
        return hits

    # --- tiny JSON cache: one file per (key, queries, region, threshold) -------------------------------
    def _cache_file(self, key: str, queries: Sequence[str], region: tuple, threshold: float) -> Path | None:
        if self.cache_dir is None or key is None:
            return None
        digest = hashlib.sha1(json.dumps([self.model_id, list(queries), region, threshold]).encode()).hexdigest()
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
        file.write_text(json.dumps([asdict(h) for h in hits]))
