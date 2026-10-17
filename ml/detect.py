"""COCO object detection with RT-DETRv2-R18 (Apache-2.0): people plus objects that matter for cash desks."""

from dataclasses import dataclass

import cv2
import numpy as np

MODEL_ID = "PekingU/rtdetr_v2_r18vd"
MODEL_REVISION = "5650961749fa93567c0d46fc7f43ea4f9e914107"  # pinned: reproducible + offline-cacheable
INPUT_SIZE = 640

# COCO classes used as signals (names as in the model config; it mixes VOC-style spellings).
LABELS_OF_INTEREST = frozenset({
    "person", "handbag", "backpack", "suitcase", "cup", "banana", "laptop",
    "cell phone", "book", "tvmonitor", "tv",
})
DEFAULT_THRESHOLDS = {"person": 0.5}
OBJECT_THRESHOLD = 0.4
NMS_IOU = 0.6

Box = tuple[float, float, float, float]  # x0, y0, x1, y1 in source-frame pixels


@dataclass(frozen=True)
class Detection:
    box: Box
    score: float
    label: str


def _pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _nms(dets: list[Detection], iou_thr: float) -> list[Detection]:
    from .track import iou  # local import keeps track.py free of torch

    kept: list[Detection] = []
    for d in sorted(dets, key=lambda d: d.score, reverse=True):
        if all(d.label != k.label or iou(d.box, k.box) < iou_thr for k in kept):
            kept.append(d)
    return kept


class Detector:
    def __init__(self, model_id: str = MODEL_ID, device: str | None = None, input_size: int = INPUT_SIZE) -> None:
        import torch
        from transformers import AutoModelForObjectDetection

        self._torch = torch
        self.device = device or _pick_device()
        self.input_size = input_size
        self.model = (
            AutoModelForObjectDetection.from_pretrained(model_id, revision=MODEL_REVISION, use_safetensors=True)
            .to(self.device)
            .eval()
        )
        self._id2label = dict(self.model.config.id2label)

    def _prep(self, frames: list[np.ndarray]):
        size = (self.input_size, self.input_size)
        batch = np.stack([cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), size) for f in frames])
        tensor = self._torch.from_numpy(batch).permute(0, 3, 1, 2).float().div_(255.0)
        return tensor.contiguous().to(self.device)

    def detect(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        if not frames:
            return []
        with self._torch.no_grad():
            out = self.model(pixel_values=self._prep(frames))
        probs = out.logits.sigmoid().cpu().numpy()  # (B, Q, all classes)
        boxes = out.pred_boxes.cpu().numpy()  # (B, Q, 4) cx, cy, w, h normalized
        return [decode_queries(probs[i], boxes[i], frames[i].shape, self._id2label) for i in range(len(frames))]


def decode_queries(probs: np.ndarray, boxes: np.ndarray, shape: tuple, id2label: dict[int, str]) -> list[Detection]:
    """Pick each query's class over ALL classes first, then keep only labels of interest."""
    h, w = shape[:2]
    best = probs.argmax(axis=1)
    scores = probs[np.arange(len(best)), best]
    dets = []
    for q in np.flatnonzero(scores >= min(OBJECT_THRESHOLD, *DEFAULT_THRESHOLDS.values())):
        label = id2label.get(int(best[q]))
        score = float(scores[q])
        if label not in LABELS_OF_INTEREST or score < DEFAULT_THRESHOLDS.get(label, OBJECT_THRESHOLD):
            continue
        cx, cy, bw, bh = boxes[q]
        x0, x1 = np.clip([(cx - bw / 2) * w, (cx + bw / 2) * w], 0, w)
        y0, y1 = np.clip([(cy - bh / 2) * h, (cy + bh / 2) * h], 0, h)
        dets.append(Detection(box=(float(x0), float(y0), float(x1), float(y1)), score=score, label=label))
    return _nms(dets, NMS_IOU)
