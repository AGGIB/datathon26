"""Mock pipeline with the real К2 signature. Lets the frontend and API work before ML is ready.

Upload a file whose name contains "clean" to get verdict=no_violation; anything else
returns two demo violations so the UI can render multiple cards and timeline markers.
"""

import base64
import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from .media import FFMPEG_SAFE_INPUT

ProgressFn = Callable[[float, str], None]

log = logging.getLogger(__name__)

PLACEHOLDER_JPEG = Path(__file__).parent / "fixtures" / "placeholder.jpg"
FFMPEG_TIMEOUT_SEC = 15
STAGES = ("decode", "detect", "rules", "verify")


def _resolve_ffmpeg() -> str:
    for candidate in ("ffmpeg", "/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg"):
        if shutil.which(candidate):
            return candidate
    return "ffmpeg"


def _frame_b64(video_path: str, t_sec: float) -> str:
    """Real frame from the uploaded video if ffmpeg can read it, placeholder otherwise."""
    cmd = [
        _resolve_ffmpeg(), "-v", "error", *FFMPEG_SAFE_INPUT, "-ss", f"{t_sec:.2f}", "-i", video_path,
        "-frames:v", "1", "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "4", "-",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=FFMPEG_TIMEOUT_SEC, check=True).stdout
        if out:
            return base64.b64encode(out).decode("ascii")
    except (OSError, subprocess.SubprocessError) as exc:
        log.info("mock: no real frame at %.1fs (%s), using placeholder", t_sec, type(exc).__name__)
    return base64.b64encode(PLACEHOLDER_JPEG.read_bytes()).decode("ascii")


def _demo_violation(video_path: str, type_name: str, t_sec: float, span: float) -> dict[str, Any]:
    frames = [t_sec - 0.5, t_sec, t_sec + 0.5]
    return {
        "type": type_name,
        "type_id": "demo",
        "timecode_sec": t_sec,
        "t_start": t_sec - span / 2,
        "t_end": t_sec + span / 2,
        "confidence": 0.9,
        "description": (
            f"ДЕМО-ответ заглушки. На {t_sec:.1f} с здесь будет описание того, что видно в кадре, "
            "и почему это может быть нарушением. Основание: ДЕМО-ссылка."
        ),
        "evidence_frames": [{"timestamp_sec": t, "image_base64": _frame_b64(video_path, t)} for t in frames],
        "rule": {
            "document": "ДЕМО: здесь будет название НПА из kb/violations.yaml",
            "clause": "п. —",
            "quote": "ДЕМО: здесь будет дословная цитата нормы.",
            "url": "https://adilet.zan.kz/",
        },
    }


def analyze(
    video_path: str,
    on_progress: ProgressFn | None = None,
    time_budget_sec: float | None = None,
    delay_sec: float = 0.0,
) -> dict[str, Any]:
    step_delay = delay_sec / len(STAGES) if STAGES else 0.0
    for i, stage in enumerate(STAGES):
        if on_progress:
            on_progress(i / len(STAGES), stage)
        time.sleep(step_delay)

    if "clean" in Path(video_path).name.lower():
        return {"verdict": "no_violation", "violations": []}
    return {
        "verdict": "violation",
        "violations": [
            _demo_violation(video_path, "ДЕМО: тип нарушения A", 3.0, 2.0),
            _demo_violation(video_path, "ДЕМО: тип нарушения B", 8.0, 4.0),
        ],
    }
