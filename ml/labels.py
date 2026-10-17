"""Loader for the organizers' open_labels.xlsx (video name, timestamp, violation flag)."""

import re
from dataclasses import dataclass
from pathlib import Path

_MIN_RE = re.compile(r"(\d+)\s*минут\w*")
_SEC_RE = re.compile(r"(\d+)\s*секунд\w*")
_CYRILLIC_X = str.maketrans({"Х": "X", "х": "X", "x": "X"})


@dataclass(frozen=True)
class Label:
    video: str
    has_violation: bool
    t_sec: float | None


def parse_timestamp(raw: str | None) -> float | None:
    """'22 секунда' -> 22.0, '1минута 8 секунда' -> 68.0, empty -> None."""
    if raw is None or not str(raw).strip():
        return None
    text = str(raw).lower()
    minutes = _MIN_RE.search(text)
    seconds = _SEC_RE.search(text)
    if not minutes and not seconds:
        raise ValueError(f"unrecognized timestamp: {raw!r}")
    return float((int(minutes.group(1)) if minutes else 0) * 60 + (int(seconds.group(1)) if seconds else 0))


def normalize_video_name(raw: str) -> str:
    """Dataset mixes Latin X and Cyrillic Х; we key everything by 'X-<n>'."""
    stem = Path(raw.strip()).stem
    return stem.translate(_CYRILLIC_X).upper()


def load_labels(xlsx_path: Path) -> list[Label]:
    from openpyxl import load_workbook

    sheet = load_workbook(xlsx_path, read_only=True).active
    labels = []
    for name, stamp, flag in sheet.iter_rows(min_row=2, max_col=3, values_only=True):
        if not name:
            continue
        has_violation = str(flag).strip().lower() == "нарушение"
        labels.append(Label(normalize_video_name(str(name)), has_violation, parse_timestamp(stamp)))
    return labels
