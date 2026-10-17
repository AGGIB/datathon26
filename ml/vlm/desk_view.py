"""Optional same-frame desk detail view; never enabled by production defaults.

Geometry comes from this clip's perception/scene, without camera or label tables.
The composite keeps one Frame and timestamp: its second panel is repeated detail,
not another person, event or independent observation. CPU composition is bounded
by source dimensions and cooperative deadlines; an individual PIL call is not
hard-interruptible. No model imports, inference, new video decode, or network.
"""

from dataclasses import dataclass
import math
from pathlib import Path
import tempfile
import time
from collections.abc import Mapping

from .contracts import Frame
from .discovery import BatchedOwnerFrames
from .mlx_backend import MAX_FRAME_FILE_BYTES, MAX_SOURCE_PIXELS
from .validation import number

COMPOSITE_SIZE = (640, 352)
CONTEXT_WIDTH = 192
POLICY_ID = "same_frame_desk_640x352_v1"
VIEW_DESCRIPTION = (
    "Each supplied image contains two views of the SAME source frame at the SAME timestamp: "
    "the left panel is the full scene, and the right panel enlarges its desk region. "
    "People and objects appearing in both panels are the SAME people and objects. "
    "Do not count the enlarged panel as another person, object, event, timestamp, or independent observation. "
    "Use the left panel for spatial relationships and the right panel only for detail.\n"
)


def _coordinates(value, size):
    if (
        not isinstance(value, (tuple, list))
        or len(value) != size
        or not all(number(v) for v in value)
    ):
        return None
    return tuple(float(v) for v in value)


def _dimensions(width, height):
    return (
        type(width) is int
        and type(height) is int
        and width > 0
        and height > 0
        and width * height <= MAX_SOURCE_PIXELS
    )


@dataclass(frozen=True)
class DeskView:
    source_width: int
    source_height: int
    roi: tuple[float, float, float, float]  # normalized x0,y0,x1,y1

    def __post_init__(self):
        roi = _coordinates(self.roi, 4)
        if (
            not _dimensions(self.source_width, self.source_height)
            or roi is None
            or not 0 <= roi[0] < roi[2] <= 1
            or not 0 <= roi[1] < roi[3] <= 1
        ):
            raise ValueError("invalid_desk_view")
        object.__setattr__(self, "roi", roi)


def view_from_scene(perception, scene) -> DeskView | None:
    """Return a clip-relative desk ROI only when both geometry sources agree."""
    info = getattr(perception, "info", None)
    workplace = getattr(getattr(perception, "roles", None), "workplace", None)
    layout = getattr(scene, "layout", None)
    width, height = getattr(info, "width", None), getattr(info, "height", None)
    if (
        not _dimensions(width, height)
        or workplace is None
        or layout is None
        or type(getattr(layout, "width", None)) is not int
        or type(getattr(layout, "height", None)) is not int
        or (layout.width, layout.height) != (width, height)
    ):
        return None
    confidence = getattr(layout, "confidence", None)
    score = confidence.get("desk") if isinstance(confidence, Mapping) else None
    desk = _coordinates(getattr(layout, "desk", None), 4)
    point = _coordinates(getattr(workplace, "desk", None), 2)
    if (
        not number(score)
        or not 0.4 <= score <= 1
        or desk is None
        or point is None
        or not 0 <= desk[0] < desk[2] <= width
        or not 0 <= desk[1] < desk[3] <= height
    ):
        return None
    from ml.scene_items import desk_zone

    x0, y0, x1, y1 = desk_zone(desk, width, height)
    if not (x0 <= point[0] <= x1 and y0 <= point[1] <= y1):
        return None
    return DeskView(width, height, (x0 / width, y0 / height, x1 / width, y1 / height))


def _check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("frame_timeout")


def _compose(path, view, deadline):
    from PIL import Image, ImageOps

    temporary = None
    try:
        with Image.open(path) as source:
            # Check metadata before loading/decoding the source pixels.
            if source.size != (view.source_width, view.source_height):
                raise ValueError("desk_view_source_dimensions")
            _check_deadline(deadline)
            with (
                source.convert("RGB") as rgb,
                Image.new("RGB", COMPOSITE_SIZE, (128, 128, 128)) as canvas,
            ):
                left = ImageOps.contain(
                    rgb, (CONTEXT_WIDTH, COMPOSITE_SIZE[1]), Image.Resampling.LANCZOS
                )
                try:
                    canvas.paste(
                        left,
                        (
                            (CONTEXT_WIDTH - left.width) // 2,
                            (COMPOSITE_SIZE[1] - left.height) // 2,
                        ),
                    )
                finally:
                    left.close()
                x0, y0, x1, y1 = view.roi
                box = (
                    math.floor(x0 * rgb.width),
                    math.floor(y0 * rgb.height),
                    math.ceil(x1 * rgb.width),
                    math.ceil(y1 * rgb.height),
                )
                with rgb.crop(box) as crop:
                    right = ImageOps.contain(
                        crop,
                        (COMPOSITE_SIZE[0] - CONTEXT_WIDTH, COMPOSITE_SIZE[1]),
                        Image.Resampling.LANCZOS,
                    )
                    try:
                        canvas.paste(
                            right,
                            (
                                CONTEXT_WIDTH
                                + (COMPOSITE_SIZE[0] - CONTEXT_WIDTH - right.width)
                                // 2,
                                (COMPOSITE_SIZE[1] - right.height) // 2,
                            ),
                        )
                    finally:
                        right.close()
                _check_deadline(deadline)
                with tempfile.NamedTemporaryFile(
                    prefix=".desk-view-", suffix=".png", dir=path.parent, delete=False
                ) as output:
                    temporary = Path(output.name)
                    canvas.save(output, format="PNG")
                _check_deadline(deadline)
                temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class DeskZoomFrames:
    """BatchFrameSource wrapper: one owner decode, then CPU-only same-frame views.

    ``applied`` and ``prompt_prefix`` describe the last successful extraction.
    Give each concurrent job its own wrapper. None means ordinary full frames.
    The owner must prepend prompt_prefix only after extraction applied the view.
    """

    timebase = "cv_index_fps"

    def __init__(self, view: DeskView | None):
        if view is not None and not isinstance(view, DeskView):
            raise ValueError("invalid_desk_view")
        self.view = view
        self.applied = False
        self._source = BatchedOwnerFrames()

    @property
    def prompt_prefix(self):
        return VIEW_DESCRIPTION if self.applied else ""

    def extract(
        self, video_path, timestamps, directory, timeout_s
    ) -> tuple[Frame, ...]:
        self.applied = False
        if not number(timeout_s):
            raise ValueError("invalid_frame_timeout")
        if timeout_s <= 0:
            raise TimeoutError("frame_timeout")
        requested = tuple(timestamps)
        if (
            not 1 <= len(requested) <= 24
            or len(set(requested)) != len(requested)
            or any(not number(t) or t < 0 for t in requested)
        ):
            raise ValueError("invalid_frame_timestamps")
        deadline = time.monotonic() + timeout_s
        _check_deadline(deadline)
        frames = tuple(
            self._source.extract(
                video_path, requested, directory, deadline - time.monotonic()
            )
        )
        _check_deadline(deadline)
        if self.view is None:
            return frames
        directory = Path(directory).resolve()
        if len(frames) > 24:
            raise ValueError("invalid_frame_count")
        seen_times, seen_paths = set(), set()
        for frame in frames:
            _check_deadline(deadline)
            if (
                not isinstance(frame, Frame)
                or not number(frame.timestamp_s)
                or not number(frame.pts_s)
                or frame.timestamp_s not in requested
                or frame.timestamp_s != frame.pts_s
                or frame.timestamp_s in seen_times
            ):
                raise ValueError("invalid_frame_timebase")
            path = Path(frame.path)
            if (
                path.is_symlink()
                or path.parent.resolve() != directory
                or not path.is_file()
                or path.stat().st_size > MAX_FRAME_FILE_BYTES
                or path.resolve() in seen_paths
            ):
                raise ValueError("invalid_frame_file")
            seen_times.add(frame.timestamp_s)
            seen_paths.add(path.resolve())
            _compose(path, self.view, deadline)
            _check_deadline(deadline)
        self.applied = bool(frames)
        return frames
