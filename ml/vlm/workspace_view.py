"""Optional same-frame workspace representation for local verifier experiments.

Only existing image files are read; no video decode or model work occurs here.
Layout boxes are uncertain attention proposals, never verified object evidence.
The caller owns the inference deadline and must retain the supplied schema/gates.
Pillow operations have cooperative deadline checks, not hard cancellation.
"""

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
import math
from pathlib import Path
import stat
import tempfile
import time

from .contracts import Frame
from .mlx_backend import MAX_FRAME_FILE_BYTES, MAX_SOURCE_PIXELS
from .validation import number

POLICY_ID = "same_frame_workspace_union_640x352_v1"
CANVAS_SIZE = (640, 352)
CONTEXT_SIZE = (192, 352)
ZOOM_SIZE = (448, 352)
MIN_DESK_CONFIDENCE = 0.4
MAX_UNION_FRAME_SHARE = 0.65
MIN_RENDER_MAGNIFICATION = 1.2
MAX_FRAMES = 12
VIEW_DESCRIPTION = (
    "Each supplied image has two views of the SAME source frame at the SAME timestamp. "
    "The left panel is the complete scene; the right panel is an enlarged region from that same image. "
    "People and objects repeated across the panels are the SAME people and objects, not additional "
    "people, events, frames, timestamps or independent evidence. The region is only an uncertain "
    "attention proposal; it does not establish the identity of any fixture or object. "
    "Use the complete left view for context and the right view for visible detail.\n"
)


def _check_deadline(deadline):
    if not number(deadline):
        raise ValueError("invalid_view_deadline")
    if time.monotonic() >= deadline:
        raise TimeoutError("workspace_view_timeout")


def _dimensions(width, height):
    return (
        type(width) is int
        and type(height) is int
        and width > 0
        and height > 0
        and width * height <= MAX_SOURCE_PIXELS
    )


def _coordinates(value, count):
    if (
        not isinstance(value, (tuple, list))
        or len(value) != count
        or not all(number(item) for item in value)
    ):
        return None
    return tuple(float(item) for item in value)


def _box(value, width, height):
    value = _coordinates(value, 4)
    return (
        value
        if value is not None
        and 0 <= value[0] < value[2] <= width
        and 0 <= value[1] < value[3] <= height
        else None
    )


def _fit(width, height, panel):
    scale = min(panel[0] / width, panel[1] / height)
    return max(1, math.floor(width * scale)), max(1, math.floor(height * scale))


@dataclass(frozen=True)
class WorkspaceView:
    source_width: int
    source_height: int
    desk_box: tuple[float, float, float, float]
    safe_box: tuple[float, float, float, float]
    desk_confidence: float
    safe_confidence: float
    plain_max_image_side: int = 640

    def __post_init__(self):
        if (
            not _dimensions(self.source_width, self.source_height)
            or type(self.plain_max_image_side) is not int
            or not 224 <= self.plain_max_image_side <= 896
        ):
            raise ValueError("invalid_workspace_dimensions")
        for name in ("desk_box", "safe_box"):
            value = _box(getattr(self, name), self.source_width, self.source_height)
            if value is None:
                raise ValueError("invalid_workspace_box")
            object.__setattr__(self, name, value)
        if (
            not number(self.desk_confidence)
            or not MIN_DESK_CONFIDENCE <= self.desk_confidence <= 1
            or not number(self.safe_confidence)
            or not 0 < self.safe_confidence <= 1
        ):
            raise ValueError("invalid_workspace_confidence")
        x0, y0, x1, y1 = self.source_box
        share = (x1 - x0) * (y1 - y0) / (self.source_width * self.source_height)
        if (
            share > MAX_UNION_FRAME_SHARE
            or self.render_magnification < MIN_RENDER_MAGNIFICATION
        ):
            raise ValueError("workspace_view_no_useful_zoom")

    @property
    def source_box(self):
        return (
            math.floor(min(self.desk_box[0], self.safe_box[0])),
            math.floor(min(self.desk_box[1], self.safe_box[1])),
            math.ceil(max(self.desk_box[2], self.safe_box[2])),
            math.ceil(max(self.desk_box[3], self.safe_box[3])),
        )

    @property
    def render_magnification(self):
        x0, y0, x1, y1 = self.source_box
        width, height = _fit(x1 - x0, y1 - y0, ZOOM_SIZE)
        reference_scale = min(
            1.0, self.plain_max_image_side / max(self.source_width, self.source_height)
        )
        return min(width / (x1 - x0), height / (y1 - y0)) / reference_scale


def workspace_view_from_scene(perception, scene, *, plain_max_image_side=640):
    """Union observed desk+safe boxes when desk/workplace geometry agrees.

    Safe confidence must be finite positive, not a semantic acceptance gate.
    Missing/invalid geometry never falls back to a camera-specific guessed ROI.
    """
    info = getattr(perception, "info", None)
    workplace = getattr(getattr(perception, "roles", None), "workplace", None)
    layout = getattr(scene, "layout", None)
    width, height = getattr(info, "width", None), getattr(info, "height", None)
    if (
        not _dimensions(width, height)
        or layout is None
        or type(getattr(layout, "width", None)) is not int
        or type(getattr(layout, "height", None)) is not int
        or (layout.width, layout.height) != (width, height)
    ):
        return None
    desk = _box(getattr(layout, "desk", None), width, height)
    safe = _box(getattr(layout, "safe", None), width, height)
    point = _coordinates(getattr(workplace, "desk", None), 2)
    confidence = getattr(layout, "confidence", None)
    if (
        desk is None
        or safe is None
        or point is None
        or not isinstance(confidence, Mapping)
        or not desk[0] <= point[0] <= desk[2]
        or not desk[1] <= point[1] <= desk[3]
    ):
        return None
    try:
        return WorkspaceView(
            width,
            height,
            desk,
            safe,
            confidence.get("desk"),
            confidence.get("safe"),
            plain_max_image_side,
        )
    except ValueError:
        return None


def _directory(value):
    path = Path(value).absolute()
    aliases = {Path("/tmp"): Path("/private/tmp"), Path("/var"): Path("/private/var")}
    for ancestor in (path, *path.parents):
        if ancestor.is_symlink() and not (
            ancestor in aliases and ancestor.resolve() == aliases[ancestor]
        ):
            raise ValueError("workspace_view_link_not_allowed")
    if not path.is_dir():
        raise ValueError("workspace_view_directory_required")
    return path.resolve()


def _validate_frames(frames, directory):
    if not isinstance(frames, (tuple, list)) or not 1 <= len(frames) <= MAX_FRAMES:
        raise ValueError("invalid_workspace_frame_count")
    frames = tuple(frames)
    times, paths = set(), set()
    for frame in frames:
        if (
            not isinstance(frame, Frame)
            or not number(frame.timestamp_s)
            or frame.timestamp_s < 0
            or not number(frame.pts_s)
            or frame.pts_s < 0
            or frame.timestamp_s in times
        ):
            raise ValueError("invalid_workspace_frame_timebase")
        path = Path(frame.path)
        if path.is_symlink() or _directory(path.parent) != directory:
            raise ValueError("invalid_workspace_source_provenance")
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or not 0 < metadata.st_size <= MAX_FRAME_FILE_BYTES
            or path.resolve() in paths
        ):
            raise ValueError("invalid_workspace_source_file")
        times.add(frame.timestamp_s)
        paths.add(path.resolve())
    return frames


class _RepresentationUnavailable(ValueError):
    pass


def _panel(rgb, source_box, canvas, panel_box):
    from PIL import Image

    x0, y0, x1, y1 = source_box
    px0, py0, px1, py1 = panel_box
    size = _fit(x1 - x0, y1 - y0, (px1 - px0, py1 - py0))
    left, top = px0 + (px1 - px0 - size[0]) // 2, py0 + (py1 - py0 - size[1]) // 2
    with (
        rgb.crop(source_box) as crop,
        crop.resize(size, Image.Resampling.LANCZOS) as fitted,
    ):
        canvas.paste(fitted, (left, top))
    sx, sy = size[0] / (x1 - x0), size[1] / (y1 - y0)
    return {
        "panel_box": list(panel_box),
        "content_box": [left, top, left + size[0], top + size[1]],
        "source_box": list(source_box),
        "source_to_canvas": [sx, 0, left - sx * x0, 0, sy, top - sy * y0],
    }


def _compose(frame, view, target, deadline):
    from PIL import Image

    _check_deadline(deadline)
    with Image.open(frame.path) as source:
        if source.size != (view.source_width, view.source_height):
            raise _RepresentationUnavailable("workspace_source_dimensions_mismatch")
        _check_deadline(deadline)
        with (
            source.convert("RGB") as rgb,
            Image.new("RGB", CANVAS_SIZE, (128, 128, 128)) as canvas,
        ):
            context = _panel(
                rgb, (0, 0, rgb.width, rgb.height), canvas, (0, 0, 192, 352)
            )
            _check_deadline(deadline)
            zoom = _panel(rgb, view.source_box, canvas, (192, 0, 640, 352))
            _check_deadline(deadline)
            canvas.save(target, format="PNG")
            _check_deadline(deadline)
    if not 0 < target.stat().st_size <= MAX_FRAME_FILE_BYTES:
        raise ValueError("workspace_output_resource_limit")
    return Frame(frame.timestamp_s, target, frame.pts_s), context, zoom


@dataclass(frozen=True)
class WorkspacePresentation:
    frames: tuple[Frame, ...]
    prompt_prefix: str
    geometry: dict


def _report(frames, view, *, applied=False, reason="geometry_unavailable"):
    return {
        "policy_id": POLICY_ID,
        "applied": applied,
        "fallback_reason": None if applied else reason,
        "attention_geometry_is_not_verified_object_evidence": True,
        "timebase": "unmodified_upstream_frame_metadata",
        "canvas_size": list(CANVAS_SIZE),
        "context_size": list(CONTEXT_SIZE),
        "zoom_size": list(ZOOM_SIZE),
        "policy": {
            "min_desk_confidence": MIN_DESK_CONFIDENCE,
            "safe_confidence_strictly_positive": True,
            "max_union_frame_share": MAX_UNION_FRAME_SHARE,
            "min_render_magnification": MIN_RENDER_MAGNIFICATION,
            "max_frames": MAX_FRAMES,
            "plain_max_image_side": view.plain_max_image_side if view else None,
        },
        "source_frames": [
            {
                "supplied_frame_index": index,
                "timestamp_s": frame.timestamp_s,
                "pts_s": frame.pts_s,
            }
            for index, frame in enumerate(frames, 1)
        ],
    }


@contextmanager
def prepare_workspace_frames(frames, view, directory, *, deadline):
    """Yield one derived image per existing Frame, then remove only our derivatives.

    The harness applies prompt_prefix to its VERIFIER request, keeps the existing
    response schema, and spends its existing budget for inference. This helper
    never calls a backend. None/unavailable views retain exact original Frames.
    A shared deadline expiry propagates; it never grants a fresh model budget.
    """
    _check_deadline(deadline)
    if view is not None and not isinstance(view, WorkspaceView):
        raise ValueError("invalid_workspace_view")
    directory = _directory(directory)
    frames = _validate_frames(frames, directory)
    _check_deadline(deadline)
    original = WorkspacePresentation(frames, "", _report(frames, view))
    if view is None:
        yield original
        return
    temporary_directory = tempfile.TemporaryDirectory(
        prefix=".workspace-view-", dir=directory
    )
    with temporary_directory as temporary:
        try:
            transformed, context, zoom = [], None, None
            for index, frame in enumerate(frames, 1):
                derived, context, zoom = _compose(
                    frame, view, Path(temporary) / f"frame-{index}.png", deadline
                )
                transformed.append(derived)
            _check_deadline(deadline)
            report = _report(frames, view, applied=True)
            report.update(
                source_size=[view.source_width, view.source_height],
                source_roi_box=list(view.source_box),
                context=context,
                zoom=zoom,
                render_magnification=view.render_magnification,
                geometry_confidence={
                    "desk": view.desk_confidence,
                    "safe": view.safe_confidence,
                },
            )
            prepared = WorkspacePresentation(
                tuple(transformed), VIEW_DESCRIPTION, report
            )
        except (_RepresentationUnavailable, OSError):
            temporary_directory.cleanup()
            _check_deadline(deadline)
            prepared = WorkspacePresentation(
                frames, "", _report(frames, view, reason="representation_unavailable")
            )
        yield prepared
