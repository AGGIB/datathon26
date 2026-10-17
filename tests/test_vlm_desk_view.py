"""Desk-view geometry and CPU composition on synthetic RGB fixtures only."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageDraw
import pytest

from ml.vlm.contracts import Frame
from ml.vlm import desk_view as module


def scene_geometry(scale=1):
    p = SimpleNamespace(
        info=SimpleNamespace(width=400 * scale, height=800 * scale),
        roles=SimpleNamespace(
            workplace=SimpleNamespace(desk=(160 * scale, 360 * scale))
        ),
    )
    s = SimpleNamespace(
        layout=SimpleNamespace(
            width=400 * scale,
            height=800 * scale,
            desk=tuple(value * scale for value in (40, 300, 280, 480)),
            confidence={"desk": 0.4},
        )
    )
    return p, s


def test_view_uses_existing_desk_zone_and_is_scale_equivariant():
    from ml.scene_items import desk_zone

    p, s = scene_geometry()
    expected = desk_zone(s.layout.desk, p.info.width, p.info.height)
    view = module.view_from_scene(p, s)
    assert view == module.DeskView(
        400, 800, tuple(v / d for v, d in zip(expected, (400, 800, 400, 800)))
    )
    assert module.view_from_scene(*scene_geometry(2)).roi == view.roi
    assert p.roles.workplace.desk == (160, 360) and s.layout.desk == (40, 300, 280, 480)


@pytest.mark.parametrize(
    "bad",
    [
        "missing_perception",
        "missing_scene",
        "missing_info",
        "missing_roles",
        "missing_workplace",
        "missing_layout",
        "missing_desk",
        "missing_point",
        "mismatch_dimensions",
        "bool_dimensions",
        "huge_dimensions",
        "low_confidence",
        "invalid_confidence",
        "bool_confidence",
        "missing_confidence",
        "nonfinite_box",
        "zero_box",
        "out_of_bounds_box",
        "point_outside",
        "nonfinite_point",
    ],
)
def test_unavailable_or_disagreeing_geometry_returns_none(bad):
    p, s = scene_geometry()
    if bad == "missing_perception":
        p = None
    elif bad == "missing_scene":
        s = None
    elif bad == "missing_info":
        p.info = None
    elif bad == "missing_roles":
        p.roles = None
    elif bad == "missing_workplace":
        p.roles.workplace = None
    elif bad == "missing_layout":
        s.layout = None
    elif bad == "missing_desk":
        s.layout.desk = None
    elif bad == "missing_point":
        p.roles.workplace.desk = None
    elif bad == "mismatch_dimensions":
        s.layout.width += 1
    elif bad == "bool_dimensions":
        p.info.width = True
    elif bad == "huge_dimensions":
        p.info.width = 30_000_000
    elif bad == "low_confidence":
        s.layout.confidence["desk"] = 0.399
    elif bad == "invalid_confidence":
        s.layout.confidence["desk"] = 1.01
    elif bad == "bool_confidence":
        s.layout.confidence["desk"] = True
    elif bad == "missing_confidence":
        s.layout.confidence = {}
    elif bad == "nonfinite_box":
        s.layout.desk = (40, float("nan"), 280, 480)
    elif bad == "zero_box":
        s.layout.desk = (40, 300, 40, 480)
    elif bad == "out_of_bounds_box":
        s.layout.desk = (-1, 300, 280, 480)
    elif bad == "point_outside":
        p.roles.workplace.desk = (390, 790)
    elif bad == "nonfinite_point":
        p.roles.workplace.desk = (float("inf"), 360)
    assert module.view_from_scene(p, s) is None


@pytest.mark.parametrize(
    "dimensions,roi",
    [
        ((True, 800), (0, 0.1, 0.9, 1)),
        ((0, 800), (0, 0.1, 0.9, 1)),
        ((400, 80_000), (0, 0.1, 0.9, 1)),
        ((400, 800), (0, 0.1, 2, 1)),
        ((400, 800), (0, 0.1, float("nan"), 1)),
        ((400, 800), (0.9, 0.1, 0.2, 1)),
    ],
)
def test_direct_view_configuration_is_bounded(dimensions, roi):
    with pytest.raises(ValueError, match="invalid_desk_view"):
        module.DeskView(*dimensions, roi)


@pytest.fixture
def synthetic_owner(monkeypatch):
    class Owner:
        def __init__(self):
            self.calls = []
            self.frames = ()
            self.after = None
            self.original_bytes = {}

        def extract(self, video_path, timestamps, directory, timeout_s):
            self.calls.append((video_path, timestamps, directory, timeout_s))
            result = []
            for index, timestamp in enumerate(timestamps):
                path = Path(directory) / f"synthetic-{index}.png"
                image = Image.new("RGB", (400, 800), (0, 0, 255))
                ImageDraw.Draw(image).rectangle(
                    (100, 300, 300, 500),
                    fill=((255, 0, 0) if index % 2 == 0 else (0, 255, 0)),
                )
                image.save(path)
                image.close()
                self.original_bytes[path] = path.read_bytes()
                result.append(Frame(timestamp, path, timestamp))
            self.frames = tuple(result)
            return self.after(self.frames) if self.after else self.frames

    owner = Owner()
    monkeypatch.setattr(module, "BatchedOwnerFrames", lambda: owner)
    return owner


def wrapper(view=None):
    return module.DeskZoomFrames(
        view or module.DeskView(400, 800, (0.25, 0.375, 0.75, 0.625))
    )


def test_exactly_one_owner_decode_same_frames_times_paths_and_aspect_fit(
    synthetic_owner, tmp_path
):
    source = wrapper()
    assert not source.applied and source.prompt_prefix == ""
    frames = source.extract("synthetic-local-video", (0.0, 2.0), tmp_path, 3)
    assert len(synthetic_owner.calls) == 1
    assert synthetic_owner.calls[0][:3] == (
        "synthetic-local-video",
        (0.0, 2.0),
        tmp_path,
    )
    assert 0 < synthetic_owner.calls[0][3] <= 3
    assert all(
        actual is expected for actual, expected in zip(frames, synthetic_owner.frames)
    )
    assert [(f.timestamp_s, f.pts_s) for f in frames] == [(0.0, 0.0), (2.0, 2.0)]
    for index, frame in enumerate(frames):
        with Image.open(frame.path) as image:
            assert image.size == module.COMPOSITE_SIZE == (640, 352)
            assert image.getpixel((7, 10)) == (
                128,
                128,
                128,
            )  # portrait aspect-fit left margin
            assert image.getpixel((9, 10)) == (
                0,
                0,
                255,
            )  # full source context remains visible
            assert image.getpixel((239, 176)) == (
                128,
                128,
                128,
            )  # square crop retains its aspect
            assert image.getpixel((416, 176)) == (
                (255, 0, 0) if index == 0 else (0, 255, 0)
            )
    assert source.applied and source.prompt_prefix == module.VIEW_DESCRIPTION
    assert (
        "SAME source frame" in source.prompt_prefix
        and "SAME people" in source.prompt_prefix
    )
    assert "independent observation" in source.prompt_prefix
    assert not list(tmp_path.glob(".desk-view-*"))


def test_missing_view_is_exact_owner_pass_through(
    synthetic_owner, tmp_path, monkeypatch
):
    source = module.DeskZoomFrames(None)
    monkeypatch.setattr(
        module, "_compose", lambda *a: pytest.fail("fallback composed pixels")
    )
    frames = source.extract("synthetic-local-video", (0.0,), tmp_path, 3)
    assert frames == synthetic_owner.frames and len(synthetic_owner.calls) == 1
    assert frames[0].path.read_bytes() == synthetic_owner.original_bytes[frames[0].path]
    assert not source.applied and source.prompt_prefix == ""


def test_source_dimensions_checked_before_rgb_decode(
    synthetic_owner, tmp_path, monkeypatch
):
    source = wrapper(module.DeskView(800, 400, (0.25, 0.375, 0.75, 0.625)))
    monkeypatch.setattr(
        Image.Image,
        "convert",
        lambda *a, **kw: pytest.fail("mismatched source decoded"),
    )
    with pytest.raises(ValueError, match="desk_view_source_dimensions"):
        source.extract("synthetic-local-video", (0.0,), tmp_path, 3)
    assert not source.applied and source.prompt_prefix == ""


@pytest.mark.parametrize(
    "bad",
    [
        "wrong_pts",
        "wrong_timestamp",
        "duplicate_time",
        "unowned_path",
        "symlink",
        "too_many",
    ],
)
def test_invalid_owner_results_do_not_reach_composition(
    synthetic_owner, tmp_path, monkeypatch, bad
):
    source = wrapper()

    def corrupt(frames):
        frame = frames[0]
        if bad == "wrong_pts":
            return (replace(frame, pts_s=1.0),)
        if bad == "wrong_timestamp":
            return (replace(frame, timestamp_s=1.0, pts_s=1.0),)
        if bad == "duplicate_time":
            return (frame, frame)
        if bad == "too_many":
            return (frame,) * 25
        if bad == "unowned_path":
            return (replace(frame, path=tmp_path.parent / "outside.png"),)
        link = tmp_path / "link.png"
        link.symlink_to(frame.path)
        return (replace(frame, path=link),)

    synthetic_owner.after = corrupt
    # Duplicate timestamps are rejected on the second result; first can be composed.
    calls = []
    monkeypatch.setattr(module, "_compose", lambda *a: calls.append(a[0]))
    with pytest.raises(ValueError):
        source.extract("synthetic-local-video", (0.0,), tmp_path, 3)
    assert len(calls) == (1 if bad == "duplicate_time" else 0)
    assert not source.applied


@pytest.mark.parametrize("timeout", [True, float("nan"), 0.0, -1.0])
def test_invalid_timeout_does_not_decode(synthetic_owner, tmp_path, timeout):
    with pytest.raises((ValueError, TimeoutError)):
        wrapper().extract("synthetic-local-video", (0.0,), tmp_path, timeout)
    assert not synthetic_owner.calls


@pytest.mark.parametrize(
    "timestamps", [(), tuple(range(25)), (0.0, 0.0), (float("nan"),), (-1.0,), (True,)]
)
def test_invalid_request_does_not_decode(synthetic_owner, tmp_path, timestamps):
    with pytest.raises(ValueError):
        wrapper().extract("synthetic-local-video", timestamps, tmp_path, 3)
    assert not synthetic_owner.calls


def test_owner_exhausting_budget_prevents_composition(
    synthetic_owner, tmp_path, monkeypatch
):
    clock = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])

    def expire(frames):
        clock[0] = 2.0
        return frames

    synthetic_owner.after = expire
    monkeypatch.setattr(
        module,
        "_compose",
        lambda *a: pytest.fail("composed after decode exhausted budget"),
    )
    source = wrapper()
    with pytest.raises(TimeoutError):
        source.extract("synthetic-local-video", (0.0,), tmp_path, 1)
    assert not source.applied


def test_deadline_after_each_compose_stops_next_frame_and_clears_applied(
    synthetic_owner, tmp_path, monkeypatch
):
    clock, calls = [0.0], []
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    original = module._compose

    def slow_compose(path, view, deadline):
        calls.append(path)
        original(path, view, deadline)
        clock[0] += 0.6

    monkeypatch.setattr(module, "_compose", slow_compose)
    source = wrapper()
    with pytest.raises(TimeoutError):
        source.extract("synthetic-local-video", (0.0, 1.0, 2.0), tmp_path, 1)
    assert len(calls) == 2 and not source.applied and source.prompt_prefix == ""
    last = synthetic_owner.frames[-1].path
    assert last.read_bytes() == synthetic_owner.original_bytes[last]


def test_timeout_during_png_save_preserves_source_and_removes_temporary(
    synthetic_owner, tmp_path, monkeypatch
):
    clock = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    original = Image.Image.save

    def slow_save(image, target, *args, **kwargs):
        original(image, target, *args, **kwargs)
        if hasattr(target, "name") and ".desk-view-" in str(target.name):
            clock[0] = 2.0

    monkeypatch.setattr(Image.Image, "save", slow_save)
    with pytest.raises(TimeoutError):
        wrapper().extract("synthetic-local-video", (0.0,), tmp_path, 1)
    frame = synthetic_owner.frames[0]
    assert frame.path.read_bytes() == synthetic_owner.original_bytes[frame.path]
    assert not list(tmp_path.glob(".desk-view-*"))


def test_applied_flag_is_reset_on_later_failed_extraction(synthetic_owner, tmp_path):
    source = wrapper()
    source.extract("synthetic-local-video", (0.0,), tmp_path, 3)
    assert source.applied
    with pytest.raises(TimeoutError):
        source.extract("synthetic-local-video", (0.0,), tmp_path, 0)
    assert not source.applied and source.prompt_prefix == ""
