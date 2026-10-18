"""Spatial evidence representation tested only on synthetic local images."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

from PIL import Image, ImageDraw
import pytest

from ml.vlm.contracts import Frame, InferenceRequest
from ml.vlm import workspace_view as module


def geometry(scale=1):
    perception = SimpleNamespace(
        info=SimpleNamespace(width=400 * scale, height=800 * scale),
        roles=SimpleNamespace(
            workplace=SimpleNamespace(desk=(160 * scale, 400 * scale))
        ),
    )
    scene = SimpleNamespace(
        layout=SimpleNamespace(
            width=400 * scale,
            height=800 * scale,
            desk=tuple(v * scale for v in (80, 340, 280, 500)),
            safe=tuple(v * scale for v in (280, 260, 350, 470)),
            confidence={"desk": 0.5, "safe": 0.14},
        )
    )
    return perception, scene


def view():
    return module.workspace_view_from_scene(*geometry())


@pytest.fixture
def frames(tmp_path):
    result = []
    for index, timestamp in enumerate((0.4, 4.6, 9.7)):
        path = tmp_path / f"synthetic-{index}.png"
        with Image.new("RGB", (400, 800), (0, 0, 220)) as image:
            draw = ImageDraw.Draw(image)
            draw.rectangle((100, 360, 250, 480), fill=(220, 0, 0))
            draw.rectangle((290, 280, 330, 450), fill=(0, 220, 0))
            image.save(path)
        result.append(Frame(timestamp, path, timestamp + 0.01))
    return tuple(result)


def test_union_is_scale_equivariant_and_keeps_weak_safe_as_attention_only():
    first = view()
    assert first is not None and first.source_box == (80, 260, 350, 500)
    assert first.safe_confidence == 0.14 and first.desk_confidence == 0.5
    doubled = module.workspace_view_from_scene(*geometry(2))
    assert doubled.source_box == tuple(v * 2 for v in first.source_box)
    assert doubled.render_magnification == pytest.approx(first.render_magnification)
    perception, scene = geometry()
    scene.layout.confidence["safe"] = 1e-9
    assert module.workspace_view_from_scene(perception, scene) is not None
    assert scene.layout.safe == (280, 260, 350, 470)


@pytest.mark.parametrize(
    "bad",
    [
        "missing_scene",
        "missing_roles",
        "missing_safe",
        "missing_desk",
        "dimension_disagreement",
        "bool_dimension",
        "huge_dimension",
        "bad_safe_box",
        "bad_desk_box",
        "bad_point",
        "disagreeing_desk_point",
        "low_desk",
        "zero_safe",
        "negative_safe",
        "nan_safe",
        "bool_safe",
        "missing_confidence",
        "confidence_over_one",
    ],
)
def test_bad_or_missing_geometry_does_not_guess_a_region(bad):
    perception, scene = geometry()
    if bad == "missing_scene":
        scene = None
    elif bad == "missing_roles":
        perception.roles = None
    elif bad == "missing_safe":
        scene.layout.safe = None
    elif bad == "missing_desk":
        scene.layout.desk = None
    elif bad == "dimension_disagreement":
        scene.layout.width += 1
    elif bad == "bool_dimension":
        perception.info.width = True
    elif bad == "huge_dimension":
        perception.info.width = 50_000_000
    elif bad == "bad_safe_box":
        scene.layout.safe = (280, -1, 350, 470)
    elif bad == "bad_desk_box":
        scene.layout.desk = (80, 340, float("nan"), 500)
    elif bad == "bad_point":
        perception.roles.workplace.desk = (float("inf"), 400)
    elif bad == "disagreeing_desk_point":
        perception.roles.workplace.desk = (10, 10)
    elif bad == "low_desk":
        scene.layout.confidence["desk"] = 0.399
    elif bad == "zero_safe":
        scene.layout.confidence["safe"] = 0
    elif bad == "negative_safe":
        scene.layout.confidence["safe"] = -0.1
    elif bad == "nan_safe":
        scene.layout.confidence["safe"] = float("nan")
    elif bad == "bool_safe":
        scene.layout.confidence["safe"] = True
    elif bad == "missing_confidence":
        scene.layout.confidence = {}
    elif bad == "confidence_over_one":
        scene.layout.confidence["safe"] = 1.01
    assert module.workspace_view_from_scene(perception, scene) is None


@pytest.mark.parametrize(
    "desk,safe",
    [
        ((0, 0, 350, 780), (350, 0, 400, 780)),
        ((0, 200, 360, 680), (360, 180, 400, 400)),
    ],
)
def test_broad_union_or_no_magnification_is_plain_fallback(desk, safe):
    perception, scene = geometry()
    scene.layout.desk, scene.layout.safe = desk, safe
    assert module.workspace_view_from_scene(perception, scene) is None


def project(transform, point):
    a, b, c, d, e, f = transform
    return round(a * point[0] + b * point[1] + c), round(
        d * point[0] + e * point[1] + f
    )


def test_same_source_reprojection_dimensions_times_and_cleanup(frames, tmp_path):
    original_bytes = [f.path.read_bytes() for f in frames]
    with module.prepare_workspace_frames(
        frames, view(), tmp_path, deadline=time.monotonic() + 5
    ) as prepared:
        assert len(prepared.frames) == len(frames) == 3
        assert [(f.timestamp_s, f.pts_s) for f in prepared.frames] == [
            (f.timestamp_s, f.pts_s) for f in frames
        ]
        assert prepared.geometry["source_frames"] == [
            {
                "supplied_frame_index": index,
                "timestamp_s": f.timestamp_s,
                "pts_s": f.pts_s,
            }
            for index, f in enumerate(frames, 1)
        ]
        assert prepared.geometry["attention_geometry_is_not_verified_object_evidence"]
        assert prepared.geometry["policy"]["safe_confidence_strictly_positive"]
        assert prepared.geometry["policy"]["min_desk_confidence"] == 0.4
        assert prepared.geometry["source_roi_box"] == [80, 260, 350, 500]
        assert (
            "SAME people" in prepared.prompt_prefix
            and "not additional" in prepared.prompt_prefix
        )
        assert "uncertain attention proposal" in prepared.prompt_prefix
        for frame in prepared.frames:
            assert frame.path not in [f.path for f in frames]
            with Image.open(frame.path) as image:
                assert image.size == (640, 352)
                for panel in ("context", "zoom"):
                    red = project(
                        prepared.geometry[panel]["source_to_canvas"], (160, 420)
                    )
                    green = project(
                        prepared.geometry[panel]["source_to_canvas"], (310, 360)
                    )
                    assert image.getpixel(red) == pytest.approx((220, 0, 0), abs=3)
                    assert image.getpixel(green) == pytest.approx((0, 220, 0), abs=3)
        derived = [f.path for f in prepared.frames]
        serialized = json.dumps(prepared.geometry)
        assert all(str(f.path) not in serialized for f in frames)
    assert all(not path.exists() for path in derived)
    assert not list(tmp_path.glob(".workspace-view-*"))
    assert [f.path.read_bytes() for f in frames] == original_bytes


def test_missing_geometry_is_exact_original_frames_without_pixel_decode(
    frames, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        Image, "open", lambda *a, **k: pytest.fail("fallback read pixels")
    )
    with module.prepare_workspace_frames(
        frames, None, tmp_path, deadline=time.monotonic() + 2
    ) as prepared:
        assert prepared.frames is frames
        assert prepared.prompt_prefix == "" and not prepared.geometry["applied"]
    assert not list(tmp_path.glob(".workspace-view-*"))


def test_mismatched_later_frame_falls_back_atomically_and_removes_partial_derivatives(
    frames, tmp_path
):
    with Image.new("RGB", (800, 400)) as wrong:
        wrong.save(frames[-1].path)
    original = [f.path.read_bytes() for f in frames]
    with module.prepare_workspace_frames(
        frames, view(), tmp_path, deadline=time.monotonic() + 3
    ) as prepared:
        assert prepared.frames is frames and prepared.prompt_prefix == ""
        assert prepared.geometry["fallback_reason"] == "representation_unavailable"
        assert not list(tmp_path.glob(".workspace-view-*"))
    assert [f.path.read_bytes() for f in frames] == original


@pytest.mark.parametrize(
    "kind",
    ["duplicate_time", "duplicate_path", "unowned", "symlink", "too_many", "bad_pts"],
)
def test_invalid_provenance_and_caps_rejected_before_composition(
    frames, tmp_path, monkeypatch, kind
):
    chosen = frames
    if kind == "duplicate_time":
        chosen = (frames[0], replace(frames[1], timestamp_s=frames[0].timestamp_s))
    elif kind == "duplicate_path":
        chosen = (frames[0], replace(frames[1], path=frames[0].path))
    elif kind == "unowned":
        chosen = (replace(frames[0], path=tmp_path.parent / "outside.png"),)
    elif kind == "too_many":
        chosen = frames * 5
    elif kind == "bad_pts":
        chosen = (replace(frames[0], pts_s=True),)
    elif kind == "symlink":
        link = tmp_path / "alias.png"
        link.symlink_to(frames[0].path)
        chosen = (replace(frames[0], path=link),)
    monkeypatch.setattr(
        module, "_compose", lambda *a: pytest.fail("invalid provenance reached pixels")
    )
    with pytest.raises((ValueError, OSError)):
        with module.prepare_workspace_frames(
            chosen, view(), tmp_path, deadline=time.monotonic() + 2
        ):
            pytest.fail("invalid source yielded")


def test_file_resource_limit_checked_before_image_open(frames, tmp_path, monkeypatch):
    monkeypatch.setattr(module, "MAX_FRAME_FILE_BYTES", 1)
    monkeypatch.setattr(
        Image, "open", lambda *a, **k: pytest.fail("oversized source opened")
    )
    with pytest.raises(ValueError, match="invalid_workspace_source_file"):
        with module.prepare_workspace_frames(
            frames, view(), tmp_path, deadline=time.monotonic() + 2
        ):
            pytest.fail("oversized source yielded")


@pytest.mark.parametrize("deadline", [True, float("nan"), float("inf"), 0.0])
def test_invalid_or_expired_deadline_never_composes(
    frames, tmp_path, monkeypatch, deadline
):
    monkeypatch.setattr(module, "_compose", lambda *a: pytest.fail("deadline bypassed"))
    with pytest.raises((ValueError, TimeoutError)):
        with module.prepare_workspace_frames(
            frames, view(), tmp_path, deadline=deadline
        ):
            pytest.fail("invalid deadline yielded")


def test_timeout_during_save_preserves_originals_and_cleans_files(
    frames, tmp_path, monkeypatch
):
    original = [f.path.read_bytes() for f in frames]
    clock = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    save = Image.Image.save

    def slow_save(image, target, *args, **kwargs):
        save(image, target, *args, **kwargs)
        clock[0] = 2.0

    monkeypatch.setattr(Image.Image, "save", slow_save)
    with pytest.raises(TimeoutError):
        with module.prepare_workspace_frames(frames, view(), tmp_path, deadline=1):
            pytest.fail("expired presentation yielded")
    assert not list(tmp_path.glob(".workspace-view-*"))
    assert [f.path.read_bytes() for f in frames] == original


def test_caller_exception_is_not_reinterpreted_as_fallback(frames, tmp_path):
    error = OSError("synthetic caller failure")
    with pytest.raises(OSError) as caught:
        with module.prepare_workspace_frames(
            frames, view(), tmp_path, deadline=time.monotonic() + 2
        ):
            raise error
    assert caught.value is error
    assert not list(tmp_path.glob(".workspace-view-*"))


def test_actual_compact_schema_budget_and_source_mapping_are_unchanged(
    frames, tmp_path
):
    from ml.kb import catalog
    from ml.vlm.compact_verified_action_backend import (
        CompactVerifiedActionBackend,
        _schema,
    )
    from ml.vlm.discovery import BudgetExhausted, DiscoveryBudget
    from ml.vlm.discovery_policy import TYPE_IDS

    class Inner:
        def __init__(self):
            self.requests = []

        def infer(self, request, timeout_s):
            self.requests.append(request)
            if "actions" in request.response_schema["properties"]:
                assert request.frames == frames
                return json.dumps(
                    {
                        "actions": [
                            {
                                "action": "food_or_drink_actually_consumed",
                                "insufficient_evidence": False,
                                "confidence": 0.95,
                                "start_frame": 1,
                                "peak_frame": 2,
                                "end_frame": 3,
                            }
                        ]
                    }
                )
            with module.prepare_workspace_frames(
                request.frames, view(), tmp_path, deadline=budget.deadline
            ) as prepared:
                transformed = replace(
                    request,
                    frames=prepared.frames,
                    prompt=prepared.prompt_prefix + request.prompt,
                )
                assert transformed.response_schema is request.response_schema
                assert transformed.response_schema == _schema(
                    [{"id": 1, "anchors": [1, 2, 3]}], 3
                )
                assert len(transformed.frames) == 3 and all(
                    f.path.exists() for f in transformed.frames
                )
                assert [f.timestamp_s for f in transformed.frames] == [0.4, 4.6, 9.7]
                return json.dumps(
                    {
                        "episodes": [
                            {
                                "id": 1,
                                "rows": [
                                    [1, "container_standing", "unknown"],
                                    [
                                        2,
                                        "cashier_drinks_identifiable_cup_at_mouth",
                                        "unknown",
                                    ],
                                    [3, "ordinary_work", "unknown"],
                                ],
                                "same_actor": "unknown",
                                "same_object": "unknown",
                            }
                        ]
                    }
                )

    budget = DiscoveryBudget(time.monotonic() + 5, max_calls=4)
    inner = Inner()
    backend = CompactVerifiedActionBackend(inner, catalog=catalog())
    request = InferenceRequest("synthetic source", frames, "checklist", None, TYPE_IDS)
    for _ in range(2):
        [finding] = json.loads(backend.infer_with_budget(request, budget))["findings"]
        assert (
            finding["t_start"],
            finding["t_peak"],
            finding["t_end"],
            finding["confidence"],
        ) == (0.4, 4.6, 9.7, 0.95)
    assert budget.calls_used == backend.calls_used == len(inner.requests) == 4
    with pytest.raises(BudgetExhausted):
        backend.infer_with_budget(request, budget)
    assert len(inner.requests) == 4 and not list(tmp_path.glob(".workspace-view-*"))


def test_import_never_loads_model_image_video_or_numeric_runtime():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import ml.vlm.workspace_view; "
            "assert not any(n in sys.modules for n in ('PIL','torch','mlx','cv2','numpy','transformers','ml.video'))",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
