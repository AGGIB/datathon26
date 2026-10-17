import base64

import cv2
import numpy as np
import pytest

from ml.evidence import MAX_FRAME_BYTES, draw_boxes, encode_jpeg_b64, render_evidence


def decode(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64, validate=True)
    assert raw[:3] == b"\xff\xd8\xff", "not a JPEG"
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    assert img is not None
    return img


@pytest.fixture
def clip(tmp_path):
    """4 s at 10 fps, 160x120; brightness grows with the frame index."""
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    for i in range(40):
        writer.write(np.full((120, 160, 3), i * 6, dtype=np.uint8))
    writer.release()
    return path


def test_encode_returns_plain_base64_jpeg():
    frame = np.zeros((48, 64, 3), np.uint8)
    b64 = encode_jpeg_b64(frame)
    assert not b64.startswith("data:")
    assert decode(b64).shape == (48, 64, 3)


def test_encode_respects_size_cap_on_noisy_large_frame():
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8)  # worst case for JPEG
    b64 = encode_jpeg_b64(noise)
    assert len(b64) <= MAX_FRAME_BYTES  # the cap applies to the base64 text that goes into JSON
    h, w = decode(b64).shape[:2]
    assert w / h == pytest.approx(1920 / 1080, rel=0.02)


def test_encode_rejects_empty_frame():
    with pytest.raises(ValueError):
        encode_jpeg_b64(np.zeros((0, 0, 3), np.uint8))


def test_draw_boxes_does_not_modify_input_and_marks_pixels():
    frame = np.zeros((100, 100, 3), np.uint8)
    out = draw_boxes(frame, [(10.0, 10.0, 60.0, 60.0)])
    assert frame.sum() == 0
    assert out.sum() > 0
    assert out[35, 35].sum() == 0  # inside stays untouched: only the outline is drawn


@pytest.mark.parametrize(
    "box",
    [
        (-50.0, -50.0, 20.0, 20.0),  # sticks out top-left
        (80.0, 80.0, 500.0, 500.0),  # sticks out bottom-right
        (-10.0, 40.0, 110.0, 60.0),  # wider than the frame
        (200.0, 200.0, 300.0, 300.0),  # fully outside
        (50.0, 50.0, 50.0, 50.0),  # degenerate
        (60.0, 60.0, 40.0, 40.0),  # inverted corners
        (float("nan"), 0.0, 10.0, 10.0),  # garbage
    ],
)
def test_draw_boxes_survives_edge_cases(box):
    out = draw_boxes(np.zeros((100, 100, 3), np.uint8), [box])
    assert out.shape == (100, 100, 3)


def test_render_evidence_returns_frames_at_requested_times(clip):
    frames = render_evidence(clip, [1.0, 2.0, 3.0], boxes=[(20.0, 20.0, 80.0, 80.0)])
    assert [f["timestamp_sec"] for f in frames] == [1.0, 2.0, 3.0]
    images = [decode(f["image_base64"]) for f in frames]
    # later timestamps are brighter in the synthetic clip -> frames really come from those times
    assert images[0].mean() < images[1].mean() < images[2].mean()


def test_render_evidence_clamps_times_beyond_video_end(clip):
    frames = render_evidence(clip, [-1.0, 99.0])
    assert len(frames) == 2
    assert frames[0]["timestamp_sec"] == 0.0
    assert 3.5 <= frames[1]["timestamp_sec"] <= 4.0
    for f in frames:
        decode(f["image_base64"])


def test_render_evidence_never_rounds_past_the_last_frame(tmp_path):
    # 33 frames at 30 fps: last frame at 1.0667 s, which naive rounding turns into 1.1 s (no frame there)
    path = tmp_path / "odd.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
    for _ in range(33):
        writer.write(np.full((48, 64, 3), 128, dtype=np.uint8))
    writer.release()
    frames = render_evidence(path, [5.0])
    assert len(frames) == 1
    assert frames[0]["timestamp_sec"] <= 32 / 30


def test_render_evidence_steps_back_when_seek_fails(clip, monkeypatch):
    import ml.evidence as evidence

    real_frame_at = evidence.frame_at
    monkeypatch.setattr(evidence, "frame_at", lambda path, t: None if t > 2.0 else real_frame_at(path, t))
    frames = render_evidence(clip, [2.4])
    assert len(frames) == 1
    assert frames[0]["timestamp_sec"] <= 2.0


def test_render_evidence_dedupes_and_caps_at_three(clip):
    frames = render_evidence(clip, [1.0, 1.0, 2.0, 3.0, 3.5])
    assert [f["timestamp_sec"] for f in frames] == [1.0, 2.0, 3.0]
