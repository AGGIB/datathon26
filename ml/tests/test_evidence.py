import base64
from pathlib import Path

import cv2
import numpy as np
import pytest

import ml.evidence as evidence
from ml.candidates import Candidate
from ml.evidence import MAX_FRAME_BYTES, draw_boxes, encode_jpeg_b64, frames_for, render_candidates, wanted_times
from ml.video import probe, sample_frames

REPO = Path(__file__).resolve().parents[2]
X11 = REPO / "data" / "open" / "videos" / "Х-11.mov"  # VFR-like h264 where seeking returned wrong frames
BOX = (20.0, 20.0, 80.0, 80.0)


def decode(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64, validate=True)
    assert raw[:3] == b"\xff\xd8\xff", "not a JPEG"
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    assert img is not None
    return img


def cand(peak=1.0, times=None, boxes=(BOX,), type_id="cash_left_on_desk_when_leaving"):
    times = times or (peak,)
    return Candidate(type_id=type_id, t_start=min(times), t_end=max(times), t_peak=peak, confidence=0.9,
                     observation="пачка на столе", evidence_times=times, highlight_boxes=boxes)


def brightness_frames(times, step=10):
    """Synthetic decoded frames: brightness encodes the timestamp."""
    return {t: np.full((120, 160, 3), int(t * step) % 255, np.uint8) for t in times}


@pytest.fixture
def clip(tmp_path):
    """4 s at 10 fps, 160x120; brightness grows with the frame index."""
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    for i in range(40):
        writer.write(np.full((120, 160, 3), i * 6, dtype=np.uint8))
    writer.release()
    return path


# --- encoding and drawing ---------------------------------------------------------------------------


def test_encode_returns_plain_base64_jpeg():
    b64 = encode_jpeg_b64(np.zeros((48, 64, 3), np.uint8))
    assert not b64.startswith("data:")
    assert decode(b64).shape == (48, 64, 3)


def test_encode_respects_size_cap_on_noisy_large_frame():
    noise = np.random.default_rng(0).integers(0, 256, (1080, 1920, 3), dtype=np.uint8)  # worst case for JPEG
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


# --- choosing frames ---------------------------------------------------------------------------------


def test_wanted_times_cover_evidence_and_peak_neighbours_inside_video():
    times = wanted_times([cand(peak=1.0, times=(1.0, 2.0)), cand(peak=3.9, times=(3.9,))], duration=4.0)
    assert times == [0.5, 1.0, 1.5, 2.0, 3.4, 3.9, 4.0]


def test_boxes_are_drawn_on_the_peak_frame_only():
    c = cand(peak=1.0, times=(1.0, 2.0, 3.0))
    frames = {t: np.zeros((120, 160, 3), np.uint8) for t in (1.0, 2.0, 3.0)}
    rendered = frames_for(c, frames, duration=4.0)
    assert [f["timestamp_sec"] for f in rendered] == [1.0, 2.0, 3.0]
    peak, *others = (decode(f["image_base64"]) for f in rendered)
    assert peak.max() > 100  # amber outline
    assert all(img.max() < 30 for img in others)  # black frames stay black: no boxes off-peak


def test_missing_peak_frame_is_replaced_by_its_neighbour_not_by_interval_edges():
    c = Candidate(type_id="x", t_start=0.0, t_end=4.0, t_peak=2.0, confidence=0.9, observation="пачка",
                  evidence_times=(2.0, 3.0), highlight_boxes=(BOX,))
    frames = brightness_frames([0.0, 1.5, 2.5, 3.0, 4.0])  # 2.0 itself failed to decode
    rendered = frames_for(c, frames, duration=4.0)
    assert [f["timestamp_sec"] for f in rendered] == [1.5, 3.0]


def test_unencodable_peak_frame_falls_back_to_neighbour(monkeypatch):
    real_encode = evidence.encode_jpeg_b64
    calls = []

    def flaky(frame, max_bytes=MAX_FRAME_BYTES):
        calls.append(1)
        if len(calls) == 1:
            raise ValueError("does not fit")
        return real_encode(frame, max_bytes)

    monkeypatch.setattr(evidence, "encode_jpeg_b64", flaky)
    rendered = frames_for(cand(peak=1.0), brightness_frames([0.5, 1.0, 1.5]), duration=4.0)
    assert [f["timestamp_sec"] for f in rendered] == [0.5]


def test_no_usable_frame_gives_empty_list():
    assert frames_for(cand(peak=1.0), {}, duration=4.0) == []


def test_render_candidates_returns_frames_from_the_right_times(clip):
    c = cand(peak=1.0, times=(1.0, 2.0, 3.0))
    [rendered] = render_candidates(clip, [c], duration=4.0).values()
    assert [f["timestamp_sec"] for f in rendered] == [1.0, 2.0, 3.0]
    images = [decode(f["image_base64"]) for f in rendered]
    centre = [img[60:110, 90:150].mean() for img in images]  # away from the box drawn on the first frame
    assert centre[0] < centre[1] < centre[2]  # later timestamps are brighter in the synthetic clip


def test_render_candidates_decodes_the_video_once(clip, monkeypatch):
    calls = []
    real = evidence.frames_at
    monkeypatch.setattr(evidence, "frames_at", lambda path, times: calls.append(times) or real(path, times))
    render_candidates(clip, [cand(peak=1.0), cand(peak=3.0, type_id="door_left_open")], duration=4.0)
    assert len(calls) == 1


def test_render_candidates_clamps_times_to_the_analysed_part(clip):
    [rendered] = render_candidates(clip, [cand(peak=3.9)], duration=2.05).values()
    assert rendered[0]["timestamp_sec"] == 2.0


@pytest.mark.parametrize("error", [ValueError("cannot open video"), OSError("disk"), cv2.error("decoder")])
def test_failed_decode_pass_drops_the_evidence_instead_of_the_job(clip, monkeypatch, caplog, error):
    def broken(path, times):
        raise error

    monkeypatch.setattr(evidence, "frames_at", broken)
    candidates = [cand(peak=1.0), cand(peak=3.0, type_id="door_left_open")]
    with caplog.at_level("ERROR", logger="ml.evidence"):
        rendered = render_candidates(clip, candidates, duration=4.0)
    assert rendered == {c: [] for c in candidates}
    assert "decode" in caplog.text


def test_render_candidates_with_nothing_to_render_does_not_touch_the_video(tmp_path):
    assert render_candidates(tmp_path / "missing.mov", [], duration=1.0) == {}


# --- real video ---------------------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(not X11.exists(), reason="data/open link is missing")
def test_real_vfr_video_evidence_matches_sampled_frame():
    t = 10.0
    info = probe(X11)
    target_idx = round(t * info.fps)
    reference = next(frame for ts, frame in sample_frames(X11, info.fps) if round(ts * info.fps) == target_idx)
    wrong = next(frame for ts, frame in sample_frames(X11, 1.0) if ts >= 18.5)  # where seeking used to land

    [rendered] = render_candidates(X11, [cand(peak=t, boxes=())], duration=info.duration).values()
    assert rendered[0]["timestamp_sec"] == t
    got = decode(rendered[0]["image_base64"])

    def diff(a, b):
        small = (a.shape[1] // 4, a.shape[0] // 4)
        return float(np.abs(cv2.resize(a, small).astype(int) - cv2.resize(b, small).astype(int)).mean())

    assert diff(got, reference) < 4.0  # JPEG noise only
    assert diff(got, reference) < diff(got, wrong)
