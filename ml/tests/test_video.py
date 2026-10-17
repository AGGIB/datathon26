import cv2
import numpy as np
import pytest

from ml.video import probe, sample_frames


@pytest.fixture
def synthetic_video(tmp_path):
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for i in range(30):  # 3 seconds at 10 fps; brightness encodes frame index
        writer.write(np.full((48, 64, 3), i * 8, dtype=np.uint8))
    writer.release()
    return path


def test_probe_reads_metadata(synthetic_video):
    info = probe(synthetic_video)
    assert info.fps == pytest.approx(10, abs=0.1)
    assert info.frame_count == 30
    assert info.duration == pytest.approx(3.0, abs=0.15)
    assert (info.width, info.height) == (64, 48)


def test_sample_frames_downsamples_with_correct_timestamps(synthetic_video):
    frames = list(sample_frames(synthetic_video, fps_out=2))
    times = [t for t, _ in frames]
    assert times == pytest.approx([0.0, 0.5, 1.0, 1.5, 2.0, 2.5], abs=0.01)
    assert frames[0][1].shape == (48, 64, 3)
    # frame at t=1.0 is source frame 10 -> brightness ~80
    assert abs(int(frames[2][1].mean()) - 80) < 12


def test_probe_rejects_unreadable_file(tmp_path):
    bad = tmp_path / "bad.mov"
    bad.write_bytes(b"not a video")
    with pytest.raises(ValueError):
        probe(bad)


def test_probe_rejects_oversized_frames(synthetic_video):
    with pytest.raises(ValueError):
        probe(synthetic_video, max_pixels=64 * 48 - 1)
