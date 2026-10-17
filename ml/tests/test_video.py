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


def test_probe_rejects_playlist_disguised_as_mov(tmp_path):
    fake = tmp_path / "x.mov"
    fake.write_bytes(b"#EXTM3U\n#EXT-X-MEDIA-SEQUENCE:0\nfile:///etc/passwd\n")
    with pytest.raises(ValueError, match="container"):
        probe(fake)


def test_frames_at_returns_exact_frames_in_any_order(synthetic_video):
    from ml.video import frames_at

    got = frames_at(synthetic_video, [2.0, 0.5, 1.0])
    # brightness encodes the source frame index (i * 8) at 10 fps
    assert abs(int(got[0.5].mean()) - 40) < 12
    assert abs(int(got[1.0].mean()) - 80) < 12
    assert abs(int(got[2.0].mean()) - 160) < 12


def test_frames_at_clamps_times_outside_the_video(synthetic_video):
    from ml.video import frames_at

    got = frames_at(synthetic_video, [-1.0, 99.0])
    assert abs(int(got[-1.0].mean()) - 0) < 12
    assert abs(int(got[99.0].mean()) - 29 * 8) < 12


def test_frame_at_matches_sample_frames_timebase(synthetic_video):
    from ml.video import frame_at

    sampled = dict(sample_frames(synthetic_video, fps_out=2))
    assert np.array_equal(frame_at(synthetic_video, 1.5), sampled[1.5])
