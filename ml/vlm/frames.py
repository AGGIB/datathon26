"""ffprobe PTS discovery and ffmpeg extraction; no audio or network protocols."""
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Sequence
from .contracts import Frame, FrameTimestamp, VideoInfo


class FrameError(RuntimeError):
    pass


def local_file(path: str | Path) -> Path:
    raw = str(path)
    if "://" in raw or raw.replace("\\", "/").startswith("//"):
        raise FrameError("local_file_required")
    resolved = Path(path).resolve()
    if str(resolved).replace("\\", "/").startswith("//"):
        raise FrameError("local_file_required")
    if not resolved.is_file():
        raise FrameError("video_missing")
    return resolved


def select_timestamps(frames: Sequence[FrameTimestamp], limit: int) -> tuple[FrameTimestamp, ...]:
    if len(frames) <= limit:
        return tuple(frames)
    return tuple(frames[round(i * (len(frames) - 1) / (limit - 1))] for i in range(limit))


class FFmpegFrames:
    def __init__(self, ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe"):
        self.ffmpeg = ffmpeg
        self.ffprobe = ffprobe

    @staticmethod
    def _run(command: list[str], timeout_s: float) -> bytes:
        if timeout_s <= 0:
            raise TimeoutError("frame_timeout")
        try:
            return subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True,
                                  timeout=timeout_s, check=True).stdout
        except subprocess.TimeoutExpired:
            raise TimeoutError("frame_timeout") from None
        except (OSError, subprocess.CalledProcessError):
            raise FrameError("frame_command_failed") from None

    def probe(self, video_path: str | Path, timeout_s: float) -> VideoInfo:
        video = local_file(video_path)
        output = self._run([self.ffprobe, "-v", "error", "-protocol_whitelist", "file,pipe",
            "-select_streams", "v:0", "-show_frames", "-show_streams", "-show_format", "-show_entries",
            "frame=best_effort_timestamp_time,duration_time:stream=start_time,duration:format=start_time,duration",
            "-of", "json", str(video)], timeout_s)
        try:
            data = json.loads(output)
            raw_frames = data["frames"]
            pts = [float(frame["best_effort_timestamp_time"]) for frame in raw_frames]
            origin = float(data["format"]["start_time"])
            if (not pts or not math.isfinite(origin) or not all(math.isfinite(t) for t in pts)
                    or pts != sorted(pts) or pts[0] < origin):
                raise ValueError
            frames = tuple(FrameTimestamp(i, t - origin, t) for i, t in enumerate(pts))
            last_duration = float(raw_frames[-1].get("duration_time", 0))
            stream = data.get("streams", [{}])[0]
            stream_start = float(stream.get("start_time", origin))
            stream_duration = float(stream.get("duration", 0))
            if (not all(math.isfinite(t) for t in (last_duration, stream_start, stream_duration))
                    or min(last_duration, stream_duration) < 0):
                raise ValueError
            # Container duration can include absolute PTS offsets (e.g. Matroska).
            # Use the video endpoint, shifted by the same origin as every frame.
            duration = max(frames[-1].timestamp_s + last_duration,
                           stream_start - origin + stream_duration if stream_duration else 0)
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError
            return VideoInfo(duration, frames)
        except (ValueError, KeyError, TypeError, IndexError):
            raise FrameError("timestamps_unavailable") from None

    def extract(self, video_path: str | Path, selected: Sequence[FrameTimestamp],
                directory: Path, timeout_s: float) -> tuple[Frame, ...]:
        video = local_file(video_path)
        deadline = time.monotonic() + timeout_s
        result = []
        for slot, frame in enumerate(selected):
            target = Path(directory) / f"frame-{slot}.png"
            filters = f"select=eq(n\\,{frame.index}),scale=640:640:force_original_aspect_ratio=decrease"
            self._run([self.ffmpeg, "-nostdin", "-v", "error", "-y", "-protocol_whitelist",
                "file,pipe", "-i", str(video), "-map", "0:v:0", "-an", "-sn", "-dn",
                "-vf", filters, "-frames:v", "1", "-fps_mode", "vfr", str(target)],
                deadline - time.monotonic())
            if not target.is_file() or target.stat().st_size == 0:
                raise FrameError("frame_missing")
            result.append(Frame(frame.timestamp_s, target, frame.pts_s))
        return tuple(result)
