"""Lazy bridge to ml.video's sequential frame_at, on its CV index/FPS timeline.

These times are NOT container PTS. pts_s in the shared legacy frame contract is
only a source coordinate here, equal to index/FPS, and never an actual PTS claim.
"""
import importlib
import logging
import multiprocessing
import os
import time
from pathlib import Path

from .contracts import Frame, FrameTimestamp, VideoInfo
from .frames import FrameError


def _silence_worker():
    logging.disable(logging.CRITICAL)
    with open(os.devnull, "w") as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)


def _owner_video_worker(connection):
    """Load owner CV code only inside the dedicated frame worker."""
    _silence_worker()
    try:
        while True:
            operation, args, deadline = connection.recv()
            try:
                if time.monotonic() >= deadline:
                    raise TimeoutError
                video = importlib.import_module("ml.video")
                if operation == "probe":
                    info = video.probe(args[0])
                    frames = tuple(FrameTimestamp(i, i / info.fps, i / info.fps)
                                   for i in range(info.frame_count))
                    result = VideoInfo(info.duration, frames)
                elif operation == "extract":
                    cv2 = importlib.import_module("cv2")
                    path, selected, directory = args
                    result = []
                    for slot, selected_frame in enumerate(selected):
                        if time.monotonic() >= deadline:
                            raise TimeoutError
                        image = video.frame_at(path, selected_frame.timestamp_s)
                        if time.monotonic() >= deadline:
                            raise TimeoutError
                        target = Path(directory) / f"frame-{slot}.png"
                        if image is None or not cv2.imwrite(str(target), image):
                            raise FrameError("frame_missing")
                        result.append(Frame(selected_frame.timestamp_s, target, selected_frame.timestamp_s))
                    result = tuple(result)
                else:
                    raise FrameError("invalid_frame_operation")
                if time.monotonic() >= deadline:
                    raise TimeoutError
                connection.send(("ok", result))
            except TimeoutError:
                connection.send(("timeout", None))
            except Exception:
                connection.send(("error", None))
    except (EOFError, BrokenPipeError, OSError):
        pass
    finally:
        connection.close()


class OwnerFrames:
    """Per-video owner frame source. Construction never imports OpenCV/CV."""
    timebase = "cv_index_fps"

    def __init__(self):
        self._process = None
        self._started = False
        self._connection = None
        self._closed = False

    def probe(self, video_path, timeout_s):
        return self._exchange("probe", (video_path,), timeout_s)

    def extract(self, video_path, selected, directory, timeout_s):
        return self._exchange("extract", (video_path, tuple(selected), directory), timeout_s)

    def _exchange(self, operation, args, timeout_s):
        from .frames import local_file
        from .validation import number
        if self._closed:
            raise FrameError("frames_unavailable")
        if not number(timeout_s) or timeout_s <= 0:
            raise TimeoutError("frame_timeout")
        deadline = time.monotonic() + timeout_s
        args = (local_file(args[0]), *args[1:])
        try:
            if self._process is None:
                context = multiprocessing.get_context("spawn")
                self._connection, child = context.Pipe()
                try:
                    self._process = context.Process(target=_owner_video_worker, args=(child,), daemon=True)
                    self._process.start()
                    self._started = True
                finally:
                    child.close()
            if not self._process.is_alive():
                raise FrameError("frames_unavailable")
            self._connection.send((operation, args, deadline))
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self._connection.poll(remaining):
                raise TimeoutError("frame_timeout")
            status, result = self._connection.recv()
            if status == "timeout" or time.monotonic() >= deadline:
                raise TimeoutError("frame_timeout")
            if status != "ok":
                raise FrameError("frames_unavailable")
            return result
        except TimeoutError:
            self.close()
            raise
        except (OSError, EOFError, FrameError):
            self.close()
            raise FrameError("frames_unavailable") from None

    def close(self):
        self._closed = True
        process, self._process = self._process, None
        connection, self._connection = self._connection, None
        started, self._started = self._started, False
        try:
            if process is not None:
                try:
                    if started:
                        if process.is_alive():
                            process.terminate()
                        process.join(timeout=0.1)
                        if process.is_alive():
                            process.kill()
                            process.join(timeout=0.1)
                finally:
                    process.close()
        finally:
            if connection is not None:
                connection.close()
