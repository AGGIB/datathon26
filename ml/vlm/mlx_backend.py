"""Optional Apple MLX backend in a separate Python environment and killable worker.

No MLX/Transformers imports or downloads happen in this module. Construct and
validate explicitly at startup; requests never load a model or restart a worker.
"""
from contextlib import contextmanager
from collections.abc import Mapping
import json
import math
import os
from pathlib import Path
import select
import subprocess
import threading
import time

from .contracts import InferenceRequest
from .preflight import ModelUnavailable, _validated_model_directory, enforce_offline

MAX_WIRE_BYTES = 65_536
MAX_FRAME_FILE_BYTES = 64 * 1024 * 1024
MAX_SOURCE_PIXELS = 20_000_000


def _decode_wire(data: bytes) -> dict:
    """One bounded strict JSON object; duplicate keys/nonfinite constants are protocol errors."""
    if len(data) > MAX_WIRE_BYTES or data.count(b"\n") != 1 or not data.endswith(b"\n"):
        raise ModelUnavailable("invalid_worker_response")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate_protocol_key")
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError("nonfinite_protocol_number")

    try:
        value = json.loads(data, object_pairs_hook=pairs, parse_constant=nonfinite)
    except (ValueError, RecursionError) as error:
        raise ModelUnavailable("invalid_worker_response") from error
    if not isinstance(value, dict):
        raise ModelUnavailable("invalid_worker_response")
    return value


class LocalMLXBackend:
    def __init__(self, model_path: str | Path, *, python_executable: str | Path,
                 max_new_tokens: int = 384, max_image_side: int = 640):
        if type(max_new_tokens) is not int or not 32 <= max_new_tokens <= 1024:
            raise ValueError("max_new_tokens must be in 32..1024")
        if type(max_image_side) is not int or not 224 <= max_image_side <= 896:
            raise ValueError("max_image_side must be in 224..896")
        self.model_path = model_path
        self.python_executable = Path(python_executable).absolute()
        self.max_new_tokens = max_new_tokens
        self.max_image_side = max_image_side
        self._process = None
        self._ready = False
        self._closed = False
        self._lock = threading.Lock()
        self._sequence = 0

    @contextmanager
    def _exclusive(self):
        if not self._lock.acquire(blocking=False):
            raise ModelUnavailable("runtime_busy")
        try:
            yield
        finally:
            self._lock.release()

    def _command(self, directory: Path) -> list[str]:
        return [str(self.python_executable), "-m", "ml.vlm.mlx_worker", str(directory),
                str(self.max_new_tokens), str(self.max_image_side)]

    def validate(self, timeout_s: float = 120) -> None:
        with self._exclusive():
            if self._closed:
                raise ModelUnavailable("runtime_not_prepared")
            if self._ready:
                self._require_ready()
                return
            deadline = self._deadline(timeout_s)
            try:
                directory = _validated_model_directory(self.model_path)
                if not self.python_executable.is_file():
                    raise ModelUnavailable("local_python_required")
                enforce_offline()
                env = dict(os.environ)
                env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
                env["PYTHONNOUSERSITE"] = "1"
                if time.monotonic() >= deadline:
                    raise TimeoutError("inference_timeout")
                self._process = subprocess.Popen(self._command(directory), stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env, bufsize=0)
                os.set_blocking(self._process.stdin.fileno(), False)
                os.set_blocking(self._process.stdout.fileno(), False)
                if self._read(deadline) != {"status": "ready"}:
                    raise ModelUnavailable("model_unavailable")
                self._ready = True
            except (OSError, ValueError, ModelUnavailable, TimeoutError):
                self._close()
                raise

    @staticmethod
    def _deadline(timeout_s: float) -> float:
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or not math.isfinite(timeout_s):
            raise ValueError("finite timeout required")
        if timeout_s <= 0:
            raise TimeoutError("inference_timeout")
        return time.monotonic() + timeout_s

    def _require_ready(self):
        if not self._ready or self._process is None or self._process.poll() is not None:
            self._close()
            raise ModelUnavailable("runtime_not_prepared")

    def infer(self, request: InferenceRequest, timeout_s: float) -> str:
        with self._exclusive():
            self._require_ready()
            deadline = self._deadline(timeout_s)
            self._sequence += 1
            frames = []
            if not 1 <= len(request.frames) <= 24 or not isinstance(request.prompt, str):
                raise ValueError("one to 24 frames and a text prompt required")
            if len(request.prompt) > MAX_WIRE_BYTES:
                raise ValueError("request too large")
            for frame in request.frames:
                path = Path(frame.path).resolve()
                if (not path.is_file() or path.stat().st_size > MAX_FRAME_FILE_BYTES
                        or isinstance(frame.timestamp_s, bool)
                        or not isinstance(frame.timestamp_s, (int, float))
                        or not math.isfinite(frame.timestamp_s) or frame.timestamp_s < 0):
                    raise ValueError("local frames required")
                frames.append({"path": str(path), "timestamp_s": frame.timestamp_s})
            payload = {"id": self._sequence, "prompt": request.prompt, "frames": frames}
            if request.response_schema is not None:
                if not isinstance(request.response_schema, Mapping):
                    raise ValueError("response schema must be a mapping")
                payload["response_schema"] = dict(request.response_schema)
            data = json.dumps(payload,
                              ensure_ascii=False, allow_nan=False).encode() + b"\n"
            if len(data) > MAX_WIRE_BYTES:
                raise ValueError("request too large")
            try:
                self._write(data, deadline)
                reply = self._read(deadline)
                if (set(reply) != {"id", "status", "text"} or type(reply.get("id")) is not int
                        or reply["id"] != self._sequence or reply.get("status") != "ok"):
                    raise ModelUnavailable("model_unavailable")
                result = reply.get("text")
                if not isinstance(result, str):
                    raise ModelUnavailable("invalid_worker_response")
                return result
            except (OSError, ValueError, ModelUnavailable, TimeoutError):
                self._close()
                raise

    def _write(self, data: bytes, deadline: float):
        fd = self._process.stdin.fileno()
        while data:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([], [fd], [], remaining)[1]:
                raise TimeoutError("inference_timeout")
            try:
                count = os.write(fd, data[:4096])
            except BlockingIOError:
                continue
            if not count:
                raise ModelUnavailable("worker_closed")
            data = data[count:]

    def _read(self, deadline: float) -> dict:
        fd = self._process.stdout.fileno()
        data = bytearray()
        while b"\n" not in data:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([fd], [], [], remaining)[0]:
                raise TimeoutError("inference_timeout")
            try:
                chunk = os.read(fd, 4096)
            except BlockingIOError:
                continue
            if not chunk:
                raise ModelUnavailable("worker_closed")
            data.extend(chunk)
            if len(data) > MAX_WIRE_BYTES:
                raise ModelUnavailable("worker_response_too_large")
        result = _decode_wire(bytes(data))
        if time.monotonic() >= deadline:
            raise TimeoutError("inference_timeout")
        return result

    def close(self) -> None:
        with self._exclusive():
            self._close()

    def _close(self):
        self._closed = True
        self._ready = False
        process, self._process = self._process, None
        if process is not None:
            try:
                if process.poll() is None:
                    try:
                        process.terminate()
                    except ProcessLookupError:
                        pass  # The child can exit between poll and terminate.
                    try:
                        process.wait(timeout=0.5)
                    except subprocess.TimeoutExpired:
                        try:
                            process.kill()
                        except ProcessLookupError:
                            pass
                        process.wait(timeout=0.5)
            finally:
                for stream in (process.stdin, process.stdout):
                    if stream is not None:
                        stream.close()
