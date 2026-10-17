"""Actual local Transformers multimodal adapter; never a cloud API client."""
from contextlib import contextmanager
import json
import logging
import multiprocessing
import os
import threading
import time
from pathlib import Path
from typing import Any
from .contracts import InferenceRequest
from .preflight import ModelUnavailable, enforce_offline, _validated_model_directory


def _token_limit(value: int) -> int:
    if type(value) is not int or not 1 <= value <= 128:
        raise ValueError("max_new_tokens must be an integer in 1..128")
    return value


class _LocalRuntime:
    """Loaded inside the inference worker only. Imports are deliberately lazy."""
    def __init__(self, model_path: str | Path, device: str, max_new_tokens: int):
        enforce_offline()
        directory = _validated_model_directory(model_path)
        import torch
        from PIL import Image
        import transformers
        self.torch = torch
        self.Image = Image
        self.max_new_tokens = _token_limit(max_new_tokens)
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        self.processor = transformers.AutoProcessor.from_pretrained(
            str(directory), local_files_only=True, trust_remote_code=False)
        if config.get("model_type") == "qwen2_5_vl":
            model_class = transformers.Qwen2_5_VLForConditionalGeneration
        else:
            model_class = transformers.AutoModelForImageTextToText
        self.model, info = model_class.from_pretrained(str(directory), local_files_only=True,
            trust_remote_code=False, use_safetensors=True, torch_dtype="auto", device_map=device,
            attn_implementation="eager", output_loading_info=True)
        if any(info.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")):
            raise ModelUnavailable("checkpoint_incomplete")
        self.model.eval()

    def infer(self, request: InferenceRequest) -> str:
        enforce_offline()
        images = []
        try:
            content: list[dict[str, Any]] = [{"type": "text", "text": request.prompt}]
            for frame in request.frames:
                with self.Image.open(frame.path) as source:
                    image = source.convert("RGB").copy()
                images.append(image)
                content.extend([{"type": "text", "text": f"Frame timestamp_s={frame.timestamp_s:.9f}"},
                                {"type": "image", "image": image}])
            messages = [{"role": "user", "content": content}]
            template = self.processor.apply_chat_template(messages, tokenize=False,
                                                          add_generation_prompt=True)
            inputs = self.processor(text=[template], images=images, padding=True, return_tensors="pt")
            inputs = inputs.to(self.model.device)
            with self.torch.inference_mode():
                generated = self.model.generate(**inputs, do_sample=False,
                                                max_new_tokens=self.max_new_tokens)
            trimmed = [output[len(prompt):] for prompt, output in zip(inputs["input_ids"], generated)]
            return self.processor.batch_decode(trimmed, skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False)[0]
        finally:
            for image in images:
                image.close()


def _worker_main(connection, model_path, device, max_new_tokens):
    enforce_offline()
    # Third-party libraries may print input paths/errors; keep the worker silent.
    logging.disable(logging.CRITICAL)
    with open(os.devnull, "w") as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
        runtime = None
        try:
            while True:
                request = connection.recv()
                try:
                    if runtime is None:
                        runtime = _LocalRuntime(model_path, device, max_new_tokens)
                    connection.send(("ok", "" if request is None else runtime.infer(request)))
                except Exception:
                    connection.send(("error", "model_unavailable"))
        except (EOFError, BrokenPipeError, OSError):
            pass
        finally:
            connection.close()


class LocalTransformersBackend:
    """One cached model in a killable process; concurrent callers fail closed busy."""
    def __init__(self, model_path: str | Path, *, device: str = "cpu", max_new_tokens: int = 128):
        enforce_offline()
        self.model_path = model_path
        self.device = device
        self.max_new_tokens = _token_limit(max_new_tokens)
        self._process = None
        self._started = False
        self._connection = None

        self._ready = False
        self._closed = False
        self._lock = threading.Lock()

    @contextmanager
    def _exclusive(self):
        # Never queue a job behind another video's inference or touch its Pipe.
        if not self._lock.acquire(blocking=False):
            raise ModelUnavailable("runtime_busy")
        try:
            yield
        finally:
            self._lock.release()

    def validate(self, timeout_s: float = 60) -> None:
        """Startup only. A failed/closed worker requires a new backend instance."""
        with self._exclusive():
            if self._closed:
                raise ModelUnavailable("runtime_not_prepared")
            if self._ready:
                self._require_ready()
                return
            self._exchange(None, timeout_s)
            self._ready = True

    def _require_ready(self) -> None:
        if not self._ready:
            raise ModelUnavailable("runtime_not_prepared")
        if self._process is None or not self._process.is_alive():
            self._close()
            raise ModelUnavailable("runtime_not_prepared")

    def infer(self, request: InferenceRequest, timeout_s: float) -> str:
        with self._exclusive():
            self._require_ready()
            return self._exchange(request, timeout_s)

    def _exchange(self, request: InferenceRequest | None, timeout_s: float) -> str:
        enforce_offline()
        deadline = time.monotonic() + timeout_s
        if timeout_s <= 0:
            raise TimeoutError("inference_timeout")
        try:
            if self._process is None:
                directory = _validated_model_directory(self.model_path)
                context = multiprocessing.get_context("spawn")
                self._connection, child = context.Pipe()
                try:
                    self._process = context.Process(target=_worker_main,
                        args=(child, str(directory), self.device, self.max_new_tokens), daemon=True)
                    self._process.start()
                    self._started = True
                finally:
                    child.close()
            assert self._connection is not None
            self._connection.send(request)
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self._connection.poll(remaining):
                raise TimeoutError("inference_timeout")
            status, result = self._connection.recv()
            if status != "ok":
                raise ModelUnavailable("model_unavailable")
            return result
        except TimeoutError:
            self._close()
            raise
        except (OSError, EOFError, ModelUnavailable):
            self._close()
            raise ModelUnavailable("model_unavailable") from None

    def close(self) -> None:
        with self._exclusive():
            self._close()

    def _close(self) -> None:
        """Cleanup while holding the exclusive worker lease."""
        self._closed = True
        self._ready = False
        process, self._process = self._process, None
        connection, self._connection = self._connection, None
        started, self._started = self._started, False
        try:
            if process is not None:
                try:
                    if started:
                        if process.is_alive():
                            process.terminate()
                        process.join(timeout=0.5)
                        if process.is_alive():
                            process.kill()
                            process.join(timeout=0.5)
                finally:
                    process.close()
        finally:
            if connection is not None:
                connection.close()
