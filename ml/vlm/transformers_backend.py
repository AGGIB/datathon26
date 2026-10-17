"""Actual local Transformers multimodal adapter; never a cloud API client."""
import json
import logging
import multiprocessing
import os
import time
from pathlib import Path
from typing import Any
from .contracts import InferenceRequest
from .preflight import ModelUnavailable, enforce_offline, _validated_model_directory


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
        self.max_new_tokens = max_new_tokens
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
    """One cached model in a killable local process. Not safe for concurrent callers."""
    def __init__(self, model_path: str | Path, *, device: str = "cpu", max_new_tokens: int = 512):
        enforce_offline()
        self.model_path = model_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._process = None
        self._connection = None

    def validate(self, timeout_s: float = 60) -> None:
        """Actually load local model/processor; no inference or accuracy claim."""
        self._exchange(None, timeout_s)

    def infer(self, request: InferenceRequest, timeout_s: float) -> str:
        return self._exchange(request, timeout_s)

    def _exchange(self, request: InferenceRequest | None, timeout_s: float) -> str:
        enforce_offline()
        directory = _validated_model_directory(self.model_path)
        deadline = time.monotonic() + timeout_s
        if timeout_s <= 0:
            raise TimeoutError("inference_timeout")
        try:
            if self._process is None:
                context = multiprocessing.get_context("spawn")
                self._connection, child = context.Pipe()
                self._process = context.Process(target=_worker_main,
                    args=(child, str(directory), self.device, self.max_new_tokens), daemon=True)
                self._process.start()
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
            self.close()
            raise
        except (OSError, EOFError, ModelUnavailable):
            self.close()
            raise ModelUnavailable("model_unavailable") from None

    def close(self) -> None:
        if self._process is not None:
            if self._process.is_alive():
                self._process.terminate()
            self._process.join(timeout=0.5)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=0.5)
            self._process.close()
            self._process = None
        if self._connection is not None:
            self._connection.close()
            self._connection = None
