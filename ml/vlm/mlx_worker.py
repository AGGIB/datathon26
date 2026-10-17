"""Private local JSON-lines worker; run with the optional MLX Python environment."""
import json
import math
from pathlib import Path
import os
import sys

from .preflight import _validated_model_directory, enforce_offline
from .mlx_backend import MAX_FRAME_FILE_BYTES, MAX_SOURCE_PIXELS, MAX_WIRE_BYTES, _decode_wire


def main():
    enforce_offline()
    # Keep library messages and paths out of the protocol and parent logs.
    protocol = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
    with open(os.devnull, "w") as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)

    def send(value):
        line = json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n"
        if len(line.encode("utf-8")) > MAX_WIRE_BYTES:
            raise ValueError("worker_response_too_large")
        protocol.write(line)

    try:
        directory = _validated_model_directory(sys.argv[1])
        tokens, side = int(sys.argv[2]), int(sys.argv[3])
        if not 32 <= tokens <= 1024 or not 224 <= side <= 896:
            raise ValueError("invalid worker limits")
        config = json.loads((directory / "config.json").read_text())
        if config.get("model_type") != "qwen3_vl":
            raise ValueError("unsupported checkpoint")
        import mlx.core as mx
        from mlx_vlm import generate, load
        from mlx_vlm.prompt_utils import apply_chat_template
        from PIL import Image

        model, processor = load(str(directory), trust_remote_code=False, local_files_only=True, strict=True)
        if hasattr(processor, "image_processor"):
            processor.image_processor.max_pixels = side * side
            if hasattr(processor.image_processor, "min_pixels"):
                processor.image_processor.min_pixels = min(processor.image_processor.min_pixels, side * side)
        send({"status": "ready"})
        while line := sys.stdin.buffer.readline(MAX_WIRE_BYTES + 1):
            request = _decode_wire(line)
            if (set(request) not in ({"id", "prompt", "frames"}, {"id", "prompt", "frames", "response_schema"})
                    or type(request["id"]) is not int
                    or request["id"] <= 0 or not isinstance(request["prompt"], str)
                    or not isinstance(request["frames"], list) or not 1 <= len(request["frames"]) <= 24
                    or ("response_schema" in request and not isinstance(request["response_schema"], dict))):
                raise ValueError("invalid worker request")
            images = []
            try:
                times = []
                for frame in request["frames"]:
                    if (not isinstance(frame, dict) or set(frame) != {"path", "timestamp_s"}
                            or not isinstance(frame["path"], str)
                            or isinstance(frame["timestamp_s"], bool)
                            or not isinstance(frame["timestamp_s"], (int, float))
                            or not math.isfinite(frame["timestamp_s"]) or frame["timestamp_s"] < 0):
                        raise ValueError("invalid frame")
                    path = Path(frame["path"])
                    if not path.is_file() or path.stat().st_size > MAX_FRAME_FILE_BYTES:
                        raise ValueError("invalid local frame")
                    with Image.open(path) as source:
                        if source.width * source.height > MAX_SOURCE_PIXELS:
                            raise ValueError("source image too large")
                        image = source.convert("RGB")
                        image.thumbnail((side, side))
                        images.append(image)
                    times.append(f"Image {len(images)}: timestamp_s={frame['timestamp_s']:.9f}")
                # Frame times come from the source contract, rather than model-written timestamps.
                text = "Image order and source times:\n" + "\n".join(times) + "\n" + request["prompt"]
                prompt = apply_chat_template(processor, config, text, num_images=len(images))
                generation_options = {}
                if "response_schema" in request:
                    from mlx_vlm.structured import build_json_schema_logits_processor

                    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
                    generation_options["logits_processors"] = [
                        build_json_schema_logits_processor(tokenizer, request["response_schema"])]
                output = generate(model, processor, prompt, images, max_tokens=tokens, temperature=0.0,
                                  verbose=False, **generation_options)
                text = output.text
                if not isinstance(text, str):
                    raise ValueError("invalid generation result")
                mx.synchronize()
                send({"id": request["id"], "status": "ok", "text": text})
            finally:
                for image in images:
                    image.close()
                mx.clear_cache()
    except Exception:
        send({"status": "error"})
    finally:
        protocol.close()


if __name__ == "__main__":
    main()
