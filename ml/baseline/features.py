"""Lazy numeric caches and strictly local SigLIP feature extraction."""
import json
import math
import os
import queue
import re
import shutil
import subprocess
import threading
from collections import deque
from contextlib import closing
from pathlib import Path

from .core import FEATURE_SPEC, artifact_path, read_json


def extract_video(video, weights, output, video_id=None, fps=1.0, max_frames=3600,
                  batch_size=8, device="cpu", cpu_threads=2, seed=0, ffmpeg="ffmpeg",
                  timeout_s=600.0, artifact_root=None):
    target = artifact_path(output, root=artifact_root)
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    ffmpeg_command(video, fps=fps, max_frames=max_frames, ffmpeg=ffmpeg)  # Fail before loading weights.
    source = Path(video)
    identifier = source.name if video_id is None else video_id
    if not isinstance(identifier, str) or not identifier.strip():
        raise ValueError("video_id must be a nonempty literal string")
    encoder = SiglipEncoder(weights, device=device, cpu_threads=cpu_threads, seed=seed)
    with closing(decode_frames(source, fps=fps, max_frames=max_frames, ffmpeg=ffmpeg, timeout_s=timeout_s)) as frames:
        times, embeddings = embed_frames(frames, encoder, batch_size=batch_size, max_frames=max_frames)
        # Resume the generator once after the final yield to check ffmpeg's exit status.
        if next(frames, None) is not None:
            raise RuntimeError("Decoder exceeded max_frames; refusing incomplete extraction")
    extraction = {"fps_requested": fps, "max_frames": max_frames, "batch_size": batch_size,
                  "n_frames": len(times), "cap_reached": len(times) == max_frames,
                  "autorotate": True, "time_origin": "container-relative PTS: copyts + start_at_zero",
                  "device": device, "seed": seed}
    stat = source.stat()
    save_cache(target, identifier, times, embeddings,
               {"extraction": extraction, "source_filename": source.name,
                "source_size_bytes": stat.st_size, "source_mtime_ns": stat.st_mtime_ns})
    return {"status": "ok", "video_id": identifier, "cache": str(target),
            "n_frames": len(times), "embedding_dim": embeddings.shape[1],
            "first_pts_s": float(times[0]), "last_pts_s": float(times[-1]),
            "cap_reached": extraction["cap_reached"]}


def _local_path(value):
    raw = str(value)
    if "://" in raw or raw.replace("\\", "/").startswith("//"):
        raise ValueError("Only local paths are allowed; network shares are forbidden")
    path = Path(value).resolve()
    if str(path).replace("\\", "/").startswith("//"):
        raise ValueError("Only local paths are allowed; network shares are forbidden")
    return path


def local_weights(weights):
    """Only an existing local Google SigLIP base/16/224 safetensors snapshot is accepted."""
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY", "DO_NOT_TRACK"):
        os.environ[key] = "1"
    path = _local_path(weights)
    if not path.is_dir():
        raise ValueError("weights must name an existing local directory, not a Hub ID/URL")
    config_path = path / "config.json"
    if not config_path.is_file():
        raise ValueError("Local weights are incomplete: config.json missing")
    config = read_json(config_path)
    vision = config.get("vision_config", {})
    # Official snapshot omits default-valued SiglipVisionConfig fields.
    expected = {"image_size": 224, "patch_size": 16, "hidden_size": 768, "num_hidden_layers": 12,
                "num_attention_heads": 12, "intermediate_size": 3072}
    if config.get("model_type") != "siglip" or any(vision.get(k, v) != v for k, v in expected.items()):
        raise ValueError("Expected Google SigLIP base patch16 224 vision configuration")
    if not (path / "preprocessor_config.json").is_file():
        raise ValueError("Local weights are incomplete: preprocessor_config.json missing")
    if not (path / "model.safetensors").is_file():
        index_path = path / "model.safetensors.index.json"
        if not index_path.is_file():
            raise ValueError("Local weights are incomplete: safetensors missing; no download or pickle fallback")
        index = read_json(index_path)
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("Local safetensors index requires a nonempty weight_map object")
        shards = set()
        for tensor, shard in weight_map.items():
            if (not isinstance(tensor, str) or not tensor.strip()
                    or not isinstance(shard, str) or not shard.strip()):
                raise ValueError("weight_map must map nonempty tensor names to relative shard filenames")
            raw = shard.replace("\\", "/")
            if (shard != shard.strip() or raw.startswith("/") or ":" in raw or "\x00" in raw
                    or ".." in raw.split("/") or Path(shard).suffix != ".safetensors"):
                raise ValueError("Only relative local safetensors shard paths without traversal are allowed")
            target = _local_path(path / shard)
            if not target.is_relative_to(path) or target.suffix != ".safetensors":
                raise ValueError("Safetensors shard must resolve inside the local weights directory")
            shards.add(target)
        # Validate every reference before probing any shard file or importing a loader.
        if any(not shard.is_file() for shard in shards):
            raise ValueError("Local weights are incomplete: safetensors shard missing; no download or fallback")
    return path


def numpy_module():
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for numeric baseline operations; see requirements-ml-baseline.txt") from exc
    return np


def validate_cache(video_id, timestamps, embeddings, metadata):
    np = numpy_module()
    if not isinstance(video_id, str) or not video_id.strip():
        raise ValueError("Cache video_id must be a nonempty literal string")
    times, values = np.asarray(timestamps), np.asarray(embeddings)
    if times.dtype.kind not in "fiu" or values.dtype.kind not in "fiu":
        raise ValueError("Caches require numeric timestamps and embeddings, never object/string arrays")
    times, values = times.astype(np.float64), values.astype(np.float32)
    if (times.ndim != 1 or values.ndim != 2 or len(times) == 0 or values.shape[1] == 0
            or len(times) != len(values)):
        raise ValueError("Cache must contain nonempty aligned 1D timestamps and 2D embeddings")
    if (not np.isfinite(times).all() or not np.isfinite(values).all()
            or (times < 0).any() or (np.diff(times) <= 0).any()):
        raise ValueError("Cache contains nonfinite features or invalid/nonincreasing actual PTS")
    if (not isinstance(metadata, dict) or metadata.get("schema_version") != 1
            or metadata.get("feature_spec") != FEATURE_SPEC):
        raise ValueError("Unsupported cache schema or incompatible feature extractor")
    return {"video_id": video_id, "timestamps_s": times, "embeddings": values, "metadata": metadata}


def save_cache(path, video_id, timestamps, embeddings, metadata=None):
    np = numpy_module()
    meta = {"schema_version": 1, "feature_spec": dict(FEATURE_SPEC)}
    if metadata:
        meta.update(metadata)
    cache = validate_cache(video_id, timestamps, embeddings, meta)
    serialized = json.dumps(meta, ensure_ascii=False, allow_nan=False)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        np.savez_compressed(handle, video_id=np.asarray(video_id),
                            timestamps_s=cache["timestamps_s"], embeddings=cache["embeddings"],
                            metadata_json=np.asarray(serialized))


def load_cache(path):
    np = numpy_module()
    try:
        with np.load(path, allow_pickle=False) as data:
            return validate_cache(str(data["video_id"].item()), data["timestamps_s"],
                                  data["embeddings"], json.loads(str(data["metadata_json"].item())))
    except (KeyError, TypeError, OSError, EOFError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid or incomplete numeric feature cache: " + str(path)) from exc


class SiglipEncoder:
    """No text/tokenizer or full-model get_image_features return-type assumptions."""

    def __init__(self, weights, device="cpu", cpu_threads=2, seed=0):
        path = local_weights(weights)  # Sets offline flags BEFORE importing the Hub/transformers.
        if device not in ("cpu", "cuda") or type(cpu_threads) is not int or cpu_threads <= 0:
            raise ValueError("device must be cpu/cuda and cpu_threads a positive integer")
        try:
            import torch
            from transformers import SiglipVisionModel
            try:
                from transformers import SiglipImageProcessorPil as SiglipProcessor
            except ImportError:  # Transformers 4.x names the PIL implementation this way.
                from transformers import SiglipImageProcessor as SiglipProcessor
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Extraction needs local torch, transformers, Pillow and NumPy; see requirements-ml-baseline.txt") from exc
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; choose --device cpu")
        torch.set_num_threads(cpu_threads)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        self.torch, self.Image, self.device = torch, Image, device
        try:
            self.processor = SiglipProcessor.from_pretrained(str(path), local_files_only=True,
                                                            trust_remote_code=False)
            self.model, info = SiglipVisionModel.from_pretrained(str(path), local_files_only=True,
                                                                 trust_remote_code=False, use_safetensors=True,
                                                                 output_loading_info=True)
            if any(info.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")):
                raise RuntimeError("Local pretrained vision weights have missing/mismatched keys; refusing random initialization")
            self.model.to(device)
            self.model.eval()
        except (OSError, ValueError) as exc:
            raise RuntimeError("Unable to load complete local pretrained SigLIP weights; no network/fallback used: " + str(exc)) from exc

    def __call__(self, raw_frames):
        if not raw_frames or any(len(raw) != 224 * 224 * 3 for raw in raw_frames):
            raise ValueError("Encoder requires nonempty RGB224 raw frame batches")
        images = [self.Image.frombytes("RGB", (224, 224), raw) for raw in raw_frames]
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            pooled = self.model(**inputs).pooler_output
        if pooled is None or len(pooled.shape) != 2 or pooled.shape[0] != len(raw_frames):
            raise RuntimeError("SigLIP did not return aligned pooled vision features")
        return pooled.detach().to(device="cpu", dtype=self.torch.float32).numpy()


def embed_frames(frames, encode_batch, batch_size=8, max_frames=3600):
    """Keep at most batch_size decoded frames and max_frames numeric embeddings."""
    np = numpy_module()
    if type(batch_size) is not int or batch_size <= 0 or type(max_frames) is not int or max_frames <= 0:
        raise ValueError("batch_size and max_frames must be positive integers")
    timestamps, output, batch = [], [], []
    dimensions = None

    def flush():
        nonlocal dimensions
        values = np.asarray(encode_batch(batch))
        if (values.dtype.kind not in "fiu" or values.ndim != 2 or len(values) != len(batch)
                or values.shape[1] == 0 or not np.isfinite(values).all()):
            raise ValueError("Encoder must return aligned finite 2D numeric embeddings")
        if dimensions is not None and dimensions != values.shape[1]:
            raise ValueError("Encoder embedding dimensions changed between batches")
        dimensions = values.shape[1]
        output.append(values.astype(np.float32))
        batch.clear()

    for pts, raw in frames:
        if not math.isfinite(pts) or pts < 0 or (timestamps and pts <= timestamps[-1]):
            raise ValueError("Embedding pipeline requires actual increasing PTS")
        timestamps.append(float(pts))
        batch.append(raw)
        if len(batch) == batch_size:
            flush()
        if len(timestamps) == max_frames:
            break
    if batch:
        flush()
    if not output:
        raise ValueError("No extracted frames; missing predictions cannot imply no_violation")
    return np.asarray(timestamps, dtype=np.float64), np.concatenate(output, axis=0)


def ffmpeg_command(video, fps=1.0, max_frames=3600, ffmpeg="ffmpeg"):
    source = _local_path(video)
    if not source.is_file():
        raise FileNotFoundError("Local video does not exist: " + str(source))
    if not math.isfinite(fps) or fps <= 0 or type(max_frames) is not int or max_frames <= 0:
        raise ValueError("fps must be positive/finite and max_frames a positive integer")
    executable = shutil.which(ffmpeg)
    if not executable:
        raise RuntimeError("ffmpeg not found; install/configure it outside this baseline")
    interval = max(0.0, 1.0 / fps - 1e-9)
    filters = f"select='isnan(prev_selected_t)+gte(t-prev_selected_t,{interval:.12g})',scale=224:224,showinfo"
    return [executable, "-nostdin", "-hide_banner", "-loglevel", "info",
            "-protocol_whitelist", "file,pipe", "-copyts", "-start_at_zero", "-autorotate",
            "-i", str(source), "-map", "0:v:0", "-an", "-sn", "-dn",
            "-vf", filters, "-frames:v", str(max_frames), "-fps_mode", "passthrough",
            "-threads", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]


def decode_frames(video, fps=1.0, max_frames=3600, ffmpeg="ffmpeg", timeout_s=600.0):
    """Stream bounded 224x224 RGB frames paired with actual ffmpeg showinfo PTS.

    Autorotation precedes filters. PTS is container-relative via copyts/start_at_zero;
    frame numbering / nominal FPS is never used to fabricate timecodes.
    No image files, pixel data or full decoder logs are written to disk/stdout.
    """
    command = ffmpeg_command(video, fps=fps, max_frames=max_frames, ffmpeg=ffmpeg)
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("decode timeout must be finite and positive")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    timestamps = queue.Queue(maxsize=max_frames + 1)
    errors = deque(maxlen=12)
    pattern = re.compile(r"\bn:\s*(\d+)\s+pts:.*?\bpts_time:([-+0-9.eE]+)")

    def read_metadata():
        try:
            for raw in process.stderr:
                line = raw.decode("utf-8", errors="replace")
                match = pattern.search(line)
                if match and "showinfo" in line:
                    timestamps.put((int(match.group(1)), float(match.group(2))))
                elif "error" in line.lower() or "invalid" in line.lower() or "failed" in line.lower():
                    errors.append(line.strip())
        finally:
            timestamps.put(None)

    reader = threading.Thread(target=read_metadata, daemon=True)
    reader.start()
    timed_out = threading.Event()

    def terminate_on_timeout():
        timed_out.set()
        process.kill()

    timer = threading.Timer(timeout_s, terminate_on_timeout)
    timer.daemon = True
    timer.start()
    count, previous = 0, -math.inf
    size = 224 * 224 * 3
    try:
        while count < max_frames:
            data = bytearray()
            while len(data) < size:
                chunk = process.stdout.read(size - len(data))
                if not chunk:
                    break
                data.extend(chunk)
            if not data:
                break
            if len(data) != size:
                raise RuntimeError("ffmpeg returned a truncated RGB frame")
            try:
                item = timestamps.get(timeout=timeout_s)
            except queue.Empty as exc:
                raise RuntimeError("ffmpeg did not provide actual frame timestamps") from exc
            if item is None or item[0] != count:
                raise RuntimeError("ffmpeg frame/PTS pairing failed; refusing guessed timestamps")
            pts = item[1]
            if not math.isfinite(pts) or pts < 0 or pts <= previous:
                raise RuntimeError("ffmpeg emitted invalid/nonincreasing PTS")
            previous = pts
            count += 1
            yield pts, bytes(data)
        status = process.wait(timeout=timeout_s)
        reader.join(timeout=5)
        if timed_out.is_set():
            raise RuntimeError("ffmpeg decode exceeded timeout; no complete extraction produced")
        if status != 0 or count == 0:
            raise RuntimeError("ffmpeg failed or decoded no video frames: " + " | ".join(errors))
    finally:
        timer.cancel()
        if process.poll() is None:
            process.kill()
        process.wait()
        process.stdout.close()
        reader.join(timeout=5)
        process.stderr.close()
