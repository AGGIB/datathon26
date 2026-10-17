"""Local, label-free dataset features; no fitting, split inference or quality claims."""
import hashlib
import math
from contextlib import contextmanager
from pathlib import Path

from . import features
from .core import artifact_path, read_json, write_json


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_fingerprint(weights):
    """Identify actual local config, processor and checkpoint bytes, including shards."""
    root = features.local_weights(weights)
    names = {"config.json", "preprocessor_config.json"}
    if (root / "model.safetensors").is_file():
        names.add("model.safetensors")
    else:
        names.add("model.safetensors.index.json")
        names.update(read_json(root / "model.safetensors.index.json")["weight_map"].values())
    digest = hashlib.sha256()
    for name in sorted(names):
        source = features._local_path(root / name)
        if not source.is_relative_to(root):
            raise ValueError("Snapshot files must remain inside the local weights directory")
        digest.update(name.encode("utf-8") + b"\0" + file_sha256(source).encode("ascii") + b"\n")
    return digest.hexdigest()


@contextmanager
def _output_ownership(target):
    """Claim a directory atomically; only the successful claimant releases it."""
    target.mkdir(parents=True, exist_ok=True)
    lock = target / ".extract.lock"
    try:
        lock.mkdir()
    except FileExistsError:
        raise ValueError("Dataset output is already owned by another extraction") from None
    # Acquisition must stay outside this finally: contenders never remove the lock.
    try:
        if any(path != lock for path in target.iterdir()):
            raise ValueError("Dataset output must be a new or empty directory; existing results are never overwritten")
        yield
    finally:
        lock.rmdir()


def extract_dataset(videos_dir, weights, output_dir, fps=1.0, max_frames=3600,
                    batch_size=8, device="cpu", cpu_threads=2, seed=0, ffmpeg="ffmpeg",
                    timeout_s=600.0, artifact_root=None):
    """Extract each literal .mov ID; exact duplicates share inference, never identity."""
    source = features._local_path(videos_dir)
    target = artifact_path(output_dir, root=artifact_root)
    videos = sorted((p for p in source.iterdir() if p.suffix.lower() == ".mov"),
                    key=lambda p: p.name)
    if not videos:
        raise ValueError("No local .mov inputs found")
    if (not math.isfinite(fps) or fps <= 0 or type(max_frames) is not int or max_frames <= 0
            or type(batch_size) is not int or batch_size <= 0
            or not math.isfinite(timeout_s) or timeout_s <= 0
            or type(cpu_threads) is not int or cpu_threads <= 0 or device not in ("cpu", "cuda")):
        raise ValueError("Invalid dataset extraction configuration")
    with _output_ownership(target):
        snapshot = snapshot_fingerprint(weights)
        encoder = features.SiglipEncoder(weights, device=device, cpu_threads=cpu_threads, seed=seed)
        report = {"schema_version": 1, "report_kind": "local_dataset_features_not_detector_predictions",
                  "quality_metrics_measured": False, "expected_videos": len(videos),
                  "successful_videos": 0, "failed_videos": 0, "capped_videos": 0,
                  "unique_contents": 0, "encoded_contents": 0, "snapshot_sha256": snapshot,
                  "extraction_config": {"fps": fps, "max_frames": max_frames, "batch_size": batch_size,
                                        "device": device, "cpu_threads": cpu_threads, "seed": seed},
                  "videos": []}
        completed = {}
        digests = set()
        for video in videos:
            record: dict = {"video_id": video.name, "status": "failed"}
            cache_file = hashlib.sha256(video.name.encode("utf-8")).hexdigest() + ".npz"
            cache_path = artifact_path(target / cache_file, root=artifact_root)
            try:
                # Reject network symlinks before opening or hashing the source.
                local_video = features._local_path(video)
                if not local_video.is_file():
                    raise ValueError("Each .mov input must be a local regular file")
                content = file_sha256(local_video)
                digests.add(content)
                record["content_sha256"] = content
                if content in completed:
                    cache = features.load_cache(completed[content])
                    record["reused_content"] = True
                else:
                    features.extract_video(local_video, weights, cache_path, video_id=video.name, fps=fps,
                                           max_frames=max_frames, batch_size=batch_size, device=device,
                                           cpu_threads=cpu_threads, seed=seed, ffmpeg=ffmpeg, timeout_s=timeout_s,
                                           artifact_root=artifact_root, encoder=encoder)
                    cache = features.load_cache(cache_path)
                    report["encoded_contents"] += 1
                    record["reused_content"] = False
                if file_sha256(local_video) != content:
                    raise ValueError("Source contents changed during extraction")
                stat = local_video.stat()
                metadata: dict = dict(cache["metadata"], content_sha256=content, snapshot_sha256=snapshot,
                                source_filename=video.name, source_size_bytes=stat.st_size,
                                source_mtime_ns=stat.st_mtime_ns)
                features.save_cache(cache_path, video.name, cache["timestamps_s"], cache["embeddings"], metadata)
                completed[content] = cache_path
                record.update(status="ok", cache_file=cache_file, n_frames=len(cache["timestamps_s"]),
                              embedding_dim=cache["embeddings"].shape[1],
                              cap_reached=bool(metadata.get("extraction", {}).get("cap_reached", True)))
                report["successful_videos"] += 1
                report["capped_videos"] += int(record["cap_reached"])
            except (ValueError, RuntimeError, OSError, TypeError, KeyError, MemoryError) as exc:
                # Error classes only: decoder/model diagnostics may contain private input details.
                record["error_class"] = type(exc).__name__
                report["failed_videos"] += 1
                if cache_path.exists():
                    cache_path.unlink()
            report["videos"].append(record)
            report["unique_contents"] = len(digests)
            write_json(target / "index.json", report)
        return report
