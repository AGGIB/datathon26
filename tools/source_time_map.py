#!/usr/bin/env python3
"""Standalone metadata PTS mapping diagnostic; never changes production sampling.

Integer frame.pts and rational stream.time_base are preserved exactly. Origin
is ONLY explicit stream.start_pts. A metadata frame index is not proof of an
OpenCV/decoder index or an evidence JPEG's identity. No pixels are produced.
"""
from dataclasses import dataclass
from fractions import Fraction
from bisect import bisect_left, bisect_right
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import select
import shutil
import stat
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 64 * 1024 * 1024
MAX_FRAMES = 100_000
MAX_REQUESTS = 64
MAX_CANDIDATES = 32


def _integer(value):
    if value is None or value in ("N/A", ""):
        return None
    if type(value) is str and re.fullmatch(r"-?\d{1,19}", value):
        value = int(value)
    if type(value) is not int or not -(2**63) <= value < 2**63:
        raise ValueError("invalid_integer_timestamp")
    return value


def _time_base(value):
    if value is None or value in ("N/A", ""):
        return None
    if not isinstance(value, str) or re.fullmatch(r"\d{1,10}/\d{1,10}", value) is None:
        raise ValueError("invalid_time_base")
    numerator, denominator = map(int, value.split("/"))
    if not 0 < numerator < 2**31 or not 0 < denominator < 2**31:
        raise ValueError("invalid_time_base")
    return Fraction(numerator, denominator)


def _request(value):
    if type(value) is bool or not isinstance(value, (str, int, float, Fraction)):
        raise ValueError("invalid_requested_time")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("invalid_requested_time")
    text = str(value)
    if len(text) > 64 or ("e" in text.lower() and abs(int(text.lower().split("e")[1])) > 18):
        raise ValueError("invalid_requested_time")
    try:
        result = Fraction(text)
    except (ValueError, ZeroDivisionError):
        raise ValueError("invalid_requested_time") from None
    if result < 0 or result.numerator.bit_length() > 256 or result.denominator.bit_length() > 256:
        raise ValueError("invalid_requested_time")
    return result


def _fraction(value):
    return {"numerator": value.numerator, "denominator": value.denominator}


@dataclass(frozen=True)
class MetadataFrame:
    metadata_frame_index: int  # Zero-based within the explicitly bound video stream.
    probe_frame_position: int  # Position in the supplied ffprobe frame array.
    pts: int | None


@dataclass(frozen=True)
class Timeline:
    status: str
    stream_index: int | None
    available_video_streams: tuple[int, ...]
    time_base: Fraction | None
    stream_start_pts: int | None
    frames: tuple[MetadataFrame, ...]


def parse_metadata(document, *, stream_index=None):
    if not isinstance(document, dict) or not isinstance(document.get("streams"), list) or not isinstance(document.get("frames"), list):
        raise ValueError("invalid_probe_document")
    if len(document["frames"]) > MAX_FRAMES:
        raise ValueError("metadata_frame_limit")
    if stream_index is not None and (type(stream_index) is not int or stream_index < 0):
        raise ValueError("invalid_stream_selection")
    streams = {}
    for stream in document["streams"]:
        if not isinstance(stream, dict):
            raise ValueError("invalid_probe_stream")
        if stream.get("codec_type") != "video":
            continue
        index = stream.get("index")
        if type(index) is not int or index < 0 or index in streams:
            raise ValueError("invalid_probe_stream")
        streams[index] = stream
    available = tuple(sorted(streams))
    if stream_index is None and len(streams) != 1:
        return Timeline("unknown_stream_selection", None, available, None, None, ())
    chosen = stream_index if stream_index is not None else available[0]
    if chosen not in streams:
        raise ValueError("video_stream_not_found")
    stream = streams[chosen]
    tb, origin = _time_base(stream.get("time_base")), _integer(stream.get("start_pts"))
    frames = []
    for position, frame in enumerate(document["frames"]):
        if not isinstance(frame, dict) or type(frame.get("stream_index")) is not int:
            return Timeline("unknown_frame_stream_binding", chosen, available, tb, origin, ())
        if frame["stream_index"] == chosen:
            frames.append(MetadataFrame(len(frames), position, _integer(frame.get("pts"))))
    known = [frame.pts for frame in frames if frame.pts is not None]
    status = ("unknown_time_base" if tb is None else "unknown_stream_start_pts" if origin is None else
              "unknown_frames" if not frames else "unknown_missing_frame_pts" if len(known) != len(frames) else
              "unknown_nonmonotonic_pts" if known != sorted(known) else "available")
    return Timeline(status, chosen, available, tb, origin, tuple(frames))


def map_requested_times(timeline, requested_times):
    if not isinstance(timeline, Timeline) or not isinstance(requested_times, (tuple, list)) or not 1 <= len(requested_times) <= MAX_REQUESTS:
        raise ValueError("invalid_mapping_request")
    requests = tuple(_request(value) for value in requested_times)
    times = tuple((frame.pts - timeline.stream_start_pts) * timeline.time_base for frame in timeline.frames) if timeline.status == "available" else ()
    rows = []
    for requested in requests:
        row = {"requested_time_seconds": _fraction(requested), "status": "unknown", "reason": timeline.status,
               "chosen_metadata_frame_index": None, "chosen_probe_frame_position": None,
               "pts": None, "pts_seconds": None, "video_stream_relative_time_seconds": None,
               "candidate_count": 0, "candidate_metadata_frame_indices": [], "candidates_truncated": False}
        if timeline.status == "available":
            if not times[0] <= requested <= times[-1]:
                row["reason"] = "outside_observed_pts_range"
            else:
                position = bisect_left(times, requested)
                nearby = {times[index] for index in (position - 1, position) if 0 <= index < len(times)}
                distance = min(abs(t - requested) for t in nearby)
                values = sorted(t for t in nearby if abs(t - requested) == distance)
                spans = [(bisect_left(times, t), bisect_right(times, t)) for t in values]
                count = sum(end - start for start, end in spans)
                indices = []
                for start, end in spans:
                    indices.extend(range(start, min(end, start + MAX_CANDIDATES - len(indices))))
                row.update(candidate_count=count, candidate_metadata_frame_indices=indices,
                           candidates_truncated=count > MAX_CANDIDATES)
                if count != 1:
                    row.update(status="ambiguous", reason="repeated_pts" if len(values) == 1 else "equidistant_pts")
                else:
                    frame = timeline.frames[indices[0]]
                    row.update(status="mapped_metadata", reason="unique_nearest_observed_pts",
                               chosen_metadata_frame_index=frame.metadata_frame_index,
                               chosen_probe_frame_position=frame.probe_frame_position, pts=frame.pts,
                               pts_seconds=_fraction(frame.pts * timeline.time_base),
                               video_stream_relative_time_seconds=_fraction((frame.pts - timeline.stream_start_pts) * timeline.time_base))
        rows.append(row)
    return {"format_version": 1, "scope": "metadata_only_nearest_pts_diagnostic", "selector": "nearest_observed_pts_relative_to_stream_start",
            "production_cv_index_fps_mapping_changed": False, "decoded_frame_identity_verified": False,
            "live_evidence_roundtrip_verified": False, "origin_kind": "stream_start_pts",
            "stream_index": timeline.stream_index, "available_video_streams": list(timeline.available_video_streams),
            "stream_start_pts": timeline.stream_start_pts, "time_base": _fraction(timeline.time_base) if timeline.time_base else None,
            "timeline_status": timeline.status, "metadata_frame_count": len(timeline.frames),
            "missing_frame_pts_count": sum(frame.pts is None for frame in timeline.frames), "mappings": rows}


def _identity(path):
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def sha256(path):
    before = _identity(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for data in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(data)
    if _identity(path) != before:
        raise ValueError("input_changed")
    return digest.hexdigest()


def _local_file(value):
    if "://" in str(value) or str(value).replace("\\", "/").startswith("//"):
        raise ValueError("local_file_required")
    raw = Path(value).absolute()
    if raw.is_symlink():
        raise ValueError("local_regular_file_required")
    path = raw.resolve()
    if not path.is_file() or not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("local_regular_file_required")
    return path


def read_metadata_json(data):
    if len(data) > MAX_BYTES:
        raise ValueError("metadata_byte_limit")
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("duplicate_json_key")
            value[key] = item
        return value
    def nonfinite(value):
        raise ValueError("nonfinite_json")
    return json.loads(data, object_pairs_hook=pairs, parse_constant=nonfinite)


def _run_probe(command, timeout_s):
    if type(timeout_s) not in (float, int) or not math.isfinite(timeout_s) or not 0 < timeout_s <= 120:
        raise ValueError("invalid_probe_timeout")
    deadline, data = time.monotonic() + timeout_s, bytearray()
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("probe_timeout")
            ready, _, _ = select.select([process.stdout], [], [], remaining)
            if not ready:
                raise TimeoutError("probe_timeout")
            chunk = os.read(process.stdout.fileno(), 65536)
            if not chunk:
                break
            data.extend(chunk)
            if len(data) > MAX_BYTES:
                raise ValueError("metadata_byte_limit")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("probe_timeout")
        if process.wait(timeout=remaining) != 0:
            raise ValueError("probe_failed")
        return bytes(data)
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        process.stdout.close()


def probe_video(path, *, ffprobe="ffprobe", timeout_s=30):
    """Explicit local-file operation; callers authorize reading that source.

    ffprobe may internally decode while reporting frame metadata. This function
    does not inspect pixels, emit images or prove any evidence frame's identity.
    """
    source = _local_file(path)
    executable = shutil.which(str(ffprobe))
    if executable is None:
        raise ValueError("ffprobe_unavailable")
    binary = Path(executable).resolve()
    source_sha, binary_sha = sha256(source), sha256(binary)
    payload = _run_probe([str(binary), "-v", "error", "-protocol_whitelist", "file,pipe", "-select_streams", "v",
                          "-show_frames", "-show_streams", "-show_entries",
                          "frame=stream_index,pts:stream=index,codec_type,time_base,start_pts", "-of", "json", str(source)], timeout_s)
    if sha256(source) != source_sha or sha256(binary) != binary_sha:
        raise ValueError("input_changed")
    return read_metadata_json(payload), {"source_media_sha256": source_sha, "ffprobe_binary_sha256": binary_sha,
                                        "probe_metadata_sha256": hashlib.sha256(payload).hexdigest()}


def _save(path, report, inputs):
    raw = Path(path).absolute()
    if ".." in raw.parts or any(part.is_symlink() for part in (raw, *raw.parents)):
        raise ValueError("fresh_ignored_output_required")
    base = ROOT.resolve() / "var"
    canonical = raw.resolve()
    if (not canonical.is_relative_to(base) or canonical == base or canonical.suffix != ".json"
            or canonical.exists() or canonical in {value.resolve() for value in inputs}):
        raise ValueError("fresh_ignored_output_required")
    raw = canonical
    raw.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=raw.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.link(temporary, raw)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class SafeParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError("invalid_arguments")


def main(argv=None):
    parser = SafeParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--probe-json", type=Path)
    inputs.add_argument("--video", type=Path)
    parser.add_argument("--stream-index", type=int)
    parser.add_argument("--times", nargs="+", required=True)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--out", type=Path, required=True)
    try:
        args = parser.parse_args(argv)
        source = _local_file(args.video if args.video else args.probe_json)
        if args.video:
            document, provenance = probe_video(source, timeout_s=args.timeout)
        else:
            if source.stat().st_size > MAX_BYTES:
                raise ValueError("metadata_byte_limit")
            before = sha256(source)
            document = read_metadata_json(source.read_bytes())
            if sha256(source) != before:
                raise ValueError("input_changed")
            provenance = {"metadata_json_sha256": before, "source_media_sha256": None}
        report = map_requested_times(parse_metadata(document, stream_index=args.stream_index), args.times)
        report["provenance"] = {**provenance, "diagnostic_source_sha256": sha256(Path(__file__))}
        _save(args.out, report, (source,))
        counts = {status: sum(row["status"] == status for row in report["mappings"]) for status in ("mapped_metadata", "ambiguous", "unknown")}
        print(json.dumps({"status": "metadata_only", "requests": len(report["mappings"]), **counts}, sort_keys=True))
        return 0
    except (OSError, ValueError, TypeError, KeyError, TimeoutError, subprocess.SubprocessError):
        print('{"status":"blocked","reason":"invalid_or_unavailable_local_metadata"}')
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
