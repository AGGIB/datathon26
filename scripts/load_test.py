#!/usr/bin/env python3
"""Realistic load testing script for Datathon 26 API against live uvicorn.

Measures upload latency, polling duration, p50/p95/max percentiles,
and asserts schema correctness and status for concurrent requests (N=6-10).
"""

import argparse
import base64
import concurrent.futures
import io
import json
import math
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def _is_valid_timestamp(val: object) -> bool:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return False
    if not math.isfinite(val) or val < 0:
        return False
    return True


def find_default_video() -> tuple[str, bytes]:
    """Find a real sample video or synthesize a valid ISO-BMFF .mov container."""
    search_dirs = [
        Path("data/open"),
        Path("../open/videos"),
        Path("var/jobs"),
    ]
    for d in search_dirs:
        if d.is_dir():
            for mov in sorted(d.glob("**/*.mov")):
                if mov.is_file() and mov.stat().st_size > 1024:
                    return mov.name, mov.read_bytes()

    # Valid ISO-BMFF container fallback (ftyp box + moov box + mdat)
    fake_mov = b"\x00\x00\x00\x14ftypqt  \x00\x00\x02\x00qt  \x00\x00\x00\x08moov\x00\x00\x04\x08mdat" + b"\x00" * 4096
    return "synthetic_iso_bmff.mov", fake_mov


def get_default_api_key() -> str:
    if os.environ.get("API_KEY"):
        return os.environ["API_KEY"]
    if os.environ.get("JURY_KEY"):
        return os.environ["JURY_KEY"]
    env_file = Path(".env")
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("API_KEYS="):
                keys = line.split("=", 1)[1].strip().strip("'\"").split(",")
                if keys and keys[0].strip():
                    return keys[0].strip()
    return "test-key"


def upload_video(base_url: str, api_key: str, filename: str, content: bytes, timeout: float = 120.0) -> tuple[int, dict, float]:
    url = f"{base_url.rstrip('/')}/api/v1/analyze"
    boundary = "----WebKitFormBoundary" + hex(int(time.time() * 1000))[2:]

    body = io.BytesIO()
    body.write(f"--{boundary}\r\n".encode())
    body.write(f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode())
    body.write(b"Content-Type: video/quicktime\r\n\r\n")
    body.write(content)
    body.write(b"\r\n")
    body.write(f"--{boundary}--\r\n".encode())

    data = body.getvalue()
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "X-API-Key": api_key,
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(data)),
        },
        method="POST",
    )

    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            elapsed = time.time() - t0
            payload = json.loads(resp.read().decode("utf-8"))
            return resp.status, payload, elapsed
    except urllib.error.HTTPError as exc:
        elapsed = time.time() - t0
        try:
            payload = json.loads(exc.read().decode("utf-8")) if exc.fp else {"detail": str(exc)}
        except Exception:
            payload = {"detail": str(exc)}
        return exc.code, payload, elapsed
    except (urllib.error.URLError, TimeoutError, OSError, ConnectionError) as exc:
        elapsed = time.time() - t0
        return 0, {"detail": f"Upload network error: {type(exc).__name__}: {exc}"}, elapsed


def poll_job(base_url: str, api_key: str, job_id: str, max_wait: float = 300.0) -> tuple[dict, float, int]:
    url = f"{base_url.rstrip('/')}/api/v1/jobs/{job_id}"
    req = urllib.request.Request(url, headers={"X-API-Key": api_key}, method="GET")

    t0 = time.time()
    poll_errors = 0
    while time.time() - t0 < max_wait:
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                status = data.get("status")
                if status in {"done", "failed"}:
                    return data, time.time() - t0, poll_errors
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError, ConnectionError):
            poll_errors += 1
        time.sleep(0.5)

    return {"status": "failed", "error": f"Job {job_id} exceeded poll timeout {max_wait}s"}, time.time() - t0, poll_errors


def validate_job_result(data: dict) -> tuple[bool, str]:
    """Strictly validates K1 contract invariants and schema correctness."""
    status = data.get("status")
    if status not in {"done", "failed"}:
        return False, f"Unknown status: {status!r}"
    if status == "failed":
        err = data.get("error", "")
        if not err or not isinstance(err, str):
            return False, "Failed status requires a descriptive string error message"
        return True, ""

    verdict = data.get("verdict")
    if verdict not in {"violation", "no_violation"}:
        return False, f"Invalid verdict: {verdict!r}"

    violations = data.get("violations")
    if not isinstance(violations, list):
        return False, f"violations must be a list, got {type(violations)}"

    if verdict == "violation":
        if len(violations) == 0:
            return False, "Verdict is 'violation' but violations list is empty"
        for idx, v in enumerate(violations):
            if not isinstance(v, dict):
                return False, f"violation[{idx}] must be a dict"
            if not v.get("type") or not isinstance(v["type"], str):
                return False, f"violation[{idx}] missing valid 'type'"
            if "timecode_sec" not in v or not _is_valid_timestamp(v["timecode_sec"]):
                return False, f"violation[{idx}] missing valid non-negative 'timecode_sec'"
            if not v.get("description") or not isinstance(v["description"], str):
                return False, f"violation[{idx}] missing valid 'description'"

            rule = v.get("rule")
            if rule is not None:
                if not isinstance(rule, dict) or not rule.get("document") or not isinstance(rule["document"], str):
                    return False, f"violation[{idx}].rule must be a dict with non-empty 'document'"

            frames = v.get("evidence_frames")
            if not isinstance(frames, list) or len(frames) == 0 or len(frames) > 3:
                return False, f"violation[{idx}] evidence_frames must be a list of 1 to 3 frames"
            for f_idx, frame in enumerate(frames):
                if not isinstance(frame, dict):
                    return False, f"violation[{idx}].evidence_frames[{f_idx}] must be a dict"
                if "timestamp_sec" not in frame or not _is_valid_timestamp(frame["timestamp_sec"]):
                    return False, f"violation[{idx}].evidence_frames[{f_idx}] missing valid non-negative 'timestamp_sec'"
                b64_str = frame.get("image_base64")
                if not isinstance(b64_str, str) or not b64_str:
                    return False, f"violation[{idx}].evidence_frames[{f_idx}] missing 'image_base64'"
                if b64_str.startswith("data:"):
                    return False, f"violation[{idx}].evidence_frames[{f_idx}] image_base64 has forbidden data: URI prefix"
                try:
                    raw_bytes = base64.b64decode(b64_str, validate=True)
                except Exception:
                    return False, f"violation[{idx}].evidence_frames[{f_idx}] invalid base64 encoding"
                if not raw_bytes.startswith(b"\xff\xd8"):
                    return False, f"violation[{idx}].evidence_frames[{f_idx}] image_base64 is not valid JPEG (missing 0xFFD8 magic header)"
    elif verdict == "no_violation":
        if len(violations) != 0:
            return False, f"Verdict is 'no_violation' but violations list is non-empty (got {len(violations)})"

    return True, ""


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    k = (len(sorted_v) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_v) - 1)
    d = k - f
    return sorted_v[f] + (sorted_v[c] - sorted_v[f]) * d


def main() -> int:
    parser = argparse.ArgumentParser(description="Datathon26 Live Load Test")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="API base URL")
    parser.add_argument("--api-key", default="", help="X-API-Key header value")
    parser.add_argument("--requests", "-n", type=int, default=8, help="Number of test requests (default: 8)")
    parser.add_argument("--concurrency", "-c", type=int, default=4, help="Concurrency workers (default: 4)")
    parser.add_argument("--video-file", default="", help="Optional specific video file to upload")
    args = parser.parse_args()

    api_key = args.api_key or get_default_api_key()
    print("=== Datathon26 Live Load Test ===")
    print(f"Target URL:    {args.url}")
    print(f"Concurrency:   {args.concurrency}")
    print(f"Requests:      {args.requests}")

    if args.video_file and Path(args.video_file).is_file():
        filename = Path(args.video_file).name
        content = Path(args.video_file).read_bytes()
    else:
        filename, content = find_default_video()

    print(f"Test video:    {filename} ({len(content) / (1024*1024):.2f} MB)")

    # Health check probe before starting
    try:
        health_req = urllib.request.Request(f"{args.url.rstrip('/')}/health")
        with urllib.request.urlopen(health_req, timeout=10) as r:
            health = json.loads(r.read().decode())
            print(f"Target health: OK (pipeline: {health.get('pipeline')})")
    except Exception as exc:
        print(f"[-] ERROR: Failed to connect to {args.url}/health: {exc}")
        return 1

    upload_times: list[float] = []
    processing_times: list[float] = []
    total_times: list[float] = []
    results: list[dict] = []
    schema_failures: list[str] = []

    start_all = time.time()

    def run_one(idx: int) -> dict:
        v_name = f"load_{idx}_{filename}"
        code, upload_res, upload_dur = upload_video(args.url, api_key, v_name, content)
        if code != 200:
            return {"index": idx, "code": code, "error": upload_res, "upload_dur": upload_dur, "poll_errors": 0}
        job_id = upload_res["job_id"]
        job_res, poll_dur, poll_errors = poll_job(args.url, api_key, job_id)
        return {
            "index": idx,
            "code": 200,
            "job_id": job_id,
            "upload_dur": upload_dur,
            "poll_dur": poll_dur,
            "poll_errors": poll_errors,
            "total_dur": upload_dur + poll_dur,
            "result": job_res,
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(run_one, i) for i in range(args.requests)]
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            results.append(res)
            if res.get("code") == 200:
                upload_times.append(res["upload_dur"])
                processing_times.append(res["poll_dur"])
                total_times.append(res["total_dur"])
                job_res = res["result"]
                status = job_res.get("status")
                verdict = job_res.get("verdict", "-")

                valid, err_msg = validate_job_result(job_res)
                if not valid:
                    schema_failures.append(f"Job {res['job_id']}: {err_msg}")
                    print(f"  [-] Job {res['job_id']}: INVALID SCHEMA: {err_msg}")
                else:
                    print(f"  [+] Job {res['job_id']}: status={status}, verdict={verdict}, duration={res['total_dur']:.2f}s")
            else:
                print(f"  [-] Request {res['index']} failed: HTTP {res['code']} ({res.get('error')})")

    wall_clock = time.time() - start_all
    success_count = sum(1 for r in results if r.get("code") == 200 and r.get("result", {}).get("status") == "done")
    failed_count = sum(1 for r in results if r.get("code") == 200 and r.get("result", {}).get("status") == "failed")
    upload_http_errors = sum(1 for r in results if r.get("code") != 200)
    polling_errors = sum(r.get("poll_errors", 0) for r in results)
    http_errors = upload_http_errors + polling_errors

    print("\n--- Summary Statistics ---")
    print(f"Total Requests:       {args.requests}")
    print(f"Completed (done):     {success_count}")
    print(f"Failed (timeout):     {failed_count}")
    print(f"HTTP/Network Errors:  {http_errors}")
    print(f"Schema Errors:        {len(schema_failures)}")
    print(f"Wall Clock Time:      {wall_clock:.2f}s")
    if total_times:
        print(f"Throughput:           {len(total_times) / wall_clock:.2f} req/s")
        print("\nLatency Percentiles (Total E2E):")
        print(f"  p50 (median):       {percentile(total_times, 0.50):.2f}s")
        print(f"  p95:                {percentile(total_times, 0.95):.2f}s")
        print(f"  Max:                {max(total_times):.2f}s")
        print("\nUpload Latency:")
        print(f"  p50:                {percentile(upload_times, 0.50):.3f}s")
        print(f"  p95:                {percentile(upload_times, 0.95):.3f}s")
        print("\nProcessing Latency:")
        print(f"  p50:                {percentile(processing_times, 0.50):.2f}s")
        print(f"  p95:                {percentile(processing_times, 0.95):.2f}s")

    # Strict assertion checks
    if http_errors > 0:
        print(f"\n[-] Load test assertion failed: {http_errors} HTTP/network errors encountered")
        return 1
    if schema_failures:
        print(f"\n[-] Load test assertion failed: {len(schema_failures)} jobs failed schema validation")
        return 1
    if success_count == 0:
        print("\n[-] Load test assertion failed: 0 jobs completed successfully (all failed)")
        return 1
    if success_count + failed_count != args.requests:
        print(f"\n[-] Load test assertion failed: incomplete results ({success_count + failed_count}/{args.requests})")
        return 1

    print("\n[+] Load test completed successfully with all invariant assertions passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
