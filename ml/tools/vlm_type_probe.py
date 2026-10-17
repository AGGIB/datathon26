"""Explicit offline VLM experiment. Labels never enter planning or inference.

Run in the CV environment; --python points to the separate MLX environment.
Outputs contain model observations and stay in ignored local var/, never Git.
Existing CV predictions are retained before VLM additions fill available slots.
This is a development probe, not evidence of generalization to a new camera.
"""
from dataclasses import asdict
from contextlib import contextmanager
import argparse
import hashlib
import json
import logging
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

from ml.labels import normalize_video_name

from ml.vlm.discovery_policy import POLICY_ID, TYPE_IDS, plan_windows

ROOT = Path(__file__).resolve().parents[2]
# Returned infrastructure failures must not look like successful CV-only runs.
# completed, budget_exhausted (normal call cap), no_windows and no_verified_types
# are legitimate bounded completion/abstention, rather than runtime failures.
FAILURE_STATUSES = frozenset({"backend_error", "video_timeout", "frames_unavailable",
                              "model_unavailable", "prompt_too_long"})


class SafeParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError("invalid_arguments")


def _number(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate_json_key")
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError("nonfinite_json")

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=nonfinite)


def _output_path(path):
    base = ROOT.absolute() / "var"
    if base.resolve() != base:
        raise ValueError("output_root_symlink")
    path = Path(path).absolute().resolve()
    if not path.is_relative_to(base) or path == base or path.suffix != ".json":
        raise ValueError("output_must_be_json_under_var")
    return path


@contextmanager
def _output_lease(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(path.suffix + ".lock")
    with lock.open("x", encoding="utf-8"):
        pass
    try:
        if path.exists() or path.with_suffix(".audit.json").exists():
            raise ValueError("new_output_path_required")
        yield
    finally:
        lock.unlink(missing_ok=True)


def _validate_baseline(document):
    if not isinstance(document, dict):
        raise ValueError("baseline_must_be_object")
    result = {}
    for name, row in document.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9Хх][A-Za-z0-9Хх_-]{0,79}", name):
            raise ValueError("invalid_baseline_video_id")
        key = normalize_video_name(name)
        if key in result or not isinstance(row, dict) or not isinstance(row.get("violations"), list):
            raise ValueError("invalid_or_duplicate_baseline")
        events = row["violations"]
        if len(events) > 5:
            raise ValueError("baseline_exceeds_output_limit")
        for event in events:
            if (not isinstance(event, dict) or not isinstance(event.get("type_id"), str)
                    or re.fullmatch(r"[a-z][a-z0-9_]{0,79}", event["type_id"]) is None
                    or not _number(event.get("t")) or event["t"] < 0
                    or not _number(event.get("conf")) or not 0 <= event["conf"] <= 1):
                raise ValueError("invalid_baseline_event")
        result[key] = dict(row, violations=[dict(event) for event in events])
    return result


def _select_videos(folder, only):
    if not folder.is_dir():
        raise ValueError("local_video_directory_required")
    requested = set()
    for value in only.split(","):
        value = value.strip()
        if not value:
            continue
        if not re.fullmatch(r"[A-Za-z0-9Хх][A-Za-z0-9Хх_-]{0,79}", value):
            raise ValueError("invalid_requested_video_id")
        requested.add(normalize_video_name(value))
    videos = {}
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in {".mov", ".mp4"} or not path.is_file():
            continue
        if not re.fullmatch(r"[A-Za-z0-9Хх][A-Za-z0-9Хх_-]{0,79}", path.stem):
            raise ValueError("invalid_video_id")
        key = normalize_video_name(path.name)
        if key in videos:
            raise ValueError("duplicate_normalized_video_id")
        videos[key] = path
    if requested - videos.keys():
        raise ValueError("requested_video_missing")
    selected = {key: path for key, path in videos.items() if not requested or key in requested}
    if not selected:
        raise ValueError("no_videos_selected")
    return selected



def merge_predictions(baseline: list[dict], additions: list[dict], limit: int = 5) -> list[dict]:
    """Reserve baseline slots first, so VLM cannot displace an existing CV event."""
    kept = [dict(event) for event in baseline]
    for event in sorted(additions, key=lambda event: -event["conf"]):
        if len(kept) >= limit:
            break
        if any(other["type_id"] == event["type_id"] and abs(other["t"] - event["t"]) <= 2.0 for other in kept):
            continue
        kept.append(dict(event))
    return sorted(kept, key=lambda event: event["t"])


def _save(path: Path, value):
    path = _output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _sha(path: Path) -> str:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for data in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(data)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError("input_changed_during_hash")
    return digest.hexdigest()


def _source_hashes():
    return {str(path.relative_to(ROOT)): _sha(path) for path in sorted((ROOT / "ml").rglob("*.py"))}


def _runtime_versions(executable):
    code = ("import importlib.metadata,json,platform;"
            "print(json.dumps({'python':platform.python_version(),'packages':"
            "{d.metadata['Name']:d.version for d in importlib.metadata.distributions()}},sort_keys=True))")
    completed = subprocess.run([str(executable), "-I", "-c", code], capture_output=True, timeout=20, check=True)
    return json.loads(completed.stdout)


def _baseline_result(base, elapsed=0.0, error=None):
    events = [dict(event) for event in base]
    result = {"verdict": "violation" if events else "no_violation", "violations": events, "sec": round(elapsed, 3)}
    if error is not None:
        result["vlm_error"] = error
    return result


def _coded_verifications(values):
    """Copy only the verifier's numeric/boolean/fixed-code diagnostics."""
    result = []
    if not isinstance(values, (tuple, list)):
        return result
    for value in values:
        if (not isinstance(value, dict) or type(value.get("episode_id")) is not int
                or value["episode_id"] < 0 or type(value.get("accepted")) is not bool
                or not isinstance(value.get("reason"), str)
                or re.fullmatch(r"[a-z][a-z0-9_]{0,79}", value["reason"]) is None):
            continue
        result.append({key: value[key] for key in ("episode_id", "accepted", "reason")})
    return result


def run(args):
    from ml.kb import KB_PATH, catalog
    from ml.video import probe
    from ml.vlm.discovery import DiscoveryBudget, discover_types
    from ml.vlm.mlx_backend import LocalMLXBackend
    from ml.vlm.preflight import _validated_model_directory

    if not _number(args.budget) or args.budget <= 0:
        raise ValueError("positive_finite_budget_required")
    if type(args.max_tokens) is not int or not 32 <= args.max_tokens <= 1024:
        raise ValueError("invalid_token_limit")
    if type(args.frame_side) is not int or not 224 <= args.frame_side <= 896:
        raise ValueError("invalid_frame_side")
    args.out = _output_path(args.out)
    audit_path = args.out.with_suffix(".audit.json")
    if args.out == audit_path or args.out.exists() or audit_path.exists():
        raise ValueError("new_output_path_required")
    videos = _select_videos(args.videos, args.only)
    baseline = _validate_baseline(_read_json(args.baseline)) if args.baseline else {}
    if args.baseline and set(videos) - baseline.keys():
        raise ValueError("selected_video_missing_from_baseline")
    directory = _validated_model_directory(args.model)
    if not args.python.is_file():
        raise ValueError("local_python_required")
    artifacts = {path.name: _sha(path) for path in sorted(directory.iterdir())
                 if path.is_file() and (path.suffix in {".json", ".jinja", ".txt", ".model", ".safetensors"})}
    action_contrast = getattr(args, "action_contrast", False)
    verify_actions = getattr(args, "verify_actions", False)
    compact_verify_actions = getattr(args, "compact_verify_actions", False)
    if type(action_contrast) is not bool:
        raise ValueError("action_contrast_must_be_boolean")
    if type(verify_actions) is not bool:
        raise ValueError("verify_actions_must_be_boolean")
    if type(compact_verify_actions) is not bool:
        raise ValueError("compact_verify_actions_must_be_boolean")
    from ml.vlm.action_backend import DECODER_POLICY

    decoder_policy = DECODER_POLICY if action_contrast else "direct_discovery"
    if compact_verify_actions:
        from ml.vlm.compact_verified_action_backend import DECODER_POLICY as VERIFIER_POLICY

        decoder_policy = VERIFIER_POLICY
    elif verify_actions:
        from ml.vlm.verified_action_backend import DECODER_POLICY as VERIFIER_POLICY

        decoder_policy = VERIFIER_POLICY
    max_model_calls = 4 if (compact_verify_actions or verify_actions) else 2
    config = {"model_directory": str(directory), "model_artifact_sha256": artifacts,
              "model_weights_sha256": artifacts.get("model.safetensors"),
              "python": str(args.python.absolute()), "runtime": _runtime_versions(args.python),
              "parent_python": str(Path(sys.executable).absolute()),
              "parent_runtime": _runtime_versions(Path(sys.executable)),
              "kb_path": str(Path(KB_PATH).absolute()), "kb_sha256": _sha(Path(KB_PATH)),
              "max_tokens": args.max_tokens, "frame_side": args.frame_side,
              "budget_sec": args.budget, "allowlist": TYPE_IDS, "sampling_policy": POLICY_ID,
              "decoder_policy": decoder_policy, "max_model_calls_per_video": max_model_calls,
              "baseline_sha256": _sha(args.baseline) if args.baseline else None,
              "source_sha256": _source_hashes(), "selected_video_ids": sorted(videos),
              "evaluation_scope": "development_only_not_new_angle",
              "timing_scope": "warm_vlm_plus_frame_io_only_CV_replayed", "ground_truth_in_model_input": False}
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    audit = {"config": config, "config_sha256": fingerprint, "videos": {}}
    results, completed = {}, {}
    types = catalog()
    backend = LocalMLXBackend(directory, python_executable=args.python,
                              max_new_tokens=args.max_tokens, max_image_side=args.frame_side)
    try:
        if compact_verify_actions:
            from ml.vlm.compact_verified_action_backend import CompactVerifiedActionBackend

            backend = CompactVerifiedActionBackend(backend, catalog=types)
        elif verify_actions:
            from ml.vlm.verified_action_backend import VerifiedActionBackend

            backend = VerifiedActionBackend(backend, catalog=types)
        elif action_contrast:
            from ml.vlm.action_backend import ActionContrastBackend

            backend = ActionContrastBackend(backend, catalog=types)
    except Exception:
        backend.close()
        raise
    with _output_lease(args.out):
        try:
            started = time.monotonic()
            try:
                backend.validate(timeout_s=180)
            except Exception as exc:
                audit["startup"] = {"status": "failed", "error_kind": type(exc).__name__}
                for key in videos:
                    results[key] = _baseline_result(baseline.get(key, {}).get("violations", []), error="vlm_startup_failed")
                _save(args.out, results)
                _save(audit_path, audit)
                return results
            audit["startup"] = {"status": "ready", "elapsed_sec": round(time.monotonic() - started, 3)}
            for key, path in videos.items():
                started = time.monotonic()
                base = baseline.get(key, {}).get("violations", [])
                try:
                    digest = _sha(path)
                    duration = probe(path).duration
                    windows = plan_windows(duration)
                    identity = (digest, duration, windows)
                    reused = identity in completed
                    if reused:
                        owner, report = completed[identity]
                    else:
                        owner = key
                        budget = DiscoveryBudget(started + args.budget, max_calls=max_model_calls)
                        report = discover_types(path, (), types, duration_sec=duration, windows=windows,
                                                allowed_type_ids=TYPE_IDS, budget=budget, backend=backend)
                        if report.status == "completed":
                            completed[identity] = (key, report)
                    additions = [{"type_id": event.type_id, "t": round(event.t_peak, 1), "conf": event.confidence}
                                 for event in report.added_candidates]
                    if _sha(path) != digest:
                        raise ValueError("video_changed_during_inference")
                    merged = merge_predictions(base, additions)
                    failure = report.status if report.status in FAILURE_STATUSES else None
                    results[key] = _baseline_result(merged, time.monotonic() - started, error=failure)
                    audit["videos"][key] = {"sha256": digest, "duration_sec": duration,
                                             "report": asdict(report), "vlm_only": additions,
                                             "inference_reused": reused, "inference_source_video": owner,
                                             "actual_calls_used": 0 if reused else report.calls_used}
                    if not reused:
                        if hasattr(backend, "last_actions"):
                            audit["videos"][key]["decoder_last_call_actions"] = list(backend.last_actions)
                        if hasattr(backend, "last_verifications"):
                            audit["videos"][key]["verifier_last_call_decisions"] = _coded_verifications(backend.last_verifications)
                    logging.info("%s status=%s calls=%d added=%d reused=%s elapsed=%.1fs", key, report.status,
                                 0 if reused else report.calls_used, len(additions), reused, time.monotonic() - started)
                except Exception as exc:
                    results[key] = _baseline_result(base, time.monotonic() - started, error=type(exc).__name__)
                    audit["videos"][key] = {"error_kind": type(exc).__name__, "baseline_retained": True}
                    logging.info("%s failed: %s", key, type(exc).__name__)
                _save(args.out, results)
                _save(audit_path, audit)
            audit["code_inputs_stable"] = (_source_hashes() == config["source_sha256"]
                                           and _sha(Path(KB_PATH)) == config["kb_sha256"])
            _save(audit_path, audit)
        finally:
            backend.close()
    return results


def main():
    parser = SafeParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--videos", type=Path, default=Path("data/open/videos"))
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", default="")
    parser.add_argument("--budget", type=float, default=180)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--frame-side", type=int, default=640)
    parser.add_argument("--action-contrast", action="store_true", help="experimental balanced physical-action decoder")
    parser.add_argument("--verify-actions", action="store_true", help="experimental blind primitive verification; overrides --action-contrast")
    parser.add_argument("--compact-verify-actions", action="store_true",
                        help="experimental compact blind verification; overrides --verify-actions and --action-contrast")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    try:
        args = parser.parse_args()
        results = run(args)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, subprocess.SubprocessError):
        print("ERROR: invalid_local_vlm_experiment", file=sys.stderr)
        return 1
    return int(any("vlm_error" in row for row in results.values()))


if __name__ == "__main__":
    raise SystemExit(main())
