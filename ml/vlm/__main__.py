"""CLI emits metadata-only summaries; evidence goes only to the explicit local output."""
import argparse
import json
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from .preflight import enforce_offline, preflight
from .transformers_backend import LocalTransformersBackend
from .validation import number, strict_json
from .verifier import remaining_budget, verify_candidates


def main(argv=None) -> int:
    enforce_offline()
    parser = argparse.ArgumentParser(description="Offline local-machine VLM verification (human review required)")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("preflight", help="Check local artifacts/dependencies; --load validates runtime")
    check.add_argument("--model", required=True)
    check.add_argument("--load", action="store_true")
    check.add_argument("--device", default="cpu")
    check.add_argument("--timeout", type=float, default=60)
    verify = commands.add_parser("verify", help="Verify the complete candidate list for one local video")
    for name in ("video", "model", "candidates", "rules", "output"):
        verify.add_argument(f"--{name}", required=True)
    verify.add_argument("--device", default="cpu")
    verify.add_argument("--max-calls", type=int, default=6)
    verify.add_argument("--timeout", type=float, default=60)
    verify.add_argument("--budget-sec", type=float, required=True, help="Remaining positive finite K2 budget")
    verify.add_argument("--video-timeout", type=float, default=None, help="Optional legacy cap; never a fresh budget")
    verify.add_argument("--startup-timeout", type=float, default=120)
    verify.add_argument("--open-checklist", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "preflight":
        result = preflight(args.model)
        if result.ready and args.load:
            backend = LocalTransformersBackend(args.model, device=args.device)
            try:
                backend.validate(args.timeout)
                result = replace(result, runtime_loaded=True)
            except Exception as exc:
                reason = "runtime_timeout" if isinstance(exc, TimeoutError) else "runtime_unavailable"
                result = replace(result, ready=False, errors=result.errors + (reason,))
            finally:
                backend.close()
        print(json.dumps(asdict(result), ensure_ascii=True, allow_nan=False))
        return 0 if result.ready else 2
    backend = None
    try:
        remaining_budget(args.budget_sec, args.video_timeout)
        if not number(args.startup_timeout) or args.startup_timeout <= 0:
            raise ValueError("invalid_startup_timeout")
        output = Path(args.output).resolve()
        if output in {Path(args.candidates).resolve(), Path(args.rules).resolve(), Path(args.video).resolve()}:
            raise ValueError("output_overlaps_input")
        candidates = strict_json(Path(args.candidates).read_text(encoding="utf-8"))
        rules = strict_json(Path(args.rules).read_text(encoding="utf-8"))
        if not isinstance(candidates, list) or not isinstance(rules, list) or not all(
            isinstance(item, dict) for item in candidates + rules
        ):
            raise ValueError("invalid_input")
        backend = LocalTransformersBackend(args.model, device=args.device)
        try:
            backend.validate(args.startup_timeout)
        except Exception:
            backend.close()
            backend = None
        report = verify_candidates(args.video, candidates, rules, backend=backend,
            max_calls=args.max_calls, enable_open_checklist=args.open_checklist,
            timeout_s=args.timeout, budget_sec=args.budget_sec, video_timeout_s=args.video_timeout)
        data = asdict(report)
        data["retained_candidates"] = report.retained_candidates
        output.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        counts = Counter(result.status for result in report.results)
        print(json.dumps({"status": "completed", "calls_used": report.calls_used,
            "confirmed": counts["confirmed"], "rejected": counts["rejected"],
            "uncertain": counts["uncertain"], "human_review_required": True}))
        return 0
    except Exception:
        print(json.dumps({"status": "error", "reason": "invalid_input_or_output"}))
        return 2
    finally:
        if backend is not None:
            backend.close()


if __name__ == "__main__":
    raise SystemExit(main())
