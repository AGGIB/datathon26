"""Run with python -m ml.baseline; no heavy imports until an operation needs them."""
import argparse
import json
import sys

from .core import REPO_ROOT, artifact_path, read_json, write_json


def argument_parser():
    parser = argparse.ArgumentParser(description="Offline local SigLIP backup candidates, not legal verdicts")
    parser.add_argument("--artifact-root", default=str(REPO_ROOT),
                        help="Project root; every output must be within its artifacts/ directory")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("extract", "extract-dataset"):
        extract = commands.add_parser(name, help="Extract local actual-PTS features; no labels or fitting")
        if name == "extract":
            extract.add_argument("--video", required=True)
            extract.add_argument("--video-id", help="Literal ID; default is complete filename including extension")
            extract.add_argument("--output", required=True)
        else:
            extract.add_argument("--videos-dir", required=True, help="Local directory; all .mov files, non-recursive")
            extract.add_argument("--output-dir", required=True, help="New/empty directory inside artifacts/")
        extract.add_argument("--weights", required=True, help="Existing local SigLIP directory; never downloads")
        extract.add_argument("--fps", type=float, default=1.0)
        extract.add_argument("--max-frames", type=int, default=3600)
        extract.add_argument("--batch-size", type=int, default=8)
        extract.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
        extract.add_argument("--cpu-threads", type=int, default=2)
        extract.add_argument("--seed", type=int, default=0)
        extract.add_argument("--ffmpeg", default="ffmpeg")
        extract.add_argument("--decode-timeout", type=float, default=600.0)
    for name in ("train", "evaluate"):
        command = commands.add_parser(name, help="Explicit angle-grouped " + name + "; no inferred split")
        command.add_argument("--manifest", required=True, help="UTF-8 JSON list of annotated video records")
        command.add_argument("--train-angle", action="append", required=True)
        command.add_argument("--validation-angle", action="append", required=True)
        command.add_argument("--holdout-angle", action="append", required=True)
        command.add_argument("--cache", action="append", required=True)
        command.add_argument("--model", required=True, help="Output NPZ for train, input NPZ for evaluate")
        command.add_argument("--report", required=True, help="Diagnostic holdout candidate JSON, not official metrics")
        if name == "train":
            command.add_argument("--positive-radius", type=float, default=2.0)
            command.add_argument("--seed", type=int, default=0)
            command.add_argument("--epochs", type=int, default=1000)
            command.add_argument("--learning-rate", type=float, default=0.1)
            command.add_argument("--l2", type=float, default=0.001)
            command.add_argument("--threshold", type=float, default=0.5)
            command.add_argument("--window-seconds", type=float, default=2.0)
    predict = commands.add_parser("predict", help="Generate candidate-only JSON from a numeric cache")
    predict.add_argument("--cache", required=True)
    predict.add_argument("--model", required=True)
    predict.add_argument("--output", required=True)
    predict.add_argument("--threshold", type=float)
    predict.add_argument("--window-seconds", type=float)
    return parser


def main(argv=None):
    parser = argument_parser()
    args = parser.parse_args(argv)
    # Windows default encoding may otherwise corrupt Cyrillic IDs/descriptions.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    try:
        from .features import extract_video, load_cache
        from .classifier import evaluate_holdout, load_model, predict_cache, save_model, train_classifier
        if args.command == "extract-dataset":
            from .dataset import extract_dataset
            report = extract_dataset(args.videos_dir, args.weights, args.output_dir, fps=args.fps,
                                     max_frames=args.max_frames, batch_size=args.batch_size, device=args.device,
                                     cpu_threads=args.cpu_threads, seed=args.seed, ffmpeg=args.ffmpeg,
                                     timeout_s=args.decode_timeout, artifact_root=args.artifact_root)
            summary = {key: value for key, value in report.items() if key != "videos"}
            print(json.dumps(summary, ensure_ascii=False))
            return 2 if report["failed_videos"] or report["capped_videos"] else 0
        elif args.command == "extract":
            result = extract_video(args.video, args.weights, args.output, video_id=args.video_id,
                                   fps=args.fps, max_frames=args.max_frames, batch_size=args.batch_size,
                                   device=args.device, cpu_threads=args.cpu_threads, seed=args.seed,
                                   ffmpeg=args.ffmpeg, timeout_s=args.decode_timeout, artifact_root=args.artifact_root)
            print(json.dumps(result, ensure_ascii=False))
        elif args.command == "predict":
            target = artifact_path(args.output, root=args.artifact_root)
            prediction = predict_cache(load_model(args.model), load_cache(args.cache),
                                       threshold=args.threshold, window_s=args.window_seconds)
            write_json(target, prediction)
            print(json.dumps({"status": "ok", "output": str(target), "video_id": prediction["video_id"]},
                             ensure_ascii=False))
        elif args.command in ("train", "evaluate"):
            report_path = artifact_path(args.report, root=args.artifact_root)
            model_path = artifact_path(args.model, root=args.artifact_root) if args.command == "train" else args.model
            rows = read_json(args.manifest)
            if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
                raise ValueError("manifest must be a JSON list of video objects")
            caches = [load_cache(path) for path in args.cache]
            if args.command == "train":
                model = train_classifier(caches, rows, args.train_angle, args.holdout_angle,
                                         positive_radius_s=args.positive_radius, seed=args.seed, epochs=args.epochs,
                                         learning_rate=args.learning_rate, l2=args.l2,
                                         threshold=args.threshold, window_s=args.window_seconds,
                                         validation_angles=args.validation_angle)
                from .core import canonical_video_id
                holdout_ids = {canonical_video_id(value) for value in model["metadata"]["holdout_video_ids"]}
                report = evaluate_holdout(model, [cache for cache in caches if canonical_video_id(cache["video_id"]) in holdout_ids],
                                          rows, args.train_angle, args.holdout_angle,
                                          validation_angles=args.validation_angle)
                save_model(model_path, model)
            else:
                report = evaluate_holdout(load_model(model_path), caches, rows, args.train_angle, args.holdout_angle,
                                          validation_angles=args.validation_angle)
            write_json(report_path, report)
            print(json.dumps({"status": "ok", "report": str(report_path), "n_holdout_videos": report["n_videos"]},
                             ensure_ascii=False))
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, "baseline error: " + str(exc) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
