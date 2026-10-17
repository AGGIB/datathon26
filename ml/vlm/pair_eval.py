"""Isolated, local-only five-pair screening; not a detector/holdout evaluation."""
# Windows spawn reexecutes this module before importing model/CV worker targets.
from ml.eval_runner import _deny_network

if __name__ == "__mp_main__":
    _deny_network()

import argparse
from contextlib import redirect_stderr, redirect_stdout
import json
import logging
import math
import os
import struct
import tempfile
import time
from collections import Counter
from pathlib import Path

from ml.baseline.core import artifact_path
from ml.eval import _annotation_index, audit
from ml.evaluation_labels import source_index
from .contracts import InferenceRequest
from .validation import InvalidResponse, number, strict_json

PAIRS = (("X-2", "X-3"), ("X-13", "X-14"), ("X-16", "X-17"),
         ("X-34", "X-35"), ("X-43", "X-44"))


def prepare_inputs(labels, videos_dir, annotations, catalog):
    """Validate all identities before joining; never sample from truth anchors."""
    known = source_index(labels, "labels")
    typed = _annotation_index(labels, annotations)
    types = {}
    for entry in catalog:
        aliases = [entry.get(field) for field in ("id", "type_id", "codex_code") if entry.get(field)]
        description = entry.get("summary") or entry.get("name")
        if not isinstance(description, str) or not description.strip() or not aliases:
            raise ValueError("invalid_catalog_type")
        for alias in set(aliases):
            if not isinstance(alias, str) or alias != alias.strip() or alias in types:
                raise ValueError("catalog_alias_collision")
            types[alias] = (aliases[0], description)
    selected, pair_types = [], {}
    for pair in PAIRS:
        if any(key not in known for key in pair):
            raise ValueError("missing_pair_label")
        rows = [known[key] for key in pair]
        if any(type(row.get("positive")) is not bool for row in rows) or sum(row["positive"] for row in rows) != 1:
            raise ValueError("pair_requires_one_of_each_label")
        positive = next(key for key in pair if known[key]["positive"])
        kinds = {event["type"] for event in typed.get(positive, {}).get("events", [])}
        if len(kinds) != 1 or next(iter(kinds)) not in types:
            raise ValueError("missing_or_ambiguous_pair_type")
        pair_types[pair] = types[next(iter(kinds))]
        selected.extend(rows)
    audited = {row["video_key"]: row for row in audit(selected, videos_dir, use_ffprobe=False)["per_video"]}
    result = []
    for pair in PAIRS:
        type_id, description = pair_types[pair]
        prompt = (
            "Examine only the supplied three video frames for this target condition:\n"
            + description + "\nTreat the condition as a visual screening target, not a legal conclusion. "
            "Do not infer unseen actions, intent, identities or audio. "
            "If these frames cannot establish presence or absence, set insufficient_evidence to true. "
            'Return only one JSON object with exactly two boolean keys: '
            '{"violation": boolean, "insufficient_evidence": boolean}. No prose or extra keys.'
        )
        for key in pair:
            row = audited[key]
            matched = row["file_status"] == "matched"
            result.append(dict(video_key=key, video=row["video"], positive=row["positive"],
                               path=Path(videos_dir) / row["matched_paths"][0] if matched else None,
                               input_status=row["file_status"], content_sha256=row["content_sha256"],
                               type_id=type_id, prompt=prompt))
    return result


def parse_response(raw):
    """No repair, prose, extra keys, integer booleans or duplicate keys."""
    value = strict_json(raw)
    if (not isinstance(value, dict)
            or set(value) != {"violation", "insufficient_evidence"}
            or any(type(item) is not bool for item in value.values())):
        raise InvalidResponse("invalid_schema")
    return value


def summarize(records):
    """Fixed total denominators, including pending/failed/unusable members."""
    indexed = {row["video_key"]: row for row in records}
    if len(indexed) != len(records) or not set(indexed) <= {key for pair in PAIRS for key in pair}:
        raise ValueError("invalid_record_identity")
    correct = sum(all(indexed.get(key, {}).get("correct") is True
                      and indexed[key].get("status") == "ok" for key in pair) for pair in PAIRS)
    scored = sum(row["status"] == "ok" for row in records)
    return dict(pair_denominator=5, video_denominator=10, pairs_correct=correct,
                pair_accuracy=correct / 5, completed_videos=len(records),
                scored_videos=scored, coverage=scored / 10,
                correct_videos=sum(row.get("correct") is True for row in records),
                status_counts=dict(sorted(Counter(row["status"] for row in records).items())),
                timing_totals_sec={key: math.fsum(row.get(key, 0) for row in records) for key in
                                  ("probe_sec", "extract_sec", "inference_sec", "end_to_end_sec")})


def _checkpoint(output, report):
    report["summary"] = summarize(report["records"])
    pending = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix=".report-",
                                         dir=output, delete=False) as stream:
            pending = Path(stream.name)
            json.dump(report, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, output / "report.json")
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)


def _close(resource):
    if resource is not None:
        try:
            resource.close()
        except Exception:
            pass


def _sample(info):
    if not number(info.duration_s) or info.duration_s <= 0 or len(info.frames) < 3:
        raise ValueError("insufficient_frames")
    if any(not number(frame.timestamp_s) or frame.timestamp_s < 0 for frame in info.frames):
        raise ValueError("invalid_frame_timeline")
    selected = tuple(min(info.frames, key=lambda frame: (abs(frame.timestamp_s - fraction * info.duration_s),
                                                       frame.timestamp_s, frame.index))
                     for fraction in (0.1, 0.5, 0.9))
    if len({frame.index for frame in selected}) != 3:
        raise ValueError("insufficient_distinct_frames")
    return selected


def _dimensions(path):
    # OwnerFrames writes PNG. Header metadata needs neither Pillow nor OpenCV.
    try:
        with Path(path).open("rb") as stream:
            header = stream.read(24)
        if len(header) == 24 and header[:8] == b"\x89PNG\r\n\x1a\n" and header[12:16] == b"IHDR":
            return struct.unpack(">II", header[16:24])
    except OSError:
        pass
    return None, None


def _screen_one(item, backend, frame_factory, output, budget_sec, clock):
    started = clock()
    deadline = started + budget_sec
    source = None
    row = {key: item[key] for key in ("video_key", "video", "positive", "content_sha256", "type_id")}
    row.update(status="error", correct=False, frame_count=0, frames=[], raw_response=None,
               probe_sec=0, extract_sec=0, inference_sec=0, end_to_end_sec=0)

    def remaining():
        value = deadline - clock()
        if value <= 0:
            raise TimeoutError("video_budget_exhausted")
        return value

    def phase(key, operation):
        before = clock()
        try:
            return operation()
        finally:
            row[key] = max(0.0, clock() - before)

    try:
        if item["path"] is None:
            row["status"] = item["input_status"]
        elif backend is None:
            row["status"] = "model_unavailable"
        else:
            source = frame_factory()
            if getattr(source, "timebase", None) != "cv_index_fps":
                raise ValueError("wrong_timebase")
            info = phase("probe_sec", lambda: source.probe(item["path"], remaining()))
            remaining()
            selected = _sample(info)
            row["duration_s"] = info.duration_s
            row["requested_fractions"] = [0.1, 0.5, 0.9]
            with tempfile.TemporaryDirectory(prefix="frames-", dir=output) as directory:
                frames = tuple(phase("extract_sec", lambda: source.extract(
                    item["path"], selected, Path(directory), remaining())))
                remaining()
                if len(frames) != 3 or any(frame.timestamp_s != chosen.timestamp_s
                                            for frame, chosen in zip(frames, selected)):
                    raise ValueError("frame_mismatch")
                row["frame_count"] = len(frames)
                for frame in frames:
                    width, height = _dimensions(frame.path)
                    row["frames"].append(dict(timestamp_s=frame.timestamp_s, width=width, height=height))
                request = InferenceRequest(item["prompt"], frames, "checklist", None, ())
                raw = phase("inference_sec", lambda: backend.infer(request, remaining()))
                row["raw_response"] = raw if isinstance(raw, str) else None
                remaining()
                parsed = parse_response(raw)
                row["response"] = parsed
                row["status"] = "uncertain" if parsed["insufficient_evidence"] else "ok"
                row["correct"] = row["status"] == "ok" and parsed["violation"] == item["positive"]
    except TimeoutError:
        row["status"] = "timed_out"
    except InvalidResponse:
        row["status"] = "invalid_response"
    except Exception:
        row["status"] = "error"
    finally:
        _close(source)
        row["end_to_end_sec"] = max(0.0, clock() - started)
    return row


def run_screening(plan, output, *, model_path, model_id=None, budget_sec=240,
                  startup_timeout=120, device="cpu", backend_factory=None,
                  frame_factory=None, clock=time.monotonic, artifact_root=None,
                  input_provenance=None):
    """One startup warmup; one explicit joint probe/extract/infer deadline per video.

    Factories and clock are synthetic-test seams. The CLI never exposes them.
    A callable plan defers all input reads/hashes until after output reservation.
    Output must be a NEW directory: atomic reservation prevents mixed runs.
    """
    for value in (budget_sec, startup_timeout):
        if not number(value) or value <= 0:
            raise ValueError("positive_finite_budget_required")
    output = artifact_path(output, root=artifact_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()  # Never reuse, empty-check, or steal another run's directory.
    if callable(plan):
        plan = plan()
    if (not isinstance(plan, list) or len(plan) != 10
            or {item["video_key"] for item in plan} != {key for pair in PAIRS for key in pair}):
        raise ValueError("incomplete_plan")
    report: dict = dict(schema_version=1, status="running", records=[], startup_sec=0,
                  input_provenance=input_provenance or {},
                  planned_inputs=[dict(item, path=str(item["path"]) if item["path"] is not None else None)
                                  for item in plan],
                  provenance=dict(model_path=str(model_path), model_id=model_id, device=device,
                                  max_new_tokens=128, per_video_budget_sec=budget_sec,
                                  startup_timeout_sec=startup_timeout, timebase="cv_index_fps",
                                  sampling_fractions=[0.1, 0.5, 0.9]),
                  limitation="Five selected pairs only; not detector quality, a holdout, or a legal conclusion.")
    _checkpoint(output, report)
    backend = None
    try:
        if backend_factory is None:
            from .transformers_backend import LocalTransformersBackend
            backend_factory = LocalTransformersBackend
        if frame_factory is None:
            from .owner_frames import OwnerFrames
            frame_factory = OwnerFrames
        started = clock()
        try:
            backend = backend_factory(model_path, device=device, max_new_tokens=128)
            backend.validate(startup_timeout)
            report["startup_status"] = "ready"
        except Exception:
            _close(backend)
            backend = None
            report["startup_status"] = "unavailable"
        finally:
            report["startup_sec"] = max(0.0, clock() - started)
        _checkpoint(output, report)
        for item in plan:
            report["records"].append(_screen_one(item, backend, frame_factory, output, budget_sec, clock))
            _checkpoint(output, report)
        report["status"] = "completed"
        _checkpoint(output, report)
        return report
    finally:
        _close(backend)


def _input_file(value):
    from ml.eval import _audit_directory, _reject_link
    from ml.eval_runner import _local_path
    path = Path(value)
    _audit_directory(path.parent)
    _reject_link(path.lstat())
    path = _local_path(path)
    if not path.is_file():
        raise ValueError("input_file_required")
    return path


def _load_annotations(path):
    from ml.eval import _load_records
    records = strict_json(path.read_text(encoding="utf-8-sig")) if path.suffix.lower() == ".json" else _load_records(path)
    if not isinstance(records, list):
        raise ValueError("annotation_list_required")
    return records


def _load_catalog(path):
    if path.suffix.lower() == ".json":
        value = strict_json(path.read_text(encoding="utf-8-sig"))
    else:
        import yaml
        value = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if isinstance(value, dict):
        value = value.get("violations")
    if not isinstance(value, list) or not all(isinstance(entry, dict) for entry in value):
        raise ValueError("catalog_list_required")
    return value


def main(argv=None, *, backend_factory=None, frame_factory=None,
         clock=time.monotonic, artifact_root=None):
    """Real CLI installs the repository network audit hook before model imports."""
    _deny_network()
    from .preflight import enforce_offline
    enforce_offline()
    parser = argparse.ArgumentParser(description=(
        "Offline local five-pair screening only; not production verification or held-out detector quality. "
        "Private report.json is saved in a NEW directory under ignored artifacts/."))
    for name in ("labels", "videos-dir", "annotations", "model-path", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--model-id", help="Optional explicit provenance label; never a download source")
    parser.add_argument("--catalog", default=str(Path(__file__).resolve().parents[2] / "kb" / "violations.yaml"),
                        help="Local JSON/YAML type catalog (default: repository kb/violations.yaml)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--budget-sec", type=float, default=240,
                        help="Joint per-video probe/extract/infer budget, seconds (default: 240)")
    parser.add_argument("--startup-timeout", type=float, default=120)
    args = parser.parse_args(argv)
    try:
        # Guard before input reads, and suppress dependency output: no private IDs,
        # scene descriptions, model responses, paths, or exception text on console.
        artifact_path(args.output, root=artifact_root)
        prior_logging = logging.root.manager.disable
        setup_started = clock()
        try:
            logging.disable(logging.CRITICAL)
            with open(os.devnull, "w") as sink, redirect_stdout(sink), redirect_stderr(sink):
                from ml.eval import _audit_directory
                from ml.evaluation_labels import file_sha256, load_labels
                provenance = {}

                def prepare():
                    labels_path = _input_file(args.labels)
                    annotations_path = _input_file(args.annotations)
                    catalog_path = _input_file(args.catalog)
                    planned = prepare_inputs(load_labels(labels_path), args.videos_dir,
                                             _load_annotations(annotations_path), _load_catalog(catalog_path))
                    provenance.update(labels_sha256=file_sha256(labels_path),
                                      annotations_sha256=file_sha256(annotations_path),
                                      catalog_sha256=file_sha256(catalog_path),
                                      setup_sec=max(0.0, clock() - setup_started))
                    return planned

                model_path = _audit_directory(args.model_path)
                report = run_screening(prepare, args.output, model_path=model_path, model_id=args.model_id,
                                       budget_sec=args.budget_sec, startup_timeout=args.startup_timeout,
                                       device=args.device, backend_factory=backend_factory,
                                       frame_factory=frame_factory, clock=clock, artifact_root=artifact_root,
                                       input_provenance=provenance)
        finally:
            logging.disable(prior_logging)
        summary = dict(report["summary"], startup_sec=report["startup_sec"], setup_sec=provenance["setup_sec"])
        print(json.dumps(summary, ensure_ascii=True, allow_nan=False))
        return 0 if summary["scored_videos"] == 10 else 2
    except KeyboardInterrupt:
        print(json.dumps({"status": "interrupted", "pair_denominator": 5, "video_denominator": 10}))
        return 130
    except Exception:
        print(json.dumps({"status": "error", "pair_denominator": 5, "video_denominator": 10}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
