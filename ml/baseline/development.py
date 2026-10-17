"""Offline development CLI: audited caches -> frozen folds -> in-memory OOF only."""
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

from .core import artifact_path, write_json


def _atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".pending")
    write_json(temporary, value)
    os.replace(temporary, path)


def run_experiment(index_path, videos_dir, labels_path, manifest_path, weights, output,
                   *, frozen_cv_path=None, seed=0, max_folds=5, artifact_root=None):
    from ml import eval as evaluation
    from ml.evaluation_labels import file_sha256
    from . import diagnostic, diagnostic_io

    # Require an existing local parent; reserve a fresh directory before any private reads.
    evaluation._audit_directory(Path(output).parent)
    try:
        metadata = Path(output).lstat()  # Inspect the raw final component, including dangling links.
    except FileNotFoundError:
        pass
    else:
        evaluation._reject_link(metadata)
        raise FileExistsError("Diagnostic output must be a fresh directory")
    target = artifact_path(output, root=artifact_root)
    target.mkdir()  # Atomic ownership; existing/other-writer output is never overwritten.
    started = time.perf_counter()
    try:
        samples, audit = diagnostic_io.audit_inputs(index_path, videos_dir, labels_path, manifest_path, weights)
        audit_s = time.perf_counter() - started
        cv, comparison = (None, {"status": "missing", "reason": "actual_hash_bound_frozen_cv_not_supplied"})
        if frozen_cv_path is not None:
            cv, comparison = diagnostic_io.load_frozen_cv(frozen_cv_path, audit)
        plan = diagnostic.make_plan(samples, seed=seed, max_folds=max_folds)
        code = {p.name: file_sha256(p) for p in
                (Path(__file__), Path(diagnostic.__file__), Path(diagnostic_io.__file__))}
        manifest = {"report_kind": "development_input_manifest", **diagnostic_io.FLAGS,
                    "manifest_sha256": plan["manifest_sha256"], "input_sha256": audit["input_sha256"],
                    "snapshot_sha256": audit["snapshot_sha256"], "code_sha256": code,
                    "comparison_provenance": comparison,
                    "python": platform.python_version(), "numpy": diagnostic.np.__version__,
                    "pooling": "arithmetic_mean_of_all_cached_frames_float64",
                    "content_units": [{k: s[k] for k in ("content_sha256", "label", "members", "source_groups")}
                                      for s in samples]}
        _atomic_json(target / "cache-audit.json", audit)
        _atomic_json(target / "folds.json", plan)
        _atomic_json(target / "input-manifest.json", manifest)
        # All choices and input hashes are on disk BEFORE any estimator sees outer outcomes.
        report = diagnostic.run_diagnostic(samples, plan, frozen_cv=cv)
        report.update(comparison_provenance=comparison, input_manifest_sha256=file_sha256(target / "input-manifest.json"))
        report["timings_s"].update(cache_audit=audit_s, full_numeric_job=time.perf_counter() - started,
                                    extraction_included=False)
        _atomic_json(target / "oof-predictions.json", report["oof"])
        _atomic_json(target / "report.json", report)
        return report
    except (ValueError, TypeError, OSError, RuntimeError, ArithmeticError, KeyError) as exc:
        _atomic_json(target / "failure.json", {"report_kind": "development_diagnostic_failure",
                     **diagnostic_io.FLAGS, "status": "failed", "error_class": type(exc).__name__})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("index-path", "videos-dir", "labels-path", "manifest-path", "weights", "output"):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--frozen-cv-path")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-folds", type=int, default=5)
    args = parser.parse_args(argv)
    # Guard before optional numerical imports; no model, decoder, sockets or downloads needed.
    from ml.eval_runner import _deny_network
    _deny_network()
    try:
        report = run_experiment(**vars(args))
    except (ValueError, TypeError, OSError, RuntimeError, ArithmeticError, KeyError) as exc:
        print(json.dumps({"status": "failed", "error_class": type(exc).__name__}))
        return 1
    metrics = report["individual"]["siglip"]
    prior = report["individual"]["train_prior"]
    complete = metrics["coverage"]["evaluated"] == metrics["coverage"]["total"]
    compared = report["joint"]["coverage"]["evaluated"] == report["n_unique_contents"]
    print(json.dumps({"development_only": True, "unique_contents": report["n_unique_contents"],
          "siglip": {k: metrics[k] for k in ("TP", "FP", "FN", "TN", "coverage", "balanced_accuracy")},
          "train_prior": {k: prior[k] for k in ("TP", "FP", "FN", "TN", "balanced_accuracy")},
          "frozen_cv_status": report["comparison_provenance"]["status"],
          "comparison_complete": compared, "oof_complete": complete}, allow_nan=False))
    return 0 if complete and compared else 2


if __name__ == "__main__":
    sys.exit(main())
