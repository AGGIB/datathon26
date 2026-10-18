"""Synthetic-only fault injection at K2, without weights or production changes."""
import argparse
import base64
import io
import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run(output):
    from ml.eval_runner import _deny_network
    _deny_network()
    import cv2
    import numpy as np
    from PIL import Image
    from backend.app.schemas import AnalysisResult
    from ml import pipeline
    from ml.baseline.core import artifact_path, write_json
    from ml.candidates import Candidate
    from ml.shared_detector import SharedDetector

    target = artifact_path(output)
    target.mkdir()  # A fresh artifact directory; never overwrite a previous run.
    source = target / "synthetic.mov"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    if not writer.isOpened():
        raise RuntimeError("Synthetic MOV encoder unavailable")
    for i in range(30):
        writer.write(np.full((48, 64, 3), i * 5, np.uint8))
    writer.release()

    class StubDetector:
        device = "cpu"
        input_size = 640

        def __init__(self):
            self.frames_seen = 0

        def detect(self, frames):
            self.frames_seen += len(frames)
            return [[] for _ in frames]

    candidate = Candidate(type_id="cash_left_on_desk_when_leaving", t_start=0.0,
                          t_end=1.0, t_peak=0.5, confidence=0.9,
                          observation="synthetic fixture, no real event",
                          evidence_times=(0.5,))
    rows = []
    for case in ("positive_control", "zero_processed_frames", "all_evidence_lost"):
        started = time.perf_counter()
        detector = StubDetector()
        with patch.dict(os.environ, {"ML_SCENE": "0", "ML_VLM_DISCOVERY": "0"}), \
                patch.object(pipeline, "_SHARED", SharedDetector(factory=lambda: detector)), \
                patch("ml.rules.find_candidates", return_value=[candidate]) as judge:
            try:
                if case == "all_evidence_lost":
                    with patch("ml.evidence.frames_at", side_effect=ValueError("synthetic decode failure")):
                        result = pipeline.analyze(str(source))
                else:
                    result = pipeline.analyze(str(source), time_budget_sec=0.01 if case == "zero_processed_frames" else None)
                AnalysisResult.model_validate(result)
                jpeg_count = 0
                for v in result["violations"]:
                    for frame in v["evidence_frames"]:
                        with Image.open(io.BytesIO(base64.b64decode(frame["image_base64"], validate=True))) as im:
                            if im.format != "JPEG":
                                raise ValueError("Evidence is not JPEG")
                            im.verify()
                        jpeg_count += 1
                outcome = result["verdict"]
                acceptable = outcome == "violation" if case == "positive_control" else False
                rows.append({"case": case, "outcome": outcome, "violation_count": len(result["violations"]),
                             "valid_jpegs": jpeg_count, "rule_calls": judge.call_count,
                             "acceptance_passed": acceptable})
            except (RuntimeError, ValueError) as exc:
                rows.append({"case": case, "outcome": "raised", "error_class": type(exc).__name__,
                             "acceptance_passed": case != "positive_control"})
        rows[-1]["elapsed_s"] = time.perf_counter() - started
        rows[-1]["stub_frames_seen_including_calibration"] = detector.frames_seen
    report = {"synthetic_only": True, "real_model_calls": 0, "service_sla_measured": False,
              "cases": rows, "acceptance_passed": all(r["acceptance_passed"] for r in rows)}
    write_json(target / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run(args.output)
    print(json.dumps(report))
    sys.exit(0 if report["acceptance_passed"] else 2)
