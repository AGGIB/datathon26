"""Dev tool: run the full pipeline over the open dataset and score it against the labels.

    python -m ml.tools.run_dataset --out var/runs/run.json [--only X-26,X-27] [--budget 200]
    python -m ml.tools.run_dataset --score var/runs/run.json      # re-score a saved run
    python -m ml.tools.run_dataset --out var/runs/run.json --capture var/cache/judge   # also save perception+scene
    python -m ml.tools.run_dataset --rules-only var/cache/judge   # re-run only the rules on saved inputs (seconds)

Scoring: verdict per video (duplicate files X-4=X-47, X-5=X-48 counted once), timecode within ±2 s of the label,
type match against kb/annotations.csv.
"""

import argparse
import csv
import json
import logging
import pickle
import time
from contextlib import contextmanager
from pathlib import Path

from ml.labels import load_labels, normalize_video_name

DATA = Path("data/open")
TOLERANCE_SEC = 2.0
DUPLICATES = {"X-47": "X-4", "X-48": "X-5"}


def _videos(only: set[str]) -> list[Path]:
    paths = sorted(DATA.glob("videos/*.mov"), key=lambda p: int(normalize_video_name(p.name).split("-")[1]))
    return [p for p in paths if not only or normalize_video_name(p.name) in only]


@contextmanager
def _capturing_rules(store: dict):
    """Wrap rules.find_candidates so the (perception, scene) of each job can be saved for rules-only runs."""
    from ml import pipeline

    real = pipeline.rules.find_candidates

    def wrapper(perception, scene=None):
        store["inputs"] = (perception, scene)
        return real(perception, scene=scene)

    pipeline.rules.find_candidates = wrapper
    try:
        yield
    finally:
        pipeline.rules.find_candidates = real


def _save(path: Path, content: bytes) -> None:
    """Keep the last complete checkpoint if a dataset run is interrupted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_bytes(content)
    temporary.replace(path)


def run(out: Path, only: set[str], budget: float, capture: Path | None, resume: bool = False) -> dict:
    from ml.pipeline import analyze

    captured: dict = {}
    results: dict = json.loads(out.read_text()) if resume and out.exists() else {}
    for path in _videos(only):
        name = normalize_video_name(path.name)
        previous = results.get(name, {})
        if resume and previous.get("verdict") and (capture is None or (capture / f"{name}.pkl").exists()):
            logging.info("%s: resume, keeping completed result", name)
            continue
        started = time.monotonic()
        captured.clear()
        try:
            with _capturing_rules(captured):
                r = analyze(str(path), time_budget_sec=budget)
            if capture and "inputs" in captured:
                _save(capture / f"{name}.pkl", pickle.dumps(captured["inputs"]))
            results[name] = {
                "verdict": r["verdict"],
                "violations": [{"type_id": v.get("type_id"), "t": v["timecode_sec"], "conf": v.get("confidence")}
                               for v in r["violations"]],
            }
        except Exception as exc:  # noqa: BLE001 — a dev tool keeps going and records the failure
            results[name] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
        results[name]["sec"] = round(time.monotonic() - started, 1)
        _save(out, json.dumps(results, ensure_ascii=False, indent=1).encode("utf-8"))
        logging.info("%s %s", name, results[name])
    return results


def rules_only(capture: Path) -> dict:
    from ml.assemble import clamp_time, select
    from ml.kb import catalog
    from ml.rules import find_candidates

    results = {}
    types = catalog()
    for pkl in sorted(capture.glob("*.pkl")):
        perception, scene = pickle.loads(pkl.read_bytes())
        cands = select(find_candidates(perception, scene=scene), types)
        results[pkl.stem] = {
            "verdict": "violation" if cands else "no_violation",
            "violations": [{"type_id": c.type_id, "t": clamp_time(c.t_peak, perception.info.duration),
                            "conf": c.confidence} for c in cands],
            "sec": 0.0,
        }
    return results


def score(results: dict) -> None:
    labels = {lb.video: lb for lb in load_labels(DATA / "open_labels.xlsx")}
    with open("kb/annotations.csv", encoding="utf-8") as handle:
        truth = {r["video"]: r["type_id"] for r in csv.DictReader(handle)}
    tp = fp = fn = tn = t_ok = type_ok = 0
    errors = failed_positive = 0
    rows, secs = [], []
    for name, res in sorted(results.items(), key=lambda kv: int(kv[0].split("-")[1])):
        if name in DUPLICATES and DUPLICATES[name] in results:
            continue
        if "error" in res:
            errors += 1
            failed_positive += labels[name].has_violation
            rows.append(f"{name:5} ERROR {res['error'][:80]}")
            continue
        secs.append(res["sec"])
        lb = labels[name]
        pred = res["verdict"] == "violation"
        tp, fp = tp + (lb.has_violation and pred), fp + (not lb.has_violation and pred)
        fn, tn = fn + (lb.has_violation and not pred), tn + (not lb.has_violation and not pred)
        anchors = [lb.t_sec] + ([labels[d].t_sec for d, src in DUPLICATES.items() if src == name and d in labels])
        hit_t = pred and lb.has_violation and any(
            abs(v["t"] - a) <= TOLERANCE_SEC for v in res["violations"] for a in anchors if a is not None)
        hit_type = pred and any(v["type_id"] == truth.get(name) for v in res["violations"])
        t_ok, type_ok = t_ok + hit_t, type_ok + (lb.has_violation and hit_type)
        flag = "TP" if lb.has_violation and pred else "FP" if pred else "FN" if lb.has_violation else "TN"
        got = [(v["type_id"], v["t"]) for v in res["violations"]][:3]
        rows.append(f"{name:5} {flag} {'t✓' if hit_t else '  '} {'type✓' if hit_type else '     '} "
                    f"gt={truth.get(name, '?')[:30]:30} pred={got}")
    print("\n".join(rows))
    total = tp + fp + fn + tn
    print(f"\nvideos={total + errors} completed={total} errors={errors} TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"recall={tp / max(1, tp + fn + failed_positive):.2f} precision={tp / max(1, tp + fp):.2f} "
          f"accuracy={(tp + tn) / max(1, total + errors):.2f}")
    print(f"timecode ±{TOLERANCE_SEC:.0f}s on TP: {t_ok}/{tp}   "
          f"type match on violations: {type_ok}/{tp + fn + failed_positive}")
    if secs:
        print(f"time per video: max {max(secs):.1f}s, mean {sum(secs) / len(secs):.1f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("var/runs/run.json"))
    ap.add_argument("--only", default="")
    ap.add_argument("--budget", type=float, default=200.0)
    ap.add_argument("--resume", action="store_true", help="keep completed results from this same code/config run")
    ap.add_argument("--score", type=Path, help="only score a saved run")
    ap.add_argument("--capture", type=Path, help="save each video's (perception, scene) here")
    ap.add_argument("--rules-only", type=Path, help="re-run only the rules on saved (perception, scene) and score")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    if args.score:
        score(json.loads(args.score.read_text()))
        return
    if args.rules_only:
        score(rules_only(args.rules_only))
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    only = {v.strip().upper() for v in args.only.split(",") if v.strip()}
    score(run(args.out, only, args.budget, args.capture, args.resume))


if __name__ == "__main__":
    main()
