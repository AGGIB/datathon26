"""Dev tool: run perception over the dataset and print events next to the labels.

    python -m ml.tools.scan_events [--only X-30,X-33] [--fps 4]
"""

import argparse
import csv
from pathlib import Path

from ml.detect import Detector
from ml.labels import load_labels, normalize_video_name
from ml.perception import SAMPLE_FPS, perceive

DATA = Path("data/open")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--fps", type=float, default=SAMPLE_FPS)
    args = ap.parse_args()

    labels = {lb.video: lb for lb in load_labels(DATA / "open_labels.xlsx")}
    notes = {r["video"]: r for r in csv.DictReader(open("kb/annotations.csv", encoding="utf-8"))}
    wanted = {v.strip().upper() for v in args.only.split(",") if v.strip()}
    videos = sorted(DATA.glob("videos/*.mov"), key=lambda p: int(normalize_video_name(p.name).split("-")[1]))

    detector = Detector()
    total = 0.0
    for path in videos:
        name = normalize_video_name(path.name)
        if wanted and name not in wanted:
            continue
        p = perceive(path, detector, sample_fps=args.fps)
        total += p.elapsed_sec
        lb = labels[name]
        note = notes.get(name, {})
        tag = f"VIOL@{lb.t_sec:.0f}s" if lb.has_violation else "clean"
        roles = p.roles
        print(f"\n{name} [{tag}] {note.get('type_id', '')} dur={p.info.duration:.1f}s "
              f"tracks={len(p.all_tracks)} in_room={len(p.tracks)} took={p.elapsed_sec:.1f}s")
        if roles is not None:
            print(f"   roles: cashier={roles.cashier_ids} visitors={roles.visitor_ids} rejected={len(roles.rejected)}")
        print(f"   codex: {note.get('what_visible', '')[:110]}")
        for e in p.events:
            extra = f"..{e.t_end:.1f}" if e.t_end is not None else ""
            extra += f" n={e.count}" if e.count else ""
            ids = e.track_ids or ((e.track_id,) if e.track_id is not None else ())
            print(f"   {e.t:6.1f}{extra:12} {e.kind:22} {ids or ''}")
    print(f"\ntotal perception time {total:.1f}s")


if __name__ == "__main__":
    main()
