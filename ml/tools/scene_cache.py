"""Dev helper: cached perception runs under var/cache/ so scene experiments do not re-run the detector.

    python -m ml.tools.scene_cache X-20,X-21      # warm the cache
"""

import argparse
import logging
import pickle
from pathlib import Path

from ml.labels import normalize_video_name
from ml.perception import Perception, perceive

DATA = Path("data/open/videos")
CACHE = Path("var/cache/perception")

log = logging.getLogger(__name__)


def video_paths(data_dir: Path = DATA) -> dict[str, Path]:
    return {normalize_video_name(p.name): p for p in data_dir.glob("*.mov")}


def cached_perception(name: str, path: Path, detector_factory, cache_dir: Path = CACHE) -> Perception:
    """Load perception for a dataset video from cache, running the detector on a miss.

    Dev-only: pickle is acceptable because the files are written by this tool into the local,
    gitignored var/cache/ and never come from users or the network.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    file = cache_dir / f"{name}.pkl"
    if file.exists():
        with file.open("rb") as fh:
            return pickle.load(fh)
    log.info("perception cache miss for %s, running detector", name)
    result = perceive(path, detector_factory())
    with file.open("wb") as fh:
        pickle.dump(result, fh)
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", help="comma-separated names like X-20,X-21")
    args = ap.parse_args()
    paths = video_paths()
    detector = None

    def factory():
        nonlocal detector
        if detector is None:
            from ml.detect import Detector

            detector = Detector()
        return detector

    for name in (v.strip().upper() for v in args.videos.split(",") if v.strip()):
        p = cached_perception(name, paths[name], factory)
        log.info("%s: %d frames, %d tracks, %.1fs", name, len(p.frames), len(p.tracks), p.elapsed_sec)


if __name__ == "__main__":
    main()
