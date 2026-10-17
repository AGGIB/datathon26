"""Upload validation and safe ffmpeg invocation for untrusted video files."""

from pathlib import Path

# Top-level ISO-BMFF / QuickTime atoms a real .mov/.mp4 can start with.
_ISOBMFF_ATOMS = frozenset({b"ftyp", b"moov", b"mdat", b"wide", b"free", b"skip"})

# Force the mov/mp4 demuxer and local-file-only I/O, so a playlist (HLS, concat)
# disguised as .mov cannot make ffmpeg read other files or reach the network.
FFMPEG_SAFE_INPUT = ("-protocol_whitelist", "file", "-f", "mov")


def looks_like_isobmff(path: Path) -> bool:
    with path.open("rb") as fh:
        head = fh.read(8)
    return len(head) == 8 and head[4:8] in _ISOBMFF_ATOMS
