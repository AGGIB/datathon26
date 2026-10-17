"""Upload validation and safe ffmpeg invocation for untrusted video files."""

from pathlib import Path

# Top-level ISO-BMFF / QuickTime atoms a real .mov/.mp4 can start with.
_ISOBMFF_ATOMS = frozenset({b"ftyp", b"moov", b"mdat", b"wide", b"free", b"skip"})

# Force the mov/mp4 demuxer and local-file-only I/O, so a playlist (HLS, concat)
# disguised as .mov cannot make ffmpeg read other files or reach the network.
FFMPEG_SAFE_INPUT = ("-protocol_whitelist", "file", "-f", "mov")


def looks_like_isobmff(path: Path) -> bool:
    with path.open("rb") as fh:
        head = fh.read(16)
    if len(head) < 16:
        return False
    atom_type = head[4:8]
    if atom_type not in _ISOBMFF_ATOMS:
        return False
    box_size = int.from_bytes(head[:4], "big")
    if box_size not in (0, 1) and box_size < 8:
        return False
    if atom_type == b"ftyp":
        major_brand = head[8:12]
        if not all(32 <= b <= 126 for b in major_brand):
            return False
    return True
