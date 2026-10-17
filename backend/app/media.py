"""Upload validation and safe ffmpeg invocation for untrusted video files."""

from pathlib import Path

# Top-level ISO-BMFF / QuickTime atoms a real .mov/.mp4 can start with.
_ISOBMFF_ATOMS = frozenset({b"ftyp", b"moov", b"mdat", b"wide", b"free", b"skip"})

# Force the mov/mp4 demuxer and local-file-only I/O, so a playlist (HLS, concat)
# disguised as .mov cannot make ffmpeg read other files or reach the network.
FFMPEG_SAFE_INPUT = ("-protocol_whitelist", "file", "-f", "mov")


def looks_like_isobmff(path: Path) -> bool:
    try:
        file_size = path.stat().st_size
    except OSError:
        return False
    if file_size < 16:
        return False

    with path.open("rb") as fh:
        head = fh.read(32)
    if len(head) < 16:
        return False

    box_size = int.from_bytes(head[:4], "big")
    atom_type = head[4:8]
    if atom_type not in _ISOBMFF_ATOMS:
        return False

    if box_size == 1:
        # 64-bit extended size in bytes 8:16
        if len(head) < 16:
            return False
        extended_size = int.from_bytes(head[8:16], "big")
        if extended_size < 16 or extended_size > file_size:
            return False
    elif box_size != 0:
        # Box size must be at least 8 bytes and cannot exceed actual file size
        if box_size < 8 or box_size > file_size:
            return False

    if atom_type == b"ftyp":
        # Standard ftyp box is at least 16 bytes (size + type + major_brand + minor_version)
        if box_size not in (0, 1) and box_size < 16:
            return False
        major_brand = head[8:12]
        if not all(32 <= b <= 126 for b in major_brand):
            return False

    return True
