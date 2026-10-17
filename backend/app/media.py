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
        # Extended-size ftyp box (box_size == 1) is at least 24 bytes (4 + 4 + 8 + 4 + 4)
        brand_offset = 16 if box_size == 1 else 8
        min_box_len = 24 if box_size == 1 else 16
        first_box_len = extended_size if box_size == 1 else box_size
        if box_size == 1 and extended_size < min_box_len:
            return False
        if box_size not in (0, 1) and box_size < min_box_len:
            return False
        if len(head) < brand_offset + 4:
            return False
        major_brand = head[brand_offset : brand_offset + 4]
        if not all(32 <= b <= 126 for b in major_brand):
            return False

        # A valid video cannot consist solely of an ftyp box; it must have a subsequent
        # atom (e.g. moov, mdat, wide, free) containing video metadata/streams.
        if first_box_len > 0:
            if file_size < first_box_len + 8:
                return False
            with path.open("rb") as fh2:
                fh2.seek(first_box_len)
                next_head = fh2.read(8)
            if len(next_head) < 8:
                return False
            next_atom = next_head[4:8]
            if next_atom not in _ISOBMFF_ATOMS:
                return False

    return True
