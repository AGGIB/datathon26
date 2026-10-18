# Metadata-only source time diagnostic

`tools/source_time_map.py` is an independent standard-library helper. It does
not modify production CV sampling, `Frame(t,t)`, evidence rendering or API K1.

```sh
python tools/source_time_map.py --probe-json LOCAL_FFPROBE.json \
  --times 0 0.04 0.1 --out var/source-time-map.json
# Explicitly authorized local media only; no network protocols:
python tools/source_time_map.py --video LOCAL_FILE.mp4 --stream-index 0 \
  --times 0 0.04 --out var/source-time-map.json
```

The second command asks local ffprobe for frame metadata and hashes the source
and ffprobe executable before/after. ffprobe may internally decode; this helper
never emits or inspects image pixels. Tests use synthetic metadata and a tiny
locally generated clip, with no private video reads.

Integer `frame.pts`, positive rational `stream.time_base`, and explicit
`stream.start_pts` are retained exactly. Zero and negative PTS are valid.
The declared origin is **stream_start_pts**, which is not automatically the
container, organiser annotation, OpenCV or evidence timebase.

The proposed selector finds the nearest observed presentation timestamp to a
requested stream-relative time. It is **not** the production index/FPS selector.
Missing PTS/origin/timebase stay unknown; best-effort timestamps, nominal FPS and
first-frame guesses never fill them. Missing any frame PTS conservatively makes
nearest selection unknown. Requests outside the observed PTS range are unknown,
including an unverified tail after the last timestamp; no endpoint clamping.
Repeated PTS and equidistant neighbours are ambiguous, with bounded candidate
indices rather than a fabricated unique choice. Multiple video streams require
explicit selection; each frame remains bound to its reported stream.

`metadata_frame_index` is zero-based within the selected ffprobe stream;
`probe_frame_position` records its position in the original metadata array.
Neither proves a decoded OpenCV index. Reports always state
`decoded_frame_identity_verified=false` and
`live_evidence_roundtrip_verified=false`. Pixel/frame/evidence round-trip testing
and an adopted production timestamp contract remain separate work.

Outputs are fresh ignored `var/*.json`, contain rational numeric mapping and
input/tool SHA256 only, and do not copy filenames, tags or private free text.
Timeout, byte/frame/query limits and fixed CLI error codes bound diagnostics.
