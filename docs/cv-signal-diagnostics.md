# Local diagnostics for source time and door measurements

These tools expose information needed for subsequent experiments. They do not
change production sampling, evidence, door rules, thresholds or model calls, and
do not establish a precision improvement.

## Source time

[source_time_map.py](../tools/source_time_map.py) preserves integer frame PTS,
rational time base and explicit stream origin. It proposes a nearest-timestamp
mapping with explicit ambiguous/unknown outcomes. A metadata index is not proof
of the identity of a decoded frame or evidence image. See the
[usage and limitations](source-time-map.md).

## Door measurement provenance

[scene_measurement_diagnostic.py](../tools/scene_measurement_diagnostic.py)
accepts already prepared arrays and returns a numeric dataclass sidecar:

```python
from tools.scene_measurement_diagnostic import diagnose_door

diagnostic = diagnose_door(
    times, grays, visible_masks, door_roi, global_gains, lighting_reject,
)
```

The caller must supply the same arrays and parameters used by the production
measurement stage. The helper performs no file/network I/O, decoding or model
inference. Connecting it to a live job remains separate work. It cannot recover
lost raw measurements from a cached final `SceneState`.

The sidecar distinguishes raw change, median smoothing and the finished score.
It records current/reference/common visible fractions, usable cells, lighting
rejection and explicit unknown measurements. A held high score does not become
a fresh observation. `raw_measurement_age_s` describes the last valid raw
measurement, not the ancestry of a smoothed score; smoothing windows and future
samples are reported separately.

Reference provenance includes start/median mode, supplied source indices and a
numeric fingerprint. A median reference can combine multiple frames. Neither
reference mode establishes that the door was physically closed; semantic state
remains unknown. Supplied indices/timestamps do not claim container-frame or
actual-PTS identity.

Existing `_region_series` and `_finish` computations are reused, and synthetic
checks compare their outputs exactly. These checks cover occlusion, lighting,
missing references, held values and smoothing. They verify observability, not
the correctness of a door alarm on private video. Keep any real-data sidecar in
ignored local `var/`; it must not include image arrays or free-text observations.
