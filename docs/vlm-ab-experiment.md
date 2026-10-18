# Local spatial-verifier A/B protocol

This development experiment changes the verifier's view of the same source
frames. Fourteen saved proposal windows across the existing content cohort are
fixed before inference. Each arm makes at most fourteen fresh verifier calls
(twenty-eight total); the first-stage proposals are cached, not regenerated.
The model, physical-atom vocabulary, source anchors, owner confidence gate,
deadlines and call limits are fixed. No training or new checkpoint is involved.

Compare machine-readable prediction maps, never narratives or model confidence
as truth. Keep baseline CV in its own file: an A/B reference arm can already
contain VLM events, so exact reference retention is not exact CV retention.

```sh
python -m ml.tools.vlm_compare \
  --baseline var/reference.json --probe var/candidate.json \
  --cv-baseline var/cv.json --audit var/candidate.audit.json \
  --labels data/labels.xlsx --annotations kb/annotations.csv \
  --hash-report var/actual-hash-report.json \
  --out var/ab-comparison.json --expect-complete
```

All inputs are explicit local files. Labels are existing CSV/XLSX organiser
tables; annotations contain one primary `video,type_id,t_peak` row per file.
The hash report is an evaluator report whose content aggregation says actual,
authoritatively verified hashes. The comparator trusts that recorded provenance;
it does not reread or decode media. Alias collisions and conflicting duplicate
labels/annotations fail closed. Quality is counted once per exact SHA256 group,
with alternative primary anchors and worst member status handled by `ml.eval`.

The comparison delegates binary metrics and event matching to `ml.eval.score`.
For the primary joint criterion it first projects predictions to the primary
type: a closer unrelated event cannot steal a correct type/time match. A type
match and a time match on two different events never form a joint hit. Confidence,
intervals and multiplicity are included in exact CV retention, but confidence
alone does not create a new semantic event. Extra positive-clip events remain
unadjudicated except for at most one newly corroborated primary joint event.
Known-clean additions are measured against the separate CV file, including
false additions inherited unchanged from the A/B reference. Without CV their
count is unknown; reference-relative additions are a separate diagnostic.

The metric gate requires complete content coverage, valid responses, exact CV
retention for both arms, no lost primary type or joint hits, zero new events on
known-clean content and at least one new primary type or joint hit. Missing audit
metadata is unknown, never a semantic negative or proof of completed inference.
Standard audit source stability, statuses and actual-call accounting are needed
for the additional `eligible_for_fresh_full_run` flag. Custom screening details
can be retained as input provenance without claiming a complete fresh run.

Only a passing screening candidate proceeds to a fresh full-cohort run with both
model stages regenerated. The reused development cohort establishes neither an
unseen-camera result nor exhaustive type precision. Warm CV-replay measurements
exclude real CV and evidence rendering; a separate nonempty full-job smoke must
exercise verification, addition assembly and original-video JPEG evidence.
No comparison flag certifies a full-pipeline SLA or authorizes production use.

Outputs are fresh numeric JSON under the project's ignored `var/`; descriptions,
observations, images, human reviewer names and exception text are never copied.

Conditional owner selection can be checked without running a model or decoding
media, using the saved screen report and explicitly trusted local pipeline
pickle caches:

```sh
python ml/tools/vlm_selection_replay.py --source-root REPO --frozen SOURCE_SHA \
  --probe SCREEN.json --audit SCREEN.screening-details.json \
  --cv-baseline CV.json --trusted-cache TRUSTED_CACHE --trust-local-pickle-cache \
  --hash-report VERIFIED_HASHES.json --out var/selection-replay.json
```

This checks complete CV candidates, including observation text, evidence order,
boxes and exact confidence, through the real discovery, gap, reserved-slot and
assembly selection gates. The report contains numeric results and hashes only.
Typed findings are recovered from their returned witness times, so relocated
sampled anchors are supported without assuming the cached proposal's times.
Missing findings and low confidence do not establish a semantic negative.
The check does not exercise inference, media decoding, JPEG evidence, deadlines
or a complete job, and must not be described as a fresh full evaluation.

The optional full-window temporal witness probe uses one original camera view
and the same already decoded frames. It is a new experimental decoder profile,
not an evaluation on another camera angle:

```sh
python -m ml.tools.vlm_type_probe --model LOCAL_CHECKPOINT --python LOCAL_MLX_PYTHON \
  --videos LOCAL_VIDEOS --baseline CV.json --out var/temporal-development.json \
  --temporal-witness-actions --max-tokens 768 --min-confidence 0.95
```

`--temporal-witness-actions` overrides the other action decoder flags and allows
four physical calls per video. The command explicitly selects 768 output tokens;
the CLI default remains 512. Supported findings use the returned witnesses'
original CV index/FPS timestamps. These are not new claims about container PTS.
The profile does not enable runtime or pipeline configuration. Audit diagnostics
retain only coded status/reason and validated 1-based source witness indices.
