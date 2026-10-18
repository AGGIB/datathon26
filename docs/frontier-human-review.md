# Private annotation scaffold for three frontier actions

Use existing development videos. No recording, model, new UI or new dependency
is needed. The scaffold groups exact bytes, creates empty human work files and
validates their structure. It never creates an event, reviewer, date, negative,
primary-label inference, camera group or legal verdict.

The three type IDs are `key_left_in_safe_lock`, `personal_items_on_desk` and
`food_or_drink_at_workplace`. Use the physical criteria and ordinary contrasts in
[the team rubric](vlm-data-without-recording.md). Review every occurrence and
contrast for each type across the full clip, or explicitly declare a partial
interval. Primary-only labels do not establish background for anything else.

## Create and check a private bundle

Run from the checkout root, using real local paths and the full source revision.
The hash report must already declare actual authoritatively verified hashes.
`init` independently streams every selected source file's bytes and compares
its hash, then uses the existing local-only numerical ffprobe helper for each
content. It never decodes or writes images. Symlinks/reparse points in inputs
are rejected; pass the explicit resolved source directory when a dataset link
is used. Choose a fresh direct child of the private artifact root.

```sh
python tools/frontier_review.py init \
  --videos-dir /absolute/local/videos --hash-report /absolute/local/hash-report.json \
  --source-revision FULL_40_HEX_REVISION --output var/frontier-review
python tools/frontier_review.py validate --bundle var/frontier-review
python tools/frontier_review.py validate --bundle var/frontier-review --require-complete
```

Optionally pass `--viewer /absolute/local/index.html` for the existing local
video/angle review page. Direct source paths in each CSV also work with an
existing video player. The tool does not open the viewer or inspect pixels.
`--artifact-root` changes the private output root explicitly. Existing output,
including an empty directory, is never overwritten.

## Human work files

- `manifest.json` and `manifest.sha256`: immutable actual content hashes, literal
  aliases, source paths, numerical duration and tool/schema fingerprints. All
  camera, angle, session and filming-source groups are initially unknown. A hash
  is a duplicate group, not a filming session or independent camera.
- `reviewer-a/coverage.csv` and `reviewer-b/coverage.csv`: one row per content and
  type. For 40 contents this is 120 unique slots and 240 independent review
  assignments. Both files start entirely `unreviewed`; they are not predictions.
- Each reviewer's `episodes.jsonl`: empty initially. Reviewers work separately,
  without the other's decisions, organiser peaks or model outputs. Record actual
  human identity/date after reviewing; distinct strings cannot prove independence.
- `adjudication.csv`: 120 pending type-scoped decisions. Keep the two original
  reviews and record disagreements. The original answers must not be rewritten
  to create apparent agreement; preserve versions before any correction.
- `adjudicated-episodes.jsonl`: consolidated human episodes, initially empty.
  Each final episode links `review_a_episode_ids` and `review_b_episode_ids` to
  actual original episodes of the same slot. At least one original link is
  required. Its times must lie in the interval reviewed by both people.

For a completed review row, set `review_status=reviewed`, real `reviewer`/`date`,
`coverage=full_clip` or `partial`, and numerical `reviewed_start_s`/`reviewed_end_s`.
`full_clip` means exactly zero through the manifest duration. Set
`type_exhaustive=true` only if this type has been exhaustively reviewed within
those bounds. `no_observed_positive` requires explicit exhaustive review; an
empty event file never implies it. For `unknown`, use a coded `unknown_reason`
and leave `type_exhaustive=false`. `events_recorded` requires at least one episode.

Episode fields/enums live in [the schema](../tools/frontier_review_schema.py).
Each JSONL record has a unique UUID `episode_id`, the CSV's `slot_id`, outcome
`observed_positive`, `observed_contrast` or `unknown`, and `t_start`/`t_peak`/`t_end`.
Unknown bounds are JSON `null` with a coded reason, never invented. Positive
bounds are known and ordered; actual consumption can have start=peak=end.
The remaining fields are `actor_role`, local anonymous `actor_track_id` and
`object_track_id`, object/contact/departure visibility, `same_actor`/`same_object`,
`unknown_reason` and `contrast_reason`. Relevant positive facts must be clear;
ordered key/coat events require recorded identity continuity. Contrasts need a
coded distinction, not the assumption that the whole video is clean.

Final adjudication can preserve `unknown` or `disputed`. Two completed reviews,
a real adjudicator/date, explicit disagreement code and a nonempty shared
reviewed interval are needed for handoff. A resolved `no_observed_positive`
also requires **both** original reviews to be type-exhaustive. Disputes or missing
consolidated episodes block handoff. No source metadata is upgraded to actual
container PTS: times remain human video-relative seconds unless a separate
source mapping establishes more.

## What validation means

A structurally valid pending draft passes ordinary validation, while
`human_handoff_ready=false` and `--require-complete` fails. This distinction is
intentional. Numeric summaries contain counts only. Validation rereads original
hashes and checks manifest/tool binding, row identity, enum/bounds, distinct
reviewer provenance and final episode links. It does not prove actual viewing,
independence, physical facts or correctness. `evaluation_truth_certified` remains
false: no automatic evaluator export is provided by this scaffold.

Keep all bundles in ignored local `var/`. Do not commit media, frames, filled
reviews, names or observations. These 40 contents remain development data; more
reviews, type-scoped coverage and duplicate aliases do not establish a new camera,
a blinded test, exhaustive labels for other types, or 97% precision.
