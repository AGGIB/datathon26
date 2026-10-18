# Personal-item attribution and precision

An item first detected while the cashier is already absent does not establish
that the cashier placed it on the desk. The personal-item rule now abstains in
that case, including when a later keyframe rediscovers the same item after the
cashier returns. The earliest qualified observation is selected across dense
samples and sparse keyframes. Items observed before departure or after the
return retain their existing behavior; rules for cash left behind are unchanged.

The change uses event intervals and item observations. It contains no video
identifiers, ground-truth lookups, new confidence thresholds or model changes.
Absence of attribution is an abstention, not a negative event label.

## Evaluation scope

The cohort contains 42 files and 40 actual unique contents, including 21 positive
and 19 clean videos. All are development data. Exact duplicate files count once.
Primary annotations identify one event, not every episode in a positive video;
therefore event precision and precision by violation type remain unmeasured.
Camera/session groups are unknown, so these results do not establish performance
on a new camera angle.

Cached perception and scene outputs are reused; the current rules and actual
assembly selection are recomputed. This is not a new detector/scene run or a
complete live service measurement.

## Cached replay, independently checked

| Variant | TP / FP / FN / TN | Video precision | Video recall | Video F1 | Correct primary type | Type and time ±2 s |
| --- | --- | --- | --- | --- | --- | --- |
| Previous CV | 15 / 5 / 6 / 14 | 75.0% | 71.4% | 73.2% | 9/21 | 4/21 |
| Corrected CV | 15 / 3 / 6 / 16 | 83.3% | 71.4% | 76.9% | 9/21 | 4/21 |
| Previous compact V5 | 16 / 5 / 5 / 14 | 76.2% | 76.2% | 76.2% | 11/21 | 5/21 |
| Corrected CV with frozen V5 findings | 16 / 3 / 5 / 16 | 84.2% | 76.2% | 80.0% | 11/21 | 5/21 |

The independent replay uses the actual discovery, runtime augmentation and
assembly paths. Every previously detected positive video and every correct
primary/type-and-time content is retained, with zero new clean-video events.
Full nonpersonal CV candidate inventories are unchanged on all 42 files.

The criteria were declared in collaboration before the first candidate result.
The cached-replay protocol file was written after that initial aggregate;
this is not blind preregistration. Fresh VLM inference has a separate protocol
recorded before outputs exist, with unchanged model settings and retention gates.

## Fresh VLM run: confirmed improvement

Both VLM stages were regenerated on all 42 files / 40 unique contents with the
corrected cached CV baseline. The measured source is `6ed922d413ed7ce64fa678e1180bd6ab812edf58`;
the before-run protocol SHA256 is
`937620f4d7eadae87757dd3db0cae1532072c33f3bf3bb959c367ca391cc1498`.
Qwen3.5-4B MLX uses the unchanged compact verifier, 512 output tokens, frame side
640, confidence threshold 0.95, 180 seconds per video and a four-call cap.

The fresh result is **TP16 / FP3 / FN5 / TN16**, video precision **84.21%** and
recall **76.19%**, video F1 **80.00%**, specificity **84.21%** and balanced
accuracy **80.20%**. Support is 21 positive and 19 clean unique contents, with
five missed positive videos. Correct primary types remain **11/21**, and type-and-time
matches remain **5/21**. Every previous correct video/type/time content is
retained; there are zero new events on known clean videos relative to the
corrected CV baseline. The improvement gate passes; **the 97% target fails**.

Two independent numerical implementations agree. A separate postflight confirms
the actual owner selection and full selected CV candidate fields on all 42 files,
plus stable source, model weights, runtime, input hashes and configuration.
There were 66 actual model calls: 52 proposal calls and 14 verifier calls, with
two exact-duplicate reuses and no timeout, fallback or exhausted budget. Three
proposal windows abstained because action order was unobserved; abstention is
not a semantic negative label.

Private artifacts are `var/runs/cv_precision_v5_fresh_6ed922d.{json,audit.json,protocol.json,independent.json}`
and the separate numerical review. The source snapshot includes inactive
research tools from PR #47; the released production CV/VLM modules are identical.
The probe's additional temporal mode remains disabled in this run. Cached
perception/scene input, development-only scope and incomplete event gold remain
the limitations described above.

## Rejected door diagnostic

A diagnostic withheld cached door measurements when the current ROI was mostly
occluded by a person. Against the previous CV, false-positive videos fell from
five to two, but detected positive videos fell from fifteen to eleven and correct
type-and-time hits fell from four to three. It fails retention and is not shipped.

The remaining door signal measures change from an image reference. Neither
reference validity nor semantic open/closed state is established by that signal
alone. Discarding these alarms merely to obtain a higher percentage is not an
accepted improvement. No thresholds were tuned from the rejected diagnostic.

A subsequent local CPU diagnostic reproduced the final door series exactly for
all eight contents with emitted door candidates. Of 583 samples, 405 had valid
raw measurements and 178 carried a smoothed value through ROI occlusion; maximum
raw-measurement age was 12.116 seconds. This is recomputed measurement provenance,
not proof that an alarm is false or that the door is closed. Historical decoded
pixel/PTS identity is not established; the existing bounded nearest-perception
lookup is recorded separately. No new door policy follows from these counts.

## Next evidence needed

- Review door state transitions and their visibility on the existing recordings,
  including when the reference image already shows an open door. Preserve
  unknowns and record the evidence before changing the rule.
- Use the [human review scaffold](frontier-human-review.md) for exhaustive
  key/coat/consumption episodes and contrasts, with independent review and
  adjudication. An empty scaffold is not completed gold.
- Record real camera/session/source metadata when it is available. Crops,
  repeated reviews and new splits of these development videos do not create
  independent camera evidence.

The 97% video-precision target is separate from the improvement gate. With
sixteen true-positive predictions, one false positive still gives 94.1%; the
current development cohort therefore needs zero false positives while retaining
useful detections. Even that would not demonstrate 97% on unseen camera angles.

## Recall and unseen-camera acceptance

Every subsequent comparison must show precision, recall, F1, TP/FP/FN/TN,
positive/negative support and failed/missing/abstained coverage together. The
existing `ml.eval` already reports end-to-end recall and F1; use that output
rather than reporting only successful model responses. A 97% precision target
does not authorize hiding misses or accepting lower recall. Preserve the actual
previous correct video, primary-type and type-and-time sets, not just their
counts, while testing recall improvements. These are development regression
criteria, not proof of generalization.

A read-only audit of the five fresh false negatives found no selected CV events
in any of them. Three primary types are outside the current seven-type discovery
frontier: `bag_accepted_without_check`, `cash_from_bag_stored_without_recount`
and `no_banknote_detector_used` (one each). The other two are
`previous_cash_left_on_desk` and `recount_after_client_left`; neither produced an
eligible proposal for verification. All five jobs completed, and none supplied a
finding to the owner gate. Lowering its confidence threshold cannot recover those
missing findings. This is a coded pipeline diagnosis, not a visual explanation of
the scenes or proof that enabling another type will work.

The next recall experiment should establish visible action sequences and matched
negative examples for these types before changing proposals or rules. Report
primary-type coverage (currently 11/21) and joint coverage (5/21) separately from
binary video recall. Exhaustive event/type recall, precision and F1 require
complete episode labels and are not established by the primary-only gold.

Judge footage will use different angles. Follow [the existing holdout workflow](angle-holdout.md):
freeze code/model/prompts/thresholds before test outcomes, keep entire source and
camera groups separate, and report precision/recall/F1 and support for each
verified angle as well as pooled totals. Report the worst defined angle result
with its support; undefined metrics and unknown camera IDs remain visible.
Reusing a test set for tuning converts it into development data. Crops, image
warps and resolution stress tests may reveal fragility but do not constitute a
new camera or establish unseen-angle quality.

[Olzhas's PR #50](https://github.com/AGGIB/datathon26/pull/50) supplies 14 public
source candidates, not a labelled holdout. As recorded at commit
`da7eacd528631bfafe4d6adfa424c6584a4a8936`, accepted/downloaded/labelled counts are
zero and camera/source groups are unverified. Suitable continuous scenes,
independent human labels and genuine source metadata are still needed before
precision or recall can be measured on those candidates.
## Merge review follow-up (2026-10-18)

Review fixes retain separate personal-item onsets after a confirmed disappearance
and cashier return. Synthetic regression tests cover both dense observations and
sparse keyframes; dense misses do not clear an item observed by OWLv2. The measured
development results above describe the original recorded snapshot, not a fresh
quality evaluation of this follow-up. No new model run or precision claim is made.
