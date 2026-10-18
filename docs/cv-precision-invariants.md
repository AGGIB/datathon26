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

| Variant | TP / FP / FN / TN | Video precision | Correct primary type | Type and time ±2 s |
| --- | --- | --- | --- | --- |
| Previous CV | 15 / 5 / 6 / 14 | 75.0% | 9/21 | 4/21 |
| Corrected CV | 15 / 3 / 6 / 16 | 83.3% | 9/21 | 4/21 |
| Previous compact V5 | 16 / 5 / 5 / 14 | 76.2% | 11/21 | 5/21 |
| Corrected CV with frozen V5 findings | 16 / 3 / 5 / 16 | 84.2% | 11/21 | 5/21 |

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
recall **76.19%**. Correct primary types remain **11/21**, and type-and-time
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
