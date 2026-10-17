"""Type discovery contracts on synthetic evidence only; no weights/model inference."""
from dataclasses import replace
import json
import time
from pathlib import Path

import pytest

from ml.candidates import Candidate
from ml.kb import catalog
from ml.vlm.contracts import Frame
from ml.vlm.discovery import (
    BatchedOwnerFrames, BudgetExhausted, DiscoveryBudget, DiscoveryWindow, _VISUAL_CRITERIA,
    _catalog_types, _decode_frames, _prompt, discover_types,
)

TYPE = "recount_after_client_left"
WINDOW = DiscoveryWindow(0., 4., (0., 1., 2., 3.))


def finding(type_id=TYPE, **changes):
    return {"type_id": type_id, "t_start": 1., "t_peak": 2., "t_end": 3.,
            "confidence": .82, "insufficient_evidence": False,
            "observations": [{"timestamp_s": 1., "text": "Посетитель выходит"},
                             {"timestamp_s": 2., "text": "Кассир пересчитывает банкноты"},
                             {"timestamp_s": 3., "text": "Пересчёт продолжается"}], **changes}


class ScriptedBackend:
    def __init__(self, replies=None):
        self.replies = list(replies or [json.dumps({"findings": [finding()]}, ensure_ascii=False)])
        self.requests = []

    def infer(self, request, timeout_s):
        self.requests.append((request, timeout_s))
        assert timeout_s > 0
        assert all(frame.path.is_file() for frame in request.frames)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class FixtureFrames:
    def __init__(self, missing=()):
        self.calls = []
        self.missing = missing
        self.paths = []

    def extract(self, video_path, timestamps, directory, timeout_s):
        self.calls.append((video_path, tuple(timestamps), timeout_s))
        frames = []
        for index, t in enumerate(timestamps):
            if t in self.missing:
                continue
            path = directory / f"synthetic-{index}.png"
            path.write_bytes(b"SYNTHETIC SCRIPTED EVIDENCE ONLY")
            self.paths.append(path)
            frames.append(Frame(t, path, t))
        return tuple(frames)


def original(type_id="safe_left_open", start=0., end=3.):
    return Candidate(type_id, start, end, start, .55, "DO_NOT_FORWARD_CV_OBSERVATION", (start,))


def run(*, originals=(), backend=None, source=None, windows=(WINDOW,), allowed=(TYPE,),
        budget=None, catalog_types=None):
    return discover_types("PRIVATE_X-TRAIN_LABEL.mov", originals, catalog_types or catalog(), duration_sec=8.,
                          windows=windows, allowed_type_ids=allowed, backend=backend,
                          frame_source=source, budget=budget or DiscoveryBudget(time.monotonic() + 3.))


def test_discovers_real_verified_catalog_candidate_and_preserves_every_original_identity():
    first, low = original(), original("door_left_open", 4., 6.)
    originals = [first, low, first]
    source, backend = FixtureFrames(), ScriptedBackend()
    report = run(originals=originals, backend=backend, source=source)
    assert all(actual is expected for actual, expected in zip(report.retained_candidates, originals))
    assert len(report.retained_candidates) == 3
    [added] = report.added_candidates
    assert added.type_id == TYPE and (added.t_start, added.t_peak, added.t_end) == (1., 2., 3.)
    assert added.confidence == .82 and added.evidence_times == (2., 1., 3.)
    assert "Кассир пересчитывает" in added.observation
    assert report.human_review_required and report.decisions[0].advisory_only
    assert report.status == "completed" and report.calls_used == report.budget_calls_used == 1
    assert not any(path.exists() for path in source.paths)


def test_prompt_contains_only_generic_criteria_and_one_based_image_time_map():
    backend = ScriptedBackend([json.dumps({"findings": []})])
    cv = original()
    report = run(originals=[cv], backend=backend, source=FixtureFrames())
    assert not report.added_candidates and report.status == "completed" and report.calls_used == 1
    assert not report.decisions
    request, _ = backend.requests[0]
    assert request.mode == "checklist" and request.candidate_id is None and request.allowed_rule_ids == (TYPE,)
    assert "PRIVATE_X-TRAIN_LABEL" not in request.prompt and cv.observation not in request.prompt
    assert catalog()[TYPE].rule.quote not in request.prompt and catalog()[TYPE].rule.document not in request.prompt
    payload = json.loads(request.prompt[request.prompt.index('{"types"'):])
    assert payload["images_in_supplied_order"] == [
        {"image_index": i, "timestamp_s": frame.timestamp_s} for i, frame in enumerate(request.frames, 1)]
    assert "money-counting machine is not evidence" in request.prompt
    schema = request.response_schema
    assert schema["additionalProperties"] is False
    finding_schema = schema["properties"]["findings"]["items"]
    assert finding_schema["properties"]["type_id"]["enum"] == [TYPE]
    assert finding_schema["properties"]["t_peak"]["enum"] == list(WINDOW.timestamps)
    assert "complete unobscured handling cycle with no gaps" in request.prompt


def test_seventeen_type_frontier_has_compact_prompt_and_explicit_t04_abstention():
    types, missing = _catalog_types(catalog(), tuple(_VISUAL_CRITERIA))
    assert len(types) == 16
    assert [(item.type_id, item.reason) for item in missing] == [("computer_left_unlocked", "no_verified_rule")]
    window = DiscoveryWindow(0., 12., tuple(float(i) for i in range(12)))
    assert len(_prompt(types, window)) < 6000
    backend = ScriptedBackend()
    report = run(backend=backend, source=FixtureFrames(), allowed=("computer_left_unlocked",))
    assert report.status == "no_verified_types" and not backend.requests
    assert report.unavailable_types == missing


def test_disabled_unknown_and_outside_frontier_types_cannot_enable_discovery():
    entries = dict(catalog())
    entries[TYPE] = replace(entries[TYPE], enabled=False)
    backend, source = ScriptedBackend(), FixtureFrames()
    report = run(backend=backend, source=source, catalog_types=entries,
                 allowed=(TYPE, "unknown", "personal_camera_phone_or_cash_in_cash_unit"))
    assert report.status == "no_verified_types" and not backend.requests and not source.calls
    assert {item.reason for item in report.unavailable_types} == {"disabled_type", "unknown_type", "outside_type_frontier"}


def test_one_batch_union_is_decoded_once_and_reused_in_two_window_calls(monkeypatch):
    import cv2
    import numpy as np
    import ml.video

    decoded = []
    def frames_at(path, timestamps):
        decoded.append((path, timestamps))
        return {t: np.zeros((4, 4, 3), np.uint8) for t in timestamps}
    monkeypatch.setattr(ml.video, "frames_at", frames_at)
    writes = []
    def save(path, image):
        Path(path).write_bytes(b"SYNTHETIC IMAGE")
        writes.append(path)
        return True
    monkeypatch.setattr(cv2, "imwrite", save)
    class OwnerFixture:
        def extract(self, path, timestamps, directory, timeout_s):
            return _decode_frames(path, timestamps, directory, time.monotonic() + timeout_s)
    second = DiscoveryWindow(2., 6., (2., 3., 4., 5.))
    backend = ScriptedBackend(['{"findings":[]}', '{"findings":[]}'])
    report = run(backend=backend, source=OwnerFixture(), windows=(WINDOW, second))
    assert decoded == [("PRIVATE_X-TRAIN_LABEL.mov", [0., 1., 2., 3., 4., 5.])]
    assert len(writes) == 6 and len(backend.requests) == 2 and report.calls_used == 2
    assert backend.requests[0][0].frames[2].path == backend.requests[1][0].frames[0].path


@pytest.mark.parametrize("raw", [
    "not JSON", '```json\n{"findings":[]}\n```', '{"findings":[],"findings":[]}',
    '{"findings":[],"citation":"invented"}', '{"findings":null}',
    json.dumps({"findings": [finding(type_id="unknown")]}),
    json.dumps({"findings": [finding(type_id="computer_left_unlocked")]}),
    json.dumps({"findings": [finding(confidence=True)]}),
    json.dumps({"findings": [finding(confidence=float("nan"))]}),
    json.dumps({"findings": [finding(t_peak=2.5)]}),
    json.dumps({"findings": [finding(t_end=7.)]}),
    json.dumps({"findings": [finding(citation="invented")]}),
    json.dumps({"findings": [finding(observations=[{"timestamp_s": 2., "text": "123"}])]}),
    json.dumps({"findings": [finding(observations=[{"timestamp_s": t, "text": "123"}
                                                   for t in [1., 2., 3.]])]}),
    json.dumps({"findings": [finding(observations=[{"timestamp_s": t, "text": "Основание: п. 20"}
                                                   for t in [1., 2., 3.]])]}),
])
def test_bad_output_abstains_and_retains_cv_unchanged(raw):
    cv = original()
    report = run(originals=[cv], backend=ScriptedBackend([raw]), source=FixtureFrames())
    assert report.retained_candidates[0] is cv and not report.added_candidates
    assert report.decisions[0].reason == "invalid_response" and report.calls_used == 1
    assert report.decisions[0].validation_code is not None


def test_payload_validation_is_atomic_no_additions_from_valid_prefix_of_bad_window():
    report = run(backend=ScriptedBackend([json.dumps({"findings": [finding(), finding(type_id="unknown")]})]),
                 source=FixtureFrames())
    assert not report.added_candidates and report.decisions[0].reason == "invalid_response"


@pytest.mark.parametrize("changes", [{"insufficient_evidence": True}, {"confidence": .69}])
def test_uncertain_or_low_confidence_finding_never_adds(changes):
    report = run(backend=ScriptedBackend([json.dumps({"findings": [finding(**changes)]})]), source=FixtureFrames())
    assert not report.added_candidates and report.decisions[0].reason == "insufficient_evidence"


def test_overlapping_existing_or_repeated_discoveries_add_no_duplicate_episode():
    cv = original(TYPE)
    report = run(originals=[cv], backend=ScriptedBackend(), source=FixtureFrames())
    assert not report.added_candidates and report.decisions[0].reason == "overlaps_existing"
    report = run(backend=ScriptedBackend([json.dumps({"findings": [finding(), finding()]})]), source=FixtureFrames())
    assert len(report.added_candidates) == 1


def test_missing_requested_frame_abstains_without_partial_evidence_inference():
    backend = ScriptedBackend()
    report = run(backend=backend, source=FixtureFrames(missing=(2.,)))
    assert not backend.requests and not report.added_candidates
    assert report.decisions[0].reason == "insufficient_frames"


def test_model_unavailable_and_expired_budget_do_not_decode_or_infer():
    cv, source = original(), FixtureFrames()
    assert run(originals=[cv], source=source).status == "model_unavailable"
    backend = ScriptedBackend()
    report = run(originals=[cv], backend=backend, source=source,
                 budget=DiscoveryBudget(time.monotonic() - 1.))
    assert report.status == "video_timeout" and report.retained_candidates[0] is cv
    assert not source.calls and not backend.requests


def test_shared_call_budget_spans_windows_and_separate_adapter_invocations():
    backend, source = ScriptedBackend(), FixtureFrames()
    budget = DiscoveryBudget(time.monotonic() + 3., max_calls=1)
    first = run(backend=backend, source=source, budget=budget, windows=(WINDOW, WINDOW))
    second = run(backend=backend, source=source, budget=budget)
    assert len(first.added_candidates) == 1 and first.status == second.status == "budget_exhausted"
    assert first.calls_used == 1 and second.calls_used == 0 and budget.calls_used == 1
    assert len(source.calls) == len(backend.requests) == 1


class BudgetedTwoStage:
    """Two physical calls, one final reply; no proposal may escape on stage failure."""
    def __init__(self, *, normal=False, fail_verify=False, delay=0):
        self.calls = []
        self.normal, self.fail_verify, self.delay = normal, fail_verify, delay

    def infer(self, *args):
        raise AssertionError("budget-aware backend must receive the owner budget")

    def infer_with_budget(self, request, budget):
        for stage in range(1 if self.normal else 2):
            if not budget.claim():
                raise BudgetExhausted()
            self.calls.append((request, stage, budget.remaining()))
            if stage == 0:
                time.sleep(self.delay)
            if stage == 1 and self.fail_verify:
                raise RuntimeError("PRIVATE VERIFIER ERROR")
        if self.normal:
            return '{"findings":[]}'
        times = [frame.timestamp_s for frame in request.frames][:3]
        item = finding(t_start=times[0], t_peak=times[1], t_end=times[2],
                       observations=[{"timestamp_s": t, "text": "Синтетическое действие"} for t in times])
        return json.dumps({"findings": [item]})


def test_two_stage_budget_counts_each_physical_call_without_precharging_window():
    second_window = DiscoveryWindow(4., 7., (4., 5., 6., 7.))
    backend, source = BudgetedTwoStage(), FixtureFrames()
    budget = DiscoveryBudget(time.monotonic() + 3, max_calls=4)
    report = run(backend=backend, source=source, windows=(WINDOW, second_window), budget=budget)
    assert report.status == "completed" and len(report.added_candidates) == 2
    assert report.calls_used == report.budget_calls_used == len(backend.calls) == 4
    assert len(source.calls) == 1
    assert backend.calls[0][0].frames is backend.calls[1][0].frames
    assert backend.calls[2][0].frames is backend.calls[3][0].frames


def test_two_stage_budget_shortage_drops_unverified_proposal_and_preserves_previous_window():
    cv = original()
    backend = BudgetedTwoStage()
    second_window = DiscoveryWindow(4., 7., (4., 5., 6., 7.))
    report = run(originals=[cv], backend=backend, source=FixtureFrames(),
                 windows=(WINDOW, second_window), budget=DiscoveryBudget(time.monotonic() + 3, max_calls=3))
    assert report.status == "budget_exhausted" and report.calls_used == len(backend.calls) == 3
    assert len(report.added_candidates) == 1 and report.retained_candidates[0] is cv
    assert report.decisions[-1].reason == "budget_exhausted"
    one = run(backend=BudgetedTwoStage(), source=FixtureFrames(),
              budget=DiscoveryBudget(time.monotonic() + 3, max_calls=1))
    assert one.status == "budget_exhausted" and one.calls_used == 1 and not one.added_candidates


def test_budget_aware_normal_skips_verify_and_failed_verification_still_counts():
    normal = BudgetedTwoStage(normal=True)
    report = run(backend=normal, source=FixtureFrames())
    assert report.status == "completed" and report.calls_used == len(normal.calls) == 1
    assert not report.added_candidates
    failure = BudgetedTwoStage(fail_verify=True)
    cv = original()
    report = run(originals=[cv], backend=failure, source=FixtureFrames())
    assert report.status == "backend_error" and report.calls_used == len(failure.calls) == 2
    assert not report.added_candidates and report.retained_candidates[0] is cv
    assert "PRIVATE VERIFIER ERROR" not in repr(report)


def test_budget_aware_counts_delta_when_budget_already_used_and_forbids_late_verify():
    budget = DiscoveryBudget(time.monotonic() + 3, max_calls=3, calls_used=1)
    report = run(backend=BudgetedTwoStage(), source=FixtureFrames(), budget=budget)
    assert report.calls_used == 2 and report.budget_calls_used == 3
    slow = BudgetedTwoStage(delay=.1)
    budget = DiscoveryBudget(time.monotonic() + .03, max_calls=2)
    started = time.monotonic()
    report = run(backend=slow, source=FixtureFrames(), budget=budget)
    assert report.status == "video_timeout" and not report.added_candidates
    assert report.calls_used == 1 and time.monotonic() - started < .2
    time.sleep(.11)
    assert len(slow.calls) == budget.calls_used == 1


def test_shared_deadline_includes_decode_and_does_not_reset_per_window():
    backend = ScriptedBackend()
    class SlowFrames:
        def extract(self, *args):
            time.sleep(.08)
            return ()
    started = time.monotonic()
    report = run(backend=backend, source=SlowFrames(), budget=DiscoveryBudget(started + .02))
    assert report.status == "video_timeout" and not backend.requests
    assert time.monotonic() - started < .3


def test_backend_ignoring_timeout_cannot_block_discovery_caller():
    class StalledBackend:
        def infer(self, *args):
            time.sleep(.1)
            return '{"findings":[]}'
    started = time.monotonic()
    report = run(backend=StalledBackend(), source=FixtureFrames(), budget=DiscoveryBudget(started + .02))
    assert report.status == "video_timeout" and not report.added_candidates and report.calls_used == 1
    assert time.monotonic() - started < .3


def test_backend_unavailable_returns_originals_instead_of_raising():
    cv = original()
    report = run(originals=[cv], backend=ScriptedBackend([RuntimeError("PRIVATE ERROR")]), source=FixtureFrames())
    assert report.status == "backend_error" and report.retained_candidates[0] is cv
    assert not report.added_candidates and "PRIVATE ERROR" not in repr(report)


@pytest.mark.parametrize("factory", [
    lambda: DiscoveryWindow(0, 4, (1., 1.)),
    lambda: DiscoveryWindow(0, 4, (2., 1.)),
    lambda: DiscoveryWindow(0, 4, (True,)),
    lambda: DiscoveryWindow(0, 4, (5.,)),
    lambda: DiscoveryBudget(float("nan")),
    lambda: DiscoveryBudget(time.monotonic() + 1, max_calls=True),
    lambda: DiscoveryBudget(time.monotonic() + 1, max_calls=7),
])
def test_invalid_plans_fail_before_processing(factory):
    with pytest.raises(ValueError):
        factory()


def test_batch_source_process_start_failure_closes_all_resources(tmp_path, monkeypatch):
    import multiprocessing
    from types import SimpleNamespace

    context = multiprocessing.get_context("spawn")
    endpoints, processes = [], []
    def pipe():
        pair = context.Pipe()
        endpoints.extend(pair)
        return pair
    def process(**kwargs):
        instance = context.Process(**kwargs)
        processes.append(instance)
        def fail():
            raise OSError("synthetic start failure")
        monkeypatch.setattr(instance, "start", fail)
        return instance
    monkeypatch.setattr(multiprocessing, "get_context", lambda *args: SimpleNamespace(Pipe=pipe, Process=process))
    with pytest.raises(OSError):
        BatchedOwnerFrames().extract("synthetic.mov", (0.,), tmp_path, 1.)
    assert len(processes) == 1 and processes[0]._closed
    assert all(endpoint.closed for endpoint in endpoints)


def test_default_owner_process_decodes_real_synthetic_video_and_assembly_uses_catalog_rule(tmp_path):
    import cv2
    import numpy as np
    from ml.assemble import build_violation

    path = tmp_path / "synthetic.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 4., (32, 32))
    assert writer.isOpened()
    for index in range(16):
        writer.write(np.full((32, 32, 3), index * 10, dtype=np.uint8))
    writer.release()
    backend = ScriptedBackend()
    report = discover_types(path, (), catalog(), duration_sec=4., windows=(WINDOW,),
                            allowed_type_ids=(TYPE,), backend=backend,
                            budget=DiscoveryBudget(time.monotonic() + 10.))
    assert report.status == "completed" and len(report.added_candidates) == 1
    [candidate] = report.added_candidates
    violation = build_violation(candidate, catalog()[TYPE], [], 4.)
    assert violation["rule"] == catalog()[TYPE].rule.as_ref()
    assert violation["type_id"] == TYPE and violation["timecode_sec"] == 2.
    assert not any(frame.path.exists() for frame in backend.requests[0][0].frames)


@pytest.mark.parametrize("absence_type", ["no_visible_client_id_check", "cash_from_bag_stored_without_recount",
                                          "bag_accepted_without_check", "no_banknote_detector_used"])
def test_sparse_default_policy_cannot_add_absence_only_type_even_with_verified_norm(absence_type):
    from ml.vlm.discovery_policy import TYPE_IDS

    cv = original()
    backend = ScriptedBackend([json.dumps({"findings": [finding(type_id=absence_type)]})])
    report = run(originals=[cv], backend=backend, source=FixtureFrames(), allowed=TYPE_IDS)
    assert report.retained_candidates[0] is cv and not report.added_candidates
    assert report.decisions[0].reason == "invalid_response"
    assert absence_type not in backend.requests[0][0].allowed_rule_ids
    assert catalog()[absence_type].status == "verified"


def test_observations_precede_classification_in_prompt_and_constrained_schema():
    from ml.vlm.discovery_policy import TYPE_IDS

    backend = ScriptedBackend(['{"findings":[]}'])
    report = run(backend=backend, source=FixtureFrames(), allowed=TYPE_IDS)
    request = backend.requests[0][0]
    assert report.status == "completed"
    finding_schema = request.response_schema["properties"]["findings"]["items"]
    assert list(finding_schema["properties"])[0] == finding_schema["required"][0] == "observations"
    assert list(finding_schema["properties"])[1] == "type_id"
    example = request.prompt[request.prompt.index("Output JSON only"):request.prompt.index("Choose t_peak")]
    assert example.index('"observations"') < example.index('"type_id"')
    payload = json.loads(request.prompt[request.prompt.index('{"types"'):])
    criteria = {entry["type_id"]: entry["visual_criterion"] for entry in payload["types"]}
    assert len(criteria) == 7 and set(criteria) == set(TYPE_IDS)
    coat = criteria["personal_items_on_desk"]
    assert "removes their own worn coat" in coat and "places that coat" in coat
    assert "client/courier bag" in coat and "does not qualify" in coat
    assert "visibly consumes" in criteria["food_or_drink_at_workplace"]
    assert "most conclusive observed consequence, not the earliest prerequisite" in request.prompt


def test_decisive_peak_after_prerequisites_still_requires_source_anchor_and_observation():
    later = finding(t_peak=3.)
    report = run(backend=ScriptedBackend([json.dumps({"findings": [later]})]), source=FixtureFrames())
    [added] = report.added_candidates
    assert added.t_start == 1. and added.t_peak == added.t_end == 3.
    assert added.evidence_times == (3., 1., 2.)
    missing_consequence = finding(t_peak=3., observations=finding()["observations"][:2])
    report = run(backend=ScriptedBackend([json.dumps({"findings": [missing_consequence]})]), source=FixtureFrames())
    assert not report.added_candidates and report.decisions[0].reason == "invalid_response"
