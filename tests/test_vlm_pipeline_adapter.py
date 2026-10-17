"""Synthetic CV/KB contracts; no real scenes, weights or model-quality claims."""
from dataclasses import dataclass
import json
from pathlib import Path

import pytest


@dataclass(frozen=True)
class FixtureCandidate:
    type_id: str = "fixture_action"
    t_start: float = 0.0
    t_end: float = 1.0
    t_peak: float = 0.5
    confidence: float = 0.8
    observation: str = "DO_NOT_FORWARD_FIXTURE_OBSERVATION"
    evidence_times: tuple = (0.5,)
    highlight_boxes: tuple = ((1, 2, 3, 4),)


def candidate(type_id="fixture_action", **changes):
    return FixtureCandidate(type_id=type_id, **changes)


@dataclass(frozen=True)
class FixtureRule:
    document: str = "FIXTURE document"
    basis: str = "FIXTURE §1"


@dataclass(frozen=True)
class FixtureCatalogType:
    id: str = "fixture_action"
    name: str = "Fixture"
    summary: str = "Artificial visible action only"
    status: str = "verified"
    rule: FixtureRule | None = FixtureRule()


def catalog_entry(type_id="fixture_action", **changes):
    return FixtureCatalogType(id=type_id, **changes)


def test_missing_model_and_missing_norm_retain_exact_original_cv_objects():
    from ml.vlm.pipeline_adapter import verify_pipeline_candidates

    first = candidate()
    pending = candidate("computer_left_unlocked")
    unknown = candidate("unknown_fixture_type")
    inputs = [first, pending, unknown, first]
    catalog = {"fixture_action": catalog_entry(),
               "computer_left_unlocked": catalog_entry("computer_left_unlocked", status="draft", rule=None)}
    report = verify_pipeline_candidates("absent-fixture.mp4", inputs, catalog, budget_sec=2.0, timebase_verified=True)

    assert len(report.retained_candidates) == len(inputs)
    assert all(actual is original for actual, original in zip(report.retained_candidates, inputs))
    assert [r.reason for r in report.verification.results] == [
        "model_unavailable", "no_verified_rule", "no_verified_rule", "model_unavailable"]
    assert [r.candidate["candidate_id"] for r in report.verification.results] == [
        "cv-1", "cv-2", "cv-3", "cv-4"]
    assert report.verification.results[0].candidate == {
        "candidate_id": "cv-1", "violation_type": "fixture_action",
        "start_s": 0.0, "end_s": 1.0, "rule_id": "fixture_action"}
    assert report.verification.calls_used == 0
    assert report.verification.human_review_required is True
    assert report.verification.checklist_status == "disabled"


@pytest.mark.parametrize("document", ["", "   ", None, 123])
def test_catalog_with_no_document_cannot_enable_inference(document):
    from ml.vlm.pipeline_adapter import verify_pipeline_candidates

    class NeverInfer:
        def infer(self, request, timeout_s):
            pytest.fail("A missing document is not a verified normative rule")

    original = candidate()
    entry = catalog_entry(rule=FixtureRule(document=document))
    report = verify_pipeline_candidates("fixture.mp4", [original], {entry.id: entry}, budget_sec=2.0, backend=NeverInfer())

    assert report.retained_candidates == (original,)
    assert report.verification.results[0].reason == "no_verified_rule"
    assert report.verification.calls_used == 0


class FixtureFrames:
    def probe(self, video_path, timeout_s):
        from ml.vlm import FrameTimestamp, VideoInfo
        return VideoInfo(1.0, tuple(FrameTimestamp(i, t, t) for i, t in enumerate((0.0, 0.5, 1.0))))

    def extract(self, video_path, selected, directory, timeout_s):
        from ml.vlm import Frame
        return tuple(Frame(f.timestamp_s, Path(directory) / f"fixture-{f.index}.png", f.pts_s)
                     for f in selected)


class FixtureBackend:
    def __init__(self, modes):
        self.modes = modes
        self.requests = []

    def infer(self, request, timeout_s):
        self.requests.append(request)
        mode = self.modes[len(self.requests) - 1]
        if mode == "bad_json":
            return "invalid fixture response"
        if mode == "timeout":
            raise TimeoutError("PRIVATE_FIXTURE_EXCEPTION")
        if mode == "error":
            raise RuntimeError("PRIVATE_FIXTURE_EXCEPTION")
        return json.dumps({"candidate_id": request.candidate_id, "rule_id": request.allowed_rule_ids[0],
                           "violation": mode != "reject", "confidence": 0.1 if mode == "low" else 0.9,
                           "insufficient_evidence": False,
                           "observations": [{"timestamp_s": 0.5, "text": "Artificial visible action"}]})


def test_public_adapter_never_removes_valid_rejection_and_preserves_cv_metadata():
    from ml.vlm import verify_pipeline_candidates

    originals = [candidate(t_peak=0.5) for _ in range(6)]
    backend = FixtureBackend(["reject", "confirm", "low", "bad_json", "timeout", "error"])
    report = verify_pipeline_candidates("fixture.mp4", originals, {"fixture_action": catalog_entry()}, budget_sec=2.0,
                                        backend=backend, frame_source=FixtureFrames(), timebase_verified=True,
                                        allow_rejection=True)

    assert [r.status for r in report.verification.results] == [
        "rejected", "confirmed", "uncertain", "uncertain", "uncertain", "uncertain"]
    assert len(report.retained_candidates) == 6
    assert all(actual is original for actual, original in zip(report.retained_candidates, originals))
    assert report.verification.calls_used == 6
    assert report.verification.results[1].citation == "FIXTURE §1"
    assert all(r.citation is None for r in report.verification.results if r.status != "confirmed")
    assert "PRIVATE_FIXTURE_EXCEPTION" not in repr(report.verification)
    assert all("DO_NOT_FORWARD_FIXTURE_OBSERVATION" not in req.prompt for req in backend.requests)
    assert all("FIXTURE §1" not in req.prompt for req in backend.requests)
    assert all(original.confidence == 0.8 and original.evidence_times == (0.5,)
               and original.highlight_boxes == ((1, 2, 3, 4),) for original in originals)


def test_adapter_requires_explicit_timebase_verification_before_inference():
    from ml.vlm import verify_pipeline_candidates

    original = candidate()
    backend = FixtureBackend(["reject"])
    report = verify_pipeline_candidates("fixture.mp4", [original], {"fixture_action": catalog_entry()}, budget_sec=2.0,
                                        backend=backend, frame_source=FixtureFrames())

    assert backend.requests == []
    assert report.verification.calls_used == 0
    assert report.verification.results[0].status == "uncertain"
    assert report.verification.results[0].reason == "unverified_timebase"
    assert report.retained_candidates[0] is original


def test_owner_ruletext_without_legacy_basis_uses_only_verified_source_fields():
    from ml.vlm import verify_pipeline_candidates

    # RuleText fields from pipeline owner 215b51ad; not a generated citation.
    @dataclass(frozen=True)
    class OwnerRule:
        document: str = "FIXTURE document"
        number_date: str | None = "FIXTURE date"
        clause: str | None = "FIXTURE clause"
        quote: str | None = None
        url: str | None = None

    original = candidate()
    entry = catalog_entry(rule=OwnerRule())
    backend = FixtureBackend(["confirm"])
    report = verify_pipeline_candidates(
        "fixture.mp4", [original], {entry.id: entry}, budget_sec=2.0,
        backend=backend, frame_source=FixtureFrames(), timebase_verified=True)

    assert report.verification.results[0].status == "confirmed"
    assert report.verification.results[0].citation == "FIXTURE document, FIXTURE date, FIXTURE clause"
    assert report.retained_candidates[0] is original
    assert "FIXTURE document" not in backend.requests[0].prompt


def test_shadow_default_retains_even_valid_advisory_rejection_until_quality_is_validated():
    from ml.vlm import verify_pipeline_candidates

    original = candidate()
    report = verify_pipeline_candidates("fixture.mp4", [original], {"fixture_action": catalog_entry()}, budget_sec=2.0,
                                        backend=FixtureBackend(["reject"]), frame_source=FixtureFrames(),
                                        timebase_verified=True)

    assert report.verification.results[0].status == "rejected"
    assert len(report.verification.retained_candidates) == 1
    assert report.retained_candidates == (original,)


@pytest.mark.parametrize("type_id", [["fixture_action"], {"id": "fixture_action"}])
def test_malformed_cv_type_id_is_retained_without_inference(type_id):
    from ml.vlm import verify_pipeline_candidates

    original = candidate(type_id)
    backend = FixtureBackend([])
    report = verify_pipeline_candidates("fixture.mp4", [original], {"fixture_action": catalog_entry()}, budget_sec=2.0,
                                        backend=backend, frame_source=FixtureFrames(), timebase_verified=True,
                                        allow_rejection=True)

    assert backend.requests == []
    assert report.verification.results[0].status == "uncertain"
    assert report.retained_candidates[0] is original


@pytest.mark.parametrize("budget", [0, 2, 6])
def test_budget_is_shared_across_whole_video_and_pending_norm_consumes_no_call(budget):
    from ml.vlm import verify_pipeline_candidates

    originals = [candidate("computer_left_unlocked")] + [candidate() for _ in range(8)]
    backend = FixtureBackend(["confirm"] * budget)
    report = verify_pipeline_candidates("fixture.mp4", originals, {"fixture_action": catalog_entry()}, budget_sec=2.0,
                                        backend=backend, frame_source=FixtureFrames(), timebase_verified=True,
                                        max_calls=budget, allow_rejection=True)

    assert report.verification.calls_used == len(backend.requests) == budget
    assert len(report.verification.results) == len(originals)
    assert report.verification.results[0].reason == "no_verified_rule"
    assert all(r.reason == "budget_exhausted" for r in report.verification.results[budget + 1:])
    assert all(actual is original for actual, original in zip(report.retained_candidates, originals))
    assert report.verification.checklist_status == "disabled"


@pytest.mark.parametrize("entry", [catalog_entry(status="draft"), catalog_entry(status="ready"),
                                   catalog_entry(status=True), catalog_entry(type_id="different_fixture_id"),
                                   catalog_entry(rule=FixtureRule(basis=""))])
def test_catalog_status_and_literal_key_must_agree_before_rule_can_be_used(entry):
    from ml.vlm import verify_pipeline_candidates

    original = candidate()
    backend = FixtureBackend([])
    report = verify_pipeline_candidates("fixture.mp4", [original], {"fixture_action": entry}, budget_sec=2.0,
                                        backend=backend, frame_source=FixtureFrames(), timebase_verified=True)
    assert backend.requests == []
    assert report.verification.results[0].reason == "no_verified_rule"
    assert report.retained_candidates[0] is original


@pytest.mark.parametrize("type_id", [" fixture_action", "fixture_action ", "Fixture_action"])
def test_cv_identifiers_are_not_trimmed_or_normalized(type_id):
    from ml.vlm import verify_pipeline_candidates

    original = candidate(type_id)
    backend = FixtureBackend([])
    report = verify_pipeline_candidates("fixture.mp4", [original], {"fixture_action": catalog_entry()}, budget_sec=2.0,
                                        backend=backend, frame_source=FixtureFrames(), timebase_verified=True)
    assert backend.requests == []
    assert report.verification.results[0].candidate["violation_type"] == type_id
    assert report.retained_candidates[0] is original


@pytest.mark.parametrize("changes", [{"t_start": -1}, {"t_start": float("nan")},
                                    {"t_start": True}, {"t_end": 2.0}])
def test_invalid_intervals_remain_uncertain_even_if_rejection_is_enabled(changes):
    from ml.vlm import verify_pipeline_candidates

    original = candidate(**changes)
    backend = FixtureBackend([])
    report = verify_pipeline_candidates("fixture.mp4", [original], {"fixture_action": catalog_entry()}, budget_sec=2.0,
                                        backend=backend, frame_source=FixtureFrames(), timebase_verified=True,
                                        allow_rejection=True)
    assert backend.requests == []
    assert report.verification.results[0].reason == "invalid_candidate"
    assert report.retained_candidates[0] is original


def test_production_missing_rule_attribute_retains_exact_originals_with_safe_warning(caplog):
    from types import SimpleNamespace
    from ml import vlm

    entry = SimpleNamespace(id="fixture_action", status="verified", summary="PRIVATE_SYNTHETIC_SCENE")
    original = candidate()
    originals = [original, candidate(), original]
    backend = FixtureBackend([])
    try:
        assert vlm.configure(backend=backend, catalog={"fixture_action": entry},
                             frame_source=FixtureFrames(), timebase_verified=True)
        retained = vlm.verify(originals, "synthetic.mp4", 1)
        assert len(retained) == len(originals)
        assert all(actual is expected for actual, expected in zip(retained, originals))
        assert backend.requests == []
        records = [record for record in caplog.records if record.name == "ml.vlm.runtime"]
        assert len(records) == 1
        assert records[0].levelname == "WARNING"
        assert records[0].getMessage() == "VLM verification unavailable: AttributeError"
        assert records[0].exc_info is None
        assert "PRIVATE_SYNTHETIC_SCENE" not in caplog.text
    finally:
        vlm.configure(catalog={})


@pytest.mark.parametrize("phase", ["construct", "adapter", "cleanup"])
def test_production_boundary_exceptions_preserve_originals_with_class_only_warning(monkeypatch, caplog, phase):
    from ml import vlm
    from ml.vlm import owner_frames, runtime

    original = candidate()
    originals = [original, candidate(), original]
    events = []
    class FailedFrames:
        def __init__(self):
            events.append("construct")
            if phase == "construct":
                raise RuntimeError("PRIVATE_SYNTHETIC_SCENE")
        def close(self):
            events.append("close")
            if phase == "cleanup":
                raise RuntimeError("PRIVATE_SYNTHETIC_SCENE")
    def adapter(*args, **kwargs):
        from types import SimpleNamespace
        events.append("adapter")
        if phase == "adapter":
            raise RuntimeError("PRIVATE_SYNTHETIC_SCENE")
        return SimpleNamespace(retained_candidates=originals[:-1])
    monkeypatch.setattr(owner_frames, "OwnerFrames", FailedFrames)
    monkeypatch.setattr(runtime, "verify_pipeline_candidates", adapter)
    try:
        assert vlm.configure(backend=FixtureBackend([]), catalog={})
        retained = vlm.verify(originals, "synthetic.mp4", 1)
        assert len(retained) == len(originals)
        assert all(actual is expected for actual, expected in zip(retained, originals))
        assert all(actual.confidence == 0.8 for actual in retained)
        assert events == (["construct"] if phase == "construct" else ["construct", "adapter", "close"])
        records = [record for record in caplog.records if record.name == "ml.vlm.runtime"]
        assert len(records) == 1
        assert records[0].levelname == "WARNING"
        assert records[0].getMessage() == "VLM verification unavailable: RuntimeError"
        assert records[0].exc_info is None
        assert "PRIVATE_SYNTHETIC_SCENE" not in caplog.text
    finally:
        vlm.configure(catalog={})


@pytest.mark.parametrize("failure", [ImportError, ValueError, OSError, RuntimeError])
@pytest.mark.parametrize("phase", ["load", "snapshot"])
def test_catalogue_startup_failure_warns_without_claiming_ready_or_retrying(monkeypatch, caplog, phase, failure):
    import sys
    from types import SimpleNamespace
    from ml import vlm
    from ml.vlm import runtime

    events = []
    class Backend(FixtureBackend):
        def validate(self, timeout_s):
            events.append("validate")
    def fail():
        events.append(phase)
        raise failure("PRIVATE_SYNTHETIC_SCENE")
    class BrokenCatalog:
        def keys(self):
            return fail()
    monkeypatch.setitem(sys.modules, "ml.kb", SimpleNamespace(catalog=fail))
    backend = Backend([])
    original = candidate()
    try:
        catalog = None if phase == "load" else BrokenCatalog()
        assert vlm.configure(backend=backend, catalog=catalog) is False
        assert runtime._configuration is None
        for _ in range(2):
            assert vlm.verify([original], "synthetic.mp4", 1)[0] is original
        assert events == [phase]
        assert backend.requests == []
        records = [record for record in caplog.records if record.name == "ml.vlm.runtime"]
        assert len(records) == 1
        assert records[0].levelname == "WARNING"
        assert records[0].getMessage() == f"VLM catalogue unavailable: {failure.__name__}"
        assert records[0].exc_info is None
        assert "PRIVATE_SYNTHETIC_SCENE" not in caplog.text
    finally:
        vlm.configure(catalog={})


def test_production_startup_seam_preserves_original_frozen_candidates():
    from ml import vlm

    class PreparedFixture(FixtureBackend):
        validations = 0

        def validate(self, timeout_s):
            self.validations += 1

    backend = PreparedFixture(["reject", "low"])
    originals = [candidate(), candidate()]
    assert vlm.configure(backend=backend, catalog={"fixture_action": catalog_entry()},
                         frame_source=FixtureFrames(), timebase_verified=True)
    try:
        retained = vlm.verify(originals, "synthetic.mp4", 1.0)
        assert isinstance(retained, list)
        assert all(actual is original for actual, original in zip(retained, originals))
        assert len(retained) == 2
        assert all(c.confidence == 0.8 for c in retained)
        assert len(backend.requests) == 2
        assert backend.validations == 1
    finally:
        vlm.configure(catalog={})


def test_production_default_frames_use_owner_cv_timeline_without_legacy_pts_gate(monkeypatch):
    from ml import vlm
    from ml.vlm.owner_frames import OwnerFrames

    calls = []
    def probe(self, *args):
        calls.append("probe")
        return FixtureFrames().probe(*args)
    def extract(self, *args):
        calls.append("extract")
        return FixtureFrames().extract(*args)
    def close(self):
        calls.append("close")
    def forbidden(*args):
        pytest.fail("production CV frame source must never run full ffprobe")
    monkeypatch.setattr(OwnerFrames, "probe", probe)
    monkeypatch.setattr(OwnerFrames, "extract", extract)
    monkeypatch.setattr(OwnerFrames, "close", close)
    monkeypatch.setattr(vlm.FFmpegFrames, "probe", forbidden)
    backend = FixtureBackend(["reject", "confirm"])
    originals = [candidate(), candidate()]
    try:
        assert vlm.configure(backend=backend, catalog={"fixture_action": catalog_entry()})
        retained = vlm.verify(originals, "synthetic.mp4", 1.0)
        assert len(backend.requests) == 2
        assert calls == ["probe", "extract", "extract", "close"]
        assert all(a is b for a, b in zip(retained, originals))
    finally:
        vlm.configure(catalog={})


def test_production_does_not_restart_budget_after_materializing_candidates(monkeypatch):
    import types
    from ml import vlm
    from ml.vlm import runtime

    clock = [0.0]
    class DelayedCandidates(list):
        def __iter__(self):
            clock[0] += 0.2
            return super().__iter__()
    monkeypatch.setattr(runtime, "time", types.SimpleNamespace(monotonic=lambda: clock[0]), raising=False)
    original = candidate()
    backend = FixtureBackend(["reject"])
    try:
        vlm.configure(backend=backend, catalog={"fixture_action": catalog_entry()},
                      frame_source=FixtureFrames(), timebase_verified=True)
        assert vlm.verify(DelayedCandidates([original]), "synthetic.mp4", 0.1)[0] is original
        assert backend.requests == []
    finally:
        vlm.configure(catalog={})


def test_production_failed_startup_is_fail_closed_without_retry():
    from ml import vlm

    class FailedStartup(FixtureBackend):
        validations = 0
        closed = False

        def validate(self, timeout_s):
            self.validations += 1
            raise vlm.ModelUnavailable("PRIVATE synthetic startup failure")

        def close(self):
            self.closed = True

    backend = FailedStartup([])
    originals = [candidate()]
    try:
        assert vlm.configure(backend=backend, catalog={"fixture_action": catalog_entry()},
                             frame_source=FixtureFrames(), timebase_verified=True) is False
        assert vlm.verify(originals, "synthetic.mp4", 0.01)[0] is originals[0]
        assert vlm.verify(originals, "synthetic.mp4", 0.01)[0] is originals[0]
        assert backend.validations == 1
        assert backend.requests == []
        assert backend.closed
    finally:
        vlm.configure(catalog={})


@pytest.mark.parametrize("flag", ["timebase_verified", "allow_rejection"])
@pytest.mark.parametrize("value", ["true", 1, None])
def test_safety_flags_require_boolean_values(flag, value):
    from ml.vlm import verify_pipeline_candidates

    with pytest.raises(ValueError, match="must be booleans"):
        verify_pipeline_candidates("fixture.mp4", [], {}, budget_sec=2.0, **{flag: value})
