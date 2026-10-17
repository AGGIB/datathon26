"""K2 augmentation preserves baseline answers; all evidence here is scripted."""
from dataclasses import replace
import json
from types import SimpleNamespace
import time

import pytest

from ml.candidates import Candidate
from ml.kb import catalog
from ml.vlm import type_runtime as runtime
from ml.vlm.contracts import Frame


def candidate(kind="safe_left_open", t=1., confidence=.6):
    return Candidate(kind, t, t + 1, t, confidence, "Синтетический пример", (t,))


@pytest.fixture(autouse=True)
def reset_runtime():
    runtime.configure_discovery()
    yield
    runtime.configure_discovery()


def test_unavailable_runtime_preserves_every_original_without_decoding(monkeypatch):
    monkeypatch.setattr(runtime, "discover_types", lambda *a, **kw: pytest.fail("unavailable model must not decode"))
    originals = [candidate(), candidate(confidence=.2)]
    result = runtime.augment_types("private.mov", originals, catalog(), duration_sec=20., deadline=time.monotonic() + 2)
    assert all(actual is expected for actual, expected in zip(result, originals))


def test_failed_startup_is_not_retried_by_video_jobs(monkeypatch):
    calls = []
    class Unavailable:
        def validate(self, timeout_s):
            calls.append(timeout_s)
            raise RuntimeError("scripted startup failure")
    assert not runtime.configure_discovery(backend=Unavailable())
    for _ in range(2):
        original = candidate()
        assert runtime.augment_types("private.mov", [original], catalog(), duration_sec=20., deadline=None) == [original]
    assert calls == [180]


def test_shared_deadline_and_baseline_slots_are_preserved(monkeypatch):
    originals = [candidate("safe_left_open", i * 3., .51) for i in range(4)]
    low = candidate("door_left_open", 18., .2)
    originals.append(low)
    additions = [candidate("food_or_drink_at_workplace", 2., .8),
                 candidate("personal_items_on_desk", 8., .99)]
    deadline = time.monotonic() + 2.
    def discover(*args, **kwargs):
        assert args[1] == originals
        assert kwargs["budget"].deadline == deadline
        assert kwargs["budget"].max_calls == 2
        return SimpleNamespace(added_candidates=additions, status="completed", calls_used=1)
    monkeypatch.setattr(runtime, "discover_types", discover)
    assert runtime.configure_discovery(backend=object())
    result = runtime.augment_types("private.mov", originals, catalog(), duration_sec=20., deadline=deadline)
    assert all(actual is expected for actual, expected in zip(result, originals))
    assert result[-1] is additions[1] and len(result) == len(originals) + 1
    from ml.assemble import select
    selected = select(result, catalog())
    assert len(selected) == 5
    assert all(original in selected for original in originals[:-1])


@pytest.mark.parametrize("failure", [TimeoutError, RuntimeError])
def test_discovery_failure_does_not_suppress_cv(monkeypatch, failure):
    assert runtime.configure_discovery(backend=object())
    def fail(*args, **kwargs):
        raise failure("scripted")
    monkeypatch.setattr(runtime, "discover_types", fail)
    original = candidate()
    assert runtime.augment_types("private.mov", [original], catalog(), duration_sec=20., deadline=None)[0] is original


def test_exhausted_deadline_and_full_baseline_do_not_spend_model_calls(monkeypatch):
    assert runtime.configure_discovery(backend=object())
    monkeypatch.setattr(runtime, "discover_types", lambda *a, **kw: pytest.fail("no calls expected"))
    original = candidate()
    assert runtime.augment_types("private.mov", [original], catalog(), duration_sec=20., deadline=0)[0] is original
    originals = [replace(original, t_start=i * 3., t_end=i * 3. + 1, t_peak=i * 3., evidence_times=(i * 3.,))
                 for i in range(5)]
    assert runtime.augment_types("private.mov", originals, catalog(), duration_sec=20., deadline=None) == originals


@pytest.mark.parametrize("stage", ["constructor", "validate_and_close"])
def test_owned_startup_and_cleanup_errors_fail_closed_without_job_reload(monkeypatch, stage):
    from ml.vlm import mlx_backend

    calls = []
    class Unavailable:
        def __init__(self, *args, **kwargs):
            calls.append("constructor")
            if stage == "constructor":
                raise ValueError("scripted constructor failure")
        def validate(self, timeout_s):
            calls.append("validate")
            raise RuntimeError("scripted load failure")
        def close(self):
            calls.append("close")
            raise RuntimeError("scripted cleanup failure")
    monkeypatch.setattr(mlx_backend, "LocalMLXBackend", Unavailable)
    assert not runtime.configure_discovery(model_path="scripted-model", python_executable="scripted-python")
    original = candidate()
    for _ in range(2):
        assert runtime.augment_types("private.mov", [original], catalog(), duration_sec=20., deadline=None) == [original]
    assert calls == (["constructor"] if stage == "constructor" else ["constructor", "validate", "close"])


class ScriptedFrames:
    def __init__(self):
        self.calls = 0

    def extract(self, video_path, timestamps, directory, timeout_s):
        self.calls += 1
        result = []
        for index, timestamp in enumerate(timestamps):
            path = directory / f"scripted-{index}.png"
            path.write_bytes(b"SYNTHETIC SCRIPTED EVIDENCE ONLY")
            result.append(Frame(timestamp, path, timestamp))
        return tuple(result)


class ScriptedBackend:
    def __init__(self, findings):
        self.findings = findings
        self.calls = 0

    def infer(self, request, timeout_s):
        self.calls += 1
        assert 0 < timeout_s <= runtime.MAX_VIDEO_SECONDS
        if isinstance(self.findings, str):
            return self.findings
        findings = []
        for kind, first, last, confidence in self.findings:
            start, end = request.frames[first].timestamp_s, request.frames[last].timestamp_s
            findings.append({"type_id": kind, "t_start": start, "t_peak": start, "t_end": end,
                             "confidence": confidence, "insufficient_evidence": False,
                             "observations": [{"timestamp_s": start, "text": "Видимое действие в начале"},
                                              {"timestamp_s": end, "text": "Видимое действие в конце"}]})
        return json.dumps({"findings": findings}, ensure_ascii=False)


@pytest.fixture
def scripted_pipeline(monkeypatch):
    import numpy as np
    from ml import evidence, pipeline
    from ml.shared_detector import SharedDetector
    from ml.video import VideoInfo

    info = VideoInfo(fps=10., frame_count=200, width=32, height=32)
    originals = [candidate("key_left_in_safe_lock", 0., .51),
                 candidate("safe_left_open", 5., .52),
                 candidate("door_left_open", 9., .53),
                 candidate("cash_left_on_desk_when_leaving", 13., .54)]
    monkeypatch.setenv("ML_SCENE", "0")
    monkeypatch.delenv("ML_VLM_DISCOVERY", raising=False)
    monkeypatch.setattr(pipeline, "_SHARED", SharedDetector(factory=object))
    monkeypatch.setattr(pipeline, "probe", lambda path: info)
    monkeypatch.setattr(pipeline, "_perceive", lambda *args: SimpleNamespace(info=info, frames=[object()]))
    monkeypatch.setattr(pipeline.rules, "find_candidates", lambda *args, **kwargs: originals)
    # Exercise actual rendering/assembly on synthetic pixels without a video/model.
    monkeypatch.setattr(evidence, "frames_at", lambda path, times: {
        timestamp: np.zeros((32, 32, 3), dtype=np.uint8) for timestamp in times})
    return pipeline, originals


@pytest.mark.parametrize("findings", [
    [("food_or_drink_at_workplace", 5, 6, .8), ("personal_items_on_desk", 7, 8, .99)],
    [("key_left_in_safe_lock", 1, 2, .99), ("personal_items_on_desk", 7, 8, .8)],
])
def test_actual_pipeline_keeps_cv_output_and_uses_free_slot_for_highest_separate_addition(
        scripted_pipeline, monkeypatch, findings):
    from backend.app.schemas import AnalysisResult

    pipeline, originals = scripted_pipeline
    baseline = pipeline.analyze("scripted.mp4")
    backend, source = ScriptedBackend(findings), ScriptedFrames()
    assert runtime.configure_discovery(backend=backend, frame_source=source)
    monkeypatch.setenv("ML_VLM_DISCOVERY", "1")
    result = pipeline.analyze("scripted.mp4", time_budget_sec=30.)
    AnalysisResult.model_validate(result)
    by_type = {row["type_id"]: row for row in result["violations"]}
    assert all(by_type[row["type_id"]] == row for row in baseline["violations"])
    assert len(result["violations"]) == 5 and "personal_items_on_desk" in by_type
    assert "food_or_drink_at_workplace" not in by_type
    assert [item.confidence for item in originals] == [.51, .52, .53, .54]
    assert backend.calls == source.calls == 1


@pytest.mark.parametrize("reply", ['{"findings":[]}', 'not JSON'])
def test_actual_pipeline_empty_or_malformed_discovery_leaves_cv_answer_identical(
        scripted_pipeline, monkeypatch, reply):
    pipeline, _ = scripted_pipeline
    baseline = pipeline.analyze("scripted.mp4")
    backend, source = ScriptedBackend(reply), ScriptedFrames()
    assert runtime.configure_discovery(backend=backend, frame_source=source)
    monkeypatch.setenv("ML_VLM_DISCOVERY", "1")
    assert pipeline.analyze("scripted.mp4", time_budget_sec=30.) == baseline
    assert backend.calls == source.calls == 1


@pytest.mark.parametrize("flag", [None, "0", "true", "yes"])
def test_actual_pipeline_discovery_requires_exact_explicit_opt_in(scripted_pipeline, monkeypatch, flag):
    pipeline, _ = scripted_pipeline
    baseline = pipeline.analyze("scripted.mp4")
    backend, source = ScriptedBackend('{"findings":[]}'), ScriptedFrames()
    assert runtime.configure_discovery(backend=backend, frame_source=source)
    monkeypatch.setattr(runtime, "configure_discovery", lambda **kwargs: pytest.fail("unexpected startup"))
    if flag is not None:
        monkeypatch.setenv("ML_VLM_DISCOVERY", flag)
    pipeline.preload()
    assert pipeline.analyze("scripted.mp4") == baseline
    assert backend.calls == source.calls == 0


def test_actual_pipeline_jobs_without_preload_never_start_a_model(scripted_pipeline, monkeypatch):
    pipeline, _ = scripted_pipeline
    baseline = pipeline.analyze("scripted.mp4")
    monkeypatch.setenv("ML_VLM_DISCOVERY", "1")
    monkeypatch.setattr(runtime, "configure_discovery", lambda **kwargs: pytest.fail("job must not configure"))
    assert pipeline.analyze("scripted.mp4", time_budget_sec=30.) == baseline


def test_actual_pipeline_discovery_uses_original_deadline_after_perception(scripted_pipeline, monkeypatch):
    pipeline, _ = scripted_pipeline
    baseline = pipeline.analyze("scripted.mp4")
    perceive = pipeline._perceive
    def delayed(*args):
        time.sleep(.03)
        return perceive(*args)
    monkeypatch.setattr(pipeline, "_perceive", delayed)
    backend, source = ScriptedBackend('{"findings":[]}'), ScriptedFrames()
    assert runtime.configure_discovery(backend=backend, frame_source=source)
    monkeypatch.setenv("ML_VLM_DISCOVERY", "1")
    assert pipeline.analyze("scripted.mp4", time_budget_sec=pipeline.FINISH_RESERVE_SEC + .01) == baseline
    assert backend.calls == source.calls == 0


class OwnedScriptedActions:
    def __init__(self, raw):
        self.raw = raw
        self.requests = []
        self.validated = []
        self.closed = 0

    def validate(self, timeout_s):
        self.validated.append(timeout_s)

    def close(self):
        self.closed += 1

    def infer(self, request, timeout_s):
        self.requests.append((request, timeout_s))
        return self.raw


def test_owned_startup_uses_catalog_action_wrapper_and_forwards_lifecycle(monkeypatch):
    from ml import kb
    from ml.vlm import action_backend, mlx_backend

    types = catalog()
    raw = OwnedScriptedActions('{"actions":[]}')
    constructed, catalog_calls = [], []
    def create(*args, **kwargs):
        constructed.append((args, kwargs))
        return raw
    monkeypatch.setattr(mlx_backend, "LocalMLXBackend", create)
    monkeypatch.setattr(kb, "catalog", lambda: catalog_calls.append(True) or types)
    assert runtime.configure_discovery(model_path="scripted-model", python_executable="scripted-python",
                                       startup_timeout_s=7)
    wrapper = runtime._configuration.backend
    assert isinstance(wrapper, action_backend.ActionContrastBackend) and wrapper.inner is raw
    assert runtime._configuration.owned and raw.validated == [7]
    assert catalog_calls == [True]
    assert constructed == [(("scripted-model",), {"python_executable": "scripted-python",
                                                  "max_new_tokens": 512, "max_image_side": 640})]
    runtime.configure_discovery()
    assert raw.closed == 1


@pytest.mark.parametrize('action,expected_addition', [
    ('visitor_or_courier_bag_handled',False),
    ('cashier_places_removed_outerwear_on_cashdesk',True),
])
def test_owned_action_decoder_normal_abstention_and_late_source_time_target(monkeypatch,action,expected_addition):
    from ml import kb
    from ml.vlm import mlx_backend
    from ml.vlm.discovery_policy import plan_windows

    types = catalog()
    payload = {'actions':[{'action':action,'insufficient_evidence':False,'confidence':.91,
                           'start_frame':10,'peak_frame':12,'end_frame':12}]}
    raw = OwnedScriptedActions(json.dumps(payload))
    constructed, loaded = [], []
    monkeypatch.setattr(mlx_backend,'LocalMLXBackend',lambda *a,**kw: constructed.append(True) or raw)
    monkeypatch.setattr(kb,'catalog',lambda: loaded.append(True) or types)
    source = ScriptedFrames()
    assert runtime.configure_discovery(model_path='scripted-model',python_executable='scripted-python',
                                       frame_source=source)
    original = candidate(confidence=.51)
    for _ in range(2):
        result = runtime.augment_types('synthetic.mp4',[original],types,duration_sec=20.,deadline=time.monotonic()+2)
        assert result[0] is original and result[0].confidence == .51
        if expected_addition:
            assert len(result)==2 and result[1].type_id=='personal_items_on_desk'
            assert result[1].t_peak == plan_windows(20.)[0].timestamps[11]
            assert result[1].t_start == plan_windows(20.)[0].timestamps[9]
        else:
            assert result == [original]
    assert constructed == loaded == [True] and raw.validated == [180]
    assert len(raw.requests)==source.calls==2
    request = raw.requests[-1][0]
    assert request.allowed_rule_ids == ()
    assert list(request.response_schema['properties']['actions']['items']['properties'])[0] == 'action'
    runtime.configure_discovery()
    assert raw.closed == 1


@pytest.mark.parametrize('stage',['catalog','wrapper','validate'])
def test_owned_startup_failure_closes_raw_worker_and_never_restarts_during_jobs(monkeypatch,stage):
    from ml import kb
    from ml.vlm import action_backend, mlx_backend

    types = catalog()
    raw = OwnedScriptedActions('{"actions":[]}')
    constructed = []
    monkeypatch.setattr(mlx_backend,'LocalMLXBackend',lambda *a,**kw: constructed.append(True) or raw)
    def fail(*args,**kwargs):
        raise RuntimeError('scripted startup failure')
    if stage=='catalog':
        monkeypatch.setattr(kb,'catalog',fail)
    elif stage=='wrapper':
        monkeypatch.setattr(action_backend,'ActionContrastBackend',fail)
    else:
        monkeypatch.setattr(raw,'validate',fail)
    assert not runtime.configure_discovery(model_path='scripted-model',python_executable='scripted-python')
    assert raw.closed == 1 and runtime._configuration is None
    original = candidate()
    for _ in range(2):
        assert runtime.augment_types('synthetic.mp4',[original],types,duration_sec=20.,deadline=None)[0] is original
    assert constructed == [True] and not raw.requests


def test_injected_backend_identity_and_lifecycle_ownership_remain_unchanged(monkeypatch):
    from ml import kb
    from ml.vlm import action_backend, mlx_backend

    types = catalog()
    injected = OwnedScriptedActions('{"findings":[]}')
    def unexpected(*args,**kwargs):
        pytest.fail('injected backend must not construct owned worker/wrapper or load catalog')
    monkeypatch.setattr(kb,'catalog',unexpected)
    monkeypatch.setattr(mlx_backend,'LocalMLXBackend',unexpected)
    monkeypatch.setattr(action_backend,'ActionContrastBackend',unexpected)
    assert runtime.configure_discovery(backend=injected,frame_source=ScriptedFrames())
    assert runtime._configuration.backend is injected and not runtime._configuration.owned
    original = candidate()
    assert runtime.augment_types('synthetic.mp4',[original],types,duration_sec=20.,deadline=None)==[original]
    assert injected.validated == [180] and len(injected.requests)==1
    runtime.configure_discovery()
    assert injected.closed == 0


def test_owned_wrapper_honors_current_catalog_disable_gate(monkeypatch):
    from ml import kb
    from ml.vlm import mlx_backend

    types = dict(catalog())
    types['personal_items_on_desk'] = replace(types['personal_items_on_desk'],enabled=False)
    raw = OwnedScriptedActions(json.dumps({'actions':[{
        'action':'cashier_places_removed_outerwear_on_cashdesk','insufficient_evidence':False,
        'confidence':.99,'start_frame':10,'peak_frame':12,'end_frame':12}]}))
    monkeypatch.setattr(mlx_backend,'LocalMLXBackend',lambda *a,**kw: raw)
    monkeypatch.setattr(kb,'catalog',lambda: types)
    assert runtime.configure_discovery(model_path='scripted-model',python_executable='scripted-python',
                                       frame_source=ScriptedFrames())
    original = candidate()
    assert runtime.augment_types('synthetic.mp4',[original],types,duration_sec=20.,deadline=None)==[original]
    allowed_actions = raw.requests[0][0].response_schema['properties']['actions']['items']['properties']['action']['enum']
    assert 'cashier_places_removed_outerwear_on_cashdesk' not in allowed_actions


@pytest.fixture
def scripted_verifier_module(monkeypatch):
    import sys
    from types import ModuleType

    module=ModuleType('ml.vlm.verified_action_backend')
    module.DECODER_POLICY='scripted_blind_verifier_policy'
    constructed=[]
    class Verified:
        def __init__(self,inner,*,catalog):
            self.inner,self.types=inner,catalog
            self.last_actions=()
            self.last_verifications=()
            constructed.append(self)
        def validate(self,timeout_s):
            self.inner.validate(timeout_s)
        def close(self):
            self.inner.close()
        def infer_with_budget(self,request,budget):
            if not budget.claim():
                raise RuntimeError('scripted budget exhausted')
            return self.inner.infer(request,budget.remaining())
    module.VerifiedActionBackend=Verified
    module.constructed=constructed
    monkeypatch.setitem(sys.modules,module.__name__,module)
    return module


def test_owned_verifier_startup_uses_single_raw_backend_current_catalog_and_lifecycle(
        monkeypatch,scripted_verifier_module):
    from ml import kb
    from ml.vlm import action_backend,mlx_backend

    types=catalog()
    raw=OwnedScriptedActions('{"findings":[]}')
    created=[]
    monkeypatch.setattr(mlx_backend,'LocalMLXBackend',lambda *a,**kw: created.append(True) or raw)
    monkeypatch.setattr(kb,'catalog',lambda: types)
    monkeypatch.setattr(action_backend,'ActionContrastBackend',lambda *a,**kw: pytest.fail('double wrapper'))
    assert runtime.configure_discovery(model_path='scripted-model',python_executable='scripted-python',
                                       verify_actions=True,frame_source=ScriptedFrames())
    config=runtime._configuration
    assert config.backend is scripted_verifier_module.constructed[0]
    assert config.backend.inner is raw and config.backend.types is types
    assert config.owned and config.max_calls==4 and raw.validated==[180] and created==[True]
    original=candidate()
    for _ in range(2):
        assert runtime.augment_types('scripted.mp4',[original],types,duration_sec=20.,deadline=None)==[original]
    assert len(raw.requests)==2 and raw.validated==[180] and created==[True]
    runtime.configure_discovery()
    assert raw.closed==1


def test_injected_verifier_flag_only_declares_cap_without_wrapper_or_lifecycle_ownership(
        monkeypatch,scripted_verifier_module):
    from ml import kb
    from ml.vlm import mlx_backend

    injected=OwnedScriptedActions('{"findings":[]}')
    def unexpected(*args,**kwargs):
        pytest.fail('injected caller owns backend and wrapper')
    monkeypatch.setattr(mlx_backend,'LocalMLXBackend',unexpected)
    monkeypatch.setattr(kb,'catalog',unexpected)
    monkeypatch.setattr(scripted_verifier_module,'VerifiedActionBackend',unexpected)
    assert runtime.configure_discovery(backend=injected,verify_actions=True)
    assert runtime._configuration.backend is injected and not runtime._configuration.owned
    seen=[]
    def discover(*args,**kwargs):
        seen.append(kwargs['budget'].max_calls)
        assert kwargs['backend'] is injected
        return SimpleNamespace(added_candidates=(),status='completed',calls_used=0)
    monkeypatch.setattr(runtime,'discover_types',discover)
    original=candidate()
    assert runtime.augment_types('scripted.mp4',[original],catalog(),duration_sec=20.,deadline=None)==[original]
    assert seen==[4] and injected.validated==[180]
    runtime.configure_discovery()
    assert injected.closed==0


@pytest.mark.parametrize('invalid',[1,0,'1',None])
def test_runtime_verifier_flag_is_exact_boolean_and_invalid_reconfiguration_preserves_configuration(invalid):
    injected=OwnedScriptedActions('{"findings":[]}')
    assert runtime.configure_discovery(backend=injected)
    config=runtime._configuration
    with pytest.raises(ValueError,match='verify_actions must be boolean'):
        runtime.configure_discovery(backend=object(),verify_actions=invalid)
    assert runtime._configuration is config and config.max_calls==2
    assert injected.validated==[180] and injected.closed==0


@pytest.mark.parametrize('stage',['wrapper','catalog','validate'])
def test_owned_verified_startup_failure_closes_single_raw_worker(monkeypatch,scripted_verifier_module,stage):
    from ml import kb
    from ml.vlm import mlx_backend

    raw=OwnedScriptedActions('{"findings":[]}')
    monkeypatch.setattr(mlx_backend,'LocalMLXBackend',lambda *a,**kw: raw)
    def fail(*args,**kwargs):
        raise RuntimeError('scripted verifier startup failure')
    if stage=='wrapper':
        monkeypatch.setattr(scripted_verifier_module,'VerifiedActionBackend',fail)
    elif stage=='catalog':
        monkeypatch.setattr(kb,'catalog',fail)
    else:
        monkeypatch.setattr(raw,'validate',fail)
    assert not runtime.configure_discovery(model_path='scripted-model',python_executable='scripted-python',verify_actions=True)
    assert raw.closed==1 and runtime._configuration is None


@pytest.mark.parametrize('discovery_flag,verify_flag,expected',[
    ('1','1',True),('1','0',False),('1','true',False),('0','1',None),
])
def test_pipeline_preload_verification_flag_requires_explicit_discovery_opt_in(
        monkeypatch,discovery_flag,verify_flag,expected):
    from ml import pipeline

    seen=[]
    monkeypatch.setenv('ML_VLM_DISCOVERY',discovery_flag)
    monkeypatch.setenv('ML_VLM_VERIFY_ACTIONS',verify_flag)
    monkeypatch.setenv('ML_VLM_MODEL','scripted-model')
    monkeypatch.setenv('ML_VLM_PYTHON','scripted-python')
    monkeypatch.setattr(runtime,'configure_discovery',lambda **kwargs: seen.append(kwargs) or True)
    pipeline.preload()
    if expected is None:
        assert not seen
    else:
        assert seen==[{'model_path':'scripted-model','python_executable':'scripted-python','verify_actions':expected}]


@pytest.mark.parametrize('proposal_action,object_kind,expected_calls,expected_addition',[
    ('visitor_or_courier_bag_handled','bag',1,False),
    ('cashier_places_removed_outerwear_on_cashdesk','outerwear',2,True),
    ('cashier_places_removed_outerwear_on_cashdesk','bag',2,False),
])
def test_actual_owned_v5_verifier_and_shared_budget_preserve_cv_without_model_or_decode(
        monkeypatch,proposal_action,object_kind,expected_calls,expected_addition):
    from ml import kb
    from ml.vlm import mlx_backend
    from ml.vlm.verified_action_backend import VerifiedActionBackend
    from ml.vlm.discovery_policy import plan_windows

    actions={'actions':[{'action':proposal_action,'insufficient_evidence':False,'confidence':.95,
                         'start_frame':10,'peak_frame':11,'end_frame':12}]}
    facts={'episodes':[{'id':1,'rows':[
        [1,'a','cashier','x',object_kind,'held','hand','none'],
        [2,'a','cashier','x',object_kind,'placing','own_workdesk','none'],
        [3,'a','cashier','x',object_kind,'resting','own_workdesk','none']]}]}
    class Raw(OwnedScriptedActions):
        def __init__(self):
            super().__init__('')
            self.replies=[json.dumps(actions),json.dumps(facts)]
        def infer(self,request,timeout_s):
            self.requests.append((request,timeout_s))
            return self.replies.pop(0)
    raw=Raw()
    types=catalog()
    monkeypatch.setattr(kb,'catalog',lambda: types)
    monkeypatch.setattr(mlx_backend,'LocalMLXBackend',lambda *a,**kw: raw)
    actual_discover=runtime.discover_types
    measured=[]
    def observe(*args,**kwargs):
        report=actual_discover(*args,**kwargs)
        measured.append((report.calls_used,kwargs['budget'].calls_used,kwargs['budget'].max_calls))
        return report
    monkeypatch.setattr(runtime,'discover_types',observe)
    assert runtime.configure_discovery(model_path='scripted-model',python_executable='scripted-python',
                                       verify_actions=True,frame_source=ScriptedFrames())
    wrapper=runtime._configuration.backend
    assert isinstance(wrapper,VerifiedActionBackend)
    original=candidate(confidence=.51)
    result=runtime.augment_types('scripted.mp4',[original],types,duration_sec=20.,deadline=time.monotonic()+2)
    assert result[0] is original and result[0].confidence==.51
    assert len(result)==1+expected_addition
    if expected_addition:
        assert result[1].type_id=='personal_items_on_desk'
        assert result[1].t_peak==plan_windows(20.)[0].timestamps[10]
    assert measured==[(expected_calls,expected_calls,4)]
    assert len(raw.requests)==wrapper.calls_used==expected_calls
    if expected_calls==2:
        proposal_frames=raw.requests[0][0].frames
        verifier_frames=raw.requests[1][0].frames
        assert len(verifier_frames)==3
        assert all(original is focused for original,focused in zip(proposal_frames[-3:],verifier_frames,strict=True))
        assert 'cashier_places_removed_outerwear_on_cashdesk' not in raw.requests[1][0].prompt
        assert wrapper.last_verifications[0]['accepted'] is expected_addition
    assert raw.validated==[180]
    runtime.configure_discovery()
    assert raw.closed==1
