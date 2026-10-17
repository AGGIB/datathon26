"""Sampling and merge policy regressions; no real media or ML runtime."""
import math

import pytest

from ml.tools.vlm_type_probe import merge_predictions, plan_windows


@pytest.mark.parametrize("duration", [0.01, 21.37445678, 30, 30.000001, 72.374533, 142.4])
def test_frozen_windows_cover_whole_clip_without_exceeding_frame_budget(duration):
    windows = plan_windows(duration)
    assert len(windows) == (1 if duration <= 30 else 2)
    times = {t for window in windows for t in window.timestamps}
    assert 0 == min(times) < max(times) < duration
    assert len(times) <= 24
    assert all(len(window.timestamps) <= 12 for window in windows)
    assert all(window.t_start <= min(window.timestamps) <= max(window.timestamps) <= window.t_end
               for window in windows)


@pytest.mark.parametrize("duration", [True, 0, -1, math.inf, math.nan])
def test_bad_durations_are_rejected(duration):
    with pytest.raises(ValueError):
        plan_windows(duration)


def test_stronger_vlm_predictions_never_displace_or_mutate_cv():
    baseline = [{"type_id": f"cv_{i}", "t": i, "conf": 0.6} for i in range(5)]
    additions = [{"type_id": "vlm", "t": 6, "conf": 0.99}]
    merged = merge_predictions(baseline, additions)
    assert merged == baseline
    assert all(a is not b for a, b in zip(merged, baseline))


def test_additions_fill_spare_slots_and_close_same_type_is_not_duplicated():
    baseline = [{"type_id": "action", "t": 1.0, "conf": 0.5}]
    additions = [{"type_id": "action", "t": 2.0, "conf": 0.99},
                 {"type_id": "new", "t": 3.0, "conf": 0.8},
                 {"type_id": "another_episode", "t": 0.0, "conf": 0.9}]
    merged = merge_predictions(baseline, additions, limit=2)
    assert merged == [additions[2], baseline[0]]
    assert baseline == [{"type_id": "action", "t": 1.0, "conf": 0.5}]


@pytest.fixture
def local_probe(tmp_path, monkeypatch):
    import json
    import sys
    from dataclasses import dataclass
    from pathlib import Path
    from types import SimpleNamespace

    from ml.tools import vlm_type_probe as probe_module
    from ml import kb, video
    from ml.vlm import discovery, mlx_backend

    monkeypatch.setattr(probe_module, 'ROOT', tmp_path)
    kb_path = tmp_path / 'kb/violations.yaml'
    kb_path.parent.mkdir()
    kb_path.write_text('violations: []\n')
    monkeypatch.setattr(kb, 'KB_PATH', kb_path)
    (tmp_path / 'ml').mkdir()
    (tmp_path / 'ml/fixture.py').write_text('# synthetic code fingerprint\n')
    videos = tmp_path / 'videos'
    videos.mkdir()
    for name in ['X-1.mov', 'X-3.mov']:
        (videos / name).write_bytes(b'SYNTHETIC IDENTICAL CONTENT; NEVER DECODED')
    model = tmp_path / 'model'
    model.mkdir()
    for name, value in [('config.json', {'model_type': 'qwen3_vl'}),
                        ('preprocessor_config.json', {}), ('tokenizer_config.json', {}), ('tokenizer.json', {})]:
        (model / name).write_text(json.dumps(value))
    (model / 'model.safetensors').write_bytes(b'SYNTHETIC WEIGHTS; NEVER LOADED')
    baseline = {'X-1': {'violations': [{'type_id': 'cv_one', 't': 1.0, 'conf': 0.5}]},
                'X-3': {'violations': [{'type_id': 'cv_other', 't': 2.0, 'conf': 0.6}]}}
    baseline_path = tmp_path / 'baseline.json'
    baseline_path.write_text(json.dumps(baseline))
    args = SimpleNamespace(model=model, python=Path(sys.executable), videos=videos, baseline=baseline_path,
                           out=tmp_path / 'var/results.json', only='', budget=10.0, max_tokens=512, frame_side=640)
    calls, prepared, closed = [], [], []

    @dataclass(frozen=True)
    class Addition:
        type_id: str = 'food_or_drink_at_workplace'
        t_peak: float = 3.0
        confidence: float = 0.9

    @dataclass
    class Report:
        status: str = 'completed'
        calls_used: int = 1
        added_candidates: tuple = (Addition(),)

    class Backend:
        def __init__(self, *args, **kwargs):
            pass

        def validate(self, timeout_s):
            prepared.append(timeout_s)

        def close(self):
            closed.append(True)

    def discover(path, originals, types, **kwargs):
        assert originals == ()  # CV predictions cannot bias discovery inputs.
        assert kwargs['windows'] == plan_windows(20.0)
        calls.append(path)
        return Report()

    monkeypatch.setattr(mlx_backend, 'LocalMLXBackend', Backend)
    monkeypatch.setattr(discovery, 'discover_types', discover)
    monkeypatch.setattr(kb, 'catalog', lambda: {})
    monkeypatch.setattr(video, 'probe', lambda path: SimpleNamespace(duration=20.0))
    monkeypatch.setattr(probe_module, '_runtime_versions', lambda path: {'python': 'synthetic', 'packages': {'mlx': '0'}})
    return SimpleNamespace(args=args, baseline=baseline, calls=calls, prepared=prepared, closed=closed,
                           module=probe_module, discovery=discovery, backend=Backend, root=tmp_path)


def test_content_deduplication_preserves_each_cv_baseline_and_fingerprints(local_probe):
    import json

    case = local_probe
    before = json.loads(case.args.baseline.read_text())
    results = case.module.run(case.args)
    assert len(case.calls) == len(case.prepared) == len(case.closed) == 1
    for key in ['X-1', 'X-3']:
        assert before[key]['violations'][0] in results[key]['violations']
    assert json.loads(case.args.baseline.read_text()) == before
    audit = json.loads(case.args.out.with_suffix('.audit.json').read_text())
    assert not audit['videos']['X-1']['inference_reused']
    assert audit['videos']['X-3']['inference_reused']
    assert audit['videos']['X-3']['actual_calls_used'] == 0
    assert audit['config']['ground_truth_in_model_input'] is False
    assert audit['config']['timing_scope'] == 'warm_vlm_plus_frame_io_only_CV_replayed'
    assert audit['config']['sampling_policy'] == 'positive_actions_uniform_halves_12_v2'
    assert audit['config']['model_artifact_sha256']['preprocessor_config.json']
    assert audit['config']['source_sha256']['ml/fixture.py']
    assert audit['config']['runtime']['packages']['mlx'] == '0'
    assert audit['config']['parent_runtime']['python'] == 'synthetic'
    assert audit['config']['parent_python']
    assert audit['config']['kb_sha256']
    assert len(audit['config_sha256']) == 64 and audit['code_inputs_stable']


def test_inference_failure_retains_baseline_and_never_logs_exception_text(local_probe, monkeypatch, caplog):
    case = local_probe

    def fail(*args, **kwargs):
        raise RuntimeError('SYNTHETIC_PRIVATE_OBSERVATION')

    monkeypatch.setattr(case.discovery, 'discover_types', fail)
    results = case.module.run(case.args)
    for key in results:
        assert results[key]['violations'] == case.baseline[key]['violations']
        assert results[key]['verdict'] == 'violation' and results[key]['vlm_error'] == 'RuntimeError'
    assert all('error' not in row for row in results.values())
    assert 'SYNTHETIC_PRIVATE_OBSERVATION' not in caplog.text
    assert len(case.closed) == 1


@pytest.mark.parametrize('status', ['backend_error', 'video_timeout', 'frames_unavailable',
                                   'model_unavailable', 'prompt_too_long'])
@pytest.mark.parametrize('partial', [False, True])
def test_returned_failure_is_flagged_without_losing_cv_or_accepted_partial_findings(local_probe, monkeypatch,
                                                                                 status, partial):
    case = local_probe
    original = case.discovery.discover_types

    def failed(*args, **kwargs):
        report = original(*args, **kwargs)
        report.status = status
        if not partial:
            report.added_candidates = ()
        return report

    monkeypatch.setattr(case.discovery, 'discover_types', failed)
    results = case.module.run(case.args)
    assert all(row['vlm_error'] == status for row in results.values())
    for key, row in results.items():
        assert all(event in row['violations'] for event in case.baseline[key]['violations'])
        assert any(event['type_id'] == 'food_or_drink_at_workplace' for event in row['violations']) is partial
    assert int(any('vlm_error' in row for row in results.values())) == 1


@pytest.mark.parametrize('status', ['completed', 'budget_exhausted', 'no_windows', 'no_verified_types'])
def test_expected_bounded_completion_or_abstention_is_not_a_runtime_error(local_probe, monkeypatch, status):
    case = local_probe
    original = case.discovery.discover_types

    def complete(*args, **kwargs):
        report = original(*args, **kwargs)
        report.status = status
        return report

    monkeypatch.setattr(case.discovery, 'discover_types', complete)
    assert not any('vlm_error' in row for row in case.module.run(case.args).values())


def test_kb_change_changes_fingerprint_and_mid_run_change_invalidates_stability(local_probe, monkeypatch):
    import json
    from pathlib import Path
    case = local_probe
    case.module.run(case.args)
    first = json.loads(case.args.out.with_suffix('.audit.json').read_text())
    kb_path = Path(first['config']['kb_path'])
    kb_path.write_text('violations: []\n# synthetic eligibility update\n')
    case.args.out = case.root / 'var/second-run.json'
    case.module.run(case.args)
    second = json.loads(case.args.out.with_suffix('.audit.json').read_text())
    assert first['config_sha256'] != second['config_sha256']
    assert first['config']['kb_sha256'] != second['config']['kb_sha256']
    assert second['code_inputs_stable']
    original = case.discovery.discover_types

    def mutate(*args, **kwargs):
        kb_path.write_text('violations: []\n# changed during inference\n')
        return original(*args, **kwargs)

    monkeypatch.setattr(case.discovery, 'discover_types', mutate)
    case.args.out = case.root / 'var/third-run.json'
    case.module.run(case.args)
    assert json.loads(case.args.out.with_suffix('.audit.json').read_text())['code_inputs_stable'] is False


def test_parent_decode_runtime_versions_are_separate_and_affect_fingerprint(local_probe, monkeypatch):
    import json
    from pathlib import Path
    import sys
    case = local_probe
    case.args.python = case.root / 'synthetic-mlx-python'
    case.args.python.write_text('not executed')
    parent_versions = {'python': 'synthetic CV', 'packages': {'opencv-python': '4.synthetic'}}
    model_versions = {'python': 'synthetic MLX', 'packages': {'mlx': '0.synthetic'}}
    monkeypatch.setattr(case.module, '_runtime_versions', lambda executable:
                        parent_versions if Path(executable) == Path(sys.executable) else model_versions)
    case.module.run(case.args)
    first = json.loads(case.args.out.with_suffix('.audit.json').read_text())
    assert first['config']['parent_runtime'] == parent_versions
    assert first['config']['runtime'] == model_versions
    parent_versions['packages']['opencv-python'] = '5.synthetic'
    case.args.out = case.root / 'var/second-run.json'
    case.module.run(case.args)
    second = json.loads(case.args.out.with_suffix('.audit.json').read_text())
    assert first['config_sha256'] != second['config_sha256']


def test_startup_failure_still_writes_protected_cv_outputs(local_probe, monkeypatch):
    import json
    case = local_probe

    def fail(*args, **kwargs):
        raise RuntimeError('SYNTHETIC_PRIVATE_PATH')

    monkeypatch.setattr(case.backend, 'validate', fail)
    results = case.module.run(case.args)
    for key in results:
        assert results[key]['violations'] == case.baseline[key]['violations']
        assert results[key]['vlm_error'] == 'vlm_startup_failed'
    assert all('error' not in row for row in results.values())
    assert case.calls == [] and len(case.closed) == 1
    audit = json.loads(case.args.out.with_suffix('.audit.json').read_text())
    assert audit['startup'] == {'status': 'failed', 'error_kind': 'RuntimeError'}
    assert 'SYNTHETIC_PRIVATE_PATH' not in case.args.out.with_suffix('.audit.json').read_text()


@pytest.mark.parametrize('mutation', ['output_outside', 'duplicate_video', 'missing_requested',
                                    'missing_baseline', 'baseline_nan', 'baseline_collision', 'invalid_config'])
def test_invalid_inputs_are_rejected_before_backend_start(local_probe, mutation):
    import json
    case = local_probe
    if mutation == 'output_outside':
        case.args.out = case.root / 'public-result.json'
    elif mutation == 'duplicate_video':
        (case.args.videos / 'Х-1.mp4').write_bytes(b'SYNTHETIC')
    elif mutation == 'missing_requested':
        case.args.only = 'X-99'
    elif mutation == 'missing_baseline':
        case.args.baseline.write_text(json.dumps({'X-1': case.baseline['X-1']}))
    elif mutation == 'baseline_nan':
        case.baseline['X-1']['violations'][0]['conf'] = math.nan
        case.args.baseline.write_text(json.dumps(case.baseline))
    elif mutation == 'baseline_collision':
        case.baseline['Х-1'] = case.baseline['X-1']
        case.args.baseline.write_text(json.dumps(case.baseline))
    else:
        case.args.max_tokens = True
    with pytest.raises(ValueError):
        case.module.run(case.args)
    assert not case.prepared
    assert not case.args.out.exists()


def test_new_camera_technical_names_are_supported_without_labels(local_probe):
    case = local_probe
    (case.args.videos / 'camera-new-control.mp4').write_bytes(b'SYNTHETIC NEW CONTENT')
    case.args.baseline = None
    case.args.only = 'camera-new-control'
    results = case.module.run(case.args)
    assert set(results) == {'CAMERA-NEW-CONTROL'}


def test_processor_change_changes_configuration_fingerprint(local_probe):
    import json
    case = local_probe
    case.module.run(case.args)
    first = json.loads(case.args.out.with_suffix('.audit.json').read_text())['config_sha256']
    (case.args.model / 'preprocessor_config.json').write_text('{"synthetic_resolution":512}')
    case.args.out = case.root / 'var/second-run.json'
    case.module.run(case.args)
    second = json.loads(case.args.out.with_suffix('.audit.json').read_text())['config_sha256']
    assert first != second


def test_atomic_save_never_follows_predictable_partial_symlink(local_probe):
    case = local_probe
    target = case.root / 'victim.txt'
    target.write_text('preserved')
    case.args.out.parent.mkdir()
    case.args.out.with_suffix('.json.part').symlink_to(target)
    case.module._save(case.args.out, {'synthetic': True})
    assert target.read_text() == 'preserved'


def test_output_lease_rechecks_results_after_slow_preparation(local_probe):
    case = local_probe
    case.args.out.parent.mkdir()
    case.args.out.write_text('existing run')
    with pytest.raises(ValueError, match='new_output_path_required'):
        with case.module._output_lease(case.args.out):
            pytest.fail('existing experiment overwritten')
    assert case.args.out.read_text() == 'existing run'
    assert not case.args.out.with_suffix('.json.lock').exists()


def test_cli_errors_never_echo_private_values(monkeypatch, capsys):
    import sys
    from ml.tools import vlm_type_probe as probe_module
    monkeypatch.setattr(sys, 'argv', ['probe', '--unknown', 'SYNTHETIC_PRIVATE_INPUT'])
    assert probe_module.main() == 1
    captured = capsys.readouterr()
    assert 'ERROR:' in captured.err and 'SYNTHETIC_PRIVATE_INPUT' not in captured.err


def test_action_contrast_is_optional_fingerprinted_and_lifecycle_owned_by_probe(local_probe, monkeypatch):
    import json
    from ml.vlm import action_backend

    case = local_probe
    wrapped = []
    class Wrapper:
        def __init__(self, inner, *, catalog):
            self.inner = inner
            self.last_actions = ()
            wrapped.append(self)
        def validate(self, timeout_s):
            self.inner.validate(timeout_s)
        def close(self):
            self.inner.close()
    monkeypatch.setattr(action_backend, 'ActionContrastBackend', Wrapper)
    case.module.run(case.args)  # legacy SimpleNamespace has no action_contrast field.
    first = json.loads(case.args.out.with_suffix('.audit.json').read_text())
    assert first['config']['decoder_policy'] == 'direct_discovery' and not wrapped
    case.args.out = case.root/'var/contrast.json'
    case.args.action_contrast = True
    original = case.discovery.discover_types
    def discover(*args, **kwargs):
        assert isinstance(kwargs['backend'], Wrapper)
        return original(*args, **kwargs)
    monkeypatch.setattr(case.discovery, 'discover_types', discover)
    case.module.run(case.args)
    second = json.loads(case.args.out.with_suffix('.audit.json').read_text())
    assert second['config']['decoder_policy'] == action_backend.DECODER_POLICY
    assert first['config_sha256'] != second['config_sha256']
    assert len(wrapped) == 1 and len(case.prepared) == len(case.closed) == 2
    assert second['videos']['X-1']['decoder_last_call_actions'] == []
    assert 'decoder_last_call_actions' not in second['videos']['X-3']  # reuse links its source.


@pytest.mark.parametrize('action_contrast',[False,True])
def test_verify_actions_overrides_contrast_uses_four_call_budget_and_coded_audit(local_probe,monkeypatch,action_contrast):
    import json
    import sys
    from types import ModuleType
    from ml.vlm import action_backend

    case = local_probe
    created=[]
    module=ModuleType('ml.vlm.verified_action_backend')
    module.DECODER_POLICY='scripted_blind_verifier_policy'
    class Verifier:
        def __init__(self,inner,*,catalog):
            self.inner=inner
            self.last_actions=({'action':'unclear','start_frame':1,'peak_frame':1,'end_frame':1,
                                'confidence':.5,'insufficient_evidence':True},)
            self.last_verifications=({'episode_id':1,'accepted':False,'reason':'unclear_object',
                                      'raw_error':'PRIVATE_UNSTRUCTURED_ERROR'},)
            created.append(self)
        def validate(self,timeout_s):
            self.inner.validate(timeout_s)
        def close(self):
            self.inner.close()
    module.VerifiedActionBackend=Verifier
    monkeypatch.setitem(sys.modules,module.__name__,module)
    monkeypatch.setattr(action_backend,'ActionContrastBackend',lambda *a,**kw: pytest.fail('double wrapper'))
    case.args.verify_actions=True
    case.args.action_contrast=action_contrast
    original=case.discovery.discover_types
    def discover(*args,**kwargs):
        assert kwargs['backend'] is created[0] and kwargs['budget'].max_calls==4
        report=original(*args,**kwargs)
        report.calls_used=4
        return report
    monkeypatch.setattr(case.discovery,'discover_types',discover)
    case.module.run(case.args)
    audit=json.loads(case.args.out.with_suffix('.audit.json').read_text())
    assert audit['config']['decoder_policy']==module.DECODER_POLICY
    assert audit['config']['max_model_calls_per_video']==4
    assert len(created)==len(case.prepared)==len(case.closed)==1
    assert audit['videos']['X-1']['actual_calls_used']==4
    assert audit['videos']['X-3']['actual_calls_used']==0
    assert audit['videos']['X-1']['verifier_last_call_decisions']==[
        {'episode_id':1,'accepted':False,'reason':'unclear_object'}]
    assert 'PRIVATE_UNSTRUCTURED_ERROR' not in json.dumps(audit)
    assert 'verifier_last_call_decisions' not in audit['videos']['X-3']
    assert 'decoder_last_call_actions' in audit['videos']['X-1']


@pytest.mark.parametrize('name',['verify_actions','action_contrast','compact_verify_actions'])
@pytest.mark.parametrize('invalid',[1,0,'true',None])
def test_experiment_mode_flags_require_exact_booleans(local_probe,name,invalid):
    case=local_probe
    setattr(case.args,name,invalid)
    with pytest.raises(ValueError,match=name+'_must_be_boolean'):
        case.module.run(case.args)
    assert not case.prepared and not case.calls and not case.closed


def test_cli_parses_verification_as_explicit_opt_in_only(monkeypatch):
    import sys
    from ml.tools import vlm_type_probe as module

    received=[]
    monkeypatch.setattr(module,'run',lambda args: received.append(args) or {})
    argv=['probe','--model','scripted-model','--python','scripted-python','--out','var/scripted.json']
    monkeypatch.setattr(sys,'argv',argv)
    assert module.main()==0 and received[-1].verify_actions is False and received[-1].compact_verify_actions is False
    monkeypatch.setattr(sys,'argv',argv+['--verify-actions','--action-contrast'])
    assert module.main()==0 and received[-1].verify_actions is True and received[-1].action_contrast is True
    monkeypatch.setattr(sys,'argv',argv+['--compact-verify-actions'])
    assert module.main()==0 and received[-1].compact_verify_actions is True and received[-1].verify_actions is False


def test_coded_verifier_diagnostics_never_copy_free_text():
    from ml.tools.vlm_type_probe import _coded_verifications

    assert _coded_verifications([{'episode_id':1,'accepted':True,'reason':'supported','observations':'PRIVATE'}])==[
        {'episode_id':1,'accepted':True,'reason':'supported'}]
    assert _coded_verifications([{'episode_id':True,'accepted':True,'reason':'supported'},
                                {'episode_id':1,'accepted':1,'reason':'supported'},
                                {'episode_id':1,'accepted':False,'reason':'PRIVATE ERROR PATH'}])==[]


@pytest.mark.parametrize('legacy,contrast',[(False,False),(False,True),(True,False),(True,True)])
def test_compact_probe_precedence_fingerprint_physical_budget_and_safe_diagnostics(
        local_probe,monkeypatch,legacy,contrast):
    import json
    import sys
    from types import ModuleType
    from ml.vlm import action_backend,verified_action_backend

    case=local_probe
    module=ModuleType('ml.vlm.compact_verified_action_backend')
    module.DECODER_POLICY='scripted_compact_blind_policy'
    created=[]
    class Compact:
        def __init__(self,inner,*,catalog):
            self.inner=inner
            self.last_actions=()
            self.last_verifications=({'episode_id':1,'accepted':False,'reason':'primitive_evidence_not_supported',
                                      'observations':'SYNTHETIC_PRIVATE_VALUE'},)
            created.append(self)
        def validate(self,timeout_s):
            self.inner.validate(timeout_s)
        def close(self):
            self.inner.close()
    module.CompactVerifiedActionBackend=Compact
    monkeypatch.setitem(sys.modules,module.__name__,module)
    def wrong_wrapper(*args,**kwargs):
        pytest.fail('compact mode must construct exactly one wrapper')
    monkeypatch.setattr(action_backend,'ActionContrastBackend',wrong_wrapper)
    monkeypatch.setattr(verified_action_backend,'VerifiedActionBackend',wrong_wrapper)
    case.args.compact_verify_actions=True
    case.args.verify_actions=legacy
    case.args.action_contrast=contrast
    original=case.discovery.discover_types
    def discover(*args,**kwargs):
        assert kwargs['backend'] is created[0] and kwargs['budget'].max_calls==4
        report=original(*args,**kwargs)
        report.calls_used=4
        return report
    monkeypatch.setattr(case.discovery,'discover_types',discover)
    results=case.module.run(case.args)
    for key in results:
        assert all(row in results[key]['violations'] for row in case.baseline[key]['violations'])
    audit=json.loads(case.args.out.with_suffix('.audit.json').read_text())
    assert audit['config']['decoder_policy']==module.DECODER_POLICY
    assert audit['config']['max_model_calls_per_video']==4
    assert len(created)==len(case.prepared)==len(case.closed)==1
    assert audit['videos']['X-1']['actual_calls_used']==4 and audit['videos']['X-3']['actual_calls_used']==0
    assert audit['videos']['X-1']['verifier_last_call_decisions']==[
        {'episode_id':1,'accepted':False,'reason':'primitive_evidence_not_supported'}]
    assert 'SYNTHETIC_PRIVATE_VALUE' not in json.dumps(audit)


def test_compact_probe_wrapper_construction_failure_closes_raw_worker(local_probe,monkeypatch):
    import sys
    from types import ModuleType

    case=local_probe
    module=ModuleType('ml.vlm.compact_verified_action_backend')
    module.DECODER_POLICY='scripted_compact_blind_policy'
    def fail(*args,**kwargs):
        raise RuntimeError('scripted compact construction failure')
    module.CompactVerifiedActionBackend=fail
    monkeypatch.setitem(sys.modules,module.__name__,module)
    case.args.compact_verify_actions=True
    with pytest.raises(RuntimeError,match='scripted compact construction failure'):
        case.module.run(case.args)
    assert case.closed==[True] and not case.prepared and not case.calls


@pytest.mark.parametrize('invalid',[True,False,None,'0.95',float('nan'),float('inf'),float('-inf'),-.01,1.01,10**1000])
def test_probe_owner_confidence_rejects_invalid_values_before_startup(local_probe,invalid):
    case=local_probe
    case.args.min_confidence=invalid
    with pytest.raises(ValueError,match='min_confidence_must_be_finite_between_zero_and_one'):
        case.module.run(case.args)
    assert not case.prepared and not case.calls and not case.closed


def test_probe_owner_confidence_default_and_explicit_gate_are_fingerprinted_and_forwarded(local_probe,monkeypatch):
    import json

    case=local_probe
    seen=[]
    original=case.discovery.discover_types
    def discover(*args,**kwargs):
        seen.append(kwargs['min_confidence'])
        return original(*args,**kwargs)
    monkeypatch.setattr(case.discovery,'discover_types',discover)
    case.module.run(case.args)
    first=json.loads(case.args.out.with_suffix('.audit.json').read_text())
    assert first['config']['min_confidence']==.7 and seen==[.7]
    case.args.out=case.root/'var/explicit-gate.json'
    case.args.min_confidence=.95
    results=case.module.run(case.args)
    second=json.loads(case.args.out.with_suffix('.audit.json').read_text())
    assert second['config']['min_confidence']==.95 and seen==[.7,.95]
    assert first['config_sha256']!=second['config_sha256']
    assert all(all(event in results[key]['violations'] for event in case.baseline[key]['violations']) for key in results)


def test_cli_owner_confidence_is_explicit_with_unchanged_default(monkeypatch):
    import sys
    from ml.tools import vlm_type_probe as module

    seen=[]
    monkeypatch.setattr(module,'run',lambda args: seen.append(args.min_confidence) or {})
    argv=['probe','--model','scripted','--python','scripted','--out','var/scripted.json']
    monkeypatch.setattr(sys,'argv',argv)
    assert module.main()==0 and seen==[.7]
    monkeypatch.setattr(sys,'argv',argv+['--min-confidence','.95'])
    assert module.main()==0 and seen==[.7,.95]
