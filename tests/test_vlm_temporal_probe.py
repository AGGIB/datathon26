"""Experimental CLI dispatch and real temporal wrapper with scripted local replies."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from ml.kb import catalog
from ml.tools import vlm_type_probe as probe
from ml.vlm import action_backend, compact_verified_action_backend, discovery, mlx_backend, verified_action_backend
from ml.vlm.contracts import Frame
from ml.vlm.temporal_witness_backend import DECODER_POLICY, TemporalWitnessBackend, _schema

COAT = 'cashier_places_removed_outerwear_on_cashdesk'


@pytest.fixture
def case(tmp_path, monkeypatch):
    from ml import kb, video
    types = catalog()
    monkeypatch.setattr(probe, 'ROOT', tmp_path)
    kb_path = tmp_path / 'kb/violations.yaml'
    kb_path.parent.mkdir()
    kb_path.write_text('synthetic catalog fingerprint only')
    monkeypatch.setattr(kb, 'KB_PATH', kb_path)
    monkeypatch.setattr(kb, 'catalog', lambda: types)
    (tmp_path / 'ml').mkdir()
    (tmp_path / 'ml/fixture.py').write_text('# synthetic source fingerprint')
    videos = tmp_path / 'videos'
    videos.mkdir()
    (videos / 'X-1.mov').write_bytes(b'SYNTHETIC_PLACEHOLDER_NEVER_DECODED')
    model = tmp_path / 'model'
    model.mkdir()
    for name, value in [('config.json', {'model_type': 'qwen3_vl'}),
                        ('preprocessor_config.json', {}), ('tokenizer_config.json', {}), ('tokenizer.json', {})]:
        (model / name).write_text(json.dumps(value))
    (model / 'model.safetensors').write_bytes(b'SYNTHETIC_WEIGHTS_NEVER_LOADED')
    baseline = tmp_path / 'baseline.json'
    cv = {'type_id': 'safe_left_open', 't': .4, 'conf': .612345}
    baseline.write_text(json.dumps({'X-1': {'violations': [cv]}}))
    state = SimpleNamespace(proposals=[], supported=True, raw_calls=[], observed=[], created=[], closed=0,
                            frame_calls=0, cv=cv, model=model)
    state.args = SimpleNamespace(model=model, python=Path(sys.executable), videos=videos, baseline=baseline,
        out=tmp_path / 'var/output.json', only='', budget=10, min_confidence=.95, max_tokens=512, frame_side=640)

    class Raw:
        def __init__(self, *args, **kwargs):
            state.created.append(kwargs)
        def validate(self, timeout_s):
            assert timeout_s > 0
        def infer(self, request, timeout_s):
            state.raw_calls.append(request)
            assert 0 < timeout_s <= state.args.budget
            fields = request.response_schema['properties']
            if 'actions' in fields:
                return json.dumps({'actions': state.proposals})
            if 'episodes' in fields:
                assert request.response_schema == _schema(12)
                assert len(request.frames) == 12
                atoms = {4: 'cashier_holds_identifiable_outerwear',
                         7: 'cashier_places_outerwear_on_own_workdesk',
                         10: 'outerwear_resting_on_own_workdesk'} if state.supported else {}
                return json.dumps({'episodes': [{'id': 1, 'rows': [
                    [i, atoms.get(i, 'ordinary_work'), 'unknown'] for i in range(1, 13)],
                    'same_actor': 'yes', 'same_object': 'yes'}]})
            return '{"findings":[]}'
        def close(self):
            state.closed += 1

    class Source:
        def extract(self, path, timestamps, directory, timeout_s):
            state.frame_calls += 1
            frames = []
            for index, t in enumerate(timestamps):
                path = directory / f'{index}.png'
                path.write_bytes(b'NOT_AN_IMAGE_NO_PIXEL_INSPECTION')
                frames.append(Frame(t, path, t))
            return tuple(frames)

    original_discover = discovery.discover_types
    def record_discover(*args, **kwargs):
        state.observed.append((kwargs['backend'], kwargs['budget']))
        return original_discover(*args, **kwargs)
    monkeypatch.setattr(discovery, 'discover_types', record_discover)
    monkeypatch.setattr(discovery, 'BatchedOwnerFrames', Source)
    monkeypatch.setattr(mlx_backend, 'LocalMLXBackend', Raw)
    monkeypatch.setattr(video, 'probe', lambda path: SimpleNamespace(duration=20.0))
    monkeypatch.setattr(probe, '_runtime_versions', lambda path: {'python': 'synthetic', 'packages': {}})
    return state


def action():
    return {'action': COAT, 'insufficient_evidence': False, 'confidence': .95,
            'start_frame': 1, 'peak_frame': 2, 'end_frame': 3}


@pytest.mark.parametrize('other_flags', [False, True])
def test_explicit_temporal_precedence_real_wrapper_budget_schema_and_relocation(case, other_flags):
    case.args.temporal_witness_actions = True
    for name in ('action_contrast', 'verify_actions', 'compact_verify_actions'):
        setattr(case.args, name, other_flags)
    case.args.max_tokens = 768  # Explicit experimental command value; default remains512.
    case.proposals = [action()]
    result = probe.run(case.args)
    backend, budget = case.observed[0]
    assert isinstance(backend, TemporalWitnessBackend)
    assert budget.max_calls == 4 and budget.calls_used == len(case.raw_calls) == 2
    assert case.created[0]['max_new_tokens'] == 768
    assert case.closed == case.frame_calls == 1
    assert case.cv in result['X-1']['violations']  # Exact numeric confidence preserved.
    expected = round(case.raw_calls[1].frames[6].timestamp_s, 1)
    assert expected != round(case.raw_calls[1].frames[1].timestamp_s, 1)
    assert {'type_id': 'personal_items_on_desk', 't': expected, 'conf': .95} in result['X-1']['violations']
    audit = json.loads(case.args.out.with_suffix('.audit.json').read_text())
    assert audit['config']['decoder_policy'] == DECODER_POLICY
    assert audit['config']['max_model_calls_per_video'] == 4
    assert audit['config']['evaluation_scope'] == 'development_only_not_new_angle'
    assert audit['videos']['X-1']['actual_calls_used'] == 2
    assert audit['videos']['X-1']['verifier_last_call_decisions'] == [{
        'episode_id': 1, 'accepted': True, 'reason': 'visible_temporal_witness_supported',
        'status': 'supported', 'witness_frame_indices': [4, 7, 10]}]


@pytest.mark.parametrize('mode,expected_calls,status,reason', [
    ('normal', 1, None, None),
    ('unsupported', 2, 'no_witness', 'primitive_evidence_not_supported'),
    ('multiple', 1, 'scope_abstention', 'multiple_proposals_not_supported'),
])
def test_temporal_abstention_diagnostics_and_raw_attempt_count(case, mode, expected_calls, status, reason):
    case.args.temporal_witness_actions = True
    case.proposals = ([{**action(), 'action': next(iter(action_backend._NORMAL))}] if mode == 'normal'
                      else [action()] * (2 if mode == 'multiple' else 1))
    case.supported = False
    result = probe.run(case.args)
    assert result['X-1']['violations'] == [case.cv]
    assert len(case.raw_calls) == case.observed[0][1].calls_used == expected_calls
    audit = json.loads(case.args.out.with_suffix('.audit.json').read_text())
    diagnostics = audit['videos']['X-1']['verifier_last_call_decisions']
    if status is None:
        assert diagnostics == []
    else:
        assert all(row['status'] == status and row['reason'] == reason
                   and row['witness_frame_indices'] == [] for row in diagnostics)


@pytest.mark.parametrize('flag,policy,cap', [
    (None, 'direct_discovery', 2),
    ('action_contrast', action_backend.DECODER_POLICY, 2),
    ('verify_actions', verified_action_backend.DECODER_POLICY, 4),
    ('compact_verify_actions', compact_verified_action_backend.DECODER_POLICY, 4),
])
def test_default_and_existing_profiles_do_not_construct_temporal_wrapper(case, monkeypatch, flag, policy, cap):
    import ml.vlm.temporal_witness_backend as temporal
    monkeypatch.setattr(temporal, 'TemporalWitnessBackend', lambda *a, **kw: pytest.fail('default-off violated'))
    if flag:
        setattr(case.args, flag, True)
    result = probe.run(case.args)
    assert result['X-1']['violations'] == [case.cv]
    audit = json.loads(case.args.out.with_suffix('.audit.json').read_text())
    assert audit['config']['decoder_policy'] == policy
    assert audit['config']['max_tokens'] == case.created[0]['max_new_tokens'] == 512
    assert case.observed[0][1].max_calls == cap and len(case.raw_calls) == 1


@pytest.mark.parametrize('value', [0, 1, 'true', None, [], {}])
def test_new_profile_rejects_nonboolean_programmatic_flags_before_backend_start(case, value):
    case.args.temporal_witness_actions = value
    with pytest.raises(ValueError, match='temporal_witness_actions_must_be_boolean'):
        probe.run(case.args)
    assert case.created == []


def test_cli_flag_defaults_and_explicit_experiment_tokens(monkeypatch):
    received = []
    monkeypatch.setattr(probe, 'run', lambda args: received.append(args) or {})
    base = ['probe', '--model', 'local-model', '--python', sys.executable, '--out', 'var/private.json']
    monkeypatch.setattr(sys, 'argv', base)
    assert probe.main() == 0
    assert received[-1].temporal_witness_actions is False and received[-1].max_tokens == 512
    monkeypatch.setattr(sys, 'argv', base + ['--temporal-witness-actions', '--compact-verify-actions', '--max-tokens', '768'])
    assert probe.main() == 0
    assert received[-1].temporal_witness_actions is True and received[-1].max_tokens == 768


@pytest.mark.parametrize('changes', [
    {'status': 'unsupported_private_text'}, {'witness_frame_indices': [True, 7, 10]},
    {'witness_frame_indices': [4, 7, 13]}, {'witness_frame_indices': [7, 4, 10]},
    {'witness_frame_indices': [4, 4, 10]}, {'witness_frame_indices': [4, 7]},
    {'witness_frame_indices': 'private'}, {'accepted': False},
    {'reason': 'primitive_evidence_not_supported'}, {'status': None},
])
def test_temporal_audit_drops_malformed_or_contradictory_optional_metadata(changes):
    row = {'episode_id': 1, 'accepted': True, 'reason': 'visible_temporal_witness_supported',
           'status': 'supported', 'witness_frame_indices': [4, 7, 10]}
    assert probe._coded_verifications([{**row, **changes}]) == []


def test_audit_legacy_shape_unchanged_and_standalone_food_witness_is_valid():
    legacy = {'episode_id': 1, 'accepted': True, 'reason': 'supported'}
    assert probe._coded_verifications([{**legacy, 'observations': 'PRIVATE_NOT_SERIALIZED'}]) == [legacy]
    food = {'episode_id': 1, 'accepted': True, 'reason': 'visible_temporal_witness_supported',
            'status': 'supported', 'witness_frame_indices': [7, 7, 7]}
    assert probe._coded_verifications([{**food, 'observations': 'PRIVATE_NOT_SERIALIZED'}]) == [food]


@pytest.mark.parametrize('flag', ['action_contrast', 'verify_actions', 'compact_verify_actions'])
def test_temporal_precedence_does_not_bypass_other_flag_type_validation(case, flag):
    case.args.temporal_witness_actions = True
    setattr(case.args, flag, 'true')
    with pytest.raises(ValueError, match=flag + '_must_be_boolean'):
        probe.run(case.args)
    assert case.created == []


@pytest.mark.parametrize('field', ['status', 'witness_frame_indices'])
def test_partial_temporal_optional_metadata_is_not_reported_as_complete(field):
    row = {'episode_id': 1, 'accepted': True, 'reason': 'visible_temporal_witness_supported',
           'status': 'supported', 'witness_frame_indices': [4, 7, 10]}
    del row[field]
    assert probe._coded_verifications([row]) == []
