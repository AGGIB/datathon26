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
    assert audit['config']['sampling_policy'] == 'uniform_halves_12_v1'
    assert audit['config']['model_artifact_sha256']['preprocessor_config.json']
    assert audit['config']['source_sha256']['ml/fixture.py']
    assert audit['config']['runtime']['packages']['mlx'] == '0'
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
