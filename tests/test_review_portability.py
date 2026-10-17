"""Portable local test paths; no private inputs or model imports."""
from pathlib import Path
import runpy
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def test_baseline_scratch_uses_platform_temp_when_tmpdir_is_unset(monkeypatch, tmp_path):
    monkeypatch.delenv('TMPDIR', raising=False)
    monkeypatch.setattr(tempfile, 'gettempdir', lambda: str(tmp_path))
    namespace = runpy.run_path(str(ROOT / 'tests/test_baseline.py'), run_name='baseline_fixture')
    assert namespace['SCRATCH'] == tmp_path


def test_baseline_scratch_respects_explicit_tmpdir(monkeypatch, tmp_path):
    monkeypatch.setenv('TMPDIR', str(tmp_path))
    namespace = runpy.run_path(str(ROOT / 'tests/test_baseline.py'), run_name='baseline_fixture')
    assert namespace['SCRATCH'] == tmp_path


def test_low_level_verifier_candidate_does_not_shadow_team_candidate():
    from ml import vlm
    from ml.candidates import Candidate
    assert hasattr(vlm, 'VerifierCandidate')
    assert 'VerifierCandidate' in vlm.__all__
    assert not hasattr(vlm, 'Candidate')
    assert vlm.VerifierCandidate is not Candidate
