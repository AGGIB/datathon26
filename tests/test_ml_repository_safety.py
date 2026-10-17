"""Repository safety rules; synthetic path names only, no credentials read."""
from configparser import ConfigParser
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('path', [
    'data/private.csv', 'ml/data/private.csv', 'backend/data/private.csv',
    'private.MoV', 'private.Mp4', 'private.AvI', 'private.MkV',
    'private.WeBm', 'private.M4v', 'private.3Gp', 'private.MtS',
    'frames/private.JpG', 'frames/private.JpEg', 'frames/private.PnG',
    'local.pem', 'local.key', 'id_rsa.backup', 'credentials-local.json', 'local.p12',
    'models/local.bin', 'weights/local.ckpt', 'artifacts/private.json',
    'reports/private.csv', 'open_labels.xlsx', '.env.local',
])
def test_private_paths_are_ignored(path):
    result = subprocess.run(
        ['git', 'check-ignore', '--no-index', '-z', '--stdin'], cwd=ROOT,
        input=path + '\0', text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, path


def test_pipeline_slow_marker_and_all_owned_tests_are_discovered():
    config = ConfigParser()
    config.read(ROOT / 'pytest.ini', encoding='utf-8')
    assert set(config['pytest']['testpaths'].split()) == {'backend/tests', 'ml/tests', 'tests'}
    assert any(line.strip().startswith('slow:') for line in config['pytest'].get('markers', '').splitlines())


def test_placeholder_exception_does_not_unignore_other_frames():
    result = subprocess.run(
        ['git', 'check-ignore', '--no-index', '-z', '--stdin'], cwd=ROOT,
        input='backend/app/fixtures/placeholder.jpg\0', text=True, capture_output=True, check=False,
    )
    assert result.returncode == 1
