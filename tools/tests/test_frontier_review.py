"""Synthetic files and human metadata only; never private data or media decoding."""
import copy
import hashlib
import json
import subprocess
import sys
from uuid import uuid4

import pytest

from tools import frontier_review as cli
from tools.frontier_review_schema import FRONTIER, draft_rows, validate_documents

REVISION = 'a' * 40


@pytest.fixture
def case(tmp_path, monkeypatch):
    videos = tmp_path / 'videos'
    videos.mkdir()
    groups = {}
    for name, data in [('A.mov', b'synthetic-content-one'), ('B.mov', b'synthetic-content-one'),
                       ('C.mp4', b'synthetic-content-two')]:
        (videos / name).write_bytes(data)
        groups.setdefault(hashlib.sha256(data).hexdigest(), []).append(name)
    report = tmp_path / 'hashes.json'
    report.write_text(json.dumps({'content_aggregation': {'hash_source': 'actual',
        'hashes_authoritatively_verified': True}, 'per_content_group': [
            {'content_sha256': digest, 'videos': aliases} for digest, aliases in groups.items()]}))
    monkeypatch.setattr(cli.shutil, 'which', lambda value: 'synthetic-ffprobe-never-executed')
    monkeypatch.setattr(cli, '_probe', lambda *args: {'status': 'ok', 'duration_sec': 20.})
    manifest = cli.build_manifest(videos, report, REVISION)
    a, b = draft_rows(manifest), draft_rows(manifest)
    adj = [{'slot_id': row['slot_id'], 'consensus': 'pending', 'reviewer': '', 'date': '',
            'disagreement_code': ''} for row in a]
    return {'root': tmp_path, 'videos': videos, 'report': report, 'manifest': manifest,
            'a': a, 'b': b, 'ea': [], 'eb': [], 'adj': adj}


def check(case):
    return validate_documents(case['manifest'], case['a'], case['ea'], case['b'], case['eb'], case['adj'])


def reviewed(row, who='Human reviewer A', **changes):
    row.update(review_status='reviewed', reviewer=who, date='2026-10-18', coverage='full_clip',
               reviewed_start_s='0', reviewed_end_s='20', type_exhaustive='true', assessment='no_observed_positive')
    row.update(changes)


def episode(row, outcome='observed_positive', **changes):
    value = {'episode_id': str(uuid4()), 'slot_id': row['slot_id'], 'outcome': outcome,
        't_start': 4, 't_peak': 5, 't_end': 6, 'actor_role': 'cashier', 'actor_track_id': 'actor-a',
        'object_track_id': 'object-a', 'object_visibility': 'clear', 'contact_visibility': 'clear',
        'departure_visibility': 'clear', 'same_actor': 'yes', 'same_object': 'yes',
        'unknown_reason': '', 'contrast_reason': ''}
    value.update(changes)
    return value


def test_exact_hash_dedup_three_actions_and_nothing_inferred(case):
    result = check(case)
    assert result['files'] == 3 and result['contents'] == 2 and result['coverage_slots'] == 6
    assert result['required_independent_review_slots'] == 12
    assert result['review_a_completed_slots'] == result['review_b_completed_slots'] == result['episode_records'] == 0
    assert not result['human_handoff_ready'] and not result['precision_measured'] and not result['evaluation_truth_certified']
    assert set(row['type_id'] for row in case['a']) == set(FRONTIER)
    assert all(all(source[key] is None for key in ('camera_id', 'angle_id', 'session_id', 'source_group'))
               for source in case['manifest']['contents'])


def test_init_publishes_private_pending_bundle_and_never_overwrites(case):
    output = case['root'] / 'var' / 'review'
    summary = cli.initialize(output, case['videos'], case['report'], REVISION, artifact_root=output.parent)
    assert summary['coverage_slots'] == 6 and not summary['human_handoff_ready']
    assert (output / 'reviewer-a/episodes.jsonl').read_bytes() == b''
    assert (output / 'reviewer-b/episodes.jsonl').read_bytes() == b''
    assert (output / 'manifest.json').stat().st_mode & 0o777 == 0o600
    before = {path: path.read_bytes() for path in output.rglob('*') if path.is_file()}
    with pytest.raises(ValueError, match='FRESH_PRIVATE_OUTPUT'):
        cli.initialize(output, case['videos'], case['report'], REVISION, artifact_root=output.parent)
    assert all(path.read_bytes() == content for path, content in before.items())
    with pytest.raises(ValueError, match='HUMAN_REVIEW_INCOMPLETE'):
        cli.validate_files(output, require_complete=True)


@pytest.mark.parametrize('damage', ['hash', 'source', 'nonactual', 'duplicate_alias'])
def test_init_rejects_wrong_or_nonactual_members_before_output(case, damage):
    data = json.loads(case['report'].read_text())
    if damage == 'hash':
        data['per_content_group'][0]['content_sha256'] = 'f' * 64
    elif damage == 'source':
        (case['videos'] / 'A.mov').write_bytes(b'changed source')
    elif damage == 'nonactual':
        data['content_aggregation']['hash_source'] = 'declared'
    else:
        data['per_content_group'][0]['videos'].append('A.mov')
    case['report'].write_text(json.dumps(data))
    output = case['root'] / 'var' / 'review'
    with pytest.raises(ValueError):
        cli.initialize(output, case['videos'], case['report'], REVISION, artifact_root=output.parent)
    assert not output.exists()


@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'wrong_type', 'false_coverage', 'fake_signature'])
def test_slot_completeness_and_pending_cannot_claim_human_work(case, damage):
    if damage == 'missing':
        case['a'].pop()
    elif damage == 'duplicate':
        case['a'][-1] = copy.deepcopy(case['a'][0])
    elif damage == 'wrong_type':
        case['a'][0]['type_id'] = 'safe_left_open'
    elif damage == 'false_coverage':
        case['a'][0]['type_exhaustive'] = 'true'
    else:
        case['a'][0]['reviewer'] = 'Human A'
    with pytest.raises(ValueError):
        check(case)


def test_empty_events_never_become_negative_without_explicit_exhaustive_review(case):
    reviewed(case['a'][0], type_exhaustive='false')
    with pytest.raises(ValueError, match='ABSENCE_REQUIRES_EXHAUSTIVE'):
        check(case)
    case['a'][0]['type_exhaustive'] = 'true'
    assert check(case)['review_a_completed_slots'] == 1
    assert not check(case)['human_handoff_ready']


@pytest.mark.parametrize('who', ['Human reviewer A', '  HUMAN  REVIEWER A  ', 'Qwen', 'Codex Agent'])
def test_second_reviewer_distinction_and_human_guard(case, who):
    reviewed(case['a'][0])
    reviewed(case['b'][0], who)
    with pytest.raises(ValueError):
        check(case)


def test_adjudication_requires_both_and_real_signature(case):
    case['adj'][0].update(consensus='no_observed_positive', reviewer='Human adjudicator', date='2026-10-18',
                          disagreement_code='none')
    with pytest.raises(ValueError, match='TWO_REVIEWS_REQUIRED'):
        check(case)
    reviewed(case['a'][0])
    reviewed(case['b'][0], 'Human reviewer B')
    assert check(case)['adjudicated_slots'] == 1


@pytest.mark.parametrize('damage', ['outside', 'reversed', 'nan', 'missing', 'unknown_role', 'unclear_object', 'no_identity'])
def test_episode_bounds_and_positive_fact_consistency(case, damage):
    reviewed(case['a'][0], assessment='events_recorded')
    value = episode(case['a'][0])
    if damage == 'outside':
        value['t_peak'] = 21
    if damage == 'reversed':
        value['t_peak'] = 3
    if damage == 'nan':
        value['t_peak'] = float('nan')
    if damage == 'missing':
        value['t_peak'] = None
    if damage == 'unknown_role':
        value['actor_role'] = 'unknown'
    if damage == 'unclear_object':
        value['object_visibility'] = 'unknown'
    if damage == 'no_identity':
        value['same_object'] = 'unknown'
    case['ea'] = [value]
    with pytest.raises(ValueError):
        check(case)


def test_consumption_single_peak_allowed_and_unknown_is_not_pending_or_negative(case):
    row = next(row for row in case['a'] if row['type_id'] == 'food_or_drink_at_workplace')
    reviewed(row, assessment='events_recorded')
    case['ea'] = [episode(row, t_start=5, t_peak=5, t_end=5, same_actor='unknown', same_object='unknown')]
    assert check(case)['episode_records'] == 1
    row = case['a'][-1]
    reviewed(row, assessment='unknown', type_exhaustive='false', unknown_reason='object_unresolved')
    assert check(case)['review_a_completed_slots'] == 2


def test_disagreement_is_preserved_and_unresolved_dispute_blocks_ready(case):
    for a, b, adj in zip(case['a'], case['b'], case['adj'], strict=True):
        reviewed(a)
        reviewed(b, 'Human reviewer B')
        adj.update(consensus='no_observed_positive', reviewer='Human adjudicator', date='2026-10-18', disagreement_code='none')
    assert check(case)['human_handoff_ready']
    case['adj'][0].update(consensus='disputed', disagreement_code='unresolved')
    assert not check(case)['human_handoff_ready']
    case['b'][0].update(assessment='unknown', type_exhaustive='false', unknown_reason='object_unresolved')
    case['adj'][0].update(consensus='unknown', disagreement_code='none')
    with pytest.raises(ValueError, match='DISAGREEMENT_MUST_BE_RECORDED'):
        check(case)


def test_manifest_binding_sources_and_csv_schema_checked_read_only(case):
    output = case['root'] / 'var' / 'review'
    cli.initialize(output, case['videos'], case['report'], REVISION, artifact_root=output.parent)
    path = output / 'manifest.json'
    path.write_text(path.read_text().replace('20.0', '21.0'))
    with pytest.raises(ValueError, match='MANIFEST_BINDING_CHANGED'):
        cli.validate_files(output)


def test_cli_failure_numeric_only_and_import_has_no_model_decode_dependencies(case, capsys):
    assert cli.main(['validate', '--bundle', str(case['root'] / 'missing')]) == 1
    assert 'invalid_or_incomplete_private_review' in capsys.readouterr().out
    code = "import sys; import tools.frontier_review; assert not {'torch','mlx','mlx_vlm','transformers','cv2','PIL','numpy'} & set(sys.modules)"
    subprocess.run([sys.executable, '-c', code], check=True, timeout=5)


def all_reviewed(case):
    for a, b, adj in zip(case['a'], case['b'], case['adj'], strict=True):
        reviewed(a)
        reviewed(b, 'Human reviewer B')
        adj.update(consensus='no_observed_positive', reviewer='Human adjudicator', date='2026-10-18',
                   disagreement_code='none')


def test_disjoint_reviewed_intervals_cannot_complete_handoff_even_with_empty_episodes(case):
    all_reviewed(case)
    for a, b in zip(case['a'], case['b'], strict=True):
        a.update(coverage='partial', reviewed_start_s='0', reviewed_end_s='5')
        b.update(coverage='partial', reviewed_start_s='10', reviewed_end_s='20')
    result = check(case)
    assert result['adjudicated_slots'] == result['coverage_slots']
    assert not result['human_handoff_ready']
    for a, b, adj in zip(case['a'], case['b'], case['adj'], strict=True):
        a.update(assessment='unknown', type_exhaustive='false', unknown_reason='object_unresolved')
        b.update(assessment='unknown', type_exhaustive='false', unknown_reason='object_unresolved')
        adj.update(consensus='unknown')
    assert not check(case)['human_handoff_ready']


def test_resolved_absence_cannot_be_derived_from_nonexhaustive_unknown_review(case):
    all_reviewed(case)
    case['b'][0].update(assessment='unknown', type_exhaustive='false', unknown_reason='object_unresolved')
    case['adj'][0]['disagreement_code'] = 'resolved_after_review'
    with pytest.raises(ValueError, match='CONSENSUS_ABSENCE_REQUIRES_BOTH_EXHAUSTIVE'):
        check(case)


def test_final_episodes_preserve_original_links_and_both_reviewed_bounds(case):
    a, b, adj = case['a'][0], case['b'][0], case['adj'][0]
    reviewed(a, assessment='events_recorded', coverage='partial', reviewed_start_s='2', reviewed_end_s='10')
    reviewed(b, 'Human reviewer B', assessment='events_recorded', coverage='partial', reviewed_start_s='4')
    case['ea'], case['eb'] = [episode(a)], [episode(b)]
    adj.update(consensus='events_recorded', reviewer='Human adjudicator', date='2026-10-18', disagreement_code='none')
    final = {**episode(a), 'review_a_episode_ids': [case['ea'][0]['episode_id']],
             'review_b_episode_ids': [case['eb'][0]['episode_id']]}
    def run():
        return validate_documents(case['manifest'], case['a'], case['ea'], case['b'], case['eb'], case['adj'], [final])
    assert run()['adjudicated_episode_records'] == 1
    final['t_start'] = 2
    with pytest.raises(ValueError, match='EPISODE_INTERVAL'):
        run()
    final['t_start'] = 4
    final['review_a_episode_ids'] = [str(uuid4())]
    with pytest.raises(ValueError, match='FINAL_EPISODE_PROVENANCE'):
        run()


def test_manifest_pins_both_schema_and_cli_and_source_changes_fail(case):
    assert case['manifest']['schema_sha256'] == cli.sha256(cli.Path(cli.__file__).with_name('frontier_review_schema.py'))
    output = case['root'] / 'var' / 'review'
    cli.initialize(output, case['videos'], case['report'], REVISION, artifact_root=output.parent)
    (case['videos'] / 'A.mov').write_bytes(b'changed after scaffold')
    with pytest.raises(ValueError, match='SOURCE_HASH_CHANGED'):
        cli.validate_files(output)


def test_invalid_cli_arguments_never_echo_private_values(capsys):
    assert cli.main(['PRIVATE_FREE_OBSERVATION']) == 1
    captured = capsys.readouterr()
    assert 'PRIVATE_FREE_OBSERVATION' not in captured.out + captured.err
    assert captured.out.strip() == '{"status":"invalid_or_incomplete_private_review"}'
