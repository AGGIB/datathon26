"""PR review regressions. Every byte, ID and hash here is a synthetic fixture."""
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

from ml.eval import audit, score
from ml.evaluation_labels import video_key


def digest(data):
    return hashlib.sha256(data).hexdigest()


def label(video, positive=True, timestamp=10, **extra):
    return dict(video=video, positive=positive, timestamp=timestamp, **extra)


def prediction(video, peaks=(10,), positive=True, **extra):
    return dict(video=video, status='ok', verdict='violation' if positive else 'no_violation',
                violations=[{'t_peak': peak, 'type_id': 'fixture'} for peak in peaks], **extra)


def test_team_latin_ids_join_cyrillic_labels_without_rewriting_provenance():
    labels = [label('Х-3.mov')]
    report = score(labels, [prediction('X-3')], annotations=[
        {'video': 'X-3', 't_peak': 10, 'type_id': 'fixture'},
        {'video': 'X-3', 't_peak': 20, 'type_id': 'fixture'}])
    assert video_key('Х-3.mov') == video_key('X-3') == 'X-3'
    assert report['overall']['confusion']['tp'] == 1
    assert report['per_video'][0]['video'] == 'Х-3.mov'
    assert report['overall']['localization']['truth_events'] == 2


@pytest.mark.parametrize('source', ['labels', 'annotations', 'predictions', 'split', 'files'])
def test_canonical_collisions_are_rejected_per_source(tmp_path, source):
    labels = [label('Х-3')]
    kwargs = {}
    predictions = []
    if source == 'labels':
        labels.append(label('X-3'))
    elif source == 'annotations':
        kwargs['annotations'] = [{'video': 'Х-3', 't_peak': 10}, {'video': 'X-3', 't_peak': 20}]
    elif source == 'predictions':
        predictions = [prediction('Х-3'), prediction('X-3')]
    elif source == 'split':
        kwargs['split'] = [dict(video=name, subset=side, angle_id=side, content_sha256=digest(side.encode()))
                           for name, side in [('Х-3', 'train'), ('X-3', 'test')]]
    if source == 'files':
        for name in ('Х-3.mov', 'X-3.mov'):
            (tmp_path / name).write_bytes(b'synthetic')
        with pytest.raises(ValueError, match='collision'):
            audit(labels, tmp_path, use_ffprobe=False)
    else:
        with pytest.raises(ValueError):
            score(labels, predictions, **kwargs)


@pytest.mark.parametrize('raw', [' X-3', 'X-3 ', 'X -3', '../X-3', r'folder\X-3',
                                 'C:X-3', '.', '..', 'X-3\x00', 'X-3\x01'])
def test_malformed_identifiers_rejected_before_team_normalization(raw):
    with pytest.raises(ValueError):
        video_key(raw)


@pytest.mark.parametrize('raw', ['fixture.', 'fixture.mov.', 'CON', 'CON.mov', 'aux.txt',
                                 'COM1.mp4', 'LPT9', 'COM¹.mov'])
def test_windows_literal_aliases_are_rejected_before_normalization(monkeypatch, raw):
    def forbidden(*args):
        pytest.fail('malformed Windows literal reached team normalization')
    monkeypatch.setattr('ml.evaluation_labels.normalize_video_name', forbidden)
    with pytest.raises(ValueError, match='identifier'):
        video_key(raw)


@pytest.mark.parametrize('source', ['labels', 'predictions', 'annotations', 'split', 'content_hashes'])
def test_trailing_dot_cannot_join_a_valid_identifier_across_sources(source):
    labels = [label('fixture'), label('other')]
    predictions = []
    kwargs = {}
    if source == 'labels':
        labels[0]['video'] = 'fixture.'
        predictions = [prediction('fixture')]
    elif source == 'predictions':
        predictions = [prediction('fixture.')]
    elif source == 'annotations':
        kwargs['annotations'] = [dict(video='fixture.', t_peak=10, type_id='fixture')]
    elif source == 'content_hashes':
        kwargs['content_hashes'] = {'fixture.': digest(b'first'), 'other': digest(b'second')}
    else:
        kwargs['split'] = [dict(video=name, subset=side, angle_id=side,
                                content_sha256=digest(side.encode()))
                           for name, side in [('fixture.', 'train'), ('other', 'test')]]
    with pytest.raises(ValueError, match='identifier'):
        score(labels, predictions, **kwargs)


def test_assignment_maximizes_inclusive_tolerance_matches_before_error():
    report = score([label('fixture')], [prediction('fixture', (21, 25))], annotations=[
        {'video': 'fixture', 't_peak': t, 'type_id': 'fixture'} for t in (10, 20)], tolerance_sec=2)
    loc = report['overall']['localization']
    assert loc['within_tolerance'] == loc['matched_events'] == 1
    assert loc['unmatched_truth_events'] == loc['unmatched_predicted_events'] == 1
    assert loc['mae_sec'] == 1
    assert report['per_video'][0]['matched_events'][0]['truth_t_peak'] == 20
    assert score([label('boundary')], [prediction('boundary', (12,))])['overall']['localization']['within_tolerance'] == 1


def test_audit_hashes_actual_bytes_reports_duplicate_anchor_alternatives(tmp_path):
    payload = b'ARTIFICIAL CONTENT DUPLICATE, NOT MEDIA'
    (tmp_path / 'nested').mkdir()
    (tmp_path / 'nested' / 'Х-3.mov').write_bytes(payload)
    (tmp_path / 'X-4.mov').write_bytes(payload)
    report = audit([label('X-3', timestamp=10), label('Х-4', timestamp=20),
                    label('X-5', positive=False, timestamp=None)], tmp_path, use_ffprobe=False)
    assert [row['content_sha256'] for row in report['per_video']] == [digest(payload), digest(payload), None]
    assert report['per_video'][0]['matched_paths'] == [str(__import__('pathlib').Path('nested') / 'Х-3.mov')]
    assert report['per_video'][0]['video'] == 'X-3'
    group, = report['duplicate_groups']
    assert group['videos'] == ['X-3', 'Х-4']
    assert group['content_sha256'] == digest(payload)
    assert group['alternative_anchors_sec'] == [10, 20]
    assert group['conflicting_anchors'] is True
    assert group['conflicting_labels'] is False
    assert report['overall']['content_groups'] == 1
    assert report['overall']['content_hashes_complete'] is False


def synthetic_link_guard(monkeypatch, linked, kind):
    """Inject link metadata on local fixtures; never construct/contact a remote target."""
    mode = linked.lstat().st_mode
    metadata = SimpleNamespace(st_mode=stat.S_IFLNK if kind == 'symlink' else mode,
                               st_file_attributes=0x400 if kind == 'reparse' else 0)
    original_stat, original_scandir = Path.stat, os.scandir
    checked = []

    def no_target_io(*args, **kwargs):
        pytest.fail('link target was followed before no-follow validation')

    def path_stat(path, *, follow_symlinks=True):
        if path == linked:
            if follow_symlinks:
                no_target_io()
            checked.append(path)
            return metadata
        if linked in path.parents:
            no_target_io()
        return original_stat(path, follow_symlinks=follow_symlinks)

    class LinkedEntry:
        name = linked.name
        path = str(linked)

        def stat(self, *, follow_symlinks=True):
            return path_stat(linked, follow_symlinks=follow_symlinks)

        def is_dir(self, *, follow_symlinks=True):
            return stat.S_ISDIR(self.stat(follow_symlinks=follow_symlinks).st_mode)

        def is_file(self, *, follow_symlinks=True):
            return stat.S_ISREG(self.stat(follow_symlinks=follow_symlinks).st_mode)

    @contextmanager
    def scandir(path):
        path = Path(path)
        if path == linked or linked in path.parents:
            no_target_io()
        with original_scandir(path) as entries:
            yield iter([LinkedEntry() if Path(entry.path) == linked else entry for entry in entries])

    monkeypatch.setattr(Path, 'stat', path_stat)
    monkeypatch.setattr(Path, 'resolve', no_target_io)
    monkeypatch.setattr(os, 'scandir', scandir)
    return checked


def link_fixture(tmp_path, location):
    directory = tmp_path / 'videos'
    directory.mkdir()
    if location == 'file':
        (directory / 'nested').mkdir()
        linked = directory / 'nested' / 'fixture.mov'
        linked.write_bytes(b'SYNTHETIC, NOT MEDIA')
    elif location == 'directory':
        linked = directory / 'nested' / 'linked'
        linked.mkdir(parents=True)
        (linked / 'fixture.mov').write_bytes(b'SYNTHETIC, NOT MEDIA')
    else:
        linked = directory
        if location == 'ancestor':
            directory = linked / 'inner'
            directory.mkdir()
    return directory, linked


@pytest.mark.parametrize('kind', ['symlink', 'reparse'])
@pytest.mark.parametrize('location', ['file', 'directory', 'root', 'ancestor'])
def test_audit_rejects_links_before_following_target_io(tmp_path, monkeypatch, kind, location):
    import ml.eval as evaluator
    directory, linked = link_fixture(tmp_path, location)
    checked = synthetic_link_guard(monkeypatch, linked, kind)

    def forbidden(*args, **kwargs):
        pytest.fail('ffprobe/hash accessed a link before audit rejected it')

    monkeypatch.setattr(evaluator, '_probe', forbidden)
    monkeypatch.setattr(evaluator, 'file_sha256', forbidden)
    with pytest.raises(ValueError, match='link|reparse'):
        audit([label('fixture')], directory)
    assert checked


def split_fixture():
    labels = [label('Х-3'), label('Х-4'), label('Х-5')]
    rows = [dict(video=f'X-{n}', subset=side, angle_id=f'camera_{side}',
                 content_sha256=digest(side.encode()))
            for n, side in [(3, 'train'), (4, 'validation'), (5, 'test')]]
    return labels, rows


@pytest.mark.parametrize('bad', [None, '', 'a' * 63, 'A' * 64, 'g' * 64, ' ' + 'a' * 64, 123])
def test_split_requires_complete_valid_lowercase_hashes(bad):
    labels = [label('train'), label('test')]
    rows = [dict(video=name, subset=name, angle_id=name, content_sha256=digest(name.encode()))
            for name in ('train', 'test')]
    if bad is None:
        del rows[0]['content_sha256']
    else:
        rows[0]['content_sha256'] = bad
    with pytest.raises(ValueError, match='content_sha256'):
        score(labels, [], split=rows)


def test_duplicate_content_cannot_cross_sides_even_with_different_cameras():
    labels = [label('train'), label('test')]
    rows = [dict(video=name, subset=name, angle_id=f'camera_{name}', content_sha256=digest(b'same content'))
            for name in ('train', 'test')]
    with pytest.raises(ValueError, match='content'):
        score(labels, [], split=rows)


def test_train_validation_test_split_and_canonical_ids():
    labels, rows = split_fixture()
    report = score(labels, [], split=rows, subset='validation')
    assert report['per_video'][0]['video'] == 'Х-4'
    assert report['split']['content_holdout_validated'] is True
    assert report['split']['calibration_partition_available'] is True
    assert report['split']['hashes_authoritatively_verified'] is False
    legacy = [rows[0], dict(rows[1], subset='train', angle_id='camera_train'), rows[2]]
    assert score(labels, [], split=legacy)['split']['calibration_partition_available'] is False


@pytest.mark.parametrize('source', ['split', 'labels'])
def test_authoritative_hashes_reject_conflicting_supplied_values(source):
    labels, rows = split_fixture()
    actual = {row['video']: row['content_sha256'] for row in rows}
    if source == 'split':
        rows[0]['content_sha256'] = digest(b'forged split hash')
    else:
        labels[0]['content_sha256'] = digest(b'forged label hash')
    with pytest.raises(ValueError, match='content_sha256.*(disagrees|conflict|mismatch)'):
        score(labels, [], split=rows, content_hashes=actual)


@pytest.mark.parametrize('peak', [10, 20, 12, 22, 15])
@pytest.mark.parametrize('hash_source', ['actual', 'labels'])
def test_duplicate_binary_anchors_are_one_truth_satisfied_by_either_anchor(peak, hash_source):
    labels = [label('first', timestamp=10), label('second', timestamp=20)]
    hashes = {row['video']: digest(b'duplicate') for row in labels}
    kwargs = {'content_hashes': hashes}
    if hash_source == 'labels':
        labels = [dict(row, content_sha256=hashes[row['video']]) for row in labels]
        kwargs = {}
    report = score(labels, [prediction(row['video'], (peak,)) for row in labels], **kwargs)
    assert len(report['per_video']) == 2
    assert report['video_only_overall']['total'] == 2
    assert report['overall']['total'] == report['overall']['confusion']['tp'] == 1
    assert report['overall']['aggregation_unit'] == 'content_group'
    assert report['content_aggregation']['available'] is True
    group, = report['per_content_group']
    assert group['alternative_anchors_sec'] == [10, 20]
    loc = report['overall']['localization']
    assert loc['truth_events'] == loc['predicted_events'] == 1
    assert loc['matched_events'] == (0 if peak == 15 else 1)
    assert loc['unmatched_truth_events'] == (1 if peak == 15 else 0)


def test_content_quality_counts_40_groups_but_retains_42_video_diagnostics():
    labels = [label(f'fixture-{i}', positive=i % 2 == 0, timestamp=10 if i % 2 == 0 else None)
              for i in range(42)]
    hashes = {row['video']: digest(str(i if i < 40 else i - 40).encode())
              for i, row in enumerate(labels)}
    predictions = [prediction(row['video'], (10,) if row['positive'] else (), row['positive'], elapsed_sec=1)
                   for row in labels]
    report = score(labels, predictions, content_hashes=hashes)
    assert len(report['per_video']) == report['video_only_overall']['total'] == 42
    assert len(report['per_content_group']) == report['overall']['total'] == 40
    assert report['overall']['label_counts'] == {'positive': 20, 'negative': 20}
    assert report['overall']['sla']['total_denominator'] == 40
    assert report['overall']['localization']['truth_events'] == 20
    assert report['overall']['end_to_end']['accuracy'] == 1


@pytest.mark.parametrize('order', [False, True])
@pytest.mark.parametrize('bad_status', ['missing', 'invalid', 'failed', 'timed_out'])
def test_duplicate_groups_never_cherry_pick_success_over_missing_or_failed_member(order, bad_status):
    labels = [label('first'), label('second')]
    if order:
        labels.reverse()
    predictions = [prediction('first', elapsed_sec=1)]
    if bad_status != 'missing':
        predictions.append(dict(video='second', status=bad_status, elapsed_sec=2))
    report = score(labels, predictions, content_hashes={row['video']: digest(b'same') for row in labels})
    assert report['video_only_overall']['scored'] == 1
    assert report['overall']['scored'] == 0
    assert report['overall']['coverage'] == report['overall']['end_to_end']['recall'] == 0
    assert report['overall']['status_counts'] == {bad_status: 1}
    assert report['overall']['confusion']['abstain_positive'] == 1
    assert report['overall']['localization']['matched_events'] == 0
    assert report['overall']['sla']['within_300_sec'] == 0


@pytest.mark.parametrize('conflict', ['label', 'verdict', 'events', 'annotation'])
def test_exact_duplicate_label_and_successful_prediction_conflicts_are_rejected(conflict):
    labels = [label('first'), label('second')]
    predictions = [prediction(row['video']) for row in labels]
    annotations = None
    if conflict == 'label':
        labels[1] = label('second', positive=False, timestamp=None)
    elif conflict == 'verdict':
        predictions[1] = prediction('second', (), False)
    elif conflict == 'events':
        predictions[1] = prediction('second', (20,))
    else:
        annotations = [{'video': row['video'], 't_peak': 10 + i, 'type_id': 'fixture'}
                       for i, row in enumerate(labels)]
    with pytest.raises(ValueError, match='conflicting.*(label|prediction|annotation)'):
        score(labels, predictions, annotations=annotations,
              content_hashes={row['video']: digest(b'same') for row in labels})


def test_absent_content_hashes_are_explicitly_video_only_not_deduped_quality():
    report = score([label('first'), label('second')], [prediction('first'), prediction('second')])
    assert report['overall']['aggregation_unit'] == 'video_diagnostic'
    assert report['content_aggregation']['available'] is False
    assert report['content_aggregation']['hashes_authoritatively_verified'] is False
    assert report['content_aggregation']['reason'] == 'complete_content_hashes_not_supplied'
    assert report['per_content_group'] == []


@pytest.mark.parametrize('sides', [('train', 'validation'), ('validation', 'test'), ('train', 'test')])
@pytest.mark.parametrize('leak', ['content', 'angle'])
def test_all_partition_pairs_are_isolated_before_subset_filtering(sides, leak):
    labels, rows = split_fixture()
    first, second = [next(row for row in rows if row['subset'] == side) for side in sides]
    field = 'content_sha256' if leak == 'content' else 'angle_id'
    second[field] = first[field]
    with pytest.raises(ValueError, match=leak):
        score(labels, [], split=rows, subset='train')


@pytest.mark.parametrize('elapsed', [(1, 301), (1, None), (300, 1)])
def test_duplicate_sla_uses_worst_member_not_fastest(elapsed):
    labels = [label('first'), label('second')]
    report = score(labels, [prediction(row['video'], elapsed_sec=value) for row, value in zip(labels, elapsed)],
                   content_hashes={row['video']: digest(b'same') for row in labels})
    sla = report['overall']['sla']
    assert sla['within_300_sec'] == (1 if elapsed == (300, 1) else 0)
    assert sla['unknown_elapsed'] == (1 if elapsed[1] is None else 0)


@pytest.mark.parametrize('layout', ['rows', 'test', 'holdout'])
def test_authoritative_hashes_verify_full_partition_and_detect_forgery(layout):
    labels, rows = split_fixture()
    actual = {row['video']: row['content_sha256'] for row in rows}
    split = rows if layout == 'rows' else {
        ('test' if side == 'test' else side) if layout == 'test' else ('holdout' if side == 'test' else side):
        [row for row in rows if row['subset'] == side] for side in ('train', 'validation', 'test')}
    report = score(labels, [], split=split, subset='validation', content_hashes=actual)
    assert report['split']['hashes_authoritatively_verified'] is True
    assert report['content_aggregation']['hashes_authoritatively_verified'] is True
    rows[0]['content_sha256'] = digest(b'forged unselected hash')
    with pytest.raises(ValueError, match='content_sha256.*disagrees'):
        score(labels, [], split=split, subset='test', content_hashes=actual)


def test_label_hash_column_is_preserved_and_validated(tmp_path):
    from ml.evaluation_labels import load_labels
    path = tmp_path / 'fixture.csv'
    sha = digest(b'fixture')
    path.write_text(f'video,timestamp,label,content_sha256\nfixture,10,нарушение,{sha}\n', encoding='utf-8')
    assert load_labels(path)[0]['content_sha256'] == sha
    path.write_text('video,timestamp,label,content_sha256\nfixture,10,нарушение,BAD\n', encoding='utf-8')
    with pytest.raises(ValueError, match='content_sha256'):
        load_labels(path)


@pytest.mark.parametrize('key_form', ['literal', 'canonical'])
def test_actual_hash_map_preserves_multidot_video_identity(key_form):
    labels = [label('fixture.part.mov')]
    literal = labels[0]['video']
    key = literal if key_form == 'literal' else video_key(literal)
    assert video_key(literal) == 'FIXTURE.PART'
    report = score(labels, [prediction(literal)], content_hashes={key: digest(b'fixture')})
    assert report['content_aggregation']['hashes_authoritatively_verified'] is True
    assert report['per_content_group'][0]['content_sha256'] == digest(b'fixture')
    assert report['per_video'][0]['video'] == literal


@pytest.mark.parametrize('reverse', [False, True])
def test_actual_hash_map_rejects_ambiguous_literal_or_canonical_identity(reverse):
    labels = [label('fixture.part.mov'), label('fixture.mov')]
    hashes = {'FIXTURE.PART': digest(b'part'), 'fixture.mov': digest(b'base')}
    if reverse:
        hashes = dict(reversed(list(hashes.items())))
    with pytest.raises(ValueError, match='ambiguous.*content_hashes'):
        score(labels, [], content_hashes=hashes)


@pytest.mark.parametrize('token', ['FIXTURE.PART.', ' FIXTURE.PART', '../FIXTURE.PART',
                                  'CON.mov', 123, None])
def test_actual_hash_map_validates_tokens_before_identity_lookup(token):
    with pytest.raises(ValueError, match='identifier'):
        score([label('fixture.part.mov')], [], content_hashes={token: digest(b'fixture')})


def test_known_canonical_hash_key_must_still_have_valid_literal_syntax():
    labels = [label('fixture..mov')]
    canonical = video_key(labels[0]['video'])
    assert canonical == 'FIXTURE.'
    with pytest.raises(ValueError, match='identifier'):
        score(labels, [], content_hashes={canonical: digest(b'fixture')})


def test_actual_hash_map_rejects_literal_and_canonical_alias_collision():
    sha = digest(b'fixture')
    with pytest.raises(ValueError, match='collision.*content_hashes'):
        score([label('fixture.part.mov')], [],
              content_hashes={'fixture.part.mov': sha, 'FIXTURE.PART': sha})


def test_actual_hash_map_rejects_unknown_identity_even_with_complete_known_keys():
    sha = digest(b'fixture')
    with pytest.raises(ValueError, match='no unknown videos'):
        score([label('fixture.part.mov')], [],
              content_hashes={'FIXTURE.PART': sha, 'UNKNOWN.PART': sha})


def cli_fixture(tmp_path):
    videos = tmp_path / 'videos'
    videos.mkdir()
    labels = tmp_path / 'labels.csv'
    labels.write_text('video,timestamp,label\nХ-3,10,нарушение\nХ-4,20,нарушение\n', encoding='utf-8')
    for name in ('X-3.mov', 'X-4.mov'):
        (videos / name).write_bytes(b'NOT MEDIA, SAME ARTIFICIAL CONTENT')
    predictions = tmp_path / 'predictions.json'
    predictions.write_text(json.dumps([prediction('X-3', (20,)), prediction('X-4', (20,))]), encoding='utf-8')
    return labels, videos, predictions


@pytest.mark.parametrize('command', ['audit', 'score', 'run'])
@pytest.mark.parametrize('videos_dir', [r'\\fixture.invalid\share', '//fixture.invalid/share',
                                      r'\\?\UNC\fixture.invalid\share', 'smb://fixture.invalid/share'])
def test_cli_rejects_nonlocal_videos_before_any_input_io(monkeypatch, command, videos_dir):
    import ml.eval as evaluator
    import ml.eval_runner as runner
    from pathlib import Path

    def forbidden(*args, **kwargs):
        pytest.fail('filesystem/input access happened before videos-dir locality validation')

    monkeypatch.setattr(evaluator, 'load_labels', forbidden)
    monkeypatch.setattr(evaluator, 'audit', forbidden)
    monkeypatch.setattr(evaluator, '_write_reports', forbidden)
    monkeypatch.setattr(runner, 'run_pipeline', forbidden)
    monkeypatch.setattr(Path, 'resolve', forbidden)
    args = [command, '--labels', 'unused.csv', '--videos-dir', videos_dir, '--output', 'unused-out']
    if command == 'score':
        args.extend(['--predictions', 'unused.json'])
    with pytest.raises(SystemExit) as error:
        evaluator.main(args)
    assert error.value.code == 2


@pytest.mark.parametrize('command', ['audit', 'score', 'run'])
@pytest.mark.parametrize('location', ['file', 'directory', 'root', 'ancestor'])
def test_cli_rejects_reparse_paths_without_resolution_or_target_io(
        tmp_path, monkeypatch, command, location):
    import ml.eval as evaluator
    import ml.eval_runner as runner
    directory, linked = link_fixture(tmp_path, location)
    labels = tmp_path / 'labels.csv'
    labels.write_text('video,timestamp,label\nfixture,10,нарушение\n', encoding='utf-8')
    checked = synthetic_link_guard(monkeypatch, linked, 'reparse')

    def forbidden(*args, **kwargs):
        pytest.fail('input/worker/media/report access happened before link rejection')

    if location in ('root', 'ancestor'):
        monkeypatch.setattr(evaluator, 'load_labels', forbidden)
    monkeypatch.setattr(evaluator, '_probe', forbidden)
    monkeypatch.setattr(evaluator, 'file_sha256', forbidden)
    monkeypatch.setattr(evaluator, '_write_reports', forbidden)
    monkeypatch.setattr(runner, 'run_pipeline', forbidden)
    output = tmp_path / 'out'
    args = [command, '--labels', str(labels), '--videos-dir', str(directory), '--output', str(output)]
    if command == 'score':
        args.extend(['--predictions', str(tmp_path / 'unread.json')])
    with pytest.raises(SystemExit) as error:
        evaluator.main(args)
    assert error.value.code == 2
    assert checked
    assert not output.exists()


@pytest.mark.parametrize('command', ['score', 'run'])
def test_cli_supplies_actual_hashes_for_content_aggregation(tmp_path, monkeypatch, command):
    import ml.eval as evaluator
    labels, videos, predictions = cli_fixture(tmp_path)
    output = tmp_path / 'out'
    annotations = tmp_path / 'annotations.csv'
    annotations.write_text('video,type_id,t_peak\nX-3,,10\nX-4,,20\n', encoding='utf-8')
    monkeypatch.setattr('ml.eval_runner.run_pipeline', lambda *a, **k: json.loads(predictions.read_text(encoding='utf-8')))
    args = [command, '--labels', str(labels), '--videos-dir', str(videos), '--output', str(output),
            '--annotations', str(annotations)]
    if command == 'score':
        args.extend(['--predictions', str(predictions)])
    assert evaluator.main(args) == 0
    report = json.loads((output / 'overall.json').read_text(encoding='utf-8'))
    assert report['overall']['total'] == report['overall']['localization']['truth_events'] == 1
    assert report['content_aggregation']['hashes_authoritatively_verified'] is True
    assert report['per_content_group'][0]['content_sha256'] == digest(b'NOT MEDIA, SAME ARTIFICIAL CONTENT')
    assert len(json.loads((output / 'per_video.json').read_text(encoding='utf-8'))) == 2


@pytest.mark.parametrize('command', ['score', 'run'])
def test_cli_rejects_forged_hashes_before_inference_or_report_writes(tmp_path, monkeypatch, command):
    import ml.eval as evaluator
    labels, videos, predictions = cli_fixture(tmp_path)
    split = tmp_path / 'split.json'
    split.write_text(json.dumps([dict(video=f'X-{n}', subset=side, angle_id=side,
                                     content_sha256=digest(side.encode()))
                                 for n, side in [(3, 'train'), (4, 'test')]]), encoding='utf-8')
    def forbidden(*args, **kwargs):
        pytest.fail('inference/report write started before actual hash validation')
    monkeypatch.setattr('ml.eval_runner.run_pipeline', forbidden)
    monkeypatch.setattr(evaluator, '_write_reports', forbidden)
    args = [command, '--labels', str(labels), '--videos-dir', str(videos), '--split', str(split),
            '--output', str(tmp_path / 'out')]
    if command == 'score':
        args.extend(['--predictions', str(predictions)])
    with pytest.raises(SystemExit) as error:
        evaluator.main(args)
    assert error.value.code == 2
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('layout', ['labels', 'actual'])
@pytest.mark.parametrize('bad', [None, '', 'A' * 64, 'g' * 64, 'a' * 63])
def test_supplied_hash_sets_must_be_valid_and_complete(layout, bad):
    labels = [label('first'), label('second')]
    hashes = {row['video']: digest(row['video'].encode()) for row in labels}
    if bad is None:
        del hashes['second']
    else:
        hashes['second'] = bad
    kwargs = {'content_hashes': hashes}
    if layout == 'labels':
        labels = [dict(row, **({'content_sha256': hashes[row['video']]} if row['video'] in hashes else {}))
                  for row in labels]
        kwargs = {}
    with pytest.raises(ValueError, match='content_sha256'):
        score(labels, [], **kwargs)


def test_declared_validation_partition_must_not_be_empty():
    labels, rows = split_fixture()
    split = {'train': [rows[0], dict(rows[1], subset='train', angle_id='camera_train')],
             'validation': [], 'test': [rows[2]]}
    with pytest.raises(ValueError, match='empty'):
        score(labels, [], split=split)


def test_hash_consistency_checks_unselected_duplicate_members_too():
    labels = [label('train-a'), label('train-b'), label('test')]
    hashes = {row['video']: digest(b'train' if i < 2 else b'test') for i, row in enumerate(labels)}
    rows = [dict(video=row['video'], subset='train' if i < 2 else 'test',
                 angle_id='train' if i < 2 else 'test', content_sha256=hashes[row['video']])
            for i, row in enumerate(labels)]
    labels[0]['content_sha256'] = hashes['train-a']  # Actual hashes may complete partial label provenance.
    predictions = [prediction('train-a'), prediction('train-b', (20,)), prediction('test')]
    with pytest.raises(ValueError, match='conflicting successful predictions'):
        score(labels, predictions, split=rows, content_hashes=hashes)


def test_consistent_explicit_duplicate_event_inventory_is_counted_once():
    labels = [label('first'), label('second')]
    predictions = [prediction(row['video'], (10, 20)) for row in labels]
    annotations = [dict(video=row['video'], t_peak=peak, type_id='fixture')
                   for row in labels for peak in (10, 20)]
    report = score(labels, predictions, annotations=annotations,
                   content_hashes={row['video']: digest(b'same') for row in labels})
    assert report['overall']['localization']['truth_events'] == 2
    assert report['overall']['localization']['matched_events'] == 2
    assert report['overall']['types']['fixture']['recall_denominator'] == 2
    assert report['overall']['types']['fixture']['precision'] == 1


@pytest.mark.parametrize('reverse_labels', [False, True])
@pytest.mark.parametrize('reverse_annotations', [False, True])
@pytest.mark.parametrize('reverse_predictions', [False, True])
def test_tied_typed_events_are_invariant_to_source_and_representative_order(
        reverse_labels, reverse_annotations, reverse_predictions):
    labels = [label('first'), label('second')]
    annotations = [dict(video=name, t_peak=10, type_id=kind)
                   for name, kinds in [('first', ['alpha', 'beta']), ('second', ['beta', 'alpha'])]
                   for kind in kinds]
    predictions = [dict(prediction(row['video']), violations=[dict(t_peak=10, type_id=kind)
                   for kind in ['alpha', 'beta'][::(-1 if reverse_predictions else 1)]])
                   for row in labels]
    if reverse_labels:
        labels.reverse()
    if reverse_annotations:
        annotations.reverse()
    original = json.dumps([labels, annotations, predictions], sort_keys=True)
    report = score(labels, predictions, annotations=annotations,
                   content_hashes={row['video']: digest(b'same') for row in labels})
    assert report['overall']['type_confusion'] == {'alpha': {'alpha': 1}, 'beta': {'beta': 1}}
    assert all(report['overall']['types'][kind]['precision'] == 1 for kind in ('alpha', 'beta'))
    assert all(report['overall']['types'][kind]['recall'] == 1 for kind in ('alpha', 'beta'))
    for row in report['per_video'] + report['per_content_group']:
        assert [(event['truth_type'], event['predicted_type']) for event in row['matched_events']] == [
            ('alpha', 'alpha'), ('beta', 'beta')]
    assert json.dumps([labels, annotations, predictions], sort_keys=True) == original


@pytest.mark.parametrize('events', [
    [{'t_peak': 10, 'type_id': 'fixture'}, {'t_peak': 20, 'type_id': 'fixture'}],
    [{'t_peak': 10, 'type_id': 'beta'}, {'t_peak': 10, 'type_id': 'alpha'}],
    [{'t_start': 1, 't_end': 3, 'type_id': 'fixture'}, {'t_peak': 10, 'type_id': 'fixture'},
     {'t_start': 4, 't_end': 6, 'type_id': 'fixture'}],
])
def test_permuted_successful_duplicate_predictions_are_the_same_inventory(events):
    labels = [label('first'), label('second')]
    predictions = [dict(prediction(row['video']), violations=ordered)
                   for row, ordered in zip(labels, [events, list(reversed(events))])]
    original = json.dumps(predictions, sort_keys=True)
    report = score(labels, predictions, content_hashes={row['video']: digest(b'same') for row in labels})
    first, second = report['per_video']
    assert first['predicted_events'] == second['predicted_events']
    assert report['per_content_group'][0]['predicted_events'] == first['predicted_events']
    assert len(first['predicted_events']) == len(events)
    assert json.dumps(predictions, sort_keys=True) == original


def test_duplicate_prediction_inventory_multiplicity_still_matters():
    labels = [label('first'), label('second')]
    with pytest.raises(ValueError, match='conflicting successful predictions'):
        score(labels, [prediction('first', (10,)), prediction('second', (10, 10))],
              content_hashes={row['video']: digest(b'same') for row in labels})


@pytest.mark.parametrize('command', ['score', 'run'])
@pytest.mark.parametrize('declared_hashes', [False, True])
def test_incomplete_actual_files_preserve_failures_without_claiming_content_quality(
        tmp_path, monkeypatch, command, declared_hashes):
    import ml.eval as evaluator
    labels, videos, predictions = cli_fixture(tmp_path)
    if declared_hashes:
        sha = digest(b'NOT MEDIA, SAME ARTIFICIAL CONTENT')
        labels.write_text('video,timestamp,label,content_sha256\n'
                          f'Х-3,10,нарушение,{sha}\nХ-4,20,нарушение,{sha}\n', encoding='utf-8')
    (videos / 'X-4.mov').unlink()
    predictions.write_text(json.dumps([prediction('X-3')]), encoding='utf-8')
    monkeypatch.setattr('ml.eval_runner.run_pipeline', lambda *a, **k: json.loads(predictions.read_text(encoding='utf-8')))
    output = tmp_path / 'out'
    args = [command, '--labels', str(labels), '--videos-dir', str(videos), '--output', str(output)]
    if command == 'score':
        args.extend(['--predictions', str(predictions)])
    assert evaluator.main(args) == 0
    report = json.loads((output / 'overall.json').read_text(encoding='utf-8'))
    assert report['overall']['aggregation_unit'] == 'video_diagnostic'
    assert report['overall']['status_counts'] == {'ok': 1, 'missing': 1}
    assert report['overall']['end_to_end']['recall'] == 0.5
    assert report['content_aggregation']['available'] is False
    assert report['content_aggregation']['hashes_authoritatively_verified'] is False
    assert report['content_aggregation']['reason'] == 'actual_content_hashes_incomplete'


@pytest.mark.parametrize('command', ['score', 'run'])
@pytest.mark.parametrize('conflict', ['labels', 'annotations', 'predictions'])
def test_incomplete_audit_keeps_known_duplicate_integrity(tmp_path, monkeypatch, capsys, command, conflict):
    import ml.eval as evaluator
    labels, videos, predictions = cli_fixture(tmp_path)
    second_label = 'без нарушения' if conflict == 'labels' else 'нарушение'
    labels.write_text(f'video,timestamp,label\nХ-3,10,нарушение\nХ-4,10,{second_label}\n'
                      'Х-5,10,нарушение\n', encoding='utf-8')
    outputs = [prediction('X-3'), prediction('X-4', (20,) if conflict == 'predictions' else (10,))]
    predictions.write_text(json.dumps(outputs), encoding='utf-8')
    worker_calls = []

    def workers(*args, **kwargs):
        worker_calls.append(True)
        return outputs

    def forbidden(*args, **kwargs):
        pytest.fail('report write happened despite a proven known-hash conflict')

    monkeypatch.setattr('ml.eval_runner.run_pipeline', workers)
    monkeypatch.setattr(evaluator, '_write_reports', forbidden)
    output = tmp_path / 'out'
    args = [command, '--labels', str(labels), '--videos-dir', str(videos), '--output', str(output)]
    if command == 'score':
        args.extend(['--predictions', str(predictions)])
    if conflict == 'annotations':
        annotations = tmp_path / 'annotations.csv'
        annotations.write_text('video,type_id,t_peak\nX-3,alpha,10\nX-4,beta,10\n', encoding='utf-8')
        args.extend(['--annotations', str(annotations)])
    with pytest.raises(SystemExit) as error:
        evaluator.main(args)
    assert error.value.code == 2
    assert 'conflicting' in capsys.readouterr().err
    assert worker_calls == ([True] if command == 'run' and conflict == 'predictions' else [])
    assert not output.exists()


def test_run_cli_exercises_real_synthetic_workers_and_content_aggregation(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    labels, videos, _ = cli_fixture(tmp_path)
    module = tmp_path / 'synthetic_evaluator_pipeline.py'
    module.write_text('def analyze(video_path, on_progress=None, time_budget_sec=None):\n'
                      '    return {"verdict": "violation", "violations": [{"timecode_sec": 20}]}\n',
                      encoding='utf-8')
    output = tmp_path / 'out'
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(tmp_path) + os.pathsep + str(root))
    process = subprocess.run([sys.executable, '-m', 'ml.eval', 'run', '--labels', str(labels),
                              '--videos-dir', str(videos), '--pipeline', 'synthetic_evaluator_pipeline:analyze',
                              '--output', str(output)], cwd=root, env=env, capture_output=True,
                             text=True, encoding='utf-8', timeout=30)
    assert process.returncode == 0, process.stderr
    report = json.loads((output / 'overall.json').read_text(encoding='utf-8'))
    assert report['overall']['total'] == report['overall']['scored'] == 1
    assert report['overall']['localization']['matched_events'] == 1
    assert report['content_aggregation']['hashes_authoritatively_verified'] is True
    assert report['video_only_overall']['total'] == 2
    assert 'X-3' not in process.stdout and 'Х-3' not in process.stdout


def test_duplicate_alternative_anchors_cannot_turn_two_predictions_into_two_truth_events():
    labels = [label('first', timestamp=10), label('second', timestamp=20)]
    report = score(labels, [prediction(row['video'], (10, 20)) for row in labels],
                   content_hashes={row['video']: digest(b'same') for row in labels})
    loc = report['overall']['localization']
    assert loc['truth_events'] == loc['matched_events'] == 1
    assert loc['predicted_events'] == 2
    assert loc['unmatched_predicted_events'] == 1


def test_console_and_written_report_do_not_hide_unavailable_content_aggregation(tmp_path):
    from ml.eval import _write_reports, console_summary
    report = score([label('fixture')], [prediction('fixture')])
    table = console_summary(report, 'score')
    assert '| content_aggregation_available | 0 |' in table
    _write_reports(report, tmp_path, 'score')
    markdown = (tmp_path / 'report.md').read_text(encoding='utf-8')
    assert 'без порога допуска' not in markdown
    assert 'внутри включительного допуска' in markdown
    assert 'video_diagnostic' in markdown


@pytest.mark.parametrize('conflict', ['labels', 'predictions'])
def test_self_reported_split_hashes_reject_known_conflicts_without_enabling_quality(conflict):
    labels = [label('train-a'), label('train-b'), label('test')]
    rows = [dict(video=row['video'], subset='train' if i < 2 else 'test',
                 angle_id='train' if i < 2 else 'test', content_sha256=digest(b'train' if i < 2 else b'test'))
            for i, row in enumerate(labels)]
    predictions = [prediction(row['video']) for row in labels]
    if conflict == 'labels':
        labels[0] = label('train-a', positive=False, timestamp=None)
    else:
        predictions[0] = prediction('train-a', (20,))
    with pytest.raises(ValueError, match='conflicting'):
        score(labels, predictions, split=rows)


@pytest.mark.parametrize('with_prediction', [False, True])
def test_unknown_single_annotations_echoing_binary_anchors_remain_one_alternative_event(with_prediction):
    labels = [label('first', timestamp=10), label('second', timestamp=20)]
    annotations = [dict(video=row['video'], type_id='', t_peak=row['timestamp']) for row in labels]
    predictions = [prediction(row['video'], (20,)) for row in labels] if with_prediction else []
    report = score(labels, predictions, annotations=annotations,
                   content_hashes={row['video']: digest(b'same') for row in labels})
    assert len(report['per_video']) == 2
    assert report['overall']['total'] == 1
    loc = report['overall']['localization']
    assert loc['truth_events'] == 1
    assert loc['matched_events'] == int(with_prediction)
    assert loc['label_only_limited'] is True
    assert report['per_content_group'][0]['alternative_anchors_sec'] == [10, 20]
    assert report['overall']['status_counts'] == {'ok' if with_prediction else 'missing': 1}


def test_genuinely_different_typed_multi_event_inventories_still_fail_closed():
    labels = [label('first', timestamp=10), label('second', timestamp=20)]
    annotations = [dict(video=row['video'], type_id='fixture_type', t_peak=peak)
                   for row, peaks in zip(labels, [(10, 30), (20, 30)]) for peak in peaks]
    with pytest.raises(ValueError, match='conflicting explicit annotations'):
        score(labels, [], annotations=annotations,
              content_hashes={row['video']: digest(b'same') for row in labels})


@pytest.mark.parametrize('with_prediction', [False, True])
def test_same_typed_single_primary_anchors_are_alternatives_not_conflicting_inventories(with_prediction):
    labels = [label('first', timestamp=10), label('second', timestamp=20)]
    annotations = [dict(video=row['video'], type_id='fixture_type', t_peak=row['timestamp']) for row in labels]
    predictions = [prediction(row['video'], (20,)) for row in labels] if with_prediction else []
    report = score(labels, predictions, annotations=annotations,
                   content_hashes={row['video']: digest(b'same') for row in labels})
    assert len(report['per_video']) == 2
    assert report['overall']['total'] == 1
    loc = report['overall']['localization']
    assert loc['truth_events'] == 1
    assert loc['matched_events'] == int(with_prediction)
    group = report['per_content_group'][0]
    assert group['alternative_anchors_sec'] == [10, 20]
    assert group['truth_events'][0]['type'] == 'fixture_type'


def test_different_types_for_duplicate_single_primary_anchors_remain_a_conflict():
    labels = [label('first', timestamp=10), label('second', timestamp=20)]
    annotations = [dict(video=row['video'], type_id=kind, t_peak=row['timestamp'])
                   for row, kind in zip(labels, ['first_type', 'second_type'])]
    with pytest.raises(ValueError, match='conflicting explicit annotations'):
        score(labels, [], annotations=annotations,
              content_hashes={row['video']: digest(b'same') for row in labels})
