"""Synthetic external intake checks. No organizer/external media or model reads."""

import copy
import hashlib
from html.parser import HTMLParser
import importlib
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import urllib.request

import pytest

TOOLS = Path(__file__).resolve().parents[2] / 'tools'
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
common = importlib.import_module('external23_common')
policy = importlib.import_module('external23_policy')
intake = importlib.import_module('external23_intake')
review = importlib.import_module('external23_review')


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


@pytest.fixture
def kit(tmp_path):
    root = tmp_path / 'private'
    (root / 'sources').mkdir(parents=True)
    (root / 'rights').mkdir()
    sources = []
    for suffix in ('a', 'b'):
        payload = ('synthetic source ' + suffix).encode()
        proof = ('synthetic rights evidence ' + suffix).encode()
        (root / 'sources' / f's-{suffix}.source').write_bytes(payload)
        (root / 'rights' / f's-{suffix}.txt').write_bytes(proof)
        rights = {'path': f'rights/s-{suffix}.txt', 'sha256': digest(proof),
                  'url': f'https://example.org/rights/{suffix}'}
        sources.append({
            'id': f's-{suffix}', 'source_url': f'https://example.org/source-{suffix}.mp4',
            'source_page_url': f'https://example.org/page-{suffix}',
            'license': {'id': 'CC-BY-4.0', 'url': policy.LICENSES['CC-BY-4.0']},
            'attribution': 'Synthetic author', 'source_group': f'group-{suffix}',
            'camera_id': f'camera-{suffix}',
            'source_context': {'country': 'KZ', 'scene_type': 'bank_cashdesk',
                               'evidence_reference': 'synthetic metadata'},
            'expected_sha256': digest(payload), 'expected_size_bytes': len(payload),
            'local_path': f'sources/s-{suffix}.source', 'rights_proof': rights,
        })
    clips = []
    for number in range(10):
        source = sources[number % 2]
        clips.append({'id': f'ext23-case-{number}', 'source_id': source['id'],
                      'start_sec': float(number), 'end_sec': float(number + 2),
                      'selection': 'selected', 'split': 'external_holdout',
                      'angle_id': 'front', 'camera_id': source['camera_id'],
                      'bank_scene_confirmed': True,
                      'bank_scene_evidence_reference': 'synthetic manual scene record',
                      'rights_proof': copy.deepcopy(source['rights_proof'])})
    document = {'schema_version': 1, 'sources': sources, 'clips': clips,
                'selection_policy': {'min_source_groups': 2}}
    manifest = root / 'manifest.json'
    common.write_json(manifest, document)
    return SimpleNamespace(root=root, document=document, manifest=manifest)


def mocked_media(monkeypatch):
    def probe(path, binary):
        return {'duration_seconds': 60.0 if Path(path).parent.name == 'sources' else 2.0,
                'width': 320, 'height': 240, 'codec': 'h264', 'average_fps': 25.0,
                'average_frame_rate': '25/1'}

    def clip(source, output, start, end, binary):
        output.write_bytes(b'synthetic clip:' + source.read_bytes() + f':{start}:{end}'.encode())

    monkeypatch.setattr(intake, 'technical_probe', probe)
    monkeypatch.setattr(intake, 'make_clip', clip)
    return clip


def do_import(kit, **kwargs):
    common.write_json(kit.manifest, kit.document)
    return intake.import_manifest(kit.root, kit.manifest, kit.document, False,
                                  'mock-ffprobe', 'mock-ffmpeg', 4096, **kwargs)


@pytest.mark.parametrize('license_id', sorted(policy.LICENSES))
def test_exact_license_allowlist(kit, license_id):
    source = kit.document['sources'][0]
    source['license'] = {'id': license_id, 'url': policy.LICENSES[license_id]}
    assert policy.license_allowed(source)
    source['license']['url'] += '?unverified=1'
    assert not policy.license_allowed(source)
    with pytest.raises(ValueError, match='LICENSE_NOT_ALLOWED'):
        policy.source_usable(source)


@pytest.mark.parametrize('license_id', ['CC-BY-SA-4.0', 'Public Domain',
                                       'LicenseRef-NARA-Undetermined', 'LicenseRef-RBS-RightsUnconfirmed'])
def test_unapproved_rights_never_materialized(kit, monkeypatch, license_id):
    source = kit.document['sources'][0]
    source['license']['id'] = license_id
    mocked_media(monkeypatch)
    with pytest.raises(ValueError, match='LICENSE_NOT_ALLOWED'):
        do_import(kit)


def test_valid_manifest_and_partial_candidate(kit):
    assert set(intake.validate_manifest(kit.document)) == {'s-a', 's-b'}
    candidate = copy.deepcopy(kit.document)
    candidate['clips'] = [candidate['clips'][0]]
    candidate['clips'][0]['selection'] = 'candidate'
    for key in policy.CLIP_REVIEW_FIELDS:
        candidate['clips'][0].pop(key, None)
    assert intake.validate_manifest(candidate, allow_partial=True)
    with pytest.raises(ValueError, match='REQUIRE_10_TO_15_CLIPS'):
        intake.validate_manifest(candidate)


@pytest.mark.parametrize(('field', 'value', 'error'), [
    ('bank_scene_confirmed', False, 'CLIP_BANK_SCENE_NOT_CONFIRMED'),
    ('bank_scene_evidence_reference', '', 'CLIP_BANK_SCENE_NOT_CONFIRMED'),
    ('angle_id', 'unknown', 'SELECTED_VIEWPOINT_UNKNOWN'),
    ('angle_id', 'UNASSIGNED', 'SELECTED_VIEWPOINT_UNKNOWN'),
    ('camera_id', 'НеИзВеСтНо', 'SELECTED_VIEWPOINT_UNKNOWN'),
    ('rights_proof', None, 'RIGHTS_PROOF_REQUIRED'),
    ('id', 'X-1', 'CLIP_ID_NAMESPACE'),
    ('split', 'train', 'EXTERNAL_HOLDOUT_ONLY'),
])
def test_selected_clip_prerequisites(kit, field, value, error):
    kit.document['clips'][0][field] = value
    with pytest.raises(ValueError, match=error):
        intake.validate_manifest(kit.document)


def test_selected_scope_and_independent_groups(kit):
    kit.document['sources'][0]['source_context']['country'] = 'US'
    with pytest.raises(ValueError, match='SOURCE_OUTSIDE_KZ_SCOPE'):
        intake.validate_manifest(kit.document)
    kit.document['sources'][0]['source_context']['country'] = 'KZ'
    kit.document['sources'][1]['source_group'] = 'group-a'
    with pytest.raises(ValueError, match='INDEPENDENT_SOURCE_GROUPS_REQUIRED'):
        intake.validate_manifest(kit.document)


def test_hash_bound_rights_proof_and_local_paths(kit, tmp_path):
    proof = kit.document['sources'][0]['rights_proof']
    policy.verify_proof(kit.root, proof)
    (kit.root / proof['path']).write_text('changed synthetic evidence')
    with pytest.raises(ValueError, match='RIGHTS_PROOF_SHA256_MISMATCH'):
        policy.verify_proof(kit.root, proof)
    for value in ['../outside', str(tmp_path / 'outside')]:
        with pytest.raises(ValueError, match='PATH_OUTSIDE_PRIVATE_ROOT'):
            common.local_path(kit.root, value)
    (kit.root / 'escape').symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match='PATH_OUTSIDE_PRIVATE_ROOT'):
        common.local_path(kit.root, 'escape/other.txt')
    alias = tmp_path / 'root-alias'
    alias.symlink_to(kit.root, target_is_directory=True)
    with pytest.raises(ValueError, match='PRIVATE_ROOT_SYMLINK'):
        common.local_path(alias, 'anything')


def test_unique_atomic_json_temp_and_duplicate_keys(kit, tmp_path):
    victim = tmp_path / 'victim.txt'
    victim.write_text('preserve')
    output = kit.root / 'report.json'
    output.with_name(output.name + '.part').symlink_to(victim)
    common.write_json(output, {'synthetic': True})
    assert victim.read_text() == 'preserve'
    assert common.read_json(output) == {'synthetic': True}
    output.write_text('{"a":1,"a":2}')
    with pytest.raises(ValueError, match='DUPLICATE_JSON_KEY'):
        common.read_json(output)


@pytest.mark.parametrize('url', ['http://example.org/v', 'ftp://example.org/v',
                                 'https://user:pass@example.org/v'])
def test_insecure_redirect_rejected_before_destination_request(monkeypatch, url):
    called = []
    monkeypatch.setattr(urllib.request.HTTPRedirectHandler, 'redirect_request',
                        lambda *args, **kwargs: called.append(True))
    handler = intake.HTTPSOnlyRedirect()
    request = urllib.request.Request('https://example.org/original')
    with pytest.raises(ValueError, match='PUBLIC_HTTPS_URL_REQUIRED'):
        handler.redirect_request(request, None, 302, 'redirect', {}, url)
    assert called == []


class Response:
    def __init__(self, payload=b'synthetic download'):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def geturl(self):
        return 'https://example.org/final'

    def read(self, size):
        payload, self.payload = self.payload, b''
        return payload


class Opener:
    def __init__(self):
        self.calls = 0

    def open(self, request, timeout):
        self.calls += 1
        return Response()


def test_download_policy_precedes_network_and_size_guard(kit, monkeypatch):
    source = kit.document['sources'][0]
    destination = kit.root / 'sources/download.source'
    opener = Opener()
    source['source_context']['country'] = None
    with pytest.raises(ValueError, match='SOURCE_OUTSIDE_KZ_SCOPE'):
        intake.download(source, destination, 1024, kit.root, opener=opener)
    assert opener.calls == 0
    source['source_context']['country'] = 'KZ'
    monkeypatch.setattr(shutil, 'disk_usage', lambda path: SimpleNamespace(free=2 * 1024 ** 3))
    with pytest.raises(ValueError, match='SOURCE_DOWNLOAD_LIMIT_EXCEEDED'):
        intake.download(source, destination, 2, kit.root, opener=opener)
    assert not destination.exists() and not destination.with_suffix('.partial').exists()
    intake.download(source, destination, 1024, kit.root, opener=opener)
    assert destination.read_bytes() == b'synthetic download'


def test_local_import_retry_preserves_sources_and_annotations(kit, monkeypatch):
    mocked_media(monkeypatch)
    monkeypatch.setattr(intake, 'download', lambda *args, **kwargs: pytest.fail('unexpected network'))
    original_sources = {p.name: p.read_bytes() for p in (kit.root / 'sources').iterdir()}
    index = do_import(kit)
    assert len(index['records']) == 10
    assert all(r['selection'] == 'selected' and r['split'] == 'external_holdout' for r in index['records'])
    first_hashes = {r['clip_id']: r['clip_sha256'] for r in index['records']}
    annotations = common.read_json(kit.root / 'annotations.json')
    assert all(r['events'] == [] and r['legal_gold'] is None for r in annotations['clips'])
    before = (kit.root / 'annotations.json').read_bytes()
    again = do_import(kit)
    assert {r['clip_id']: r['clip_sha256'] for r in again['records']} == first_hashes
    assert (kit.root / 'annotations.json').read_bytes() == before
    assert original_sources == {p.name: p.read_bytes() for p in (kit.root / 'sources').iterdir()}


def test_content_identity_immutable_and_context_changes_journaled(kit):
    old = copy.deepcopy(kit.document)
    kit.document['clips'][0]['start_sec'] = 0.5
    with pytest.raises(ValueError, match='MANIFEST_CHANGED_CONTENT_IDENTITY'):
        policy.journal_changes(old, kit.document)
    kit.document = copy.deepcopy(old)
    context = kit.document['sources'][0]['source_context']
    context['evidence_reference'] = 'new synthetic evidence'
    with pytest.raises(ValueError, match='CONTEXT_CHANGE_NOT_JOURNALED'):
        policy.journal_changes(old, kit.document)
    kit.document['change_log'] = [{
        'entity': 'sources', 'id': 's-a', 'field': 'source_context',
        'before_sha256': common.value_sha256(old['sources'][0]['source_context']),
        'after_sha256': common.value_sha256(context), 'changed_by': 'Synthetic reviewer',
        'changed_at': '2026-10-17T12:00:00+05:00', 'reason': 'Synthetic correction',
    }]
    changes = policy.journal_changes(old, kit.document)
    assert len(changes) == 1 and changes[0]['field'] == 'source_context'
    kit.document['change_log'][0]['before_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='CONTEXT_CHANGE_NOT_JOURNALED'):
        policy.journal_changes(old, kit.document)


def test_import_context_journal_updates_intake_without_reencoding(kit, monkeypatch):
    mocked_media(monkeypatch)
    first = do_import(kit)
    old = copy.deepcopy(kit.document)
    context = kit.document['sources'][0]['source_context']
    context['evidence_reference'] = 'reviewed synthetic evidence'
    kit.document['change_log'] = [{
        'entity': 'sources', 'id': 's-a', 'field': 'source_context',
        'before_sha256': common.value_sha256(old['sources'][0]['source_context']),
        'after_sha256': common.value_sha256(context), 'changed_by': 'Synthetic reviewer',
        'changed_at': '2026-10-17T12:00:00+05:00', 'reason': 'Synthetic correction',
    }]
    monkeypatch.setattr(intake, 'make_clip', lambda *args: pytest.fail('unexpected reencoding'))
    again = do_import(kit)
    assert len(again['previous_manifest_sha256']) == 1
    assert len(again['manifest_change_log'][0]['changes']) == 1
    assert again['records'][0]['source_context'] == context
    assert [r['clip_sha256'] for r in first['records']] == [r['clip_sha256'] for r in again['records']]


def test_orphan_clip_retry_requires_pending_identity(kit, monkeypatch):
    good_clip = mocked_media(monkeypatch)
    kit.document['clips'] = [kit.document['clips'][0]]
    kit.document['clips'][0]['selection'] = 'candidate'

    def interrupted(source, output, start, end, binary):
        output.write_bytes(b'untrusted interrupted synthetic output')
        raise subprocess.TimeoutExpired('mock-ffmpeg', 1)

    monkeypatch.setattr(intake, 'make_clip', interrupted)
    with pytest.raises(subprocess.TimeoutExpired):
        do_import(kit, allow_partial=True)
    output = kit.root / 'clips/ext23-case-0.mp4'
    assert output.exists()
    monkeypatch.setattr(intake, 'make_clip', good_clip)
    index = do_import(kit, allow_partial=True)
    assert len(index['records']) == 1
    archives = list((kit.root / 'orphaned').glob('*.mp4'))
    assert len(archives) == 1 and archives[0].read_bytes() == b'untrusted interrupted synthetic output'
    assert not (kit.root / 'pending/ext23-case-0.json').exists()


class Tags(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))


def test_review_xss_csp_and_local_media_only(kit, monkeypatch):
    monkeypatch.setattr(review, 'PRIVATE', kit.root)
    malicious = '</textarea><script src="https://example.org/leak"></script><img onerror=alert(1)>'
    clip_id = "ext23-" + malicious
    annotations = {'schema_version': 1, 'clips': [{'clip_id': clip_id, 'notes': malicious}]}
    index = {'records': [{'clip_id': clip_id, 'clip_path': 'clips/synthetic.mp4',
                          'source_url': malicious, 'attribution': malicious,
                          'license': kit.document['sources'][0]['license'],
                          'source_context': {'country': 'KZ', 'scene_type': 'bank_cashdesk'}}]}
    model = [{'clip_id': clip_id, 'raw_response': malicious}]
    page, clips, models = review.render(index, annotations, model, kit.root / 'review/index.html')
    parser = Tags()
    parser.feed(page)
    assert (clips, models) == (1, 1)
    assert not any(tag == 'img' for tag, _ in parser.tags)
    assert not any(tag == 'script' and 'src' in attrs for tag, attrs in parser.tags)
    videos = [attrs for tag, attrs in parser.tags if tag == 'video']
    assert len(videos) == 1 and 'autoplay' not in videos[0]
    assert videos[0]['src'].startswith('../clips/')
    assert "connect-src 'none'" in page and "default-src 'none'" in page
    assert '<script src=' not in page and '\\u003c/textarea\\u003e' in page
    index['records'][0]['clip_path'] = '../outside.mp4'
    with pytest.raises(ValueError, match='review_path_outside_private'):
        review.render(index, annotations, model, kit.root / 'review/index.html')

annotations_api = importlib.import_module('external23_annotations')
export_api = importlib.import_module('external23_export')
vlm = importlib.import_module('external23_vlm')


def reviewed_pair(kit, monkeypatch):
    mocked_media(monkeypatch)
    index = do_import(kit)
    index['records'] = index['records'][:2]
    drafts = annotations_api.draft_annotations(index['records'])
    for row in drafts['clips']:
        row.update(annotation_status='reviewed_human', reviewer='Synthetic reviewer',
                   reviewed_on='2026-10-17', full_clip_reviewed=True,
                   normative_scenario={'jurisdiction': 'KZ', 'basis': 'indicator_benchmark',
                                       'decision_reference': 'synthetic policy'},
                   applicability={'jurisdiction': 'KZ', 'scene_type': 'bank_cashdesk',
                                  'roles_established': True, 'normative_conditions_established': True,
                                  'evidence_references': ['synthetic complete review']},
                   legal_gold={'verdict': 'no_violation', 'type_ids': []})
    event = {'provenance': 'reviewed_human', 'type_id': 'computer_left_unlocked',
             't_start': 0.0, 't_end': 1.0, 't_peak': 0.5, 'angle_id': 'front',
             'action_label': 'synthetic indicator', 'actor': 'cashier',
             'roles': [{'role': 'cashier', 'status': 'human_reviewed'}],
             'observations': 'Synthetic visible fact',
             'origin': {'reference': 'synthetic review', 'reviewer': 'Synthetic reviewer',
                        'reviewed_on': '2026-10-17'}}
    drafts['clips'][0]['events'] = [event]
    drafts['clips'][0]['legal_gold'] = {'verdict': 'violation', 'type_ids': ['computer_left_unlocked']}
    return index, drafts


def test_gold_t04_without_norm_and_reviewed_negative_export(kit, monkeypatch):
    import csv
    index, drafts = reviewed_pair(kit, monkeypatch)
    annotations_api.validate_annotations(drafts, index['records'])
    second_event = copy.deepcopy(drafts['clips'][0]['events'][0])
    second_event['t_peak'] = 0.7
    drafts['clips'][0]['events'].insert(0, second_event)
    result = export_api.export_selected(kit.root, index, drafts)
    assert result == {'exported_clips': 2, 'annotation_rows': 3, 'negative_clips': 1,
                      'evaluator_split_written': False}
    with (kit.root / 'exports/annotations.csv').open(newline='') as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        assert reader.fieldnames == export_api.ANNOTATION_FIELDS
    negative = next(r for r in rows if r['type_id'] == 'none')
    assert all(negative[k] == '' for k in ['t_start', 't_end', 't_peak', 'actor', 'what_visible'])
    with (kit.root / 'exports/labels.csv').open(newline='') as stream:
        reader = csv.DictReader(stream)
        labels = list(reader)
        assert reader.fieldnames == export_api.LABEL_FIELDS
    assert labels[0]['label'] == 'нарушение' and float(labels[0]['timestamp']) == 0.5
    assert labels[1]['label'] == 'без нарушения' and labels[1]['timestamp'] == ''
    assert [r['content_sha256'] for r in labels] == [r['clip_sha256'] for r in index['records']]
    holdout = common.read_json(kit.root / 'exports/external_holdout.json')
    assert holdout['kind'] == 'external_holdout_manifest' and not holdout['ready_for_evaluator']
    assert len(holdout['items']) == 2 and all(r['split'] == 'external_holdout' for r in holdout['items'])
    assert not (kit.root / 'exports/split.json').exists()


@pytest.mark.parametrize(('change', 'error'), [
    ('missing_human', 'annotation_gold_review_required'),
    ('not_full_clip', 'annotation_gold_review_required'),
    ('no_scenario', 'annotation_scenario_required'),
    ('no_visible_fact', 'annotation_gold_event_invalid'),
    ('no_peak', 'annotation_gold_event_invalid'),
    ('peak_outside_event', 'annotation_gold_event_invalid'),
    ('mixed_angle', 'annotation_mixed_angles'),
    ('duplicate_type_peak', 'annotation_duplicate_type_peak'),
    ('negative_has_event', 'annotation_negative_has_events'),
    ('unknown_type', 'annotation_gold_invalid'),
])
def test_gold_does_not_accept_unreviewed_or_inconsistent_evidence(kit, monkeypatch, change, error):
    index, drafts = reviewed_pair(kit, monkeypatch)
    first = drafts['clips'][0]
    event = first['events'][0]
    if change == 'missing_human':
        first['annotation_status'] = 'pending_review'
    elif change == 'not_full_clip':
        first['full_clip_reviewed'] = False
    elif change == 'no_scenario':
        first.pop('normative_scenario')
    elif change == 'no_visible_fact':
        event['observations'] = ''
    elif change == 'no_peak':
        event.pop('t_peak')
    elif change == 'peak_outside_event':
        event['t_peak'] = 1.5
    elif change == 'mixed_angle':
        event['angle_id'] = 'side'
    elif change == 'duplicate_type_peak':
        first['events'].append(copy.deepcopy(event))
    elif change == 'negative_has_event':
        drafts['clips'][1]['events'] = [copy.deepcopy(event)]
    elif change == 'unknown_type':
        first['legal_gold']['type_ids'] = ['not_in_active_catalog']
    with pytest.raises(ValueError, match=error):
        annotations_api.validate_annotations(drafts, index['records'])


def test_pending_exports_no_negative_and_refuses_evaluator_split(kit, monkeypatch):
    import csv
    index, _ = reviewed_pair(kit, monkeypatch)
    pending = annotations_api.draft_annotations(index['records'])
    result = export_api.export_selected(kit.root, index, pending)
    assert result['exported_clips'] == result['annotation_rows'] == result['negative_clips'] == 0
    for name in ['annotations.csv', 'labels.csv']:
        with (kit.root / 'exports' / name).open(newline='') as stream:
            assert list(csv.DictReader(stream)) == []
    with pytest.raises(ValueError, match='EVALUATOR_TRAIN_INPUTS_REQUIRED'):
        export_api.export_selected(kit.root, {}, {}, split_json=True)


@pytest.mark.parametrize(('change', 'error'), [
    ('namespace', 'CLIP_ID_NAMESPACE'),
    ('hash_tamper', 'EXPORT_CLIP_SHA256_MISMATCH'),
    ('duplicate_hash', 'EXPORT_DUPLICATE_CLIP_CONTENT'),
    ('normalized_id_collision', 'EXPORT_VIDEO_ID_COLLISION'),
    ('one_group', 'INDEPENDENT_SOURCE_GROUPS_REQUIRED'),
    ('unknown_camera', 'SELECTED_VIEWPOINT_UNKNOWN'),
    ('unconfirmed_bank', 'CLIP_BANK_SCENE_NOT_CONFIRMED'),
    ('bad_rights_proof', 'RIGHTS_PROOF_SHA256_MISMATCH'),
])
def test_export_identity_rights_and_viewpoint_guards(kit, monkeypatch, change, error):
    index, drafts = reviewed_pair(kit, monkeypatch)
    first, second = index['records']
    if change == 'namespace':
        first['clip_id'] = drafts['clips'][0]['clip_id'] = 'X-1'
    elif change == 'hash_tamper':
        (kit.root / first['clip_path']).write_bytes(b'tampered synthetic output')
    elif change == 'duplicate_hash':
        (kit.root / second['clip_path']).write_bytes((kit.root / first['clip_path']).read_bytes())
        second['clip_sha256'] = first['clip_sha256']
    elif change == 'normalized_id_collision':
        old_path = kit.root / second['clip_path']
        aliases = kit.root / 'case-aliases'
        aliases.mkdir()
        new_path = aliases / Path(first['clip_path']).name.upper()
        new_path.write_bytes(old_path.read_bytes())
        second['clip_path'] = str(new_path.relative_to(kit.root))
    elif change == 'one_group':
        second['source_group'] = first['source_group']
    elif change == 'unknown_camera':
        first['camera_id'] = 'unknown'
    elif change == 'unconfirmed_bank':
        first['bank_scene_confirmed'] = False
    elif change == 'bad_rights_proof':
        (kit.root / first['rights_proof']['path']).write_bytes(b'changed synthetic rights')
    with pytest.raises(ValueError, match=error):
        export_api.export_selected(kit.root, index, drafts)


def test_csv_formula_escaped_without_executing_content(kit, monkeypatch):
    import csv
    index, drafts = reviewed_pair(kit, monkeypatch)
    drafts['clips'][0]['events'][0]['observations'] = '=SYNTHETIC_FORMULA()'
    export_api.export_selected(kit.root, index, drafts)
    with (kit.root / 'exports/annotations.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]['what_visible'] == "'=SYNTHETIC_FORMULA()"


def test_experimental_vlm_disabled_before_files_or_processes(monkeypatch):
    monkeypatch.setattr(Path, 'open', lambda *args, **kwargs: pytest.fail('unexpected file read'))
    monkeypatch.setattr(subprocess, 'run', lambda *args, **kwargs: pytest.fail('unexpected subprocess'))
    with pytest.raises(ValueError, match='experimental_vlm_disabled_unapproved_checkpoint'):
        vlm.run(SimpleNamespace())


@pytest.mark.parametrize('module', [intake, export_api, review, vlm], ids=['intake', 'export', 'review', 'vlm'])
def test_cli_argument_errors_never_echo_private_values(monkeypatch, capsys, module):
    monkeypatch.setattr(sys, 'argv', ['synthetic', '--synthetic-private-argument-SECRET'])
    assert module.main() == 1
    captured = capsys.readouterr()
    assert 'SECRET' not in captured.out + captured.err


def test_vlm_cli_disabled_without_reading_supplied_paths(monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['synthetic', '--manifest', 'NOT-READ-SECRET.json',
                                     '--output', 'NOT-WRITTEN-SECRET.json'])
    monkeypatch.setattr(Path, 'open', lambda *args, **kwargs: pytest.fail('unexpected file access'))
    monkeypatch.setattr(subprocess, 'run', lambda *args, **kwargs: pytest.fail('unexpected process'))
    assert vlm.main() == 1
    captured = capsys.readouterr()
    assert 'SECRET' not in captured.out + captured.err and 'failed' in captured.out


def test_context_journal_history_is_append_only(kit):
    previous = copy.deepcopy(kit.document)
    previous['change_log'] = [{'entity': 'sources', 'id': 's-a', 'field': 'source_context',
                               'before_sha256': 'a' * 64, 'after_sha256': 'b' * 64,
                               'changed_by': 'Synthetic reviewer',
                               'changed_at': '2026-10-17T12:00:00+05:00', 'reason': 'Synthetic history'}]
    current = copy.deepcopy(previous)
    current['change_log'] = []
    with pytest.raises(ValueError, match='MANIFEST_CHANGE_LOG_NOT_APPEND_ONLY'):
        policy.journal_changes(previous, current)
    current = copy.deepcopy(previous)
    current['change_log'][0]['reason'] = 'Rewritten history'
    with pytest.raises(ValueError, match='MANIFEST_CHANGE_LOG_NOT_APPEND_ONLY'):
        policy.journal_changes(previous, current)
    current = copy.deepcopy(previous)
    current['change_log'].append(dict(previous['change_log'][0], reason='New history entry'))
    assert policy.journal_changes(previous, current) == []


@pytest.mark.parametrize('proof_failure', ['missing', 'stale_source', 'stale_clip'])
def test_local_candidate_requires_verified_rights_before_media(kit, monkeypatch, proof_failure):
    clip = kit.document['clips'][0]
    clip['selection'] = 'candidate'
    kit.document['clips'] = [clip]
    source = kit.document['sources'][0]
    if proof_failure == 'missing':
        source.pop('rights_proof')
        clip.pop('rights_proof')
        expected = 'RIGHTS_PROOF_REQUIRED'
    elif proof_failure == 'stale_source':
        clip.pop('rights_proof')
        source['rights_proof']['sha256'] = '0' * 64
        expected = 'RIGHTS_PROOF_SHA256_MISMATCH'
    else:
        source.pop('rights_proof')
        clip['rights_proof']['sha256'] = '0' * 64
        expected = 'RIGHTS_PROOF_SHA256_MISMATCH'
    original_hash = intake.sha256

    def no_source_read(path):
        assert Path(path).parent.name != 'sources', 'source hash attempted before rights proof'
        return original_hash(path)

    monkeypatch.setattr(intake, 'sha256', no_source_read)
    for name in ['technical_probe', 'make_clip', 'download']:
        monkeypatch.setattr(intake, name, lambda *args, **kwargs: pytest.fail('media processing before rights proof'))
    with pytest.raises(ValueError, match=expected):
        do_import(kit, allow_partial=True)
    assert not list((kit.root / 'clips').glob('*.mp4'))


@pytest.mark.parametrize('proof_location', ['source', 'clip'])
def test_candidate_can_use_verified_source_or_clip_proof(kit, monkeypatch, proof_location):
    mocked_media(monkeypatch)
    clip = kit.document['clips'][0]
    clip['selection'] = 'candidate'
    kit.document['clips'] = [clip]
    source = kit.document['sources'][0]
    if proof_location == 'source':
        clip.pop('rights_proof')
        expected = source['rights_proof']
    else:
        source.pop('rights_proof')
        expected = clip['rights_proof']
    index = do_import(kit, allow_partial=True)
    assert len(index['records']) == 1 and index['records'][0]['rights_proof'] == expected
    assert index['records'][0]['selection'] == 'candidate'


def test_metadata_only_reserve_requires_no_rights_proof_or_media(kit, monkeypatch):
    kit.document['clips'] = [kit.document['clips'][0]]
    kit.document['clips'][0]['selection'] = 'reserve'
    kit.document['clips'][0].pop('rights_proof')
    for source in kit.document['sources']:
        source.pop('rights_proof')
    for name in ['technical_probe', 'make_clip', 'download']:
        monkeypatch.setattr(intake, name, lambda *args, **kwargs: pytest.fail('reserve must remain metadata-only'))
    index = do_import(kit, allow_partial=True)
    assert index['records'] == []
    assert not list((kit.root / 'clips').glob('*.mp4'))


def test_context_journal_old_transition_cannot_authorize_replay(kit):
    state_a = copy.deepcopy(kit.document['sources'][0]['source_context'])
    state_b = dict(state_a, evidence_reference='Synthetic state B')

    def entry(before, after, reason):
        return {'entity': 'sources', 'id': 's-a', 'field': 'source_context',
                'before_sha256': common.value_sha256(before), 'after_sha256': common.value_sha256(after),
                'changed_by': 'Synthetic reviewer', 'changed_at': '2026-10-17T12:00:00+05:00',
                'reason': reason}

    previous = copy.deepcopy(kit.document)
    previous['change_log'] = [entry(state_a, state_b, 'Historical A to B'),
                              entry(state_b, state_a, 'Historical B to A')]
    current = copy.deepcopy(previous)
    current['sources'][0]['source_context'] = state_b
    with pytest.raises(ValueError, match='CONTEXT_CHANGE_NOT_JOURNALED'):
        policy.journal_changes(previous, current)
    current['change_log'].append(entry(state_a, state_b, 'Fresh transition approval'))
    changes = policy.journal_changes(previous, current)
    assert len(changes) == 1 and changes[0]['reason'] == 'Fresh transition approval'


@pytest.mark.parametrize('field', ['angle_id', 'camera_id'])
@pytest.mark.parametrize('value', [' front', 'front ', '\tfront', 'front\n', '\u00a0front',
                                   '=front', '+front', '-front', '@front'])
def test_structured_viewpoint_ids_are_not_trimmed_or_formula_escaped(kit, field, value):
    clip = copy.deepcopy(kit.document['clips'][0])
    clip[field] = value
    with pytest.raises(ValueError, match='SELECTED_VIEWPOINT_UNKNOWN'):
        policy.selected_clip(kit.document['sources'][0], clip)
    assert clip[field] == value


def test_export_rejects_train_split_instead_of_relabeling_holdout(kit, monkeypatch):
    index, drafts = reviewed_pair(kit, monkeypatch)
    index['records'][0]['split'] = 'train'
    with pytest.raises(ValueError, match='EXTERNAL_HOLDOUT_ONLY'):
        export_api.export_selected(kit.root, index, drafts)
    assert not (kit.root / 'exports').exists()


@pytest.mark.parametrize('folder', ['quarantine', 'excluded'])
@pytest.mark.parametrize('input_kind', ['clip', 'proof'])
def test_export_rejects_excluded_media_and_proof_even_with_matching_hash(kit, monkeypatch, folder, input_kind):
    index, drafts = reviewed_pair(kit, monkeypatch)
    record = index['records'][0]
    excluded = kit.root / folder
    excluded.mkdir()
    if input_kind == 'clip':
        destination = excluded / Path(record['clip_path']).name
        destination.write_bytes((kit.root / record['clip_path']).read_bytes())
        record['clip_path'] = str(destination.relative_to(kit.root))
        assert common.sha256(destination) == record['clip_sha256']
    else:
        proof = copy.deepcopy(record['rights_proof'])
        destination = excluded / 'synthetic-proof.txt'
        destination.write_bytes((kit.root / proof['path']).read_bytes())
        proof['path'] = str(destination.relative_to(kit.root))
        record['rights_proof'] = proof
        assert common.sha256(destination) == proof['sha256']
    with pytest.raises(ValueError, match='EXCLUDED_INPUT_OR_OUTPUT'):
        export_api.export_selected(kit.root, index, drafts)
    assert not (kit.root / 'exports').exists()


@pytest.mark.parametrize('folder', ['quarantine', 'excluded'])
def test_local_candidate_rejects_excluded_source_before_processing(kit, monkeypatch, folder):
    clip = kit.document['clips'][0]
    clip['selection'] = 'candidate'
    kit.document['clips'] = [clip]
    source = kit.document['sources'][0]
    excluded = kit.root / folder
    excluded.mkdir()
    destination = excluded / 'synthetic-source.bin'
    destination.write_bytes((kit.root / source['local_path']).read_bytes())
    source['local_path'] = str(destination.relative_to(kit.root))
    assert common.sha256(destination) == source['expected_sha256']
    for name in ['technical_probe', 'make_clip', 'download']:
        monkeypatch.setattr(intake, name, lambda *args, **kwargs: pytest.fail('excluded media must not be processed'))
    with pytest.raises(ValueError, match='EXCLUDED_INPUT_OR_OUTPUT'):
        do_import(kit, allow_partial=True)
    assert not list((kit.root / 'clips').glob('*.mp4'))


@pytest.mark.parametrize('folder', ['QuArAnTiNe', 'ExClUdEd'])
@pytest.mark.parametrize('cli_option', ['--output-dir', '--index', '--annotations'])
def test_mixed_case_excluded_output_rejected_before_creation_in_api_and_cli(kit, monkeypatch, capsys, folder, cli_option):
    index, drafts = reviewed_pair(kit, monkeypatch)
    output_dir = folder + '/exports'
    monkeypatch.setattr(export_api, 'csv_file', lambda *args, **kwargs: pytest.fail('excluded output write'))
    monkeypatch.setattr(export_api, 'write_json', lambda *args, **kwargs: pytest.fail('excluded output write'))
    with pytest.raises(ValueError, match='EXCLUDED_INPUT_OR_OUTPUT'):
        export_api.export_selected(kit.root, index, drafts, output_dir=output_dir)
    assert not (kit.root / folder).exists()
    monkeypatch.setattr(export_api, 'PRIVATE', kit.root)
    monkeypatch.setattr(sys, 'argv', ['synthetic', cli_option, output_dir])
    monkeypatch.setattr(export_api, 'read_json', lambda *args, **kwargs: pytest.fail('input read before output guard'))
    assert export_api.main() == 1
    assert not (kit.root / folder).exists()
    assert 'ERROR:' in capsys.readouterr().err


@pytest.mark.parametrize('folder', ['QuArAnTiNe', 'ExClUdEd'])
def test_review_rejects_case_alias_media_before_generating_url(kit, monkeypatch, folder):
    monkeypatch.setattr(review, 'PRIVATE', kit.root)
    monkeypatch.setattr(review, 'quote', lambda *args, **kwargs: pytest.fail('excluded media URL generated'))
    clip_id = 'ext23-synthetic'
    index = {'records': [{'clip_id': clip_id, 'clip_path': folder + '/synthetic.mp4',
                          'license': kit.document['sources'][0]['license'],
                          'source_context': {'country': 'KZ', 'scene_type': 'bank_cashdesk'}}]}
    annotations = {'schema_version': 1, 'clips': [{'clip_id': clip_id}]}
    with pytest.raises(ValueError, match='EXCLUDED_INPUT_OR_OUTPUT'):
        review.render(index, annotations, [], kit.root / 'review/index.html')


@pytest.mark.parametrize('folder', ['QuArAnTiNe', 'ExClUdEd'])
@pytest.mark.parametrize('cli_option', ['--index', '--annotations', '--model-results', '--output'])
def test_review_rejects_case_alias_inputs_and_output_before_reading(kit, monkeypatch, capsys, folder, cli_option):
    monkeypatch.setattr(review, 'PRIVATE', kit.root)
    args = ['synthetic', '--index', str(kit.root / 'intake.json'),
            '--annotations', str(kit.root / 'annotations.json'),
            '--output', str(kit.root / 'review/index.html'),
            cli_option, str(kit.root / folder / 'synthetic.html')]
    monkeypatch.setattr(sys, 'argv', args)
    monkeypatch.setattr(review, 'read_json', lambda *args: pytest.fail('excluded input read'))
    assert review.main() == 1
    assert not (kit.root / folder).exists()
    assert 'ERROR:' in capsys.readouterr().err
