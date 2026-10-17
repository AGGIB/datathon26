"""Pilot review regression fixtures: eight fake byte files, no real media or models."""

import copy
import csv
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import uuid

import pytest

TOOLS = Path(__file__).resolve().parents[2] / 'tools'
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
pilot = importlib.import_module('pilot_review')
schema = importlib.import_module('pilot_review_schema')
TYPE_ID = 'computer_left_unlocked'


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def human(name='Synthetic reviewer A', decision=None):
    result = {'kind': 'human', 'reviewer': name, 'date': '2026-10-17'}
    if decision is not None:
        result['decision'] = decision
    return result


@pytest.fixture
def make_case(tmp_path, monkeypatch):
    root = tmp_path / 'synthetic-repo'
    root.mkdir()
    monkeypatch.setattr(pilot, 'ROOT', root)
    monkeypatch.setattr(pilot, '_probe', lambda *args, **kwargs: {
        'status': 'ok', 'duration_sec': 10.0, 'width': 320, 'height': 240,
        'fps': 25.0, 'video_streams': 1,
    })

    def create(duplicate=False):
        videos = root / 'fixtures/videos'
        videos.mkdir(parents=True)
        ids = [f'X-{101 + i}' for i in range(8)]
        paths = []
        for i, name in enumerate(ids):
            payload = f'synthetic non-media payload {6 if duplicate and i == 7 else i}'.encode()
            path = videos / (name + '.mov')
            path.write_bytes(payload)
            paths.append(path)
        labels = root / 'fixtures/labels.csv'
        with labels.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['video', 'timestamp', 'label'])
            writer.writerows([(name, '1.25' if i == 0 else '',
                               'нарушение' if i == 0 else 'без нарушения') for i, name in enumerate(ids)])
        id_path = root / 'fixtures/pilots.json'
        save_json(id_path, ids)
        result = SimpleNamespace(root=root, videos=videos, labels=labels, ids=ids, paths=paths,
                                 id_path=id_path, output=root / 'data/pilot-review')
        return result

    return create


def initialized_case(result):
    pilot.initialize(result.id_path, result.videos, result.labels, ffprobe=sys.executable)
    result.inventory_path = result.output / 'inventory.json'
    result.adjudication_path = result.output / 'adjudication.json'
    result.inventory = json.loads(result.inventory_path.read_text())
    result.adjudication = json.loads(result.adjudication_path.read_text())
    result.inventory_hash = result.adjudication['inventory_sha256']
    return result


@pytest.fixture
def case(make_case):
    return initialized_case(make_case())


@pytest.fixture
def duplicate_case(make_case):
    return initialized_case(make_case(duplicate=True))


def check(case, document=None):
    return schema.validate_documents(case.inventory, document if document is not None else case.adjudication,
                                     case.inventory_hash, type_ids={TYPE_ID})


def event_for(clip, *, peak=1.25):
    return {'event_id': str(uuid.uuid4()), 'video_key': clip['video_key'],
            'content_sha256': clip['content_sha256'], 'type_id': TYPE_ID,
            'action': 'Synthetic visible action', 'actor_role': 'unknown', 'object': 'unknown',
            'visibility': 'clear', 't_start': 1.0, 't_peak': peak, 't_end': 1.5,
            'primary': human(decision='confirmed'),
            'secondary': human('Synthetic reviewer B', decision='confirmed'),
            'consensus': 'confirmed', 'disagreement': None,
            'evidence_timestamp_sec': None, 'source_metadata': None}


def clarify_organizer(case, clip):
    source = next(row for row in case.inventory['clips'] if row['video_key'] == clip['video_key'])
    clip['organizer_clarification'] = {
        'reference': 'inventory.organizer', 'source_sha256': source['organizer']['source_sha256'],
        'note': 'Synthetic explicit source-versus-human annotation clarification',
    }


def full_review(case):
    document = copy.deepcopy(case.adjudication)
    for clip in document['clips']:
        clip.update(coverage='full_clip', primary_review=human(), secondary_review=None,
                    clip_decision='no_violation', events=[], disagreements=[])
    first = document['clips'][0]
    first['clip_decision'] = 'has_confirmed_event'
    first['events'] = [event_for(first)]
    return document


def test_pending_is_structurally_valid_but_not_handoff(case):
    result = check(case)
    assert result['structurally_valid']
    assert not result['review_handoff_ready'] and not result['full_truth_ready']
    assert not result['complete']
    assert all(c['primary_review'] is None and not c['events'] for c in case.adjudication['clips'])


def test_full_review_uses_unknown_viewpoints_without_blocking_truth(case):
    document = full_review(case)
    result = check(case, document)
    assert result['review_handoff_ready'] and result['full_truth_ready'] and result['complete']


def test_primary_only_unclear_is_diagnostic_handoff(case):
    document = copy.deepcopy(case.adjudication)
    for clip in document['clips']:
        clip.update(coverage='primary_only', primary_review=human(),
                    secondary_review=human('Synthetic reviewer B'), clip_decision='unclear',
                    events=[], disagreements=['Synthetic coverage remains incomplete'])
    result = check(case, document)
    assert result['review_handoff_ready'] and result['complete']
    assert not result['full_truth_ready']


def test_one_unreviewed_clip_blocks_handoff(case):
    document = full_review(case)
    document['clips'][-1] = copy.deepcopy(case.adjudication['clips'][-1])
    result = check(case, document)
    assert not result['review_handoff_ready'] and not result['full_truth_ready']


@pytest.mark.parametrize(('field', 'value'), [('t_start', -0.1), ('t_peak', 11.0),
                                           ('t_end', 11.0), ('t_peak', True),
                                           ('t_start', '1.0'), ('visibility', 'occluded'),
                                           ('actor_role', 'manager'), ('consensus', 'verified'),
                                           ('type_id', 'not_active_type')])
def test_event_boundaries_and_enums_rejected(case, field, value):
    document = full_review(case)
    document['clips'][0]['events'][0][field] = value
    with pytest.raises(ValueError):
        check(case, document)


@pytest.mark.parametrize('name', ['Synthetic reviewer A', 'synthetic REVIEWER a', 'Synthetic  reviewer A'])
def test_event_second_reviewer_must_be_independent(case, name):
    document = full_review(case)
    document['clips'][0]['events'][0]['secondary'] = human(name, decision='confirmed')
    with pytest.raises(ValueError):
        check(case, document)


@pytest.mark.parametrize('name', ['Codex', 'ChatGPT', 'local_model'])
def test_machine_identifiers_cannot_claim_human_review(case, name):
    document = full_review(case)
    document['clips'][1]['primary_review'] = human(name)
    with pytest.raises(ValueError):
        check(case, document)


def test_confirmed_candidate_missing_second_review_never_becomes_ready(case):
    document = full_review(case)
    event = document['clips'][0]['events'][0]
    event.update(secondary=None, consensus='rejected')
    result = check(case, document)
    assert not result['review_handoff_ready'] and not result['full_truth_ready']


def test_disagreement_requires_second_human_and_remains_diagnostic(case):
    document = full_review(case)
    event = document['clips'][0]['events'][0]
    event.update(consensus='disputed', disagreement='Synthetic reviewers disagree')
    event['secondary']['decision'] = 'rejected'
    document['clips'][0]['disagreements'] = ['Synthetic reviewers disagree']
    result = check(case, document)
    assert result['review_handoff_ready'] and not result['full_truth_ready']


def test_separate_events_survive_equal_rounded_display_times(case):
    document = full_review(case)
    first = document['clips'][0]
    first['events'] = [event_for(first, peak=1.2501), event_for(first, peak=1.2502)]
    before = copy.deepcopy(first['events'])
    assert round(before[0]['t_peak'], 2) == round(before[1]['t_peak'], 2)
    assert check(case, document)['full_truth_ready']
    assert first['events'] == before and len(first['events']) == 2


def test_duplicate_event_uuid_rejected(case):
    document = full_review(case)
    first = document['clips'][0]
    first['events'].append(copy.deepcopy(first['events'][0]))
    with pytest.raises(ValueError):
        check(case, document)


@pytest.mark.parametrize('aliases', [
    ['X-101', 'Х-101'], ['X-101', 'x-101.mov'], ['Й-101', 'И\u0306-101'],
])
def test_pilot_alias_and_nfc_collisions_rejected(make_case, aliases):
    candidate = make_case()
    save_json(candidate.id_path, aliases + candidate.ids[2:])
    with pytest.raises(ValueError):
        pilot.load_pilots(candidate.id_path)


@pytest.mark.parametrize('count', [0, 7, 9])
def test_exactly_eight_unique_pilots_required(make_case, count):
    candidate = make_case()
    save_json(candidate.id_path, [f'X-{i}' for i in range(count)])
    with pytest.raises(ValueError):
        pilot.load_pilots(candidate.id_path)


def test_video_namespace_collision_rejected_before_choosing_alias(make_case):
    candidate = make_case()
    (candidate.videos / 'Х-101.mov').write_bytes(b'distinct synthetic alias bytes')
    with pytest.raises(ValueError):
        pilot.build_inventory(candidate.id_path, candidate.videos, candidate.labels, ffprobe=sys.executable)


def test_source_labels_and_filled_sidecars_unchanged_on_resume(case):
    source_before = case.labels.read_bytes()
    save_json(case.adjudication_path, full_review(case))
    inventory_before = case.inventory_path.read_bytes()
    adjudication_before = case.adjudication_path.read_bytes()
    pilot.initialize(case.id_path, case.videos, case.labels, ffprobe=sys.executable)
    assert case.labels.read_bytes() == source_before
    assert case.inventory_path.read_bytes() == inventory_before
    assert case.adjudication_path.read_bytes() == adjudication_before


def test_actual_source_byte_mutation_rejected_without_rewriting_sidecars(case):
    before = case.adjudication_path.read_bytes(), case.inventory_path.read_bytes(), case.labels.read_bytes()
    original = case.paths[0].read_bytes()
    case.paths[0].write_bytes(bytes([original[0] ^ 1]) + original[1:])
    assert case.paths[0].stat().st_size == len(original)
    with pytest.raises(ValueError):
        pilot.validate_files()
    assert before == (case.adjudication_path.read_bytes(), case.inventory_path.read_bytes(), case.labels.read_bytes())


def test_inventory_adjudication_hash_binding(case):
    with pytest.raises(ValueError):
        schema.validate_documents(case.inventory, case.adjudication, '0' * 64, type_ids={TYPE_ID})
    document = full_review(case)
    document['clips'][0]['content_sha256'] = '0' * 64
    with pytest.raises(ValueError):
        check(case, document)


def test_output_directory_symlink_never_writes_outside(make_case, tmp_path):
    candidate = make_case()
    outside = tmp_path / 'outside'
    outside.mkdir()
    candidate.output.parent.mkdir(parents=True)
    candidate.output.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        pilot.initialize(candidate.id_path, candidate.videos, candidate.labels, ffprobe=sys.executable)
    assert not list(outside.iterdir())


def test_init_never_overwrites_an_input_used_as_inventory(make_case):
    candidate = make_case()
    candidate.output.mkdir(parents=True)
    input_path = candidate.output / 'inventory.json'
    save_json(input_path, candidate.ids)
    before = input_path.read_bytes()
    with pytest.raises(ValueError):
        pilot.initialize(input_path, candidate.videos, candidate.labels, ffprobe=sys.executable)
    assert input_path.read_bytes() == before


def test_unselected_files_are_not_hashed_or_probed(make_case, monkeypatch):
    candidate = make_case()
    unused = candidate.videos / 'X-999.mov'
    unused.write_bytes(b'unselected synthetic bytes')
    original_hash = pilot.sha256
    probed = []

    def guarded_hash(path):
        assert Path(path) != unused
        return original_hash(path)

    def probe(path, *args, **kwargs):
        assert Path(path) != unused
        probed.append(Path(path))
        return {'status': 'ok', 'duration_sec': 10.0}

    monkeypatch.setattr(pilot, 'sha256', guarded_hash)
    monkeypatch.setattr(pilot, '_probe', probe)
    inventory = pilot.build_inventory(candidate.id_path, candidate.videos, candidate.labels, ffprobe=sys.executable)
    assert len(inventory['clips']) == len(probed) == 8


def test_file_changed_during_mocked_probe_rejected_before_outputs(make_case, monkeypatch):
    candidate = make_case()

    def probe(path, *args, **kwargs):
        path.write_bytes(path.read_bytes() + b'changed')
        return {'status': 'ok', 'duration_sec': 10.0}

    monkeypatch.setattr(pilot, '_probe', probe)
    with pytest.raises(ValueError):
        pilot.initialize(candidate.id_path, candidate.videos, candidate.labels, ffprobe=sys.executable)
    assert not (candidate.output / 'inventory.json').exists()
    assert not (candidate.output / 'adjudication.json').exists()


@pytest.mark.parametrize('duration', [0.0, -1.0, float('nan'), True, '10'])
def test_probe_duration_requires_real_positive_numeric_value(make_case, monkeypatch, duration):
    candidate = make_case()
    monkeypatch.setattr(pilot, '_probe', lambda *args, **kwargs: {'status': 'ok', 'duration_sec': duration})
    with pytest.raises(ValueError):
        pilot.initialize(candidate.id_path, candidate.videos, candidate.labels, ffprobe=sys.executable)
    assert not (candidate.output / 'inventory.json').exists()


def test_partial_pair_recovers_only_missing_pending_adjudication(case):
    before = case.inventory_path.read_bytes()
    case.adjudication_path.unlink()
    result = pilot.initialize(case.id_path, case.videos, case.labels, ffprobe=sys.executable)
    assert result['resumed'] and case.inventory_path.read_bytes() == before
    pending = json.loads(case.adjudication_path.read_text())
    assert all(c['primary_review'] is None for c in pending['clips'])


def test_adjudication_without_inventory_is_never_replaced(case):
    save_json(case.adjudication_path, full_review(case))
    before = case.adjudication_path.read_bytes()
    case.inventory_path.unlink()
    with pytest.raises(ValueError):
        pilot.initialize(case.id_path, case.videos, case.labels, ffprobe=sys.executable)
    assert case.adjudication_path.read_bytes() == before and not case.inventory_path.exists()


def test_output_name_traversal_and_existing_human_file_rejected(case):
    before = case.adjudication_path.read_bytes()
    with pytest.raises(ValueError):
        pilot.write_new_json('../escape.json', {'synthetic': True})
    with pytest.raises(ValueError):
        pilot.write_new_json('adjudication.json', {'synthetic': True})
    assert case.adjudication_path.read_bytes() == before
    assert not (case.output.parent / 'escape.json').exists()


def test_relative_selection_path_cannot_escape_video_directory(make_case):
    candidate = make_case()
    save_json(candidate.id_path, {'pilots': [{'video': name, 'relative_path': '../' + name + '.mov'}
                                            for name in candidate.ids]})
    with pytest.raises(ValueError):
        pilot.build_inventory(candidate.id_path, candidate.videos, candidate.labels, ffprobe=sys.executable)


def test_reviewed_context_updates_without_rehashing_inventory_or_losing_reviews(case):
    document = full_review(case)
    before_reviews = copy.deepcopy(document['clips'][0]['events'])
    before_inventory, before_selection = case.inventory_path.read_bytes(), case.id_path.read_bytes()
    clip = document['clips'][0]
    clip['reviewed_context'] = {'angle_id': 'synthetic-front', 'source_group': 'synthetic-session',
                                'kind': 'human', 'reviewer': 'Synthetic reviewer A',
                                'date': '2026-10-17', 'source_reference': 'Synthetic local metadata review'}
    assert check(case, document)['full_truth_ready']
    effective = schema.effective_context(case.inventory['clips'][0], clip)
    assert effective['angle_id'] == 'synthetic-front' and effective['source_group'] == 'synthetic-session'
    save_json(case.adjudication_path, document)
    assert pilot.validate_files()['full_truth_ready']
    assert case.inventory_path.read_bytes() == before_inventory
    assert case.id_path.read_bytes() == before_selection
    assert clip['events'] == before_reviews and document['inventory_sha256'] == case.inventory_hash


def test_validation_detects_earlier_clip_changed_while_later_clip_is_hashed(case, monkeypatch):
    original_hash = pilot.sha256
    before = case.inventory_path.read_bytes(), case.adjudication_path.read_bytes()
    changed = []

    def mutate_earlier_after_later_hash(path):
        result = original_hash(path)
        if Path(path) == case.paths[1] and not changed:
            payload = case.paths[0].read_bytes()
            case.paths[0].write_bytes(bytes([payload[0] ^ 1]) + payload[1:])
            changed.append(True)
        return result

    monkeypatch.setattr(pilot, 'sha256', mutate_earlier_after_later_hash)
    with pytest.raises(ValueError):
        pilot.validate_files()
    assert changed and before == (case.inventory_path.read_bytes(), case.adjudication_path.read_bytes())


def test_pending_cli_summary_is_safe_and_require_complete_refuses(case, capsys):
    assert pilot.main(['summary']) == 0
    output = capsys.readouterr()
    assert 'review_handoff_ready' in output.out
    assert all(value not in output.out + output.err for value in case.ids)
    assert pilot.main(['validate', '--require-complete']) == 1
    output = capsys.readouterr()
    assert all(value not in output.out + output.err for value in case.ids)
    assert pilot.main(['--unknown-PRIVATE-SECRET']) == 1
    output = capsys.readouterr()
    assert 'PRIVATE-SECRET' not in output.out + output.err


def test_full_clip_negative_with_confirmed_event_never_ready(case):
    document = full_review(case)
    document['clips'][0]['clip_decision'] = 'no_violation'
    document['clips'][0]['secondary_review'] = human('Synthetic reviewer B')
    clarify_organizer(case, document['clips'][0])
    result = check(case, document)
    assert not result['review_handoff_ready'] and not result['full_truth_ready']


@pytest.mark.parametrize('field', ['t_start', 't_peak', 't_end'])
def test_missing_manual_event_time_is_diagnostic_not_full_truth(case, field):
    document = full_review(case)
    document['clips'][0]['events'][0][field] = None
    result = check(case, document)
    assert result['review_handoff_ready'] and not result['full_truth_ready']


def test_manual_event_time_order_and_end_boundary(case):
    document = full_review(case)
    event = document['clips'][0]['events'][0]
    event.update(t_start=9.0, t_peak=9.5, t_end=10.0)
    assert check(case, document)['full_truth_ready']
    event.update(t_start=1.5, t_peak=1.25, t_end=1.0)
    with pytest.raises(ValueError):
        check(case, document)


@pytest.mark.parametrize(('metadata', 'evidence'), [
    (None, 1.25), (None, True), (None, float('nan')),
    ({'kind': 'source_metadata', 'reference': 'Synthetic metadata', 'evidence_timestamp_sec': 1.25}, 1.5),
    ({'kind': 'source_metadata', 'reference': 'Synthetic metadata', 'evidence_timestamp_sec': True}, 1.0),
])
def test_evidence_timestamps_require_matching_numeric_source_metadata(case, metadata, evidence):
    document = full_review(case)
    event = document['clips'][0]['events'][0]
    event.update(source_metadata=metadata, evidence_timestamp_sec=evidence)
    with pytest.raises(ValueError):
        check(case, document)


def test_supported_evidence_timestamp_never_replaces_manual_event_time(case):
    document = full_review(case)
    event = document['clips'][0]['events'][0]
    event.update(source_metadata={'kind': 'source_metadata', 'reference': 'Synthetic metadata',
                                  'evidence_timestamp_sec': 0.75}, evidence_timestamp_sec=0.75)
    assert check(case, document)['full_truth_ready']
    assert event['t_peak'] == 1.25 and event['evidence_timestamp_sec'] == 0.75


def test_exact_duplicates_require_explicit_resolution(duplicate_case):
    document = full_review(duplicate_case)
    assert len(document['duplicate_decisions']) == 1
    result = check(duplicate_case, document)
    assert result['structurally_valid'] and not result['review_handoff_ready']
    group = document['duplicate_decisions'][0]
    group.update(consensus='unclear', reviewers=[human(), human('Synthetic reviewer B')],
                 note='Synthetic duplicate context remains unclear')
    result = check(duplicate_case, document)
    assert result['review_handoff_ready'] and not result['full_truth_ready']
    group.update(consensus='confirmed', note='Synthetic duplicate comparison completed')
    assert check(duplicate_case, document)['full_truth_ready']
    group['reviewers'] = [human()]
    assert not check(duplicate_case, document)['review_handoff_ready']


def test_duplicate_consensus_reviewers_must_be_distinct(duplicate_case):
    document = full_review(duplicate_case)
    document['duplicate_decisions'][0].update(
        consensus='confirmed', reviewers=[human(), human('synthetic REVIEWER a')], note='Synthetic resolution')
    with pytest.raises(ValueError):
        check(duplicate_case, document)


def test_pending_duplicate_resolution_can_leave_new_event_links_incomplete(duplicate_case):
    document = full_review(duplicate_case)
    group = document['duplicate_decisions'][0]
    links = []
    for key in group['video_keys']:
        clip = next(c for c in document['clips'] if c['video_key'] == key)
        clip.update(clip_decision='has_confirmed_event', events=[event_for(clip)])
        clarify_organizer(duplicate_case, clip)
        links.append({'video_key': key, 'event_id': clip['events'][0]['event_id']})
    assert group['consensus'] == 'pending' and group['event_links'] == []
    result = check(duplicate_case, document)
    assert result['structurally_valid'] and not result['review_handoff_ready']
    group.update(consensus='confirmed', reviewers=[human(), human('Synthetic reviewer B')],
                 note='Synthetic explicit event linkage', event_links=links)
    assert check(duplicate_case, document)['full_truth_ready']
    group['event_links'] = []
    assert not check(duplicate_case, document)['review_handoff_ready']
    group['event_links'] = links + copy.deepcopy(links)
    with pytest.raises(ValueError):
        check(duplicate_case, document)


def test_duplicate_group_cannot_be_removed_or_reassigned(duplicate_case):
    document = full_review(duplicate_case)
    document['duplicate_decisions'] = []
    with pytest.raises(ValueError):
        check(duplicate_case, document)
    document = full_review(duplicate_case)
    document['duplicate_decisions'][0]['video_keys'] = [document['clips'][0]['video_key']]
    with pytest.raises(ValueError):
        check(duplicate_case, document)


def test_exact_duplicate_binary_contradiction_requires_explicit_unclear_resolution(duplicate_case):
    document = full_review(duplicate_case)
    group = document['duplicate_decisions'][0]
    positive_key, negative_key = group['video_keys']
    positive = next(c for c in document['clips'] if c['video_key'] == positive_key)
    negative = next(c for c in document['clips'] if c['video_key'] == negative_key)
    assert negative['clip_decision'] == 'no_violation' and negative['coverage'] == 'full_clip'
    positive.update(clip_decision='has_confirmed_event', events=[event_for(positive)])
    clarify_organizer(duplicate_case, positive)
    group.update(consensus='confirmed', reviewers=[human(), human('Synthetic reviewer B')],
                 note='Generic synthetic duplicate resolution',
                 event_links=[{'video_key': positive_key, 'event_id': positive['events'][0]['event_id']}])
    result = check(duplicate_case, document)
    assert not result['review_handoff_ready'] and not result['full_truth_ready']
    group.update(consensus='unclear', note='Synthetic identical bytes have conflicting human binary decisions')
    result = check(duplicate_case, document)
    assert result['review_handoff_ready'] and not result['full_truth_ready']


def test_organizer_negative_to_agreed_positive_needs_explicit_source_clarification(case):
    document = full_review(case)
    clip = document['clips'][1]
    clip.update(clip_decision='has_confirmed_event', events=[event_for(clip)])
    result = check(case, document)
    assert not result['review_handoff_ready'] and not result['full_truth_ready']
    clarify_organizer(case, clip)
    result = check(case, document)
    assert result['review_handoff_ready'] and result['full_truth_ready']


def test_organizer_positive_to_reviewed_negative_needs_two_humans_and_source_clarification(case):
    document = full_review(case)
    clip = document['clips'][0]
    clip.update(clip_decision='no_violation', events=[], secondary_review=human('Synthetic reviewer B'))
    result = check(case, document)
    assert not result['review_handoff_ready'] and not result['full_truth_ready']
    clarify_organizer(case, clip)
    assert check(case, document)['full_truth_ready']
    clip['secondary_review'] = None
    result = check(case, document)
    assert not result['review_handoff_ready'] and not result['full_truth_ready']


@pytest.mark.parametrize('content_kind', ['bad_zip', 'bad_xml'])
def test_malformed_xlsx_cli_failure_never_echoes_private_path_or_traceback(make_case, capsys, content_kind):
    import zipfile
    candidate = make_case()
    malformed = candidate.root / 'fixtures/PRIVATE-PATH-SENTINEL.xlsx'
    if content_kind == 'bad_zip':
        malformed.write_bytes(b'not a ZIP workbook')
    else:
        with zipfile.ZipFile(malformed, 'w') as archive:
            archive.writestr('xl/workbook.xml', '<invalid-xml')
    result = pilot.main(['init', '--pilot-ids', str(candidate.id_path),
                         '--videos-dir', str(candidate.videos), '--labels', str(malformed),
                         '--ffprobe', sys.executable])
    assert result == 1
    captured = capsys.readouterr()
    assert 'SENTINEL' not in captured.out + captured.err
    assert str(malformed) not in captured.out + captured.err
    assert 'Traceback' not in captured.out + captured.err
    assert not (candidate.output / 'inventory.json').exists()
    assert not (candidate.output / 'adjudication.json').exists()


def test_inventory_uses_shared_local_protocol_numeric_probe(make_case, monkeypatch):
    import ml.eval as evaluator

    candidate = make_case()
    monkeypatch.setattr(pilot, '_probe', evaluator._probe)
    calls = []

    def local_metadata(command, **kwargs):
        assert command[command.index('-protocol_whitelist') + 1] == 'file,pipe'
        assert '-show_frames' not in command and '-show_packets' not in command
        assert 'tags' not in command[command.index('-show_entries') + 1]
        calls.append(command[-1])
        return SimpleNamespace(returncode=0, stdout=json.dumps({
            'format': {'duration': '10', 'size': '32'},
            'streams': [{'codec_type': 'video', 'width': 16, 'height': 16}],
        }).encode())

    monkeypatch.setattr(evaluator.subprocess, 'run', local_metadata)
    inventory = pilot.build_inventory(candidate.id_path, candidate.videos,
                                      candidate.labels, ffprobe=sys.executable)
    assert len(inventory['clips']) == 8
    assert set(calls) == {str(path) for path in candidate.paths}
