#!/usr/bin/env python3
"""Synthetic checks only; creates test media locally, never downloads or reads dataset videos."""

import copy
import json
from pathlib import Path
import subprocess
import tempfile

from external23_annotations import draft_annotations, validate_annotations
from external23_intake import import_manifest, local_path, read_json, sha256, validate_manifest, write_json


def rejected(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError('invalid synthetic input accepted')


def fixture():
    return {
        'schema_version': 1,
        'sources': [{
            'id': 'synthetic', 'source_url': 'https://example.invalid/synthetic.mp4',
            'source_page_url': 'https://example.invalid/synthetic',
            'license': {'id': 'SYNTHETIC_TEST_ONLY', 'url': 'https://example.invalid/license'},
            'attribution': 'Locally generated synthetic test', 'source_group': 'synthetic-scene',
            'camera_id': 'synthetic-camera', 'source_context': {
                'country': None, 'scene_type': 'bank_cashdesk', 'evidence_reference': 'synthetic-context-only'},
            'local_path': 'inputs/synthetic.mp4',
        }],
        'clips': [{'id': f'synthetic-{index:02}', 'source_id': 'synthetic', 'start_sec': None, 'end_sec': None}
                  for index in range(10)],
    }


def main():
    valid = fixture()
    validate_manifest(valid)
    for change in ['count', 'duplicate', 'missing_license', 'http', 'nan', 'path_id', 'bool_schema']:
        doc = copy.deepcopy(valid)
        if change == 'count':
            doc['clips'].pop()
        elif change == 'duplicate':
            doc['clips'][1]['id'] = doc['clips'][0]['id']
        elif change == 'missing_license':
            doc['sources'][0].pop('license')
        elif change == 'http':
            doc['sources'][0]['source_url'] = 'http://example.invalid/synthetic.mp4'
        elif change == 'nan':
            doc['clips'][0].update(start_sec=0, end_sec=float('nan'))
        elif change == 'path_id':
            doc['clips'][0]['id'] = '../../escape'
        else:
            doc['schema_version'] = True
        rejected(lambda: validate_manifest(doc))

    with tempfile.TemporaryDirectory(prefix='external23-synthetic-') as temp:
        root = Path(temp)
        rejected(lambda: local_path(root, '../escape'))
        (root / 'outside-link').symlink_to(root.parent, target_is_directory=True)
        rejected(lambda: local_path(root, 'outside-link/escape'))
        (root / 'inputs').mkdir()
        source = root / 'inputs/synthetic.mp4'
        subprocess.run(['/opt/homebrew/bin/ffmpeg', '-nostdin', '-v', 'error', '-f', 'lavfi',
                        '-i', 'color=c=black:s=64x64:r=10:d=1', '-metadata', 'comment=PRIVATE_TAG_MUST_NOT_LEAK',
                        '-an', '-c:v', 'libx264', str(source)], capture_output=True, check=True)
        valid['sources'][0]['expected_sha256'] = sha256(source)
        manifest = root / 'manifest.json'
        write_json(manifest, valid)
        index = import_manifest(root, manifest, valid, False, '/opt/homebrew/bin/ffprobe',
                                '/opt/homebrew/bin/ffmpeg', 1024 ** 2)
        assert len(index['records']) == 10
        assert len({row['source_group'] for row in index['records']}) == 1
        assert all(row['legal_gold'] is None for row in index['records'])
        assert all('PRIVATE_TAG_MUST_NOT_LEAK' not in json.dumps(row) for row in index['records'])
        annotations = read_json(root / 'annotations.json')
        assert all(not row['events'] and row['legal_gold'] is None for row in annotations['clips'])
        before = (root / 'annotations.json').read_bytes()
        repeated = import_manifest(root, manifest, valid, False, '/opt/homebrew/bin/ffprobe',
                                   '/opt/homebrew/bin/ffmpeg', 1024 ** 2)
        assert len(repeated['records']) == 10 and (root / 'annotations.json').read_bytes() == before
        bad_hash = copy.deepcopy(valid)
        bad_hash['sources'][0]['expected_sha256'] = '0' * 64
        write_json(manifest, bad_hash)
        rejected(lambda: import_manifest(root, manifest, bad_hash, False, '/opt/homebrew/bin/ffprobe',
                                         '/opt/homebrew/bin/ffmpeg', 1024 ** 2))

        records = index['records']
        draft = draft_annotations(records)
        validate_annotations(draft, records)
        for provenance in ['source_human', 'local_model', 'reviewed_human']:
            annotated = copy.deepcopy(draft)
            origin = {'reference': 'synthetic-only'}
            if provenance == 'source_human':
                origin['original_source_interval'] = {'start': 0, 'end': 0.5, 'unit': 'seconds'}
            elif provenance == 'local_model':
                origin.update(model_id='synthetic-model', version='synthetic-v1')
            else:
                origin.update(reviewer='synthetic-reviewer', reviewed_on='2026-10-17')
            annotated['clips'][0]['events'] = [{'provenance': provenance, 't_start': 0, 't_end': 0.5,
                                               'action_label': 'synthetic_action', 'roles': [], 'origin': origin}]
            validate_annotations(annotated, records)
        unreviewed = copy.deepcopy(draft)
        unreviewed['clips'][0]['legal_gold'] = {'verdict': 'no_violation', 'type_ids': []}
        rejected(lambda: validate_annotations(unreviewed, records))
        unknown_records = copy.deepcopy(records)
        unknown_records[0]['source_context']['scene_type'] = 'unknown'
        forged = copy.deepcopy(draft)
        forged['clips'][0].update(annotation_status='reviewed_human', reviewer='synthetic-reviewer',
                                  reviewed_on='2026-10-17', legal_gold={'verdict': 'violation', 'type_ids': ['synthetic']},
                                  applicability={'jurisdiction': 'KZ', 'scene_type': 'bank_cashdesk',
                                                 'roles_established': True, 'normative_conditions_established': True,
                                                 'evidence_references': ['synthetic-only']})
        rejected(lambda: validate_annotations(forged, unknown_records))
        simulated = copy.deepcopy(forged)
        simulated['clips'][0]['simulation_context'] = {
            'basis': 'scenario_assumption', 'jurisdiction': 'KZ', 'scene_type': 'bank_cashdesk',
            'decision_reference': 'synthetic-explicit-scenario-decision'}
        simulated['clips'][0]['legal_gold']['statement_scope'] = 'simulated_indicator_benchmark'
        rejected(lambda: validate_annotations(simulated, unknown_records))
        validate_annotations(forged, records)  # Country can be unknown; confirmed bank scene is required.
    print('OK: synthetic manifest/provenance/path/hash/ffprobe/import/resume checks; no downloads or dataset reads')


if __name__ == '__main__':
    main()
