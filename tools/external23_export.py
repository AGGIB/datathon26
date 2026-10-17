#!/usr/bin/env python3
"""Export reviewed selected benchmark labels; pending clips never become negatives."""
import csv
from pathlib import Path
import re
import sys
import tempfile

from external23_annotations import validate_annotations
from external23_common import SafeParser, active_path, local_path, read_json, require, sha256, write_json
from external23_policy import selected_record, verify_proof

REPO = Path(__file__).resolve().parents[1]
PRIVATE = REPO / 'data/external23'
ANNOTATION_FIELDS = ['video', 'type_id', 't_start', 't_end', 't_peak', 'actor',
                     'what_visible', 'angle_id', 'comment']
LABEL_FIELDS = ['video', 'timestamp', 'label', 'content_sha256']


def csv_file(path, fields, rows):
    def safe(value):
        return "'" + value if isinstance(value, str) and value.lstrip().startswith(('=', '+', '-', '@')) else value
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', newline='', dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: safe(row.get(k, '')) for k in fields} for row in rows)
    temp.replace(path)


def export_selected(root, index, annotations, output_dir='exports', split_json=False):
    # External-only material has no valid train set. Never fabricate evaluator split.json.
    require(not split_json, 'EVALUATOR_TRAIN_INPUTS_REQUIRED')
    sys.path.insert(0, str(REPO))
    from ml.labels import normalize_video_name
    root = Path(root).absolute()
    validate_annotations(annotations, index['records'])
    drafts = {row['clip_id']: row for row in annotations['clips']}
    selected = [r for r in index['records'] if r.get('selection') == 'selected'
                and drafts[r['clip_id']].get('annotation_status') == 'reviewed_human'
                and drafts[r['clip_id']].get('legal_gold') is not None]
    minimum = index.get('manifest_snapshot', {}).get('selection_policy', {}).get('min_source_groups', 2)
    require(type(minimum) is int and minimum >= 2, 'MIN_SOURCE_GROUPS_INVALID')
    require(not selected or len({r['source_group'] for r in selected}) >= minimum,
            'INDEPENDENT_SOURCE_GROUPS_REQUIRED')
    annotations_out, labels, items = [], [], []
    seen_ids, seen_hashes = set(), set()
    for record in selected:
        require(re.fullmatch(r'ext23-[a-z0-9][a-z0-9_-]{0,73}', record['clip_id']), 'CLIP_ID_NAMESPACE')
        selected_record(record)  # Same rights/geography/viewpoint gate as intake.
        verify_proof(root, record['rights_proof'])
        path = active_path(root, record['clip_path'])
        require(path.is_file() and re.fullmatch('[0-9a-f]{64}', record.get('clip_sha256', ''))
                and sha256(path) == record['clip_sha256'], 'EXPORT_CLIP_SHA256_MISMATCH')
        video = path.name
        video_id = normalize_video_name(video)
        require(video_id.startswith('EXT23-') and video_id not in seen_ids, 'EXPORT_VIDEO_ID_COLLISION')
        require(record['clip_sha256'] not in seen_hashes, 'EXPORT_DUPLICATE_CLIP_CONTENT')
        seen_ids.add(video_id)
        seen_hashes.add(record['clip_sha256'])
        draft = drafts[record['clip_id']]
        positive = draft['legal_gold']['verdict'] == 'violation'
        events = draft['events'] if positive else []
        peak = min(event['t_peak'] for event in events) if events else ''
        labels.append({'video': video, 'timestamp': peak,
                       'label': 'нарушение' if positive else 'без нарушения',
                       'content_sha256': record['clip_sha256']})
        if not positive:
            annotations_out.append({'video': video, 'type_id': 'none', 'angle_id': record['angle_id'],
                                    'comment': 'reviewed_human; full_clip_reviewed; indicator_benchmark'})
        for event in events:
            annotations_out.append({'video': video, 'type_id': event['type_id'],
                                    't_start': event['t_start'], 't_end': event['t_end'],
                                    't_peak': event['t_peak'], 'actor': event.get('actor', 'unknown'),
                                    'what_visible': event['observations'], 'angle_id': record['angle_id'],
                                    'comment': 'reviewed_human; indicator_benchmark'})
        items.append({'video_id': video_id, 'angle_id': record['angle_id'],
                      'content_sha256': record['clip_sha256'], 'source_group': record['source_group'],
                      'split': 'external_holdout'})
    output = local_path(root, output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name in ('annotations.csv', 'labels.csv', 'external_holdout.json'):
        local_path(root, output / name)
    csv_file(output / 'annotations.csv', ANNOTATION_FIELDS, annotations_out)
    csv_file(output / 'labels.csv', LABEL_FIELDS, labels)
    write_json(output / 'external_holdout.json', {
        'schema_version': 1, 'kind': 'external_holdout_manifest', 'ready_for_evaluator': False,
        'requires': 'merge with reviewed base train labels/split; validate angle/hash/ID isolation with PR30 evaluator',
        'items': items})
    return {'exported_clips': len(selected), 'annotation_rows': len(annotations_out),
            'negative_clips': sum(row['label'] == 'без нарушения' for row in labels), 'evaluator_split_written': False}


def main():
    parser = SafeParser(description=__doc__)
    parser.add_argument('--index', default='intake.json')
    parser.add_argument('--annotations', default='annotations.json')
    parser.add_argument('--output-dir', default='exports')
    parser.add_argument('--split-json', action='store_true', help='Refused: reviewed base train inputs are required')
    try:
        args = parser.parse_args()
        for value in (args.index, args.annotations, args.output_dir):
            path = local_path(PRIVATE.absolute(), value)
            require(not {'excluded', 'quarantine'} & set(path.relative_to(PRIVATE).parts), 'EXCLUDED_INPUT_OR_OUTPUT')
        summary = export_selected(PRIVATE.absolute(), read_json(local_path(PRIVATE, args.index)),
                                  read_json(local_path(PRIVATE, args.annotations)), args.output_dir, args.split_json)
    except (OSError, ValueError, KeyError, TypeError):
        print('ERROR: external23_export_invalid', file=sys.stderr)
        return 1
    print('OK: exported_clips={exported_clips} annotation_rows={annotation_rows} negative_clips={negative_clips}'.format(**summary))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
