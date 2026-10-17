"""Import a provided human analysis bundle locally, without interpreting media.

Descriptions are copied only into private artifacts. Stdout contains counts only.
No provided normative claim becomes a verified rule automatically.
"""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

from .eval import audit
from .evaluation_labels import load_labels, parse_timestamp, video_key


def _json(path):
    path = Path(path)
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError('analysis JSON exceeds the local import size limit')
    return json.loads(path.read_text(encoding='utf-8-sig'))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def import_analysis(analysis_dir, labels_path, videos_dir, output_dir):
    """Join registry to actual files by SHA256, not by repairing identifiers."""
    source = Path(analysis_dir)
    videos = Path(videos_dir)
    output = Path(output_dir).resolve()
    allowed = Path(__file__).resolve().parents[1] / 'artifacts'
    if not output.is_relative_to(allowed):
        raise ValueError('private import output must stay inside project artifacts/')
    data = _json(source / 'report_data.json')
    manifest = _json(source / 'manifest.json')
    labels = load_labels(labels_path)
    file_audit = audit(labels, videos, use_ffprobe=False)
    if any(row['file_status'] != 'matched' for row in file_audit['per_video']):
        raise ValueError('every source label requires exactly one literal local file')
    local_by_hash = {}
    for label, row in zip(labels, file_audit['per_video']):
        path = videos / row['matched_paths'][0]
        digest = _sha256(path)
        local_by_hash.setdefault(digest, []).append((label, path))
    by_source_id = {}
    for item in manifest:
        digest = item['sha256']
        if not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest):
            raise ValueError('source hash must be literal lowercase SHA256')
        if digest not in local_by_hash or item['id'] in by_source_id:
            raise ValueError('unknown source hash or duplicate source identifier')
        matches = local_by_hash[digest]
        if len(matches) > 1:
            source_name = video_key(item.get('name'))
            matches = [entry for entry in matches if video_key(entry[1].name) == source_name]
        if len(matches) != 1:
            raise ValueError('duplicate content requires an unambiguous literal source name')
        by_source_id[item['id']] = matches[0]
    if len(by_source_id) != len(labels) or len({entry[0]['video'] for entry in by_source_id.values()}) != len(labels):
        raise ValueError('manifest must map every local labeled video exactly once')
    sources = {row[0]: {'id': row[0], 'title': row[1], 'description': row[2], 'url': row[3]}
               for row in data['sources']}
    catalog = []
    codes = set()
    for row in data['types']:
        if len(row) != 8 or not all(isinstance(value, str) for value in row[:7]) or not isinstance(row[7], list):
            raise ValueError('unsupported provided taxonomy row schema')
        code, name, examples, visible, law, limits, use, refs = row
        if not code or code in codes or any(ref not in sources for ref in refs):
            raise ValueError('duplicate type code or unresolved source reference')
        codes.add(code)
        catalog.append({'source_type_code': code, 'type_id': code, 'name': name,
                        'source_examples': examples, 'visible_features': visible,
                        'normative_claim_unverified': law, 'limitations': limits,
                        'usage_notes': use, 'source_references': [sources[ref] for ref in refs],
                        'status': 'draft', 'verified': False, 'rule': None})
    annotations = {}
    baseline_manifest = []
    provenance = []
    for record in data['registry']:
        source_id = record['id']
        if source_id not in by_source_id:
            raise ValueError('registry references an unknown source identifier')
        label, path = by_source_id[source_id]
        key = video_key(label['video'])
        if key in annotations:
            raise ValueError('registry contains duplicate video assignments')
        flag = str(record['original_label']).strip().lower()
        if flag not in ('нарушение', 'без нарушения') or (flag == 'нарушение') != label['positive']:
            raise ValueError('provided registry contradicts organizer binary label')
        assessment = record.get('assessment')
        if label['positive']:
            if not isinstance(assessment, list) or len(assessment) != 2 or assessment[0] not in codes:
                raise ValueError('positive registry row requires an existing provided type code')
            peak = parse_timestamp(record['timestamp'])
            if label['timestamp'] is None or abs(peak - label['timestamp']) > 1e-6:
                raise ValueError('provided peak contradicts organizer timestamp')
            kind = assessment[0]
        else:
            if assessment is not None:
                raise ValueError('negative registry row unexpectedly has an assessment')
            peak, kind = None, 'none'
        observation = record.get('observation') or {}
        if not isinstance(observation, dict):
            raise ValueError('observation metadata must be an object')
        angle = record.get('angle_id') or observation.get('angle_id') or 'unknown'
        if not isinstance(angle, str) or not angle.strip() or angle != angle.strip():
            raise ValueError('explicit angle_id must be a nonempty literal string')
        annotations[key] = {'video': label['video'], 'type_id': kind,
                            't_start': '', 't_end': '', 't_peak': peak if peak is not None else '',
                            'actor': '', 'what_visible': observation.get('observation') or '',
                            'angle_id': angle,
                            'comment': 'Imported primary type only; normative sources unverified; intervals not inferred.'}
        baseline_manifest.append({'video_id': path.name, 'angle_id': angle,
                                  'content_sha256': _sha256(path),
                                  'label': int(label['positive']),
                                  'positive_times_s': [peak] if peak is not None else []})
        provenance.append({'source_id': source_id, 'source_name': record.get('name'),
                           'literal_local_video': path.name,
                           'provided_observed_interval': observation.get('observed_interval'),
                           'provided_assessment': assessment})
    if len(annotations) != len(labels):
        raise ValueError('registry must cover every labeled video')
    rows = [annotations[video_key(label['video'])] for label in labels]
    unknown = sum(row['angle_id'].lower() in ('unknown', 'неизвестно', 'unassigned') for row in rows)
    summary = {'video_count': len(rows), 'type_count': len(catalog),
               'unique_content_count': len(local_by_hash),
               'duplicate_content_groups': sum(len(group) > 1 for group in local_by_hash.values()),
               'source_count': len(sources), 'additional_check_count': len(data.get('additional_checks', [])),
               'typed_positive_count': sum(row['type_id'] != 'none' for row in rows),
               'negative_count': sum(row['type_id'] == 'none' for row in rows),
               'unknown_angle_count': unknown,
               'ready_for_angle_holdout': unknown == 0 and len({row['angle_id'] for row in rows}) >= 2,
               'normative_rules_verified': False, 'event_intervals_inferred': False,
               'join_method': 'exact_sha256', 'input_media_modified': False}
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'annotations.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for name, value in [('catalog.draft.json', catalog), ('baseline-manifest.draft.json', baseline_manifest),
                        ('provenance.json', provenance), ('summary.json', summary),
                        ('additional-checks.unverified.json', data.get('additional_checks', []))]:
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description='Private local import of a provided human taxonomy bundle.')
    parser.add_argument('--analysis-dir', required=True, type=Path)
    parser.add_argument('--labels', required=True, type=Path)
    parser.add_argument('--videos-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = import_analysis(args.analysis_dir, args.labels, args.videos_dir, args.output)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(type(error).__name__ + ': invalid or inconsistent local import input')
    print(json.dumps(result))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
