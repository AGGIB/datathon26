#!/usr/bin/env python3
"""Private three-action human annotation scaffold. No models, frames or labels."""
import csv
import json
import os
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ml.eval import _audit_directory, _probe  # noqa: E402
from ml.evaluation_labels import validate_content_sha256, video_key  # noqa: E402
from tools.audit_dataset import file_identity, json_object, sha256  # noqa: E402
from tools.pilot_review import SafeParser, guarded_json, local_file, selected_paths  # noqa: E402
from tools.pilot_review_schema import require  # noqa: E402
from tools.frontier_review_schema import (ADJUDICATION_FIELDS, FRONTIER, REVIEW_FIELDS,  # noqa: E402
                                         draft_rows, validate_documents)


def build_manifest(videos_dir, hash_report, source_revision, ffprobe=None):
    require(isinstance(source_revision, str) and re.fullmatch('[0-9a-f]{40}', source_revision), 'SOURCE_REVISION')
    report_path = local_file(hash_report)
    require(report_path.stat().st_size <= 64 * 1024 * 1024, 'HASH_REPORT_TOO_LARGE')
    document, report_sha = guarded_json(report_path)
    require(isinstance(document, dict), 'HASH_REPORT_SCHEMA')
    metadata = document.get('content_aggregation', {})
    require(metadata.get('hash_source') == 'actual' and metadata.get('hashes_authoritatively_verified') is True,
            'AUTHORITATIVE_HASH_REPORT_REQUIRED')
    groups, members, contents = document.get('per_content_group'), {}, {}
    require(isinstance(groups, list) and groups, 'HASH_GROUPS_REQUIRED')
    for group in groups:
        digest = validate_content_sha256(group['content_sha256'])
        require(digest not in contents and isinstance(group.get('videos'), list) and group['videos'], 'HASH_GROUP_DUPLICATE')
        contents[digest] = []
        for alias in group['videos']:
            key = video_key(alias)
            require(key not in members, 'ALIAS_DUPLICATE')
            members[key] = digest
    pilots = [{'video': key} for key in sorted(members)]
    directory, paths = selected_paths(videos_dir, pilots)
    executable = shutil.which(ffprobe or 'ffprobe')
    require(executable is not None, 'FFPROBE_REQUIRED')
    identities = {}
    for path in paths:
        key = video_key(path.name)
        before = file_identity(path)
        digest = sha256(path)
        require(digest == members[key], 'ACTUAL_SOURCE_HASH_MISMATCH')
        identities[path] = before
        contents[digest].append({'video_key': key, 'path': str(path),
                                'relative_path': path.relative_to(directory).as_posix(),
                                'content_sha256': digest, 'size_bytes': path.stat().st_size})
    rows = []
    for digest, aliases in sorted(contents.items()):
        aliases.sort(key=lambda row: row['video_key'])
        numeric = _probe(Path(aliases[0]['path']), executable)
        duration = numeric.get('duration_sec')
        require(numeric.get('status') == 'ok' and type(duration) in (int, float)
                and 0 < duration < float('inf'), 'SOURCE_DURATION_REQUIRED')
        rows.append({'content_sha256': digest, 'duration_sec': duration, 'aliases': aliases,
            'metadata_provenance': {'kind': 'ffprobe_numeric', 'source_content_sha256': digest},
            'camera_id': None, 'angle_id': None, 'session_id': None, 'source_group': None})
    require(all(file_identity(path) == before for path, before in identities.items())
            and sha256(report_path) == report_sha, 'SOURCE_CHANGED_DURING_INIT')
    return {'schema_version': 1, 'kind': 'frontier_review_manifest', 'usage': 'development',
        'source_revision': source_revision, 'scaffold_sha256': sha256(Path(__file__)),
        'schema_sha256': sha256(Path(__file__).with_name('frontier_review_schema.py')),
        'videos_root': str(directory), 'hash_report': {'path': str(report_path), 'sha256': report_sha},
        'frontier': FRONTIER, 'timebase': 'human_video_relative_seconds_not_verified_container_PTS',
        'contents': rows}


def _write(path, data):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8', newline='') as stream:
        stream.write(data)


def _write_csv(path, fields, rows):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def _csv(path, fields):
    path = local_file(path)
    require(path.stat().st_size <= 16 * 1024 * 1024, 'REVIEW_FILE_TOO_LARGE')
    with path.open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == fields, 'CSV_HEADER')
        return list(reader)


def _episodes(path):
    path = local_file(path)
    require(path.stat().st_size <= 16 * 1024 * 1024, 'REVIEW_FILE_TOO_LARGE')
    def nonfinite(value):
        raise ValueError('NONFINITE_EPISODE')
    return [json.loads(line, object_pairs_hook=json_object, parse_constant=nonfinite)
            for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def verify_sources(manifest, ffprobe=None):
    require(manifest['scaffold_sha256'] == sha256(Path(__file__))
            and manifest['schema_sha256'] == sha256(Path(__file__).with_name('frontier_review_schema.py')),
            'SCAFFOLD_OR_SCHEMA_CHANGED')
    report, digest = guarded_json(local_file(manifest['hash_report']['path']))
    require(digest == manifest['hash_report']['sha256'], 'HASH_REPORT_CHANGED')
    metadata = report['content_aggregation']
    require(metadata['hash_source'] == 'actual' and metadata['hashes_authoritatively_verified'] is True,
            'HASH_REPORT_NOT_ACTUAL')
    expected = {video_key(name): row['content_sha256'] for row in report['per_content_group'] for name in row['videos']}
    seen = {}
    root = _audit_directory(manifest['videos_root'])
    for content in manifest['contents']:
        for alias in content['aliases']:
            path = local_file(alias['path'])
            require(path.is_relative_to(root) and path.relative_to(root).as_posix() == alias['relative_path']
                    and video_key(path.name) == alias['video_key'] and alias['video_key'] not in seen,
                    'SOURCE_PATH_BINDING')
            require(path.stat().st_size == alias['size_bytes'] and sha256(path) == content['content_sha256']
                    == alias['content_sha256'], 'SOURCE_HASH_CHANGED')
            seen[alias['video_key']] = content['content_sha256']
    require(seen == expected, 'MANIFEST_HASH_COVERAGE')
    # The adjacent checksum detects edits, but is not an independent authority.
    # Re-derive duration and all generated metadata from the bound source bytes.
    regenerated = build_manifest(root, manifest['hash_report']['path'], manifest['source_revision'], ffprobe)
    require(manifest == regenerated, 'SOURCE_METADATA_CHANGED')


def validate_files(bundle, require_complete=False, *, ffprobe=None):
    bundle = _audit_directory(bundle)
    manifest, digest = guarded_json(local_file(bundle / 'manifest.json'))
    require(local_file(bundle / 'manifest.sha256').read_text().strip() == digest, 'MANIFEST_BINDING_CHANGED')
    paths = [bundle / 'manifest.json', bundle / 'manifest.sha256', bundle / 'adjudication.csv', bundle / 'adjudicated-episodes.jsonl']
    paths += [bundle / name / file for name in ('reviewer-a', 'reviewer-b') for file in ('coverage.csv', 'episodes.jsonl')]
    identities = {path: file_identity(local_file(path)) for path in paths}
    a, b = [_csv(bundle / name / 'coverage.csv', REVIEW_FIELDS) for name in ('reviewer-a', 'reviewer-b')]
    ea, eb = [_episodes(bundle / name / 'episodes.jsonl') for name in ('reviewer-a', 'reviewer-b')]
    summary = validate_documents(manifest, a, ea, b, eb, _csv(bundle / 'adjudication.csv', ADJUDICATION_FIELDS),
                                 _episodes(bundle / 'adjudicated-episodes.jsonl'))
    verify_sources(manifest, ffprobe)
    require(all(file_identity(path) == before for path, before in identities.items()), 'REVIEW_INPUT_CHANGED')
    if require_complete:
        require(summary['human_handoff_ready'], 'HUMAN_REVIEW_INCOMPLETE')
    return summary


def initialize(output, videos_dir, hash_report, source_revision, *, artifact_root=None, ffprobe=None, viewer=None):
    base = Path(artifact_root or ROOT / 'var').absolute()
    output = Path(output).absolute()
    require('..' not in output.parts and base == base.resolve() and output.is_relative_to(base)
            and output != base and not output.exists() and not output.is_symlink(), 'FRESH_PRIVATE_OUTPUT_REQUIRED')
    base.mkdir(parents=True, exist_ok=True)
    _audit_directory(base)
    parent = output.parent
    require(parent == base, 'OUTPUT_MUST_BE_DIRECT_CHILD_OF_ARTIFACT_ROOT')
    if viewer is not None:
        viewer = local_file(viewer)
    manifest = build_manifest(videos_dir, hash_report, source_revision, ffprobe)
    rows = draft_rows(manifest)
    adj = [{'slot_id': row['slot_id'], 'consensus': 'pending', 'reviewer': '', 'date': '', 'disagreement_code': ''}
           for row in rows]
    validate_documents(manifest, rows, [], rows, [], adj)
    output.mkdir(mode=0o700)
    _write(output / 'manifest.json', json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    _write(output / 'manifest.sha256', sha256(output / 'manifest.json') + '\n')
    for reviewer in ('reviewer-a', 'reviewer-b'):
        (output / reviewer).mkdir(mode=0o700)
        _write_csv(output / reviewer / 'coverage.csv', REVIEW_FIELDS, rows)
        _write(output / reviewer / 'episodes.jsonl', '')
    _write_csv(output / 'adjudication.csv', ADJUDICATION_FIELDS, adj)
    _write(output / 'adjudicated-episodes.jsonl', '')
    viewer_link = f'[Existing local video/angle viewer]({viewer})\n' if viewer else ''
    _write(output / 'README.md', f'''# Private frontier annotation draft

{len(manifest['contents'])} development contents, {sum(len(c['aliases']) for c in manifest['contents'])} aliases,
{len(rows)} unique content/type coverage slots, two independent human passes.
Every row is UNREVIEWED. Empty episodes do not mean a negative. No labels or
model proposals were used to create this bundle. All camera/source/session IDs
remain unknown; hashes identify duplicate bytes, not filming sessions.

{viewer_link}
Use the existing local viewer/player and original source paths in coverage.csv.
Reviewer A and B each fill only their own folder without reading the other's
answers. Actual independence/viewing cannot be established by the validator.
Do not copy organizer/VLM answers. Fill actual human name/date only after work.
For each of the three types inspect the full clip or record exact partial bounds;
unreviewed remainder is not background. Record all positive/contrast/unknown
episodes separately. Unknown is a reviewed uncertainty, not UNREVIEWED.

Read tools/frontier_review_schema.py for fields/enums and docs/frontier-human-review.md
for the episode template. Keep times video-relative; do not claim actual PTS.
After both reviews, retain originals and fill adjudication.csv, recording any
disagreement. Consolidated episodes go in adjudicated-episodes.jsonl with links
to the original review episode IDs and bounds covered by BOTH reviewers. This scaffold does not export certified evaluation truth or prove
97% precision, new cameras, legal violations, or complete labels for other types.

Validation (local metadata/hash reads only):
python "{Path(__file__).resolve()}" validate --bundle "{output}"
python "{Path(__file__).resolve()}" validate --bundle "{output}" --require-complete

Only numeric counts/errors are printed. All filled files stay private/ignored.
''')
    return validate_files(output, ffprobe=ffprobe)


def main(argv=None):
    parser = SafeParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True, parser_class=SafeParser)
    init = sub.add_parser('init')
    for key in ('output', 'videos-dir', 'hash-report', 'source-revision'):
        init.add_argument('--' + key, required=True)
    init.add_argument('--artifact-root')
    init.add_argument('--ffprobe')
    init.add_argument('--viewer')
    check = sub.add_parser('validate')
    check.add_argument('--bundle', required=True)
    check.add_argument('--require-complete', action='store_true')
    check.add_argument('--ffprobe')
    try:
        args = parser.parse_args(argv)
        if args.command == 'init':
            result = initialize(args.output, args.videos_dir, args.hash_report, args.source_revision,
                                artifact_root=args.artifact_root, ffprobe=args.ffprobe, viewer=args.viewer)
        else:
            result = validate_files(args.bundle, args.require_complete, ffprobe=args.ffprobe)
    except (OSError, ValueError, TypeError, KeyError, OverflowError, csv.Error):
        print('{"status":"invalid_or_incomplete_private_review"}')
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
