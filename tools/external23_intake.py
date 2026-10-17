#!/usr/bin/env python3
"""Local external-video intake. Stdout contains aggregate counts or fixed error codes only."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from urllib.parse import urlsplit

from external23_annotations import draft_annotations, validate_annotations

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / 'data' / 'external23'
ID = re.compile(r'[a-z0-9][a-z0-9_-]{0,79}')


def require(condition, code):
    if not condition:
        raise ValueError(code)


def local_path(root, value):
    root = root.resolve()
    path = Path(value)
    path = (path if path.is_absolute() else root / path).resolve()
    require(path.is_relative_to(root) and path != root, 'PATH_OUTSIDE_PRIVATE_ROOT')
    return path


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'DUPLICATE_JSON_KEY')
            result[key] = value
        return result
    return json.loads(path.read_text(), object_pairs_hook=unique)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as output:
        temp = Path(output.name)
        json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write('\n')
    temp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def text(value):
    return isinstance(value, str) and bool(value.strip())


def number(value):
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        return False


def https(value):
    require(text(value), 'SOURCE_URL_REQUIRED')
    parsed = urlsplit(value)
    require(parsed.scheme == 'https' and parsed.hostname and not parsed.username
            and not parsed.password, 'PUBLIC_HTTPS_URL_REQUIRED')


def validate_manifest(document, allow_partial=False):
    require(isinstance(document, dict) and type(document.get('schema_version')) is int
            and document['schema_version'] == 1, 'MANIFEST_SCHEMA')
    sources, clips = document.get('sources'), document.get('clips')
    require(isinstance(sources, list) and isinstance(clips, list), 'MANIFEST_LISTS')
    minimum = 1 if allow_partial else 10
    selected = sum(isinstance(clip, dict) and clip.get('selection', 'selected') == 'selected' for clip in clips)
    require((not sources and not clips) or 1 <= len(clips) <= 50 and selected <= 15
            and (allow_partial or minimum <= selected), 'REQUIRE_10_TO_15_CLIPS')
    indexed = {}
    for source in sources:
        require(isinstance(source, dict), 'SOURCE_SCHEMA')
        sid = source.get('id')
        require(isinstance(sid, str) and ID.fullmatch(sid) and sid not in indexed, 'SOURCE_ID')
        for field in ['source_url', 'source_page_url']:
            https(source.get(field))
        license_info = source.get('license')
        require(isinstance(license_info, dict) and text(license_info.get('id')), 'LICENSE_REQUIRED')
        https(license_info.get('url'))
        for field in ['attribution', 'source_group', 'camera_id']:
            require(text(source.get(field)), 'PROVENANCE_REQUIRED')
        expected = source.get('expected_sha256')
        require(expected is None or isinstance(expected, str) and re.fullmatch('[0-9a-f]{64}', expected),
                'EXPECTED_SHA256_INVALID')
        expected_sha1 = source.get('expected_sha1')
        require(expected_sha1 is None or isinstance(expected_sha1, str)
                and re.fullmatch('[0-9a-f]{40}', expected_sha1), 'EXPECTED_SHA1_INVALID')
        expected_size = source.get('expected_size_bytes')
        require(expected_size is None or type(expected_size) is int and expected_size > 0, 'EXPECTED_SIZE_INVALID')
        context = source.get('source_context')
        require(isinstance(context, dict) and 'country' in context and text(context.get('scene_type')),
                'SOURCE_CONTEXT_REQUIRED')
        require(context['country'] is None or text(context['country']), 'SOURCE_COUNTRY_INVALID')
        if 'local_path' in source:
            require(text(source['local_path']), 'LOCAL_SOURCE_PATH_INVALID')
        indexed[sid] = source
    ids = set()
    for clip in clips:
        require(isinstance(clip, dict), 'CLIP_SCHEMA')
        cid = clip.get('id')
        require(isinstance(cid, str) and ID.fullmatch(cid) and cid not in ids, 'CLIP_ID')
        ids.add(cid)
        require(clip.get('selection', 'selected') in {'selected', 'reserve', 'candidate'}, 'SELECTION_INVALID')
        require(clip.get('split', 'external_holdout') == 'external_holdout', 'EXTERNAL_HOLDOUT_ONLY')
        require(isinstance(clip.get('source_id'), str) and clip['source_id'] in indexed, 'CLIP_SOURCE_MISSING')
        if clip.get('selection', 'selected') == 'selected':
            context = indexed[clip['source_id']]['source_context']
            require(context.get('scene_type') == 'bank_cashdesk' and text(context.get('evidence_reference')),
                    'SELECTED_BANK_CONTEXT_NOT_CONFIRMED')
        require('start_sec' in clip and 'end_sec' in clip, 'SOURCE_OFFSETS_REQUIRED')
        start, end = clip['start_sec'], clip['end_sec']
        require(start is None and end is None or number(start) and number(end) and 0 <= start < end,
                'SOURCE_OFFSETS_INVALID')
    return indexed


def technical_probe(path, binary):
    command = [binary, '-v', 'error', '-show_entries',
               'format=duration,size:stream=codec_type,codec_name,width,height,avg_frame_rate',
               '-of', 'json', '-i', str(path)]
    result = subprocess.run(command, capture_output=True, timeout=120)
    require(result.returncode == 0, 'FFPROBE_FAILED')
    raw = json.loads(result.stdout)
    streams = [stream for stream in raw.get('streams', []) if stream.get('codec_type') == 'video']
    require(streams, 'VIDEO_STREAM_MISSING')
    stream = streams[0]
    duration = float(raw.get('format', {}).get('duration', 'nan'))
    require(number(duration) and duration > 0, 'VIDEO_DURATION_INVALID')
    width, height = stream.get('width'), stream.get('height')
    require(isinstance(width, int) and width > 0 and isinstance(height, int) and height > 0, 'VIDEO_SIZE_INVALID')
    codec = stream.get('codec_name', '')
    require(isinstance(codec, str) and re.fullmatch('[a-z0-9_]+', codec), 'VIDEO_CODEC_INVALID')
    rate = stream.get('avg_frame_rate', '0/0')
    require(isinstance(rate, str) and re.fullmatch(r'\d+/\d+', rate), 'VIDEO_FPS_INVALID')
    numerator, denominator = map(int, rate.split('/'))
    return {'duration_seconds': duration, 'width': width, 'height': height,
            'codec': codec, 'average_fps': numerator / denominator if denominator else None,
            'average_frame_rate': rate}


def download(source, destination, max_bytes):
    request = urllib.request.Request(source['source_url'], headers={'User-Agent': 'external23-local-intake/1'})
    partial = destination.with_suffix('.partial')
    require(not partial.exists(), 'PARTIAL_SOURCE_ALREADY_EXISTS')
    try:
        with urllib.request.urlopen(request, timeout=60) as response, partial.open('xb') as output:
            https(response.geturl())
            size = 0
            while block := response.read(1024 * 1024):
                size += len(block)
                require(size <= max_bytes, 'SOURCE_DOWNLOAD_LIMIT_EXCEEDED')
                require(shutil.disk_usage(destination.parent).free > 1024 ** 3, 'LOW_DISK_SPACE')
                output.write(block)
        require(size > 0, 'EMPTY_SOURCE')
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def make_clip(source_path, output_path, start, end, binary):
    partial = output_path.with_suffix('.partial.mp4')
    require(not partial.exists(), 'PARTIAL_CLIP_ALREADY_EXISTS')
    try:
        result = subprocess.run([binary, '-nostdin', '-v', 'error', '-n', '-i', str(source_path),
                                 '-ss', str(start), '-t', str(end - start), '-map', '0:v:0', '-an',
                                 '-map_metadata', '-1', '-c:v', 'libx264', '-preset', 'veryfast',
                                 '-crf', '20', '-movflags', '+faststart', str(partial)],
                                capture_output=True, timeout=1800)
        require(result.returncode == 0, 'FFMPEG_CLIP_FAILED')
        partial.replace(output_path)
    finally:
        partial.unlink(missing_ok=True)


def import_manifest(root, manifest_path, document, allow_download, ffprobe, ffmpeg, max_bytes, allow_partial=False):
    sources = validate_manifest(document, allow_partial)
    require(document['clips'], 'MANIFEST_PENDING_NO_CLIPS')
    root = root.resolve()
    for folder in ['sources', 'clips']:
        local_path(root, folder).mkdir(parents=True, exist_ok=True)
    index_path = local_path(root, 'intake.json')
    manifest_hash = sha256(manifest_path)
    index = read_json(index_path) if index_path.exists() else {
        'schema_version': 1, 'manifest_sha256': manifest_hash, 'manifest_snapshot': document,
        'records': [], 'status': 'importing'}
    if index.get('manifest_sha256') != manifest_hash:
        previous = index.get('manifest_snapshot', {})
        for field in ['sources', 'clips']:
            current = {row['id']: row for row in document[field]}
            def immutable(row):
                return {key: value for key, value in row.items() if key not in {'selection', 'split', 'selection_reason'}}
            require(all(row['id'] in current and immutable(current[row['id']]) == immutable(row)
                        for row in previous.get(field, [])),
                    'MANIFEST_CHANGED_EXISTING_RECORDS')
        index.setdefault('previous_manifest_sha256', []).append(index['manifest_sha256'])
        index['manifest_sha256'], index['manifest_snapshot'] = manifest_hash, document
    records = {record['clip_id']: record for record in index['records']}
    require(set(records) <= {clip['id'] for clip in document['clips']}, 'INTAKE_CLIP_SET_MISMATCH')
    cache = {}
    for clip in document['clips']:
        source = sources[clip['source_id']]
        if source['id'] not in cache:
            path = local_path(root, source.get('local_path', 'sources/' + source['id'] + '.source'))
            if not path.exists():
                require(allow_download and 'local_path' not in source, 'SOURCE_MISSING_DOWNLOAD_NOT_ENABLED')
                download(source, path, max_bytes)
            require(path.is_file() and path.stat().st_size <= max_bytes, 'SOURCE_FILE_LIMIT')
            require(not source.get('expected_size_bytes') or path.stat().st_size == source['expected_size_bytes'],
                    'SOURCE_PUBLISHED_SIZE_MISMATCH')
            digest = sha256(path)
            require(not source.get('expected_sha256') or digest == source['expected_sha256'], 'SOURCE_SHA256_MISMATCH')
            sha1 = hashlib.sha1()
            with path.open('rb') as original:
                for block in iter(lambda: original.read(1024 * 1024), b''):
                    sha1.update(block)
            require(not source.get('expected_sha1') or sha1.hexdigest() == source['expected_sha1'],
                    'SOURCE_PUBLISHED_SHA1_MISMATCH')
            proof = source.get('rights_proof')
            if proof:
                require(sha256(local_path(root, proof['path'])) == proof['sha256'], 'RIGHTS_PROOF_SHA256_MISMATCH')
            cache[source['id']] = (path, digest, sha1.hexdigest(), technical_probe(path, ffprobe))
        path, source_hash, source_sha1, source_probe = cache[source['id']]
        start, end = clip['start_sec'], clip['end_sec']
        whole_source = start is None
        if whole_source:
            start, end = 0.0, source_probe['duration_seconds']
        require(end <= source_probe['duration_seconds'], 'CLIP_EXCEEDS_SOURCE_DURATION')
        output = local_path(root, 'clips/' + clip['id'] + '.mp4')
        if output.exists():
            old = records.get(clip['id'])
            require(old and old['source_sha256'] == source_hash and old['clip_sha256'] == sha256(output),
                    'EXISTING_CLIP_NOT_MATCHED_TO_INTAKE')
            old['selection'] = clip.get('selection', 'selected')
            old['split'] = 'external_holdout'
            old['selection_reason'] = clip.get('selection_reason')
            continue
        make_clip(path, output, start, end, ffmpeg)
        record = {
            'clip_id': clip['id'], 'source_id': source['id'], 'source_url': source['source_url'],
            'source_page_url': source['source_page_url'], 'license': source['license'],
            'attribution': source['attribution'], 'source_group': source['source_group'],
            'camera_id': source['camera_id'], 'source_context': source['source_context'],
            'source_path': str(path.relative_to(root)), 'source_sha256': source_hash,
            'source_sha1': source_sha1,
            'source_checksum_status': 'matched_expected' if source.get('expected_sha256') or source.get('expected_sha1') else 'recorded_only',
            'source_ffprobe': source_probe, 'clip_path': str(output.relative_to(root)),
            'clip_sha256': sha256(output), 'ffprobe': technical_probe(output, ffprobe),
            'source_offsets': {'start_sec': start, 'end_sec': end, 'time_basis': 'source_seconds',
                               'origin': 'whole_source' if whole_source else 'manifest'},
            'annotation_status': 'pending_review', 'legal_gold': None,
            'imported_at_utc': datetime.now(timezone.utc).isoformat(),
            'temporal_criteria_note': 'shorter_than_5_seconds' if end - start < 5 else None,
            'rights_proof': source.get('rights_proof'),
            'selection': clip.get('selection', 'selected'), 'split': 'external_holdout',
            'selection_reason': clip.get('selection_reason'),
        }
        records[clip['id']] = record
        index['records'] = list(records.values())
        write_json(index_path, index)
    selected = sum(row.get('selection') == 'selected' for row in index['records'])
    index['status'] = ('partial_collection_annotations_pending_review' if selected < 10
                       else 'intake_complete_annotations_pending_review')
    write_json(index_path, index)
    annotation_path = local_path(root, 'annotations.json')
    if not annotation_path.exists():
        write_json(annotation_path, draft_annotations(index['records']))
    annotations = read_json(annotation_path)
    seen = {row['clip_id'] for row in annotations['clips']}
    new_records = [row for row in index['records'] if row['clip_id'] not in seen]
    if new_records:
        annotations['clips'].extend(draft_annotations(new_records)['clips'])
        write_json(annotation_path, annotations)
    validate_annotations(annotations, index['records'])
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['init', 'validate', 'import', 'check-annotations'])
    parser.add_argument('--manifest', default='manifest.json', help='Path relative to ignored data/external23')
    parser.add_argument('--annotations', default='annotations.json')
    parser.add_argument('--download', action='store_true', help='Explicitly allow source downloads during import')
    parser.add_argument('--allow-partial', action='store_true', help='Import 1–9 pending candidates without declaring selection complete')
    parser.add_argument('--max-source-mib', type=int, default=2048)
    parser.add_argument('--ffprobe', default='/opt/homebrew/bin/ffprobe')
    parser.add_argument('--ffmpeg', default='/opt/homebrew/bin/ffmpeg')
    args = parser.parse_args()
    try:
        root = PRIVATE.resolve()
        require(root == PRIVATE.absolute(), 'PRIVATE_ROOT_SYMLINK_NOT_ALLOWED')
        path = local_path(root, args.manifest)
        require(args.max_source_mib > 0, 'DOWNLOAD_LIMIT_INVALID')
        if args.command == 'init':
            require(not path.exists(), 'MANIFEST_ALREADY_EXISTS')
            write_json(path, {'schema_version': 1, 'scope': 'external_action_proxies_not_legal_gold',
                              'sources': [], 'clips': []})
            print('OK: manifest_pending sources=0 clips=0 annotations=0')
        elif args.command == 'validate':
            document = read_json(path)
            validate_manifest(document, args.allow_partial)
            selected = sum(row.get('selection', 'selected') == 'selected' for row in document['clips'])
            print(f"OK: manifest sources={len(document['sources'])} clips={len(document['clips'])} selected={selected}")
        elif args.command == 'check-annotations':
            index = read_json(local_path(root, 'intake.json'))
            validate_annotations(read_json(local_path(root, args.annotations)), index['records'])
            print(f"OK: annotations_checked clips={len(index['records'])}")
        else:
            index = import_manifest(root, path, read_json(path), args.download, args.ffprobe,
                                    args.ffmpeg, args.max_source_mib * 1024 ** 2, args.allow_partial)
            groups = {row['source_group'] for row in index['records']}
            hashes = {row['clip_sha256'] for row in index['records']}
            selected = sum(row['selection'] == 'selected' for row in index['records'])
            print(f"OK: intake clips={len(index['records'])} selected={selected} groups={len(groups)} unique_sha256={len(hashes)} annotations=pending_review")
        return 0
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        code = str(error) if type(error) is ValueError and re.fullmatch('[A-Za-z0-9_]+', str(error)) else 'INTAKE_OPERATION_FAILED'
        print('ERROR: ' + code, file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
