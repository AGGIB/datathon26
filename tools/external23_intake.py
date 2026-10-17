#!/usr/bin/env python3
"""Local external-video intake. Stdout contains aggregate counts or fixed error codes only."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request

from external23_annotations import draft_annotations, validate_annotations
from external23_common import SafeParser, active_path, binary, https, local_path, number, read_json, require, sha256, text, write_json
from external23_policy import journal_changes, selected_clip, source_usable, verify_proof

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / 'data' / 'external23'
ID = re.compile(r'[a-z0-9][a-z0-9_-]{0,79}')
CLIP_ID = re.compile(r'ext23-[a-z0-9][a-z0-9_-]{0,73}')


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
        require(isinstance(cid, str) and CLIP_ID.fullmatch(cid) and cid not in ids, 'CLIP_ID_NAMESPACE')
        ids.add(cid)
        require(clip.get('selection') in {'selected', 'reserve', 'candidate'}, 'SELECTION_INVALID')
        require(clip.get('split', 'external_holdout') == 'external_holdout', 'EXTERNAL_HOLDOUT_ONLY')
        require(isinstance(clip.get('source_id'), str) and clip['source_id'] in indexed, 'CLIP_SOURCE_MISSING')
        if clip.get('selection', 'selected') == 'selected':
            selected_clip(indexed[clip['source_id']], clip)
        require('start_sec' in clip and 'end_sec' in clip, 'SOURCE_OFFSETS_REQUIRED')
        start, end = clip['start_sec'], clip['end_sec']
        require(start is None and end is None or number(start) and number(end) and 0 <= start < end,
                'SOURCE_OFFSETS_INVALID')
    policy = document.get('selection_policy', {})
    require(isinstance(policy, dict), 'SELECTION_POLICY_INVALID')
    minimum = policy.get('min_source_groups', 2)
    require(type(minimum) is int and 2 <= minimum <= 15, 'MIN_SOURCE_GROUPS_INVALID')
    if selected:
        groups = {indexed[c['source_id']]['source_group'] for c in clips if c.get('selection') == 'selected'}
        require(len(groups) >= minimum, 'INDEPENDENT_SOURCE_GROUPS_REQUIRED')
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


class HTTPSOnlyRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, file, code, message, headers, newurl):
        https(newurl)  # Reject before urllib can request the redirect destination.
        return super().redirect_request(request, file, code, message, headers, newurl)


def download(source, destination, max_bytes, root=None, opener=None):
    source_usable(source)
    require(source['source_context'].get('scene_type') == 'bank_cashdesk', 'SOURCE_BANK_CONTEXT_REQUIRED')
    verify_proof(root or destination.parent.parent, source.get('rights_proof'))
    https(source['source_url'])
    request = urllib.request.Request(source['source_url'], headers={'User-Agent': 'external23-local-intake/1'})
    partial = destination.with_suffix('.partial')
    require(not partial.exists(), 'PARTIAL_SOURCE_ALREADY_EXISTS')
    try:
        opener = opener or urllib.request.build_opener(HTTPSOnlyRedirect())
        with opener.open(request, timeout=60) as response, partial.open('xb') as output:
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
    with tempfile.NamedTemporaryFile(dir=output_path.parent, suffix='.mp4', delete=False) as stream:
        partial = Path(stream.name)
    try:
        result = subprocess.run([binary, '-nostdin', '-v', 'error', '-y', '-threads', '2',
                                 '-filter_threads', '1', '-ss', str(start), '-i', str(source_path),
                                 '-t', str(end - start), '-map', '0:v:0', '-an', '-sn', '-dn',
                                 '-map_metadata', '-1', '-c:v', 'libx264', '-preset', 'veryfast',
                                 '-crf', '20', '-threads', '2', '-movflags', '+faststart', str(partial)],
                                capture_output=True, timeout=1800)
        require(result.returncode == 0, 'FFMPEG_CLIP_FAILED')
        require(not output_path.exists(), 'CLIP_ALREADY_EXISTS')
        partial.replace(output_path)
    finally:
        partial.unlink(missing_ok=True)


def import_manifest(root, manifest_path, document, allow_download, ffprobe, ffmpeg, max_bytes, allow_partial=False):
    sources = validate_manifest(document, allow_partial)
    require(document['clips'], 'MANIFEST_PENDING_NO_CLIPS')
    root = root.resolve()
    for folder in ['sources', 'clips', 'pending']:
        local_path(root, folder).mkdir(parents=True, exist_ok=True)
    for clip in document['clips']:
        if clip['selection'] == 'selected':
            verify_proof(root, clip['rights_proof'])
    index_path = local_path(root, 'intake.json')
    manifest_hash = sha256(manifest_path)
    index = read_json(index_path) if index_path.exists() else {
        'schema_version': 1, 'manifest_sha256': manifest_hash, 'manifest_snapshot': document,
        'records': [], 'status': 'importing'}
    if index.get('manifest_sha256') != manifest_hash:
        changes = journal_changes(index.get('manifest_snapshot', {}), document)
        index.setdefault('manifest_change_log', []).append({
            'from_sha256': index['manifest_sha256'], 'to_sha256': manifest_hash, 'changes': changes})
        index.setdefault('previous_manifest_sha256', []).append(index['manifest_sha256'])
        index['manifest_sha256'], index['manifest_snapshot'] = manifest_hash, document
    records = {record['clip_id']: record for record in index['records']}
    require(set(records) <= {clip['id'] for clip in document['clips']}, 'INTAKE_CLIP_SET_MISMATCH')
    cache = {}
    for clip in document['clips']:
        source = sources[clip['source_id']]
        if clip['selection'] == 'reserve':
            require(clip['id'] not in records, 'MATERIALIZED_RESERVE_REQUIRES_QUARANTINE')
            continue  # Reserve metadata never authorizes reading or rendering its media.
        verify_proof(root, clip.get('rights_proof') or source.get('rights_proof'))
        if source['id'] not in cache:
            source_usable(source)
            path = active_path(root, source.get('local_path', 'sources/' + source['id'] + '.source'))
            if not path.exists():
                require(allow_download and 'local_path' not in source, 'SOURCE_MISSING_DOWNLOAD_NOT_ENABLED')
                download(source, path, max_bytes, root)
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
            if proof and 'url' in proof:
                verify_proof(root, proof)
            cache[source['id']] = (path, digest, sha1.hexdigest(), technical_probe(path, ffprobe))
        path, source_hash, source_sha1, source_probe = cache[source['id']]
        start, end = clip['start_sec'], clip['end_sec']
        whole_source = start is None
        if whole_source:
            start, end = 0.0, source_probe['duration_seconds']
        require(end <= source_probe['duration_seconds'], 'CLIP_EXCEEDS_SOURCE_DURATION')
        output = local_path(root, 'clips/' + clip['id'] + '.mp4')
        pending = local_path(root, 'pending/' + clip['id'] + '.json')
        identity = {'source_sha256': source_hash, 'start_sec': start, 'end_sec': end}
        if output.exists():
            old = records.get(clip['id'])
            if old:
                require(old['source_sha256'] == source_hash and old['clip_sha256'] == sha256(output),
                        'EXISTING_CLIP_NOT_MATCHED_TO_INTAKE')
                update_review_metadata(old, source, clip)
                continue
            require(pending.is_file() and read_json(pending) == identity, 'UNMATCHED_ORPHAN_CLIP')
            archive = local_path(root, 'orphaned/' + clip['id'] + '-' + sha256(output) + '.mp4')
            archive.parent.mkdir(parents=True, exist_ok=True)
            require(not archive.exists(), 'ORPHAN_ARCHIVE_ALREADY_EXISTS')
            output.replace(archive)
        write_json(pending, identity)
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
            'rights_proof': clip.get('rights_proof'),
            'selection': clip.get('selection', 'selected'), 'split': 'external_holdout',
            'selection_reason': clip.get('selection_reason'),
        }
        update_review_metadata(record, source, clip)
        records[clip['id']] = record
        index['records'] = list(records.values())
        write_json(index_path, index)
        pending.unlink(missing_ok=True)
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


def update_review_metadata(record, source, clip):
    record.update(source_context=source['source_context'], selection=clip['selection'],
                  split='external_holdout', selection_reason=clip.get('selection_reason'),
                  camera_id=clip.get('camera_id', source['camera_id']))
    for field in ['angle_id', 'bank_scene_confirmed', 'bank_scene_evidence_reference', 'rights_proof']:
        if field == 'rights_proof' and (clip.get(field) or source.get(field)):
            record[field] = clip.get(field) or source[field]
        elif field in clip:
            record[field] = clip[field]
        else:
            record.pop(field, None)


def main():
    parser = SafeParser(description=__doc__)
    parser.add_argument('command', choices=['init', 'validate', 'import', 'check-annotations'])
    parser.add_argument('--manifest', default='manifest.json', help='Path relative to ignored data/external23')
    parser.add_argument('--annotations', default='annotations.json')
    parser.add_argument('--download', action='store_true', help='Explicitly allow source downloads during import')
    parser.add_argument('--allow-partial', action='store_true', help='Import 1–9 pending candidates without declaring selection complete')
    parser.add_argument('--max-source-mib', type=int, default=2048)
    parser.add_argument('--ffprobe')
    parser.add_argument('--ffmpeg')
    try:
        args = parser.parse_args()
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
            index = import_manifest(root, path, read_json(path), args.download,
                                    binary(args.ffprobe, 'ffprobe'), binary(args.ffmpeg, 'ffmpeg'),
                                    args.max_source_mib * 1024 ** 2, args.allow_partial)
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
