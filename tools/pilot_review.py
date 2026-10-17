#!/usr/bin/env python3
"""Local pilot inventory and human adjudication. No frames, inference or exports."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
import unicodedata
from xml.etree.ElementTree import ParseError
from zipfile import BadZipFile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ml.eval import _audit_directory, _probe  # noqa: E402
from ml.evaluation_labels import (parse_timestamp, read_table, source_index,  # noqa: E402
                                  validate_content_sha256, video_key)
from tools.audit_dataset import file_identity, read_json, sha256  # noqa: E402
from tools.pilot_review_schema import draft_adjudication, validate_documents  # noqa: E402

OUTPUT_NAMES = {'inventory.json', 'adjudication.json'}


def require(condition, code):
    if not condition:
        raise ValueError(code)


def canonical(value):
    require(isinstance(value, str), 'PILOT_ID_INVALID')
    return video_key(unicodedata.normalize('NFC', value))


def local_file(value):
    path = Path(value)
    require('..' not in path.parts, 'INPUT_PATH_INVALID')
    if not path.is_absolute():
        path = ROOT / path
    parent = _audit_directory(path.parent)
    path = parent / path.name
    require(not path.is_symlink() and stat.S_ISREG(path.lstat().st_mode), 'INPUT_FILE_INVALID')
    return path


def guarded_json(path):
    before = file_identity(path)
    digest = sha256(path)
    result = read_json(path)
    require(file_identity(path) == before, 'INPUT_CHANGED')
    return result, digest


def load_pilots(path):
    document = read_json(local_file(path))
    rows = document.get('pilots') if isinstance(document, dict) else document
    require(isinstance(rows, list) and len(rows) == 8, 'EXACTLY_EIGHT_PILOTS_REQUIRED')
    result = []
    for row in rows:
        row = {'video': row} if isinstance(row, str) else row
        require(isinstance(row, dict) and set(row) <= {'video', 'relative_path', 'angle_id', 'source_group'},
                'PILOT_SELECTION_INVALID')
        canonical(row.get('video'))
        relative = row.get('relative_path')
        if relative is not None:
            require(isinstance(relative, str) and relative and '\\' not in relative,
                    'PILOT_RELATIVE_PATH_INVALID')
            path = Path(relative)
            require(not path.is_absolute() and not {'.', '..'} & set(path.parts)
                    and canonical(path.name) == canonical(row['video']), 'PILOT_RELATIVE_PATH_INVALID')
        for key in ('angle_id', 'source_group'):
            require(row.get(key) is None or isinstance(row[key], str) and bool(row[key].strip()),
                    'PILOT_CONTEXT_INVALID')
        result.append(dict(row))
    source_index([{'video': unicodedata.normalize('NFC', row['video'])} for row in result], 'pilots')
    return result


def selected_paths(videos_dir, pilots):
    directory = _audit_directory(videos_dir)
    wanted = {canonical(row['video']) for row in pilots}
    matches = defaultdict(list)
    # Only directory entries are inspected here. Bytes/probes are limited to the explicit eight.
    for folder, directories, filenames in os.walk(directory, followlinks=False):
        directories[:] = [name for name in directories if not (Path(folder) / name).is_symlink()]
        for name in filenames:
            try:
                key = canonical(name)
            except ValueError:
                continue
            if key in wanted:
                matches[key].append(Path(folder) / name)
    result = []
    for row in pilots:
        candidates = matches[canonical(row['video'])]
        require(len(candidates) == 1, 'PILOT_FILE_MISSING_OR_AMBIGUOUS')
        path = local_file(candidates[0])
        relative = path.relative_to(directory).as_posix()
        if row.get('relative_path') is not None:
            require(unicodedata.normalize('NFC', relative)
                    == unicodedata.normalize('NFC', row['relative_path']), 'PILOT_LITERAL_PATH_MISMATCH')
        result.append(path)
    return directory, result


def organizer_rows(path):
    before = file_identity(path)
    digest = sha256(path)
    table = read_table(path)
    require(table and isinstance(table[0], list), 'ORGANIZER_TABLE_INVALID')
    aliases = {'video': ('video', 'наименование файла'), 'timestamp': ('timestamp', 'таймстемп'),
               'label': ('label', 'нарушение / без нарушения')}
    header = [str(cell).strip().lower() for cell in table[0]]
    columns = {}
    for field, names in aliases.items():
        indices = [i for i, cell in enumerate(header) if cell in names]
        require(len(indices) == 1, 'ORGANIZER_COLUMNS_INVALID')
        columns[field] = indices[0]
    rows = []
    for row_number, values in enumerate(table[1:], 2):
        if not any(str(cell).strip() for cell in values):
            continue
        def cell(field):
            index = columns[field]
            return values[index] if index < len(values) else ''
        literal, label, timestamp = cell('video'), cell('label'), cell('timestamp')
        canonical(literal)
        require(isinstance(label, str) and label.strip().lower() in ('нарушение', 'без нарушения'),
                'ORGANIZER_LABEL_INVALID')
        seconds = None if timestamp is None or isinstance(timestamp, str) and not timestamp.strip() else parse_timestamp(timestamp)
        rows.append({'video': literal, 'label': label, 'timestamp': timestamp,
                     'positive': label.strip().lower() == 'нарушение', 'timestamp_sec': seconds,
                     'source_sha256': digest, 'source_row': row_number})
    source_index([{'video': unicodedata.normalize('NFC', row['video'])} for row in rows], 'organizer')
    require(file_identity(path) == before, 'ORGANIZER_SOURCE_CHANGED')
    return {canonical(row['video']): row for row in rows}, digest


def output_path(name, inputs=()):
    require(name in OUTPUT_NAMES, 'OUTPUT_NAME_INVALID')
    root = Path(ROOT).absolute()
    require(root == root.resolve(), 'OUTPUT_ROOT_SYMLINK')
    output = root / 'data' / 'pilot-review' / name
    current = root
    for part in output.relative_to(root).parts:
        current = current / part
        require(not current.is_symlink(), 'OUTPUT_SYMLINK')
    require(not output.exists() or output.is_file(), 'OUTPUT_NOT_FILE')
    for source in inputs:
        source = Path(source)
        require(output.resolve() != source.resolve()
                and not (source.is_dir() and output.resolve().is_relative_to(source.resolve())),
                'OUTPUT_OVERLAPS_INPUT')
        require(not (output.exists() and source.is_file()
                     and os.path.samestat(output.stat(), source.stat())), 'OUTPUT_OVERLAPS_INPUT')
    return output


def write_new_json(name, document, inputs=()):
    output = output_path(name, inputs)
    require(not output.exists(), 'OUTPUT_ALREADY_EXISTS')
    output.parent.mkdir(parents=True, exist_ok=True)
    output_path(name, inputs)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=output.parent,
                                         prefix='.pilot-review-', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(document, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        output_path(name, inputs)
        # Atomic publication without replacing a concurrently created human file.
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output


def build_inventory(pilot_ids_path, videos_dir, labels_path, ffprobe=None):
    selection_path, labels_path = local_file(pilot_ids_path), local_file(labels_path)
    selection_before = file_identity(selection_path)
    selection_hash = sha256(selection_path)
    pilots = load_pilots(selection_path)
    directory, paths = selected_paths(videos_dir, pilots)
    organizers, source_hash = organizer_rows(labels_path)
    executable = shutil.which(ffprobe or 'ffprobe')
    require(executable is not None, 'FFPROBE_REQUIRED')
    clips = []
    identities = {}
    for pilot, path in zip(pilots, paths):
        key = canonical(pilot['video'])
        require(key in organizers, 'PILOT_ORGANIZER_LABEL_MISSING')
        before = file_identity(path)
        digest = validate_content_sha256(sha256(path))
        metadata = _probe(path, executable)
        duration = metadata.get('duration_sec')
        require(metadata.get('status') == 'ok' and isinstance(duration, (int, float))
                and not isinstance(duration, bool) and 0 < duration < float('inf'), 'PILOT_DURATION_REQUIRED')
        require(file_identity(path) == before, 'PILOT_CHANGED_DURING_METADATA')
        identities[path] = before
        clips.append({'video': pilot['video'], 'video_key': key,
                      'relative_path': path.relative_to(directory).as_posix(),
                      'content_sha256': digest, 'size_bytes': path.stat().st_size,
                      'duration_sec': duration,
                      'metadata_provenance': {'kind': 'ffprobe_numeric', 'tool': 'ffprobe',
                                              'source_content_sha256': digest},
                      'organizer': organizers[key], 'angle_id': pilot.get('angle_id'),
                      'source_group': pilot.get('source_group'), 'content_group': 'sha256:' + digest})
    require(all(file_identity(path) == before for path, before in identities.items()), 'PILOT_INPUT_CHANGED')
    require(file_identity(selection_path) == selection_before and sha256(selection_path) == selection_hash,
            'PILOT_SELECTION_CHANGED')
    require(sha256(labels_path) == source_hash, 'ORGANIZER_SOURCE_CHANGED')
    return {'schema_version': 1, 'kind': 'pilot_review_inventory', 'usage': 'development',
            'selection': {'path': str(selection_path), 'sha256': selection_hash,
                          'pilot_ids': [row['video'] for row in pilots]},
            'organiser_source': {'path': str(labels_path), 'content_sha256': source_hash},
            'videos_root': str(directory), 'clips': clips}


def verify_inventory_inputs(inventory):
    selection = local_file(inventory['selection']['path'])
    labels = local_file(inventory['organiser_source']['path'])
    identities = {path: file_identity(path) for path in (selection, labels)}
    require(sha256(selection) == inventory['selection']['sha256'], 'PILOT_SELECTION_SHA256_MISMATCH')
    require(sha256(labels) == inventory['organiser_source']['content_sha256'], 'ORGANIZER_SHA256_MISMATCH')
    pilots = load_pilots(selection)
    require([row['video'] for row in pilots] == inventory['selection']['pilot_ids'], 'PILOT_SELECTION_MISMATCH')
    directory, paths = selected_paths(inventory['videos_root'], pilots)
    originals, source_hash = organizer_rows(labels)
    clips = {row['video_key']: row for row in inventory['clips']}
    for pilot, path in zip(pilots, paths):
        identities[path] = file_identity(path)
        key = canonical(pilot['video'])
        row = clips[key]
        require(row['video'] == pilot['video'] and row['relative_path'] == path.relative_to(directory).as_posix(),
                'PILOT_LITERAL_PATH_MISMATCH')
        require(row['organizer'] == originals[key] and row['organizer']['source_sha256'] == source_hash,
                'ORGANIZER_PROVENANCE_MISMATCH')
        require(row['angle_id'] == pilot.get('angle_id') and row['source_group'] == pilot.get('source_group'),
                'PILOT_CONTEXT_PROVENANCE_MISMATCH')
        require(path.stat().st_size == row['size_bytes'] and sha256(path) == row['content_sha256'],
                'PILOT_CONTENT_SHA256_MISMATCH')
    require(all(file_identity(path) == before for path, before in identities.items()),
            'PILOT_INPUT_CHANGED')
    return [selection, labels, directory, *paths]


def validate_files(require_complete=False):
    inventory_path, adjudication_path = output_path('inventory.json'), output_path('adjudication.json')
    inventory, digest = guarded_json(inventory_path)
    adjudication, adjudication_digest = guarded_json(adjudication_path)
    summary = validate_documents(inventory, adjudication, digest)
    inputs = verify_inventory_inputs(inventory)
    for name in OUTPUT_NAMES:
        output_path(name, inputs)
    require(sha256(inventory_path) == digest and sha256(adjudication_path) == adjudication_digest,
            'REVIEW_FILES_CHANGED')
    if require_complete:
        require(summary['review_handoff_ready'], 'REVIEW_HANDOFF_INCOMPLETE')
    return summary


def initialize(pilot_ids_path, videos_dir, labels_path, ffprobe=None):
    inputs = [local_file(pilot_ids_path), local_file(labels_path), _audit_directory(videos_dir)]
    inventory_path, adjudication_path = (output_path(name, inputs) for name in ('inventory.json', 'adjudication.json'))
    if inventory_path.exists():
        inventory, digest = guarded_json(inventory_path)
        require(inventory['selection']['path'] == str(inputs[0])
                and inventory['organiser_source']['path'] == str(inputs[1])
                and inventory['videos_root'] == str(inputs[2]), 'INIT_INPUTS_DIFFER_FROM_EXISTING')
        if not adjudication_path.exists():
            document = draft_adjudication(inventory, digest)
            validate_documents(inventory, document, digest)
            protected = verify_inventory_inputs(inventory)
            write_new_json('adjudication.json', document, protected)
        result = validate_files()
        return dict(result, resumed=True)
    require(not adjudication_path.exists(), 'INVENTORY_MISSING_EXISTING_ADJUDICATION')
    inventory = build_inventory(pilot_ids_path, videos_dir, labels_path, ffprobe)
    protected = verify_inventory_inputs(inventory)
    # Validate the pending schema before publishing either file.
    draft = draft_adjudication(inventory, '0' * 64)
    validate_documents(inventory, draft, '0' * 64)
    path = write_new_json('inventory.json', inventory, protected)
    digest = sha256(path)
    draft = draft_adjudication(inventory, digest)
    write_new_json('adjudication.json', draft, protected)
    return dict(validate_files(), resumed=False)


class SafeParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError('PILOT_ARGUMENTS_INVALID')


def main(argv=None):
    parser = SafeParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True, parser_class=SafeParser)
    init = commands.add_parser('init')
    init.add_argument('--pilot-ids', required=True)
    init.add_argument('--videos-dir', required=True)
    init.add_argument('--labels', required=True)
    init.add_argument('--ffprobe')
    validate = commands.add_parser('validate')
    validate.add_argument('--require-complete', action='store_true')
    commands.add_parser('summary')
    try:
        args = parser.parse_args(argv)
        if args.command == 'init':
            summary = initialize(args.pilot_ids, args.videos_dir, args.labels, args.ffprobe)
        else:
            summary = validate_files(getattr(args, 'require_complete', False))
    except (OSError, ValueError, KeyError, TypeError, OverflowError, BadZipFile, ParseError):
        print('ERROR: pilot_review_invalid_or_incomplete', file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
