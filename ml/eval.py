"""Local, dependency-free evaluation. No media is sent to any service."""
from collections import Counter, defaultdict
import math
import argparse
import csv
import json
from fractions import Fraction
from pathlib import Path
import shutil
import subprocess
from xml.etree import ElementTree as ET
import zipfile

from .evaluation_labels import load_labels, parse_timestamp, read_table, video_key


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _metrics(tp, tn, fp, fn, total):
    return {'accuracy': _ratio(tp + tn, total), 'precision': _ratio(tp, tp + fp),
            'recall': _ratio(tp, tp + fn), 'f1': _ratio(2 * tp, 2 * tp + fp + fn),
            'accuracy_denominator': total, 'positive_denominator': tp + fn}


def _prediction(record):
    result = {'status': 'invalid', 'positive': None, 'events': [], 'elapsed_sec': None, 'error': None}
    if record is None:
        result['status'] = 'missing'
        return result
    try:
        elapsed = record.get('elapsed_sec')
        if elapsed is not None:
            result['elapsed_sec'] = parse_timestamp(elapsed)
        status = record.get('status')
        if status in ('failed', 'error'):
            result['status'] = 'failed'
            return result
        if status in ('timed_out', 'timeout', 'timedout'):
            result['status'] = 'timed_out'
            return result
        if status not in ('done', 'ok', 'success', 'completed'):
            raise ValueError('missing or unsupported status')
        if 'result' in record:
            payload = record['result']
            if not isinstance(payload, dict):
                raise ValueError('result envelope must be an object')
            for field in ('verdict', 'violations'):
                if field in record and field in payload and record[field] != payload[field]:
                    raise ValueError('conflicting top-level and envelope fields')
            record = dict(record, **{key: value for key, value in payload.items() if key in ('verdict', 'violations')})
        verdict = record.get('verdict')
        if not isinstance(verdict, str):
            raise ValueError('verdict must be a string')
        verdict = verdict.strip().lower()
        verdicts = {'нарушение': True, 'без нарушения': False, 'нарушений нет': False,
                    'violation': True, 'no_violation': False}
        if verdict not in verdicts:
            raise ValueError('unknown verdict')
        events = record.get('violations')
        if not isinstance(events, list):
            raise ValueError('violations must be a list')
        if status == 'done' and verdicts[verdict] and not events:
            raise ValueError('done violation requires at least one event (K1)')
        if not verdicts[verdict] and events:
            raise ValueError('negative verdict contradicts events')
        parsed = []
        for event in events:
            if not isinstance(event, dict):
                raise ValueError('violation must be an object')
            kind = event.get('type_id') or event.get('type')
            kind = 'unknown' if kind is None or kind == '' else kind
            if not isinstance(kind, str) or not kind.strip():
                raise ValueError('event type must be a nonempty string or absent')
            times = {field: parse_timestamp(event[field]) if event.get(field) is not None else None
                     for field in ('t_start', 't_peak', 't_end')}
            if event.get('timecode_sec') is not None:
                canonical_peak = parse_timestamp(event['timecode_sec'])
                if times['t_peak'] is not None and canonical_peak != times['t_peak']:
                    raise ValueError('conflicting timecode_sec and t_peak')
                times['t_peak'] = canonical_peak
            supplied_times = [value for value in times.values() if value is not None]
            if supplied_times != sorted(supplied_times):
                raise ValueError('event must satisfy t_start <= t_peak <= t_end')
            parsed.append(dict(times, type=kind))
        result.update(status='ok', positive=verdicts[verdict], events=parsed)
    except (ValueError, TypeError) as error:
        result['error'] = str(error)
    return result


def _match_events(truth, predicted):
    """Minimum absolute-time cost, maximum-cardinality monotone assignment."""
    truth = sorted([e for e in truth if e['t_peak'] is not None], key=lambda e: e['t_peak'])
    predicted = sorted([e for e in predicted if e['t_peak'] is not None], key=lambda e: e['t_peak'])
    n, m = len(truth), len(predicted)
    states = {(0, 0): (0, 0.0)}
    parent = {}
    for i in range(n + 1):
        for j in range(m + 1):
            if i == j == 0:
                continue
            choices = []
            if i:
                choices.append((states[i - 1, j], (i - 1, j), False))
            if j:
                choices.append((states[i, j - 1], (i, j - 1), False))
            if i and j:
                count, cost = states[i - 1, j - 1]
                choices.append(((count + 1, cost + abs(truth[i - 1]['t_peak'] - predicted[j - 1]['t_peak'])), (i - 1, j - 1), True))
            value, previous, matched = min(choices, key=lambda item: (-item[0][0], item[0][1]))
            states[i, j] = value
            parent[i, j] = (previous, matched)
    result = []
    i, j = n, m
    while i or j:
        previous, matched = parent[i, j]
        if matched:
            result.append({'truth_t_peak': truth[i - 1]['t_peak'], 'predicted_t_peak': predicted[j - 1]['t_peak'],
                           'error_sec': abs(truth[i - 1]['t_peak'] - predicted[j - 1]['t_peak']),
                           'truth_type': truth[i - 1]['type'], 'predicted_type': predicted[j - 1]['type']})
        i, j = previous
    return list(reversed(result))


def _annotation_index(labels, annotations):
    known = {video_key(row['video']): row for row in labels}
    indexed = defaultdict(lambda: {'angle_id': 'unknown', 'events': []})
    for annotation in annotations or []:
        if not isinstance(annotation, dict):
            raise ValueError('annotation must be an object')
        key = video_key(annotation.get('video'))
        if key not in known:
            raise ValueError('annotation video absent from labels')
        angle = annotation.get('angle_id') or 'unknown'
        if not isinstance(angle, str) or not angle.strip() or angle != angle.strip():
            raise ValueError('angle_id must be a nonempty string without edge whitespace')
        target = indexed[key]
        if target['angle_id'] != 'unknown' and angle != 'unknown' and target['angle_id'] != angle:
            raise ValueError('conflicting angle_id for video')
        if angle != 'unknown':
            target['angle_id'] = angle
        peak = annotation.get('t_peak')
        kind = annotation.get('type_id') or annotation.get('type') or 'unknown'
        if not isinstance(kind, str) or not kind.strip() or kind != kind.strip():
            raise ValueError('annotation type must be a nonempty string without edge whitespace')
        if kind == 'none' and not known[key]['positive'] and peak in (None, ''):
            continue
        if peak is not None and peak != '':
            if not known[key]['positive']:
                raise ValueError('event annotation contradicts negative label')
            event = {'t_peak': parse_timestamp(peak), 'type': kind}
            if event in target['events']:
                raise ValueError('duplicate annotated event')
            target['events'].append(event)
        elif kind != 'unknown':
            raise ValueError('event type requires t_peak')
    return indexed


def _validate_split(labels, split, annotations):
    if isinstance(split, dict):
        if set(split) not in ({'train', 'test'}, {'train', 'holdout'}):
            raise ValueError('split JSON must have train and test/holdout lists')
        records = []
        for side, values in split.items():
            if not isinstance(values, list):
                raise ValueError('split sides must be lists')
            for value in values:
                if not isinstance(value, dict):
                    raise ValueError('split entries must include video and angle_id')
                records.append(dict(value, subset='test' if side == 'holdout' else side))
        split = records
    if not isinstance(split, list):
        raise ValueError('split must be a list or partition object')
    known = {video_key(row['video']) for row in labels}
    result = {}
    angles = {'train': set(), 'test': set()}
    for row in split:
        key = video_key(row.get('video', row.get('video_id')))
        side = row.get('subset')
        angle = row.get('angle_id')
        if key in result or key not in known or side not in angles:
            raise ValueError('split must assign every known video exactly once to train/test')
        if not isinstance(angle, str) or not angle.strip() or angle != angle.strip() or angle.lower() in ('unknown', 'неизвестно', 'unassigned'):
            raise ValueError('explicit known angle_id required for every split video')
        annotation_angle = annotations.get(key, {}).get('angle_id', 'unknown')
        if annotation_angle != 'unknown' and annotation_angle != angle:
            raise ValueError('split angle disagrees with annotation')
        result[key] = {'subset': side, 'angle_id': angle}
        angles[side].add(angle)
    if set(result) != known or not all(angles.values()) or angles['train'] & angles['test']:
        raise ValueError('incomplete, empty or leaking angle holdout')
    return result, {side: sorted(values) for side, values in angles.items()}


def score(labels, predictions, annotations=None, split=None, subset=None, tolerance_sec=2.0):
    """Score all labels; optional explicit event and angle annotations override anchors."""
    tolerance_sec = parse_timestamp(tolerance_sec)
    if isinstance(predictions, dict):
        keys = [key for key in ('predictions', 'results') if key in predictions]
        if len(keys) != 1:
            raise ValueError('wrapper must contain exactly one predictions/results list')
        predictions = predictions[keys[0]]
    if not isinstance(predictions, list):
        raise ValueError('predictions must be a list')
    annotation_index = _annotation_index(labels, annotations)
    subset = subset or ('test' if split is not None else 'all')
    if subset not in ('train', 'test', 'all') or (split is None and subset != 'all'):
        raise ValueError('train/test subset requires an explicit split')
    partition, split_angles = _validate_split(labels, split, annotation_index) if split is not None else ({}, {})
    selected = {video_key(row['video']) for row in labels if subset == 'all' or partition[video_key(row['video'])]['subset'] == subset}
    excluded_predictions = 0
    index = {}
    known = {video_key(row['video']) for row in labels}
    for record in predictions:
        if not isinstance(record, dict):
            raise ValueError('prediction must be an object')
        key = video_key(record.get('video', record.get('video_id')))
        if 'video' in record and 'video_id' in record and video_key(record['video_id']) != key:
            raise ValueError('conflicting video and video_id')
        if key in index:
            raise ValueError('duplicate prediction video')
        if key not in known:
            raise ValueError('prediction video is absent from labels')
        index[key] = record
        if key not in selected:
            excluded_predictions += 1
    rows = []
    confusion = dict(tp=0, tn=0, fp=0, fn=0, abstain_positive=0, abstain_negative=0)
    errors, false_positives, misses = [], [], []
    truth_event_count, predicted_event_count = 0, 0
    type_truth, type_predicted, type_correct, type_metric_predicted = Counter(), Counter(), Counter(), Counter()
    type_metric_truth = Counter()
    type_confusion = defaultdict(Counter)
    limited = False
    for label in labels:
        key = video_key(label['video'])
        if key not in selected:
            continue
        pred = _prediction(index.get(key))
        truth = label['positive']
        if pred['status'] == 'ok':
            outcome = ('t' if pred['positive'] == truth else 'f') + ('p' if pred['positive'] else 'n')
        else:
            outcome = 'abstain_positive' if truth else 'abstain_negative'
        confusion[outcome] += 1
        if outcome == 'fp':
            false_positives.append(label['video'])
        if truth and outcome != 'tp':
            misses.append(label['video'])
        annotation = dict(annotation_index.get(key, {'angle_id': 'unknown', 'events': []}))
        if partition:
            annotation['angle_id'] = partition[key]['angle_id']
        truth_events = annotation['events']
        label_limited = truth and not truth_events
        limited = limited or label_limited
        if label_limited and label['timestamp'] is not None:
            truth_events = [{'t_peak': label['timestamp'], 'type': 'unknown'}]
        matches = _match_events(truth_events, pred['events'])
        matched_errors = [match['error_sec'] for match in matches]
        errors.extend(matched_errors)
        truth_event_count += len(truth_events)
        predicted_event_count += len(pred['events'])
        type_truth.update(e['type'] for e in truth_events)
        type_predicted.update(e['type'] for e in pred['events'])
        if not label_limited and all(event['type'] != 'unknown' for event in truth_events):
            type_metric_predicted.update(e['type'] for e in pred['events'])
            type_metric_truth.update(e['type'] for e in truth_events)
            localized_matches = [match for match in matches if match['error_sec'] <= tolerance_sec]
            type_correct.update(match['truth_type'] for match in localized_matches if match['truth_type'] == match['predicted_type'])
            matched_truth = Counter(match['truth_type'] for match in localized_matches)
            matched_predicted = Counter(match['predicted_type'] for match in localized_matches)
            for match in localized_matches:
                type_confusion[match['truth_type']][match['predicted_type']] += 1
            for kind, count in (Counter(e['type'] for e in truth_events) - matched_truth).items():
                type_confusion[kind]['<missed>'] += count
            for kind, count in (Counter(e['type'] for e in pred['events']) - matched_predicted).items():
                type_confusion['<spurious>'][kind] += count
        rows.append(dict(video=label['video'], truth_positive=truth, truth_timestamp=label['timestamp'],
                         status=pred['status'], predicted_positive=pred['positive'], outcome=outcome,
                         elapsed_sec=pred['elapsed_sec'], sla_met=pred['status'] == 'ok' and pred['elapsed_sec'] is not None and pred['elapsed_sec'] <= 300,
                         angle_id=annotation['angle_id'], timecode_errors_sec=matched_errors,
                         truth_events=truth_events, predicted_events=pred['events'], matched_events=matches,
                         label_only_localization=label_limited, error=pred['error']))
    tp, tn, fp, fn = (confusion[x] for x in ('tp', 'tn', 'fp', 'fn'))
    total, valid = len(rows), tp + tn + fp + fn
    elapsed = [row['elapsed_sec'] for row in rows if row['elapsed_sec'] is not None]
    angles = {}
    for angle in sorted({row['angle_id'] for row in rows}):
        group = [row for row in rows if row['angle_id'] == angle]
        angles[angle] = {'total': len(group), 'scored': sum(row['status'] == 'ok' for row in group),
                         'correct': sum(row['outcome'] in ('tp', 'tn') for row in group)}
    kinds = sorted(set(type_truth) | set(type_predicted) | {'unknown'})
    types = {kind: {'truth_events': type_truth[kind], 'predicted_events': type_predicted[kind],
                    'matched_correct_type': type_correct[kind],
                    'precision': _ratio(type_correct[kind], type_metric_predicted[kind]) if kind != 'unknown' else None,
                    'recall': _ratio(type_correct[kind], type_metric_truth[kind]) if kind != 'unknown' else None,
                    'precision_denominator': type_metric_predicted[kind] if kind != 'unknown' else 0,
                    'recall_denominator': type_metric_truth[kind] if kind != 'unknown' else 0} for kind in kinds}
    overall = {'total': total, 'scored': valid, 'coverage': _ratio(valid, total),
               'label_counts': {'positive': sum(row['truth_positive'] for row in rows), 'negative': sum(not row['truth_positive'] for row in rows)},
               'type_metrics_limited': bool(limited or type_truth['unknown']),
               'status_counts': dict(Counter(row['status'] for row in rows)), 'confusion': confusion,
               'scored_only': _metrics(tp, tn, fp, fn, valid),
               'end_to_end': _metrics(tp, tn, fp, fn + confusion['abstain_positive'], total),
               'false_positives': false_positives, 'misses': misses,
               'processing_time': {'measured': len(elapsed), 'sum_sec': sum(elapsed), 'mean_sec': _ratio(sum(elapsed), len(elapsed)), 'max_sec': max(elapsed) if elapsed else None},
               'sla': {'limit_sec': 300, 'within_300_sec': sum(row['sla_met'] for row in rows), 'total_denominator': total,
                       'unknown_elapsed': total - len(elapsed),
                       'completed_late': sum(row['status'] == 'ok' and row['elapsed_sec'] is not None and row['elapsed_sec'] > 300 for row in rows),
                       'rate_total': _ratio(sum(row['sla_met'] for row in rows), total),
                       'timely_correct_rate_total': _ratio(sum(row['sla_met'] and row['outcome'] in ('tp', 'tn') for row in rows), total)},
               'localization': {'truth_events': truth_event_count, 'predicted_events': predicted_event_count,
                                'tolerance_sec': tolerance_sec,
                                'within_tolerance': sum(error <= tolerance_sec for error in errors),
                                'within_tolerance_rate_truth': _ratio(sum(error <= tolerance_sec for error in errors), truth_event_count),
                                'matched_events': len(errors), 'unmatched_truth_events': truth_event_count - len(errors),
                                'unmatched_predicted_events': predicted_event_count - len(errors),
                                'mae_sec': _ratio(sum(errors), len(errors)), 'label_only_limited': bool(limited)},
               'angles': angles, 'types': types,
               'type_confusion': {kind: dict(counts) for kind, counts in type_confusion.items()}}
    return {'overall': overall, 'per_video': rows,
            'split': {'supplied': split is not None, 'subset': subset, 'angles': split_angles,
                      'excluded_predictions': excluded_predictions, 'angle_holdout_validated': split is not None}}


def _probe(path, executable):
    """Only numerical metadata; never decode frames, export tags, or call a network API."""
    command = [executable, '-v', 'error', '-show_entries',
               'format=duration,size:stream=codec_type,width,height,avg_frame_rate,time_base,nb_frames',
               '-of', 'json', str(path)]
    try:
        process = subprocess.run(command, capture_output=True, timeout=20)
        if process.returncode:
            return {'status': 'failed'}
        data = json.loads(process.stdout)
        streams = [stream for stream in data.get('streams', []) if stream.get('codec_type') == 'video']
        if not streams:
            return {'status': 'no_video_stream'}
        stream = streams[0]
        def number(value):
            try:
                result = float(Fraction(str(value)))
                return result if math.isfinite(result) and result >= 0 else None
            except (ValueError, ZeroDivisionError):
                return None
        return {'status': 'ok', 'video_streams': len(streams),
                'duration_sec': number(data.get('format', {}).get('duration')),
                'width': number(stream.get('width')), 'height': number(stream.get('height')),
                'fps': number(stream.get('avg_frame_rate')), 'time_base_sec': number(stream.get('time_base')),
                'frame_count': number(stream.get('nb_frames')), 'size_bytes': number(data.get('format', {}).get('size'))}
    except subprocess.TimeoutExpired:
        return {'status': 'timed_out'}
    except (OSError, ValueError, TypeError):
        return {'status': 'failed'}


def audit(labels, videos_dir, use_ffprobe=True):
    """Recursively match exact literal identifiers and optional lowercase .mov."""
    directory = Path(videos_dir)
    if not directory.is_dir():
        raise ValueError('videos-dir must be a directory')
    files = defaultdict(list)
    for path in sorted(directory.rglob('*')):
        if path.is_file() and (path.suffix.lower() == '.mov' or not path.suffix):
            files[video_key(path.name)].append(path)
    executable = shutil.which('ffprobe') if use_ffprobe else None
    rows = []
    for label in labels:
        matches = files.get(video_key(label['video']), [])
        state = 'matched' if len(matches) == 1 else ('missing' if not matches else 'ambiguous')
        metadata = _probe(matches[0], executable) if executable and len(matches) == 1 else {'status': 'disabled' if not use_ffprobe else ('unavailable' if not executable else 'not_probed')}
        duration = metadata.get('duration_sec')
        rows.append(dict(label, file_status=state, matched_paths=[str(path.relative_to(directory)) for path in matches],
                         metadata=metadata, angle_id='unknown', type='unknown',
                         anchor_after_duration=label['timestamp'] > duration if label['timestamp'] is not None and duration is not None else None))
    known = {video_key(label['video']) for label in labels}
    overall = {'total': len(labels), 'positive': sum(label['positive'] for label in labels),
               'negative': sum(not label['positive'] for label in labels),
               'matched_files': sum(row['file_status'] == 'matched' for row in rows),
               'missing_files': sum(row['file_status'] == 'missing' for row in rows),
               'ambiguous_files': sum(row['file_status'] == 'ambiguous' for row in rows),
               'unlabelled_files': sum(len(paths) for key, paths in files.items() if key not in known),
               'metadata_status_counts': dict(Counter(row['metadata']['status'] for row in rows)),
               'ffprobe_available': bool(executable), 'ffprobe_requested': bool(use_ffprobe),
               'anchors_after_duration': sum(row['anchor_after_duration'] is True for row in rows),
               'angle_id': 'unknown', 'type': 'unknown'}
    return {'overall': overall, 'per_video': rows}


def _load_records(path):
    rows = read_table(path)
    if not rows:
        raise ValueError('empty CSV table')
    header = rows[0]
    if len(set(header)) != len(header):
        raise ValueError('duplicate CSV column')
    result = []
    for row in rows[1:]:
        if not any(row):
            continue
        if len(row) > len(header):
            raise ValueError('CSV row has extra columns')
        result.append(dict(zip(header, row + [''] * (len(header) - len(row)))))
    return result


def _write_reports(report, output, mode):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    def dump(name, value):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    overview = dict(report, mode=mode)
    rows = overview.pop('per_video')
    dump('overall.json', overview)
    dump('per_video.json', rows)
    with (output / 'per_video.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        fields = list(rows[0]) if rows else ['video']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, allow_nan=False) if isinstance(value, (list, dict)) else value for key, value in row.items()})
    text = '# Локальный ML-отчёт: ' + mode + '\n\n'
    text += 'Не публиковать: отчёт содержит закрытую разметку и идентификаторы.\n\n'
    text += 'Детализация: `per_video.csv`, `per_video.json`; агрегаты: `overall.json`.\n\n'
    text += 'Необработанные, ошибочные и отсутствующие ответы не являются чистыми отрицательными. '
    text += 'Метрики `scored_only` условны по валидным ответам; `end_to_end` учитывает пропущенные положительные. '
    text += 'Порог SLA: <= 300 секунд; неизвестное время не подтверждает SLA. '
    text += 'Сопоставление событий взаимно-однозначное по минимуму суммарной временной ошибки, без порога допуска. '
    text += 'Разметка только бинарными метками даёт ограниченный одиночный временной якорь; тип и ракурс не выводятся из изображения.\n\n'
    text += '```json\n' + json.dumps(overview, ensure_ascii=False, indent=2, allow_nan=False) + '\n```\n'
    (output / 'report.md').write_text(text, encoding='utf-8')


def console_summary(report, mode):
    """Aggregate numbers only: never print media, video IDs or observed actions."""
    overall = report['overall']
    rows = [('total', overall['total'])]
    if mode == 'audit':
        rows.extend((key, overall[key]) for key in
                    ('positive', 'negative', 'matched_files', 'missing_files',
                     'ambiguous_files', 'anchors_after_duration'))
    else:
        rows.extend((key, overall[key]) for key in ('scored', 'coverage'))
        rows.extend((key, overall['end_to_end'][key]) for key in
                    ('accuracy', 'precision', 'recall', 'f1'))
        rows.extend((('false_positives', len(overall['false_positives'])),
                     ('misses', len(overall['misses'])),
                     ('localization_within_tolerance', overall['localization']['within_tolerance_rate_truth']),
                     ('localization_mae_sec', overall['localization']['mae_sec']),
                     ('mean_elapsed_sec', overall['processing_time']['mean_sec']),
                     ('sla_rate_total', overall['sla']['rate_total'])))
    def display(value):
        if value is None:
            return 'n/a'
        return f'{value:.4f}' if isinstance(value, float) else str(value)
    return '\n'.join(['| metric | value |', '| --- | ---: |'] +
                     [f'| {key} | {display(value)} |' for key, value in rows])


def main(argv=None):
    parser = argparse.ArgumentParser(description='Local-only label audit and prediction scoring (stdlib).')
    commands = parser.add_subparsers(dest='command', required=True)
    audit_parser = commands.add_parser('audit')
    score_parser = commands.add_parser('score')
    run_parser = commands.add_parser('run', help='Run actual local analyze(), then score its outputs')
    for command in (audit_parser, score_parser, run_parser):
        command.add_argument('--labels', required=True, type=Path)
        command.add_argument('--output', required=True, type=Path)
    audit_parser.add_argument('--videos-dir', required=True, type=Path)
    audit_parser.add_argument('--no-ffprobe', action='store_true', help='Skip optional local numeric metadata probing')
    score_parser.add_argument('--predictions', required=True, type=Path)
    for command in (score_parser, run_parser):
        command.add_argument('--annotations', type=Path)
        command.add_argument('--split', type=Path)
        command.add_argument('--subset', choices=('test', 'train', 'all'), help='Default: test with split, all otherwise')
    run_parser.add_argument('--videos-dir', required=True, type=Path)
    run_parser.add_argument('--pipeline', default='ml.pipeline:analyze')
    run_parser.add_argument('--time-budget-sec', type=float, default=240)
    run_parser.add_argument('--timeout-sec', type=float, default=300)
    args = parser.parse_args(argv)
    try:
        labels = load_labels(args.labels)
        if args.command == 'audit':
            report = audit(labels, args.videos_dir, use_ffprobe=not args.no_ffprobe)
        else:
            annotations = _load_records(args.annotations) if args.annotations else None
            split = None
            if args.split:
                split = json.loads(args.split.read_text(encoding='utf-8-sig')) if args.split.suffix.lower() == '.json' else _load_records(args.split)
            if args.command == 'run':
                # Validate metadata before starting workers or writing a report.
                score(labels, [], annotations=annotations, split=split, subset=args.subset)
                from .eval_runner import run_pipeline
                predictions = run_pipeline(labels, args.videos_dir, pipeline=args.pipeline,
                                           time_budget_sec=args.time_budget_sec,
                                           timeout_sec=args.timeout_sec, output_dir=args.output)
            else:
                predictions = json.loads(args.predictions.read_text(encoding='utf-8-sig'))
            report = score(labels, predictions, annotations=annotations, split=split, subset=args.subset)
        _write_reports(report, args.output, args.command)
    except (ValueError, OSError, KeyError, ET.ParseError, zipfile.BadZipFile) as error:
        parser.error(str(error))
    # Deliberately no per-video labels, identifiers, descriptions or media in stdout.
    print(console_summary(report, args.command))
    print(json.dumps({'mode': args.command, 'total': report['overall']['total']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
