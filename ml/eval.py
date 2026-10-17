"""Local, dependency-free evaluation. No media is sent to any service."""
from collections import Counter, defaultdict
import math
import argparse
import csv
import json
from fractions import Fraction
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
from xml.etree import ElementTree as ET
import zipfile

from .evaluation_labels import (
    file_sha256, load_labels, parse_timestamp, read_table, source_index,
    validate_content_sha256, video_key,
)


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _metrics(tp, tn, fp, fn, total):
    return {'accuracy': _ratio(tp + tn, total), 'precision': _ratio(tp, tp + fp),
            'recall': _ratio(tp, tp + fn), 'f1': _ratio(2 * tp, 2 * tp + fp + fn),
            'accuracy_denominator': total, 'positive_denominator': tp + fn}


def _canonical_events(events):
    """Stable inventory order, including type/interval tie breaks and absent times."""
    def key(event):
        def time(field):
            value = event.get(field)
            return (value is None, value if value is not None else 0.0)
        return time('t_peak'), event['type'], time('t_start'), time('t_end')
    return sorted(events, key=key)


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
        result.update(status='ok', positive=verdicts[verdict], events=_canonical_events(parsed))
    except (ValueError, TypeError) as error:
        result['error'] = str(error)
    return result


def _match_events(truth, predicted, tolerance_sec):
    """Maximize pairs inside inclusive tolerance, then minimize their time error."""
    truth = _canonical_events(e for e in truth if e['t_peak'] is not None)
    predicted = _canonical_events(e for e in predicted if e['t_peak'] is not None)
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
            if i and j and abs(truth[i - 1]['t_peak'] - predicted[j - 1]['t_peak']) <= tolerance_sec:
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
    source_index(annotations or [], 'annotations', repeated=True)
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
    for target in indexed.values():
        target['events'] = _canonical_events(target['events'])
    return indexed


def _content_hash_index(labels, content_hashes):
    """Resolve complete label hashes and/or caller-verified actual file hashes."""
    known = {video_key(row['video']) for row in labels}
    supplied = {video_key(row['video']): validate_content_sha256(row['content_sha256'])
                for row in labels if 'content_sha256' in row}
    if content_hashes is None:
        if supplied and set(supplied) != known:
            raise ValueError('content_sha256 required for every label when hashes are supplied')
        return supplied
    if not isinstance(content_hashes, dict):
        raise ValueError('content_hashes must map every video to its actual content_sha256')
    actual = {}
    for literal, digest in content_hashes.items():
        key = video_key(literal)
        if literal in known:
            if key in known and key != literal:
                raise ValueError('ambiguous literal/canonical video identity in content_hashes')
            key = literal
        if key in actual:
            raise ValueError('canonical video collision in content_hashes')
        actual[key] = validate_content_sha256(digest)
    if set(actual) != known:
        raise ValueError('actual content_sha256 required for every known video, and no unknown videos')
    if any(actual[key] != digest for key, digest in supplied.items()):
        raise ValueError('label content_sha256 disagrees with actual content hashes')
    return actual


def _validate_split(labels, split, annotations, content_hashes=None):
    if isinstance(split, dict):
        if set(split) not in ({'train', 'test'}, {'train', 'holdout'},
                              {'train', 'validation', 'test'}, {'train', 'validation', 'holdout'}):
            raise ValueError('split JSON must have train, optional validation, and test/holdout lists')
        records = []
        for side, values in split.items():
            if not isinstance(values, list):
                raise ValueError('split sides must be lists')
            if not values:
                raise ValueError('empty split partition')
            for value in values:
                if not isinstance(value, dict):
                    raise ValueError('split entries must include video, angle_id and content_sha256')
                records.append(dict(value, subset='test' if side == 'holdout' else side))
        split = records
    if not isinstance(split, list):
        raise ValueError('split must be a list or partition object')
    source_index(split, 'split')
    known = {video_key(row['video']) for row in labels}
    result = {}
    angles = {'train': set(), 'test': set()}
    if any(row.get('subset') == 'validation' for row in split):
        angles['validation'] = set()
    # Validate every angle before hashes, so malformed camera IDs cannot be masked.
    for row in split:
        key = video_key(row.get('video', row.get('video_id')))
        side = row.get('subset')
        angle = row.get('angle_id')
        if key in result or key not in known or side not in angles:
            raise ValueError('split must assign every known video exactly once to train/validation/test')
        if not isinstance(angle, str) or not angle.strip() or angle != angle.strip() or angle.lower() in ('unknown', 'неизвестно', 'unassigned'):
            raise ValueError('explicit known angle_id required for every split video')
        annotation_angle = annotations.get(key, {}).get('angle_id', 'unknown')
        if annotation_angle != 'unknown' and annotation_angle != angle:
            raise ValueError('split angle disagrees with annotation')
        result[key] = {'subset': side, 'angle_id': angle}
        angles[side].add(angle)
    if set(result) != known or not all(angles.values()):
        raise ValueError('incomplete or empty angle holdout')
    sides = sorted(angles)
    if any(angles[a] & angles[b] for i, a in enumerate(sides) for b in sides[i + 1:]):
        raise ValueError('leaking angle holdout')
    contents = {}
    for row in split:
        key = video_key(row.get('video', row.get('video_id')))
        digest = validate_content_sha256(row.get('content_sha256'))
        if content_hashes and digest != content_hashes[key]:
            raise ValueError('split content_sha256 disagrees with supplied/actual content hashes')
        side = result[key]['subset']
        if digest in contents and contents[digest] != side:
            raise ValueError('duplicate content crosses split partitions')
        contents[digest] = side
        result[key]['content_sha256'] = digest
    return result, {side: sorted(values) for side, values in angles.items()}


def _score_videos(labels, predictions, annotations=None, split=None, subset=None, tolerance_sec=2.0,
                  content_hashes=None):
    """Score all labels; optional explicit event and angle annotations override anchors."""
    source_index(labels, 'labels')
    hashes = _content_hash_index(labels, content_hashes)
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
    if subset not in ('train', 'validation', 'test', 'all') or (split is None and subset != 'all'):
        raise ValueError('train/validation/test subset requires an explicit split')
    partition, split_angles = _validate_split(labels, split, annotation_index, hashes) if split is not None else ({}, {})
    if subset != 'all' and subset not in split_angles:
        raise ValueError('requested subset is absent from split')
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
        # Team CSVs repeat the primary binary anchor with a blank/unknown type;
        # this is not evidence of a separate, exhaustive event inventory.
        anchor_echo = (len(truth_events) == 1 and truth_events[0]['type'] == 'unknown'
                       and truth_events[0]['t_peak'] == label['timestamp'])
        label_limited = truth and (not truth_events or anchor_echo)
        limited = limited or label_limited
        if label_limited and label['timestamp'] is not None:
            truth_events = [{'t_peak': label['timestamp'], 'type': 'unknown'}]
        matches = _match_events(truth_events, pred['events'], tolerance_sec)
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
                      'excluded_predictions': excluded_predictions, 'angle_holdout_validated': split is not None,
                      'content_holdout_validated': split is not None,
                      'calibration_partition_available': 'validation' in split_angles,
                      'hashes_authoritatively_verified': split is not None and content_hashes is not None}}


def _primary_anchor_type(members):
    """Same-type single events echoing each binary anchor are alternatives.

    This does not merge differing types or multi-event annotation inventories.
    """
    if len(members) < 2 or not all(row['truth_positive'] for row in members):
        return None
    kinds = set()
    for row in members:
        events = row['truth_events']
        if len(events) != 1 or events[0]['t_peak'] != row['truth_timestamp']:
            return None
        kinds.add(events[0]['type'])
    return next(iter(kinds)) if len(kinds) == 1 else None


def _validate_content_consistency(rows, hashes):
    """Check every known-hash member, even when other videos have no actual hash."""
    groups = defaultdict(list)
    for row in rows:
        key = video_key(row['video'])
        if key in hashes:
            groups[hashes[key]].append(row)
    for members in groups.values():
        if len({row['truth_positive'] for row in members}) != 1:
            raise ValueError('conflicting binary labels for exact duplicate content')
        inventories = [row['truth_events'] if not row['label_only_localization'] else [] for row in members]
        if (len({json.dumps(_canonical_events(events), sort_keys=True)
                 for events in inventories}) > 1 and _primary_anchor_type(members) is None):
            raise ValueError('conflicting explicit annotations for exact duplicate content')


def _group_prediction(members):
    """Union successful inventories without multiplying identical events across duplicate runs.

    Keep the maximum multiplicity seen in one member, so an actual repeated prediction is still
    counted. Differences stay visible in metadata; they do not invalidate an entire dataset report.
    """
    successful = [row for row in members if row['status'] == 'ok']
    counts, events = Counter(), {}
    for row in successful:
        inventory = Counter(json.dumps(event, sort_keys=True) for event in row['predicted_events'])
        counts |= inventory
        events.update({json.dumps(event, sort_keys=True): event for event in row['predicted_events']})
    combined = _canonical_events(events[key] for key, count in counts.items() for _ in range(count))
    differ = len({json.dumps([row['predicted_positive'], row['predicted_events']], sort_keys=True)
                  for row in successful}) > 1
    return any(row['predicted_positive'] for row in successful), combined, differ


def score(labels, predictions, annotations=None, split=None, subset=None, tolerance_sec=2.0,
          content_hashes=None):
    """Keep literal-video diagnostics; count each known exact content once in quality."""
    report = _score_videos(labels, predictions, annotations, split, subset, tolerance_sec,
                           content_hashes)
    hashes = _content_hash_index(labels, content_hashes)
    report['video_only_overall'] = report['overall']
    report['video_only_overall']['aggregation_unit'] = 'video_diagnostic'
    report['content_aggregation'] = {
        'available': bool(hashes), 'hash_source': 'actual' if content_hashes is not None else ('labels' if hashes else None),
        'hashes_authoritatively_verified': content_hashes is not None,
        'reason': None if hashes else 'complete_content_hashes_not_supplied',
    }
    report['per_content_group'] = []
    consistency_hashes = hashes
    if not hashes and split is not None:
        partition, _ = _validate_split(labels, split, _annotation_index(labels, annotations))
        consistency_hashes = {key: row['content_sha256'] for key, row in partition.items()}
    if consistency_hashes:
        all_rows = report['per_video']
        if split is not None and report['split']['subset'] != 'all':
            all_rows = _score_videos(labels, predictions, annotations, tolerance_sec=tolerance_sec,
                                     content_hashes=content_hashes)['per_video']
        _validate_content_consistency(all_rows, consistency_hashes)
    if not hashes:
        return report
    groups = defaultdict(list)
    for row in report['per_video']:
        groups[hashes[video_key(row['video'])]].append(row)
    group_labels, group_predictions, group_annotations, metadata = [], [], [], {}
    for digest, members in groups.items():
        first = members[0]
        video = first['video']
        positive, events, predictions_differ = _group_prediction(members)
        anchors = sorted({row['truth_timestamp'] for row in members
                          if row['truth_positive'] and row['truth_timestamp'] is not None})
        anchor = None
        if anchors:
            peaks = [event['t_peak'] for event in events if event['t_peak'] is not None]
            anchor = min(anchors, key=lambda t: (min((abs(t - peak) for peak in peaks), default=math.inf), t))
        group_labels.append({'video': video, 'positive': first['truth_positive'], 'timestamp': anchor})
        status = min((row['status'] for row in members),
                     key=lambda value: ('invalid', 'timed_out', 'failed', 'missing', 'ok').index(value))
        elapsed = [row['elapsed_sec'] for row in members]
        elapsed = max(elapsed) if all(value is not None for value in elapsed) else None
        if status != 'missing':
            group_predictions.append({'video': video, 'status': status,
                                      'verdict': 'violation' if positive else 'no_violation',
                                      'violations': events if status == 'ok' else [],
                                      'elapsed_sec': elapsed})
        angle_ids = sorted({row['angle_id'] for row in members})
        angle = angle_ids[0] if len(angle_ids) == 1 else 'multiple'
        anchor_type = _primary_anchor_type(members)
        if anchor_type is not None and anchor_type != 'unknown':
            group_annotations.append({'video': video, 'angle_id': angle,
                                      't_peak': anchor, 'type': anchor_type})
        elif first['truth_events'] and not first['label_only_localization']:
            group_annotations.extend(dict(event, video=video, angle_id=angle) for event in first['truth_events'])
        else:
            group_annotations.append({'video': video, 'angle_id': angle})
        metadata[video] = {'content_sha256': digest, 'videos': [row['video'] for row in members],
                           'member_predictions_differ': predictions_differ,
                           'prediction_aggregation': 'successful_inventory_union',
                           'angle_ids': angle_ids, 'alternative_anchors_sec': anchors,
                           'member_status_counts': dict(Counter(row['status'] for row in members))}
    grouped = _score_videos(group_labels, group_predictions, annotations=group_annotations,
                            tolerance_sec=tolerance_sec)
    report['overall'] = dict(grouped['overall'], aggregation_unit='content_group')
    report['per_content_group'] = [dict(row, **metadata[row['video']]) for row in grouped['per_video']]
    report['content_aggregation'].update(total=report['overall']['total'],
                                         selected_videos=len(report['per_video']))
    return report


def _probe(path, executable):
    """Only numerical metadata; never decode frames, export tags, or call a network API."""
    command = [executable, '-v', 'error', '-protocol_whitelist', 'file,pipe', '-show_entries',
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


def _reject_link(metadata):
    if (stat.S_ISLNK(metadata.st_mode)
            or getattr(metadata, 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT):
        raise ValueError('videos-dir must not contain symlinks or reparse points')


def _audit_directory(value):
    """Check locality and every ancestor top-down, without resolving link targets."""
    text = os.fspath(value)
    if not isinstance(text, str) or text.startswith(('\\\\', '//')) or (
            re.match(r'^[A-Za-z][A-Za-z0-9+.-]*:', text)
            and not re.match(r'^[A-Za-z]:[\\/]', text)):
        raise ValueError('only local filesystem paths are allowed')
    path = Path(text)
    if not path.is_absolute():
        path = Path.cwd() / path
    # macOS exposes its own temporary roots through these immutable system aliases.
    # Canonicalize only the verified OS alias; retain no-follow checks for every user link.
    if sys.platform == 'darwin' and len(path.parts) > 1 and path.parts[1] in ('var', 'tmp'):
        alias = Path('/') / path.parts[1]
        target = Path('/private') / path.parts[1]
        if alias.is_symlink() and alias.parent / alias.readlink() == target:
            path = target.joinpath(*path.parts[2:])
    if os.name == 'nt':
        import ctypes
        if ctypes.windll.kernel32.GetDriveTypeW(str(path.anchor)) == 4:
            raise ValueError('network drives are not allowed')
    # Do not resolve/abspath first: even link/../child must reject the link.
    for ancestor in (*reversed(path.parents), path):
        metadata = ancestor.lstat()
        _reject_link(metadata)
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError('videos-dir must be a directory')
    return path


def _audit_files(directory):
    """Inventory no-follow metadata completely before any probe or content read."""
    pending, files = [directory], []
    while pending:
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                metadata = entry.stat(follow_symlinks=False)
                _reject_link(metadata)
                path = Path(entry.path)
                if stat.S_ISDIR(metadata.st_mode):
                    pending.append(path)
                elif stat.S_ISREG(metadata.st_mode) and (path.suffix.lower() == '.mov' or not path.suffix):
                    files.append(path)
    return sorted(files)


def audit(labels, videos_dir, use_ffprobe=True):
    """Canonical metadata joins; matched_paths always retain literal filesystem names."""
    source_index(labels, 'labels')
    directory = _audit_directory(videos_dir)
    files = defaultdict(list)
    source_names = {}
    for path in _audit_files(directory):
        key = video_key(path.name)
        if key in source_names and source_names[key] != path.name:
            raise ValueError('canonical video collision in local files')
        source_names[key] = path.name
        files[key].append(path)
    executable = shutil.which('ffprobe') if use_ffprobe else None
    rows = []
    for label in labels:
        matches = files.get(video_key(label['video']), [])
        state = 'matched' if len(matches) == 1 else ('missing' if not matches else 'ambiguous')
        metadata = _probe(matches[0], executable) if executable and len(matches) == 1 else {'status': 'disabled' if not use_ffprobe else ('unavailable' if not executable else 'not_probed')}
        duration = metadata.get('duration_sec')
        rows.append(dict(label, video_key=video_key(label['video']),
                         content_sha256=file_sha256(matches[0]) if len(matches) == 1 else None,
                         file_status=state, matched_paths=[str(path.relative_to(directory)) for path in matches],
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
    groups = defaultdict(list)
    for row in rows:
        if row['content_sha256'] is not None:
            groups[row['content_sha256']].append(row)
    duplicates = []
    for digest, members in groups.items():
        if len(members) > 1:
            anchors = sorted({row['timestamp'] for row in members if row['timestamp'] is not None})
            duplicates.append({'content_sha256': digest, 'videos': [row['video'] for row in members],
                               'alternative_anchors_sec': anchors if all(row['positive'] for row in members) else [],
                               'anchors_sec': anchors, 'conflicting_anchors': len(anchors) > 1,
                               'conflicting_labels': len({row['positive'] for row in members}) > 1})
    overall.update(content_groups=len(groups), duplicate_groups=len(duplicates),
                   content_hashes_complete=all(row['content_sha256'] is not None for row in rows))
    return {'overall': overall, 'per_video': rows, 'duplicate_groups': duplicates}


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
    text += 'Сопоставление сначала максимизирует число пар внутри включительного допуска, затем минимизирует временную ошибку. '
    text += 'При полных SHA256 качество считается один раз на content_group; без них доступны только video_diagnostic, не deduped quality. '
    text += 'Все исходные видео и отказы сохраняются отдельно; успешная копия не заменяет missing/failed output другой копии. '
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
        rows.append(('content_aggregation_available', int(report['content_aggregation']['available'])))
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
    score_parser.add_argument('--videos-dir', type=Path, help='Hash actual local matched files before scoring')
    for command in (score_parser, run_parser):
        command.add_argument('--annotations', type=Path)
        command.add_argument('--split', type=Path)
        command.add_argument('--subset', choices=('test', 'train', 'validation', 'all'), help='Default: test with split, all otherwise')
    run_parser.add_argument('--videos-dir', required=True, type=Path)
    run_parser.add_argument('--pipeline', default='ml.pipeline:analyze')
    run_parser.add_argument('--time-budget-sec', type=float, default=240)
    run_parser.add_argument('--timeout-sec', type=float, default=300)
    args = parser.parse_args(argv)
    try:
        if args.videos_dir is not None:
            args.videos_dir = _audit_directory(args.videos_dir)
        labels = load_labels(args.labels)
        if args.command == 'audit':
            report = audit(labels, args.videos_dir, use_ffprobe=not args.no_ffprobe)
        else:
            annotations = _load_records(args.annotations) if args.annotations else None
            split = None
            if args.split:
                split = json.loads(args.split.read_text(encoding='utf-8-sig')) if args.split.suffix.lower() == '.json' else _load_records(args.split)
            content_hashes, hashes_complete, scoring_labels = None, None, labels
            known_hashes = {}
            if args.videos_dir is not None:
                audited = audit(labels, args.videos_dir, use_ffprobe=False)
                hashes_complete = audited['overall']['content_hashes_complete']
                actual = {row['video']: row['content_sha256'] for row in audited['per_video']}
                known_hashes = {video_key(video): digest for video, digest in actual.items() if digest is not None}
                for label in labels:
                    expected = label.get('content_sha256')
                    observed = actual[label['video']]
                    if expected is not None and observed is not None and expected != observed:
                        raise ValueError('label content_sha256 disagrees with actual file hash')
                if hashes_complete:
                    content_hashes = actual
                elif split is not None:
                    raise ValueError('actual content_sha256 required for every split video; local files missing/ambiguous')
                else:
                    # Preserve file-failure coverage, but never substitute self-reports for missing actual hashes.
                    scoring_labels = [{key: value for key, value in label.items() if key != 'content_sha256'}
                                      for label in labels]
            # Validate all metadata and actual bytes before workers or report writes.
            preflight = score(scoring_labels, [], annotations=annotations, split=split, subset=args.subset,
                              content_hashes=content_hashes)
            if hashes_complete is False:
                _validate_content_consistency(preflight['per_video'], known_hashes)
            if args.command == 'run':
                from .eval_runner import run_pipeline
                predictions = run_pipeline(labels, args.videos_dir, pipeline=args.pipeline,
                                           time_budget_sec=args.time_budget_sec,
                                           timeout_sec=args.timeout_sec, output_dir=args.output)
            else:
                predictions = json.loads(args.predictions.read_text(encoding='utf-8-sig'))
            report = score(scoring_labels, predictions, annotations=annotations, split=split, subset=args.subset,
                           content_hashes=content_hashes)
            if hashes_complete is False:
                _validate_content_consistency(report['per_video'], known_hashes)
                report['content_aggregation']['reason'] = 'actual_content_hashes_incomplete'
        _write_reports(report, args.output, args.command)
    except (ValueError, OSError, KeyError, ET.ParseError, zipfile.BadZipFile) as error:
        parser.error(str(error))
    # Deliberately no per-video labels, identifiers, descriptions or media in stdout.
    print(console_summary(report, args.command))
    print(json.dumps({'mode': args.command, 'total': report['overall']['total']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
