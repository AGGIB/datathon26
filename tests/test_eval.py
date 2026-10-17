"""Synthetic predictions/labels only; no model inference or competition data."""
import importlib.util
import unittest
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


class ScoreTests(unittest.TestCase):
    def test_invalid_annotation_metadata_is_rejected_explicitly(self):
        from ml.eval import score
        labels = [{'video': 'synthetic', 'positive': True, 'timestamp': 1}]
        cases = [[{'video': 'synthetic', 't_peak': 1, 'type': ['not-a-type']}],
                 [{'video': 'synthetic', 'angle_id': '   '}],
                 [{'video': 'synthetic', 't_peak': 'nan'}],
                 [{'video': 'synthetic', 'type': 'synthetic'}],
                 [{'video': 'synthetic', 'angle_id': 'a'}, {'video': 'synthetic', 'angle_id': 'b'}],
                 [{'video': 'synthetic', 't_peak': 1}, {'video': 'synthetic', 't_peak': 1}],
                 ['not-an-object']]
        for annotations in cases:
            with self.subTest(annotations=annotations):
                with self.assertRaises(ValueError):
                    score(labels, [], annotations=annotations)

    def test_partial_type_annotations_cannot_inflate_precision(self):
        from ml.eval import score
        labels = [{'video': video, 'positive': True, 'timestamp': 1} for video in ('partial_synthetic', 'full_synthetic')]
        annotations = [{'video': 'partial_synthetic', 'type': 'known_synthetic', 't_peak': 1},
                       {'video': 'partial_synthetic', 't_peak': 2},
                       {'video': 'full_synthetic', 'type': 'known_synthetic', 't_peak': 1}]
        predictions = [{'video': row['video'], 'status': 'ok', 'verdict': 'violation', 'violations': [{'type': 'known_synthetic', 't_peak': 1}]} for row in labels]
        types = score(labels, predictions, annotations=annotations)['overall']['types']['known_synthetic']
        self.assertEqual(types['precision'], 1)
        self.assertEqual(types['recall'], 1)
        self.assertEqual(types['precision_denominator'], 1)
        self.assertEqual(types['recall_denominator'], 1)
        self.assertEqual(types['truth_events'], 2)

    def test_unknown_types_do_not_manufacture_zero_type_accuracy(self):
        from ml.eval import score
        labels = [{'video': 'synthetic', 'positive': True, 'timestamp': 10}]
        predictions = [{'video': 'synthetic', 'status': 'ok', 'verdict': 'violation', 'violations': [{'type': 'synthetic_type', 't_peak': 10}]}]
        report = score(labels, predictions)['overall']
        self.assertTrue(report['type_metrics_limited'])
        self.assertIsNone(report['types']['synthetic_type']['precision'])
        self.assertEqual(report['types']['synthetic_type']['precision_denominator'], 0)
        self.assertEqual(report['sla']['unknown_elapsed'], 1)
        self.assertEqual(report['sla']['rate_total'], 0)
        self.assertEqual(report['sla']['timely_correct_rate_total'], 0)
        predictions[0]['elapsed_sec'] = 301
        report = score(labels, predictions)['overall']
        self.assertEqual(report['sla']['completed_late'], 1)
        self.assertEqual(report['processing_time']['max_sec'], 301)
        self.assertEqual(report['sla']['timely_correct_rate_total'], 0)
        predictions[0]['elapsed_sec'] = 300
        report = score(labels, predictions)['overall']
        self.assertEqual(report['sla']['timely_correct_rate_total'], 1)
        self.assertEqual(report['label_counts'], {'positive': 1, 'negative': 0})

    def test_invalid_api_events_abstain_without_clean_negatives(self):
        from ml.eval import score
        labels = [{'video': 'synthetic', 'positive': False, 'timestamp': None}]
        base = {'video': 'synthetic', 'status': 'ok', 'verdict': 'violation', 'violations': [{'type': 'synthetic', 't_start': 1, 't_peak': 2, 't_end': 3}], 'elapsed_sec': 1}
        malformed = [dict(base, violations=[{'t_start': -1, 't_peak': 2}]),
                     dict(base, violations=[{'t_start': 3, 't_peak': 2, 't_end': 4}]),
                     dict(base, violations=[{'t_start': 1, 't_peak': 2, 't_end': 'nan'}]),
                     dict(base, violations=[{'type': {'bad': 'object'}, 't_peak': 2}]),
                     dict(base, violations=[{'t_peak': float('inf')}]),
                     dict(base, elapsed_sec=-1), dict(base, verdict='no_violation'),
                     dict(base, violations=None), dict(base, status='processing'),
                     dict(base, result={'verdict': 'no_violation', 'violations': []})]
        for prediction in malformed:
            with self.subTest(prediction=prediction):
                report = score(labels, [prediction])['overall']
                self.assertEqual(report['status_counts'], {'invalid': 1})
                self.assertEqual(report['confusion']['tn'], 0)
                self.assertEqual(report['confusion']['abstain_negative'], 1)
        with self.assertRaises(ValueError):
            score(labels, [dict(base, video_id='different')])
        with self.assertRaises(ValueError):
            score(labels, [base, dict(base, video='synthetic.mov')])
        self.assertEqual(score(labels, [])['overall']['confusion']['tn'], 0)
        with self.assertRaises(ValueError):
            score(labels, [], annotations=[{'video': 'synthetic', 'type': 'event', 't_peak': 1}])

    def test_explicit_angle_holdout_filters_without_leakage(self):
        from ml.eval import score
        labels = [{'video': name, 'positive': True, 'timestamp': 1} for name in ('train_synthetic', 'test_synthetic')]
        split = [{'video': 'train_synthetic', 'subset': 'train', 'angle_id': 'angle_a'},
                 {'video': 'test_synthetic.mov', 'subset': 'test', 'angle_id': 'angle_b'}]
        predictions = [{'video': name, 'status': 'ok', 'verdict': 'violation', 'violations': []} for name in ('train_synthetic', 'test_synthetic')]
        report = score(labels, predictions, split=split)
        self.assertEqual(report['overall']['total'], 1)
        self.assertEqual(report['per_video'][0]['video'], 'test_synthetic')
        self.assertEqual(report['overall']['angles']['angle_b']['total'], 1)
        self.assertEqual(report['split']['excluded_predictions'], 1)
        self.assertEqual(score(labels, predictions, split=split, subset='all')['overall']['total'], 2)
        self.assertEqual(score(labels, predictions, split=split, subset='train')['per_video'][0]['video'], 'train_synthetic')
        for invalid in [split[:1], [dict(split[0]), dict(split[1], angle_id='angle_a')],
                        [dict(split[0], angle_id='unknown'), split[1]], split + [split[0]]]:
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    score(labels, predictions, split=invalid)
        with self.assertRaises(ValueError):
            score(labels, predictions, subset='test')
        with self.assertRaises(ValueError):
            score(labels, predictions, split=split, annotations=[{'video': 'test_synthetic', 'angle_id': 'angle_a'}])

    def test_annotations_match_events_one_to_one_and_envelopes(self):
        from ml.eval import score
        labels = [{'video': 'synthetic', 'positive': True, 'timestamp': 10}]
        predictions = {'predictions': [{'video_id': 'synthetic.mov', 'status': 'completed', 'elapsed_sec': 2,
                        'result': {'verdict': 'violation', 'violations': [{'type': 'test_type', 't_start': 9, 't_peak': 10, 't_end': 11}]}}]}
        annotations = [{'video': 'synthetic', 'type': 'test_type', 't_peak': 10, 'angle_id': 'camera_synthetic'},
                       {'video': 'synthetic', 'type': 'test_type', 't_peak': 11, 'angle_id': 'camera_synthetic'}]
        report = score(labels, predictions, annotations=annotations)
        localization = report['overall']['localization']
        self.assertEqual(localization['truth_events'], 2)
        self.assertEqual(localization['matched_events'], 1)
        self.assertEqual(localization['unmatched_truth_events'], 1)
        self.assertEqual(localization['mae_sec'], 0)
        self.assertFalse(localization['label_only_limited'])
        self.assertEqual(report['overall']['angles']['camera_synthetic']['total'], 1)
        self.assertEqual(report['overall']['types']['test_type'], {'truth_events': 2, 'predicted_events': 1, 'matched_correct_type': 1, 'precision': 1.0, 'recall': 0.5, 'precision_denominator': 1, 'recall_denominator': 2})
        # The global assignment must beat greedily choosing the closest pair first.
        predictions['predictions'][0]['result']['violations'] = [{'t_peak': 3}, {'t_peak': 4}]
        annotations[0]['t_peak'], annotations[1]['t_peak'] = 0, 3
        self.assertEqual(score(labels, predictions, annotations=annotations)['overall']['localization']['mae_sec'], 2)

    def test_failures_abstain_and_missing_positives_reduce_end_to_end_recall(self):
        self.assertIsNotNone(importlib.util.find_spec('ml.eval'), 'evaluator missing')
        from ml.eval import score
        labels = [{'video': 'p%d' % i, 'positive': True, 'timestamp': 10.0} for i in range(4)] + [{'video': 'n%d' % i, 'positive': False, 'timestamp': None} for i in range(3)]
        predictions = [
            {'video': 'p0.mov', 'status': 'ok', 'verdict': 'нарушение', 'violations': [{'type': 'synthetic', 't_peak': 12}], 'elapsed_sec': 9},
            {'video': 'p1', 'status': 'failed', 'verdict': 'без нарушения', 'violations': []},
            {'video': 'p2', 'status': 'timed_out', 'verdict': 'без нарушения', 'violations': [], 'elapsed_sec': 301},
            {'video': 'n0', 'status': 'ok', 'verdict': 'без нарушения', 'violations': [], 'elapsed_sec': 300},
            {'video': 'n1', 'status': 'ok', 'verdict': 'нарушение', 'violations': [], 'elapsed_sec': 301},
            {'video': 'n2', 'status': 'ok', 'verdict': 'garbage', 'violations': []},
        ]
        report = score(labels, predictions)
        overall = report['overall']
        self.assertEqual(overall['total'], 7)
        self.assertEqual(overall['status_counts'], {'ok': 3, 'failed': 1, 'timed_out': 1, 'missing': 1, 'invalid': 1})
        self.assertEqual(overall['coverage'], 3 / 7)
        self.assertEqual(overall['confusion'], {'tp': 1, 'tn': 1, 'fp': 1, 'fn': 0, 'abstain_positive': 3, 'abstain_negative': 1})
        self.assertEqual(overall['end_to_end']['recall'], 1 / 4)
        self.assertEqual(overall['end_to_end']['accuracy'], 2 / 7)
        self.assertEqual(overall['scored_only']['accuracy'], 2 / 3)
        self.assertEqual(overall['end_to_end']['precision'], 1 / 2)
        self.assertEqual(overall['end_to_end']['f1'], 1 / 3)
        self.assertEqual(overall['misses'], ['p1', 'p2', 'p3'])
        self.assertEqual(overall['false_positives'], ['n1'])
        self.assertEqual(overall['sla']['within_300_sec'], 2)
        self.assertEqual(overall['sla']['total_denominator'], 7)
        self.assertEqual(overall['localization']['matched_events'], 1)
        self.assertEqual(overall['localization']['mae_sec'], 2)
        self.assertTrue(overall['localization']['label_only_limited'])
        self.assertEqual(overall['angles']['unknown']['total'], 7)


class CLITests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, '-m', 'ml.eval', *map(str, args)], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, encoding='utf-8')

    def test_audit_and_score_write_reports_from_synthetic_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            labels = root / 'labels.csv'
            with labels.open('w', encoding='utf-8', newline='') as stream:
                csv.writer(stream).writerows([['video', 'timestamp', 'label'], ['X_synthetic', '22 секунда', 'нарушение'], ['Х_synthetic', '', 'без нарушения']])
            media = root / 'media'
            (media / 'nested').mkdir(parents=True)
            (media / 'nested' / 'X_synthetic.mov').write_bytes(b'')
            audit = self.run_cli('audit', '--labels', labels, '--videos-dir', media, '--output', root / 'audit', '--no-ffprobe')
            self.assertEqual(audit.returncode, 0, audit.stderr)
            self.assertTrue((root / 'audit' / 'overall.json').exists(), 'audit CLI did not generate reports')
            overview = json.loads((root / 'audit' / 'overall.json').read_text(encoding='utf-8'))['overall']
            self.assertEqual(overview['total'], 2)
            self.assertEqual(overview['positive'], 1)
            self.assertEqual(overview['matched_files'], 1)
            self.assertEqual(overview['missing_files'], 1)
            predictions = root / 'predictions.json'
            predictions.write_text(json.dumps({'results': [{'video': 'X_synthetic.mov', 'status': 'ok', 'elapsed_sec': 1,
                'result': {'verdict': 'violation', 'violations': [{'type': 'synthetic', 't_peak': 23}]}}]}), encoding='utf-8')
            annotations = root / 'annotations.csv'
            annotations.write_text('video,type,t_peak,angle_id\nX_synthetic,synthetic,22,angle_a\nХ_synthetic,,,angle_b\n', encoding='utf-8')
            split = root / 'split.json'
            split.write_text(json.dumps({'train': [{'video': 'Х_synthetic', 'angle_id': 'angle_b'}], 'test': [{'video': 'X_synthetic', 'angle_id': 'angle_a'}]}), encoding='utf-8')
            result = self.run_cli('score', '--labels', labels, '--predictions', predictions, '--annotations', annotations, '--split', split, '--output', root / 'score')
            self.assertEqual(result.returncode, 0, result.stderr)
            for name in ('overall.json', 'per_video.json', 'per_video.csv', 'report.md'):
                self.assertTrue((root / 'score' / name).exists(), name)
            report = json.loads((root / 'score' / 'overall.json').read_text(encoding='utf-8'))
            self.assertEqual(report['overall']['end_to_end']['recall'], 1)
            self.assertEqual(report['overall']['localization']['mae_sec'], 1)
            self.assertEqual(report['split']['subset'], 'test')
            rows = json.loads((root / 'score' / 'per_video.json').read_text(encoding='utf-8'))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['video'], 'X_synthetic')
            with (root / 'score' / 'per_video.csv').open(encoding='utf-8-sig', newline='') as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 1)
            bad = self.run_cli('score', '--labels', labels, '--predictions', predictions, '--subset', 'test', '--output', root / 'bad')
            self.assertEqual(bad.returncode, 2)
            self.assertFalse((root / 'bad').exists())


if __name__ == '__main__':
    unittest.main()
