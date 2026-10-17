"""Jury API contract regressions; all records below are artificial fixtures."""
import unittest

from ml.eval import score


class JuryContractTests(unittest.TestCase):
    def test_done_api_record_uses_canonical_timecode(self):
        labels = [{'video': 'fixture', 'positive': True, 'timestamp': 10}]
        predictions = [{'video': 'fixture.mov', 'job_id': 'test-only', 'status': 'done',
                        'verdict': 'violation', 'elapsed_sec': 2,
                        'violations': [{'type': 'fixture-type', 'timecode_sec': 11,
                                        'description': 'Artificial test fixture.',
                                        'evidence_frames': [{'timestamp_sec': 11,
                                                             'image_base64': '/9j/2Q=='}]}]}]
        report = score(labels, predictions)['overall']
        self.assertEqual(report['scored'], 1)
        self.assertEqual(report['confusion']['tp'], 1)
        self.assertEqual(report['localization']['mae_sec'], 1)

    def test_disagreeing_timestamp_aliases_are_invalid(self):
        labels = [{'video': 'fixture', 'positive': True, 'timestamp': 10}]
        predictions = [{'video': 'fixture', 'status': 'ok', 'verdict': 'violation',
                        'violations': [{'type': 'fixture-type', 'timecode_sec': 10,
                                        't_peak': 90}]}]
        report = score(labels, predictions)['overall']
        self.assertEqual(report['status_counts'], {'invalid': 1})
        self.assertEqual(report['confusion']['tp'], 0)

    def test_team_type_ids_and_negative_annotation_rows(self):
        labels = [{'video': 'fixture-p', 'positive': True, 'timestamp': 10},
                  {'video': 'fixture-n', 'positive': False, 'timestamp': None}]
        predictions = [{'video': 'fixture-p', 'status': 'ok', 'verdict': 'violation',
                        'violations': [{'type': 'Display name', 'type_id': 'fixture_type',
                                        'timecode_sec': 11}]},
                       {'video': 'fixture-n', 'status': 'done', 'verdict': 'no_violation',
                        'violations': []}]
        annotations = [{'video': 'fixture-p', 'type_id': 'fixture_type', 't_peak': '10',
                        'angle_id': 'A'},
                       {'video': 'fixture-n', 'type_id': 'none', 't_peak': '',
                        'angle_id': 'B'}]
        report = score(labels, predictions, annotations=annotations)['overall']
        self.assertFalse(report['type_metrics_limited'])
        self.assertEqual(report['types']['fixture_type']['matched_correct_type'], 1)
        self.assertEqual(report['confusion']['tn'], 1)
        self.assertEqual(report['localization']['truth_events'], 1)

    def test_localization_tolerance_and_type_confusion_do_not_reward_far_events(self):
        labels = [{'video': 'fixture', 'positive': True, 'timestamp': 10}]
        predictions = [{'video': 'fixture', 'status': 'ok', 'verdict': 'violation',
                        'violations': [{'type_id': 'wrong_type', 'timecode_sec': 12},
                                       {'type_id': 'fixture_type', 'timecode_sec': 23}]}]
        annotations = [{'video': 'fixture', 'type_id': 'fixture_type', 't_peak': t}
                       for t in (10, 20, 40)]
        report = score(labels, predictions, annotations=annotations)['overall']
        self.assertEqual(report['localization']['within_tolerance'], 1)
        self.assertEqual(report['localization']['within_tolerance_rate_truth'], 1 / 3)
        self.assertEqual(report['types']['fixture_type']['matched_correct_type'], 0)
        self.assertEqual(report['type_confusion']['fixture_type']['wrong_type'], 1)
        self.assertEqual(report['type_confusion']['fixture_type']['<missed>'], 2)

    def test_numeric_console_table_does_not_print_identifiers_or_observations(self):
        import ml.eval as evaluator
        labels = [{'video': 'private-fixture-p', 'positive': True, 'timestamp': 10},
                  {'video': 'private-fixture-n', 'positive': False, 'timestamp': None}]
        report = score(labels, [{'video': 'private-fixture-p', 'status': 'done',
                                'verdict': 'violation', 'elapsed_sec': 1,
                                'violations': [{'type': 'private-category',
                                                'timecode_sec': 10}]}])
        self.assertTrue(hasattr(evaluator, 'console_summary'), 'numeric console table missing')
        table = evaluator.console_summary(report, 'score')
        self.assertIn('| accuracy | 0.5000 |', table)
        self.assertIn('| recall | 1.0000 |', table)
        self.assertNotIn('private-', table)

    def test_done_cannot_claim_violation_with_no_events(self):
        labels = [{'video': 'fixture', 'positive': True, 'timestamp': 10}]
        prediction = {'video': 'fixture', 'status': 'done', 'verdict': 'violation',
                      'violations': []}
        report = score(labels, [prediction])['overall']
        self.assertEqual(report['status_counts'], {'invalid': 1})
        self.assertEqual(report['confusion']['tp'], 0)


if __name__ == '__main__':
    unittest.main()
