"""Artificial analysis bundles only; no competition observations or media."""
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path


class AnalysisImportTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1] / 'artifacts' / 'test-import'
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.analysis = self.root / 'analysis'
        self.videos = self.root / 'videos'
        self.analysis.mkdir()
        self.videos.mkdir()
        self.labels = self.root / 'labels.csv'
        self.labels.write_text('video,timestamp,label\nХ-fixture,2,нарушение\nX-clean,,без нарушения\n', encoding='utf-8')
        files = [('Х-fixture.mov', b'artificial-positive', 1), ('X-clean.mov', b'artificial-negative', 2)]
        manifest = []
        for name, data, source_id in files:
            (self.videos / name).write_bytes(data)
            manifest.append({'id': source_id, 'name': 'external-fixture-name',
                             'sha256': hashlib.sha256(data).hexdigest()})
        self.report = {'sources': [['s1', 'Artificial source', 'Test-only source', 'https://example.invalid']],
                       'types': [['T01', 'Fixture type', 'external-fixture-name', 'Artificial observation',
                                  'Unverified fixture claim', 'Fixture limits', 'Fixture use', ['s1']]],
                       'additional_checks': [],
                       'registry': [{'id': 1, 'name': 'external-fixture-name',
                                     'original_label': 'нарушение', 'timestamp': 2,
                                     'assessment': ['T01', 'Artificial assessment'],
                                     'observation': {'observed_interval': '1-3', 'observation': 'Test-only observation'}},
                                    {'id': 2, 'name': 'other-external-name',
                                     'original_label': 'без нарушения', 'timestamp': None,
                                     'assessment': None, 'observation': {'observation': 'Test-only normal action'}}]}
        (self.analysis / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
        self.save_report()

    def save_report(self):
        (self.analysis / 'report_data.json').write_text(json.dumps(self.report), encoding='utf-8')

    def test_hash_join_preserves_real_names_and_never_marks_law_verified(self):
        import importlib.util
        self.assertIsNotNone(importlib.util.find_spec('ml.import_analysis'), 'local analysis importer missing')
        from ml.import_analysis import import_analysis
        output = self.root / 'imported'
        result = import_analysis(self.analysis, self.labels, self.videos, output)
        self.assertEqual(result['video_count'], 2)
        self.assertEqual(result['typed_positive_count'], 1)
        self.assertEqual(result['unknown_angle_count'], 2)
        self.assertFalse(result['ready_for_angle_holdout'])
        with (output / 'annotations.csv').open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([row['video'] for row in rows], ['Х-fixture', 'X-clean'])
        self.assertEqual([row['type_id'] for row in rows], ['T01', 'none'])
        self.assertEqual(rows[0]['t_peak'], '2.0')
        catalog = json.loads((output / 'catalog.draft.json').read_text(encoding='utf-8'))
        self.assertFalse(catalog[0]['verified'])
        self.assertIsNone(catalog[0]['rule'])
        self.assertEqual(catalog[0]['source_type_code'], 'T01')

    def test_duplicate_content_requires_literal_name_to_disambiguate(self):
        from ml.import_analysis import import_analysis
        payload = b'shared artificial fixture content'
        manifest = []
        for source_id, name in [(1, 'Х-fixture.mov'), (2, 'X-clean.mov')]:
            (self.videos / name).write_bytes(payload)
            manifest.append({'id': source_id, 'name': name[:-4],
                             'sha256': hashlib.sha256(payload).hexdigest()})
        (self.analysis / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
        result = import_analysis(self.analysis, self.labels, self.videos, self.root / 'imported')
        self.assertEqual(result['video_count'], 2)
        self.assertEqual(result['unique_content_count'], 1)
        self.assertEqual(result['duplicate_content_groups'], 1)

    def test_invalid_join_or_timestamp_does_not_write_partial_output(self):
        from ml.import_analysis import import_analysis
        self.report['registry'][0]['timestamp'] = 90
        self.save_report()
        output = self.root / 'invalid-output'
        with self.assertRaises(ValueError):
            import_analysis(self.analysis, self.labels, self.videos, output)
        self.assertFalse(output.exists())

    def test_ambiguous_hash_without_literal_name_is_not_guessed(self):
        from ml.import_analysis import import_analysis
        payload = b'shared artificial fixture content'
        for name in ['Х-fixture.mov', 'X-clean.mov']:
            (self.videos / name).write_bytes(payload)
        manifest = [{'id': source_id, 'name': 'unmatched-fixture',
                     'sha256': hashlib.sha256(payload).hexdigest()} for source_id in [1, 2]]
        (self.analysis / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
        with self.assertRaises(ValueError):
            import_analysis(self.analysis, self.labels, self.videos, self.root / 'invalid-output')
        self.assertFalse((self.root / 'invalid-output').exists())


if __name__ == '__main__':
    unittest.main()
