"""End-to-end CLI test with a deliberately failing artificial local pipeline."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class RunCLITests(unittest.TestCase):
    def test_run_scores_real_worker_failure_instead_of_inventing_clean_result(self):
        project = Path(__file__).resolve().parents[1]
        scratch = project / 'artifacts' / 'test-eval-cli'
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            root = Path(directory)
            videos = root / 'videos'
            videos.mkdir()
            (videos / 'fixture.mov').write_bytes(b'artificial test fixture, not video')
            labels = root / 'labels.csv'
            labels.write_text('video,timestamp,label\nfixture,2,нарушение\n', encoding='utf-8')
            (root / 'fixture_cli_pipeline.py').write_text(
                'def analyze(video_path, on_progress=None, time_budget_sec=None):\n'
                '    raise NotImplementedError("artificial failure fixture")\n', encoding='utf-8')
            env = os.environ.copy()
            env['PYTHONPATH'] = str(root) + os.pathsep + env.get('PYTHONPATH', '')
            output = root / 'results'
            process = subprocess.run([sys.executable, '-m', 'ml.eval', 'run',
                                      '--labels', str(labels), '--videos-dir', str(videos),
                                      '--pipeline', 'fixture_cli_pipeline:analyze',
                                      '--output', str(output)],
                                     cwd=project, env=env, capture_output=True,
                                     encoding='utf-8', timeout=20)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertIn('| coverage | 0.0000 |', process.stdout)
            predictions = json.loads((output / 'predictions.json').read_text(encoding='utf-8'))
            self.assertEqual(predictions[0]['status'], 'failed')
            self.assertEqual(predictions[0]['error'], 'NotImplementedError')
            self.assertNotIn('verdict', predictions[0])
            overall = json.loads((output / 'overall.json').read_text(encoding='utf-8'))['overall']
            self.assertEqual(overall['total'], 1)
            self.assertEqual(overall['end_to_end']['recall'], 0)
            self.assertEqual(overall['confusion']['tn'], 0)


if __name__ == '__main__':
    unittest.main()
