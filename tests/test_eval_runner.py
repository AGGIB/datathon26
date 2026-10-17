"""Synthetic subprocess tests only: no model, real media, or model metrics."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import textwrap
import time
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class EvalRunnerTests(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / 'artifacts' / 'eval-runner-tests'
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.videos = self.base / 'videos'
        self.videos.mkdir()
        self.output = self.base / 'output'
        self.modules = self.base / 'modules'
        self.modules.mkdir()
        self.env = patch.dict(os.environ, {
            'PYTHONPATH': str(self.modules) + os.pathsep + os.environ.get('PYTHONPATH', ''),
            'PYTHONDONTWRITEBYTECODE': '1',
        })
        self.env.start()
        self.addCleanup(self.env.stop)

    def fixture(self, body, name='runner_fixture'):
        (self.modules / (name + '.py')).write_text(textwrap.dedent(body), encoding='utf-8')
        return name + ':analyze'

    def label(self, name='synthetic', present=True, positive=True):
        if present:
            (self.videos / (name + '.mov')).write_bytes(b'not-media')
        return {'video': name, 'positive': positive, 'timestamp': 12 if positive else None}

    def run_pipeline(self, labels, **kwargs):
        from ml.eval_runner import run_pipeline
        return run_pipeline(labels, self.videos, output_dir=self.output, **kwargs)

    def test_real_subprocess_wraps_success_measures_wall_time_and_is_quiet(self):
        target = self.fixture('''
            import os
            import time
            from pathlib import Path
            def analyze(video_path, on_progress=None, time_budget_sec=None):
                assert Path(video_path).is_file()
                assert on_progress is None
                assert time_budget_sec == 2
                print('PRIVATE_MEDIA_' + 'A' * 2000000)
                os.write(2, b'PRIVATE_DESCRIPTION')
                time.sleep(0.08)
                return {'verdict': 'no_violation', 'violations': [],
                        'elapsed_sec': 999, 'video': 'forged'}
        ''')
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            records = self.run_pipeline([self.label()], pipeline=target, time_budget_sec=2, timeout_sec=5)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['video'], 'synthetic')
        self.assertEqual(records[0]['status'], 'done')
        self.assertEqual(records[0]['verdict'], 'no_violation')
        self.assertEqual(records[0]['violations'], [])
        self.assertGreaterEqual(records[0]['elapsed_sec'], 0.08)
        self.assertLess(records[0]['elapsed_sec'], 5)
        self.assertEqual(stdout.getvalue() + stderr.getvalue(), '')
        self.assertEqual(json.loads((self.output / 'predictions.json').read_text(encoding='utf-8')), records)
        self.assertEqual(sorted(path.name for path in self.output.iterdir()), ['predictions.json'])

    def test_interrupt_at_second_worker_preserves_exact_first_checkpoint(self):
        from ml.eval_runner import _run_worker
        target = self.fixture('''
            def analyze(video_path, on_progress=None, time_budget_sec=None):
                return {'verdict': 'no_violation', 'violations': []}
        ''')
        self.output.mkdir()
        predictions = self.output / 'predictions.json'
        predictions.write_text('[{"video": "stale"}]', encoding='utf-8')
        snapshots = []

        def interrupt_second(command, env, timeout):
            snapshots.append(json.loads(predictions.read_text(encoding='utf-8')))
            if len(snapshots) == 2:
                raise KeyboardInterrupt()
            return _run_worker(command, env, timeout)

        labels = [self.label('first'), self.label('interrupted')]
        with patch('ml.eval_runner._run_worker', side_effect=interrupt_second), \
                patch('ml.eval_runner.time.perf_counter', return_value=10), \
                self.assertRaises(KeyboardInterrupt):
            self.run_pipeline(labels, pipeline=target)
        expected = [{'status': 'done', 'verdict': 'no_violation', 'violations': [],
                     'video': 'first', 'elapsed_sec': 0}]
        self.assertEqual(json.loads(predictions.read_text(encoding='utf-8')), expected)
        self.assertEqual(snapshots, [[], expected])
        self.assertEqual(sorted(path.name for path in self.output.iterdir()), ['predictions.json'])

    def test_file_failures_are_checkpointed_before_next_worker_interrupt(self):
        labels = [self.label('absent', present=False), self.label('duplicate'), self.label('next')]
        nested = self.videos / 'nested'
        nested.mkdir()
        (nested / 'duplicate.mov').write_bytes(b'not-media')
        expected = [{'video': 'absent', 'status': 'failed', 'error': 'MissingFile', 'elapsed_sec': 0},
                    {'video': 'duplicate', 'status': 'failed', 'error': 'AmbiguousFile', 'elapsed_sec': 0}]
        with patch('ml.eval_runner._run_worker', side_effect=KeyboardInterrupt), \
                patch('ml.eval_runner.time.perf_counter', return_value=10), \
                self.assertRaises(KeyboardInterrupt):
            self.run_pipeline(labels)
        self.assertEqual(json.loads((self.output / 'predictions.json').read_text(encoding='utf-8')), expected)

    def test_request_failure_before_worker_is_checkpointed_and_processing_continues(self):
        from ml.eval_runner import _dump
        expected = [{'status': 'failed', 'error': 'PermissionError', 'video': 'first', 'elapsed_sec': 0}]

        def fail_first_request(path, value):
            if Path(path).name == '0-request.json':
                raise PermissionError('PRIVATE_REQUEST_ERROR')
            return _dump(path, value)

        with patch('ml.eval_runner._dump', side_effect=fail_first_request), \
                patch('ml.eval_runner._run_worker', side_effect=KeyboardInterrupt) as worker, \
                patch('ml.eval_runner.time.perf_counter', return_value=10), \
                self.assertRaises(KeyboardInterrupt):
            self.run_pipeline([self.label('first'), self.label('next')])
        worker.assert_called_once()
        self.assertEqual(json.loads((self.output / 'predictions.json').read_text(encoding='utf-8')), expected)
        self.assertEqual(sorted(path.name for path in self.output.iterdir()), ['predictions.json'])

    def test_audit_failure_before_workers_replaces_previous_run_with_empty_snapshot(self):
        self.output.mkdir()
        predictions = self.output / 'predictions.json'
        predictions.write_text('[{"video": "stale"}]', encoding='utf-8')
        with patch('ml.eval_runner.audit', side_effect=ValueError('PRIVATE_AUDIT_ERROR')), \
                patch('ml.eval_runner._run_worker') as worker, self.assertRaises(ValueError):
            self.run_pipeline([])
        worker.assert_not_called()
        self.assertEqual(json.loads(predictions.read_text(encoding='utf-8')), [])
        self.assertEqual(sorted(path.name for path in self.output.iterdir()), ['predictions.json'])

    def test_checkpoint_publish_failure_keeps_previous_snapshot_and_removes_pending_file(self):
        from ml.eval_runner import _checkpoint
        self.output.mkdir()
        predictions = self.output / 'predictions.json'
        previous = b'[{"video":"completed","status":"failed","error":"ValueError","elapsed_sec":1}]\n'
        predictions.write_bytes(previous)
        with patch('ml.eval_runner.os.replace', side_effect=PermissionError), \
                self.assertRaises(PermissionError):
            _checkpoint(self.output, [])
        self.assertEqual(predictions.read_bytes(), previous)
        self.assertEqual(sorted(path.name for path in self.output.iterdir()), ['predictions.json'])

    def test_parent_strips_extra_event_fields_from_worker_response_before_checkpoint(self):
        target = self.fixture('''
            import json
            import os
            import sys
            from pathlib import Path
            def analyze(video_path, on_progress=None, time_budget_sec=None):
                response = {'status': 'done', 'verdict': 'violation', 'violations': [
                    {'type': 'synthetic', 'timecode_sec': 2.5,
                     'evidence_frames': [{'image_base64': 'PRIVATE_FRAME'}],
                     'description': 'PRIVATE_DESCRIPTION', 'nested': {'raw': 'PRIVATE'}}]}
                Path(sys.argv[-1]).write_text(json.dumps(response), encoding='utf-8')
                os._exit(0)
        ''')
        records = self.run_pipeline([self.label()], pipeline=target)
        self.assertEqual(records[0]['status'], 'done')
        self.assertEqual(set(records[0]['violations'][0]), {'type', 'timecode_sec'})
        self.assertEqual(records[0]['violations'], [{'type': 'synthetic', 'timecode_sec': 2.5}])
        self.assertEqual(json.loads((self.output / 'predictions.json').read_text(encoding='utf-8')), records)
        self.assertEqual(sorted(path.name for path in self.output.rglob('*')), ['predictions.json'])
        self.assertFalse(b'PRIVATE' in (self.output / 'predictions.json').read_bytes())

    def test_missing_and_ambiguous_files_fail_without_skipping_later_labels(self):
        target = self.fixture('''
            def analyze(video_path, on_progress=None, time_budget_sec=None):
                return {'verdict': 'no_violation', 'violations': []}
        ''')
        labels = [self.label('absent', present=False), self.label('duplicate'), self.label('last')]
        nested = self.videos / 'nested'
        nested.mkdir()
        (nested / 'duplicate.mov').write_bytes(b'not-media')
        records = self.run_pipeline(labels, pipeline=target)
        self.assertEqual([r['video'] for r in records], ['absent', 'duplicate', 'last'])
        self.assertEqual([r['status'] for r in records], ['failed', 'failed', 'done'])
        self.assertEqual([r['error'] for r in records[:2]], ['MissingFile', 'AmbiguousFile'])
        for record in records[:2]:
            self.assertNotIn('verdict', record)
            self.assertNotIn('positive', record)
            self.assertGreaterEqual(record['elapsed_sec'], 0)

    def test_worker_exceptions_are_class_only_and_all_labels_are_processed(self):
        target = self.fixture('''
            from pathlib import Path
            def analyze(video_path, on_progress=None, time_budget_sec=None):
                if Path(video_path).stem == 'unimplemented':
                    raise NotImplementedError('SECRET_PRIVATE_ERROR')
                raise RuntimeError('SECRET_PRIVATE_ERROR')
        ''')
        records = self.run_pipeline([self.label('unimplemented'), self.label('broken')], pipeline=target)
        self.assertEqual([r['status'] for r in records], ['failed', 'failed'])
        self.assertEqual([r['error'] for r in records], ['NotImplementedError', 'RuntimeError'])
        self.assertNotIn('SECRET', json.dumps(records))
        self.assertTrue(all('verdict' not in record for record in records))
        imported = self.run_pipeline([self.label()], pipeline='absent_synthetic_module:analyze')
        self.assertEqual(imported[0]['error'], 'ModuleNotFoundError')

    def test_hard_timeout_kills_worker_and_continues(self):
        target = self.fixture('''
            import time
            from pathlib import Path
            def analyze(video_path, on_progress=None, time_budget_sec=None):
                if Path(video_path).stem == 'hang':
                    time.sleep(2)
                    Path(video_path).with_suffix('.escaped').write_text('late')
                return {'verdict': 'no_violation', 'violations': []}
        ''')
        started = time.perf_counter()
        records = self.run_pipeline([self.label('hang'), self.label('next')], pipeline=target,
                                    timeout_sec=0.8)
        self.assertLess(time.perf_counter() - started, 2)
        self.assertEqual([r['status'] for r in records], ['timed_out', 'done'])
        self.assertEqual(records[0]['error'], 'TimeoutExpired')
        self.assertGreaterEqual(records[0]['elapsed_sec'], 0.8)
        self.assertNotIn('verdict', records[0])
        time.sleep(2)
        self.assertFalse((self.videos / 'hang.escaped').exists())

    def test_network_is_denied_even_during_pipeline_import(self):
        # RED connects only to a closed loopback port, never to the Internet.
        target = self.fixture('''
            import socket
            socket.create_connection(('127.0.0.1', 1), timeout=0.05)
            def analyze(video_path, on_progress=None, time_budget_sec=None):
                return {'verdict': 'no_violation', 'violations': []}
        ''')
        records = self.run_pipeline([self.label()], pipeline=target)
        self.assertEqual(records[0]['status'], 'failed')
        self.assertEqual(records[0]['error'], 'PermissionError')

    def test_offline_flags_override_inherited_environment_before_import(self):
        target = self.fixture('''
            import os
            KEYS = ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY')
            assert all(os.environ.get(key) == '1' for key in KEYS)
            def analyze(video_path, on_progress=None, time_budget_sec=None):
                return {'verdict': 'no_violation', 'violations': []}
        ''')
        original = os.environ.get('HF_HUB_OFFLINE')
        with patch.dict(os.environ, {key: '0' for key in
                        ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY')}):
            records = self.run_pipeline([self.label()], pipeline=target)
        self.assertEqual(records[0]['status'], 'done')
        self.assertEqual(os.environ.get('HF_HUB_OFFLINE'), original)

    def test_invalid_returns_are_failed_not_negative_predictions(self):
        target = self.fixture('''
            from pathlib import Path
            def analyze(video_path, on_progress=None, time_budget_sec=None):
                bad = {
                    'empty': {}, 'list': [], 'null': None,
                    'verdict': {'verdict': 'unknown', 'violations': []},
                    'contradiction': {'verdict': 'no_violation', 'violations': [{}]},
                    'positive_empty': {'verdict': 'violation', 'violations': []},
                    'events': {'verdict': 'violation', 'violations': 'bad'},
                    'event': {'verdict': 'violation', 'violations': [42]},
                    'bad_type': {'verdict': 'violation', 'violations': [{'type': []}]},
                    'time_order': {'verdict': 'violation', 'violations': [{'t_start': 3, 'timecode_sec': 2, 't_end': 1}]},
                    'negative_time': {'verdict': 'violation', 'violations': [{'timecode_sec': -1}]},
                    'nan': {'verdict': 'violation', 'violations': [{'timecode_sec': float('nan')}]},
                    'not_json': {'verdict': 'violation', 'violations': [{'x': {1, 2}}]},
                    'status': {'status': 'invented', 'verdict': 'no_violation', 'violations': []},
                }
                return bad[Path(video_path).stem]
        ''')
        names = ['empty', 'list', 'null', 'verdict', 'contradiction', 'positive_empty',
                 'events', 'event', 'bad_type', 'time_order', 'negative_time', 'nan', 'not_json', 'status']
        records = self.run_pipeline([self.label(name) for name in names], pipeline=target)
        self.assertEqual(len(records), len(names))
        for record in records:
            with self.subTest(video=record['video']):
                self.assertEqual(record['status'], 'failed')
                self.assertNotIn('verdict', record)
                self.assertRegex(record['error'], r'^[A-Za-z]+Error$')
        json.dumps(records, allow_nan=False)

    def test_explicit_failures_abstain_and_large_evidence_never_reaches_worker_response(self):
        from ml.eval_runner import _run_worker
        target = self.fixture('''
            from pathlib import Path
            def analyze(video_path, on_progress=None, time_budget_sec=None):
                mode = Path(video_path).stem
                if mode in ('failed', 'timed_out'):
                    return {'status': mode, 'error': 'SECRET_ERROR_TEXT',
                            'verdict': 'no_violation', 'violations': []}
                return {'status': 'done', 'video_id': 'forged', 'verdict': 'violation',
                        'violations': [{'type': 'synthetic', 'timecode_sec': 2.5,
                                        't_start': 1, 't_peak': 2.5, 't_end': 3,
                                        'description': 'PRIVATE_DESCRIPTION',
                                        'observations': ['PRIVATE_OBSERVATION'],
                                        'arbitrary': {'nested': {'raw': 'PRIVATE_NESTED'}},
                                        'evidence_frames': [{'timestamp_sec': 2.5,
                                                             'image_base64': 'A' * 2000000}]}]}
        ''')
        worker_payloads = []

        def inspect_worker_response(command, env, timeout):
            _run_worker(command, env, timeout)
            worker_payloads.append(Path(command[-1]).read_bytes())

        with patch('ml.eval_runner._run_worker', side_effect=inspect_worker_response):
            records = self.run_pipeline([self.label(name) for name in ('failed', 'timed_out', 'positive')],
                                        pipeline=target)
        self.assertEqual([r['status'] for r in records], ['failed', 'timed_out', 'done'])
        self.assertTrue(all('verdict' not in r for r in records[:2]))
        self.assertNotIn('SECRET', json.dumps(records[:2]))
        self.assertNotIn('video_id', records[2])
        event = records[2]['violations'][0]
        expected = {'type': 'synthetic', 'timecode_sec': 2.5, 't_start': 1, 't_peak': 2.5, 't_end': 3}
        self.assertEqual(set(event), set(expected))
        self.assertEqual(event, expected)
        for payload in worker_payloads:
            self.assertLess(len(payload), 4096)
            self.assertFalse(b'PRIVATE_' in payload)
            self.assertFalse(b'image_base64' in payload)
        self.assertEqual(json.loads((self.output / 'predictions.json').read_text(encoding='utf-8')), records)
        self.assertEqual(sorted(path.name for path in self.output.rglob('*')), ['predictions.json'])
        persisted = (self.output / 'predictions.json').read_bytes()
        self.assertLess(len(persisted), 4096)
        self.assertFalse(b'PRIVATE_' in persisted)
        self.assertFalse(b'image_base64' in persisted)

    def test_crashes_and_missing_or_corrupt_responses_do_not_reuse_stale_results(self):
        target = self.fixture('''
            import os
            import sys
            from pathlib import Path
            def analyze(video_path, on_progress=None, time_budget_sec=None):
                mode = Path(video_path).stem
                if mode == 'crash':
                    os._exit(7)
                if mode == 'no_response':
                    os._exit(0)
                if mode == 'corrupt':
                    Path(sys.argv[-1]).write_text('not-json', encoding='utf-8')
                    os._exit(0)
                return {'verdict': 'no_violation', 'violations': []}
        ''')
        self.run_pipeline([self.label('previous')], pipeline=target)
        records = self.run_pipeline([self.label(name) for name in
                                    ('crash', 'no_response', 'corrupt', 'last')], pipeline=target)
        self.assertEqual([r['status'] for r in records], ['failed', 'failed', 'failed', 'done'])
        self.assertEqual([r['error'] for r in records[:3]],
                         ['CalledProcessError', 'FileNotFoundError', 'JSONDecodeError'])
        self.assertEqual(json.loads((self.output / 'predictions.json').read_text(encoding='utf-8')), records)
        self.assertEqual(sorted(path.name for path in self.output.iterdir()), ['predictions.json'])

    def test_budgets_must_be_positive_finite_and_at_most_300(self):
        for options in ({'time_budget_sec': 0}, {'timeout_sec': -1},
                        {'timeout_sec': float('nan')}, {'time_budget_sec': float('inf')},
                        {'timeout_sec': 301}, {'time_budget_sec': 301},
                        {'time_budget_sec': True}, {'timeout_sec': '300'}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.run_pipeline([], **options)
        self.assertEqual(self.run_pipeline([], time_budget_sec=300, timeout_sec=300), [])
        self.assertEqual(self.run_pipeline([], time_budget_sec=3, timeout_sec=2), [])

    def test_pipeline_target_is_a_validated_module_and_single_function_name(self):
        for target in ('', 'ml.pipeline', '../bad:analyze', 'module:func.attr',
                       'module:func;command', 'module:func:extra', 'module:func\n', None, 3):
            with self.subTest(target=target), self.assertRaises(ValueError):
                self.run_pipeline([], pipeline=target)

    def test_nonlocal_output_is_rejected_before_filesystem_access(self):
        from ml.eval_runner import run_pipeline
        with self.assertRaises(ValueError):
            run_pipeline([], self.videos, output_dir='https://127.0.0.1/private')

    def test_json_shaped_but_invalid_worker_responses_fail_safely(self):
        target = self.fixture('''
            import os
            import sys
            from pathlib import Path
            def analyze(video_path, on_progress=None, time_budget_sec=None):
                replies = {'array': '[]', 'null': 'null',
                           'bad_done': '{"status": "done"}',
                           'bad_error': '{"status": "failed", "error": "SECRET raw text"}'}
                Path(sys.argv[-1]).write_text(replies[Path(video_path).stem], encoding='utf-8')
                os._exit(0)
        ''')
        records = self.run_pipeline([self.label(name) for name in
                                    ('array', 'null', 'bad_done', 'bad_error')], pipeline=target)
        self.assertEqual([r['status'] for r in records], ['failed'] * 4)
        self.assertEqual([r['error'] for r in records], ['ValueError'] * 4)
        self.assertNotIn('SECRET', json.dumps(records))

    def test_timeout_stops_pipeline_child_processes(self):
        (self.modules / 'runner_child.py').write_text(
            'import sys, time\nfrom pathlib import Path\ntime.sleep(2)\n'
            'Path(sys.argv[1]).with_suffix(".escaped").write_text("late")\n', encoding='utf-8')
        target = self.fixture('''
            import subprocess
            import sys
            import time
            from pathlib import Path
            def analyze(video_path, on_progress=None, time_budget_sec=None):
                subprocess.Popen([sys.executable, str(Path(__file__).with_name('runner_child.py')), video_path])
                time.sleep(10)
        ''')
        records = self.run_pipeline([self.label('child')], pipeline=target,
                                    time_budget_sec=0.3, timeout_sec=0.8)
        self.assertEqual(records[0]['status'], 'timed_out')
        time.sleep(2.5)
        self.assertFalse((self.videos / 'child.escaped').exists())

    def test_canonical_joins_preserve_literal_ids_fresh_processes_and_private_requests(self):
        target = self.fixture('''
            import json
            import sys
            import tempfile
            from pathlib import Path
            calls = 0
            def analyze(video_path, on_progress=None, time_budget_sec=None):
                global calls
                calls += 1
                request = Path(sys.argv[-2])
                data = json.loads(request.read_text(encoding='utf-8'))
                assert set(data) == {'pipeline', 'video_path', 'time_budget_sec'}
                assert Path(tempfile.gettempdir()).resolve() == request.parent.resolve()
                assert data['pipeline'] not in sys.argv
                return {'verdict': 'violation', 'violations': [{'timecode_sec': calls}]}
        ''')
        labels = [self.label('LatinХ'), self.label('OtherX', positive=False)]
        (self.videos / 'LatinХ.mov').rename(self.videos / 'LatinX.mov')
        old_cwd = Path.cwd()
        try:
            os.chdir(self.base)
            from ml.eval_runner import run_pipeline
            records = run_pipeline(labels, self.videos, pipeline=target, output_dir='output')
        finally:
            os.chdir(old_cwd)
        self.assertEqual([r['video'] for r in records], ['LatinХ', 'OtherX'])
        self.assertEqual([r['status'] for r in records], ['done', 'done'])
        self.assertEqual([r['violations'][0]['timecode_sec'] for r in records], [1, 1])
        self.assertFalse(any('positive' in r or 'timestamp' in r for r in records))


if __name__ == '__main__':
    unittest.main()
