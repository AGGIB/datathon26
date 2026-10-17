"""Local subprocess runner; media and pipeline console output never enter logs."""
import importlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

from .eval import audit


_ROOT = Path(__file__).resolve().parents[1]


def _target(value):
    identifier = r'[A-Za-z_][A-Za-z0-9_]*'
    if not isinstance(value, str) or not re.fullmatch(identifier + r'(?:\.' + identifier + ')*:' + identifier, value):
        raise ValueError('pipeline must be module:function')
    return value.split(':')


def _dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def _checkpoint(output, records):
    with tempfile.NamedTemporaryFile(prefix='.predictions-', dir=output, delete=False) as stream:
        pending = Path(stream.name)
    try:
        _dump(pending, records)
        os.replace(pending, output / 'predictions.json')
    finally:
        pending.unlink(missing_ok=True)


def _local_path(value):
    text = os.fspath(value)
    if not isinstance(text, str) or text.startswith(('\\\\', '//')) or (
            re.match(r'^[A-Za-z][A-Za-z0-9+.-]*:', text)
            and not re.match(r'^[A-Za-z]:[\\/]', text)):
        raise ValueError('only local filesystem paths are allowed')
    path = Path(text).absolute()
    if os.name == 'nt':
        import ctypes
        if ctypes.windll.kernel32.GetDriveTypeW(str(path.anchor)) == 4:
            raise ValueError('network drives are not allowed')
    path = path.resolve()
    if str(path).startswith(('\\\\', '//')):
        raise ValueError('network paths are not allowed')
    return path


def _run_worker(command, env, timeout):
    flags = (subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW) if os.name == 'nt' else 0
    with subprocess.Popen(command, cwd=str(_ROOT), env=env, shell=False,
                          stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL, creationflags=flags,
                          start_new_session=os.name != 'nt') as process:
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            # Stop ordinary descendants (e.g. ffmpeg), not only the interpreter.
            try:
                if os.name == 'nt':
                    taskkill = Path(os.environ.get('SystemRoot', r'C:\Windows')) / 'System32' / 'taskkill.exe'
                    subprocess.run([str(taskkill), '/PID', str(process.pid), '/T', '/F'],
                                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, timeout=3, check=False, shell=False)
                else:
                    import signal
                    os.killpg(process.pid, signal.SIGKILL)
            except (OSError, subprocess.SubprocessError):
                pass
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait()
            raise
        if code:
            raise subprocess.CalledProcessError(code, command)


def run_pipeline(labels, videos_dir, pipeline='ml.pipeline:analyze',
                 time_budget_sec=240, timeout_sec=300, output_dir=None):
    """Run one fresh local interpreter per label and return JSON-safe records."""
    _target(pipeline)
    for value in (time_budget_sec, timeout_sec):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 300:
            raise ValueError('budgets must be finite numbers in (0, 300]')
    directory = _local_path(videos_dir)
    output = _local_path(output_dir if output_dir is not None else _ROOT / 'artifacts' / 'eval-runner')
    output.mkdir(parents=True, exist_ok=True)
    _checkpoint(output, [])
    rows = audit(labels, directory, use_ffprobe=False)['per_video']
    records = []
    with tempfile.TemporaryDirectory(prefix='run-', dir=output) as temporary:
        temporary = Path(temporary)
        for index, row in enumerate(rows):
            started = time.perf_counter()
            if row['file_status'] != 'matched':
                records.append({'video': row['video'], 'status': 'failed',
                                'error': 'MissingFile' if row['file_status'] == 'missing' else 'AmbiguousFile',
                                'elapsed_sec': time.perf_counter() - started})
                _checkpoint(output, records)
                continue
            request = temporary / (str(index) + '-request.json')
            response = temporary / (str(index) + '-response.json')
            try:
                _dump(request, {'pipeline': pipeline, 'video_path': str(directory / row['matched_paths'][0]),
                                'time_budget_sec': time_budget_sec})
                env = os.environ.copy()
                env.update({key: '1' for key in
                            ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY')})
                env['PYTHONPATH'] = str(_ROOT) + os.pathsep + env.get('PYTHONPATH', '')
                env.update({key: str(temporary) for key in ('TMPDIR', 'TEMP', 'TMP')})
                _run_worker([sys.executable, '-m', 'ml.eval_runner', '_worker', str(request), str(response)],
                            env, max(0.001, timeout_sec - (time.perf_counter() - started)))
                result = _read_response(response)
            except subprocess.TimeoutExpired:
                result = {'status': 'timed_out', 'error': 'TimeoutExpired'}
            except (OSError, subprocess.CalledProcessError, ValueError, TypeError) as error:
                result = {'status': 'failed', 'error': type(error).__name__}
            records.append(dict(result, video=row['video'], elapsed_sec=time.perf_counter() - started))
            _checkpoint(output, records)
    return records


def _deny_network():
    import ipaddress
    import socket

    # Permit local runtime initialization, not connections, sends or name queries.
    # CPython audit events also cover direct _socket use. This is not an OS sandbox.
    def guard(event, args):
        if not event.startswith('socket.') or event in ('socket.__new__', 'socket.gethostname'):
            return
        if event == 'socket.bind':
            local, address = args
            if local.family in (socket.AF_INET, socket.AF_INET6) and isinstance(address, tuple):
                try:
                    if ipaddress.ip_address(address[0]).is_loopback:
                        return
                except (ValueError, TypeError, IndexError):
                    pass
        raise PermissionError()
    sys.addaudithook(guard)


def _validated_result(result):
    if not isinstance(result, dict):
        raise ValueError()
    status = result.get('status', 'done')
    if status in ('failed', 'timed_out'):
        return {'status': status, 'error': 'RuntimeError' if status == 'failed' else 'TimeoutExpired'}
    if status != 'done':
        raise ValueError()
    verdict, events = result.get('verdict'), result.get('violations')
    if verdict not in ('violation', 'no_violation') or not isinstance(events, list):
        raise ValueError()
    if bool(events) != (verdict == 'violation'):
        raise ValueError()
    for event in events:
        if not isinstance(event, dict):
            raise ValueError()
        kind = event.get('type')
        if kind not in (None, '') and (not isinstance(kind, str) or not kind.strip()):
            raise ValueError()
        for key in ('timecode_sec', 't_start', 't_peak', 't_end'):
            value = event.get(key)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not math.isfinite(value) or value < 0):
                raise ValueError()
        peak = event.get('timecode_sec') if event.get('timecode_sec') is not None else event.get('t_peak')
        ordered = [value for value in (event.get('t_start'), peak, event.get('t_end')) if value is not None]
        if ordered != sorted(ordered):
            raise ValueError()
    # Reject NaN, non-JSON data, cycles, etc., inside the worker's exception boundary.
    json.dumps(result, allow_nan=False)
    return {'status': 'done', 'verdict': verdict,
            'violations': [{key: event[key] for key in
                            ('type', 'timecode_sec', 't_start', 't_peak', 't_end') if key in event}
                           for event in events]}


def _read_response(path):
    result = json.loads(path.read_text(encoding='utf-8'))
    if isinstance(result, dict) and result.get('status') in ('failed', 'timed_out'):
        error = result.get('error')
        if not isinstance(error, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{0,79}', error):
            raise ValueError()
        return {'status': result['status'], 'error': error}
    return _validated_result(result)


def _worker(request, response):
    _deny_network()
    try:
        data = json.loads(Path(request).read_text(encoding='utf-8'))
        module, function = _target(data['pipeline'])
        analyze = getattr(importlib.import_module(module), function)
        video_path = _local_path(data['video_path'])
        if not video_path.is_file():
            raise FileNotFoundError()
        result = analyze(str(video_path), on_progress=None, time_budget_sec=data['time_budget_sec'])
        result = _validated_result(result)
    except Exception as error:
        result = {'status': 'failed', 'error': type(error).__name__}
    _dump(response, result)


if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[1] != '_worker':
        raise SystemExit(2)
    _worker(sys.argv[2], sys.argv[3])
