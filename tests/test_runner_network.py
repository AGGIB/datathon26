"""Isolated real workers, synthetic inputs, and no outbound OS contact."""
import importlib.util
import json
import os
from pathlib import Path
import socket
import tempfile
import textwrap

import pytest

from ml.eval_runner import run_pipeline


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def worker(monkeypatch):
    scratch = ROOT / 'artifacts' / 'runner-network-tests'
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as temporary:
        base = Path(temporary)
        videos = base / 'videos'
        videos.mkdir()
        (videos / 'synthetic.mov').write_bytes(b'not-media')
        monkeypatch.setenv('PYTHONPATH', str(base) + os.pathsep + os.environ.get('PYTHONPATH', ''))
        monkeypatch.setenv('PYTHONDONTWRITEBYTECODE', '1')

        def run(import_body, timeout=5):
            (base / 'network_fixture.py').write_text(
                textwrap.dedent(import_body) + '\n'
                'def analyze(video_path, on_progress=None, time_budget_sec=None):\n'
                "    return {'verdict': 'no_violation', 'violations': []}\n",
                encoding='utf-8')
            output = base / 'output'
            records = run_pipeline(
                [{'video': 'synthetic', 'positive': False, 'timestamp': None}], videos,
                pipeline='network_fixture:analyze', timeout_sec=timeout, output_dir=output)
            assert json.loads((output / 'predictions.json').read_text(encoding='utf-8')) == records
            return records[0]

        yield run


def test_worker_permits_hostname_socket_creation_and_loopback_bind(worker):
    result = worker('''
        import socket
        assert isinstance(socket.gethostname(), str)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as local:
            local.bind(('127.0.0.1', 0))
            assert local.getsockname()[0] == '127.0.0.1'
    ''')
    assert result['status'] == 'done', result
    assert result['verdict'] == 'no_violation'
    assert result['violations'] == []


@pytest.mark.parametrize('module', ['socket', '_socket'])
def test_direct_and_public_socket_local_initialization(worker, module):
    result = worker(f'''
        import {module} as api
        assert isinstance(api.gethostname(), str)
        local = api.socket(api.AF_INET, api.SOCK_STREAM)
        try:
            local.bind(('127.0.0.1', 0))
            assert local.getsockname()[0] == '127.0.0.1'
        finally:
            local.close()
    ''')
    assert result['status'] == 'done', result


@pytest.mark.skipif(not socket.has_ipv6, reason='IPv6 is unavailable')
def test_worker_permits_ipv6_loopback_bind(worker):
    result = worker('''
        import socket
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as local:
            local.bind(('::1', 0))
            assert local.getsockname()[0] == '::1'
    ''')
    assert result['status'] == 'done', result


# Runs after the production hook, only inside the child. If that hook regresses,
# this tripwire prevents test connections, packets, DNS queries and unsafe binds
# from reaching the OS, and raises a different exception so the test still fails.
TRIPWIRE = '''
import sys
def no_os_contact(event, args):
    if event.startswith('socket.') and event not in ('socket.__new__', 'socket.gethostname'):
        raise AssertionError('network operation escaped production guard')
sys.addaudithook(no_os_contact)
'''


@pytest.mark.parametrize('module', ['socket', '_socket'])
@pytest.mark.parametrize('family,kind,operation', [
    ('AF_INET', 'SOCK_STREAM', "local.connect(('127.0.0.1', 1))"),
    ('AF_INET', 'SOCK_STREAM', "local.connect_ex(('127.0.0.1', 1))"),
    ('AF_INET', 'SOCK_STREAM', "local.connect(('203.0.113.1', 1))"),
    ('AF_INET', 'SOCK_STREAM', "local.connect_ex(('203.0.113.1', 1))"),
    ('AF_INET', 'SOCK_DGRAM', "local.sendto(b'synthetic', ('203.0.113.1', 1))"),
    ('AF_INET', 'SOCK_DGRAM', "local.sendto(b'synthetic', ('127.0.0.1', 1))"),
    pytest.param('AF_INET', 'SOCK_DGRAM',
                 "local.sendmsg([b'synthetic'], [], 0, ('203.0.113.1', 1))",
                 marks=pytest.mark.skipif(not hasattr(socket.socket, 'sendmsg'),
                                          reason='sendmsg is unavailable')),
    ('AF_INET', 'SOCK_STREAM', "local.bind(('0.0.0.0', 0))"),
    ('AF_INET', 'SOCK_STREAM', "local.bind(('', 0))"),
    ('AF_INET', 'SOCK_STREAM', "local.bind(('203.0.113.1', 0))"),
    pytest.param('AF_INET6', 'SOCK_STREAM', "local.bind(('::', 0))",
                 marks=pytest.mark.skipif(not socket.has_ipv6, reason='IPv6 is unavailable')),
    pytest.param('AF_INET6', 'SOCK_STREAM', "local.bind(('2001:db8::1', 0))",
                 marks=pytest.mark.skipif(not socket.has_ipv6, reason='IPv6 is unavailable')),
])
def test_worker_denies_io_and_unsafe_bind_before_os_contact(worker, module, family, kind, operation):
    result = worker(f'import {module} as api\n' + TRIPWIRE + f'''
local = api.socket(api.{family}, api.{kind})
try:
    try:
        {operation}
    except PermissionError:
        pass
    else:
        raise AssertionError('operation was not denied')
finally:
    local.close()
''')
    assert result['status'] == 'done', result


@pytest.mark.parametrize('module', ['socket', '_socket'])
@pytest.mark.parametrize('query', [
    "api.getaddrinfo('network-fixture.invalid', 443)",
    "api.gethostbyname('network-fixture.invalid')",
    "api.gethostbyname_ex('network-fixture.invalid')",
    "api.gethostbyaddr('203.0.113.1')",
    "api.getnameinfo(('203.0.113.1', 443), 0)",
])
def test_worker_denies_name_queries_before_os_contact(worker, module, query):
    result = worker(f'import {module} as api\n' + TRIPWIRE + f'''
try:
    {query}
except PermissionError:
    pass
else:
    raise AssertionError('name query was not denied')
''')
    assert result['status'] == 'done', result


@pytest.mark.skipif(importlib.util.find_spec('torch') is None, reason='Torch is not installed')
def test_worker_imports_real_local_torch(worker):
    result = worker('''
        import torch
        assert torch.__version__
        assert torch.tensor([1]).item() == 1
    ''', timeout=30)
    assert result['status'] == 'done', result
