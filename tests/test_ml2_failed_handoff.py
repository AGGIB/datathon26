"""Consumer regression checks of the existing failed path, using synthetic faults.

These establish the API exception contract, not a fix to the K2 fault sites.
"""
import io

import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app


@pytest.mark.parametrize('reason', ['zero_processed_frames', 'all_evidence_lost'])
def test_exception_is_failed_without_verdict_persists_and_releases_worker(tmp_path, reason):
    settings = Settings(api_keys=frozenset({'test-key'}), data_dir=tmp_path,
                        max_upload_mb=5, workers=1, pipeline='mock',
                        mock_delay_sec=0.0, job_timeout_sec=30)
    calls = []

    def injected_pipeline(video_path, on_progress=None, time_budget_sec=None):
        calls.append(video_path)
        if len(calls) == 1:
            raise RuntimeError(reason)
        return {'verdict': 'no_violation', 'violations': []}

    headers = {'X-API-Key': 'test-key'}
    # Upload validation fixture only; the injected pipeline does not decode this.
    content = b'\x00\x00\x00\x14ftypqt  \x00\x00\x02\x00qt  \x00\x00\x00\x08moov' + b'\x00' * 64

    def submit(client):
        response = client.post('/api/v1/analyze', headers=headers,
                               files={'file': ('synthetic.mov', io.BytesIO(content), 'video/quicktime')})
        assert response.status_code == 200
        job_id = response.json()['job_id']
        client.app.state.runner.wait(job_id, timeout=10)
        result = client.get(f'/api/v1/jobs/{job_id}', headers=headers)
        assert result.status_code == 200
        return job_id, result.json()

    with TestClient(create_app(settings, pipeline_fn=injected_pipeline)) as client:
        failed_id, failed = submit(client)
        assert failed['status'] == 'failed'
        assert failed['error']
        assert {'verdict', 'violations', 'result'}.isdisjoint(failed)
        record = client.app.state.store.get(failed_id)
        assert record['stage'] == 'failed' and record.get('result') is None
        assert client.app.state.runner.lost_slots == 0
        _, next_job = submit(client)
        assert next_job['status'] == 'done' and next_job['verdict'] == 'no_violation'
        assert next_job['violations'] == []

    with TestClient(create_app(settings, pipeline_fn=injected_pipeline)) as restarted:
        response = restarted.get(f'/api/v1/jobs/{failed_id}', headers=headers)
        assert response.status_code == 200
        assert response.json() == failed
    assert len(calls) == 2
