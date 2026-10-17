"""API contract tests (К1): auth, upload validation, job lifecycle, persistence."""

import io

import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.schemas import AnalysisResult

API_KEY = "test-key"


@pytest.fixture
def settings(tmp_path):
    return Settings(
        api_keys=frozenset({API_KEY}),
        data_dir=tmp_path,
        max_upload_mb=5,
        workers=1,
        pipeline="mock",
        mock_delay_sec=0.0,
        job_timeout_sec=30,
    )


@pytest.fixture
def client(settings):
    app = create_app(settings)
    with TestClient(app) as c:
        yield c


def _upload(client, name="clip.mov", content=b"fake-video", key=API_KEY):
    headers = {"X-API-Key": key} if key is not None else {}
    files = {"file": (name, io.BytesIO(content), "video/quicktime")}
    return client.post("/api/v1/analyze", files=files, headers=headers)


def _wait_done(client, job_id):
    client.app.state.runner.wait(job_id, timeout=10)
    return client.get(f"/api/v1/jobs/{job_id}", headers={"X-API-Key": API_KEY})


def test_health_is_public(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_analyze_without_key_is_rejected(client):
    assert _upload(client, key=None).status_code == 401


def test_analyze_with_wrong_key_is_rejected(client):
    assert _upload(client, key="nope").status_code == 401


def test_jobs_endpoint_requires_key(client):
    assert client.get("/api/v1/jobs/whatever").status_code == 401


def test_rejects_non_video_extension(client):
    resp = _upload(client, name="notes.txt")
    assert resp.status_code == 415


def test_rejects_too_large_upload(client):
    resp = _upload(client, content=b"x" * (6 * 1024 * 1024))
    assert resp.status_code == 413


def test_rejects_empty_upload(client):
    assert _upload(client, content=b"").status_code == 400


def test_upload_returns_job_id_and_processing_status(client):
    resp = _upload(client)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in {"processing", "done"}
    assert isinstance(body["job_id"], str) and body["job_id"]


def test_job_completes_with_contract_shaped_result(client):
    job_id = _upload(client).json()["job_id"]
    resp = _wait_done(client, job_id)
    assert resp.status_code == 200
    body = resp.json()
    assert body["job_id"] == job_id
    assert body["status"] == "done"
    result = AnalysisResult.model_validate(body)
    assert result.verdict in {"violation", "no_violation"}
    for v in result.violations:
        assert 1 <= len(v.evidence_frames) <= 3


def test_uppercase_mov_extension_is_accepted(client):
    assert _upload(client, name="X-1.MOV").status_code == 200


def test_unknown_job_returns_404(client):
    resp = client.get("/api/v1/jobs/doesnotexist", headers={"X-API-Key": API_KEY})
    assert resp.status_code == 404


def test_pipeline_error_marks_job_failed(settings):
    def boom(video_path, on_progress=None, time_budget_sec=None):
        raise RuntimeError("decoder exploded")

    app = create_app(settings, pipeline_fn=boom)
    with TestClient(app) as c:
        job_id = _upload(c).json()["job_id"]
        body = _wait_done(c, job_id).json()
    assert body["status"] == "failed"
    assert "error" in body


def test_finished_job_survives_restart(settings):
    app = create_app(settings)
    with TestClient(app) as c:
        job_id = _upload(c).json()["job_id"]
        _wait_done(c, job_id)

    restarted = create_app(settings)
    with TestClient(restarted) as c:
        resp = c.get(f"/api/v1/jobs/{job_id}", headers={"X-API-Key": API_KEY})
    assert resp.status_code == 200
    assert resp.json()["status"] == "done"
