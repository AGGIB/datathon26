"""Load, concurrency, and negative payload robustness tests (Issue #12)."""

import concurrent.futures
import io
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app

API_KEY = "test-key"
# Valid ISO-BMFF header (ftyp box)
FAKE_MOV = b"\x00\x00\x00\x14ftypqt  \x00\x00\x02\x00qt  " + b"\x00" * 1024


@pytest.fixture
def load_settings(tmp_path):
    return Settings(
        api_keys=frozenset({API_KEY}),
        data_dir=tmp_path,
        max_upload_mb=50,
        workers=2,
        pipeline="mock",
        mock_delay_sec=0.1,  # fast simulation for load test
        job_timeout_sec=240,
    )


def test_concurrent_five_uploads_all_complete_under_budget(load_settings):
    """5 concurrent video uploads processed by 2 workers finish < time budget."""
    app = create_app(load_settings)
    with TestClient(app) as client:
        start_time = time.time()

        def submit_and_wait(idx: int) -> dict:
            headers = {"X-API-Key": API_KEY}
            files = {"file": (f"video_{idx}.mov", io.BytesIO(FAKE_MOV), "video/quicktime")}
            res = client.post("/api/v1/analyze", files=files, headers=headers)
            assert res.status_code == 200, f"Upload {idx} failed: {res.text}"
            job_id = res.json()["job_id"]

            # Poll until done
            for _ in range(50):
                r = client.get(f"/api/v1/jobs/{job_id}", headers=headers)
                if r.json().get("status") == "done":
                    return r.json()
                time.sleep(0.05)
            raise TimeoutError(f"Job {job_id} did not finish in time")

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(submit_and_wait, range(5)))

        total_duration = time.time() - start_time
        assert total_duration < 30.0, f"Total execution took too long: {total_duration:.2f}s"
        assert len(results) == 5
        for res in results:
            assert res["status"] == "done"
            assert res["verdict"] in {"violation", "no_violation"}
            if res["verdict"] == "violation":
                assert len(res["violations"]) >= 1
                for v in res["violations"]:
                    assert "evidence_frames" in v
                    assert 1 <= len(v["evidence_frames"]) <= 3


def test_corrupt_and_invalid_payloads_return_clear_errors(load_settings):
    """Empty file, invalid extension, corrupted ISO-BMFF header return clear errors."""
    app = create_app(load_settings)
    with TestClient(app) as client:
        headers = {"X-API-Key": API_KEY}

        # 1. Empty file
        empty = client.post(
            "/api/v1/analyze",
            files={"file": ("empty.mov", io.BytesIO(b""), "video/quicktime")},
            headers=headers,
        )
        assert empty.status_code == 400
        assert "Пустой файл" in empty.json()["detail"]

        # 2. Invalid extension (.txt)
        txt = client.post(
            "/api/v1/analyze",
            files={"file": ("notes.txt", io.BytesIO(b"hello world"), "text/plain")},
            headers=headers,
        )
        assert txt.status_code == 415
        assert "Поддерживаются только" in txt.json()["detail"]

        # 3. Corrupted payload (.mov name but plain text content)
        corrupted = client.post(
            "/api/v1/analyze",
            files={"file": ("fake.mov", io.BytesIO(b"NOT A VIDEO AT ALL JUST TEXT"), "video/quicktime")},
            headers=headers,
        )
        assert corrupted.status_code == 415
        assert "не похож на видео" in corrupted.json()["detail"]
