"""API contract tests (К1): auth, upload validation, job lifecycle, persistence."""

import io
import threading
import time

import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.runner import JobRunner
from backend.app.schemas import AnalysisResult
from backend.app.store import JobStore

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


# Minimal ISO-BMFF container: 'ftyp' box followed by 'moov' atom.
FAKE_MOV = b"\x00\x00\x00\x14ftypqt  \x00\x00\x02\x00qt  \x00\x00\x00\x08moov" + b"\x00" * 64


def _upload(client, name="clip.mov", content=FAKE_MOV, key=API_KEY):
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


def test_rejects_non_isobmff_payload_with_mov_name(client):
    # e.g. an HLS playlist renamed to .mov — ffmpeg would follow its URLs
    playlist = b"#EXTM3U\n#EXT-X-MEDIA-SEQUENCE:0\nfile:///etc/passwd\n"
    assert _upload(client, content=playlist).status_code == 415


def test_oversized_content_length_rejected_before_body_is_read(client):
    resp = client.post(
        "/api/v1/analyze",
        content=b"tiny",
        headers={"X-API-Key": API_KEY, "Content-Length": str(50 * 1024 * 1024), "Content-Type": "multipart/form-data; boundary=x"},
    )
    assert resp.status_code == 413


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


def test_pipeline_hard_timeout_marks_job_failed(settings):
    import time

    def hanging_pipeline(video_path, on_progress=None, time_budget_sec=None):
        time.sleep(2)
        return {"verdict": "no_violation", "violations": []}

    short_budget = Settings(
        api_keys=settings.api_keys,
        data_dir=settings.data_dir,
        max_upload_mb=settings.max_upload_mb,
        workers=1,
        pipeline="mock",
        mock_delay_sec=0.0,
        job_timeout_sec=0.1,  # 100ms hard watchdog
        min_budget_sec=0.01,
    )
    app = create_app(short_budget, pipeline_fn=hanging_pipeline)
    with TestClient(app) as c:
        job_id = _upload(c).json()["job_id"]
        c.app.state.runner.wait(job_id, timeout=2)
        resp = c.get(f"/api/v1/jobs/{job_id}", headers={"X-API-Key": API_KEY})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "failed"
        assert "Превышено" in body["error"]


def test_damaged_job_json_does_not_crash_store_or_startup(settings, tmp_path):
    # Corrupt job.json (0 bytes) should not cause unhandled crash loop on restart
    job_dir = tmp_path / "jobs" / "badjob123456"
    job_dir.mkdir(parents=True)
    (job_dir / "job.json").write_text("", encoding="utf-8")

    damaged_settings = Settings(
        api_keys=settings.api_keys,
        data_dir=tmp_path,
        max_upload_mb=settings.max_upload_mb,
        workers=1,
        pipeline="mock",
        mock_delay_sec=0.0,
        job_timeout_sec=settings.job_timeout_sec,
    )
    app = create_app(damaged_settings)
    with TestClient(app) as c:
        assert c.get("/health").status_code == 200
        assert c.get("/api/v1/jobs/badjob123456", headers={"X-API-Key": API_KEY}).status_code == 404



def test_late_progress_callback_does_not_overwrite_completed_job(settings):
    callback_holder = []

    def pipeline_saving_callback(video_path, on_progress=None, time_budget_sec=None):
        callback_holder.append(on_progress)
        return {"verdict": "no_violation", "violations": []}

    app = create_app(settings, pipeline_fn=pipeline_saving_callback)
    with TestClient(app) as c:
        job_id = _upload(c).json()["job_id"]
        _wait_done(c, job_id)
        done_record = c.get(f"/api/v1/jobs/{job_id}", headers={"X-API-Key": API_KEY}).json()
        assert done_record["status"] == "done"

        # Simulate late callback from lingering thread
        if callback_holder and callback_holder[0]:
            callback_holder[0](0.25, "late_stage")

        after_record = c.get(f"/api/v1/jobs/{job_id}", headers={"X-API-Key": API_KEY}).json()
        assert after_record["status"] == "done"
        assert after_record.get("stage") != "late_stage"


def test_unfinished_job_resumes_and_completes_after_restart(settings, monkeypatch):
    """If service crashes (kill -9) while job is processing, restart resumes it to done."""

    # Create a job directly in store in 'processing' state as if killed mid-run
    store = JobStore(settings.data_dir / "jobs")
    job_id = "aabbccdd1122"
    job_dir = store.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    fake_video = job_dir / "input.mov"
    fake_video.write_bytes(b"dummy")
    store.create(job_id, fake_video, "input.mov")

    DUMMY_JPEG_B64 = (
        "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP/////////////////////////////////////////////////////////////////"
        "/////////////////////wgALCAABAAEBAREA/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPxA="
    )

    def mock_pipeline_slow(video_path, on_progress=None, time_budget_sec=None):
        return {
            "verdict": "violation",
            "violations": [
                {
                    "type": "Оставление денег без присмотра",
                    "timecode_sec": 12.0,
                    "description": "Деньги на стойке",
                    "evidence_frames": [{"timestamp_sec": 12.0, "image_base64": DUMMY_JPEG_B64}],
                }
            ],
        }

    restarted = create_app(settings, pipeline_fn=mock_pipeline_slow)
    with TestClient(restarted) as c:
        resp = _wait_done(c, job_id)
        assert resp.status_code == 200
        assert resp.json()["status"] == "done"
        assert resp.json()["verdict"] == "violation"


def test_rejects_corrupted_short_isobmff_header(client):
    short_header = b"\x00\x00\x00\x08ftyp"
    resp = _upload(client, name="corrupt.mov", content=short_header)
    assert resp.status_code == 415
    assert "не похож на видео" in resp.json()["detail"]


def test_frontend_dist_mounted_and_serves_ui(settings, tmp_path):
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    (dist_dir / "index.html").write_text("<!doctype html><html><body>Test UI</body></html>", encoding="utf-8")

    ui_settings = Settings(
        api_keys=settings.api_keys,
        data_dir=settings.data_dir,
        max_upload_mb=settings.max_upload_mb,
        workers=settings.workers,
        pipeline=settings.pipeline,
        mock_delay_sec=0.0,
        job_timeout_sec=settings.job_timeout_sec,
        frontend_dist=dist_dir,
    )
    app = create_app(ui_settings)
    with TestClient(app) as c:
        resp = c.get("/")
        assert resp.status_code == 200
        assert "Test UI" in resp.text
        # Ensure /health and API endpoints still work
        assert c.get("/health").status_code == 200


def test_upload_clean_file_returns_no_violation(client):
    resp = _upload(client, name="cashier_clean_sample.mov")
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    done = _wait_done(client, job_id)
    assert done.status_code == 200
    body = done.json()
    assert body["status"] == "done"
    assert body["verdict"] == "no_violation"
    assert body["violations"] == []


def test_extended_64bit_ftyp_box_accepted(client):
    import struct
    # 64-bit extended size box: size=1, atom='ftyp', extended_size=24, brand='qt  ', minor=0, followed by moov
    hdr = struct.pack(">I4sQ4sI", 1, b"ftyp", 24, b"qt  ", 0) + struct.pack(">I4s", 8, b"moov") + b"\x00" * 32
    resp = _upload(client, name="extended.mov", content=hdr)
    assert resp.status_code == 200


def test_extended_64bit_ftyp_box_too_short_rejected(client):
    import struct
    # 64-bit extended size box claims size=16, but minimum for extended ftyp is 24
    hdr = struct.pack(">I4sQ4sI", 1, b"ftyp", 16, b"qt  ", 0) + b"\x00" * 32
    resp = _upload(client, name="extended_short.mov", content=hdr)
    assert resp.status_code == 415
    assert "не похож на видео" in resp.json()["detail"]


def test_header_only_ftyp_without_media_rejected(client):
    # A 20-byte file with only an ftyp atom and no subsequent media atoms (moov/mdat)
    content = b"\x00\x00\x00\x14ftypqt  \x00\x00\x00\x00qt  "
    resp = _upload(client, name="header-only.mov", content=content)
    assert resp.status_code == 415
    assert "не похож на видео" in resp.json()["detail"]


def test_queue_timeout_marks_job_failed(client, tmp_path):
    import time
    runner = client.app.state.runner
    store = client.app.state.store

    video_path = tmp_path / "dummy.mov"
    video_path.write_bytes(b"\x00" * 32)
    store.create("aabbccddee11", video_path, "dummy.mov")
    # Simulate job created 300 seconds ago (exceeds 30s budget)
    store.update("aabbccddee11", created_at=time.time() - 300)

    runner.submit("aabbccddee11", video_path)
    runner.wait("aabbccddee11", timeout=5)

    res = store.get("aabbccddee11")
    assert res is not None
    assert res["status"] == "failed"
    assert "очереди" in res["error"]


def test_timed_out_pipeline_holds_slot_until_thread_exits(tmp_path):
    import threading
    import time
    from backend.app.runner import JobRunner

    store = JobStore(tmp_path / "jobs")
    pipeline_finish = threading.Event()

    def slow_hanging_pipeline(video_path, on_progress=None, time_budget_sec=None):
        pipeline_finish.wait(timeout=2.0)
        return {"verdict": "no_violation", "violations": []}

    # runner with workers=1, time_budget_sec=0.15s, soft_budget_sec=0.05s
    runner = JobRunner(store=store, pipeline=slow_hanging_pipeline, workers=1, time_budget_sec=0.15, soft_budget_sec=0.05, min_budget_sec=0.01)

    v1 = tmp_path / "v1.mov"
    v1.write_bytes(b"dummy")
    store.create("112233445566", v1, "v1.mov")
    runner.submit("112233445566", v1)
    runner.wait("112233445566", timeout=2.0)

    # Job 1 has timed out and marked failed
    j1 = store.get("112233445566")
    assert j1["status"] == "failed"
    assert "таймаут" in j1["error"]

    # Concurrency guard: slot is retained until pipeline thread actually exits
    assert runner._sem._value == 0

    # Once pipeline finishes, slot is released
    pipeline_finish.set()
    time.sleep(0.1)
    assert runner._sem._value == 1
    runner.shutdown()


def test_job_rejected_when_remaining_budget_under_minimum(client, tmp_path):
    import time

    runner = client.app.state.runner
    store = client.app.state.store

    job_id = "1122334455bb"
    video_path = tmp_path / "dummy_budget.mov"
    video_path.write_bytes(b"\x00" * 32)
    store.create(job_id, video_path, "dummy_budget.mov")
    # Job created 15 seconds ago on a 30s budget (remaining = 15s < min_budget_sec = 20s)
    store.update(job_id, created_at=time.time() - 15)

    runner.submit(job_id, video_path)
    runner.wait(job_id, timeout=5)

    res = store.get(job_id)
    assert res is not None
    assert res["status"] == "failed"
    assert "Недостаточно времени для обработки видео" in res["error"]


def test_health_reports_lost_slots_on_thread_timeout(tmp_path, caplog):
    import logging
    import threading
    import time
    from backend.app.config import Settings
    from backend.app.main import create_app

    pipeline_finish = threading.Event()

    def hanging_pipeline(video_path, on_progress=None, time_budget_sec=None):
        pipeline_finish.wait(timeout=2.0)
        return {"verdict": "no_violation", "violations": []}

    settings = Settings(
        api_keys=frozenset({API_KEY}),
        data_dir=tmp_path,
        max_upload_mb=5,
        workers=1,
        pipeline="mock",
        job_timeout_sec=0.15,
        pipeline_timeout_sec=0.05,
        min_budget_sec=0.01,
    )
    app = create_app(settings, pipeline_fn=hanging_pipeline)
    with TestClient(app) as c:
        # Initial health check: lost_slots is 0
        h0 = c.get("/health").json()
        assert h0.get("lost_slots") == 0

        # Upload a job that will hang and hit hard watchdog timeout
        with caplog.at_level(logging.ERROR):
            job_id = _upload(c).json()["job_id"]
            c.app.state.runner.wait(job_id, timeout=2.0)

            # Job timed out and slot is retained as lost
            j = c.app.state.store.get(job_id)
            assert j["status"] == "failed"
            assert "таймаут" in j["error"]
            assert c.app.state.runner.lost_slots == 1

            h1 = c.get("/health").json()
            assert h1.get("lost_slots") == 1

            # Verify ERROR log was emitted
            error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
            assert any("exceeded hard timeout" in r.message and "compute slot retained" in r.message for r in error_records)

        # Release thread and verify lost_slots returns to 0
        pipeline_finish.set()
        time.sleep(0.1)
        assert c.app.state.runner.lost_slots == 0
        h2 = c.get("/health").json()
        assert h2.get("lost_slots") == 0


def test_short_hard_timeout_rejected_by_default_min_budget(tmp_path):
    store = JobStore(tmp_path / "jobs")
    pipeline_called = []

    def dummy_pipeline(video_path, on_progress=None, time_budget_sec=None):
        pipeline_called.append(time_budget_sec)
        return {"verdict": "no_violation", "violations": []}

    # hard_timeout = 10s with default min_budget_sec = 20s
    runner = JobRunner(store, dummy_pipeline, workers=1, time_budget_sec=10.0)
    video = tmp_path / "v_short.mov"
    video.write_bytes(b"dummy")
    job_id = "1122334455cc"
    store.create(job_id, video, "v_short.mov")
    runner.submit(job_id, video)
    runner.wait(job_id, timeout=2.0)

    res = store.get(job_id)
    assert res is not None
    assert res["status"] == "failed"
    assert "Недостаточно времени для обработки видео" in res["error"]
    assert len(pipeline_called) == 0
    runner.shutdown()


def test_tiny_soft_budget_rejected_by_default_min_budget(tmp_path):
    store = JobStore(tmp_path / "jobs")
    pipeline_called = []

    def dummy_pipeline(video_path, on_progress=None, time_budget_sec=None):
        pipeline_called.append(time_budget_sec)
        return {"verdict": "no_violation", "violations": []}

    # hard_timeout = 30s, but soft_budget_sec = 0.001s (< default min_budget_sec = 20s)
    runner = JobRunner(store, dummy_pipeline, workers=1, time_budget_sec=30.0, soft_budget_sec=0.001)
    video = tmp_path / "v_tiny.mov"
    video.write_bytes(b"dummy")
    job_id = "1122334455dd"
    store.create(job_id, video, "v_tiny.mov")
    runner.submit(job_id, video)
    runner.wait(job_id, timeout=2.0)

    res = store.get(job_id)
    assert res is not None
    assert res["status"] == "failed"
    assert "Недостаточно времени для обработки видео" in res["error"]
    assert len(pipeline_called) == 0
    runner.shutdown()


def test_watchdog_boundary_race_does_not_leak_lost_slots(tmp_path):
    store = JobStore(tmp_path / "jobs")
    finish_ev = threading.Event()

    def quick_pipeline(video_path, on_progress=None, time_budget_sec=None):
        # finishes right around the timeout boundary
        finish_ev.wait(timeout=0.06)
        return {"verdict": "no_violation", "violations": []}

    runner = JobRunner(store, quick_pipeline, workers=1, time_budget_sec=0.08, soft_budget_sec=0.05, min_budget_sec=0.01)
    video = tmp_path / "v_race.mov"
    video.write_bytes(b"dummy")
    job_id = "1122334455ee"
    store.create(job_id, video, "v_race.mov")
    runner.submit(job_id, video)
    runner.wait(job_id, timeout=2.0)

    finish_ev.set()
    time.sleep(0.05)
    # Regardless of whether it timed out or succeeded, lost_slots must be 0 once thread exits
    assert runner.lost_slots == 0
    assert runner._sem.acquire(blocking=False) is True
    runner._sem.release()
    runner.shutdown()






