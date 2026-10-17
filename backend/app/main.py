"""FastAPI app: jury API (ТЗ раздел 10) + static UI."""

import functools
import logging
import re
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

_FALLBACK_INDEX_HTML = """<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ForteBank Datathon 26 — Видео-контроль кассовой дисциплины</title>
  <style>
    :root {
      --bg: #0d1117;
      --card-bg: #161b22;
      --border: #30363d;
      --text: #c9d1d9;
      --text-heading: #f0f6fc;
      --accent: #58a6ff;
      --green: #3fb950;
      --font: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    body { margin: 0; padding: 0; background: var(--bg); color: var(--text); font-family: var(--font); display: flex; flex-direction: column; align-items: center; min-height: 100vh; }
    header { width: 100%; max-width: 900px; padding: 40px 24px 20px; box-sizing: border-box; border-bottom: 1px solid var(--border); }
    .badge { display: inline-block; padding: 4px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; background: rgba(63, 185, 80, 0.15); color: var(--green); border: 1px solid var(--green); margin-bottom: 12px; }
    h1 { color: var(--text-heading); font-size: 28px; margin: 0 0 10px; }
    p.lead { color: #8b949e; font-size: 16px; margin: 0; }
    main { width: 100%; max-width: 900px; padding: 30px 24px; box-sizing: border-box; display: grid; grid-gap: 20px; }
    .card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 24px; }
    h2 { color: var(--text-heading); font-size: 18px; margin-top: 0; margin-bottom: 16px; }
    .btn { display: inline-block; background: #238636; color: #fff; padding: 8px 16px; border-radius: 6px; text-decoration: none; font-weight: 500; font-size: 14px; margin-right: 10px; }
    .btn.secondary { background: #21262d; border: 1px solid var(--border); color: var(--text-heading); }
    code { background: rgba(110, 118, 129, 0.2); padding: 2px 6px; border-radius: 4px; font-family: monospace; font-size: 13px; color: #e6edf3; }
  </style>
</head>
<body>
  <header>
    <div class="badge">● Сервис активен и готов к проверке</div>
    <h1>ForteBank Datathon 26: Видео-контроль кассовой дисциплины</h1>
    <p class="lead">Бэкенд API для проверки нарушений регламента кассовых операций (ТЗ п. 10, Контракт К1).</p>
  </header>
  <main>
    <div class="card">
      <h2>🚀 Интерактивная документация и точки входа</h2>
      <p>Сервис полностью сконфигурирован и отвечает по контракту жюри:</p>
      <div style="margin: 20px 0;">
        <a class="btn" href="/docs">Swagger UI (/docs)</a>
        <a class="btn secondary" href="/health">Healthcheck (/health)</a>
      </div>
      <p>Точка анализа видео: <code>POST /api/v1/analyze</code> с заголовком <code>X-API-Key</code>.</p>
    </div>
  </main>
</body>
</html>
"""

from . import mock_pipeline
from .config import Settings
from .media import looks_like_isobmff
from .runner import JobRunner, PipelineFn
from .schemas import JobAccepted, JobView
from .store import JobStore, new_job_id

log = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = frozenset({".mov", ".mp4"})
CHUNK_BYTES = 1024 * 1024
UPLOAD_PATH = "/api/v1/analyze"
MULTIPART_OVERHEAD_BYTES = 64 * 1024


def load_pipeline(settings: Settings) -> PipelineFn:
    if settings.pipeline == "mock":
        return functools.partial(mock_pipeline.analyze, delay_sec=settings.mock_delay_sec)
    from ml.pipeline import analyze  # heavy imports only when the real pipeline is enabled

    return analyze


def _record_to_view(record: dict) -> JobView:
    base = {"job_id": record["job_id"], "status": record["status"]}
    if record["status"] == "done":
        return JobView(**base, **record["result"])
    if record["status"] == "failed":
        return JobView(**base, error=record.get("error", "Ошибка обработки"))
    return JobView(**base, progress=record.get("progress"), stage=record.get("stage"))


async def _save_upload(upload: UploadFile, dest: Path, max_bytes: int) -> int:
    size = 0
    with dest.open("wb") as fh:
        while chunk := await upload.read(CHUNK_BYTES):
            size += len(chunk)
            if size > max_bytes:
                fh.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, f"Файл больше {max_bytes // CHUNK_BYTES} МБ")
            fh.write(chunk)
    return size


def create_app(settings: Settings, pipeline_fn: PipelineFn | None = None) -> FastAPI:
    store = JobStore(settings.data_dir / "jobs")
    runner = JobRunner(
        store,
        pipeline_fn or load_pipeline(settings),
        workers=settings.workers,
        time_budget_sec=settings.job_timeout_sec,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        resumed = runner.resume_unfinished()
        if resumed:
            log.info("resumed %d unfinished jobs", resumed)
        yield
        runner.shutdown()

    app = FastAPI(title="Datathon26 cash-discipline video check", version="0.1.0", lifespan=lifespan)
    app.state.store = store
    app.state.runner = runner
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["X-API-Key", "Content-Type"],
    )

    max_upload_bytes = settings.max_upload_mb * CHUNK_BYTES

    @app.middleware("http")
    async def limit_upload_size(request: Request, call_next):
        # Starlette spools the whole multipart body to disk before the handler runs,
        # so the size cap must be enforced on Content-Length up front.
        if request.method == "POST" and request.url.path == UPLOAD_PATH:
            length = request.headers.get("content-length")
            if length is None:
                return JSONResponse({"detail": "Нужен заголовок Content-Length"}, status_code=411)
            if not length.isdigit() or int(length) > max_upload_bytes + MULTIPART_OVERHEAD_BYTES:
                return JSONResponse({"detail": f"Файл больше {settings.max_upload_mb} МБ"}, status_code=413)
        return await call_next(request)

    def require_key(x_api_key: str | None = Header(default=None)) -> None:
        if not x_api_key or not any(secrets.compare_digest(x_api_key, k) for k in settings.api_keys):
            raise HTTPException(401, "Неверный или отсутствующий X-API-Key")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "pipeline": settings.pipeline}

    @app.post("/api/v1/analyze", response_model=JobAccepted, dependencies=[Depends(require_key)])
    async def analyze(file: UploadFile = File(...)) -> JobAccepted:
        filename = file.filename or ""
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(415, f"Поддерживаются только {', '.join(sorted(ALLOWED_EXTENSIONS))}")

        job_id = new_job_id()
        job_dir = store.job_dir(job_id)
        job_dir.mkdir(parents=True)
        stem = re.sub(r"[^a-zA-Z0-9_\-]", "_", Path(filename).stem)[:32]
        video_name = f"input_{stem}{ext}" if stem else f"input{ext}"
        video_path = job_dir / video_name
        size = await _save_upload(file, video_path, max_upload_bytes)
        if size == 0:
            video_path.unlink(missing_ok=True)
            raise HTTPException(400, "Пустой файл")
        if not looks_like_isobmff(video_path):
            video_path.unlink(missing_ok=True)
            raise HTTPException(415, "Файл не похож на видео .mov/.mp4")

        store.create(job_id, video_path, filename)
        runner.submit(job_id, video_path)
        log.info("job %s accepted: %s (%.1f MB)", job_id, filename, size / CHUNK_BYTES)
        return JobAccepted(job_id=job_id, status="processing")

    @app.get(
        "/api/v1/jobs/{job_id}",
        response_model=JobView,
        response_model_exclude_none=True,
        dependencies=[Depends(require_key)],
    )
    def get_job(job_id: str) -> JobView:
        record = store.get(job_id)
        if record is None:
            raise HTTPException(404, "Задание не найдено")
        return _record_to_view(record)

    if settings.frontend_dist and settings.frontend_dist.is_dir():
        app.mount("/", StaticFiles(directory=settings.frontend_dist, html=True), name="ui")
    else:
        dist_index = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist" / "index.html"
        if dist_index.is_file():
            html_content = dist_index.read_text(encoding="utf-8")
        else:
            html_content = _FALLBACK_INDEX_HTML

        @app.get("/", response_class=HTMLResponse)
        def index() -> HTMLResponse:
            return HTMLResponse(content=html_content)

    return app


def app_from_env() -> FastAPI:
    """Entry point: uvicorn --factory backend.app.main:app_from_env"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return create_app(Settings.from_env())
