"""Background execution of the analysis pipeline (К2) with a fixed worker pool."""

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path
from typing import Any, Callable, Protocol

from pydantic import ValidationError

from . import mock_pipeline
from .schemas import AnalysisResult
from .store import JobStore

log = logging.getLogger(__name__)

ProgressFn = Callable[[float, str], None]


class PipelineFn(Protocol):
    def __call__(
        self,
        video_path: str,
        on_progress: ProgressFn | None = None,
        time_budget_sec: float | None = None,
    ) -> dict[str, Any]: ...


class JobRunner:
    def __init__(self, store: JobStore, pipeline: PipelineFn, workers: int, time_budget_sec: float) -> None:
        self._store = store
        self._pipeline = pipeline
        self._budget = time_budget_sec
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="job")
        self._futures: dict[str, Future] = {}
        self._lock = threading.Lock()

    def submit(self, job_id: str, video_path: Path) -> None:
        future = self._pool.submit(self._run, job_id, video_path)
        with self._lock:
            self._futures[job_id] = future

    def resume_unfinished(self) -> int:
        """Re-queue jobs that were processing when the service stopped."""
        pending = self._store.unfinished()
        for record in pending:
            self.submit(record["job_id"], Path(record["input_path"]))
        return len(pending)

    def wait(self, job_id: str, timeout: float) -> None:
        with self._lock:
            future = self._futures.get(job_id)
        if future is None:
            return
        try:
            future.result(timeout=timeout)
        except FutureTimeout:
            raise TimeoutError(f"job {job_id} still running after {timeout}s") from None

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    def _run(self, job_id: str, video_path: Path) -> None:
        def on_progress(progress: float, stage: str) -> None:
            self._store.update(job_id, progress=round(max(0.0, min(1.0, progress)), 3), stage=stage)

        pipeline_result: list[dict[str, Any]] = []
        pipeline_exc: list[Exception] = []

        def target() -> None:
            try:
                res = self._pipeline(
                    str(video_path),
                    on_progress=on_progress,
                    time_budget_sec=self._budget,
                )
                pipeline_result.append(res)
            except Exception as exc:  # noqa: BLE001
                pipeline_exc.append(exc)

        thread = threading.Thread(target=target, name=f"pipe-{job_id}", daemon=True)
        thread.start()
        thread.join(timeout=self._budget)

        try:
            if thread.is_alive():
                log.warning(
                    "job %s: pipeline exceeded time budget %.1fs, returning fast fallback stage",
                    job_id,
                    self._budget,
                )
                raw = mock_pipeline.fast_fallback(str(video_path))
            elif pipeline_exc:
                raise pipeline_exc[0]
            elif pipeline_result:
                raw = pipeline_result[0]
            else:
                raw = mock_pipeline.fast_fallback(str(video_path))

            result = AnalysisResult.model_validate(raw)
        except ValidationError as exc:
            log.exception("job %s: pipeline returned an invalid result", job_id)
            self._store.update(job_id, status="failed", stage="failed", error=f"Некорректный результат анализа: {exc.error_count()} ошибок схемы")
            return
        except Exception as exc:  # noqa: BLE001 — any pipeline failure must end as a failed job, not a hung one
            log.exception("job %s: pipeline failed", job_id)
            self._store.update(job_id, status="failed", stage="failed", error=f"Не удалось обработать видео ({type(exc).__name__})")
            return

        self._store.update(
            job_id,
            status="done",
            stage="done",
            progress=1.0,
            result=result.model_dump(mode="json", exclude_none=True),
        )
