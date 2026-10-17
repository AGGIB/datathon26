"""Background execution of the analysis pipeline (К2) with a fixed worker pool."""

import logging
import threading
import time
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
    def __init__(
        self,
        store: JobStore,
        pipeline: PipelineFn,
        workers: int,
        time_budget_sec: float = 270.0,
        soft_budget_sec: float | None = None,
    ) -> None:
        self._store = store
        self._pipeline = pipeline
        self._hard_timeout = float(time_budget_sec)
        self._soft_budget = float(soft_budget_sec) if soft_budget_sec is not None else min(200.0, self._hard_timeout * 0.75)
        self._workers = workers
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="job")
        self._futures: dict[str, Future] = {}
        self._lock = threading.Lock()
        self._sem = threading.Semaphore(workers)

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
        try:
            self._run_internal(job_id, video_path)
        except Exception as exc:  # noqa: BLE001
            log.exception("job %s: unhandled exception in _run", job_id)
            self._store.update(
                job_id,
                status="failed",
                stage="failed",
                error=f"Ошибка обработки видео ({type(exc).__name__})",
            )

    def _run_internal(self, job_id: str, video_path: Path) -> None:
        def on_progress(progress: float, stage: str) -> None:
            # Atomically updates progress only while job status is still 'processing'
            self._store.update_progress(
                job_id,
                progress=round(max(0.0, min(1.0, progress)), 3),
                stage=stage,
            )

        record = self._store.get(job_id) or {}
        created_at = record.get("created_at")
        if created_at is not None:
            elapsed = time.time() - float(created_at)
            hard_remaining = self._hard_timeout - elapsed
            if hard_remaining <= 0:
                log.warning("job %s: queue wait (%.1fs) exceeded hard timeout %.1fs", job_id, elapsed, self._hard_timeout)
                self._store.update(
                    job_id,
                    status="failed",
                    stage="failed",
                    error="Время ожидания в очереди превысило лимит",
                )
                return
        else:
            hard_remaining = self._hard_timeout

        # Dual budgets: soft budget passed to pipeline, hard timeout enforced by watchdog
        grace_sec = max(10.0, self._hard_timeout - self._soft_budget)
        pipeline_budget = max(1.0, min(self._soft_budget, hard_remaining - grace_sec))

        # Concurrency guard: ensure active ML executions do not exceed workers
        if not self._sem.acquire(timeout=max(0.001, hard_remaining)):
            log.warning("job %s: could not acquire compute slot within %.1fs", job_id, hard_remaining)
            self._store.update(
                job_id,
                status="failed",
                stage="failed",
                error="Не удалось выделить вычислительные ресурсы (очередь перегружена)",
            )
            return

        pipeline_result: list[dict[str, Any]] = []
        pipeline_exc: list[Exception] = []

        try:
            def target() -> None:
                try:
                    res = self._pipeline(
                        str(video_path),
                        on_progress=on_progress,
                        time_budget_sec=pipeline_budget,
                    )
                    pipeline_result.append(res)
                except Exception as exc:  # noqa: BLE001
                    pipeline_exc.append(exc)

            thread = threading.Thread(target=target, name=f"pipe-{job_id}", daemon=True)
            thread.start()
            thread.join(timeout=hard_remaining)

            if thread.is_alive():
                log.warning(
                    "job %s: pipeline exceeded hard timeout %.1fs (allocated %.1fs)",
                    job_id,
                    self._hard_timeout,
                    pipeline_budget,
                )
                self._store.update(
                    job_id,
                    status="failed",
                    stage="failed",
                    error="Превышено максимальное время обработки видео (таймаут)",
                )
                return

            if pipeline_exc:
                log.exception("job %s: pipeline raised exception", job_id, exc_info=pipeline_exc[0])
                self._store.update(
                    job_id,
                    status="failed",
                    stage="failed",
                    error=f"Не удалось обработать видео ({type(pipeline_exc[0]).__name__})",
                )
                return

            if not pipeline_result:
                self._store.update(
                    job_id,
                    status="failed",
                    stage="failed",
                    error="Пайплайн не вернул результат",
                )
                return

            try:
                result = AnalysisResult.model_validate(pipeline_result[0])
            except ValidationError as exc:
                log.exception("job %s: pipeline returned invalid schema", job_id)
                self._store.update(
                    job_id,
                    status="failed",
                    stage="failed",
                    error=f"Некорректный результат анализа: {exc.error_count()} ошибок схемы",
                )
                return

            self._store.update(
                job_id,
                status="done",
                stage="done",
                progress=1.0,
                result=result.model_dump(mode="json", exclude_none=True),
            )
        finally:
            self._sem.release()
