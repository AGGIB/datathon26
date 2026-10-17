"""Background execution of the analysis pipeline (К2) with a fixed worker pool."""

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path
from typing import Any, Callable, Protocol

from pydantic import ValidationError

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
        min_budget_sec: float = 20.0,
    ) -> None:
        self._store = store
        self._pipeline = pipeline
        self._hard_timeout = float(time_budget_sec)
        self._soft_budget = float(soft_budget_sec) if soft_budget_sec is not None else min(200.0, self._hard_timeout * 0.75)
        if self._hard_timeout < min_budget_sec:
            self._min_budget = min(0.01, self._hard_timeout * 0.1)
        else:
            self._min_budget = float(min_budget_sec)
        self._workers = workers
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="job")
        self._futures: dict[str, Future] = {}
        self._lock = threading.Lock()
        self._sem = threading.Semaphore(workers)
        self._lost_slots = 0

    @property
    def lost_slots(self) -> int:
        with self._lock:
            return self._lost_slots

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
        start_time = float(created_at) if created_at is not None else time.time()
        elapsed = time.time() - start_time
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

        if hard_remaining < self._min_budget:
            log.warning(
                "job %s: remaining time before acquire (%.1fs) below minimum budget %.1fs",
                job_id,
                hard_remaining,
                self._min_budget,
            )
            self._store.update(
                job_id,
                status="failed",
                stage="failed",
                error="Недостаточно времени для обработки видео",
            )
            return

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

        # Recompute remaining budget AFTER acquiring the slot to account for wait time
        elapsed_after_acquire = time.time() - start_time
        hard_remaining = self._hard_timeout - elapsed_after_acquire
        if hard_remaining <= 0:
            self._sem.release()
            log.warning("job %s: expired while waiting for compute slot (%.1fs)", job_id, elapsed_after_acquire)
            self._store.update(
                job_id,
                status="failed",
                stage="failed",
                error="Время ожидания в очереди превысило лимит",
            )
            return

        # Dual budgets: soft budget passed to pipeline, hard timeout enforced by watchdog
        grace_sec = min(10.0, self._hard_timeout * 0.1) if self._hard_timeout < 60 else max(10.0, self._hard_timeout - self._soft_budget)
        available = max(0.001, hard_remaining - grace_sec)

        if available < self._min_budget:
            self._sem.release()
            log.warning(
                "job %s: available budget (%.1fs) below minimum budget %.1fs",
                job_id,
                available,
                self._min_budget,
            )
            self._store.update(
                job_id,
                status="failed",
                stage="failed",
                error="Недостаточно времени для обработки видео",
            )
            return

        pipeline_budget = min(self._soft_budget, available)

        pipeline_result: list[dict[str, Any]] = []
        pipeline_exc: list[Exception] = []
        slot_released = threading.Event()
        is_lost_slot = [False]

        def release_slot() -> None:
            if not slot_released.is_set():
                slot_released.set()
                with self._lock:
                    if is_lost_slot[0]:
                        self._lost_slots = max(0, self._lost_slots - 1)
                self._sem.release()

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
            finally:
                release_slot()

        thread_started = False
        try:
            thread = threading.Thread(target=target, name=f"pipe-{job_id}", daemon=True)
            thread.start()
            thread_started = True
            thread.join(timeout=hard_remaining)

            if thread.is_alive():
                with self._lock:
                    is_lost_slot[0] = True
                    self._lost_slots += 1
                log.error(
                    "job %s: pipeline exceeded hard timeout %.1fs (allocated %.1fs); compute slot retained by lingering thread",
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
                # Note: slot is intentionally NOT released here while the thread is still executing,
                # preventing accumulation of concurrent heavy ML runs past workers limit.
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
        except Exception:
            if not thread_started:
                release_slot()
            raise
