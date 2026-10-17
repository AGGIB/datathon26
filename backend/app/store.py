"""Disk-backed job store: one directory per job with job.json, so state survives restarts."""

import json
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any

JOB_ID_RE = re.compile(r"^[a-f0-9]{8,32}$")
JOB_FILE = "job.json"


def new_job_id() -> str:
    return uuid.uuid4().hex[:12]


def is_valid_job_id(job_id: str) -> bool:
    # Guards against path traversal: job ids map directly to directories.
    return bool(JOB_ID_RE.match(job_id))


class JobStore:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def job_dir(self, job_id: str) -> Path:
        if not is_valid_job_id(job_id):
            raise ValueError(f"invalid job id: {job_id!r}")
        return self._root / job_id

    def create(self, job_id: str, input_path: Path, filename: str) -> dict[str, Any]:
        now = time.time()
        record = {
            "job_id": job_id,
            "status": "processing",
            "progress": 0.0,
            "stage": "queued",
            "filename": filename,
            "input_path": str(input_path),
            "created_at": now,
            "updated_at": now,
        }
        with self._lock:
            self._write(job_id, record)
        return record

    def get(self, job_id: str) -> dict[str, Any] | None:
        if not is_valid_job_id(job_id):
            return None
        path = self._root / job_id / JOB_FILE
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def update(self, job_id: str, **changes: Any) -> dict[str, Any]:
        with self._lock:
            current = self.get(job_id)
            if current is None:
                raise KeyError(job_id)
            updated = {**current, **changes, "updated_at": time.time()}
            self._write(job_id, updated)
        return updated

    def update_progress(self, job_id: str, progress: float, stage: str) -> bool:
        """Atomically update progress and stage only if job is still in 'processing' status."""
        with self._lock:
            current = self.get(job_id)
            if current is None or current.get("status") != "processing":
                return False
            updated = {**current, "progress": progress, "stage": stage, "updated_at": time.time()}
            self._write(job_id, updated)
            return True

    def unfinished(self) -> list[dict[str, Any]]:
        records = (self.get(p.name) for p in self._root.iterdir() if p.is_dir())
        return [r for r in records if r is not None and r["status"] == "processing"]

    def _write(self, job_id: str, record: dict[str, Any]) -> None:
        directory = self.job_dir(job_id)
        directory.mkdir(parents=True, exist_ok=True)
        tmp = directory / f".{JOB_FILE}.tmp"
        tmp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, directory / JOB_FILE)
