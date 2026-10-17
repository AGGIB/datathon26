"""Runtime settings, read once from environment variables."""

import os
from dataclasses import dataclass
from pathlib import Path

PIPELINES = frozenset({"mock", "real"})


@dataclass(frozen=True)
class Settings:
    api_keys: frozenset[str]
    data_dir: Path
    require_api_key: bool = False
    max_upload_mb: int = 200
    workers: int = 2
    pipeline: str = "mock"
    mock_delay_sec: float = 2.0
    # Jury hard limit is 300 s per video; leave margin for upload and queueing.
    job_timeout_sec: float = 240.0
    frontend_dist: Path | None = None

    def __post_init__(self) -> None:
        if self.require_api_key and not self.api_keys:
            raise ValueError("API_KEYS must be set when REQUIRE_API_KEY is enabled")
        if self.pipeline not in PIPELINES:
            raise ValueError(f"PIPELINE must be one of {sorted(PIPELINES)}, got {self.pipeline!r}")
        if self.workers < 1:
            raise ValueError("WORKERS must be >= 1")

    @classmethod
    def from_env(cls) -> "Settings":
        keys = frozenset(k.strip() for k in os.environ.get("API_KEYS", "").split(",") if k.strip())
        dist = os.environ.get("FRONTEND_DIST")
        return cls(
            api_keys=keys,
            data_dir=Path(os.environ.get("DATA_DIR", "./var")).resolve(),
            require_api_key=os.environ.get("REQUIRE_API_KEY", "false").lower() == "true",
            max_upload_mb=int(os.environ.get("MAX_UPLOAD_MB", "200")),
            workers=int(os.environ.get("WORKERS", "2")),
            pipeline=os.environ.get("PIPELINE", "mock"),
            mock_delay_sec=float(os.environ.get("MOCK_DELAY_SEC", "2.0")),
            job_timeout_sec=float(os.environ.get("JOB_TIMEOUT_SEC", "240")),
            frontend_dist=Path(dist).resolve() if dist else None,
        )
