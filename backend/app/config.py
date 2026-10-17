"""Runtime settings, read once from environment variables."""

import os
from dataclasses import dataclass
from pathlib import Path

PIPELINES = frozenset({"mock", "real"})


def _strip_inline_comment(line: str) -> str:
    in_quote: str | None = None
    for i, ch in enumerate(line):
        if ch in ('"', "'"):
            if in_quote == ch:
                in_quote = None
            elif in_quote is None:
                in_quote = ch
        elif ch == "#" and in_quote is None:
            return line[:i].strip()
    return line.strip()


def _parse_env_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    res: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = _strip_inline_comment(line)
        if not line:
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip("'\"")
            if k:
                res[k] = v
    return res


@dataclass(frozen=True)
class Settings:
    api_keys: frozenset[str]
    data_dir: Path
    max_upload_mb: int = 200
    workers: int = 2
    pipeline: str = "mock"
    mock_delay_sec: float = 2.0
    # Soft pipeline budget (K2 returns best effort result) and hard watchdog limit
    pipeline_timeout_sec: float = 200.0
    job_timeout_sec: float = 270.0
    min_budget_sec: float = 20.0
    frontend_dist: Path | None = None

    def __post_init__(self) -> None:
        if not self.api_keys:
            raise ValueError("At least one API key is required")
        for key in self.api_keys:
            if key.startswith("change-me-"):
                raise ValueError(f"Placeholder API key {key!r} is not allowed. Configure real keys.")
        if self.pipeline not in PIPELINES:
            raise ValueError(f"PIPELINE must be one of {sorted(PIPELINES)}, got {self.pipeline!r}")
        if self.workers < 1:
            raise ValueError("WORKERS must be >= 1")
        if self.min_budget_sec <= 0:
            raise ValueError("MIN_BUDGET_SEC must be > 0")
        if self.pipeline_timeout_sec <= 0:
            raise ValueError("PIPELINE_TIMEOUT_SEC must be > 0")
        if self.job_timeout_sec <= 0:
            raise ValueError("JOB_TIMEOUT_SEC must be > 0")

    @classmethod
    def from_env(cls, env_path: Path | None = None) -> "Settings":
        file_vars: dict[str, str] = {}
        candidates = [env_path] if env_path is not None else [Path(".env"), Path(__file__).resolve().parent.parent.parent / ".env"]
        for candidate in candidates:
            if candidate and candidate.is_file():
                file_vars = _parse_env_file(candidate)
                break

        def get_val(key: str, default: str | None = None) -> str | None:
            # os.environ has precedence, followed by .env file, then default
            if key in os.environ:
                return os.environ[key]
            if key in file_vars:
                return file_vars[key]
            return default

        raw_keys = get_val("API_KEYS", "") or ""
        keys = frozenset(k.strip() for k in raw_keys.split(",") if k.strip())
        if not keys:
            raise RuntimeError("API_KEYS is not set (comma-separated list, e.g. JURY_KEY,UI_KEY)")

        dist = get_val("FRONTEND_DIST")
        if not dist:
            default_dist = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
            if default_dist.is_dir():
                dist = str(default_dist)

        return cls(
            api_keys=keys,
            data_dir=Path(get_val("DATA_DIR", "./var")).resolve(),
            max_upload_mb=int(get_val("MAX_UPLOAD_MB", "200")),
            workers=int(get_val("WORKERS", "2")),
            pipeline=get_val("PIPELINE", "mock"),
            mock_delay_sec=float(get_val("MOCK_DELAY_SEC", "2.0")),
            pipeline_timeout_sec=float(get_val("PIPELINE_TIMEOUT_SEC", "200")),
            job_timeout_sec=float(get_val("JOB_TIMEOUT_SEC", "270")),
            min_budget_sec=float(get_val("MIN_BUDGET_SEC", "20.0")),
            frontend_dist=Path(dist).resolve() if dist else None,
        )
