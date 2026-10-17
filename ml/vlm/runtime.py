"""Startup-owned production seam; no CV or model imports during module import."""
from dataclasses import dataclass
import logging
import time
from pathlib import Path
from typing import Mapping, Sequence

from .contracts import Backend, FrameSource
from .pipeline_adapter import CatalogType, T, verify_pipeline_candidates
from .validation import number
from .verifier import remaining_budget


@dataclass(frozen=True)
class _Configuration:
    backend: Backend | None
    catalog: Mapping[str, CatalogType]
    frame_source: FrameSource | None
    timebase_verified: bool
    owned: bool = False


_configuration: _Configuration | None = None
logger = logging.getLogger(__name__)


def configure(model_path: str | Path | None = None, *,
              catalog: Mapping[str, CatalogType] | None = None,
              backend: Backend | None = None, frame_source: FrameSource | None = None,
              timebase_verified: bool = False, startup_timeout_s: float = 120,
              device: str = "cpu", max_new_tokens: int = 128) -> bool:
    """Prepare once at service startup, outside the remaining per-video K2 budget.

    Return runtime availability, not a model-quality claim. Reconfiguration is
    explicit; verification never retries a failed load or starts a new model.
    `catalog` is the owner's validated ml.kb catalog, not raw YAML.
    """
    global _configuration
    if type(timebase_verified) is not bool:
        raise ValueError("timebase_verified must be boolean")
    if not number(startup_timeout_s) or startup_timeout_s <= 0:
        raise ValueError("startup timeout must be finite positive")
    if _configuration is not None and _configuration.owned:
        _configuration.backend.close()
    _configuration = None
    try:
        if catalog is None:
            from ml.kb import catalog as owner_catalog
            catalog = owner_catalog()
        catalog_snapshot = dict(catalog)
    except Exception as exc:
        logger.warning("VLM catalogue unavailable: %s", type(exc).__name__)
        return False
    owned = backend is None and model_path is not None
    if owned:
        from .transformers_backend import LocalTransformersBackend
        backend = LocalTransformersBackend(model_path, device=device, max_new_tokens=max_new_tokens)
    if backend is not None:
        try:
            validate = getattr(backend, "validate", None)
            if validate is not None:
                validate(startup_timeout_s)
        except Exception:
            close = getattr(backend, "close", None)
            if close is not None:
                close()
            backend = None
            owned = False
    _configuration = _Configuration(backend, catalog_snapshot, frame_source, timebase_verified, owned)
    return backend is not None


def verify(candidates: Sequence[T], video_path: str | Path, budget_sec: float) -> list[T]:
    """Return every original frozen team Candidate, without confidence changes.

    One call covers the complete video candidate set. Rejections remain advisory
    until independent calibration authorizes a different integration contract.
    """
    budget = remaining_budget(budget_sec)
    deadline = time.monotonic() + budget
    originals = list(candidates)
    config = _configuration
    if config is None or time.monotonic() >= deadline:
        return originals
    owned_source = config.frame_source is None
    source = None
    retained = originals
    try:
        from .owner_frames import OwnerFrames
        source = OwnerFrames() if owned_source else config.frame_source
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return originals
        report = verify_pipeline_candidates(video_path, originals, config.catalog,
            backend=config.backend, frame_source=source,
            timebase_verified=owned_source or config.timebase_verified, budget_sec=remaining,
            _deadline=deadline)
        retained = list(report.retained_candidates)
    except Exception as exc:
        logger.warning("VLM verification unavailable: %s", type(exc).__name__)
    finally:
        if owned_source and source is not None:
            try:
                source.close()
            except Exception as exc:
                logger.warning("VLM verification unavailable: %s", type(exc).__name__)
                retained = originals
    return retained
