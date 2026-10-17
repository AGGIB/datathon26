"""Explicit startup and optional K2 augmentation. Unavailable VLM preserves CV."""
from dataclasses import dataclass
import logging
import math
from pathlib import Path
import time

from .contracts import Backend
from .discovery import DiscoveryBudget, discover_types
from .discovery_policy import TYPE_IDS, plan_windows

log = logging.getLogger(__name__)
MAX_VIDEO_SECONDS = 180.0


@dataclass(frozen=True)
class _Configuration:
    backend: Backend
    frame_source: object | None = None
    owned: bool = False
    max_calls: int = 2


_configuration: _Configuration | None = None


def configure_discovery(*, model_path: str | Path | None = None,
                        python_executable: str | Path | None = None,
                        backend: Backend | None = None, frame_source=None,
                        startup_timeout_s: float = 180, verify_actions: bool = False) -> bool:
    """Call once at startup, before accepting jobs; no network or implicit installs.

    Injected backends own their lifecycle. Owned MLX workers are closed on
    explicit reconfiguration and use the same action decoder as the measured
    experimental probe. A failed load stays unavailable for later jobs.
    For an injected backend, verify_actions declares the four-call video cap;
    the caller supplies its own verifier wrapper and still owns its lifecycle.
    """
    global _configuration
    if type(verify_actions) is not bool:
        raise ValueError("verify_actions must be boolean")
    if (isinstance(startup_timeout_s, bool) or not isinstance(startup_timeout_s, (float, int))
            or not math.isfinite(startup_timeout_s) or startup_timeout_s <= 0):
        raise ValueError("positive finite startup timeout required")
    previous, _configuration = _configuration, None
    if previous is not None and previous.owned:
        try:
            previous.backend.close()
        except Exception:
            log.warning("vlm discovery: previous worker could not close")
    owned = backend is None and model_path is not None and python_executable is not None
    try:
        if owned:
            from .mlx_backend import LocalMLXBackend
            backend = LocalMLXBackend(model_path, python_executable=python_executable,
                                      max_new_tokens=512, max_image_side=640)
            # Keep the raw worker in `backend` until wrapper creation succeeds,
            # so catalog/import/constructor failures still close its resources.
            from ml.kb import catalog

            if verify_actions:
                from .verified_action_backend import VerifiedActionBackend

                backend = VerifiedActionBackend(backend, catalog=catalog())
            else:
                from .action_backend import ActionContrastBackend

                backend = ActionContrastBackend(backend, catalog=catalog())
        if backend is None:
            return False
        validate = getattr(backend, "validate", None)
        if validate is not None:
            validate(timeout_s=startup_timeout_s)
    except Exception:
        if owned and backend is not None:
            try:
                backend.close()
            except Exception:
                log.warning("vlm discovery: unavailable worker could not close")
        log.warning("vlm discovery: local runtime unavailable")
        return False
    _configuration = _Configuration(backend, frame_source, owned, 4 if verify_actions else 2)
    return True


def augment_types(video_path, originals, types, *, duration_sec: float, deadline: float | None):
    """Add events only in free answer slots, within the caller's original deadline.

    CV selection is performed first and retained; even a higher-confidence VLM
    addition cannot displace it at assembly's five-event cap. Neither rejection
    nor a backend failure lowers an existing CV confidence.
    """
    config = _configuration
    originals = list(originals)
    if config is None:
        return originals
    now = time.monotonic()
    end = min(deadline, now + MAX_VIDEO_SECONDS) if deadline is not None else now + MAX_VIDEO_SECONDS
    if end <= now:
        return originals
    try:
        from ml.assemble import MAX_VIOLATIONS, MERGE_GAP_SEC, select

        baseline = select(originals, types)
        if len(baseline) >= MAX_VIOLATIONS:
            return originals
        report = discover_types(video_path, originals, types, duration_sec=duration_sec,
            windows=plan_windows(duration_sec), allowed_type_ids=TYPE_IDS,
            budget=DiscoveryBudget(end, max_calls=config.max_calls), backend=config.backend, frame_source=config.frame_source)
        # Final assembly merges near same-type episodes. Even non-overlapping
        # additions must stay outside that gap or they can rewrite a CV episode.
        separate = [addition for addition in report.added_candidates if not any(
            addition.type_id == cv.type_id and addition.t_start <= cv.t_end + MERGE_GAP_SEC
            and cv.t_start <= addition.t_end + MERGE_GAP_SEC for cv in baseline)]
        additions = sorted(select(separate, types), key=lambda item: -item.confidence)[
            :MAX_VIOLATIONS - len(baseline)]
        log.info("vlm discovery: status=%s calls=%d additions=%d", report.status, report.calls_used, len(additions))
        return [*originals, *additions]
    except Exception:
        log.warning("vlm discovery: failed, returning CV candidates")
        return originals
