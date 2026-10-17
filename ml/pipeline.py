"""К2 — pipeline entry point used by the backend when PIPELINE=real.

Contract (do not change without the team lead):
    analyze(video_path, on_progress=None, time_budget_sec=None) -> dict
returns a dict valid against backend.app.schemas.AnalysisResult:
    {"verdict": "violation" | "no_violation", "violations": [...]}
on_progress(progress: float 0..1, stage: str) may be called any number of times.
time_budget_sec: soft deadline; on overrun return the best result so far instead of failing.
"""

from typing import Any, Callable

ProgressFn = Callable[[float, str], None]


def analyze(
    video_path: str,
    on_progress: ProgressFn | None = None,
    time_budget_sec: float | None = None,
) -> dict[str, Any]:
    raise NotImplementedError("real pipeline is not wired yet — run the backend with PIPELINE=mock")
