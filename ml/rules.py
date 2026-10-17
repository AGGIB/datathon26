"""Stage 2: rules turn perception (+ scene state) into violation candidates. Owner: Агыбай.

Contract used by ml/pipeline.py:
    find_candidates(perception, scene=None) -> list[Candidate]
`scene` is whatever ml/scene.py produces (layout + per-time states); rules must tolerate None.
"""

from typing import Any

from .candidates import Candidate
from .perception import Perception


def find_candidates(perception: Perception, scene: Any | None = None) -> list[Candidate]:
    return []
