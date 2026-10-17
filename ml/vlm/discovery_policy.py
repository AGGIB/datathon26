"""Frozen first experiment policy, independent of labels, names and camera geometry."""
import math

from .discovery import DiscoveryWindow

TYPE_IDS = (
    "banknote_returned_after_check", "no_visible_client_id_check", "previous_cash_left_on_desk",
    "cash_from_bag_stored_without_recount", "bag_accepted_without_check", "recount_after_client_left",
    "no_banknote_detector_used", "personal_items_on_desk", "food_or_drink_at_workplace",
    "key_left_in_safe_lock", "stamps_and_book_left_on_desk",
)
POLICY_ID = "uniform_halves_12_v1"


def plan_windows(duration: float) -> tuple[DiscoveryWindow, ...]:
    """12 images over a short clip, or two ordered halves over a longer clip.

    Sparse samples cannot establish that a check never happened. The prompt
    explicitly requires abstention for incomplete procedural evidence.
    """
    if isinstance(duration, bool) or not math.isfinite(duration) or duration <= 0:
        raise ValueError("positive finite duration required")
    last = max(0.0, round(duration - min(0.1, duration / 100), 6))
    middle = round(last / 2, 6)
    bounds = [(0.0, last)] if duration <= 30 else [(0.0, middle), (middle, last)]
    return tuple(DiscoveryWindow(start, end, tuple(sorted(set(
        round(start + (end - start) * i / 11, 6) for i in range(12))))) for start, end in bounds)
