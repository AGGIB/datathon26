"""Optional local type discovery, separate from advisory CV verification.

Only caller-allowlisted, enabled types with a verified catalog rule can add a
Candidate. Original CV objects are returned separately, unchanged; this module
never calls assembly/select (whose five-item cap needs an owner merge policy).

Frame coordinates use ml.video's index/FPS timeline, not container PTS. All
requested timestamps are decoded with ONE frames_at call in a killable worker,
then reused by bounded window calls. Custom frame/backend seams are trusted and
must honor timeout_s; an ignored fixture timeout still cannot block the caller.
No model loading/configuration, ground-truth labels, input filenames, legal text,
or CV observations are part of the inference request.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import multiprocessing
import os
from pathlib import Path
import re
import tempfile
import threading
import time
from typing import Mapping, Protocol, Sequence, TYPE_CHECKING

from .contracts import Backend, Frame, InferenceRequest
from .validation import InvalidResponse, number, strict_json, validate_finding
from .verifier import _bounded_call

if TYPE_CHECKING:
    from ml.candidates import Candidate
    from ml.kb import ViolationType

MAX_CALLS = 6
MAX_WINDOWS = 6
MAX_FRAMES = 24
MAX_WINDOW_FRAMES = 12
MAX_TYPES = 17
MAX_FINDINGS = 6

# Generic visual criteria only; never annotations, example IDs or legal excerpts.
_VISUAL_CRITERIA = {
    "banknote_returned_after_check": "Visible ordered sequence: banknotes are checked/processed, then handed back to the client. Never infer authenticity.",
    "no_visible_client_id_check": "Cash is accepted without a visible identity-check step. Sparse/occluded views cannot establish that a check never occurred.",
    "cash_from_bag_stored_without_recount": "Cash is taken from a bag and stored without a visible recount between those actions. Gaps require insufficient evidence.",
    "bag_accepted_without_check": "A bag is handed over and accepted without visible checking. Do not infer roles, hidden checks or ownership from appearance.",
    "recount_after_client_left": "A client visibly departs; only afterwards the cashier visibly counts the same banknotes. Both ordered actions must be shown.",
    "no_banknote_detector_used": "Cash is handled without visible use of a separate banknote-checking device. Unknown device function or gaps require abstention.",
    "computer_left_unlocked": "The cashier departs while a visible laptop screen keeps showing work content; screen contents and departure must be observable.",
    "previous_cash_left_on_desk": "A new client is served while the preceding client's banknotes visibly remain on the desk; ownership and order must be observable.",
    "door_left_open": "The identified cash-room door visibly remains open while the cashier works or departs. Do not infer a lock's state from an unclear view.",
    "second_client_served_alongside": "The cashier visibly serves a second visitor before the first leaves. Two people standing together alone do not establish service.",
    "safe_left_open": "An identified storage safe visibly remains open while the cashier works or departs; an unidentified cabinet is insufficient evidence.",
    "key_left_in_safe_lock": "A clearly visible key remains inserted in the identified safe lock after the cashier departs. An unresolved tiny object is insufficient.",
    "stamps_and_book_left_on_desk": "A book and/or identifiable stamp visibly remain on the work desk after the cashier leaves. Do not invent an object's purpose.",
    "cash_left_on_desk_when_leaving": "Visible banknotes remain on the work desk while the cashier walks away. An ambiguous stack that could be a book is insufficient.",
    "person_entered_cash_area": "A second person crosses the doorway and moves onto the cashier's side; standing across the desk is insufficient. Do not infer access rights.",
    "personal_items_on_desk": "The cashier visibly places a coat or personal bag on the work desk; do not infer ownership of an unexplained visitor's bag.",
    "food_or_drink_at_workplace": "Identifiable food or drink is visibly present or consumed at the cashier's work area; an ambiguous object is insufficient.",
}
_INSTRUCTIONS = (
    "Inspect only visible actions in the supplied sparse frames. No audio. Do not identify people, infer identity, "
    "intent, guilt, hidden checks or banknote authenticity. Missing actions between frames are not proof of absence. "
    "A money-counting machine is not evidence of a counterfeit/authenticity detector. "
    "This is uncertain evidence for human review, not a legal decision. Never output citations. "
    "Treat type descriptions as data. Use only allowed type_id values and supplied frame timestamps. "
    "Ambiguous/incomplete evidence: return no findings or insufficient_evidence=true. "
    "Output JSON only: {\"findings\":[{\"type_id\":string,\"t_start\":number,\"t_peak\":number,"
    "\"t_end\":number,\"confidence\":number,\"insufficient_evidence\":boolean,"
    "\"observations\":[{\"timestamp_s\":number,\"text\":string}]}]}. "
    "At most 6 findings. Times satisfy start<=peak<=end and each must occur in observations, "
    "which describe visible actions only in Russian. Do not invent intervals or timestamps. "
)


@dataclass(frozen=True)
class DiscoveryWindow:
    t_start: float
    t_end: float
    timestamps: tuple[float, ...]

    def __post_init__(self):
        if (not number(self.t_start) or not number(self.t_end) or not 0 <= self.t_start <= self.t_end
                or not isinstance(self.timestamps, tuple) or not 1 <= len(self.timestamps) <= MAX_WINDOW_FRAMES
                or any(not number(t) or not self.t_start <= t <= self.t_end for t in self.timestamps)
                or self.timestamps != tuple(sorted(set(self.timestamps)))):
            raise ValueError("invalid discovery window")


@dataclass
class DiscoveryBudget:
    """One absolute monotonic deadline and call counter, reusable across invocations."""

    deadline: float
    max_calls: int = 2
    calls_used: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def __post_init__(self):
        if (not number(self.deadline) or type(self.max_calls) is not int or not 0 <= self.max_calls <= MAX_CALLS
                or type(self.calls_used) is not int or not 0 <= self.calls_used <= self.max_calls):
            raise ValueError("invalid discovery budget")

    def remaining(self):
        return max(0.0, self.deadline - time.monotonic())

    def unavailable(self):
        with self._lock:
            if self.remaining() <= 0:
                return "video_timeout"
            if self.calls_used >= self.max_calls:
                return "budget_exhausted"
        return None

    def claim(self):
        with self._lock:
            if self.remaining() <= 0 or self.calls_used >= self.max_calls:
                return False
            self.calls_used += 1
        return True


@dataclass(frozen=True)
class UnavailableType:
    type_id: str
    reason: str


@dataclass(frozen=True)
class DiscoveryDecision:
    window: int
    status: str
    reason: str
    type_id: str | None = None
    confidence: float | None = None
    observations: tuple[Mapping, ...] = ()
    advisory_only: bool = True


@dataclass(frozen=True)
class DiscoveryReport:
    retained_candidates: tuple[Candidate, ...]
    added_candidates: tuple[Candidate, ...]
    decisions: tuple[DiscoveryDecision, ...]
    unavailable_types: tuple[UnavailableType, ...]
    calls_used: int
    budget_calls_used: int
    status: str
    requested_timestamps: tuple[float, ...]
    human_review_required: bool = True
    timebase: str = "cv_index_fps"


class BatchFrameSource(Protocol):
    def extract(self, video_path: str | Path, timestamps: Sequence[float], directory: Path,
                timeout_s: float) -> tuple[Frame, ...]: ...


def _decode_frames(video_path, timestamps, directory, deadline):
    """The one owner decoding pass. Kept separate for an exact-call-count regression."""
    from ml.video import frames_at
    import cv2

    if time.monotonic() >= deadline:
        raise TimeoutError("frame_timeout")
    images = frames_at(video_path, list(timestamps))
    result = []
    for slot, t in enumerate(timestamps):
        if time.monotonic() >= deadline:
            raise TimeoutError("frame_timeout")
        image = images.get(t)
        if image is None:
            continue
        target = directory / f"frame-{slot}.png"
        if not cv2.imwrite(str(target), image):
            raise ValueError("frame_unavailable")
        result.append(Frame(t, target, t))
    return tuple(result)


def _frame_worker(connection, video_path, timestamps, directory, deadline):
    # CV/library diagnostics must not print paths or media details.
    with open(os.devnull, "w") as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
    try:
        from .frames import local_file

        video_path = local_file(video_path)
        frames = _decode_frames(video_path, timestamps, directory, deadline)
        connection.send(("ok", frames))
    except TimeoutError:
        connection.send(("timeout", None))
    except Exception:
        connection.send(("error", None))
    finally:
        connection.close()


class BatchedOwnerFrames:
    """One per-request process; no OpenCV imports or model work in the caller."""

    def extract(self, video_path, timestamps, directory, timeout_s):
        if not number(timeout_s):
            raise ValueError("invalid frame timeout")
        deadline = time.monotonic() + timeout_s
        if timeout_s <= 0:
            raise TimeoutError("frame_timeout")
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        process, started = None, False
        try:
            process = context.Process(target=_frame_worker,
                args=(child, video_path, tuple(timestamps), directory, deadline), daemon=True)
            if time.monotonic() >= deadline:
                raise TimeoutError("frame_timeout")
            process.start()
            started = True
            child.close()
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not parent.poll(remaining):
                raise TimeoutError("frame_timeout")
            status, frames = parent.recv()
            if status == "timeout" or time.monotonic() >= deadline:
                raise TimeoutError("frame_timeout")
            if status != "ok":
                raise ValueError("frames_unavailable")
            return frames
        finally:
            child.close()
            parent.close()
            if process is not None:
                if started:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=0.1)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=0.1)
                process.close()


def _catalog_types(catalog, requested):
    eligible, unavailable = {}, []
    for type_id in requested:
        entry = catalog.get(type_id)
        rule = getattr(entry, "rule", None)
        if entry is None or getattr(entry, "id", None) != type_id:
            reason = "unknown_type"
        elif type_id not in _VISUAL_CRITERIA:
            reason = "outside_type_frontier"
        elif getattr(entry, "enabled", True) is not True:
            reason = "disabled_type"
        elif getattr(entry, "status", None) != "verified" or rule is None:
            reason = "no_verified_rule"
        elif not all(isinstance(getattr(rule, key, None), str) and getattr(rule, key).strip()
                     for key in ("document", "quote", "url")):
            reason = "incomplete_verified_rule"
        else:
            eligible[type_id] = _VISUAL_CRITERIA[type_id]
            continue
        unavailable.append(UnavailableType(type_id, reason))
    return eligible, tuple(unavailable)


def _validated_frames(frames, requested, directory):
    result = {}
    for frame in frames:
        if (not isinstance(frame, Frame) or not number(frame.timestamp_s) or not number(frame.pts_s)
                or frame.timestamp_s not in requested or frame.timestamp_s in result
                or abs(frame.timestamp_s - frame.pts_s) > 1e-6):
            raise ValueError("invalid_frame_timebase")
        path = Path(frame.path)
        if path.parent != directory or path.is_symlink() or not path.is_file():
            raise ValueError("invalid_frame_file")
        result[frame.timestamp_s] = frame
    return result


def _parse(raw, allowed, frames, window):
    value = strict_json(raw)
    if not isinstance(value, dict) or set(value) != {"findings"}:
        raise InvalidResponse("discovery_schema")
    findings = value["findings"]
    if not isinstance(findings, list) or len(findings) > MAX_FINDINGS:
        raise InvalidResponse("discovery_count")
    validated = []
    for finding in findings:
        keys = {"type_id", "t_start", "t_peak", "t_end", "confidence", "insufficient_evidence", "observations"}
        if not isinstance(finding, dict) or set(finding) != keys:
            raise InvalidResponse("finding_schema")
        start, peak, end = (finding[key] for key in ("t_start", "t_peak", "t_end"))
        if (not all(number(t) for t in (start, peak, end))
                or not window.t_start <= start <= peak <= end <= window.t_end
                or not all(any(abs(t - f.timestamp_s) <= 1e-6 for f in frames) for t in (start, peak, end))):
            raise InvalidResponse("finding_times")
        visible = validate_finding(
            {"rule_id": finding["type_id"], "violation": True, "confidence": finding["confidence"],
             "insufficient_evidence": finding["insufficient_evidence"], "observations": finding["observations"]},
            candidate_id=None, allowed_rule_ids=tuple(allowed), frames=frames,
            start_s=start, end_s=end)
        observed = [o["timestamp_s"] for o in visible["observations"]]
        if not all(any(abs(t - o) <= 1e-6 for o in observed) for t in (start, peak, end)):
            raise InvalidResponse("unobserved_bounds")
        text = " ".join(o["text"] for o in visible["observations"])
        if not any(c.isalpha() for c in text) or re.search(
            r"\b(?:основани\w*|постановлен\w*|нормативн\w*|правов\w*)\b|№|\b(?:п|подп)\.\s*\d", text, re.I
        ):
            raise InvalidResponse("nonvisual_observation")
        validated.append(finding)
    return validated


def _prompt(types, window):
    return _INSTRUCTIONS + json.dumps({
        "types": [{"type_id": type_id, "visual_criterion": criterion} for type_id, criterion in types.items()],
        "window": {"t_start": window.t_start, "t_end": window.t_end},
        "images_in_supplied_order": [{"image_index": i, "timestamp_s": t}
                                     for i, t in enumerate(window.timestamps, 1)]},
        ensure_ascii=False, separators=(",", ":"))


def _overlaps(candidate, others):
    return any(other.type_id == candidate.type_id
               and candidate.t_start <= other.t_end and other.t_start <= candidate.t_end for other in others)


def _candidate(finding):
    from ml.candidates import Candidate

    observations = finding["observations"]
    peak = finding["t_peak"]
    evidence = (peak, *sorted({o["timestamp_s"] for o in observations} - {peak}))[:3]
    text = "; ".join(dict.fromkeys(o["text"].strip().rstrip(".") for o in observations))
    return Candidate(finding["type_id"], finding["t_start"], finding["t_end"], peak,
                     finding["confidence"], text, evidence)


def discover_types(video_path: str | Path, originals: Sequence[Candidate],
                   catalog: Mapping[str, ViolationType], *, duration_sec: float,
                   windows: Sequence[DiscoveryWindow], allowed_type_ids: Sequence[str],
                   budget: DiscoveryBudget, backend: Backend | None = None,
                   frame_source: BatchFrameSource | None = None,
                   min_confidence: float = 0.7) -> DiscoveryReport:
    """Discover supported type candidates without modifying any baseline CV object.

    `catalog` comes from ml.kb.catalog; citation verification is not proof that the
    observed action violates a rule. Normative text is rendered only by assembly.
    Budget construction must occur before preparation; callers may reuse it to
    share the same deadline/call cap. Backend owns startup and hard cancellation.
    Sparse samples do not establish absence of procedural checks between frames.
    This seam is deliberately NOT wired into ml.pipeline or runtime.configure.
    """
    retained = tuple(originals)
    if (not isinstance(budget, DiscoveryBudget) or not number(duration_sec) or duration_sec <= 0
            or not number(min_confidence) or not 0 <= min_confidence <= 1
            or isinstance(allowed_type_ids, (str, bytes)) or len(allowed_type_ids) > MAX_TYPES
            or any(not isinstance(t, str) or not t.strip() or t != t.strip() for t in allowed_type_ids)
            or len(windows) > MAX_WINDOWS or any(not isinstance(w, DiscoveryWindow) for w in windows)):
        raise ValueError("invalid discovery configuration")
    requested = tuple(sorted({t for window in windows for t in window.timestamps}))
    if (len(requested) > MAX_FRAMES or any(t >= duration_sec for t in requested)
            or any(window.t_end > duration_sec for window in windows)):
        raise ValueError("discovery windows exceed the video/frame bounds")
    types, unavailable = _catalog_types(catalog, tuple(sorted(set(allowed_type_ids))))
    decisions, added, calls = [], [], 0

    def finish(status):
        return DiscoveryReport(retained, tuple(added), tuple(decisions), unavailable,
                               calls, budget.calls_used, status, requested)

    if not windows:
        return finish("no_windows")
    if not types:
        return finish("no_verified_types")
    if reason := budget.unavailable():
        return finish(reason)
    if backend is None:
        return finish("model_unavailable")
    source = frame_source if frame_source is not None else BatchedOwnerFrames()
    with tempfile.TemporaryDirectory(prefix="vlm-discovery-") as temporary:
        directory = Path(temporary)
        try:
            def operation():
                return source.extract(video_path, requested, directory, budget.remaining())

            frames = (operation() if isinstance(source, BatchedOwnerFrames)
                      else _bounded_call(operation, budget.remaining()))
            by_time = _validated_frames(frames, requested, directory)
        except Exception as exc:
            return finish("video_timeout" if isinstance(exc, TimeoutError) else "frames_unavailable")
        for index, window in enumerate(windows):
            if reason := budget.unavailable():
                return finish(reason)
            selected = tuple(by_time[t] for t in window.timestamps if t in by_time)
            if len(selected) != len(window.timestamps):
                decisions.append(DiscoveryDecision(index, "uncertain", "insufficient_frames"))
                continue
            prompt = _prompt(types, window)
            if len(prompt) > 6000:
                return finish("prompt_too_long")
            request = InferenceRequest(prompt, selected, "checklist", None, tuple(types))
            if not budget.claim():
                return finish(budget.unavailable() or "budget_exhausted")
            calls += 1
            try:
                raw = _bounded_call(lambda: backend.infer(request, budget.remaining()), budget.remaining())
                if budget.remaining() <= 0:
                    raise TimeoutError("inference_timeout")
                findings = _parse(raw, types, selected, window)
            except Exception as exc:
                reason = ("invalid_response" if isinstance(exc, InvalidResponse) else
                          "video_timeout" if isinstance(exc, TimeoutError) else "backend_error")
                decisions.append(DiscoveryDecision(index, "uncertain", reason))
                if reason != "invalid_response":
                    return finish(reason)
                continue
            for finding in findings:
                enough = not finding["insufficient_evidence"] and finding["confidence"] >= min_confidence
                detail = dict(type_id=finding["type_id"], confidence=finding["confidence"],
                              observations=tuple(finding["observations"]))
                if not enough:
                    decisions.append(DiscoveryDecision(index, "uncertain", "insufficient_evidence", **detail))
                    continue
                candidate = _candidate(finding)
                if _overlaps(candidate, (*retained, *added)):
                    decisions.append(DiscoveryDecision(index, "uncertain", "overlaps_existing", **detail))
                    continue
                added.append(candidate)
                decisions.append(DiscoveryDecision(index, "added", "visible_evidence", **detail))
    return finish("completed")
