"""Structural/completeness checks for private human reviews; never proves viewing.

No media reads, model calls, output, dates or reviewer identities are generated.
"""

from collections import defaultdict
from datetime import date
import json
import math
from pathlib import Path, PurePosixPath
import re
import unicodedata
from uuid import UUID
from ml.evaluation_labels import (
    parse_timestamp,
    source_index as shared_source_index,
    video_key as shared_video_key,
)

HEX = re.compile(r"[0-9a-f]{64}")
DECISIONS = {"confirmed", "rejected", "unclear"}
CONSENSUS = DECISIONS | {"pending", "disputed"}
MACHINE = re.compile(
    r"^(?:codex|chatgpt|qwen|mlx|gpt|smolvlm|siglip|llm|vlm|ai|model|agent|assistant|модель|агент|ии)(?:\d.*)?$"
)


def require(condition, code):
    if not condition:
        raise ValueError(code)


def text(value):
    return isinstance(value, str) and bool(value.strip())


def in_enum(value, allowed):
    return isinstance(value, str) and value in allowed


def number(value):
    try:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )
    except OverflowError:
        return False


def digest(value):
    return isinstance(value, str) and HEX.fullmatch(value) is not None


def video_key(value):
    require(isinstance(value, str), "VIDEO_ID_INVALID")
    try:
        return shared_video_key(unicodedata.normalize("NFC", value))
    except ValueError as error:
        raise ValueError("VIDEO_ID_INVALID") from error


def indexed_ids(values):
    try:
        return shared_source_index(
            [{"video": unicodedata.normalize("NFC", value)} for value in values],
            "pilot selection",
        )
    except (ValueError, TypeError) as error:
        raise ValueError("PILOT_SELECTION_INVALID") from error


def human(value, decision=False):
    require(
        isinstance(value, dict) and value.get("kind") == "human", "REVIEW_KIND_INVALID"
    )
    require(text(value.get("reviewer")), "REVIEWER_REQUIRED")
    name = " ".join(unicodedata.normalize("NFC", value["reviewer"]).casefold().split())
    words = re.findall(r"[^\W_]+", name, flags=re.UNICODE)
    require(
        words and not any(MACHINE.fullmatch(word) for word in words),
        "REVIEWER_NOT_HUMAN",
    )
    stamp = value.get("date")
    require(isinstance(stamp, str), "REVIEW_DATE_INVALID")
    try:
        require(date.fromisoformat(stamp).isoformat() == stamp, "REVIEW_DATE_INVALID")
    except ValueError as error:
        raise ValueError("REVIEW_DATE_INVALID") from error
    if decision:
        require(in_enum(value.get("decision"), DECISIONS), "REVIEW_DECISION_INVALID")
    return name


def pair(first, second, decision=False):
    left = human(first, decision) if first is not None else None
    right = human(second, decision) if second is not None else None
    require(not left or not right or left != right, "REVIEWERS_NOT_DISTINCT")
    return left is not None, right is not None


def context(value):
    require(isinstance(value, dict), "REVIEWED_CONTEXT_INVALID")
    human(value)
    require(text(value.get("source_reference")), "CONTEXT_SOURCE_REQUIRED")
    for key in ("angle_id", "source_group"):
        require(
            key in value and (value[key] is None or text(value[key])),
            "REVIEWED_CONTEXT_INVALID",
        )


def effective_context(inventory_clip, adjudication_clip):
    reviewed = adjudication_clip.get("reviewed_context")
    if reviewed is not None:
        context(reviewed)
        return {key: reviewed[key] for key in ("angle_id", "source_group")}
    return {key: inventory_clip[key] for key in ("angle_id", "source_group")}


def inventory_rows(inventory):
    require(
        isinstance(inventory, dict)
        and type(inventory.get("schema_version")) is int
        and inventory["schema_version"] == 1
        and inventory.get("kind") == "pilot_review_inventory"
        and inventory.get("usage") == "development",
        "INVENTORY_SCHEMA_INVALID",
    )
    selection, organiser = inventory.get("selection"), inventory.get("organiser_source")
    require(
        isinstance(selection, dict) and isinstance(organiser, dict),
        "INVENTORY_SOURCE_INVALID",
    )
    for entry, hash_key in ((selection, "sha256"), (organiser, "content_sha256")):
        require(
            text(entry.get("path"))
            and Path(entry["path"]).is_absolute()
            and digest(entry.get(hash_key)),
            "INVENTORY_SOURCE_INVALID",
        )
    require(
        text(inventory.get("videos_root"))
        and Path(inventory["videos_root"]).is_absolute(),
        "VIDEOS_ROOT_INVALID",
    )
    ids = selection.get("pilot_ids")
    require(
        isinstance(ids, list)
        and len(ids) == 8
        and all(text(v) for v in ids)
        and len({video_key(v) for v in ids}) == 8,
        "PILOT_SELECTION_INVALID",
    )
    require(len(indexed_ids(ids)) == 8, "PILOT_SELECTION_INVALID")
    clips = inventory.get("clips")
    require(isinstance(clips, list) and len(clips) == 8, "INVENTORY_CLIPS_INVALID")
    rows, groups = {}, defaultdict(list)
    for row in clips:
        require(
            isinstance(row, dict)
            and text(row.get("video"))
            and text(row.get("video_key")),
            "INVENTORY_CLIP_INVALID",
        )
        key = row["video_key"]
        require(
            key == video_key(row["video"]) and key not in rows,
            "INVENTORY_VIDEO_KEY_INVALID",
        )
        relative = row.get("relative_path")
        require(
            text(relative)
            and "\\" not in relative
            and "\x00" not in relative
            and not PurePosixPath(relative).is_absolute()
            and ".." not in PurePosixPath(relative).parts
            and relative not in {".", ""},
            "INVENTORY_RELATIVE_PATH_INVALID",
        )
        require(
            digest(row.get("content_sha256"))
            and type(row.get("size_bytes")) is int
            and row["size_bytes"] > 0
            and number(row.get("duration_sec"))
            and row["duration_sec"] > 0,
            "INVENTORY_CONTENT_INVALID",
        )
        metadata = row.get("metadata_provenance")
        require(
            isinstance(metadata, dict)
            and metadata.get("kind") == "ffprobe_numeric"
            and metadata.get("tool") == "ffprobe"
            and metadata.get("source_content_sha256") == row["content_sha256"],
            "INVENTORY_METADATA_INVALID",
        )
        original = row.get("organizer")
        require(
            isinstance(original, dict)
            and text(original.get("video"))
            and video_key(original["video"]) == key
            and isinstance(original.get("label"), str)
            and type(original.get("positive")) is bool
            and type(original.get("source_row")) is int
            and original["source_row"] > 0
            and original.get("source_sha256") == organiser["content_sha256"],
            "ORGANIZER_PROVENANCE_INVALID",
        )
        raw = original.get("timestamp")
        require(
            raw is None or isinstance(raw, str) or number(raw),
            "ORGANIZER_TIMESTAMP_INVALID",
        )
        parsed = original.get("timestamp_sec")
        require(
            parsed is None or number(parsed) and parsed >= 0,
            "ORGANIZER_TIMESTAMP_INVALID",
        )
        try:
            expected = (
                None
                if raw is None or isinstance(raw, str) and not raw.strip()
                else parse_timestamp(raw)
            )
            require(expected == parsed, "ORGANIZER_TIMESTAMP_INVALID")
        except ValueError as error:
            raise ValueError("ORGANIZER_TIMESTAMP_INVALID") from error
        for field in ("angle_id", "source_group"):
            require(
                field in row and (row[field] is None or text(row[field])),
                "INVENTORY_CONTEXT_INVALID",
            )
        require(
            row.get("content_group") == "sha256:" + row["content_sha256"],
            "CONTENT_GROUP_INVALID",
        )
        rows[key] = row
        groups[row["content_sha256"]].append(key)
    require(set(rows) == {video_key(v) for v in ids}, "PILOT_SELECTION_MISMATCH")
    return rows, {sha: keys for sha, keys in groups.items() if len(keys) > 1}


def draft_adjudication(inventory, inventory_sha256):
    rows, duplicates = inventory_rows(inventory)
    require(digest(inventory_sha256), "INVENTORY_SHA_INVALID")
    return {
        "schema_version": 1,
        "kind": "pilot_review_adjudication",
        "inventory_sha256": inventory_sha256,
        "clips": [
            {
                "video_key": key,
                "content_sha256": row["content_sha256"],
                "coverage": "pending",
                "primary_review": None,
                "secondary_review": None,
                "reviewed_context": None,
                "organizer_clarification": None,
                "clip_decision": "pending",
                "events": [],
                "disagreements": [],
            }
            for key, row in rows.items()
        ],
        "duplicate_decisions": [
            {
                "content_sha256": sha,
                "video_keys": keys,
                "consensus": "pending",
                "reviewers": [],
                "note": None,
                "event_links": [],
            }
            for sha, keys in duplicates.items()
        ],
    }


def allowed_ids(type_ids):
    if type_ids is not None:
        return set(type_ids)
    path = Path(__file__).resolve().parents[1] / "kb/type-ids.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        row["id"] for row in data["types"] if row["catalog_section"] == "violations"
    }


def event_state(event, row, allowed, event_ids, clip_disagreements):
    require(isinstance(event, dict), "EVENT_SCHEMA_INVALID")
    eid = event.get("event_id")
    require(isinstance(eid, str), "EVENT_ID_INVALID")
    try:
        require(str(UUID(eid)) == eid and eid not in event_ids, "EVENT_ID_INVALID")
    except (ValueError, AttributeError) as error:
        raise ValueError("EVENT_ID_INVALID") from error
    event_ids.add(eid)
    require(
        event.get("video_key") == row["video_key"]
        and event.get("content_sha256") == row["content_sha256"],
        "EVENT_CONTENT_MISMATCH",
    )
    kind = event.get("type_id")
    require(
        kind is None or isinstance(kind, str) and kind in allowed, "EVENT_TYPE_INVALID"
    )
    require(
        text(event.get("action"))
        and text(event.get("object"))
        and in_enum(
            event.get("actor_role"),
            {"cashier", "client", "collector", "other", "unknown"},
        ),
        "EVENT_ACTION_INVALID",
    )
    require(
        in_enum(event.get("visibility"), {"clear", "partial", "unclear"}),
        "EVENT_VISIBILITY_INVALID",
    )
    times = [event.get(key) for key in ("t_start", "t_peak", "t_end")]
    require(
        all(key in event for key in ("t_start", "t_peak", "t_end"))
        and all(
            t is None or number(t) and 0 <= t <= row["duration_sec"] for t in times
        ),
        "EVENT_TIME_INVALID",
    )
    known = [t for t in times if t is not None]
    require(known == sorted(known), "EVENT_TIME_ORDER_INVALID")
    metadata, evidence = (
        event.get("source_metadata"),
        event.get("evidence_timestamp_sec"),
    )
    require(
        "source_metadata" in event and "evidence_timestamp_sec" in event,
        "EVIDENCE_PROVENANCE_INVALID",
    )
    if metadata is not None:
        require(
            isinstance(metadata, dict)
            and metadata.get("kind") == "source_metadata"
            and text(metadata.get("reference"))
            and number(metadata.get("evidence_timestamp_sec"))
            and 0 <= metadata["evidence_timestamp_sec"] <= row["duration_sec"],
            "EVIDENCE_PROVENANCE_INVALID",
        )
    require(
        evidence is None
        or number(evidence)
        and metadata is not None
        and evidence == metadata["evidence_timestamp_sec"],
        "EVIDENCE_TIME_UNSUPPORTED",
    )
    require("primary" in event and "secondary" in event, "EVENT_REVIEW_FIELDS_REQUIRED")
    primary, secondary = event["primary"], event["secondary"]
    first, second = pair(primary, secondary, True)
    consensus = event.get("consensus")
    require(in_enum(consensus, CONSENSUS), "EVENT_CONSENSUS_INVALID")
    disagreement = event.get("disagreement")
    require(
        "disagreement" in event and (disagreement is None or text(disagreement)),
        "EVENT_DISAGREEMENT_INVALID",
    )
    choices = [r["decision"] for r in (primary, secondary) if r is not None]
    positive = "confirmed" in choices or consensus == "confirmed"
    unclear = (
        "unclear" in choices
        or consensus == "unclear"
        or event["visibility"] == "unclear"
    )
    disputed = (
        len(set(choices)) > 1 or consensus == "disputed" or disagreement is not None
    )
    needs_second = positive or unclear or disputed
    listed = disagreement is not None or bool(clip_disagreements)
    coherent = bool(choices) and (
        consensus in {"unclear", "disputed"}
        and listed
        or consensus in DECISIONS
        and (
            all(v == consensus for v in choices)
            or consensus in choices
            and second
            and listed
        )
    )
    ready = (
        first and (not needs_second or second) and consensus != "pending" and coherent
    )
    ready = ready and (not (unclear or disputed) or listed)
    full = (
        ready
        and consensus in {"confirmed", "rejected"}
        and not unclear
        and not disputed
    )
    full = full and event["visibility"] == "clear" and all(t is not None for t in times)
    return {
        "ready": ready,
        "full": full,
        "positive": positive,
        "unclear": unclear,
        "disputed": disputed,
        "missing_times": any(t is None for t in times),
        "disagreement": disagreement is not None,
    }


def validate_documents(inventory, adjudication, inventory_sha256, type_ids=None):
    rows, duplicate_groups = inventory_rows(inventory)
    require(digest(inventory_sha256), "INVENTORY_SHA_INVALID")
    require(
        isinstance(adjudication, dict)
        and type(adjudication.get("schema_version")) is int
        and adjudication["schema_version"] == 1
        and adjudication.get("kind") == "pilot_review_adjudication",
        "ADJUDICATION_SCHEMA_INVALID",
    )
    require(
        adjudication.get("inventory_sha256") == inventory_sha256,
        "ADJUDICATION_INVENTORY_SHA_MISMATCH",
    )
    clips = adjudication.get("clips")
    require(isinstance(clips, list) and len(clips) == 8, "ADJUDICATION_CLIPS_INVALID")
    allowed = allowed_ids(type_ids)
    seen, event_ids, group_events = set(), set(), defaultdict(set)
    counts = dict(
        clips=8,
        primary_reviewed_clips=0,
        secondary_reviewed_clips=0,
        events=0,
        positive_events=0,
        unclear_events=0,
        disputed_events=0,
        duplicate_groups=len(duplicate_groups),
        agreed_duplicate_groups=0,
        unresolved_disagreements=0,
        missing_event_times=0,
        known_angle_clips=0,
        known_source_group_clips=0,
    )
    handoff, full = True, True
    final_positive, explicit_negative = defaultdict(set), defaultdict(set)
    for clip in clips:
        require(
            isinstance(clip, dict)
            and text(clip.get("video_key"))
            and clip.get("video_key") in rows
            and clip["video_key"] not in seen,
            "ADJUDICATION_CLIP_INVALID",
        )
        key = clip["video_key"]
        seen.add(key)
        row = rows[key]
        require(
            clip.get("content_sha256") == row["content_sha256"],
            "ADJUDICATION_CONTENT_MISMATCH",
        )
        clarification = clip.get("organizer_clarification")
        if clarification is not None:
            require(
                isinstance(clarification, dict)
                and clarification.get("reference") == "inventory.organizer"
                and clarification.get("source_sha256")
                == row["organizer"]["source_sha256"]
                and text(clarification.get("note")),
                "ORGANIZER_CLARIFICATION_INVALID",
            )
        require(
            in_enum(clip.get("coverage"), {"pending", "primary_only", "full_clip"}),
            "COVERAGE_INVALID",
        )
        require(
            in_enum(
                clip.get("clip_decision"),
                {"pending", "has_confirmed_event", "no_violation", "unclear"},
            ),
            "CLIP_DECISION_INVALID",
        )
        if clip["clip_decision"] == "no_violation":
            require(clip["coverage"] == "full_clip", "NO_VIOLATION_REQUIRES_FULL_CLIP")
        require(
            "primary_review" in clip and "secondary_review" in clip,
            "CLIP_REVIEW_FIELDS_REQUIRED",
        )
        first, second = pair(clip["primary_review"], clip["secondary_review"])
        counts["primary_reviewed_clips"] += first
        counts["secondary_reviewed_clips"] += second
        disagreements, events = clip.get("disagreements"), clip.get("events")
        require(
            isinstance(disagreements, list) and all(text(v) for v in disagreements),
            "CLIP_DISAGREEMENTS_INVALID",
        )
        require(isinstance(events, list), "CLIP_EVENTS_INVALID")
        counts["unresolved_disagreements"] += len(disagreements)
        state = []
        for event in events:
            item = event_state(event, row, allowed, event_ids, disagreements)
            state.append(item)
            group_events[row["content_sha256"]].add((key, event["event_id"]))
            counts["events"] += 1
            for dest, source in [
                ("positive_events", "positive"),
                ("unclear_events", "unclear"),
                ("disputed_events", "disputed"),
                ("missing_event_times", "missing_times"),
                ("unresolved_disagreements", "disagreement"),
            ]:
                counts[dest] += bool(item[source])
        event_disputes_covered = bool(state) and any(
            s["unclear"] or s["disputed"] for s in state
        )
        event_disputes_covered = event_disputes_covered and all(
            s["ready"] for s in state
        )
        needs_second = (
            clip["clip_decision"] == "unclear"
            or bool(disagreements)
            and not event_disputes_covered
            or row["organizer"]["positive"]
            and clip["clip_decision"] == "no_violation"
        )
        ready = (
            first
            and clip["coverage"] != "pending"
            and clip["clip_decision"] != "pending"
        )
        ready = (
            ready and (not needs_second or second) and all(s["ready"] for s in state)
        )
        if clip["clip_decision"] == "has_confirmed_event":
            ready = ready and any(s["positive"] for s in state)
        if clip["clip_decision"] == "no_violation":
            ready = ready and not any(e["consensus"] == "confirmed" for e in events)
            explicit_negative[row["content_sha256"]].add(key)
        if any(e["consensus"] == "confirmed" for e in events):
            final_positive[row["content_sha256"]].add(key)
        if clip["clip_decision"] == "unclear":
            ready = ready and bool(disagreements)
        source_conflict = (
            row["organizer"]["positive"]
            and clip["clip_decision"] == "no_violation"
            or not row["organizer"]["positive"]
            and (
                clip["clip_decision"] == "has_confirmed_event"
                or any(e["consensus"] == "confirmed" for e in events)
            )
        )
        if source_conflict:
            ready = ready and clarification is not None
        handoff = handoff and ready
        full = (
            full
            and ready
            and clip["coverage"] == "full_clip"
            and clip["clip_decision"] != "unclear"
        )
        full = full and not disagreements and all(s["full"] for s in state)
        current = effective_context(row, clip)
        for field, dest in [
            ("angle_id", "known_angle_clips"),
            ("source_group", "known_source_group_clips"),
        ]:
            value = current[field]
            counts[dest] += value is not None and value.strip().casefold() not in {
                "unknown",
                "unassigned",
                "неизвестно",
            }
    require(seen == set(rows), "ADJUDICATION_CLIP_SET_MISMATCH")
    decisions = adjudication.get("duplicate_decisions")
    require(
        isinstance(decisions, list) and len(decisions) == len(duplicate_groups),
        "DUPLICATE_DECISIONS_REQUIRED",
    )
    seen_groups = set()
    for decision in decisions:
        require(isinstance(decision, dict), "DUPLICATE_DECISION_INVALID")
        sha = decision.get("content_sha256")
        require(
            digest(sha) and sha in duplicate_groups and sha not in seen_groups,
            "DUPLICATE_GROUP_INVALID",
        )
        seen_groups.add(sha)
        keys = decision.get("video_keys")
        require(
            isinstance(keys, list)
            and all(text(key) for key in keys)
            and len(keys) == len(set(keys))
            and set(keys) == set(duplicate_groups[sha]),
            "DUPLICATE_VIDEO_KEYS_INVALID",
        )
        links = decision.get("event_links")
        require(
            isinstance(links, list)
            and all(
                isinstance(v, dict)
                and text(v.get("video_key"))
                and text(v.get("event_id"))
                for v in links
            ),
            "DUPLICATE_EVENT_LINKS_INVALID",
        )
        actual = [(v["video_key"], v["event_id"]) for v in links]
        require(
            len(actual) == len(set(actual)) and set(actual) <= group_events[sha],
            "DUPLICATE_EVENT_LINKS_INVALID",
        )
        consensus = decision.get("consensus")
        require(
            in_enum(consensus, DECISIONS | {"pending"}), "DUPLICATE_CONSENSUS_INVALID"
        )
        reviewers = decision.get("reviewers")
        require(isinstance(reviewers, list), "DUPLICATE_REVIEWERS_INVALID")
        identities = [human(v) for v in reviewers]
        require(len(identities) == len(set(identities)), "REVIEWERS_NOT_DISTINCT")
        note = decision.get("note")
        require(
            "note" in decision and (note is None or text(note)),
            "DUPLICATE_NOTE_INVALID",
        )
        agreed = (
            consensus != "pending"
            and len(identities) >= 2
            and text(note)
            and set(actual) == group_events[sha]
        )
        counts["agreed_duplicate_groups"] += bool(agreed)
        handoff = handoff and bool(agreed)
        full = full and bool(agreed) and consensus != "unclear"
        contradiction = any(
            p != n for p in final_positive[sha] for n in explicit_negative[sha]
        )
        if contradiction:
            handoff = handoff and consensus == "unclear"
            full = False
    return {
        "structurally_valid": True,
        "review_handoff_ready": bool(handoff),
        "complete": bool(handoff),
        "full_truth_ready": bool(full),
        **counts,
    }
