"""Structural checks for private, type-scoped human review drafts.

Recorded reviewer identities are checked for distinction, not actual viewing or
independence. This schema never creates truth, negatives, dates or reviewers.
"""
from collections import Counter
import re
from uuid import UUID

from tools.pilot_review_schema import digest, human, number, require

FRONTIER = {
    'key_left_in_safe_lock': 'key_remains_in_safe_lock_after_departure',
    'personal_items_on_desk': 'cashier_places_removed_outerwear_on_cashdesk',
    'food_or_drink_at_workplace': 'food_or_drink_actually_consumed',
}
REVIEW_FIELDS = ['slot_id', 'content_sha256', 'type_id', 'source_path', 'aliases', 'review_status', 'reviewer', 'date',
    'coverage', 'reviewed_start_s', 'reviewed_end_s', 'type_exhaustive', 'assessment', 'unknown_reason']
ADJUDICATION_FIELDS = ['slot_id', 'consensus', 'reviewer', 'date', 'disagreement_code']
ASSESSMENTS = {'events_recorded', 'no_observed_positive', 'unknown'}
UNKNOWN = {'object_unresolved', 'actor_unresolved', 'contact_occluded', 'departure_not_visible',
           'continuity_unresolved', 'timing_unresolved', 'other_unknown'}
EPISODE_FIELDS = {'episode_id', 'slot_id', 'outcome', 't_start', 't_peak', 't_end', 'actor_role',
    'actor_track_id', 'object_track_id', 'object_visibility', 'contact_visibility',
    'departure_visibility', 'same_actor', 'same_object', 'unknown_reason', 'contrast_reason'}


def slots_for(manifest):
    require(isinstance(manifest, dict) and type(manifest.get('schema_version')) is int and manifest['schema_version'] == 1
            and manifest.get('kind') == 'frontier_review_manifest'
            and manifest.get('usage') == 'development'
            and manifest.get('frontier') == FRONTIER
            and isinstance(manifest.get('source_revision'), str)
            and re.fullmatch('[0-9a-f]{40}', manifest['source_revision'])
            and digest(manifest.get('scaffold_sha256')) and digest(manifest.get('schema_sha256')), 'MANIFEST_SCHEMA')
    contents = manifest.get('contents')
    require(isinstance(contents, list) and contents, 'MANIFEST_CONTENTS')
    slots, aliases, hashes = {}, set(), set()
    for source in contents:
        require(isinstance(source, dict), 'MANIFEST_CONTENT')
        content = source.get('content_sha256')
        require(digest(content) and content not in hashes and number(source.get('duration_sec'))
                and source['duration_sec'] > 0, 'MANIFEST_CONTENT')
        hashes.add(content)
        require(all(source.get(key) is None for key in ('camera_id', 'angle_id', 'session_id', 'source_group')),
                'SCAFFOLD_CONTEXT_MUST_REMAIN_UNKNOWN')
        require(source.get('aliases') and isinstance(source['aliases'], list), 'MANIFEST_ALIASES')
        for alias in source['aliases']:
            require(isinstance(alias, dict), 'MANIFEST_ALIAS_IDENTITY')
            key = alias.get('video_key')
            require(isinstance(key, str) and key and key not in aliases and alias.get('content_sha256') == content,
                    'MANIFEST_ALIAS_IDENTITY')
            aliases.add(key)
        for kind in FRONTIER:
            slot = content + ':' + kind
            slots[slot] = (source, kind)
    return slots


def draft_rows(manifest):
    return [{**dict.fromkeys(REVIEW_FIELDS, ''), 'slot_id': slot, 'content_sha256': source['content_sha256'],
             'type_id': kind, 'source_path': source['aliases'][0]['path'],
             'aliases': ';'.join(alias['video_key'] for alias in source['aliases']), 'review_status': 'unreviewed', 'coverage': 'unreviewed',
             'type_exhaustive': 'false', 'assessment': 'unreviewed'}
            for slot, (source, kind) in slots_for(manifest).items()]


def _rows(values, expected, fields):
    require(isinstance(values, list) and all(isinstance(row, dict) and set(row) == set(fields)
            and all(isinstance(value, str) for value in row.values()) for row in values), 'CSV_ROW_SCHEMA')
    result = {row['slot_id']: row for row in values}
    require(len(result) == len(values) and set(result) == set(expected), 'COVERAGE_SLOT_SET')
    return result


def _timestamp(value):
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError('REVIEW_TIME_INVALID') from error
    require(number(result), 'REVIEW_TIME_INVALID')
    return result


def _review(row, source, kind):
    require(row['content_sha256'] == source['content_sha256'] and row['type_id'] == kind
            and row['source_path'] == source['aliases'][0]['path']
            and row['aliases'] == ';'.join(alias['video_key'] for alias in source['aliases']),
            'REVIEW_CONTENT_IDENTITY')
    require(row['type_exhaustive'] in {'true', 'false'}, 'TYPE_EXHAUSTIVE_BOOLEAN')
    if row['review_status'] == 'unreviewed':
        require(row['coverage'] == row['assessment'] == 'unreviewed' and row['type_exhaustive'] == 'false'
                and not any(row[key] for key in ('reviewer', 'date', 'reviewed_start_s', 'reviewed_end_s', 'unknown_reason')),
                'UNREVIEWED_HAS_CLAIMS')
        return None
    require(row['review_status'] == 'reviewed' and row['coverage'] in {'partial', 'full_clip'}
            and row['assessment'] in ASSESSMENTS, 'REVIEW_STATUS')
    name = human({'kind': 'human', 'reviewer': row['reviewer'], 'date': row['date']})
    start, end = _timestamp(row['reviewed_start_s']), _timestamp(row['reviewed_end_s'])
    require(0 <= start < end <= source['duration_sec'], 'REVIEW_INTERVAL')
    require(row['coverage'] != 'full_clip' or (start == 0 and end == source['duration_sec']), 'FULL_CLIP_INTERVAL')
    require(row['unknown_reason'] in UNKNOWN if row['assessment'] == 'unknown' else row['unknown_reason'] == '',
            'UNKNOWN_REASON')
    require(row['assessment'] != 'unknown' or row['type_exhaustive'] == 'false', 'UNKNOWN_NOT_EXHAUSTIVE')
    require(row['assessment'] != 'no_observed_positive' or row['type_exhaustive'] == 'true',
            'ABSENCE_REQUIRES_EXHAUSTIVE_REVIEW')
    return {'name': name, 'start': start, 'end': end}


def _episodes(values, slots, reviews):
    require(isinstance(values, list), 'EPISODE_LIST')
    indexed, counts = {}, Counter()
    for event in values:
        require(isinstance(event, dict) and set(event) == EPISODE_FIELDS, 'EPISODE_SCHEMA')
        eid = event['episode_id']
        try:
            require(isinstance(eid, str) and str(UUID(eid)) == eid and eid not in indexed, 'EPISODE_ID')
        except (ValueError, TypeError, AttributeError) as error:
            raise ValueError('EPISODE_ID') from error
        slot = event['slot_id']
        require(isinstance(slot, str) and slot in slots and reviews[slot] is not None, 'EPISODE_REVIEW_REQUIRED')
        require(event['outcome'] in {'observed_positive', 'observed_contrast', 'unknown'}, 'EPISODE_OUTCOME')
        times = [event[key] for key in ('t_start', 't_peak', 't_end')]
        known = [time for time in times if time is not None]
        interval = reviews[slot]
        require(all(number(time) and interval['start'] <= time <= interval['end'] for time in known)
                and known == sorted(known), 'EPISODE_INTERVAL')
        require(event['unknown_reason'] in UNKNOWN if event['outcome'] == 'unknown' or len(known) != 3
                else event['unknown_reason'] == '', 'EPISODE_UNKNOWN_REASON')
        require(event['outcome'] != 'observed_positive' or len(known) == 3, 'POSITIVE_TIMES_REQUIRED')
        require(event['actor_role'] in {'cashier', 'client', 'other', 'unknown'}
                and all(isinstance(event[key], str) and len(event[key]) <= 120
                        for key in ('actor_track_id', 'object_track_id')), 'EPISODE_TRACKS')
        require(all(event[key] in {'clear', 'partial', 'unknown'} for key in
                    ('object_visibility', 'contact_visibility', 'departure_visibility'))
                and all(event[key] in {'yes', 'no', 'unknown'} for key in ('same_actor', 'same_object')),
                'EPISODE_VISIBILITY')
        require(event['contrast_reason'] in {'different_object', 'different_actor', 'different_location',
                'no_mouth_contact', 'key_removed_before_departure', 'other_observed_contrast'}
                if event['outcome'] == 'observed_contrast' else event['contrast_reason'] == '', 'CONTRAST_REASON')
        if event['outcome'] == 'observed_positive':
            kind = slots[slot][1]
            require(event['actor_role'] == 'cashier' and event['object_visibility'] == 'clear', 'POSITIVE_VISIBLE_FACTS')
            required = 'departure_visibility' if kind == 'key_left_in_safe_lock' else 'contact_visibility'
            require(event[required] == 'clear', 'POSITIVE_VISIBLE_FACTS')
            if kind != 'food_or_drink_at_workplace':
                require(event['same_actor'] == event['same_object'] == 'yes'
                        and event['actor_track_id'] and event['object_track_id'] and times[0] < times[1],
                        'POSITIVE_CONTINUITY')
        indexed[eid] = event
        counts[slot] += 1
    return indexed, counts


def validate_documents(manifest, review_a, episodes_a, review_b, episodes_b, adjudication, final_episodes=None):
    slots = slots_for(manifest)
    a, b = (_rows(rows, slots, REVIEW_FIELDS) for rows in (review_a, review_b))
    reviewed_a = {slot: _review(a[slot], *values) for slot, values in slots.items()}
    reviewed_b = {slot: _review(b[slot], *values) for slot, values in slots.items()}
    events_a, counts_a = _episodes(episodes_a, slots, reviewed_a)
    events_b, counts_b = _episodes(episodes_b, slots, reviewed_b)
    for rows, events, counts in ((a, events_a, counts_a), (b, events_b, counts_b)):
        for slot, row in rows.items():
            require(row['assessment'] != 'events_recorded' or counts[slot] > 0, 'RECORDED_EPISODES_REQUIRED')
            require(row['assessment'] != 'no_observed_positive' or not any(event['slot_id'] == slot
                    and event['outcome'] == 'observed_positive' for event in events.values()), 'ABSENCE_CONTRADICTS_EPISODE')
    adjudicated, decisions, final_reviews = 0, Counter(), {}
    final_episodes = [] if final_episodes is None else final_episodes
    for slot, row in _rows(adjudication, slots, ADJUDICATION_FIELDS).items():
        left, right = reviewed_a[slot], reviewed_b[slot]
        require(not left or not right or left['name'] != right['name'], 'REVIEWERS_NOT_DISTINCT')
        if row['consensus'] == 'pending':
            require(not row['reviewer'] and not row['date'] and not row['disagreement_code'], 'PENDING_HAS_SIGNATURE')
            continue
        require(left is not None and right is not None, 'TWO_REVIEWS_REQUIRED')
        human({'kind': 'human', 'reviewer': row['reviewer'], 'date': row['date']})
        require(row['consensus'] in ASSESSMENTS | {'disputed'}, 'CONSENSUS_ENUM')
        require(row['disagreement_code'] in {'none', 'outcome_difference', 'timing_difference',
                'episode_inventory_difference', 'resolved_after_review', 'unresolved'}, 'DISAGREEMENT_CODE')
        require(row['consensus'] != 'no_observed_positive'
                or a[slot]['type_exhaustive'] == b[slot]['type_exhaustive'] == 'true',
                'CONSENSUS_ABSENCE_REQUIRES_BOTH_EXHAUSTIVE')
        require(row['consensus'] != 'disputed' or row['disagreement_code'] == 'unresolved', 'DISPUTE_REASON')
        require(row['disagreement_code'] != 'unresolved' or row['consensus'] in {'unknown', 'disputed'},
                'UNRESOLVED_IS_NOT_KNOWN')
        require(a[slot]['assessment'] == b[slot]['assessment'] == row['consensus']
                or row['disagreement_code'] != 'none', 'DISAGREEMENT_MUST_BE_RECORDED')
        if row['consensus'] != 'disputed' and max(left['start'], right['start']) < min(left['end'], right['end']):
            final_reviews[slot] = {'start': max(left['start'], right['start']), 'end': min(left['end'], right['end'])}
        adjudicated += 1
        decisions[row['consensus']] += 1
    require(isinstance(final_episodes, list), 'FINAL_EPISODE_LIST')
    base_events = []
    refs = {'review_a_episode_ids': events_a, 'review_b_episode_ids': events_b}
    for event in final_episodes:
        require(isinstance(event, dict) and set(event) == EPISODE_FIELDS | set(refs), 'FINAL_EPISODE_SCHEMA')
        require(event['slot_id'] in final_reviews, 'FINAL_ADJUDICATION_REQUIRED')
        linked = []
        for field, originals in refs.items():
            ids = event[field]
            require(isinstance(ids, list) and all(isinstance(eid, str) and eid in originals
                    and originals[eid]['slot_id'] == event['slot_id'] for eid in ids)
                    and len(ids) == len(set(ids)), 'FINAL_EPISODE_PROVENANCE')
            linked += ids
        require(linked, 'FINAL_EPISODE_PROVENANCE')
        base_events.append({key: event[key] for key in EPISODE_FIELDS})
    final_events, final_counts = _episodes(base_events, slots, {slot: final_reviews.get(slot) for slot in slots})
    adjudication_by_slot = {row['slot_id']: row for row in adjudication}
    for event in final_events.values():
        require(adjudication_by_slot[event['slot_id']]['consensus'] != 'no_observed_positive'
                or event['outcome'] != 'observed_positive', 'FINAL_ABSENCE_CONTRADICTS_EPISODE')
        require(adjudication_by_slot[event['slot_id']]['consensus'] != 'unknown'
                or event['outcome'] == 'unknown', 'FINAL_UNKNOWN_CONTRADICTS_EPISODE')
    episode_handoff = all(row['consensus'] != 'events_recorded' or final_counts[slot] > 0
                          for slot, row in adjudication_by_slot.items())
    return {'structurally_valid': True, 'contents': len(manifest['contents']),
        'files': sum(len(row['aliases']) for row in manifest['contents']), 'coverage_slots': len(slots), 'frontier_types': len(FRONTIER),
        'required_independent_review_slots': 2 * len(slots),
        'review_a_completed_slots': sum(value is not None for value in reviewed_a.values()),
        'review_b_completed_slots': sum(value is not None for value in reviewed_b.values()),
        'adjudicated_slots': adjudicated, 'adjudication_counts': dict(decisions),
        'episode_records': len(episodes_a) + len(episodes_b),
        'adjudicated_episode_records': len(final_episodes),
        'pending_adjudication_slots': len(slots) - adjudicated,
        'human_handoff_ready': adjudicated == len(slots) and len(final_reviews) == len(slots)
                                 and decisions['disputed'] == 0 and episode_handoff, 'evaluation_truth_certified': False,
        'new_camera_holdout': False, 'precision_measured': False}
