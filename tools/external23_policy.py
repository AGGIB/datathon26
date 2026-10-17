"""Explicit external KZ-bank source policy; metadata does not prove licence facts."""
from datetime import datetime
import re

from external23_common import active_path, https, require, sha256, text, value_sha256

LICENSES = {
    'CC0-1.0': 'https://creativecommons.org/publicdomain/zero/1.0/',
    'CC-BY-4.0': 'https://creativecommons.org/licenses/by/4.0/',
    'PDM-1.0': 'https://creativecommons.org/publicdomain/mark/1.0/',
    'PD-US-GOV': 'https://www.copyright.gov/title17/92chap1.html#105',
}
UNKNOWN = {'unknown', 'unconfirmed', 'pending', 'none', 'null', 'unassigned', 'неизвестно'}
CLIP_REVIEW_FIELDS = {'angle_id', 'camera_id', 'bank_scene_confirmed',
                      'bank_scene_evidence_reference', 'rights_proof'}


def known(value):
    return (text(value) and value == value.strip() and value.casefold() not in UNKNOWN
            and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.:-]{0,79}', value) is not None)


def license_allowed(source):
    info = source.get('license', {})
    return isinstance(info, dict) and info.get('id') in LICENSES and info.get('url') == LICENSES[info['id']]


def proof_shape(proof):
    require(isinstance(proof, dict) and text(proof.get('path'))
            and isinstance(proof.get('sha256'), str)
            and re.fullmatch('[0-9a-f]{64}', proof['sha256']), 'RIGHTS_PROOF_REQUIRED')
    https(proof.get('url'))


def verify_proof(root, proof):
    proof_shape(proof)
    path = active_path(root, proof['path'])
    require(path.is_file() and sha256(path) == proof['sha256'], 'RIGHTS_PROOF_SHA256_MISMATCH')


def source_usable(source):
    require(license_allowed(source), 'LICENSE_NOT_ALLOWED')
    context = source.get('source_context')
    require(isinstance(context, dict) and context.get('country') == 'KZ', 'SOURCE_OUTSIDE_KZ_SCOPE')


def selected_clip(source, clip):
    source_usable(source)
    require(clip.get('bank_scene_confirmed') is True
            and text(clip.get('bank_scene_evidence_reference')), 'CLIP_BANK_SCENE_NOT_CONFIRMED')
    require(known(clip.get('angle_id')) and known(clip.get('camera_id', source.get('camera_id'))),
            'SELECTED_VIEWPOINT_UNKNOWN')
    proof_shape(clip.get('rights_proof'))


def selected_record(record):
    require(record.get('selection') == 'selected', 'ANNOTATION_NOT_SELECTED')
    require(record.get('split') == 'external_holdout', 'EXTERNAL_HOLDOUT_ONLY')
    selected_clip(record, record)


def journal_changes(previous, current):
    """Keep content identity immutable; context edits require an explicit journal."""
    changes = []
    journal = current.get('change_log', [])
    require(isinstance(journal, list), 'MANIFEST_CHANGE_LOG_INVALID')
    history = previous.get('change_log', [])
    require(isinstance(history, list) and journal[:len(history)] == history, 'MANIFEST_CHANGE_LOG_NOT_APPEND_ONLY')
    for collection in ('sources', 'clips'):
        rows = {r['id']: r for r in current[collection]}
        for old in previous.get(collection, []):
            require(old['id'] in rows, 'MANIFEST_REMOVED_EXISTING_RECORD')
            new = rows[old['id']]
            mutable = {'source_context'} if collection == 'sources' else CLIP_REVIEW_FIELDS | {
                'selection', 'selection_reason', 'split'}
            require({k: v for k, v in old.items() if k not in mutable}
                    == {k: v for k, v in new.items() if k not in mutable},
                    'MANIFEST_CHANGED_CONTENT_IDENTITY')
            for field in mutable:
                if old.get(field) == new.get(field):
                    continue
                change = {'entity': collection, 'id': old['id'], 'field': field,
                          'before_sha256': value_sha256(old.get(field)),
                          'after_sha256': value_sha256(new.get(field))}
                if field not in {'selection', 'selection_reason', 'split'}:
                    matches = [entry for entry in journal[len(history):] if isinstance(entry, dict)
                               and all(entry.get(k) == v for k, v in change.items())]
                    require(bool(matches), 'CONTEXT_CHANGE_NOT_JOURNALED')
                    entry = matches[-1]
                    require(text(entry.get('changed_by')) and text(entry.get('reason'))
                            and text(entry.get('changed_at')), 'CONTEXT_CHANGE_LOG_INVALID')
                    try:
                        datetime.fromisoformat(entry['changed_at'].replace('Z', '+00:00'))
                    except ValueError as error:
                        raise ValueError('CONTEXT_CHANGE_LOG_INVALID') from error
                    change.update({k: entry[k] for k in ('changed_by', 'changed_at', 'reason')})
                changes.append(change)
    return changes
