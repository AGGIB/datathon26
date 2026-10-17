import hashlib

import pytest

from ml.eval import score
from ml.eval_runner import _validated_result


def test_saved_actual_assembly_keeps_machine_type_and_confidence_but_not_images():
    from ml.assemble import assemble
    from ml.candidates import Candidate
    from ml.kb import catalog

    candidate = Candidate('safe_left_open', 1, 9, 3, 0.7, 'дверца остаётся открытой', (3,))
    result = assemble([candidate], lambda selected: {
        item: [{'timestamp_sec': 3.0, 'image_base64': 'PRIVATE_IMAGE'}] for item in selected
    }, types=catalog(), duration=10)
    saved = _validated_result(result)
    event = saved['violations'][0]
    assert event['type_id'] == 'safe_left_open'
    assert event['type'] != event['type_id']
    assert event['confidence'] == 0.7
    assert 'evidence_frames' not in event and 'PRIVATE_IMAGE' not in str(saved)


@pytest.mark.parametrize('field,value', [('type_id', 123), ('confidence', True),
                                       ('confidence', float('nan')), ('confidence', 1.01)])
def test_saved_extension_fields_must_be_valid(field, value):
    event = {'type': 'display', 'type_id': 'safe_left_open', 'timecode_sec': 3.0, field: value}
    with pytest.raises(ValueError):
        _validated_result({'verdict': 'violation', 'violations': [event]})


@pytest.mark.parametrize('reverse', [False, True])
def test_duplicate_runs_with_jitter_or_different_verdicts_keep_a_group_report(reverse):
    labels = [{'video': 'first', 'positive': True, 'timestamp': 10.0},
              {'video': 'second', 'positive': True, 'timestamp': 20.0}]
    predictions = [
        {'video': 'first', 'status': 'done', 'verdict': 'no_violation', 'violations': []},
        {'video': 'second', 'status': 'done', 'verdict': 'violation',
         'violations': [{'type_id': 'fixture', 'timecode_sec': 20.4}]},
    ]
    if reverse:
        labels.reverse()
        predictions.reverse()
    same = hashlib.sha256(b'same').hexdigest()
    report = score(labels, predictions, content_hashes={row['video']: same for row in labels})
    assert report['overall']['total'] == 1
    assert report['overall']['confusion']['tp'] == 1
    assert report['overall']['localization']['matched_events'] == 1
    assert report['per_content_group'][0]['member_predictions_differ'] is True


def test_duplicate_successful_timestamp_jitter_is_not_a_metadata_error():
    labels = [{'video': name, 'positive': True, 'timestamp': 57.0} for name in ('X-4', 'X-47')]
    predictions = [{'video': name, 'status': 'done', 'verdict': 'violation',
                    'violations': [{'type_id': 'fixture', 'timecode_sec': peak}]}
                   for name, peak in [('X-4', 57.0), ('X-47', 57.4)]]
    same = hashlib.sha256(b'same').hexdigest()
    report = score(labels, predictions, content_hashes={row['video']: same for row in labels})
    assert report['overall']['total'] == 1
    assert report['overall']['localization']['within_tolerance'] == 1
    assert report['per_content_group'][0]['member_predictions_differ'] is True
