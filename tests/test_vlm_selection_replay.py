"""Scripted CPU selection fixtures; no model/media/real annotations."""
from dataclasses import asdict, replace
import json

import pytest

from ml.assemble import select
from ml.candidates import Candidate
from ml.kb import catalog
from ml.tools.vlm_selection_replay import flat, full_inventory, recover_findings, replay_candidates
from ml.vlm import discovery, type_runtime
from ml.vlm.action_backend import _ACTIONS
from ml.vlm.discovery_policy import plan_windows

FOOD = 'food_or_drink_at_workplace'
COAT = 'personal_items_on_desk'


def finding(kind=FOOD, confidence=.95):
    window = plan_windows(20)[0]
    start, peak, end = window.timestamps[2:5]
    spec = next(spec for spec in _ACTIONS.values() if spec.type_id == kind)
    phases = {end: spec.phases[2], start: spec.phases[0], peak: spec.phases[1]}
    return {'type_id': kind, 't_start': start, 't_peak': peak, 't_end': end,
            'confidence': confidence, 'insufficient_evidence': False,
            'observations': [{'timestamp_s': t, 'text': phases[t]} for t in sorted(phases)]}


def screen(values):
    accepted = [discovery._candidate(value) for value in values if value['confidence'] >= .95]
    return {'duration_sec': 20, 'report': {'status': 'completed', 'timebase': 'cv_index_fps',
        'retained_candidates': [], 'requested_timestamps': list(plan_windows(20)[0].timestamps),
        'decisions': [{'window': 0, 'status': 'added' if value['confidence'] >= .95 else 'uncertain',
            'reason': 'visible_evidence' if value['confidence'] >= .95 else 'insufficient_evidence',
            'type_id': value['type_id'], 'confidence': value['confidence'],
            'observations': value['observations']} for value in values],
        'added_candidates': json.loads(json.dumps([asdict(c) for c in accepted]))}}


def cv(kind='safe_left_open', peak=1, confidence=.6):
    return Candidate(kind, peak, peak + .5, peak, confidence,
                     'Synthetic visible state', (peak,), ((1., 2., 3., 4.),))


def test_recovers_full_candidate_and_reuses_exact_source_anchors():
    value = finding()
    recovered, counts = recover_findings(screen([value]), catalog(), .95)
    assert recovered[plan_windows(20)[0].timestamps] == [value]
    assert counts == {'visible_evidence': 1, 'saved_full_candidates_exact_recovered': 1}


def test_low_confidence_is_replayed_but_never_called_semantic_negative():
    value = finding(confidence=.9)
    recovered, counts = recover_findings(screen([value]), catalog(), .95)
    selected, checks = replay_candidates([], recovered, catalog(), 20, .95)
    assert selected == [] and checks['scripted_postverification_calls'] == 1
    assert counts['insufficient_evidence'] == 1
    assert not any('negative' in key or 'reject' in key for key in counts)


def test_untyped_unknown_has_no_fabricated_finding_or_negative():
    item = screen([])
    item['report']['decisions'] = [{'window': 0, 'status': 'uncertain',
        'reason': 'invalid_response', 'type_id': None, 'confidence': None, 'observations': []}]
    recovered, counts = recover_findings(item, catalog(), .95)
    assert not any(recovered.values()) and counts['untyped_or_uncertain_decisions'] == 1


@pytest.mark.parametrize('field,new', [('confidence', .951), ('evidence_times', [0]),
    ('observation', 'Different synthetic observation'), ('highlight_boxes', [[1, 2, 3, 4]])])
def test_recovery_rejects_full_candidate_field_tampering(field, new):
    item = screen([finding()])
    item['report']['added_candidates'][0][field] = new
    with pytest.raises(ValueError, match='full_saved_candidate'):
        recover_findings(item, catalog(), .95)


def test_owner_outcome_and_exact_phase_text_are_checked():
    item = screen([finding()])
    item['report']['decisions'][0]['reason'] = 'overlaps_existing'
    with pytest.raises(ValueError, match='owner_outcome'):
        recover_findings(item, catalog(), .95)
    item = screen([finding()])
    item['report']['decisions'][0]['observations'][0]['text'] = 'Unrecoverable synthetic phase'
    with pytest.raises(ValueError, match='phase_recovery'):
        recover_findings(item, catalog(), .95)


def test_actual_discovery_overlap_retains_original_object_and_every_field():
    value = finding(COAT)
    original = Candidate(COAT, value['t_start'], value['t_end'], value['t_peak'], .6,
                         'Existing synthetic CV state', (value['t_peak'], value['t_start']), ((1, 2, 3, 4),))
    recovered, _ = recover_findings(screen([value]), catalog(), .95)
    selected, checks = replay_candidates([original], recovered, catalog(), 20, .95)
    assert selected == [original] and selected[0] is original
    assert checks['selected_cv_full_fields_exactly_retained']
    assert full_inventory(selected) == full_inventory([original])


def test_actual_merge_gap_suppresses_nonoverlapping_same_type_before_assembly():
    value = finding(COAT)
    original = Candidate(COAT, 0, value['t_start'] - .5, 0, .6, 'Synthetic existing CV', (0,))
    assert not discovery._overlaps(discovery._candidate(value), [original])
    recovered, _ = recover_findings(screen([value]), catalog(), .95)
    selected, _ = replay_candidates([original], recovered, catalog(), 20, .95)
    assert selected == [original]  # Without the owner gap gate assembly rewrites this interval.


def test_five_reserved_cv_slots_skip_even_higher_confidence_proposal():
    originals = [cv(peak=i * 3) for i in range(5)]
    recovered, _ = recover_findings(screen([finding()]), catalog(), .95)
    selected, checks = replay_candidates(originals, recovered, catalog(), 20, .95)
    assert selected == select(originals, catalog())
    assert checks['scripted_postverification_calls'] == checks['scripted_extract_calls'] == 0


def test_four_reserved_slots_allow_one_addition_without_confidence_rewriting():
    originals = [cv(peak=i * 3, confidence=.501) for i in range(4)]
    value = finding()
    recovered, _ = recover_findings(screen([value]), catalog(), .95)
    selected, checks = replay_candidates(originals, recovered, catalog(), 20, .95)
    assert len(selected) == 5 and all(c in selected for c in originals)
    assert checks['selected_cv_count'] == 4
    assert flat(selected, 20)['violations'][0]['conf'] == .501


def test_inventory_is_sensitive_to_evidence_order_boxes_confidence_and_observation():
    original = replace(cv(), evidence_times=(1, 0))
    for changed in (replace(original, confidence=.601), replace(original, observation='Other synthetic text'),
                    replace(original, evidence_times=(1, .5)), replace(original, highlight_boxes=())):
        assert full_inventory([original]) != full_inventory([changed])


def test_replay_restores_runtime_configuration_even_on_inference_failure(monkeypatch):
    marker = type_runtime._configuration
    recovered, _ = recover_findings(screen([finding()]), catalog(), .95)
    def fail(*args, **kwargs):
        raise RuntimeError('synthetic selection failure')
    monkeypatch.setattr(type_runtime, 'augment_types', fail)
    with pytest.raises(RuntimeError):
        replay_candidates([], recovered, catalog(), 20, .95)
    assert type_runtime._configuration is marker


def test_temporal_relocation_uses_returned_sampled_witnesses_not_cached_v4_times():
    value = finding(COAT)
    window = plan_windows(20)[0]
    shifted = window.timestamps[6:9]
    spec = next(spec for spec in _ACTIONS.values() if spec.type_id == COAT)
    relocated = dict(value, t_start=shifted[0], t_peak=shifted[1], t_end=shifted[2],
                     observations=[{'timestamp_s': t, 'text': phase} for t, phase in zip(shifted, spec.phases)])
    recovered, _ = recover_findings(screen([relocated]), catalog(), .95)
    selected, _ = replay_candidates([], recovered, catalog(), 20, .95)
    assert selected[0].t_peak == shifted[1] != value['t_peak']
    assert selected[0].evidence_times == (shifted[1], shifted[0], shifted[2])


def test_repeated_peak_end_anchor_is_recovered_without_fabricating_third_witness():
    value = finding()
    spec = next(spec for spec in _ACTIONS.values() if spec.type_id == FOOD)
    value['t_end'] = value['t_peak']
    value['observations'] = [value['observations'][0],
                             {'timestamp_s': value['t_peak'], 'text': spec.phases[1]}]
    recovered, _ = recover_findings(screen([value]), catalog(), .95)
    selected, _ = replay_candidates([], recovered, catalog(), 20, .95)
    assert len(selected[0].evidence_times) == 2
    assert selected[0].t_peak == selected[0].t_end
