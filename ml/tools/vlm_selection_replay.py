"""Replay saved post-verification findings through the real CPU owner selection.

Only explicitly trusted, local pipeline pickle caches may be loaded. This is a
conditional selection check, never inference, decoding, evidence or SLA testing.
Run this file directly with --source-root to inspect another frozen checkout.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import importlib.abc
import json
import logging
from pathlib import Path
import pickle
import subprocess
import sys


class NoModelRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'mlx', 'mlx_vlm', 'tensorflow', 'jax'}:
            raise RuntimeError('model_runtime_forbidden_in_cpu_replay')
        return None


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def full_inventory(candidates):
    """Include confidence, all times, observation, evidence order and boxes."""
    return Counter(canonical(asdict(c)) for c in candidates)


def flat(candidates, duration):
    from ml.assemble import clamp_time
    events = [{'type_id': c.type_id, 't': clamp_time(c.t_peak, duration), 'conf': c.confidence}
              for c in candidates]
    return {'verdict': 'violation' if events else 'no_violation', 'violations': events}


def recover_findings(item, types, threshold):
    """Recover only actually returned typed findings, including low confidence.

    No returned finding is not a semantic negative. Unknown/malformed outcomes
    remain diagnostic counts, and are not reconstructed from cached proposals.
    """
    from ml.vlm import discovery
    from ml.vlm.action_backend import _ACTIONS
    from ml.vlm.contracts import Frame
    from ml.vlm.discovery_policy import TYPE_IDS, plan_windows

    report = item['report']
    if (report['status'] != 'completed' or report['retained_candidates'] != []
            or report.get('timebase') != 'cv_index_fps'):
        raise ValueError('completed_empty_original_screen_required')
    windows = plan_windows(item['duration_sec'])
    if report['requested_timestamps'] != sorted({t for w in windows for t in w.timestamps}):
        raise ValueError('source_timestamps_mismatch')
    eligible, unavailable = discovery._catalog_types(types, TYPE_IDS)
    if unavailable:
        raise ValueError('catalog_frontier_changed')
    specs = {spec.type_id: spec for spec in _ACTIONS.values()}
    recovered = {window.timestamps: [] for window in windows}
    accepted, counts = [], Counter()
    for decision in report['decisions']:
        index = decision['window']
        if type(index) is not int or not 0 <= index < len(windows):
            raise ValueError('window_identity')
        if decision.get('type_id') is None:
            if decision.get('confidence') is not None or decision.get('observations'):
                raise ValueError('untyped_decision_values')
            counts['untyped_or_uncertain_decisions'] += 1
            continue
        kind, confidence, observations = decision['type_id'], decision['confidence'], decision['observations']
        if kind not in specs or kind not in eligible:
            raise ValueError('unrecognized_action_type')
        spec = specs[kind]
        times = [row['timestamp_s'] for row in observations]
        peaks = [row['timestamp_s'] for row in observations if row['text'] == spec.phases[1]]
        if not times or times != sorted(set(times)) or len(peaks) != 1:
            raise ValueError('phase_recovery')
        start, peak, end = min(times), peaks[0], max(times)
        phases = {end: spec.phases[2], start: spec.phases[0], peak: spec.phases[1]}
        if (observations != [{'timestamp_s': t, 'text': phases[t]} for t in sorted(phases)]
                or spec.ordered and start >= peak):
            raise ValueError('phase_recovery')
        finding = {'type_id': kind, 't_start': start, 't_peak': peak, 't_end': end,
                   'confidence': confidence, 'insufficient_evidence': False, 'observations': observations}
        window = windows[index]
        frames = tuple(Frame(t, Path('unused-replay-frame'), t) for t in window.timestamps)
        discovery._parse(json.dumps({'findings': [finding]}, ensure_ascii=False), eligible, frames, window)
        candidate = discovery._candidate(finding)
        expected = ('insufficient_evidence' if confidence < threshold else
                    'overlaps_existing' if discovery._overlaps(candidate, accepted) else 'visible_evidence')
        if decision['reason'] != expected:
            raise ValueError('saved_owner_outcome_mismatch')
        if expected == 'visible_evidence':
            accepted.append(candidate)
        counts[expected] += 1
        recovered[window.timestamps].append(finding)
    if any(len(values) > 2 for values in recovered.values()):
        raise ValueError('window_finding_bound')
    if json.loads(canonical([asdict(c) for c in accepted])) != report['added_candidates']:
        raise ValueError('full_saved_candidate_recovery_mismatch')
    counts['saved_full_candidates_exact_recovered'] = len(accepted)
    return recovered, dict(counts)


def replay_candidates(originals, findings, types, duration, threshold):
    """Exercise discover_types + real augment_types + assemble.select unchanged."""
    from ml.assemble import select
    from ml.vlm import type_runtime
    from ml.vlm.contracts import Frame

    class PlaceholderFrames:
        calls = 0
        def extract(self, unused_path, timestamps, directory, timeout_s):
            self.calls += 1
            result = []
            for index, t in enumerate(timestamps):
                path = directory / f'{index}.png'
                path.write_bytes(b'NOT_AN_IMAGE_CPU_REPLAY_ONLY')
                result.append(Frame(t, path, t))
            return tuple(result)

    class SavedBackend:
        calls = 0
        def infer(self, request, timeout_s):
            self.calls += 1
            key = tuple(frame.timestamp_s for frame in request.frames)
            return json.dumps({'findings': findings[key]}, ensure_ascii=False)

    backend, frames = SavedBackend(), PlaceholderFrames()
    before = full_inventory(originals)
    previous = type_runtime._configuration
    try:
        if not type_runtime.configure_discovery(backend=backend, frame_source=frames,
                compact_verify_actions=True, min_confidence=threshold):
            raise ValueError('scripted_runtime_unavailable')
        augmented = type_runtime.augment_types('unused-cpu-replay', originals, types,
                                               duration_sec=duration, deadline=None)
        chosen, cv = select(augmented, types), select(originals, types)
    finally:
        type_runtime._configuration = previous
    if (len(augmented) < len(originals) or any(a is not b for a, b in zip(augmented, originals))
            or full_inventory(originals) != before):
        raise ValueError('original_cv_candidate_mutated')
    missing = full_inventory(cv) - full_inventory(chosen)
    if missing:
        raise ValueError('selected_full_cv_candidate_changed')
    return chosen, {'original_cv_objects_retained': True,
        'selected_cv_full_fields_exactly_retained': True,
        'original_candidate_count': len(originals), 'selected_cv_count': len(cv),
        'selected_count': len(chosen), 'scripted_extract_calls': frames.calls,
        'scripted_postverification_calls': backend.calls,
        'selected_candidate_inventory_sha256': hashlib.sha256(canonical(sorted(full_inventory(chosen).items())).encode()).hexdigest()}


def _read(path):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError('duplicate_json_key')
            value[key] = item
        return value
    def nonfinite(value):
        raise ValueError('nonfinite_json')
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError('report_too_large')
    return json.loads(path.read_text(), object_pairs_hook=pairs, parse_constant=nonfinite)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'probe', 'audit', 'cv-baseline', 'trusted-cache', 'hash-report', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--frozen', required=True)
    parser.add_argument('--trust-local-pickle-cache', action='store_true', required=True)
    args = parser.parse_args(argv)
    root = args.source_root.resolve()
    if 'ml' in sys.modules and Path(sys.modules['ml'].__file__).resolve().parent != root / 'ml':
        raise ValueError('run_file_directly_for_other_source_root')
    sys.path.insert(0, str(root))
    sys.meta_path.insert(0, NoModelRuntime())
    from ml import rules
    from ml.assemble import select
    from ml.evaluation_labels import video_key
    from ml.kb import KB_PATH, catalog

    logging.disable(logging.CRITICAL)
    inputs = {key: getattr(args, key).resolve() for key in ('probe', 'audit', 'cv_baseline', 'hash_report')}
    inputs['kb'] = KB_PATH
    digests = {key: sha(path) for key, path in inputs.items()}
    probe, audit, baseline, groups = (_read(inputs[key]) for key in ('probe', 'audit', 'cv_baseline', 'hash_report'))
    if (not audit.get('source_stable') or audit.get('baseline_sha256') != digests['cv_baseline']
            or audit.get('kb_sha256') != sha(KB_PATH) or set(probe) != set(baseline)
            or set(baseline) != set(audit['videos'])):
        raise ValueError('screen_provenance_mismatch')
    # Compare the whole source inventory, not only files exercised by this replay.
    sources = {str(p.relative_to(root)): sha(p) for p in (root / 'ml').rglob('*.py')}
    if sources != audit['source_hashes']:
        raise ValueError('screen_source_changed')
    frozen = subprocess.check_output(['git', '-C', str(root), 'rev-parse', args.frozen], text=True).strip()
    for path, digest in {**sources, str(KB_PATH.relative_to(root)): sha(KB_PATH)}.items():
        raw = subprocess.check_output(['git', '-C', str(root), 'show', frozen + ':' + path])
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError('source_differs_from_frozen')
    metadata = groups['content_aggregation']
    if metadata.get('hash_source') != 'actual' or metadata.get('hashes_authoritatively_verified') is not True:
        raise ValueError('verified_actual_hash_groups_required')
    group_hashes = {video_key(name): row['content_sha256'] for row in groups['per_content_group'] for name in row['videos']}
    hashes = {name: group_hashes[video_key(name)] for name in baseline}
    if (len(hashes) != len(group_hashes) or len(hashes) != sum(len(row['videos']) for row in groups['per_content_group'])
            or set(audit['videos']) != set(baseline)):
        raise ValueError('complete_hash_and_audit_coverage_required')
    types, threshold = catalog(), audit['min_confidence']
    if type(threshold) not in (float, int) or not 0 <= threshold <= 1:
        raise ValueError('invalid_owner_threshold')
    outputs, per_file, counts, cache_digests = {}, {}, Counter(), {}
    calls = Counter()
    for name, expected in baseline.items():
        item = audit['videos'][name]
        if item['sha256'] != hashes[name]:
            raise ValueError('content_identity_mismatch')
        if any(type(item.get(key)) is not int or not 0 <= item[key] <= 2
               for key in ('actual_verifier_calls', 'cached_proposal_attempts')):
            raise ValueError('invalid_screen_call_counts')
        if item.get('inference_reused'):
            owner = item['inference_source_video']
            if (owner == name or owner not in audit['videos'] or hashes[owner] != hashes[name]
                    or item['actual_verifier_calls'] != 0 or item['cached_proposal_attempts'] != 0
                    or item['report'] != audit['videos'][owner]['report']):
                raise ValueError('duplicate_reuse_mismatch')
            calls['duplicate_files_reused'] += 1
        elif item['report']['calls_used'] != item['actual_verifier_calls'] + item['cached_proposal_attempts']:
            raise ValueError('screen_call_accounting_mismatch')
        calls['fresh_screen_verifier_calls'] += item['actual_verifier_calls']
        calls['cached_screen_proposal_attempts'] += item['cached_proposal_attempts']
        cache = args.trusted_cache.resolve() / (name + '.pkl')
        if name != Path(name).name or cache.is_symlink() or not cache.is_file():
            raise ValueError('trusted_cache_identity')
        cache_digests[name] = sha(cache)
        with cache.open('rb') as stream:
            perception, scene = pickle.load(stream)  # Explicit local trusted-cache opt-in only.
        originals = list(rules.find_candidates(perception, scene=scene))
        duration = perception.info.duration
        if duration != item['duration_sec'] or flat(select(originals, types), duration) != {
                key: expected[key] for key in ('verdict', 'violations')}:
            raise ValueError('cached_cv_baseline_mismatch')
        del perception, scene
        findings, recovered = recover_findings(item, types, threshold)
        counts.update(recovered)
        chosen, checks = replay_candidates(originals, findings, types, duration, threshold)
        outputs[name] = flat(chosen, duration)
        checks['matches_screen_numeric_answer'] = outputs[name] == {key: probe[name][key] for key in ('verdict', 'violations')}
        per_file[name] = checks
        if sha(cache) != cache_digests[name]:
            raise ValueError('cache_changed')
    if (sources != {str(p.relative_to(root)): sha(p) for p in (root / 'ml').rglob('*.py')}
            or any(sha(inputs[key]) != digest for key, digest in digests.items())):
        raise ValueError('replay_inputs_changed')
    if any(name.split('.')[0] in {'torch', 'transformers', 'mlx', 'mlx_vlm', 'tensorflow', 'jax'} for name in sys.modules):
        raise ValueError('unexpected_model_runtime_import')
    changed = [name for name, item in per_file.items() if not item['matches_screen_numeric_answer']]
    result = {'status': 'completed_cpu_conditional_selection_replay', 'frozen_sha': frozen,
        'scope': {'saved_postverification_outputs_only': True, 'development_only': True,
            'actual_model_calls': 0, 'model_runtime_imports': 0, 'actual_media_decodes': 0,
            'evidence_renders': 0, 'full_job_or_deadline_equivalence_tested': False,
            'hashes_from_trusted_report_not_rehashed_media': True,
            'missing_or_low_confidence_findings_are_not_semantic_negatives': True},
        'files': len(outputs), 'actual_contents': len(set(hashes.values())),
        'all_screen_outputs_match_runtime_selection': not changed,
        'selection_difference_files': changed, 'source_stable': True,
        'min_confidence': threshold, 'recovered': dict(counts), 'screen_call_inventory': dict(calls), 'per_file': per_file,
        'input_sha256': digests, 'cache_sha256': cache_digests, 'predictions': outputs,
        'replay_source_sha256': sha(Path(__file__))}
    out = args.out.absolute()
    if ('..' in out.parts or out.exists() or out.suffix != '.json' or not out.is_relative_to(root / 'var')
            and not out.is_relative_to(Path(__file__).resolve().parents[2] / 'var')
            or any(path.is_symlink() for path in (out, *out.parents))):
        raise ValueError('fresh_private_var_output_required')
    out.parent.mkdir(parents=True, exist_ok=True)
    import os
    import tempfile
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', dir=out.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write('\n')
        os.link(temporary, out)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(json.dumps({'status': result['status'], 'files': len(outputs), 'contents': result['actual_contents'],
        'all_screen_outputs_match_runtime_selection': not changed, 'selection_difference_count': len(changed),
        'actual_model_calls': 0, 'actual_media_decodes': 0}))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError, TypeError):
        print('{"status":"blocked","reason":"replay_input_or_contract_mismatch"}')
        raise SystemExit(2)
