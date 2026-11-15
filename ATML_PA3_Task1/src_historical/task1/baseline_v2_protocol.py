"""Versioned conditional adoption and verified original qualitative/probe reuse."""
import copy
import json
from pathlib import Path

from archive_io import digest, safe_relative
from baseline_runtime import load_generation
from evaluation_records import identity, publish
from launch_batch8_pilot import validate_result

PILOT_RUN = 'task1_batch8_pilot_df8cf72da3c149b9ae6e99d5abf04942'
OLD_PROTOCOL = '2b0cdf3ca8dd889e6e63996af55aba94672110f5e0d79e7ca1e843434d7cf297'
NAMESPACE = 'results_batch8_V2'
TASK_ORDER = ('hellaswag', 'boolq', 'arc_easy', 'gsm8k_cot')


def pilot_gate(root):
    root = Path(root)
    gate_path = root / 'configs' / f'BATCH8_PILOT_GATE_{PILOT_RUN}.json'
    gate = json.loads(gate_path.read_text())
    assert gate['approval_id'] == 'T1-A003' and gate['conditional_adoption_gate_passed'] is True
    result_path = root / str(safe_relative(gate['result_path']))
    assert digest(result_path) == gate['result_sha256']
    result = json.loads(result_path.read_text()); assert validate_result(result, 0, PILOT_RUN)
    assert result['status'] == 'TASK1_BATCH8_PILOT_PASSED'
    comparison = root / str(safe_relative(gate['comparison_path']))
    assert digest(comparison) == gate['comparison_sha256']
    return {'approval_id': 'T1-A003', 'source_run': PILOT_RUN,
            'gate_path': str(gate_path.relative_to(root)), 'gate_sha256': digest(gate_path),
            'result_sha256': digest(result_path), 'all_required_pilot_checks_passed': True}


def make_protocol(old):
    assert identity(old) == OLD_PROTOCOL
    protocol = copy.deepcopy(old)
    protocol['approval_ids'] = list(old['approval_ids']) + ['T1-A003']
    protocol.update(schema_version=2, implementation_protocol='V2_BATCH8',
        amendment='T1-A003 / D10B', parent_protocol_identity=OLD_PROTOCOL,
        generation_batch_size=8, generation_batch_size_scope='GSM8K only',
        qualitative_generation_batch_size=1, task_order=list(TASK_ORDER),
        physical_chunk_membership='Fixed full source-order chunks: generation 25, likelihood 100; native harness ordering within chunks',
        raw_journals='Whole physical generation batches; complete fixed likelihood chunks',
        result_namespace=NAMESPACE,
        preliminary_batch_one_results_retained_at='results/original',
        GSM_baseline='Regenerate under V2; do not mix preliminary batch-one answers',
        baseline_qualitative='Reuse verified original-model batch-one qualitative responses by reference',
        activation_probes='Reuse verified original-model FP32 prompt-only activations for all methods')
    return protocol


def verified_references(root, ready, model_identity):
    root = Path(root)
    qpath = root / str(safe_relative(ready['qualitative_path'])); assert digest(qpath) == ready['qualitative_sha256']
    questions = {row['id']: row for row in json.loads(qpath.read_text())}; assert len(questions) == 24
    qmanifest_path = root / 'results/original/QUALITATIVE_COMPLETE.json'
    qmanifest = json.loads(qmanifest_path.read_text()); source_audit = qmanifest['audit']
    assert source_audit['condition'] == 'original' and source_audit['model_identity'] == model_identity
    assert source_audit['protocol_identity'] == OLD_PROTOCOL and qmanifest['count'] == 24
    seen = set()
    for entry in qmanifest['entries']:
        assert entry['id'] in questions and entry['id'] not in seen
        path = root / str(safe_relative(entry['path'])); assert digest(path) == entry['sha256']
        row = json.loads(path.read_text()); original = questions[entry['id']]
        assert row['contract'] == {'audit': source_audit, 'prepared_prompt_identity': identity(original)}
        assert row['messages'] == original['messages'] and isinstance(row['response'], str)
        raw_path = root / str(safe_relative(row['metadata']['raw_record']))
        raw_contract = json.loads(raw_path.read_text())['contract']
        raw = load_generation(raw_path, raw_contract)
        assert raw_contract['audit'] == source_audit and raw_contract['input_ids'] == original['prompt_input_ids']
        assert raw_contract['max_length'] - len(raw_contract['input_ids']) == 512
        assert raw_contract['generation_kwargs']['do_sample'] is False
        seen.add(entry['id'])
    assert seen == set(questions)
    probe_path = root / str(safe_relative(ready['probe_path'])); assert digest(probe_path) == ready['probe_sha256']
    expected = {(p['task'], p['source_index']): p for p in json.loads(probe_path.read_text())}
    pmanifest_path = root / 'results/original/ACTIVATION_PROBES_COMPLETE.json'
    pmanifest = json.loads(pmanifest_path.read_text())
    assert pmanifest['audit'] == source_audit and pmanifest['count'] == 64
    seen = set()
    for entry in pmanifest['entries']:
        key = (entry['task'], entry['source_index']); assert key in expected and key not in seen
        path = root / str(safe_relative(entry['path'])); assert digest(path) == entry['sha256']
        meta = json.loads(path.with_suffix('.json').read_text())
        assert meta['sha256'] == entry['sha256']
        assert meta['contract'] == {'audit': source_audit,
            'prompt_identity': identity(expected[key]), 'forward_dtype': 'float32'}
        seen.add(key)
    assert seen == set(expected) and len(seen) == 64
    return {'scope': 'Unchanged original-baseline qualitative responses; original FP32 probes are shared for all methods',
        'model_identity': model_identity, 'source_protocol_identity': OLD_PROTOCOL,
        'qualitative_count': 24, 'probe_count': 64,
        'qualitative_manifest': str(qmanifest_path.relative_to(root)), 'qualitative_manifest_sha256': digest(qmanifest_path),
        'probe_manifest': str(pmanifest_path.relative_to(root)), 'probe_manifest_sha256': digest(pmanifest_path),
        'human_qualitative_annotation_pending': True, 'previous_GSM_answers_reused_for_main_V2_metrics': False}


def prepare(root):
    root = Path(root)
    evidence = pilot_gate(root)
    ready = json.loads((root / 'configs/BASELINE_CPU_READY_V1.json').read_text())
    old = json.loads((root / ready['protocol_path']).read_text())
    protocol = make_protocol(old); protocol_id = identity(protocol)
    originals = json.loads((root / 'manifests/ORIGINAL_MODEL_FILES.json').read_text())
    model_id = identity({'model': originals, 'evaluation_dtype': 'float16'})
    references = verified_references(root, ready, model_id)
    publish(root / 'configs/EVALUATION_PROTOCOL_V2_BATCH8.json', protocol)
    publish(root / 'provenance/T1_A003_BATCH8_ADOPTION.json', {**evidence,
        'new_protocol_identity': protocol_id, 'old_protocol_identity': OLD_PROTOCOL,
        'preliminary_outputs_preserved': True})
    publish(root / NAMESPACE / 'original/REUSED_REFERENCE_MANIFEST.json', references)
    return {'protocol': protocol, 'protocol_identity': protocol_id,
            'model_identity': model_id, 'ready': ready, 'references': references, 'adoption': evidence}
