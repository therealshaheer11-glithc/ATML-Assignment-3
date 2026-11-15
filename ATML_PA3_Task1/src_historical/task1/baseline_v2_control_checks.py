"""CPU adoption/publication and real parent-process tests; no experimental inference."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from baseline_runtime import atomic, save_generation
from evaluation_records import COUNTS, identity, publish
from release_coordination import finalize
from launch_baseline_v2 import execute


def protocol_checks(directory):
    import baseline_v2_protocol as protocol_module
    from archive_io import digest
    from pilot_launch_checks import example
    root = Path(directory); root.mkdir(parents=True)
    def put(rel, value):
        path = root / rel; publish(path, value); return path
    old = {'approval_ids': ['T1-A001', 'T1-A002'], 'generation_batch_size': 1,
        'likelihood_batch_size': 2, 'gsm_max_new_tokens': 1024, 'num_fewshot': 0,
        'counts_per_condition': COUNTS, 'dtype': 'float16', 'dropout': False,
        'thinking': False, 'qualitative_max_new_tokens': 512,
        'conditions': ['original', 'full_mid', 'full', 'l2sp', 'lora4', 'lora16', 'oplora16']}
    original_constant = protocol_module.OLD_PROTOCOL
    protocol_module.OLD_PROTOCOL = identity(old)
    try:
        put('configs/old.json', old)
        originals = {'model': {'repo': 'CPU_FIXTURE_ONLY'}, 'files': {}}
        put('manifests/ORIGINAL_MODEL_FILES.json', originals)
        model = identity({'model': originals, 'evaluation_dtype': 'float16'})
        audit = {'condition': 'original', 'model_identity': model,
            'protocol_identity': identity(old), 'gpu_implementation_identity': 'CPU_FIXTURE_ONLY'}
        questions, entries = [], []
        for i in range(24):
            row = {'id': f'fixture_{i}', 'messages': [{'role': 'user', 'content': 'CPU_FIXTURE_ONLY'}],
                   'prompt_input_ids': [i + 100], 'max_new_tokens': 512}
            questions.append(row)
            contract = {'audit': audit, 'input_ids': row['prompt_input_ids'],
                        'max_length': 513, 'generation_kwargs': {'do_sample': False}}
            raw_rel = f'results/original/raw_generation/{identity(contract)}.json'
            save_generation(root / raw_rel, contract, [[i + 100, 9]], {})
            rel = f'results/original/qualitative/fixture_{i}.json'
            path = put(rel, {'contract': {'audit': audit, 'prepared_prompt_identity': identity(row)},
                'messages': row['messages'], 'response': 'CPU_FIXTURE_ONLY', 'metadata': {'raw_record': raw_rel}})
            entries.append({'id': row['id'], 'path': rel, 'sha256': digest(path)})
        qpath = put('evaluation_inputs/qual.json', questions)
        put('results/original/QUALITATIVE_COMPLETE.json', {'audit': audit, 'count': 24, 'entries': entries})
        probes, entries = [], []
        for task in ('gsm8k_cot', 'hellaswag'):
            for i in range(32):
                row = {'task': task, 'source_index': i, 'prompt_input_ids': [i + 1000]}; probes.append(row)
                rel = f'results/original/activation_probes/{task}_{i}.safetensors'
                path = root / rel; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'CPU_HASH_FIXTURE_ONLY')
                put(str(Path(rel).with_suffix('.json')), {'sha256': digest(path),
                    'contract': {'audit': audit, 'prompt_identity': identity(row), 'forward_dtype': 'float32'}})
                entries.append({'task': task, 'source_index': i, 'path': rel, 'sha256': digest(path)})
        ppath = put('evaluation_inputs/probes.json', probes)
        put('results/original/ACTIVATION_PROBES_COMPLETE.json', {'audit': audit, 'count': 64, 'entries': entries})
        put('configs/BASELINE_CPU_READY_V1.json', {'protocol_path': 'configs/old.json',
            'qualitative_path': 'evaluation_inputs/qual.json', 'qualitative_sha256': digest(qpath),
            'probe_path': 'evaluation_inputs/probes.json', 'probe_sha256': digest(ppath)})
        result = example(protocol_module.PILOT_RUN)
        rpath = put('logs/pilot_result.json', result)
        cpath = put('pilot_runs/COMPARISON.json', {'result': result})
        put(f'configs/BATCH8_PILOT_GATE_{protocol_module.PILOT_RUN}.json', {
            'approval_id': 'T1-A003', 'conditional_adoption_gate_passed': True,
            'result_path': 'logs/pilot_result.json', 'result_sha256': digest(rpath),
            'comparison_path': 'pilot_runs/COMPARISON.json', 'comparison_sha256': digest(cpath)})
        old_marker = put('results/original/gsm8k_cot/00000.json', {'preliminary_fixture': 'PRESERVE'})
        marker_sha = digest(old_marker)
        first = protocol_module.prepare(root); second = protocol_module.prepare(root)
        assert first == second and digest(old_marker) == marker_sha
        assert first['protocol']['generation_batch_size'] == 8 and first['protocol']['likelihood_batch_size'] == 2
        assert first['protocol']['qualitative_generation_batch_size'] == 1
        assert first['protocol']['counts_per_condition'] == COUNTS and first['protocol']['gsm_max_new_tokens'] == 1024
        for key in ('dtype', 'dropout', 'thinking', 'qualitative_max_new_tokens', 'conditions'):
            assert first['protocol'][key] == old[key]
        assert first['references']['qualitative_count'] == 24 and first['references']['probe_count'] == 64
        (root / 'results/original/qualitative/fixture_0.json').write_text('{}')
        try: protocol_module.prepare(root)
        except AssertionError: pass
        else: raise AssertionError('Changed reference was accepted')
        return {'actual_new_protocol_and_reference_publications_executed': True,
            'repeat_preparation_is_idempotent': True, 'old_preliminary_GSM_evidence_preserved': True,
            'all_24_qualitative_and_64_probe_reference_hashes_checked': True,
            'changed_reference_rejected': True, 'required_counts_caps_precision_policy_preserved': True}
    finally:
        protocol_module.OLD_PROTOCOL = original_constant


def parent_case(bundle, directory, kind):
    root = Path(directory); (root / 'logs').mkdir(parents=True)
    run = 'task1_v2_fixture_' + kind; now = time.time()
    state, event, released = root / 'state.json', root / 'event.json', root / 'released.txt'
    atomic(state, {'deadline_wall': now + 15, 'progress_wall': now, 'progress_timeout': 900,
        'worker_pid': None, 'parent_pid': os.getpid(), 'status': 'setup',
        'run_id': run, 'evidence_directory': str(root / 'logs')})
    env = os.environ.copy(); env['TASK1_SAFETY_SELFTEST'] = '1'
    guard = subprocess.Popen([sys.executable, str(bundle / 'gpu_guardian_R3.py'), str(state), str(event), str(released)],
        env=env, start_new_session=True)
    summary = {'status': 'TASK1_ORIGINAL_BASELINE_V2_COMPLETE', 'benchmark_counts': COUNTS,
               'audit': {'model_identity': 'CPU_FIXTURE', 'protocol_identity': 'CPU_FIXTURE'}}
    value = {'status': 'TASK1_BASELINE_V2_COMPLETE_SAVED', 'run_id': run, 'official_training_updates': 0,
        'qualitative_responses_reused': 24, 'FP32_probe_files_reused': 64,
        'benchmark_completed': COUNTS, 'baseline_summary': summary}
    if kind == 'partial': value['status'] = 'TASK1_BASELINE_V2_PARTIAL_SAVED'
    if kind == 'false_complete': value['benchmark_completed'] = {name: 0 for name in COUNTS}
    final = root / 'logs' / f'{run}_session_final.json'
    if kind == 'save_error': final.write_text('{"preserved":true}')
    complete = root / 'results_batch8_V2/original/BASELINE_COMPLETE.json'; publish(complete, summary)
    payload = root / 'payload.json'; payload.write_text(json.dumps(value))
    dummy = root / 'dummy.py'
    dummy.write_text('''import json,os,pathlib,sys,time
s=pathlib.Path(sys.argv[1]);r=pathlib.Path(sys.argv[2]);p=pathlib.Path(sys.argv[3]);kind=sys.argv[4]
while json.loads(s.read_text()).get('worker_pid')!=os.getpid():time.sleep(.01)
if kind!='missing':r.write_bytes(p.read_bytes())
d=json.loads(s.read_text());d.update(status='worker_complete',progress_wall=time.time(),worker_completed_wall=time.time())
t=s.with_suffix('.worker_pending');t.write_text(json.dumps(d));t.replace(s);time.sleep(.1)
sys.exit(7 if kind=='worker_error' else 0)
''')
    success = False
    try:
        try:
            execute([sys.executable, str(dummy), str(state), str(root / 'logs' / f'{run}_result.json'),
                str(payload), kind], env,
                {'root': str(root), 'state': str(state), 'started_wall': now, 'rate': 1.94},
                {'run_id': run, 'model_identity': 'CPU_FIXTURE', 'protocol_identity': 'CPU_FIXTURE'})
            success = True
        except Exception:
            assert kind in ('false_complete', 'save_error', 'missing', 'worker_error')
        assert success == (kind in ('complete', 'partial'))
        assert not released.exists()
        outcome = finalize(state, event, lambda: (_ for _ in ()).throw(AssertionError('Unexpected fallback')), success, wait_seconds=3)
        guard.wait(timeout=5)
        assert outcome['primary_request_accepted'] and released.read_text() == 'SIMULATED_RELEASE_ONLY\n'
        assert json.loads(event.read_text())['reason'] == ('finished' if success else 'failed')
        if kind == 'save_error': assert json.loads(final.read_text()) == {'preserved': True}
        return True
    finally:
        if guard.poll() is None: guard.kill(); guard.wait()


def run(bundle, directory):
    bundle, directory = Path(bundle), Path(directory)
    return {'protocol_and_reference_checks': protocol_checks(directory / 'protocol'),
        'parent_controller_cases': {kind: parent_case(bundle, directory / kind, kind)
            for kind in ('complete', 'partial', 'false_complete', 'save_error', 'missing', 'worker_error')},
        'scope': 'CPU synthetic control-flow/provenance tests; no original-model numerical execution'}
