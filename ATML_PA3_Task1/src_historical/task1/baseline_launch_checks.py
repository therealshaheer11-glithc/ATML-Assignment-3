"""Exercise the actual new parent controller and raw-token cache on CPU."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from baseline_runtime import atomic, termination, load_generation, save_generation
from launch_baseline import execute_worker
from release_coordination import finalize


def cache_checks(directory):
    directory = Path(directory); directory.mkdir(parents=True)
    assert termination([1, 2], [9], ['STOP'], 'text', 2)['truncated']
    assert not termination([1, 9], [9], [], 'text', 2)['truncated']
    assert termination([1, 9], [9], [], 'text', 2)['at_token_cap']
    assert not termination([1, 2], [9], ['STOP'], 'STOP', 2)['truncated']
    contract = {'input_ids': [7, 8], 'model': 'original'}
    path = directory / 'raw.json'
    assert load_generation(path, contract) is None
    saved = save_generation(path, contract, [[7, 8, 1, 9]], {'generated_tokens': 2})
    assert load_generation(path, contract) == saved
    try: load_generation(path, {**contract, 'model': 'different'})
    except RuntimeError: pass
    else: raise AssertionError('Cross-model raw generation reused')
    changed = json.loads(path.read_text()); changed['sequence_ids'][0][-1] = 3
    path.write_text(json.dumps(changed))
    try: load_generation(path, contract)
    except RuntimeError: pass
    else: raise AssertionError('Edited tokens reused')
    return {'capped_answer_retained_and_flagged': True, 'EOS_at_cap_is_normal_termination': True,
            'stock_stop_at_cap_is_normal_termination': True, 'raw_tokens_reused': True,
            'raw_model_identity_checked': True, 'edited_raw_tokens_rejected': True}


def parent_case(bundle, directory, kind):
    directory = Path(directory); directory.mkdir(parents=True)
    state, event, released = directory / 'state.json', directory / 'event.json', directory / 'released.txt'
    now = time.time()
    atomic(state, {'deadline_wall': now + 15, 'progress_wall': now, 'progress_timeout': 900,
                   'worker_pid': None, 'status': 'setup'})
    env = os.environ.copy(); env['TASK1_SAFETY_SELFTEST'] = '1'
    guard = subprocess.Popen([sys.executable, str(bundle / 'gpu_guardian_R2.py'), str(state),
        str(event), str(released)], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, start_new_session=True)
    result_path = directory / 'result.json'; final = directory / 'session_final.json'
    if kind == 'final_save_error':
        final.write_text('{"preserved_preexisting_evidence":true}\n')
    worker = directory / 'dummy.py'
    worker.write_text('''import json,os,pathlib,sys,time
s=pathlib.Path(sys.argv[1]);r=pathlib.Path(sys.argv[2]);kind=sys.argv[3]
while json.loads(s.read_text()).get('worker_pid')!=os.getpid():time.sleep(.01)
if kind!='missing_result':
 status='TASK1_BASELINE_PARTIAL_SAVED' if kind=='partial' else 'TASK1_BASELINE_COMPLETE_SAVED'
 counts={'gsm8k_cot':1319,'hellaswag':10042,'boolq':3270,'arc_easy':2376}
 result={'run_id':'fixture','status':status,'official_training_updates':0,
  'benchmark_completed':counts,'qualitative_completed':24,'activation_probes_completed':64,
  'strict_gate':{'status':'TASK1_STRICT_BACKWARD_GATE_OK'},
  'baseline_summary':{'status':'TASK1_ORIGINAL_BASELINE_COMPLETE','benchmark_counts':counts}}
 if kind=='false_complete':result['benchmark_completed']={}
 r.write_text(json.dumps(result))
d=json.loads(s.read_text());d.update(status='worker_complete',progress_wall=time.time(),worker_completed_wall=time.time())
p=s.with_suffix('.worker_pending');p.write_text(json.dumps(d));p.replace(s)
time.sleep(.15)
sys.exit(7 if kind=='worker_error' else 0)
''')
    success = False; exceptions = []
    try:
        try:
            result = execute_worker([sys.executable, str(worker), str(state), str(result_path), kind],
                env, state, result_path, final,
                {'run_id': 'fixture', 'started_wall': now, 'displayed_units_per_hour': 2.02}, directory / 'console.txt')
            success = True
            assert kind in ('complete', 'partial')
            assert json.loads(final.read_text())['worker_result_validated']
            assert not released.exists(), 'Guardian released before parent finalized the saved result'
        except Exception as error:
            exceptions.append(type(error).__name__)
            assert kind in ('worker_error', 'missing_result', 'final_save_error', 'false_complete'), (kind, error)
        outcome = finalize(state, event, lambda: (_ for _ in ()).throw(AssertionError('Unexpected fallback')),
                           success, wait_seconds=3)
        guard.wait(timeout=5)
        assert outcome['primary_request_accepted'] and not outcome['fallback_called']
        assert released.read_text() == 'SIMULATED_RELEASE_ONLY\n'
        assert json.loads(event.read_text())['reason'] == ('finished' if success else 'failed')
        assert success == (kind in ('complete', 'partial'))
        if kind in ('worker_error', 'false_complete'):
            assert not json.loads(final.read_text())['worker_result_validated']
        if kind == 'final_save_error': assert json.loads(final.read_text()) == {'preserved_preexisting_evidence': True}
        return True
    finally:
        if guard.poll() is None: guard.kill(); guard.wait()


def run(bundle, directory):
    bundle, directory = Path(bundle), Path(directory)
    result = {'raw_generation_checks': cache_checks(directory / 'cache')}
    result['parent_integration_checks'] = {kind: parent_case(bundle, directory / kind, kind)
        for kind in ('complete', 'partial', 'worker_error', 'missing_result', 'final_save_error', 'false_complete')}
    return result


if __name__ == '__main__': print(json.dumps(run(sys.argv[1], sys.argv[2]), indent=2))
