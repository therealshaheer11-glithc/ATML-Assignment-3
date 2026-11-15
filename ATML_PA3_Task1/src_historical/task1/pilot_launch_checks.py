"""Real CPU-process tests of the pilot launcher and R3 release evidence."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from baseline_runtime import atomic
from launch_batch8_pilot import CHECKS, execute, validate_result
from release_coordination import finalize
from watchdog_evidence import cpu_checks


def example(run_id, status='TASK1_BATCH8_PILOT_PASSED'):
    return {'run_id': run_id, 'status': status, 'approval_id': 'T1-A003',
        'official_training_updates': 0, 'official_benchmark_answers_generated': 0,
        'conditional_adoption_gate_passed': True, 'checks': {key: True for key in CHECKS},
        'token_agreement_by_prompt': [True] * 8, 'postprocessed_text_agreement_by_prompt': [True] * 8,
        'model_inference_speedup': 3., 'headroom_fraction': .5,
        'max_absolute_logit_difference': .01, 'relative_RMS_logit_difference': .001}


def case(bundle, folder, kind):
    folder = Path(folder); folder.mkdir(parents=True)
    root = folder / 'drive'; (root / 'logs').mkdir(parents=True)
    state, event, released = folder / 'state.json', folder / 'event.json', folder / 'released.txt'
    run = 'task1_pilot_fixture_' + kind; now = time.time()
    atomic(state, {'deadline_wall': now + 15, 'progress_wall': now, 'progress_timeout': 900,
        'status': 'setup', 'worker_pid': None, 'parent_pid': os.getpid(),
        'evidence_directory': str(root / 'logs'), 'run_id': run})
    env = os.environ.copy(); env['TASK1_SAFETY_SELFTEST'] = '1'
    guard = subprocess.Popen([sys.executable, str(bundle / 'gpu_guardian_R3.py'), str(state), str(event), str(released)],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    value = example(run)
    if kind == 'rejected':
        value.update(status='TASK1_BATCH8_PILOT_REJECTED', conditional_adoption_gate_passed=False)
        value['checks']['speedup_at_least_2'] = False
    if kind == 'false_pass': value['model_inference_speedup'] = 1.
    if kind == 'incomplete': value.update(status='TASK1_BATCH8_PILOT_INCOMPLETE', conditional_adoption_gate_passed=False)
    payload = folder / 'payload.json'; payload.write_text(json.dumps(value))
    journal = root / 'pilot_runs' / run; journal.mkdir(parents=True)
    (journal / 'COMPARISON.json').write_text(json.dumps({'result': value}))
    dummy = folder / 'dummy.py'
    dummy.write_text('''import json,os,pathlib,sys,time
s=pathlib.Path(sys.argv[1]);r=pathlib.Path(sys.argv[2]);p=pathlib.Path(sys.argv[3]);kind=sys.argv[4]
while json.loads(s.read_text()).get('worker_pid')!=os.getpid():time.sleep(.01)
if kind!='missing':r.write_bytes(p.read_bytes())
d=json.loads(s.read_text());d.update(status='worker_complete',progress_wall=time.time(),worker_completed_wall=time.time())
t=s.with_suffix('.pending_worker');t.write_text(json.dumps(d));t.replace(s);time.sleep(.1)
sys.exit(7 if kind=='worker_error' else 0)
''')
    context = {'root': str(root), 'job': str(folder), 'state': str(state), 'journal': str(journal),
               'started_wall': now, 'rate': 1.94}
    session = {'run_id': run, 'implementation_identity': 'CPU_FIXTURE_ONLY'}
    success = False
    try:
        try:
            execute([sys.executable, str(dummy), str(state), str(root / 'logs' / f'{run}_result.json'),
                str(payload), kind], env, context, session)
            success = True
        except Exception:
            assert kind in ('missing', 'worker_error', 'false_pass')
        assert success == (kind in ('passed', 'rejected', 'incomplete'))
        assert not released.exists(), 'Release occurred before parent acknowledgement'
        outcome = finalize(state, event, lambda: (_ for _ in ()).throw(AssertionError('Unexpected fallback')),
                           success, wait_seconds=3)
        guard.wait(timeout=5)
        assert outcome['primary_request_accepted']
        assert released.read_text() == 'SIMULATED_RELEASE_ONLY\n'
        intent = json.loads((root / 'logs' / f'{run}_release_intent.json').read_text())
        assert intent['reason'] == ('finished' if success else 'failed')
        gate = root / 'configs' / f'BATCH8_PILOT_GATE_{run}.json'
        assert gate.exists() == (kind == 'passed')
        return True
    finally:
        if guard.poll() is None: guard.kill(); guard.wait()


def parent_death(bundle, directory):
    directory = Path(directory); directory.mkdir(parents=True)
    state, event, released = directory / 'state.json', directory / 'event.json', directory / 'released.txt'
    worker = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'], start_new_session=True)
    atomic(state, {'deadline_wall': time.time() + 10, 'progress_wall': time.time(),
        'worker_pid': worker.pid, 'parent_pid': 99999999, 'status': 'running',
        'evidence_directory': str(directory), 'run_id': 'task1_parent_fixture'})
    env = os.environ.copy(); env['TASK1_SAFETY_SELFTEST'] = '1'
    guard = subprocess.Popen([sys.executable, str(bundle / 'gpu_guardian_R3.py'), str(state), str(event), str(released)], env=env)
    try:
        guard.wait(timeout=6); worker.wait(timeout=3)
        assert worker.returncode != 0
        assert json.loads(event.read_text())['reason'] == 'parent_exited'
        assert json.loads((directory / 'task1_parent_fixture_release_intent.json').read_text())['reason'] == 'parent_exited'
        return True
    finally:
        if worker.poll() is None: worker.kill(); worker.wait()
        if guard.poll() is None: guard.kill(); guard.wait()


def run(bundle, directory):
    bundle, directory = Path(bundle), Path(directory)
    assert validate_result(example('fixture'), 0, 'fixture')
    invalid = example('fixture'); invalid['official_benchmark_answers_generated'] = 1
    assert not validate_result(invalid, 0, 'fixture')
    results = {'evidence_writer': cpu_checks(directory / 'evidence')}
    results['pilot_parent_cases'] = {kind: case(bundle, directory / kind, kind)
        for kind in ('passed', 'rejected', 'incomplete', 'missing', 'worker_error', 'false_pass')}
    results['parent_death_releases_and_saves_reason'] = parent_death(bundle, directory / 'parent_death')
    import importlib.util
    spec = importlib.util.spec_from_file_location('guardian_signal_failure_fixture', bundle / 'gpu_guardian_R3.py')
    guardian = importlib.util.module_from_spec(spec); spec.loader.exec_module(guardian)
    folder = directory / 'signal_error'; folder.mkdir()
    state, event = folder / 'state.json', folder / 'event.json'
    atomic(state, {'status': 'failed', 'deadline_wall': time.time() + 10, 'progress_wall': time.time()})
    calls = []
    def denied(pid): raise PermissionError('INTENTIONAL_SIGNAL_FAILURE_FIXTURE')
    guardian.terminate = denied
    guardian.release = lambda test_file=None: calls.append(1) or 'SIMULATED_RELEASE_ONLY'
    saved = guardian.supervise(state, event)
    assert calls == [1] and 'worker_termination_error' in saved and saved['release_result'] == 'SIMULATED_RELEASE_ONLY'
    results['worker_signal_failure_still_requests_release'] = True
    return results


if __name__ == '__main__': print(json.dumps(run(sys.argv[1], sys.argv[2]), indent=2))
