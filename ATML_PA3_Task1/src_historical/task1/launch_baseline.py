"""Verified restoration and repaired parent/worker completion handoff."""
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from archive_io import digest, restore, safe_relative
from baseline_runtime import atomic, progress, publish_bytes
from evaluation_records import COUNTS, identity, publish
from restore_baseline_cpu import inventory_complete


PROTOCOL = '2b0cdf3ca8dd889e6e63996af55aba94672110f5e0d79e7ca1e843434d7cf297'
ACCEPTED = ('TASK1_BASELINE_COMPLETE_SAVED', 'TASK1_BASELINE_PARTIAL_SAVED')


def valid_worker_result(result, code, session):
    valid = (code == 0 and result.get('status') in ACCEPTED
        and result.get('run_id') == session['run_id'] and result.get('official_training_updates') == 0)
    if valid and result['status'] == 'TASK1_BASELINE_COMPLETE_SAVED':
        valid = (result.get('benchmark_completed') == COUNTS
            and result.get('qualitative_completed') == 24 and result.get('activation_probes_completed') == 64
            and result.get('strict_gate', {}).get('status') == 'TASK1_STRICT_BACKWARD_GATE_OK'
            and result.get('baseline_summary', {}).get('status') == 'TASK1_ORIGINAL_BASELINE_COMPLETE'
            and result.get('baseline_summary', {}).get('benchmark_counts') == COUNTS)
    return valid


def restore_archive(root, destination, record, entries, job, state):
    progress(state, 'verify local software/input inventory')
    if inventory_complete(destination, entries):
        progress(state, 'local software/input inventory verified')
        return
    source = root / str(safe_relative(record['archive_path']))
    copy = job / source.name
    progress(state, 'copy frozen archive ' + source.name)
    hasher = hashlib.sha256(); last = time.monotonic()
    with source.open('rb') as src, copy.open('xb') as output:
        for block in iter(lambda: src.read(8 * 1024**2), b''):
            output.write(block); hasher.update(block)
            if time.monotonic() - last > 2:
                progress(state, 'copy ' + source.name); last = time.monotonic()
    assert hasher.hexdigest() == record['archive_sha256'], 'Frozen archive hash differs'
    restore(copy, destination, entries, lambda phase: progress(state, phase))
    copy.unlink()


def execute_worker(command, env, state_path, result_path, final_session_path, session, log_path):
    """The guardian receives the parent's ACK only after this function returns.

    The worker waits for worker_pid publication before taking ownership of state.
    Final session evidence is read back before the parent can acknowledge success.
    """
    worker = None
    try:
        with Path(log_path).open('x') as log:
            worker = subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
            state = json.loads(Path(state_path).read_text())
            state.update(worker_pid=worker.pid, status='running', progress_wall=time.time())
            atomic(state_path, state)
            for line in worker.stdout:
                print(line, end='', flush=True); log.write(line); log.flush()
            code = worker.wait()
        result = json.loads(Path(result_path).read_text())
        valid = valid_worker_result(result, code, session)
        finished = {**session, 'status': result['status'], 'worker_exit_code': code,
            'worker_result_validated': valid, 'result_path': str(result_path),
            'result_sha256': digest(result_path), 'parent_finalized_wall': time.time(),
            'elapsed_allocation_seconds_at_final_save': time.time() - session['started_wall'],
            'release_request_still_pending': True, 'physical_release_verified': False}
        finished['estimated_units_so_far_with_margin'] = (
            finished['elapsed_allocation_seconds_at_final_save'] / 3600 * session['displayed_units_per_hour'] * 1.1)
        publish(final_session_path, finished)
        if not valid:
            raise RuntimeError('Worker did not finish successfully; its result and final session evidence were preserved')
        return result
    finally:
        if worker is not None and worker.poll() is None:
            try: os.killpg(worker.pid, signal.SIGTERM)
            except ProcessLookupError: pass
            try: worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try: os.killpg(worker.pid, signal.SIGKILL)
                except ProcessLookupError: pass
                worker.wait(timeout=5)


def run(context, bundle):
    root, job, bundle = Path(context['root']), Path(context['job']), Path(bundle)
    state = context['state']; run_id = job.name
    progress(state, 'validate CPU readiness and supervisor R2')
    ready = json.loads((root / 'configs/BASELINE_CPU_READY_V1.json').read_text())
    assert ready['protocol_identity'] == PROTOCOL
    assert all(all(v.values()) for v in ready['stock_scoring_checks'].values())
    assert all(all(v.values()) for v in ready['resume_engine_checks'].values())
    assert all(ready['record_checks'].values()) and all(ready['asset_manifest_regressions'].values())
    for name, expected in ready['files'].items():
        assert '/' not in name and digest(bundle / name) == expected, 'CPU-checked evaluation code differs: ' + name
    safety = json.loads((root / 'configs/SUPERVISOR_R2_READY.json').read_text())
    assert all(safety['prior_safety_tests'].values())
    assert all(safety['completion_tests'].values()) and all(safety['release_coordination_tests'].values())
    assert digest(job / 'guardian.py') == safety['files']['gpu_guardian_R2.py']
    assert digest(bundle / 'release_coordination.py') == safety['files']['release_coordination.py']
    files = {path.name: digest(path) for path in bundle.iterdir() if path.is_file() and path.suffix == '.py'}
    implementation = identity(files)
    session = {'run_id': run_id, 'started_wall': context['started_wall'],
        'status': 'STARTED', 'balance_at_start': context['balance'],
        'displayed_units_per_hour': context['rate'], 'budget_reserve_compute_units': 23.26,
        'approved_total_Task1_cap': 55., 'budget_rate_margin': 1.1,
        'soft_wall': context['soft_wall'], 'deadline_wall': context['deadline_wall'],
        'release_reserve_seconds': 45, 'official_training_updates': 0,
        'protocol_identity': PROTOCOL, 'implementation_identity': implementation,
        'billed_units': 'not measured; elapsed-time estimates include restoration but release may take additional time'}
    publish(root / 'logs' / (run_id + '_session_started.json'), session)
    code_backup = root / 'execution' / ('baseline_gpu_V1_' + implementation)
    for name in sorted(files):
        publish_bytes(code_backup / name, (bundle / name).read_bytes())
    publish(code_backup / 'CONTROLLER_MANIFEST.json', {
        'protocol_identity': PROTOCOL, 'implementation_identity': implementation,
        'files': files, 'supervisor_R2_verified': True,
        'CPU_checked_evaluation_files': ready['files'],
        'strict_deterministic_GPU_gate_required_before_baseline': True})
    local = Path('/content/atml_pa3_task1'); local.mkdir(exist_ok=True)
    if local.is_symlink(): raise RuntimeError('Linked private runtime')
    old = json.loads((root / 'configs/GPU_PREFLIGHT_KIT_READY.json').read_text())
    runtime = old['portable_runtime']
    inv = root / 'runtime' / ('inventory_' + runtime['inventory_sha256'] + '.json')
    assert digest(inv) == runtime['inventory_sha256']
    software = json.loads(inv.read_text())
    assert software['base_directory'] == str(local) and software['python_version'] == '3.12.12'
    restore_archive(root, local, runtime, software['files'], job, state)
    inputs = local / 'gpu_inputs'
    inv = root / str(safe_relative(old['input_inventory_path']))
    assert digest(inv) == old['fixed_inputs']['inventory_sha256']
    restore_archive(root, inputs, old['fixed_inputs'], json.loads(inv.read_text()), job, state)
    progress(state, 'copy and verify new evaluation request inputs')
    extras = {'configs/BASELINE_CPU_READY_V1.json': digest(root / 'configs/BASELINE_CPU_READY_V1.json'),
        ready['protocol_path']: digest(root / ready['protocol_path']),
        'configs/HARNESS_SOURCE_HASHES.json': digest(root / 'configs/HARNESS_SOURCE_HASHES.json'),
        'sources/first_plot_questions.yaml': '215abde5b02811e922b6ac563ff33c490e0dff9ab58439df840f3135e9387173',
        ready['qualitative_path']: ready['qualitative_sha256'], ready['probe_path']: ready['probe_sha256']}
    extras.update({entry['path']: entry['sha256'] for entry in ready['benchmarks'].values()})
    # The old GPU transport contains tokenizer copies under original_model,
    # but intentionally omits the canonical sources/tokenizer directory.
    token_manifest = json.loads((inputs / 'manifests/TOKENIZER_FILES.json').read_text())
    extras.update({'sources/tokenizer/' + name: token_manifest[name]
                  for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json')})
    for rel, expected in extras.items():
        safe_relative(rel); payload = (root / rel).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == expected
        publish_bytes(inputs / rel, payload)
    # Check the exact observed protocol, rather than accepting newly selected settings.
    assert identity(json.loads((inputs / ready['protocol_path']).read_text())) == PROTOCOL
    assets = json.loads((inputs / 'manifests/CPU_ASSETS_READY.json').read_text())
    assert digest(bundle / 'training_core.py') == assets['training_core_sha256']
    assert digest(bundle / 'training_ops.py') == old['files']['training_ops.py']
    result_path = root / 'logs' / (run_id + '_result.json')
    config = {**context, 'run_id': run_id, 'inputs': str(inputs), 'result': str(result_path),
              'implementation_identity': implementation, 'protocol_identity': PROTOCOL}
    config_path = job / 'worker_config.json'; atomic(config_path, config)
    env = os.environ.copy()
    for name in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'UV_SYSTEM_PYTHON'): env.pop(name, None)
    env.update(PYTHONNOUSERSITE='1', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
        HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false',
        CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    print('Frozen environment and inputs restored. Starting strict backward gate, then original baseline.', flush=True)
    result = execute_worker([str(local / 'venv312/bin/python'), '-I', '-u',
        str(bundle / 'baseline_worker.py'), str(config_path)], env, state, result_path,
        root / 'logs' / (run_id + '_session_final.json'), session,
        root / 'logs' / (run_id + '_worker_console.txt'))
    print('Saved result:', result_path, flush=True)
    print('Saved status:', result['status'], flush=True)
    return result
