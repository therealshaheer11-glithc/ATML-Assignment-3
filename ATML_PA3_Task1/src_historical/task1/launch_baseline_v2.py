"""Adopt only a passed pilot, preserve old evidence, execute bounded V2 baseline."""
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from archive_io import digest, safe_relative
from baseline_runtime import atomic, progress, publish_bytes
from baseline_v2_protocol import prepare, NAMESPACE
from evaluation_records import COUNTS, identity, publish
from launch_baseline import restore_archive
from launch_batch8_pilot import copy_model_file


def valid_result(value, code, session):
    if code != 0 or value.get('run_id') != session['run_id'] or value.get('official_training_updates') != 0:
        return False
    if value.get('qualitative_responses_reused') != 24 or value.get('FP32_probe_files_reused') != 64:
        return False
    counts = value.get('benchmark_completed', {})
    if set(counts) != set(COUNTS) or not all(type(n) is int and 0 <= n <= COUNTS[k] for k, n in counts.items()):
        return False
    if value.get('status') == 'TASK1_BASELINE_V2_PARTIAL_SAVED': return True
    summary = value.get('baseline_summary', {})
    return (value.get('status') == 'TASK1_BASELINE_V2_COMPLETE_SAVED' and counts == COUNTS
        and summary.get('status') == 'TASK1_ORIGINAL_BASELINE_V2_COMPLETE'
        and summary.get('benchmark_counts') == COUNTS
        and summary.get('audit', {}).get('protocol_identity') == session['protocol_identity']
        and summary.get('audit', {}).get('model_identity') == session['model_identity'])


def execute(command, env, context, session):
    root = Path(context['root']); run_id = session['run_id']; worker = None
    try:
        with (root / 'logs' / f'{run_id}_worker_console.txt').open('x') as log:
            worker = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, start_new_session=True)
            state = json.loads(Path(context['state']).read_text())
            state.update(worker_pid=worker.pid, status='running', progress_wall=time.time())
            atomic(context['state'], state)
            for line in worker.stdout:
                print(line, end='', flush=True); log.write(line); log.flush()
            code = worker.wait()
        result_path = root / 'logs' / f'{run_id}_result.json'
        value = json.loads(result_path.read_text()); valid = valid_result(value, code, session)
        if valid and value['status'] == 'TASK1_BASELINE_V2_COMPLETE_SAVED':
            summary = json.loads((root / NAMESPACE / 'original/BASELINE_COMPLETE.json').read_text())
            assert summary == value['baseline_summary']
        publish(root / 'logs' / f'{run_id}_session_final.json', {**session,
            'status': value['status'], 'worker_exit_code': code, 'worker_result_validated': valid,
            'result_sha256': digest(result_path), 'elapsed_allocation_seconds_at_final_save': time.time() - context['started_wall'],
            'estimated_units_so_far_with_margin': (time.time() - context['started_wall']) / 3600 * context['rate'] * 1.1,
            'release_request_pending': True, 'physical_release_verified': False})
        if not valid: raise RuntimeError('Baseline V2 worker failed or its saved result was inconsistent; review preserved evidence')
        print('Saved baseline V2 result:', result_path, flush=True)
        return value
    finally:
        if worker is not None and worker.poll() is None:
            try: os.killpg(worker.pid, signal.SIGTERM)
            except ProcessLookupError: pass
            try: worker.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try: os.killpg(worker.pid, signal.SIGKILL)
                except ProcessLookupError: pass
                worker.wait(timeout=3)


def run(context, bundle):
    root, job, bundle = Path(context['root']), Path(context['job']), Path(bundle)
    state = context['state']; run_id = job.name
    progress(state, 'verify passed pilot, freeze V2 protocol, verify reusable references')
    adopted = prepare(root)
    files = {p.name: digest(p) for p in bundle.iterdir() if p.is_file() and p.suffix == '.py'}
    assert digest(job / 'guardian.py') == files['gpu_guardian_R3.py']
    assert digest(job / 'watchdog_evidence.py') == files['watchdog_evidence.py']
    assert digest(job / 'release_coordination.py') == files['release_coordination.py']
    implementation = identity(files)
    session = {**context, 'run_id': run_id, 'status': 'STARTED', 'approval_id': 'T1-A003',
        'implementation_identity': implementation, 'protocol_identity': adopted['protocol_identity'],
        'model_identity': adopted['model_identity'], 'adoption_evidence': adopted['adoption'],
        'reference_identity': identity(adopted['references']), 'result_namespace': NAMESPACE,
        'official_training_updates': 0, 'original_batch_one_results_preserved': True}
    publish(root / 'logs' / f'{run_id}_session_started.json', session)
    backup = root / 'execution' / f'baseline_batch8_V2_{implementation}'
    for name in files: publish_bytes(backup / name, (bundle / name).read_bytes())
    publish(backup / 'FILES.json', files)
    old = json.loads((root / 'configs/GPU_PREFLIGHT_KIT_READY.json').read_text())
    runtime = old['portable_runtime']; local = Path('/content/atml_pa3_task1'); local.mkdir(exist_ok=True)
    if local.is_symlink(): raise RuntimeError('Linked private software directory')
    inventory_path = root / 'runtime' / f"inventory_{runtime['inventory_sha256']}.json"
    assert digest(inventory_path) == runtime['inventory_sha256']
    inventory = json.loads(inventory_path.read_text()); assert inventory['base_directory'] == str(local)
    assert inventory['python_version'] == '3.12.12'
    originals = json.loads((root / 'manifests/ORIGINAL_MODEL_FILES.json').read_text())
    needed = runtime['archive_bytes'] + runtime['uncompressed_file_bytes'] + 2 * 2**30
    needed += sum(entry['bytes'] for entry in originals['files'].values())
    assert shutil.disk_usage(local).free >= needed, 'Insufficient runtime disk for verified baseline restoration'
    restore_archive(root, local, runtime, inventory['files'], job, state)
    inputs = local / 'baseline_v2_inputs'
    frozen = json.loads((root / 'configs/SOURCE_REVISIONS.json').read_text())['model']
    assert originals['model'] == frozen
    required = {name: entry['sha256'] for name, entry in originals['files'].items()}
    for name, expected in json.loads((root / 'manifests/TOKENIZER_FILES.json').read_text()).items():
        if name in required: assert required[name] == expected
        required[name] = expected
    for name, expected in required.items():
        safe_relative(name); assert '/' not in name
        progress(state, 'copy original model file ' + name)
        copy_model_file(root / 'sources/original_model' / name,
            inputs / 'sources/original_model' / name, expected, state)
    ready = adopted['ready']
    extras = {'configs/BASELINE_CPU_READY_V1.json': digest(root / 'configs/BASELINE_CPU_READY_V1.json'),
        'configs/EVALUATION_PROTOCOL_V2_BATCH8.json': digest(root / 'configs/EVALUATION_PROTOCOL_V2_BATCH8.json'),
        'configs/HARNESS_SOURCE_HASHES.json': digest(root / 'configs/HARNESS_SOURCE_HASHES.json'),
        'manifests/BENCHMARK_FILES.json': digest(root / 'manifests/BENCHMARK_FILES.json')}
    for entry in ready['benchmarks'].values(): extras[entry['path']] = entry['sha256']
    raw = json.loads((root / 'manifests/BENCHMARK_FILES.json').read_text())
    for entry in raw.values(): extras[entry['path']] = entry['sha256']
    for rel, expected in extras.items():
        safe_relative(rel); progress(state, 'copy frozen evaluation input ' + rel)
        payload = (root / rel).read_bytes()
        import hashlib
        assert hashlib.sha256(payload).hexdigest() == expected
        publish_bytes(inputs / rel, payload)
    result = root / 'logs' / f'{run_id}_result.json'
    config = {**context, 'run_id': run_id, 'inputs': str(inputs), 'result': str(result),
        'implementation_identity': implementation, 'protocol_identity': adopted['protocol_identity'],
        'model_identity': adopted['model_identity'], 'reference_identity': identity(adopted['references'])}
    config_path = job / 'worker_config.json'; atomic(config_path, config)
    env = os.environ.copy()
    for name in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'UV_SYSTEM_PYTHON'): env.pop(name, None)
    env.update(PYTHONNOUSERSITE='1', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
        HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false', CUBLAS_WORKSPACE_CONFIG=':4096:8',
        OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    return execute([str(local / 'venv312/bin/python'), '-I', '-u', str(bundle / 'baseline_v2_worker.py'),
                    str(config_path)], env, context, session)
