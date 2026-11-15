"""Restore only needed frozen assets, execute T1-A003, validate the saved gate."""
import json
import math
import os
import shutil
import signal
import subprocess
import time
import uuid
from pathlib import Path

from archive_io import digest, safe_relative
from baseline_runtime import atomic, progress, publish_bytes
from evaluation_records import identity, publish
from launch_baseline import restore_archive

CPU_ID = '98e17cfd9d8c66f395d0a1296100b45bf66bb068062d2e495d1e9597163040c9'
PROMPT_ID = '80961084a25dfdbc535dd2c4a6389ab901295ddb88fdca11646c263faa8ffad9'
CHECKS = ('finite_prefill_logits_within_approved_tolerances', 'all_prefill_next_tokens_agree',
    'all_8_semantic_token_sequences_agree', 'all_8_native_postprocessed_outputs_agree',
    'speedup_at_least_2', 'reserved_memory_headroom_at_least_10_percent')


def validate_result(result, exit_code, run_id):
    if (result.get('run_id') != run_id or result.get('official_training_updates') != 0
            or result.get('official_benchmark_answers_generated') != 0): return False
    status = result.get('status')
    if status == 'TASK1_BATCH8_PILOT_PASSED':
        try:
            return (exit_code == 0 and result.get('approval_id') == 'T1-A003'
                and result.get('conditional_adoption_gate_passed') is True
                and set(result['checks']) == set(CHECKS)
                and all(result['checks'][key] is True for key in CHECKS)
                and len(result['token_agreement_by_prompt']) == 8 and all(result['token_agreement_by_prompt'])
                and len(result['postprocessed_text_agreement_by_prompt']) == 8 and all(result['postprocessed_text_agreement_by_prompt'])
                and math.isfinite(result['model_inference_speedup']) and result['model_inference_speedup'] >= 2
                and math.isfinite(result['headroom_fraction']) and result['headroom_fraction'] >= .1
                and 0 <= result['max_absolute_logit_difference'] <= .1
                and 0 <= result['relative_RMS_logit_difference'] <= .005)
        except (KeyError, TypeError): return False
    if status == 'TASK1_BATCH8_PILOT_REJECTED':
        return (exit_code == 0 and result.get('conditional_adoption_gate_passed') is False
            and set(result.get('checks', {})) == set(CHECKS)
            and not all(result['checks'].values()))
    return (exit_code == 0 and status == 'TASK1_BATCH8_PILOT_INCOMPLETE'
            and result.get('conditional_adoption_gate_passed') is False)


def execute(command, env, context, session):
    root, job = Path(context['root']), Path(context['job'])
    run_id = session['run_id']; worker = None
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
        path = root / 'logs' / f'{run_id}_result.json'
        result = json.loads(path.read_text())
        valid = validate_result(result, code, run_id)
        if valid and result['status'] == 'TASK1_BATCH8_PILOT_PASSED':
            comparison_path = Path(context['journal']) / 'COMPARISON.json'
            comparison = json.loads(comparison_path.read_text())
            assert all(comparison['result']['checks'].values())
            assert comparison['result']['model_inference_speedup'] == result['model_inference_speedup']
            gate = {'approval_id': 'T1-A003', 'CPU_implementation_identity': CPU_ID,
                'GPU_implementation_identity': session['implementation_identity'],
                'source_run': run_id, 'result_path': str(path.relative_to(root)), 'result_sha256': digest(path),
                'comparison_path': str(comparison_path.relative_to(root)), 'comparison_sha256': digest(comparison_path),
                'conditional_adoption_gate_passed': True,
                'new_official_evaluation_protocol_not_yet_published': True}
            publish(root / 'configs' / f'BATCH8_PILOT_GATE_{run_id}.json', gate)
        final = {**session, 'status': result['status'], 'worker_exit_code': code,
            'worker_result_validated': valid, 'result_sha256': digest(path),
            'elapsed_allocation_seconds_at_final_save': time.time() - context['started_wall'],
            'physical_release_verified': False, 'release_request_pending': True}
        final['estimated_units_so_far_with_margin'] = final['elapsed_allocation_seconds_at_final_save'] / 3600 * context['rate'] * 1.1
        publish(root / 'logs' / f'{run_id}_session_final.json', final)
        if not valid: raise RuntimeError('Pilot failed or its saved result was inconsistent; preserved evidence must be reviewed')
        print('Saved pilot result:', path, flush=True)
        return result
    finally:
        if worker is not None and worker.poll() is None:
            try: os.killpg(worker.pid, signal.SIGTERM)
            except ProcessLookupError: pass
            try: worker.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try: os.killpg(worker.pid, signal.SIGKILL)
                except ProcessLookupError: pass
                worker.wait(timeout=3)


def copy_model_file(source, target, expected, state):
    import hashlib
    if target.exists():
        assert not target.is_symlink() and digest(target) == expected
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.with_name(target.name + '.pending_' + uuid.uuid4().hex)
    hasher = hashlib.sha256(); last = time.monotonic()
    with source.open('rb') as src, pending.open('xb') as output:
        for block in iter(lambda: src.read(8 * 1024**2), b''):
            output.write(block); hasher.update(block)
            if time.monotonic() - last > 2:
                progress(state, 'copy original model ' + source.name); last = time.monotonic()
    assert hasher.hexdigest() == expected
    pending.rename(target); assert digest(target) == expected


def run(context, bundle):
    root, job, bundle = Path(context['root']), Path(context['job']), Path(bundle)
    state = context['state']; run_id = job.name
    progress(state, 'verify approved pilot and CPU readiness')
    ready = json.loads((root / 'configs' / f'BATCH8_CPU_CHECKS_{CPU_ID}.json').read_text())
    assert ready['status'] == 'TASK1_BATCH8_CPU_CHECKS_OK'
    assert ready['implementation_identity'] == identity(ready['files']) == CPU_ID
    assert ready['fixed_prompt_identity'] == PROMPT_ID
    assert all(value is True for value in ready['record_checks'].values())
    for key, value in ready['native_CPU_checks'].items():
        if key not in ('scope', 'actual_FP16_L4_pilot_passed'): assert value is True
    assert ready['pilot_limits'] == {'maximum_minutes': 15, 'maximum_estimated_units': .6,
        'rate_margin': 1.1, 'minimum_headroom': .1, 'minimum_speedup': 2.}
    for name, expected in ready['files'].items(): assert digest(bundle / name) == expected, name
    approval = json.loads((root / 'provenance/APPROVAL_T1_A003.json').read_text())
    assert approval['approval_id'] == 'T1-A003' and approval['user_answer'] == 'Approve pilot and conditional adoption (Recommended)'
    proposal = root / 'provenance' / f"D10B_proposal_{approval['proposal_sha256']}.md"
    assert digest(proposal) == approval['proposal_sha256'] == 'aa251f073c0d994ad1f5364978bb4b65c6c67f8ecf8377d73a3e8f91470ed8b6'
    files = {path.name: digest(path) for path in bundle.iterdir() if path.is_file() and path.suffix == '.py'}
    assert digest(job / 'guardian.py') == files['gpu_guardian_R3.py']
    assert digest(job / 'watchdog_evidence.py') == files['watchdog_evidence.py']
    assert digest(job / 'release_coordination.py') == files['release_coordination.py']
    implementation = identity(files)
    session = {**context, 'run_id': run_id, 'status': 'STARTED', 'approval_id': 'T1-A003',
        'implementation_identity': implementation, 'CPU_implementation_identity': CPU_ID,
        'fixed_prompt_identity': PROMPT_ID, 'maximum_minutes': 15, 'maximum_estimated_units': .6,
        'official_benchmark_answers_generated': 0, 'official_training_updates': 0,
        'previous_baseline_shutdown_review': 'Reviewed saved console: no terminal receipt or traceback; cause remains unknown.',
        'guardian_version': 'R3; nonblocking progress and release-intent evidence; parent-death monitoring'}
    publish(root / 'logs' / f'{run_id}_session_started.json', session)
    backup = root / 'execution' / f'pilot_gpu_R3_{implementation}'
    for name in files: publish_bytes(backup / name, (bundle / name).read_bytes())
    publish(backup / 'FILES.json', files)
    old = json.loads((root / 'configs/GPU_PREFLIGHT_KIT_READY.json').read_text())
    runtime = old['portable_runtime']; local = Path('/content/atml_pa3_task1'); local.mkdir(exist_ok=True)
    if local.is_symlink(): raise RuntimeError('Linked private runtime directory')
    inventory = root / 'runtime' / f"inventory_{runtime['inventory_sha256']}.json"
    assert digest(inventory) == runtime['inventory_sha256']
    inventory = json.loads(inventory.read_text())
    assert inventory['base_directory'] == str(local) and inventory['python_version'] == '3.12.12'
    originals = json.loads((root / 'manifests/ORIGINAL_MODEL_FILES.json').read_text())
    frozen = json.loads((root / 'configs/SOURCE_REVISIONS.json').read_text())['model']
    assert originals['model'] == frozen
    required = {name: entry['sha256'] for name, entry in originals['files'].items()}
    tokens = json.loads((root / 'manifests/TOKENIZER_FILES.json').read_text())
    for name, expected in tokens.items():
        if name in required: assert required[name] == expected
        required[name] = expected
    needed = runtime['archive_bytes'] + runtime['uncompressed_file_bytes'] + 2 * 2**30
    needed += sum((root / 'sources/original_model' / name).stat().st_size for name in required)
    assert shutil.disk_usage(local).free >= needed, 'Insufficient runtime disk for the bounded pilot'
    restore_archive(root, local, runtime, inventory['files'], job, state)
    inputs = local / 'pilot_inputs'
    for name, expected in required.items():
        safe_relative(name); assert '/' not in name
        progress(state, 'verify/copy original model file ' + name)
        copy_model_file(root / 'sources/original_model' / name,
                        inputs / 'sources/original_model' / name, expected, state)
    result = root / 'logs' / f'{run_id}_result.json'
    journal = root / 'pilot_runs' / run_id
    config = {**context, 'inputs': str(inputs), 'journal': str(journal), 'run_id': run_id,
        'result': str(result), 'implementation_identity': implementation,
        'model_identity': identity({'model': originals, 'evaluation_dtype': 'float16'}),
        'maximum_prepared_GSM_prompt_tokens': ready['maximum_prepared_GSM_prompt_tokens']}
    context = {**context, 'journal': str(journal)}
    config_path = job / 'worker_config.json'; atomic(config_path, config)
    env = os.environ.copy()
    for name in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'UV_SYSTEM_PYTHON'): env.pop(name, None)
    env.update(PYTHONNOUSERSITE='1', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
        HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false', CUBLAS_WORKSPACE_CONFIG=':4096:8',
        OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    print('Starting bounded original-model pilot: eight fixed non-benchmark prompts, no training.', flush=True)
    return execute([str(local / 'venv312/bin/python'), '-I', '-u', str(bundle / 'batch8_pilot_entry.py'),
                    str(config_path)], env, context, session)
