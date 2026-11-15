# Task 1 - Block 12: CPU-only recovery and measured baseline speed audit.
# Paste the entire file into one CPU Colab cell. No ZIP or package installation.
import datetime
import gzip
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import threading
import time
import uuid
from pathlib import Path

PROTOCOL = '2b0cdf3ca8dd889e6e63996af55aba94672110f5e0d79e7ca1e843434d7cf297'
EXPECTED_IMPLEMENTATION = 'd5c5b51f6107bd08adb362f878b528d6b1c92738e07e432405ddfeb7fc3ff56e'
COUNTS = {'gsm8k_cot': 1319, 'hellaswag': 10042, 'boolq': 3270, 'arc_easy': 2376}


def encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False,
                       separators=(',', ':'), allow_nan=False) + '\n').encode()


def identity(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_path(root, rel):
    root = Path(root)
    path = root / rel
    if Path(rel).is_absolute() or '..' in Path(rel).parts or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Unexpected saved relative path')
    return path


def cpu_guard():
    evidence = []
    if os.environ.get('COLAB_GPU', '0').lower().strip() not in ('', '0', 'false', 'none'):
        evidence.append('GPU allocation flag')
    if list(Path('/dev').glob('nvidia[0-9]*')):
        evidence.append('GPU device')
    if any(os.environ.get(k) for k in ('COLAB_TPU_ADDR', 'TPU_NAME', 'TPU_WORKER_ID')) or os.environ.get('PJRT_DEVICE', '').upper() == 'TPU':
        evidence.append('TPU allocation')
    if shutil.which('nvidia-smi'):
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                                    capture_output=True, text=True, timeout=10)
            if result.returncode or result.stdout.strip(): evidence.append('GPU present or status uncertain')
        except Exception:
            evidence.append('GPU status uncertain')
    if evidence:
        print('CPU required; requesting release:', evidence, flush=True)
        def release():
            try:
                from google.colab import runtime
                runtime.unassign()
            except Exception as error:
                print('Release request:', type(error).__name__, flush=True)
        thread = threading.Thread(target=release, daemon=True); thread.start(); thread.join(10)
        raise RuntimeError('Select CPU; end this notebook\'s remaining L4 session through Manage sessions if active.')


def validate_raw(path, expected_audit, allowed_prompts, eos_ids):
    record = json.loads(path.read_text())
    checksum = record.pop('content_sha256')
    assert identity(record) == checksum, 'Raw-token checksum differs'
    contract = record['contract']; assert contract['audit'] == expected_audit
    assert path.stem == identity(contract), 'Raw cache key differs'
    prefix = contract['input_ids']; cap = contract['max_length'] - len(prefix)
    assert (tuple(prefix), cap) in allowed_prompts, 'Unknown prepared generation prompt'
    assert contract['generation_kwargs']['do_sample'] is False
    sequence = record['sequence_ids']; assert len(sequence) == 1 and sequence[0][:len(prefix)] == prefix
    assert all(type(t) is int and t >= 0 for t in sequence[0])
    new = sequence[0][len(prefix):]; meta = record['metadata']
    assert len(new) <= cap and meta['generated_tokens'] == len(new) and meta['max_new_tokens'] == cap
    assert meta['at_token_cap'] == (len(new) == cap)
    assert meta['ended_with_eos'] == bool(new and new[-1] in eos_ids)
    assert meta['truncated'] == bool(meta['at_token_cap'] and not meta['ended_with_eos'] and not meta['stock_stop_string_present'])
    assert math.isfinite(meta['original_inference_seconds']) and meta['original_inference_seconds'] >= 0
    return record, cap


def audit(root, implementation=EXPECTED_IMPLEMENTATION):
    root = Path(root); errors = []
    def read(rel): return json.loads(safe_path(root, rel).read_text())
    ready = read('configs/BASELINE_CPU_READY_V1.json')
    protocol = read(ready['protocol_path'])
    assert identity(protocol) == ready['protocol_identity'] == PROTOCOL
    original = read('manifests/ORIGINAL_MODEL_FILES.json')
    audit_contract = {'condition': 'original', 'protocol_identity': PROTOCOL,
        'model_identity': identity({'model': original, 'evaluation_dtype': 'float16'}),
        'gpu_implementation_identity': implementation}
    store_contract = {key: audit_contract[key] for key in ('condition', 'model_identity', 'protocol_identity')}
    assert read('results/original/CONTRACT.json') == store_contract
    sources, counts = {}, {}
    for name, count in COUNTS.items():
        entry = ready['benchmarks'][name]; path = safe_path(root, entry['path'])
        assert sha(path) == entry['sha256']
        with gzip.open(path, 'rt') as stream: rows = [json.loads(line) for line in stream]
        assert [r['source_index'] for r in rows] == list(range(count))
        sources[name] = rows
        saved = {}
        for path in (root / 'results/original' / name).glob('*.json'):
            try:
                row = json.loads(path.read_text()); index = row['source_index']
                assert type(index) is int and 0 <= index < count and path.stem == f'{index:05}'
                assert row['task'] == name and row['status'] == 'complete' and row['contract'] == store_contract
                request = {'source_index': index, 'requests': rows[index]['requests']}
                assert row['request_identity'] == identity(request)
                assert index not in saved and all(v in (0., 1.) for v in row['scores'].values())
                assert protocol['primary_metrics'][name] in row['scores']
                saved[index] = row
            except Exception as error:
                errors.append(str(path.relative_to(root)) + ': ' + type(error).__name__ + ': ' + str(error))
        counts[name] = {'scored_answers_validated': len(saved), 'required': count}
        sources[name + '_saved'] = saved
    print('Scored answer counts:', counts, flush=True)
    qpath = safe_path(root, ready['qualitative_path']); assert sha(qpath) == ready['qualitative_sha256']
    prompts = json.loads(qpath.read_text()); assert len(prompts) == 24
    allowed = {(tuple(row['prompt_input_ids']), 1024) for row in sources['gsm8k_cot']}
    allowed.update((tuple(row['prompt_input_ids']), 512) for row in prompts)
    generation_config = root / 'sources/original_model/generation_config.json'
    assert sha(generation_config) == original['files']['generation_config.json']['sha256']
    eos_ids = json.loads(generation_config.read_text())['eos_token_id']
    eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
    raw_by_rel, raw_groups = {}, {1024: [], 512: []}
    for path in (root / 'results/original/raw_generation').glob('*.json'):
        try:
            raw, cap = validate_raw(path, audit_contract, allowed, eos_ids)
            raw_by_rel[str(path.relative_to(root))] = raw; raw_groups[cap].append(raw)
        except Exception as error:
            errors.append(str(path.relative_to(root)) + ': ' + type(error).__name__ + ': ' + str(error))
    for row in sources['gsm8k_cot_saved'].values():
        try:
            assert len(row['metadata']['requests']) == 1
            raw = raw_by_rel[row['metadata']['requests'][0]['raw_record']]
            assert raw['contract']['input_ids'] == sources['gsm8k_cot'][row['source_index']]['prompt_input_ids']
        except Exception as error:
            errors.append('GSM scored/raw link: ' + type(error).__name__ + ': ' + str(error))
    qualitative_count = 0
    for prompt in prompts:
        path = root / 'results/original/qualitative' / (prompt['id'] + '.json')
        if not path.exists(): continue
        try:
            row = json.loads(path.read_text())
            assert row['contract'] == {'audit': audit_contract, 'prepared_prompt_identity': identity(prompt)}
            assert row['id'] == prompt['id'] and row['messages'] == prompt['messages']
            raw = raw_by_rel[row['metadata']['raw_record']]
            assert raw['contract']['input_ids'] == prompt['prompt_input_ids']
            qualitative_count += 1
        except Exception as error:
            errors.append('Qualitative ' + prompt['id'] + ': ' + type(error).__name__ + ': ' + str(error))
    probe_count = 0
    path = root / 'results/original/ACTIVATION_PROBES_COMPLETE.json'
    if path.exists():
        try:
            manifest = json.loads(path.read_text()); assert manifest['audit'] == audit_contract and manifest['count'] == 64
            ppath = safe_path(root, ready['probe_path']); assert sha(ppath) == ready['probe_sha256']
            probes = {(p['task'], p['source_index']): p for p in json.loads(ppath.read_text())}
            seen = set()
            for entry in manifest['entries']:
                key = (entry['task'], entry['source_index']); assert key in probes and key not in seen
                tensor_path = safe_path(root, entry['path']); assert sha(tensor_path) == entry['sha256']
                meta = json.loads(tensor_path.with_suffix('.json').read_text())
                assert meta['sha256'] == entry['sha256']
                assert meta['contract'] == {'audit': audit_contract, 'prompt_identity': identity(probes[key]), 'forward_dtype': 'float32'}
                seen.add(key)
            assert seen == set(probes); probe_count = len(seen)
        except Exception as error:
            errors.append('Activation probe manifest/bytes: ' + type(error).__name__ + ': ' + str(error))
    stats = {}
    for cap, group in raw_groups.items():
        durations = [r['metadata']['original_inference_seconds'] for r in group]
        stats['gsm8k' if cap == 1024 else 'qualitative'] = {
            'unique_raw_generations_validated': len(group),
            'generated_tokens': sum(r['metadata']['generated_tokens'] for r in group),
            'at_cap_count': sum(r['metadata']['at_token_cap'] for r in group),
            'truncated_flag_count': sum(r['metadata']['truncated'] for r in group),
            'total_recorded_model_inference_seconds': sum(durations),
            'mean_model_inference_seconds': statistics.mean(durations) if durations else None,
            'median_model_inference_seconds': statistics.median(durations) if durations else None}
    sessions = [json.loads(p.read_text()) for p in (root / 'logs').glob('task1_baseline_*_session_started.json')]
    sessions = [s for s in sessions if s.get('implementation_identity') == implementation]
    latest = max(sessions, key=lambda s: s['started_wall']) if sessions else None
    terminal = {'worker_result_found': False, 'final_parent_session_found': False,
                'stop_reason': 'Not established: no final receipt observed in the supplied console.'}
    if latest:
        run = latest['run_id']
        for suffix, key in (('_result.json', 'worker_result'), ('_session_final.json', 'final_parent_session')):
            path = root / 'logs' / (run + suffix)
            if path.exists():
                value = json.loads(path.read_text()); terminal[key + '_found'] = True
                terminal[key] = {k: value[k] for k in ('status', 'stop_reason', 'error_type', 'error',
                    'worker_exit_code', 'worker_result_validated', 'elapsed_worker_seconds') if k in value}
                if key == 'worker_result': terminal['stop_reason'] = value.get('stop_reason', value.get('error', value['status']))
        terminal['bootstrap_error_found'] = (root / 'logs' / (run + '_bootstrap_error.txt')).exists()
        gate = root / 'logs' / (run + '_strict_gate.json')
        terminal['strict_gate_receipt'] = json.loads(gate.read_text()) if gate.exists() else None
        pk = datetime.timezone(datetime.timedelta(hours=5))
        terminal['scheduled_soft_stop_Pakistan'] = datetime.datetime.fromtimestamp(latest['soft_wall'], pk).isoformat()
        terminal['scheduled_release_timer_Pakistan'] = datetime.datetime.fromtimestamp(latest['deadline_wall'], pk).isoformat()
    if counts['gsm8k_cot']['scored_answers_validated']:
        if qualitative_count != 24 or probe_count != 64:
            errors.append('GSM scored records exist, but the earlier qualitative/probe stages lack complete validated durable coverage.')
    if terminal.get('worker_result', {}).get('status') == 'TASK1_BASELINE_COMPLETE_SAVED':
        if any(row['scored_answers_validated'] != row['required'] for row in counts.values()):
            errors.append('Terminal receipt claims complete baseline but durable validated benchmark coverage is incomplete.')
    gsm_raw_prompts = {tuple(r['contract']['input_ids']) for r in raw_groups[1024]}
    cached_gsm_cases = sum(tuple(row['prompt_input_ids']) in gsm_raw_prompts for row in sources['gsm8k_cot'])
    estimate = None
    mean = stats['gsm8k']['mean_model_inference_seconds']
    if mean is not None and latest:
        seconds = (1319 - cached_gsm_cases) * mean
        estimate = {'remaining_GSM_answers_without_cached_tokens': 1319 - cached_gsm_cases,
            'rough_remaining_baseline_GSM_inference_hours': seconds / 3600,
            'rough_remaining_baseline_GSM_units_at_recorded_rate': seconds / 3600 * latest['displayed_units_per_hour'],
            'recorded_L4_units_per_hour': latest['displayed_units_per_hour'],
            'limitations': 'Inference-only extrapolation from observed prompts; excludes restores, saves, other tasks and training. Future prompt/condition lengths may differ.'}
    return {'status': 'TASK1_CPU_RECOVERY_AUDIT_OK' if not errors else 'TASK1_CPU_RECOVERY_REQUIRES_REVIEW',
        'scored_answers': counts, 'qualitative_responses_validated': qualitative_count,
        'FP32_probe_files_hash_validated': probe_count, 'raw_generation_statistics': stats,
        'GSM_answers_with_reusable_raw_tokens': cached_gsm_cases, 'latest_run_evidence': terminal,
        'inference_only_estimate': estimate, 'integrity_errors': errors,
        'GPU_work_run': False, 'new_model_inference_run': False, 'results_modified': False,
        'scores_recomputed': False, 'physical_GPU_deallocation_independently_verified': False,
        'next': 'Review this audit and compute cost before another GPU run; do not change prompts, cap or batch size silently.'}


def main():
    cpu_guard()
    from google.colab import drive
    drive.mount('/content/drive')
    root = Path('/content/drive/MyDrive/ATML-PA-3/Task1')
    print('CPU recovery: verifying saved answers, token records, probes and terminal receipts...', flush=True)
    result = audit(root)
    path = root / 'logs' / ('block12_cpu_recovery_' + uuid.uuid4().hex + '.json')
    data = encoded(result); path.write_bytes(data); assert path.read_bytes() == data
    print(json.dumps(result, indent=2), flush=True)
    print('Saved audit:', path, flush=True)
    print('Keep CPU selected and send the complete JSON summary.', flush=True)


if __name__ == '__main__': main()
