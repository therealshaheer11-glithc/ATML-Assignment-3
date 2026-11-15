"""Complete the original baseline under the approved fixed batch-eight protocol."""
import gzip
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from archive_io import digest
from baseline_runtime import SoftStop, atomic, boundary, progress, wait_for_parent
from baseline_v2_protocol import NAMESPACE, TASK_ORDER
from evaluation_records import COUNTS, RecordStore, aggregate, identity, publish


def run(config, result):
    import torch
    import importlib.metadata
    import lm_eval
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from stock_evaluation import load_task
    from evaluation_v2_engine import run_chunk
    from evaluation_v2_hflm import EvaluationHFLM
    root, inputs = Path(config['root']), Path(config['inputs'])
    assert torch.__version__ == '2.8.0+cu126' and importlib.metadata.version('lm_eval') == '0.4.9.2'
    assert importlib.metadata.version('transformers') == '4.57.1'
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    assert torch.cuda.device_count() == 1 and torch.cuda.get_device_name(0) == 'NVIDIA L4'
    torch.set_num_threads(2); torch.manual_seed(6304); torch.cuda.manual_seed_all(6304)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    protocol = json.loads((inputs / 'configs/EVALUATION_PROTOCOL_V2_BATCH8.json').read_text())
    assert identity(protocol) == config['protocol_identity']
    assert protocol['generation_batch_size'] == 8 and protocol['likelihood_batch_size'] == 2
    assert protocol['task_order'] == list(TASK_ORDER)
    hashes = json.loads((inputs / 'configs/HARNESS_SOURCE_HASHES.json').read_text())
    package = Path(lm_eval.__file__).parent
    for rel, expected in hashes.items(): assert digest(package / rel) == expected
    ready = json.loads((inputs / 'configs/BASELINE_CPU_READY_V1.json').read_text())
    refs = json.loads((root / NAMESPACE / 'original/REUSED_REFERENCE_MANIFEST.json').read_text())
    assert identity(refs) == config['reference_identity'] and refs['qualitative_count'] == 24 and refs['probe_count'] == 64
    boundary(config, 'load unchanged original FP16 model for batch-eight baseline')
    print('Loading original FP16 model; old qualitative responses and FP32 probes are reused by reference.', flush=True)
    model = AutoModelForCausalLM.from_pretrained(inputs / 'sources/original_model',
        local_files_only=True, dtype=torch.float16, attn_implementation='sdpa', low_cpu_mem_usage=True)
    model.requires_grad_(False).eval().to('cuda'); model.config.use_cache = True
    tokenizer = AutoTokenizer.from_pretrained(inputs / 'sources/original_model', local_files_only=True)
    audit = {'condition': 'original', 'model_identity': config['model_identity'],
        'protocol_identity': config['protocol_identity'], 'gpu_implementation_identity': config['implementation_identity']}
    lm = EvaluationHFLM(pretrained=model, tokenizer=tokenizer, dtype='float16', batch_size=8,
        enable_thinking=False, add_bos_token=False, job_config=config, audit_contract=audit,
        journal=root / NAMESPACE / 'original/raw_generation_batches')
    assert lm.max_length == protocol['model_max_context_tokens']
    assert all(p.dtype == torch.float16 for p in model.parameters())
    assert sum(p.numel() for p in model.parameters()) == 596049920 and not model.training
    store = RecordStore(root / NAMESPACE, 'original', config['model_identity'], config['protocol_identity'])
    metrics = {}
    for name in TASK_ORDER:
        boundary(config, 'prepare frozen task ' + name)
        entry = ready['benchmarks'][name]; path = inputs / entry['path']
        assert digest(path) == entry['sha256']
        with gzip.open(path, 'rt') as stream: rows = [json.loads(line) for line in stream]
        assert [r['source_index'] for r in rows] == list(range(COUNTS[name]))
        task, _, yaml_path = load_task(inputs, name)
        assert digest(yaml_path) == entry['stock_yaml_sha256'] and task.config.num_fewshot == 0
        assert all(lm.tok_encode(row['context']) == row['prompt_input_ids'] for row in rows)
        chunk_size = 25 if name == 'gsm8k_cot' else 100
        completed = []
        print('Baseline V2 task:', name, 'required questions:', len(rows), flush=True)
        for start in range(0, len(rows), chunk_size):
            boundary(config, name + ' fixed chunk ' + str(start // chunk_size + 1))
            chunk = run_chunk(task, name, rows[start:start + chunk_size], store, lm.infer,
                root / NAMESPACE / 'original/raw_likelihood_chunks', audit)
            completed.extend(chunk)
            result['benchmark_completed'][name] = len(completed)
            progress(config['state'], f'saved V2 {name}: {len(completed)}/{len(rows)}')
            print(f'{name}: {len(completed)}/{len(rows)} V2 answers saved.', flush=True)
        metrics[name] = aggregate(completed, name)
        publish(root / NAMESPACE / 'original' / f'{name}_COMPLETE.json',
                {'audit': audit, 'count': len(rows), 'metrics': metrics[name]})
    summary = {'status': 'TASK1_ORIGINAL_BASELINE_V2_COMPLETE', 'audit': audit,
        'benchmark_counts': COUNTS, 'metrics': metrics, 'references': refs,
        'qualitative_responses': 24, 'FP32_activation_probes': 64,
        'human_qualitative_annotation_pending': True, 'official_training_updates': 0,
        'preliminary_batch_one_GSM_outputs_used_in_main_metrics': False}
    publish(root / NAMESPACE / 'original/BASELINE_COMPLETE.json', summary)
    result['baseline_summary'] = summary


def main(config_path):
    config = json.loads(Path(config_path).read_text()); wait_for_parent(config['state'])
    started = time.monotonic(); code = 0
    result = {'status': 'TASK1_BASELINE_V2_FAILED', 'run_id': config['run_id'],
        'official_training_updates': 0, 'benchmark_completed': {name: 0 for name in COUNTS},
        'qualitative_responses_reused': 24, 'FP32_probe_files_reused': 64}
    try:
        run(config, result); result['status'] = 'TASK1_BASELINE_V2_COMPLETE_SAVED'
    except SoftStop as error:
        result.update(status='TASK1_BASELINE_V2_PARTIAL_SAVED', stop_reason=str(error),
            resume='Use the exact same V2 package; saved physical groups and scored examples are reused.')
    except BaseException as error:
        code = 1; result.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    result['elapsed_worker_seconds'] = time.monotonic() - started
    try:
        import torch
        result['GPU_memory'] = {'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'total_gpu_bytes': torch.cuda.get_device_properties(0).total_memory}
    except Exception: result['GPU_memory'] = 'not available'
    publish(config['result'], result)
    print(json.dumps(result, indent=2), flush=True)
    state = json.loads(Path(config['state']).read_text())
    state.update(status='worker_complete', worker_completed_wall=time.time(), progress_wall=time.time())
    atomic(config['state'], state)
    return code


if __name__ == '__main__': sys.exit(main(sys.argv[1]))
