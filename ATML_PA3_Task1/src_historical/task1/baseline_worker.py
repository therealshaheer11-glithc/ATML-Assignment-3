"""Original-model baseline and fixed FP32 probes; no official training updates."""
import gc
import gzip
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_runtime import (SoftStop, atomic, boundary, progress, publish_bytes, wait_for_parent)
from evaluation_records import COUNTS, RecordStore, aggregate, identity, publish


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''): h.update(block)
    return h.hexdigest()


def probe_cache(config, ready, audit):
    import torch
    from safetensors.torch import load_file, save
    from transformers import AutoModelForCausalLM
    root, inputs = Path(config['root']), Path(config['inputs'])
    prompts_path = inputs / ready['probe_path']
    assert sha(prompts_path) == ready['probe_sha256']
    prompts = json.loads(prompts_path.read_text()); assert len(prompts) == 64
    entries, model = [], None
    try:
        for prompt in prompts:
            boundary(config, 'original FP32 activation probe')
            stem = prompt['task'] + '_' + str(prompt['source_index'])
            path = root / 'results/original/activation_probes' / (stem + '.safetensors')
            meta_path = path.with_suffix('.json')
            contract = {'audit': audit, 'prompt_identity': identity(prompt), 'forward_dtype': 'float32'}
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                assert meta['contract'] == contract and sha(path) == meta['sha256']
                tensors = load_file(path)
                assert set(tensors) == {'layer_0', 'layer_13', 'layer_27'}
                assert all(v.dtype == torch.float32 and list(v.shape) == meta['shapes'][key]
                           and bool(torch.isfinite(v).all()) for key, v in tensors.items())
            else:
                if model is None:
                    boundary(config, 'load fresh original FP32 model for probes')
                    model = AutoModelForCausalLM.from_pretrained(inputs / 'sources/original_model',
                        local_files_only=True, dtype=torch.float32, attn_implementation='sdpa', low_cpu_mem_usage=True)
                    model.requires_grad_(False).eval().to('cuda'); model.config.use_cache = False
                    assert all(p.dtype == torch.float32 for p in model.parameters())
                tensors, handles = {}, []
                for layer in (0, 13, 27):
                    def hook(module, args, layer=layer):
                        tensors['layer_' + str(layer)] = args[0][0, -1].detach().float().cpu().contiguous()
                    handles.append(model.get_submodule('model.layers.' + str(layer) + '.mlp.down_proj').register_forward_pre_hook(hook))
                try:
                    ids = torch.tensor([prompt['prompt_input_ids']], device='cuda', dtype=torch.long)
                    with torch.inference_mode():
                        model.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
                    torch.cuda.synchronize()
                finally:
                    for handle in handles: handle.remove()
                assert set(tensors) == {'layer_0', 'layer_13', 'layer_27'}
                assert all(v.dtype == torch.float32 and v.shape == (model.config.intermediate_size,)
                           and bool(torch.isfinite(v).all()) for v in tensors.values())
                progress(config['state'], 'save original FP32 probe', saving=True)
                digest = publish_bytes(path, save(tensors))
                meta = {'contract': contract, 'sha256': digest,
                        'shapes': {key: list(value.shape) for key, value in tensors.items()},
                        'task': prompt['task'], 'source_index': prompt['source_index'],
                        'last_nonpadding_prompt_token': True, 'answer_continuation_used': False}
                publish(meta_path, meta)
            entries.append({'task': prompt['task'], 'source_index': prompt['source_index'],
                'path': str(path.relative_to(root)), 'sha256': meta['sha256']})
        manifest = {'audit': audit, 'count': 64, 'entries': entries,
                    'same_original_activations_for_all_methods': True, 'dtype': 'float32'}
        publish(root / 'results/original/ACTIVATION_PROBES_COMPLETE.json', manifest)
        return manifest
    finally:
        del model
        gc.collect(); torch.cuda.empty_cache()


def run_baseline(config, ready, audit, result):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from lm_eval.api.instance import Instance
    from baseline_hflm import AuditedHFLM
    from evaluation_engine import run_chunk
    from stock_evaluation import load_task
    root, inputs = Path(config['root']), Path(config['inputs'])
    boundary(config, 'load fresh original FP16 model for baseline')
    model = AutoModelForCausalLM.from_pretrained(inputs / 'sources/original_model',
        local_files_only=True, dtype=torch.float16, attn_implementation='sdpa', low_cpu_mem_usage=True)
    model.requires_grad_(False).eval().to('cuda'); model.config.use_cache = True
    assert all(p.dtype == torch.float16 for p in model.parameters())
    tokenizer = AutoTokenizer.from_pretrained(inputs / 'sources/original_model', local_files_only=True)
    lm = AuditedHFLM(pretrained=model, tokenizer=tokenizer, batch_size=1, dtype='float16',
        enable_thinking=False, add_bos_token=False, job_config=config, audit_contract=audit)
    protocol = json.loads((inputs / ready['protocol_path']).read_text())
    assert lm.max_length == protocol['model_max_context_tokens']
    assert sum(p.numel() for p in model.parameters()) == 596049920
    assert not model.training
    store = RecordStore(root / 'results', 'original', audit['model_identity'], audit['protocol_identity'])
    # Finish the small qualitative set before the larger benchmark run.
    qpath = inputs / ready['qualitative_path']; assert sha(qpath) == ready['qualitative_sha256']
    qualitative = json.loads(qpath.read_text()); assert len(qualitative) == 24
    qual_entries = []
    for row in qualitative:
        path = root / 'results/original/qualitative' / (row['id'] + '.json')
        contract = {'audit': audit, 'prepared_prompt_identity': identity(row)}
        if path.exists():
            saved = json.loads(path.read_text()); assert saved['contract'] == contract
        else:
            boundary(config, 'qualitative prompt ' + row['id'])
            assert lm.tok_encode(row['context']) == row['prompt_input_ids']
            req = Instance(request_type='generate_until', doc={}, idx=0,
                arguments=(row['context'], {'max_gen_toks': 512, 'do_sample': False,
                                           'until': [tokenizer.eos_token]}),
                metadata=('qualitative', row['id'], 1))
            outputs, metadata = lm.infer('generate_until', [req])
            saved = {'contract': contract, 'id': row['id'], 'question_id': row['question_id'],
                'format': row['format'], 'messages': row['messages'], 'response': outputs[0],
                'metadata': metadata[0], 'human_annotation_required': row['annotate'],
                'human_annotation_complete': False}
            progress(config['state'], 'save qualitative answer', saving=True); publish(path, saved)
        qual_entries.append({'id': row['id'], 'path': str(path.relative_to(root)), 'sha256': sha(path)})
        result['qualitative_completed'] = len(qual_entries)
    publish(root / 'results/original/QUALITATIVE_COMPLETE.json',
            {'audit': audit, 'count': 24, 'entries': qual_entries, 'human_annotation_pending': True})
    all_metrics = {}
    for name, count in COUNTS.items():
        boundary(config, 'load prepared benchmark ' + name)
        entry = ready['benchmarks'][name]; path = inputs / entry['path']
        assert sha(path) == entry['sha256']
        with gzip.open(path, 'rt') as stream: rows = [json.loads(line) for line in stream]
        assert [r['source_index'] for r in rows] == list(range(count))
        task, _, yaml_path = load_task(inputs, name)
        assert sha(yaml_path) == entry['stock_yaml_sha256']
        # Audit native tokenization against every frozen prompt before inference.
        assert all(lm.tok_encode(row['context']) == row['prompt_input_ids'] for row in rows)
        chunk = 25 if name == 'gsm8k_cot' else 100
        completed = []
        for start in range(0, count, chunk):
            boundary(config, name + ' chunk ' + str(start // chunk + 1))
            completed.extend(run_chunk(task, name, rows[start:start + chunk], store, lm.infer))
            result['benchmark_completed'][name] = len(completed)
            progress(config['state'], 'saved ' + name + ': ' + str(len(completed)) + '/' + str(count))
            print(name + ': ' + str(len(completed)) + '/' + str(count) + ' answers saved.', flush=True)
        all_metrics[name] = aggregate(completed, name)
        publish(root / 'results/original' / (name + '_COMPLETE.json'),
                {'audit': audit, 'count': count, 'metrics': all_metrics[name]})
    summary = {'status': 'TASK1_ORIGINAL_BASELINE_COMPLETE', 'audit': audit,
        'benchmark_counts': COUNTS, 'metrics': all_metrics,
        'qualitative_responses': 24, 'activation_probes': 64,
        'human_qualitative_annotation_pending': True, 'official_training_updates': 0}
    publish(root / 'results/original/BASELINE_COMPLETE.json', summary)
    result['baseline_summary'] = summary


def main(config_path):
    config = json.loads(Path(config_path).read_text())
    wait_for_parent(config['state'])
    result = {'status': 'TASK1_BASELINE_FAILED', 'run_id': config['run_id'],
        'official_training_updates': 0, 'qualitative_completed': 0,
        'benchmark_completed': {name: 0 for name in COUNTS}, 'activation_probes_completed': 0}
    started = time.monotonic(); code = 0
    try:
        import torch
        import importlib.metadata
        torch.set_num_threads(2)
        assert torch.cuda.device_count() == 1 and 'L4' in torch.cuda.get_device_name(0)
        assert torch.__version__ == '2.8.0+cu126' and importlib.metadata.version('lm_eval') == '0.4.9.2'
        assert importlib.metadata.version('transformers') == '4.57.1'
        assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.manual_seed(6304); torch.cuda.manual_seed_all(6304)
        root, inputs = Path(config['root']), Path(config['inputs'])
        ready = json.loads((inputs / 'configs/BASELINE_CPU_READY_V1.json').read_text())
        protocol = json.loads((inputs / ready['protocol_path']).read_text())
        assert identity(protocol) == ready['protocol_identity'] == config['protocol_identity']
        import lm_eval
        manifest = inputs / 'configs/HARNESS_SOURCE_HASHES.json'
        assert sha(manifest) == protocol['harness_source_manifest_sha256']
        package = Path(lm_eval.__file__).parent
        for rel, expected in json.loads(manifest.read_text()).items():
            assert sha(package / rel) == expected, 'Harness source differs: ' + rel
        from prepare_baseline_cpu import verify_evaluation_assets
        verify_evaluation_assets(inputs)
        originals = json.loads((inputs / 'manifests/ORIGINAL_MODEL_FILES.json').read_text())
        audit = {'condition': 'original', 'protocol_identity': ready['protocol_identity'],
            'model_identity': identity({'model': originals, 'evaluation_dtype': 'float16'}),
            'gpu_implementation_identity': config['implementation_identity']}
        from strict_backward_gate import run as strict_gate
        gate_path = root / 'logs' / (config['run_id'] + '_strict_gate.json')
        # Recheck under each allocated runtime rather than assuming a prior GPU's behavior.
        result['strict_gate'] = strict_gate(config)
        progress(config['state'], 'save strict deterministic backward gate', saving=True)
        publish(gate_path, result['strict_gate'])
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(6304); torch.cuda.manual_seed_all(6304)
        print('Strict backward gate passed. Loading fresh original weights for baseline work.', flush=True)
        probes = probe_cache(config, ready, audit)
        result['activation_probes_completed'] = probes['count']
        run_baseline(config, ready, audit, result)
        result['status'] = 'TASK1_BASELINE_COMPLETE_SAVED'
    except SoftStop as error:
        result.update(status='TASK1_BASELINE_PARTIAL_SAVED', stop_reason=str(error),
                      resume='Reuse this same block; completed answers are preserved and reused.')
    except BaseException as error:
        result.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        code = 1
    result['elapsed_worker_seconds'] = time.monotonic() - started
    try:
        result['GPU_memory'] = {'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'total_gpu_bytes': torch.cuda.get_device_properties(0).total_memory,
            'scope': 'baseline work after strict gate' if 'strict_gate' in result else 'strict gate or earlier failure'}
    except Exception:
        result['GPU_memory'] = 'not available'
    result['physical_runtime_release_verified'] = False
    # Save/read back the result BEFORE announcing worker completion to the guardian.
    publish(config['result'], result)
    print(json.dumps(result, indent=2), flush=True)
    state = json.loads(Path(config['state']).read_text())
    state.update(status='worker_complete', worker_completed_wall=time.time(), progress_wall=time.time())
    atomic(config['state'], state)
    return code


if __name__ == '__main__': sys.exit(main(sys.argv[1]))
