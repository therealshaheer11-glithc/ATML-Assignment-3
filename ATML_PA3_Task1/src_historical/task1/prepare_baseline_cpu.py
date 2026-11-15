"""Freeze every evaluation request and validate stock scoring/resume on CPU."""
import gzip
import hashlib
import importlib.metadata
import json
import os
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation_records import COUNTS, CONDITIONS, RecordStore, encoded, identity, publish, cpu_checks
from stock_evaluation import load_task, request_for, serialized_requests, scoring_checks, chat_formatter
from evaluation_engine import run_chunk


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_bytes(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError('Frozen evaluation input differs: ' + str(path))
        return
    pending = path.with_name(path.name + '.pending_' + uuid.uuid4().hex)
    pending.write_bytes(data); assert pending.read_bytes() == data
    if path.exists(): raise RuntimeError('Concurrent writer')
    pending.rename(path); assert path.read_bytes() == data


def verify_evaluation_assets(root):
    """Check each file against its owning frozen manifest, including copied files.

    prepare_data owns TOKENIZER_FILES (flat filename -> sha256). prepare_assets
    owns ORIGINAL_MODEL_FILES (model metadata plus nested per-file metadata).
    Tokenizer copies in original_model are intentionally absent from the latter.
    """
    root = Path(root)
    original = json.loads((root / 'manifests/ORIGINAL_MODEL_FILES.json').read_text())
    tokenizer = json.loads((root / 'manifests/TOKENIZER_FILES.json').read_text())
    frozen = json.loads((root / 'configs/SOURCE_REVISIONS.json').read_text())['model']
    if original['model'] != frozen:
        raise RuntimeError('Original-model source revision differs from the source freeze')
    checked = {}
    for name in ('config.json', 'generation_config.json'):
        expected = original['files'][name]['sha256']
        if sha(root / 'sources/original_model' / name) != expected:
            raise RuntimeError('Original model configuration hash differs: ' + name)
        checked['sources/original_model/' + name] = expected
    for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json'):
        expected = tokenizer[name]
        for directory in ('sources/tokenizer', 'sources/original_model'):
            rel = directory + '/' + name
            if sha(root / rel) != expected:
                raise RuntimeError('Frozen tokenizer configuration/copy hash differs: ' + rel)
            checked[rel] = expected
    return checked


def main(root, local, bundle):
    import lm_eval
    import torch
    import yaml
    from transformers import AutoTokenizer, AutoConfig
    root, local, bundle = Path(root), Path(local), Path(bundle)
    started = time.monotonic()
    assert not torch.cuda.is_available(), 'This stage requires CPU'
    assert importlib.metadata.version('lm_eval') == '0.4.9.2'
    assert torch.__version__ == '2.8.0+cu126'
    assert importlib.metadata.version('transformers') == '4.57.1'
    torch.set_num_threads(2)
    print('1/5 Verifying the frozen harness source and tokenizer...', flush=True)
    package_root = Path(lm_eval.__file__).parent
    hashes = json.loads((root / 'configs/HARNESS_SOURCE_HASHES.json').read_text())
    actual_files = {str(p.relative_to(package_root)) for p in package_root.rglob('*')
                    if p.is_file() and p.suffix in ('.py', '.yaml')}
    assert actual_files == set(hashes)
    for rel, expected in hashes.items():
        assert sha(package_root / rel) == expected, rel
    model_dir = root / 'sources/original_model'
    checked_assets = verify_evaluation_assets(root)
    print('Tokenizer and model configuration files match their owning frozen manifests.', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    max_length = int(config.max_position_embeddings)
    assert max_length > 2048 and tokenizer.eos_token == '<|im_end|>'
    # These are formatting/scoring fixtures, not an evaluation of model behavior.
    fixture_root = local / ('evaluation_CPU_fixtures_' + uuid.uuid4().hex)
    from baseline_asset_checks import run_checks as check_asset_manifests
    asset_regressions = check_asset_manifests(verify_evaluation_assets, fixture_root / 'asset_regressions')
    fixtures = cpu_checks(fixture_root / 'record_tests')
    benchmark_manifest = json.loads((root / 'manifests/BENCHMARK_FILES.json').read_text())
    protocol = {'schema_version': 1, 'approval_ids': ['T1-A001', 'T1-A002'],
        'conditions': list(CONDITIONS), 'counts_per_condition': COUNTS, 'seed': 6304,
        'num_fewshot': 0, 'dtype': 'float16', 'dropout': False,
        'thinking': False, 'generation_batch_size': 1, 'likelihood_batch_size': 2,
        'gsm_chunk_size': 25, 'likelihood_chunk_size': 100,
        'gsm_max_new_tokens': 1024, 'qualitative_max_new_tokens': 512,
        'chat_tasks': ['gsm8k_cot', 'qualitative'],
        'completion_tasks': ['hellaswag', 'boolq', 'arc_easy'],
        'primary_metrics': {'gsm8k_cot': 'exact_match,flexible-extract',
            'hellaswag': 'acc_norm,none', 'boolq': 'acc,none', 'arc_easy': 'acc_norm,none'},
        'secondary_metrics': {'hellaswag': 'acc,none', 'arc_easy': 'acc,none'},
        'benchmark_files': benchmark_manifest, 'model_max_context_tokens': max_length,
        'qualitative_source_sha256': '215abde5b02811e922b6ac563ff33c490e0dff9ab58439df840f3135e9387173',
        'harness_source_manifest_sha256': sha(root / 'configs/HARNESS_SOURCE_HASHES.json'),
        'baseline_model': json.loads((root / 'configs/SOURCE_REVISIONS.json').read_text())['model'],
        'baseline_checkpoint': 'unchanged original posttrained model; FP16 evaluation',
        'probe_forward_dtype': 'float32', 'probe_layers': [0, 13, 27],
        'probe_location': 'down_proj input, final nonpadding prompt token, before any continuation',
        'aggregation': 'stock per-document metrics; final means only at complete split coverage',
        'truncation': 'retain capped outputs in scores; save raw token termination metadata',
        'inference_implementation': 'native pinned HFLM; native stock filters and process_results',
        'no_result_based_prompt_or_subset_selection': True}
    protocol_id = identity(protocol)
    publish(root / 'configs/EVALUATION_PROTOCOL_V1.json', protocol)
    print('2/5 Preparing and auditing ALL four complete benchmark splits...', flush=True)
    prepared_manifest, prompt_stats, score_checks, engine_checks = {}, {}, {}, {}
    probe_ids = json.loads((root / 'configs/ACTIVATION_PROBE_IDS.json').read_text())
    probe_prompts = []
    for name, count in COUNTS.items():
        task, stock_config, yaml_path = load_task(root, name)
        docs = task.eval_docs
        assert len(docs) == count and task.config.num_fewshot == 0
        if name == 'gsm8k_cot':
            assert stock_config['num_fewshot'] == 8
            assert task.config.generation_kwargs['max_gen_toks'] == 1024
            assert task.config.generation_kwargs['until'] == stock_config['generation_kwargs']['until']
            assert task.config.filter_list == stock_config['filter_list']
        score_checks[name] = scoring_checks(task, name, tokenizer)
        rows, lengths, choice_count = [], [], 0
        selected = {r['source_index'] for r in probe_ids.get(name, [])}
        for i, doc in enumerate(docs):
            context, requests = request_for(task, name, i, doc, tokenizer)
            raw_text = task.doc_to_text(doc)
            if name == 'gsm8k_cot':
                assert raw_text == 'Q: ' + doc['question'] + '\nA:'
                assert context == chat_formatter(tokenizer)([{'role': 'user', 'content': raw_text}])
                assert '<think>\n\n</think>\n\n' in context
            else:
                assert context == raw_text
                assert '<|im_start|>' not in context
                assert all(r.request_type == 'loglikelihood' for r in requests)
            ids = tokenizer.encode(context, add_special_tokens=False)
            cap = 1024 if name == 'gsm8k_cot' else 0
            assert len(ids) + cap <= max_length, (name, i, 'Context would be truncated')
            for request in requests:
                if request.request_type == 'loglikelihood':
                    full = tokenizer.encode(request.args[0] + request.args[1], add_special_tokens=False)
                    assert len(full) <= max_length, (name, i, 'Likelihood context would be truncated')
            lengths.append(len(ids)); choice_count += len(requests)
            row = {'source_index': i, 'document': doc, 'context': context,
                   'prompt_input_ids': ids, 'requests': serialized_requests(requests)}
            rows.append(row)
            if i in selected:
                probe_prompts.append({'task': name, 'source_index': i,
                    'prompt_input_ids': ids, 'context': context, 'layers': [0, 13, 27]})
        payload = gzip.compress(b''.join(encoded(r) for r in rows), mtime=0)
        rel = 'evaluation_inputs/' + name + '_requests_V1.jsonl.gz'
        freeze_bytes(root / rel, payload)
        prepared_manifest[name] = {'path': rel, 'sha256': sha(root / rel), 'count': count,
            'stock_yaml_sha256': sha(yaml_path), 'requests': choice_count}
        prompt_stats[name] = {'count': count, 'min_prompt_tokens': min(lengths),
            'max_prompt_tokens': max(lengths), 'total_prompt_tokens': sum(lengths),
            'chat_template': name == 'gsm8k_cot', 'num_fewshot': 0,
            'context_truncation_required': False}
        # Exercise the production resume engine on two real request structures.
        # Fixture answers consult the gold label; no model is loaded or evaluated.
        store = RecordStore(fixture_root / 'engine_tests' / name, 'original', 'CPU_FIXTURE_ONLY', protocol_id)
        calls = []
        def fixture_infer(kind, native_requests):
            calls.append(len(native_requests)); out = []
            for request in native_requests:
                target = task.doc_to_target(request.doc)
                if kind == 'generate_until':
                    out.append('The answer is ' + str(target) + '.')
                else:
                    choices = task.doc_to_choice(request.doc)
                    gold = int(target) if not isinstance(target, str) else choices.index(target)
                    out.append((0. if request.idx == gold else -1000., request.idx == gold))
            return out, [{'fixture_only': True} for _ in out]
        first = run_chunk(task, name, rows[:1], store, fixture_infer)
        recovered = run_chunk(task, name, rows[:2], store, fixture_infer)
        reused = run_chunk(task, name, rows[:2], store, fixture_infer)
        assert len(calls) == 2 and len(first) == 1 and recovered == reused
        assert all(r['scores'][protocol['primary_metrics'][name]] == 1. for r in recovered)
        engine_checks[name] = {'resume_only_infers_missing_example': True,
                              'repeat_chunk_makes_no_new_inference_calls': True,
                              'native_request_roundtrip_preserved': True}
        print(name + ': ' + str(count) + ' prompts checked; stock scoring and resume fixtures passed.', flush=True)
    print('3/5 Preparing all qualitative prompts and the SAME 64 fixed activation probes...', flush=True)
    source = root / 'sources/first_plot_questions.yaml'
    assert sha(source) == protocol['qualitative_source_sha256']
    prompts = json.loads((root / 'configs/QUALITATIVE_PROMPTS.json').read_text())
    assert len(prompts) == 24
    raw_prompts = yaml.safe_load(source.read_text())
    assert len(raw_prompts) == len(prompts)
    assert sum(bool(row['annotate']) for row in prompts) == 6
    qualitative = []
    for row, original in zip(prompts, raw_prompts):
        messages = row['messages']
        original_messages = ([{'role': 'system', 'content': original['system']}]
                             if original.get('system') else [])
        original_messages.append({'role': 'user', 'content': original['paraphrases'][0]})
        assert row['id'] == original['id'] and messages == original_messages
        context = chat_formatter(tokenizer)(messages)
        ids = tokenizer.encode(context, add_special_tokens=False)
        assert len(ids) + 512 <= max_length
        qualitative.append({**row, 'context': context, 'prompt_input_ids': ids,
                            'max_new_tokens': 512, 'do_sample': False, 'enable_thinking': False})
    qual_rel = 'evaluation_inputs/qualitative_requests_V1.json'
    publish(root / qual_rel, qualitative)
    assert len(probe_prompts) == 64
    # Match the previously frozen random draw order, not source traversal order.
    probe_index = {(r['task'], r['source_index']): r for r in probe_prompts}
    ordered_probes = [probe_index[(name, row['source_index'])]
                     for name in ('gsm8k_cot', 'hellaswag') for row in probe_ids[name]]
    publish(root / 'evaluation_inputs/activation_prompt_inputs_V1.json', ordered_probes)
    print('4/5 Publishing evaluation code and immutable request fingerprints...', flush=True)
    names = ('evaluation_records.py', 'stock_evaluation.py', 'evaluation_engine.py',
             'prepare_baseline_cpu.py', 'archive_io.py', 'restore_baseline_cpu.py',
             'baseline_asset_checks.py')
    file_hashes = {name: sha(bundle / name) for name in names}
    kit_id = identity(file_hashes)
    destination = root / 'execution' / ('evaluation_V1_' + kit_id)
    for name in names:
        freeze_bytes(destination / name, (bundle / name).read_bytes())
    supervisor = json.loads((root / 'configs/SUPERVISOR_R2_READY.json').read_text())
    assert supervisor['legacy_block08_must_not_be_reused']
    assert all(supervisor['completion_tests'].values())
    for name, expected in supervisor['files'].items():
        assert sha(root / supervisor['implementation_path'] / name) == expected
    ready = {'schema_version': 1, 'protocol_identity': protocol_id,
        'protocol_path': 'configs/EVALUATION_PROTOCOL_V1.json', 'kit_id': kit_id,
        'kit_path': str(destination.relative_to(root)), 'files': file_hashes,
        'benchmarks': prepared_manifest, 'qualitative_path': qual_rel,
        'qualitative_sha256': sha(root / qual_rel), 'qualitative_count': 24,
        'probe_path': 'evaluation_inputs/activation_prompt_inputs_V1.json',
        'probe_sha256': sha(root / 'evaluation_inputs/activation_prompt_inputs_V1.json'),
        'probe_count': 64, 'stock_scoring_checks': score_checks,
        'implementation_revision': 'R1', 'asset_manifest_regressions': asset_regressions,
        'resume_engine_checks': engine_checks, 'record_checks': fixtures,
        'supervisor_R2_verified': True, 'strict_deterministic_gpu_check_pending': True,
        'GPU_launch_controller_integration_pending': True,
        'actual_model_evaluation_run': False, 'training_started': False}
    publish(root / 'configs/BASELINE_CPU_READY_V1.json', ready)
    print('5/5 Saving the CPU audit summary...', flush=True)
    receipt = {'status': 'TASK1_BLOCK10_CPU_BASELINE_KIT_OK',
        'implementation_revision': 'R1', 'model_tokenizer_asset_checks': checked_assets,
        'asset_manifest_regressions': asset_regressions,
        'protocol_identity': protocol_id, 'benchmarks': prompt_stats,
        'total_benchmark_questions_per_condition': sum(COUNTS.values()),
        'qualitative_prompts': 24, 'activation_probes': 64,
        'stock_scoring_checks': score_checks, 'resume_engine_checks': engine_checks,
        'record_checks': fixtures, 'strict_deterministic_GPU_check_pending': True,
        'actual_model_evaluation_run': False, 'training_started': False,
        'GPU_work_run': False, 'elapsed_seconds': round(time.monotonic() - started, 1),
        'next_stage': 'Integrate repaired supervisor and strict GPU gate; then run unchanged-model baseline.'}
    publish(root / 'logs' / ('block10_' + uuid.uuid4().hex + '.json'), receipt)
    print(json.dumps(receipt, indent=2), flush=True)
    print('TASK1_BLOCK10_CPU_BASELINE_KIT_OK — keep CPU selected and send the final summary.', flush=True)


if __name__ == '__main__': main(*sys.argv[1:])
