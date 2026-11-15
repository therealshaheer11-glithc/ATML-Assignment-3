"""T1-A003 pilot only; no official benchmark generation or training."""
import gc
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_runtime import SoftStop, atomic, boundary, progress, wait_for_parent
from evaluation_records import identity, publish


PROMPTS = [
    'Compute 17 + 26. Explain your calculation.',
    'A ribbon is 84 centimetres long and is cut into seven equal pieces. How long is each piece? Show the calculation.',
    'Explain why multiplying any whole number by zero gives zero.',
    'Write a short Python function that returns the larger of two numbers, and explain it.',
    'Explain how evaporation differs from condensation in a short paragraph.',
    'Give a brief explanation of why a calendar needs leap years.',
    'Summarize the following fictional note in three bullet points: The library opens at nine. The reading room closes at six. Returned books go in the blue box. Volunteers sort donations on Fridays.',
    'Compare a triangle and a rectangle using their sides and angles. Include one example of each.',
]


def make_requests(tokenizer, questions=PROMPTS, cap=1024):
    from lm_eval.api.instance import Instance
    result = []
    for index, question in enumerate(questions):
        text = tokenizer.apply_chat_template([{'role': 'user', 'content': 'Q: ' + question + '\nA:'}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        result.append(Instance(request_type='generate_until', doc={'diagnostic_only': True},
            arguments=(text, {'max_gen_toks': cap, 'do_sample': False,
                             'until': ['Q:', '</s>', '<|im_end|>']}), idx=0,
            metadata=('nonbenchmark_batch8_pilot', index, 1)))
    return result


def prefill_logits(lm, contexts):
    import torch
    ids, mask = lm.tok_batch_encode(contexts)
    ids, mask = ids.to(lm.device), mask.to(lm.device)
    # Use the model's native generation preparation for left-padding position IDs.
    prepared = lm.model.prepare_inputs_for_generation(ids, attention_mask=mask,
        cache_position=torch.arange(ids.shape[1], device=lm.device), use_cache=False)
    with torch.inference_mode():
        result = lm.model(**prepared).logits[:, -1, :].float().cpu()
    if lm.device.type == 'cuda': torch.cuda.synchronize()
    assert torch.isfinite(result).all()
    return result


def run(config):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from batch8_hflm import BatchJournalHFLM
    inputs = Path(config['inputs']); destination = Path(config['journal'])
    destination.mkdir(parents=True, exist_ok=False)
    boundary(config, 'load unchanged original model for approved non-benchmark pilot')
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(6304); torch.cuda.manual_seed_all(6304)
    model = AutoModelForCausalLM.from_pretrained(inputs / 'sources/original_model',
        local_files_only=True, dtype=torch.float16, attn_implementation='sdpa', low_cpu_mem_usage=True)
    model.requires_grad_(False).eval().to('cuda'); model.config.use_cache = True
    assert sum(p.numel() for p in model.parameters()) == 596049920
    tokenizer = AutoTokenizer.from_pretrained(inputs / 'sources/original_model', local_files_only=True)
    contract = {'condition': 'NONBENCHMARK_PILOT_ONLY', 'approval_id': 'T1-A003',
        'model_identity': config['model_identity'], 'implementation_identity': config['implementation_identity'],
        'prompt_identity': identity(PROMPTS)}
    lm = BatchJournalHFLM(pretrained=model, tokenizer=tokenizer, dtype='float16', batch_size=8,
        enable_thinking=False, add_bos_token=False, job_config=config, audit_contract=contract,
        journal=destination / 'batched')
    requests = make_requests(tokenizer)
    publish(destination / 'PROMPTS.json', {'questions': PROMPTS,
        'rendered_requests': [r.args for r in requests], 'official_benchmark_questions_used': 0})
    boundary(config, 'compare serial and batch-eight native prefill logits')
    texts = [request.args[0] for request in requests]
    batched_logits = prefill_logits(lm, texts)
    serial_logits = torch.cat([prefill_logits(lm, [text]) for text in texts], dim=0)
    difference = batched_logits - serial_logits
    max_error = float(difference.abs().max())
    relative_rms = float(difference.square().mean().sqrt() / serial_logits.square().mean().sqrt().clamp_min(1e-12))
    next_agreement = bool((batched_logits.argmax(-1) == serial_logits.argmax(-1)).all())
    logits_ok = max_error <= .1 and relative_rms <= .005 and next_agreement
    del batched_logits, serial_logits, difference
    boundary(config, 'conservative full-length batch-eight cache memory check')
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    length = max(config['maximum_prepared_GSM_prompt_tokens'],
                 max(len(lm.tok_encode(text)) for text in texts)) + 1024
    ids = torch.randint(0, 151000, (8, length), dtype=torch.long, device=lm.device)
    with torch.inference_mode():
        output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True)
        assert torch.isfinite(output.logits[:, -1, :]).all()
    torch.cuda.synchronize()
    reserved = torch.cuda.max_memory_reserved(); total = torch.cuda.get_device_properties(0).total_memory
    headroom = 1 - reserved / total
    del output, ids
    gc.collect(); torch.cuda.empty_cache()
    boundary(config, 'measure eight fixed non-benchmark prompts serially')
    lm.journal = destination / 'serial'
    one_text, one_meta, one_time = lm.generate_journaled(requests, 1)
    assert one_time['cache_hits'] == 0
    boundary(config, 'measure the same fixed prompts in batch eight')
    lm.journal = destination / 'batched'
    eight_text, eight_meta, eight_time = lm.generate_journaled(requests, 8)
    assert eight_time['cache_hits'] == 0
    agreements = [a['semantic_new_ids'] == b['semantic_new_ids'] for a, b in zip(one_meta, eight_meta)]
    same_text = [a == b for a, b in zip(one_text, eight_text)]
    speedup = one_time['model_inference_seconds_this_call'] / max(eight_time['model_inference_seconds_this_call'], 1e-12)
    checks = {'finite_prefill_logits_within_approved_tolerances': logits_ok,
        'all_prefill_next_tokens_agree': next_agreement,
        'all_8_semantic_token_sequences_agree': all(agreements),
        'all_8_native_postprocessed_outputs_agree': all(same_text),
        'speedup_at_least_2': speedup >= 2., 'reserved_memory_headroom_at_least_10_percent': headroom >= .1}
    result = {'status': 'TASK1_BATCH8_PILOT_PASSED' if all(checks.values()) else 'TASK1_BATCH8_PILOT_REJECTED',
        'checks': checks, 'approval_id': 'T1-A003', 'official_training_updates': 0,
        'official_benchmark_answers_generated': 0, 'model_inference_speedup': speedup,
        'serial_time': one_time, 'batch8_time': eight_time, 'max_absolute_logit_difference': max_error,
        'relative_RMS_logit_difference': relative_rms, 'token_agreement_by_prompt': agreements,
        'postprocessed_text_agreement_by_prompt': same_text, 'peak_reserved_bytes': reserved,
        'total_GPU_bytes': total, 'headroom_fraction': headroom, 'synthetic_memory_check_length': length,
        'conditional_adoption_gate_passed': all(checks.values()),
        'no_unseen_prompt_equality_guarantee': True}
    publish(destination / 'COMPARISON.json', {'result': result,
        'serial_outputs': one_text, 'batch8_outputs': eight_text,
        'serial_metadata': one_meta, 'batch8_metadata': eight_meta})
    return result


def main(config_path):
    config = json.loads(Path(config_path).read_text()); wait_for_parent(config['state'])
    started = time.monotonic(); code = 0
    try:
        import os
        import torch
        assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
        assert torch.cuda.device_count() == 1 and 'L4' in torch.cuda.get_device_name(0)
        result = run(config)
    except SoftStop as error:
        result = {'status': 'TASK1_BATCH8_PILOT_INCOMPLETE', 'error': str(error), 'conditional_adoption_gate_passed': False}
    except BaseException as error:
        result = {'status': 'TASK1_BATCH8_PILOT_FAILED', 'error': str(error),
            'traceback': traceback.format_exc(), 'conditional_adoption_gate_passed': False}; code = 1
    result.update(run_id=config['run_id'], elapsed_worker_seconds=time.monotonic() - started,
        official_training_updates=0, official_benchmark_answers_generated=0)
    publish(config['result'], result)
    print(json.dumps(result, indent=2), flush=True)
    state = json.loads(Path(config['state']).read_text())
    state.update(status='worker_complete', worker_completed_wall=time.time(), progress_wall=time.time())
    atomic(config['state'], state)
    return code


if __name__ == '__main__': sys.exit(main(sys.argv[1]))
