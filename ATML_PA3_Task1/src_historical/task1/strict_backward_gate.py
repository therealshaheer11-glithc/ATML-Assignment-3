"""Strict deterministic full-length backward, with no optimizer updates."""
import gc
import json
import time
from pathlib import Path

from baseline_runtime import boundary, progress


def check_kind(config, kind):
    import torch
    import training_core as core
    import training_ops as ops
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM
    inputs = Path(config['inputs'])
    assets = json.loads((inputs / 'manifests/CPU_ASSETS_READY.json').read_text())
    boundary(config, 'strict backward: load ' + kind)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(inputs / 'sources/original_model',
        local_files_only=True, dtype=torch.float16 if kind == 'oplora' else torch.float32,
        attn_implementation='sdpa', low_cpu_mem_usage=True)
    model.config.use_cache = False
    if kind == 'oplora':
        model.requires_grad_(False)
        init = load_file(inputs / assets['initializer_files']['16']['path'])
        for item in assets['bases']:
            name = item['key'][:-len('.weight')]
            basis = load_file(inputs / 'bases' / (item['key'] + '.safetensors'))
            layer = core.AdapterLinear(model.get_submodule(name), init[name + '.A'], init[name + '.B'],
                basis['U'], basis['V'], alpha=32, dropout=.05)
            parent, leaf = name.rsplit('.', 1)
            setattr(model.get_submodule(parent), leaf, layer)
        del init, basis, layer
    model.to('cuda').train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assert all(p.dtype == torch.float32 for _, p in named)
    assert sum(p.numel() for _, p in named) == (10092544 if kind == 'oplora' else 596049920)
    reference = {n: p.detach().clone() for n, p in model.named_parameters()} if kind == 'full_with_l2_reference' else None
    optimizer = torch.optim.AdamW([p for _, p in named], lr=2e-5,
        betas=(.9, .999), eps=1e-8, weight_decay=0., foreach=False, fused=False)
    # Reserve real FP32 Adam moment storage without executing an AdamW step.
    for _, parameter in named:
        optimizer.state[parameter] = {'step': torch.tensor(0., dtype=torch.float32),
            'exp_avg': torch.zeros_like(parameter), 'exp_avg_sq': torch.zeros_like(parameter)}
    scaler = torch.amp.GradScaler('cuda', init_scale=65536., growth_factor=2.,
        backoff_factor=.5, growth_interval=2000)
    torch.manual_seed(6304); torch.cuda.manual_seed_all(6304)
    ids = torch.randint(0, 151000, (2, 2048), device='cuda'); ids[:, -1] = 151645
    labels = ids.clone(); labels[:, :64] = -100; mask = torch.ones_like(ids)
    denominator = int(labels[:, 1:].ne(-100).sum()) * 16
    cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    retries = []
    for retry in range(9):
        boundary(config, 'strict backward: ' + kind + ', retry ' + str(retry))
        torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state(gpu_rng)
        optimizer.zero_grad(set_to_none=True)
        for micro in range(16):
            progress(config['state'], 'strict ' + kind + ': microbatch ' + str(micro + 1) + '/16')
            with torch.autocast('cuda', dtype=torch.float16):
                loss, count = ops.chunked_response_sum(model, ids, labels, mask)
                assert count * 16 == denominator
                objective = loss / denominator
            if reference is not None and micro == 15:
                objective = objective + 1e-3 * core.l2_sp(model, reference)
            if not torch.isfinite(objective): raise FloatingPointError('Nonfinite strict backward objective')
            scaler.scale(objective).backward()
            del loss, objective
        scaler.unscale_(optimizer)
        finite = all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for _, p in named)
        if finite: break
        # update(new_scale=...) changes only the scratch scaler, never parameters.
        before = scaler.get_scale(); scaler.update(new_scale=before * .5)
        retries.append({'retry': retry, 'old_scale': before, 'new_scale': scaler.get_scale()})
        if retry == 8: raise FloatingPointError('Strict backward overflow persisted after eight retries')
    torch.nn.utils.clip_grad_norm_([p for _, p in named], 1., error_if_nonfinite=True)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_reserved(); total = torch.cuda.get_device_properties(0).total_memory
    assert peak <= .9 * total, 'Strict backward leaves less than the approved 10% memory headroom'
    assert torch.are_deterministic_algorithms_enabled() and not torch.is_deterministic_algorithms_warn_only_enabled()
    result = {'kind': kind, 'microbatch': 2, 'accumulation': 16, 'sequence_length': 2048,
        'strict_deterministic_backward_passed': True, 'FP32_Adam_moments_reserved': True,
        'FP32_L2_reference_reserved': reference is not None,
        'optimizer_updates': 0, 'official_training_updates': 0, 'overflow_retries': retries,
        'peak_reserved_bytes': peak, 'total_gpu_bytes': total, 'headroom_fraction': 1 - peak / total,
        'elapsed_seconds': time.monotonic() - started}
    return result


def run(config):
    import torch
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.is_deterministic_algorithms_warn_only_enabled()
    outcomes = []
    for kind in ('full_with_l2_reference', 'oplora'):
        outcomes.append(check_kind(config, kind))
        gc.collect(); torch.cuda.empty_cache()
        print(json.dumps(outcomes[-1], indent=2), flush=True)
    return {'status': 'TASK1_STRICT_BACKWARD_GATE_OK', 'checks': outcomes,
        'scratch_models_discarded': True, 'optimizer_updates': 0, 'official_training_updates': 0}
