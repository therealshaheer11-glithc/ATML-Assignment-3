"""Approved source-order training arithmetic and five independently built methods."""
import gzip
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

import training_core as core
import training_ops as ops

METHODS = ('full', 'l2sp', 'lora4', 'lora16', 'oplora')
MANDATORY = (312, 625, 937, 1250)


def require_training_gpu():
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    assert torch.__version__ == '2.8.0+cu126'
    assert torch.are_deterministic_algorithms_enabled() and not torch.is_deterministic_algorithms_warn_only_enabled()
    assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
    assert torch.cuda.device_count() == 1 and torch.cuda.get_device_name(0) == 'NVIDIA L4'


def seed_training():
    random.seed(6304); np.random.seed(6304); torch.manual_seed(6304)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(6304)


def rng_state(cuda):
    n = np.random.get_state()
    return {'python': random.getstate(), 'numpy': [n[0], n[1].tolist(), n[2], n[3], n[4]],
        'torch_cpu': torch.get_rng_state().tolist(),
        'torch_cuda': [r.tolist() for r in torch.cuda.get_rng_state_all()] if cuda else []}


def restore_rng(value):
    def tuples(v): return tuple(tuples(x) for x in v) if isinstance(v, (list, tuple)) else v
    random.setstate(tuples(value['python']))
    n = value['numpy']; np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(torch.tensor(value['torch_cpu'], dtype=torch.uint8))
    if value['torch_cuda']:
        assert len(value['torch_cuda']) == torch.cuda.device_count()
        torch.cuda.set_rng_state_all([torch.tensor(s, dtype=torch.uint8) for s in value['torch_cuda']])


def optimizer_for(model, peak):
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assert named and all(p.dtype == torch.float32 for _, p in named)
    groups = [{'params': [p for n, p in named if p.ndim >= 2 and not n.endswith('bias')], 'weight_decay': .01},
              {'params': [p for n, p in named if p.ndim < 2 or n.endswith('bias')], 'weight_decay': 0.}]
    optimizer = torch.optim.AdamW(groups, lr=peak, betas=(.9, .999), eps=1e-8, foreach=False, fused=False)
    return named, optimizer


def collate(rows, pad_id, device):
    assert len(rows) == 2
    width = max(len(r['input_ids']) for r in rows)
    assert 1 < width <= 2048
    ids = torch.full((2, width), pad_id, dtype=torch.long)
    labels = torch.full_like(ids, -100); mask = torch.zeros_like(ids)
    for i, r in enumerate(rows):
        values = r['input_ids']; start = r['response_start']; size = len(values)
        assert 0 < start < size and size - start == r['n_supervised_tokens']
        ids[i, :size] = torch.tensor(values, dtype=torch.long)
        labels[i, start:size] = ids[i, start:size]; mask[i, :size] = 1
    assert int(labels[:, 1:].ne(-100).sum()) == sum(r['n_supervised_tokens'] for r in rows)
    return tuple(t.to(device) for t in (ids, labels, mask))


class SourceOrderData:
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root / 'manifests/TRAINING_DATA_READY.json').read_text())
        assert self.manifest['selected_count'] == 40000 and self.manifest['total_supervised_tokens'] == 7100056
        assert self.manifest['full_saved_training_audit_passed'] and len(self.manifest['shards']) == 40
        self.ids = json.loads((self.root / 'configs/TRAIN_SOURCE_IDS.json').read_text())
        assert len(self.ids) == 40000 and self.ids == sorted(set(self.ids))
        self.loaded, self.records = None, None

    def __getitem__(self, i):
        assert 0 <= i < 40000
        shard = i // 1000
        if shard != self.loaded:
            entry = self.manifest['shards'][shard]; path = self.root / entry['path']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256']
            with gzip.open(path, 'rt') as stream: records = [json.loads(line) for line in stream]
            assert len(records) == entry['count'] == 1000
            assert [r['source_index'] for r in records] == self.ids[shard * 1000:(shard + 1) * 1000]
            self.loaded, self.records = shard, records
        return self.records[i % 1000]

    def batch(self, update):
        assert 1 <= update <= 1250
        return [self[i] for i in range((update - 1) * 32, update * 32)]


def effective_step(model, named, optimizer, scaler, rows, update, peak, pad_id, device,
                   coherence, reference=None, progress=lambda phase: None):
    """Retry an identical 32-example batch on overflow; commit in the caller."""
    device = torch.device(device); cuda = device.type == 'cuda'
    if cuda: require_training_gpu()
    assert len(rows) == 32 and all(p.requires_grad and p.dtype == torch.float32 for _, p in named)
    denominator = sum(r['n_supervised_tokens'] for r in rows); assert denominator > 0
    before = rng_state(cuda); coherence.update(coherent=True, pre_update_rng=before)
    previous_lrs = [g['lr'] for g in optimizer.param_groups]
    lr = core.learning_rate(update, peak); skipped = []; begin = time.monotonic()
    coherence['overflow_attempts'] = skipped
    def sync():
        if cuda: torch.cuda.synchronize()
    try:
        for retry in range(9):
            restore_rng(before); optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups: group['lr'] = lr
            sync(); start = time.monotonic(); sft = 0.; omega_value = 0.
            for micro in range(16):
                progress(f'update {update}/1250, microbatch {micro+1}/16, retry {retry}')
                ids, labels, mask = collate(rows[micro*2:micro*2+2], pad_id, device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=cuda):
                    loss, count = ops.chunked_response_sum(model, ids, labels, mask)
                    assert count == sum(r['n_supervised_tokens'] for r in rows[micro*2:micro*2+2])
                    objective = loss / denominator
                sft += float(objective.detach())
                if reference is not None and micro == 15:
                    omega = core.l2_sp(model, reference); omega_value = float(omega.detach())
                    objective = objective + 1e-3 * omega
                if not bool(torch.isfinite(objective)): raise FloatingPointError('Nonfinite forward objective')
                scaler.scale(objective).backward()
                del ids, labels, mask, loss, objective
                if reference is not None and micro == 15: del omega
            scaler.unscale_(optimizer)
            finite = all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for _, p in named)
            if not finite:
                old_scale = scaler.get_scale()
                # GradScaler must skip AdamW entirely, then back off the same batch.
                scaler.step(optimizer); scaler.update(); sync()
                assert scaler.get_scale() < old_scale
                skipped.append({'retry': retry, 'old_scale': old_scale, 'new_scale': scaler.get_scale(),
                                'seconds': time.monotonic() - start})
                if retry == 8: raise FloatingPointError('Overflow after first attempt and eight retries')
                continue
            norm = torch.nn.utils.clip_grad_norm_([p for _, p in named], 1., error_if_nonfinite=True, foreach=False)
            # A partially executed optimizer call is not safe for emergency serialization.
            coherence['coherent'] = False
            scaler.step(optimizer); scaler.update(); sync()
            for _, parameter in named:
                state = optimizer.state[parameter]
                assert all(v.dtype == torch.float32 for v in state.values() if isinstance(v, torch.Tensor))
                assert int(state['step'].item()) == update
            record = {'update': update, 'examples_seen': update * 32, 'supervised_tokens': denominator,
                'source_ids': [r['source_index'] for r in rows], 'lr': lr, 'sft_loss': sft,
                'omega': omega_value, 'lambda': 1e-3 if reference is not None else 0.,
                'penalized_objective': sft + (1e-3 * omega_value if reference is not None else 0.),
                'gradient_norm_before_clip': float(norm), 'scale': scaler.get_scale(),
                'overflow_attempts': skipped, 'successful_attempt_seconds': time.monotonic() - start,
                'total_update_seconds_including_retries': time.monotonic() - begin}
            optimizer.zero_grad(set_to_none=True)
            return record
        raise AssertionError('Unreachable retry loop')
    except BaseException:
        if coherence['coherent']:
            optimizer.zero_grad(set_to_none=True); restore_rng(before)
            for group, lr_before in zip(optimizer.param_groups, previous_lrs): group['lr'] = lr_before
        raise


def build_method(inputs, method, device):
    from transformers import AutoModelForCausalLM
    from safetensors.torch import load_file
    assert method in METHODS
    assert torch.device(device).type == 'cuda'
    require_training_gpu()
    inputs = Path(inputs); assets = json.loads((inputs / 'manifests/CPU_ASSETS_READY.json').read_text())
    model = AutoModelForCausalLM.from_pretrained(inputs / 'sources/original_model', local_files_only=True,
        dtype=torch.float32, attn_implementation='sdpa', low_cpu_mem_usage=True)
    assert sum(p.numel() for p in model.parameters()) == 596049920
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        h.update(name.encode() + b'\0'); h.update(memoryview(p.detach().contiguous().numpy()).cast('B'))
    expected = json.loads((inputs / 'configs/CPU_MODEL_CONTRACT.json').read_text())
    assert h.hexdigest() == expected['fp32_master_state_sha256']
    if method in ('lora4', 'lora16', 'oplora'):
        model.requires_grad_(False).half()
        rank = 4 if method == 'lora4' else 16; alpha = 8 if rank == 4 else 32
        entry = assets['initializer_files'][str(rank)]; path = inputs / entry['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256']
        initial = load_file(path, device='cpu')
        for item in assets['bases']:
            name = item['key'][:-len('.weight')]; U = V = None
            if method == 'oplora':
                path = inputs / 'bases' / (item['key'] + '.safetensors')
                assert hashlib.sha256(path.read_bytes()).hexdigest() == item['basis_sha256']
                basis = load_file(path, device='cpu'); U, V = basis['U'], basis['V']
            layer = core.AdapterLinear(model.get_submodule(name), initial[name+'.A'], initial[name+'.B'], U, V, alpha, .05)
            assert torch.equal(layer.A, initial[name+'.A']) and torch.equal(layer.B, initial[name+'.B'])
            parent, leaf = name.rsplit('.', 1); setattr(model.get_submodule(parent), leaf, layer)
        assert sum(isinstance(m, core.AdapterLinear) for m in model.modules()) == 196
    else: model.requires_grad_(True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False, 'preserve_rng_state': True})
    model.to(device).train()
    peak = 2e-5 if method in ('full', 'l2sp') else 2e-4
    named, optimizer = optimizer_for(model, peak)
    expected_count = 596049920 if method in ('full', 'l2sp') else assets['adapter_parameter_counts']['4' if method == 'lora4' else '16']
    assert sum(p.numel() for _, p in named) == expected_count
    assert all(p.dtype == torch.float16 for p in model.parameters() if not p.requires_grad)
    reference = {n: p.detach().clone() for n, p in model.named_parameters()} if method == 'l2sp' else None
    scaler = torch.amp.GradScaler('cuda', init_scale=65536., growth_factor=2., backoff_factor=.5, growth_interval=2000)
    seed_training()
    return model, named, optimizer, scaler, reference, peak
