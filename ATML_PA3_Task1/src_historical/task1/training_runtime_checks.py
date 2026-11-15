"""CPU fixtures for the real training step, exact state reload and failure boundaries."""
import copy
import json
import random
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

import official_training as training
from evaluation_records import identity, publish
from training_checkpoints import (write_archive, verify_archive, save_checkpoint, restore_checkpoint,
                                  prune_rolling, state_header)


class TinyBase(nn.Module):
    def __init__(self):
        super().__init__(); self.embedding = nn.Embedding(19, 8); self.dropout = nn.Dropout(.2)
        self.proj = nn.Linear(8, 8)
    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(last_hidden_state=self.proj(self.dropout(self.embedding(input_ids))))


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__(); self.model = TinyBase(); self.lm_head = nn.Linear(8, 19, bias=False)
        self.lm_head.weight = self.model.embedding.weight
    def train(self, mode=True): return super().train(mode)


def rows():
    return [{'source_index': i, 'input_ids': [1, 3, 4, 5, 6, 2][:4 + i % 3],
             'response_start': 2 + i % 2, 'n_supervised_tokens': (4 + i % 3) - (2 + i % 2)} for i in range(32)]


def fresh(state=None):
    model = TinyLM().train()
    if state is not None: model.load_state_dict(state)
    named, opt = training.optimizer_for(model, 2e-5)
    scaler = torch.amp.GradScaler('cpu', init_scale=65536., growth_factor=2., backoff_factor=.5, growth_interval=2000)
    return model, named, opt, scaler


def perform(bundle, update, batch=None, reference=None):
    model, named, opt, scaler = bundle
    coherence = {}
    record = training.effective_step(model, named, opt, scaler, batch or rows(), update, 2e-5, 0, 'cpu', coherence, reference)
    assert coherence['coherent'] is False
    coherence.update(coherent=True, pre_update_rng=None)
    return record


def checks(directory):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    training.seed_training(); base = TinyLM().state_dict()
    a = fresh(base); training.seed_training()
    history = [perform(a, 1), perform(a, 2)]
    boundary_rng = training.rng_state(False)
    contract = {'fixture': 'CPU_ONLY', 'method': 'full', 'peak_lr': 2e-5}
    header = state_header('full', contract, 2, history, a[2], a[1], a[3], False)
    split = directory / 'many_shards.tar'
    manifest = write_archive(split, header, a[1], a[2], max_bytes=128)
    assert len(manifest['shards']) > 3
    verify_archive(split, manifest)
    # Full production checkpoint path: CPU shard writes, archive copy/readback, COMPLETE last.
    saved, marker = save_checkpoint(directory/'remote', directory, 'full', contract, 2, history,
                                   a[1], a[2], a[3], False, 'exit')
    restored_bundle = fresh(base)
    loaded = restore_checkpoint(saved, contract, restored_bundle[0], restored_bundle[1], restored_bundle[2], restored_bundle[3])
    assert identity(loaded['rng']) == identity(boundary_rng)
    assert identity(training.rng_state(False)) == identity(boundary_rng)
    assert restored_bundle[3].state_dict() == a[3].state_dict()
    for (n, p), (n2, q) in zip(a[1], restored_bundle[1]):
        assert n == n2 and torch.equal(p, q)
        assert a[2].state[p].keys() == restored_bundle[2].state[q].keys()
        for key, v in a[2].state[p].items(): assert torch.equal(v, restored_bundle[2].state[q][key])
    # Compare actual next-update continuation from the saved RNG, including dropout.
    training.restore_rng(boundary_rng); next_a = perform(a, 3)
    expected_rng = training.rng_state(False)
    training.restore_rng(loaded['rng']); next_b = perform(restored_bundle, 3)
    for (_, p), (_, q) in zip(a[1], restored_bundle[1]):
        torch.testing.assert_close(p, q, rtol=1e-5, atol=1e-6); assert torch.equal(p, q)
    assert identity(training.rng_state(False)) == identity(expected_rng)
    assert next_a['source_ids'] == next_b['source_ids'] and next_a['supervised_tokens'] == next_b['supervised_tokens']
    assert restored_bundle[0].lm_head.weight is restored_bundle[0].model.embedding.weight
    # L2-SP is charged once on all unique parameters, including the bias.
    l2 = fresh(base); reference = {n:p.detach().clone() for n,p in l2[0].named_parameters()}
    training.seed_training(); perform(l2,1,reference=reference)
    expected_omega = float(training.core.l2_sp(l2[0], reference).detach())
    penalty_record = perform(l2,2,reference=reference)
    assert expected_omega > 0 and penalty_record['omega'] == expected_omega and penalty_record['lambda'] == 1e-3
    # Adapter-only updates never change a frozen original weight or a protected basis.
    for protected in (False, True):
        model = TinyLM().train(); model.requires_grad_(False)
        A = torch.randn(4,8)*.02; B = torch.zeros(8,4)
        U = torch.linalg.qr(torch.randn(8,2)).Q if protected else None
        V = torch.linalg.qr(torch.randn(8,2)).Q if protected else None
        adapter = training.core.AdapterLinear(model.model.proj,A,B,U,V,8,.05); model.model.proj = adapter
        frozen = {n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
        named,opt = training.optimizer_for(model,2e-4)
        scaler = torch.amp.GradScaler('cpu')
        coherence = {}; training.seed_training()
        training.effective_step(model,named,opt,scaler,rows(),1,2e-4,0,'cpu',coherence)
        assert all(torch.equal(p,frozen[n]) for n,p in model.named_parameters() if not p.requires_grad)
        assert bool(adapter.B.ne(0).any()) and len(named)==2
        if protected:
            ae,be=adapter.effective_factors();delta=adapter.scale*be@ae
            assert float((torch.linalg.norm(adapter.U.T@delta)/torch.linalg.norm(delta)).detach()) <= 1e-5
            assert float((torch.linalg.norm(delta@adapter.V)/torch.linalg.norm(delta)).detach()) <= 1e-5
            assert torch.equal(adapter.U,U) and torch.equal(adapter.V,V)
    # A single injected gradient overflow uses the SAME dropout draw and no skipped Adam update.
    overflow = fresh(base); clean = fresh(base); hook_calls = [0]
    def once(grad):
        hook_calls[0] += 1
        return torch.full_like(grad, float('inf')) if hook_calls[0] == 1 else grad
    hook = overflow[1][0][1].register_hook(once)
    training.seed_training(); rec = perform(overflow, 1); after_overflow_rng = training.rng_state(False); hook.remove()
    training.seed_training(); clean_rec = perform(clean, 1)
    assert len(rec['overflow_attempts']) == 1 and rec['overflow_attempts'][0]['new_scale'] == 32768.
    assert identity(training.rng_state(False)) == identity(after_overflow_rng)
    for (_, p), (_, q) in zip(overflow[1], clean[1]): assert torch.equal(p, q)
    # Persistent overflow stops after 9 attempts without changing weights or successful steps.
    persistent = fresh(base); p_before = [p.detach().clone() for _, p in persistent[1]]
    count = [0]
    def always(grad): count[0] += 1; return torch.full_like(grad, float('inf'))
    hook = persistent[1][0][1].register_hook(always); coherence = {}; training.seed_training()
    try: training.effective_step(*persistent, rows(), 1, 2e-5, 0, 'cpu', coherence)
    except FloatingPointError: pass
    else: raise AssertionError('Persistent overflow accepted')
    hook.remove()
    assert count[0] == 16 * 9 and len(coherence['overflow_attempts']) == 9 and coherence['coherent']
    assert not persistent[2].state
    for (_, p), before in zip(persistent[1], p_before): assert torch.equal(p, before)
    # Forward errors fail immediately, restore boundary RNG, and do not reduce the scale.
    failed = fresh(base); original_scale = failed[3].get_scale(); training.seed_training(); before_rng = training.rng_state(False)
    import official_training
    with patch.object(official_training.ops, 'chunked_response_sum', side_effect=RuntimeError('CPU_FIXTURE_FORWARD_ERROR')):
        try: training.effective_step(*failed, rows(), 1, 2e-5, 0, 'cpu', {})
        except RuntimeError as e: assert str(e) == 'CPU_FIXTURE_FORWARD_ERROR'
        else: raise AssertionError('Forward exception swallowed')
    assert failed[3].get_scale() == original_scale and identity(training.rng_state(False)) == identity(before_rng)
    # Artificial partial optimizer mutation is marked unsafe; it cannot be emergency-saved.
    unsafe = fresh(base); coherence = {}
    def broken_step(*args, **kwargs):
        with torch.no_grad(): unsafe[1][0][1].add_(1)
        raise RuntimeError('CPU_FIXTURE_PARTIAL_OPTIMIZER_ERROR')
    with patch.object(unsafe[2], 'step', broken_step):
        try: training.effective_step(*unsafe, rows(), 1, 2e-5, 0, 'cpu', coherence)
        except RuntimeError: pass
        else: raise AssertionError('Partial optimizer error swallowed')
    assert coherence['coherent'] is False
    # Update 1250 executes AdamW bookkeeping at zero LR, with no extra update.
    zero = fresh(base)
    for _,p in zero[1]:zero[2].state[p]={'step':torch.tensor(1249.,dtype=torch.float32),'exp_avg':torch.zeros_like(p),'exp_avg_sq':torch.zeros_like(p)}
    zero_before=[p.detach().clone() for _,p in zero[1]]
    last=perform(zero,1250)
    assert last['lr']==0. and all(torch.equal(p,b) for (_,p),b in zip(zero[1],zero_before))
    # Retention touches only completed, owned rolling states for this contract.
    prune_dir = directory/'prune'; prune_dir.mkdir()
    for u, kind in [(50,'rolling'),(100,'rolling'),(150,'rolling'),(312,'mandatory')]:
        folder = prune_dir/f'{kind}_{u}'; folder.mkdir()
        owner = {'workflow':'ATML_PA3_TASK1_TRAINING_V1','method':'full','contract_identity':identity(contract),'kind':kind,'update':u}
        publish(folder/'OWNER.json', owner)
        publish(folder/'COMPLETE.json', {'method':'full','kind':kind,'contract_identity':identity(contract),'successful_updates':u,'created_wall':float(u)})
    (prune_dir/'unrelated').mkdir(); (prune_dir/'incomplete').mkdir()
    prune_rolling(prune_dir,'full',identity(contract)); assert not (prune_dir/'rolling_50').exists()
    assert (prune_dir/'rolling_100').is_dir() and (prune_dir/'rolling_150').is_dir()
    prune_rolling(prune_dir,'full',identity(contract),312)
    assert (prune_dir/'mandatory_312').is_dir() and (prune_dir/'unrelated').is_dir() and (prune_dir/'incomplete').is_dir()
    # Archive corruption is rejected before recovery can be called complete.
    with (saved/'state.tar').open('ab') as stream: stream.write(b'CORRUPTION_FIXTURE')
    bad = fresh(base)
    try: restore_checkpoint(saved,contract,bad[0],bad[1],bad[2],bad[3])
    except AssertionError: pass
    else: raise AssertionError('Changed completed archive accepted')
    return {'token_loss_and_projection_checks': official_training.core.cpu_math_checks(),
        'chunked_head_gradient_checks': official_training.ops.check_chunked_gradients(),
        'sharded_tensor_and_optimizer_values_reload_exactly': True,
        'Python_NumPy_torch_RNG_and_scaler_reload_exactly': True,
        'actual_next_update_CPU_continuation_matches': True,
        'tied_embedding_and_head_preserved_on_reload': True,
        'L2_SP_once_per_effective_batch_and_frozen_reference_checked': True,
        'LoRA_and_OPLoRA_freeze_backbone_and_preserve_protected_bases': True,
        'last_update_zero_LR_still_executes_exactly_one_Adam_step': True,
        'single_overflow_retries_same_batch_and_RNG_without_extra_update': True,
        'persistent_overflow_stops_after_8_retries_plus_initial_attempt': True,
        'forward_error_stops_immediately_and_restores_boundary': True,
        'partially_executed_optimizer_is_not_coherent': True,
        'rolling_retention_preserves_mandatory_unrelated_and_incomplete_data': True,
        'changed_checkpoint_archive_rejected': True,
        'fixture_torch_version': torch.__version__, 'GPU_work_run': False,
        'official_training_updates': 0}
