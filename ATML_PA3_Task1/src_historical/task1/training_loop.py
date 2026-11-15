"""One method per bounded job; save only coherent completed-update states."""
import json
import time
from pathlib import Path

import torch

from baseline_runtime import progress
from evaluation_records import identity
from official_training import MANDATORY, effective_step, restore_rng
from training_checkpoints import save_checkpoint, prune_rolling


def run_updates(config, model, named, optimizer, scaler, data, peak, pad_id, reference=None,
                restored=None, clock=time.time):
    root, local = Path(config['root']), Path(config['job'])
    contract = config['training_contract']; method = contract['method']
    directory = root / 'checkpoints' / method / identity(contract)
    state = {'update': restored['successful_updates'] if restored else 0,
             'history': list(restored['history']) if restored else []}
    next_mandatory = next((u for u in MANDATORY if u > state['update']), None)
    if next_mandatory is None: return {'status': 'ALREADY_TRAINED', 'successful_updates': 1250}
    coherence = {'coherent': True}; last_checkpoint = config.get('resume_checkpoint'); last_saved = -1
    device = next(model.parameters()).device
    timings = {'training_seconds': 0., 'checkpoint_seconds': 0.}
    def beat(phase, saving=False): progress(config['state'], phase, saving=saving)
    def checkpoint(kind):
        nonlocal last_checkpoint, last_saved
        assert coherence['coherent'] and len(state['history']) == state['update']
        optimizer.zero_grad(set_to_none=True)
        start = time.monotonic()
        path, marker = save_checkpoint(directory, local, method, contract, state['update'], state['history'],
            named, optimizer, scaler, device.type == 'cuda', kind,
            progress=lambda phase: beat(phase, True))
        timings['checkpoint_seconds'] += time.monotonic() - start
        last_checkpoint, last_saved = str(path.relative_to(root)), state['update']
        removed = prune_rolling(directory, method, identity(contract), state['update'] if kind == 'mandatory' else None)
        print(json.dumps({'checkpoint_verified': last_checkpoint, 'successful_updates': state['update'],
              'examples_seen': state['update']*32, 'kind': kind, 'bytes': marker['archive_bytes'],
              'obsolete_owned_rolling_checkpoints_removed': removed}), flush=True)
    try:
        while state['update'] < next_mandatory:
            if clock() >= config['soft_wall']:
                if last_saved != state['update']: checkpoint('exit')
                return {'status': 'TRAINING_PARTIAL_SAVED', 'successful_updates': state['update'],
                        'checkpoint': last_checkpoint, 'timings': timings, 'stop_reason': 'soft_limit_at_complete_update'}
            update = state['update'] + 1
            beat('load fixed source-order effective batch ' + str(update))
            rows = data.batch(update)
            start = time.monotonic()
            record = effective_step(model, named, optimizer, scaler, rows, update, peak, pad_id, device,
                                    coherence, reference, lambda phase: beat(phase))
            state['history'].append(record); state['update'] = update; coherence['coherent'] = True
            coherence['pre_update_rng'] = None
            timings['training_seconds'] += time.monotonic() - start
            print(json.dumps(record), flush=True)
            if update in MANDATORY:
                checkpoint('mandatory')
                return {'status': 'TRAINING_FINAL_SAVED' if update == 1250 else 'TRAINING_MANDATORY_SAVED',
                        'successful_updates': update, 'checkpoint': last_checkpoint, 'timings': timings,
                        'FullFT_midpoint_evaluation_due': method == 'full' and update == 625}
            if update % 50 == 0: checkpoint('rolling')
        raise AssertionError('Missing mandatory training boundary')
    except BaseException as error:
        if coherence['coherent']:
            # A forward/backward failure can leave partial gradients and consumed RNG,
            # but no parameter update. Return to the previous completed-update boundary.
            if coherence.get('pre_update_rng') is not None: restore_rng(coherence['pre_update_rng'])
            optimizer.zero_grad(set_to_none=True)
            try: checkpoint('error')
            except BaseException as save_error:
                error.add_note('Emergency save failed: ' + type(save_error).__name__ + ': ' + str(save_error))
        else:
            error.add_note('Optimizer mutation may be partial. No emergency checkpoint was saved; use the last verified checkpoint.')
        error.add_note('last_verified_checkpoint=' + str(last_checkpoint))
        raise
