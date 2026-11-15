"""Whole physical-batch journals preserve padding and membership on resume."""
import json
from pathlib import Path

from evaluation_records import identity, publish


def load_batch(path, contract):
    path = Path(path)
    if not path.exists(): return None
    value = json.loads(path.read_text())
    checksum = value.pop('content_sha256')
    if identity(value) != checksum or value['contract'] != contract:
        raise RuntimeError('Batch cache content/contract differs')
    rows = value['sequence_ids']; prompts = contract['padded_input_ids']
    if len(rows) != len(prompts) or len(value['row_metadata']) != len(rows):
        raise RuntimeError('Batch cache row coverage differs')
    width = len(prompts[0])
    for row, prompt in zip(rows, prompts):
        if row[:width] != prompt or any(type(t) is not int or t < 0 for t in row):
            raise RuntimeError('Batch cache tokens/prefix differ')
        if not width <= len(row) <= contract['max_length']:
            raise RuntimeError('Cached length exceeds fixed allowance')
    return value


def save_batch(path, contract, sequence_ids, row_metadata, elapsed):
    value = {'contract': contract, 'sequence_ids': sequence_ids,
             'row_metadata': row_metadata, 'shared_model_inference_seconds': elapsed}
    publish(path, {**value, 'content_sha256': identity(value)})
    return load_batch(path, contract)


def cache_checks(directory):
    directory = Path(directory); directory.mkdir(parents=True)
    contract = {'padded_input_ids': [[0, 2], [3, 4]], 'max_length': 4,
                'audit': {'fixture': True}}
    path = directory / (identity(contract) + '.json')
    assert load_batch(path, contract) is None
    first = save_batch(path, contract, [[0, 2, 8, 9], [3, 4, 9, 0]], [{}, {}], .1)
    assert load_batch(path, contract) == first
    checks = {'whole_batch_commit_and_reuse': True}
    try: load_batch(path, {**contract, 'padded_input_ids': [[3, 4], [0, 2]]})
    except RuntimeError: checks['membership_and_order_frozen'] = True
    else: raise AssertionError('Changed membership was accepted')
    changed = json.loads(path.read_text()); changed['sequence_ids'][0][-1] = 6
    path.write_text(json.dumps(changed))
    try: load_batch(path, contract)
    except RuntimeError: checks['changed_tokens_rejected'] = True
    else: raise AssertionError('Changed tokens were accepted')
    pending = directory / 'uncommitted.json.pending'; pending.write_text('{}')
    assert load_batch(directory / 'uncommitted.json', contract) is None
    checks['uncommitted_file_is_not_an_answer'] = True
    return checks
