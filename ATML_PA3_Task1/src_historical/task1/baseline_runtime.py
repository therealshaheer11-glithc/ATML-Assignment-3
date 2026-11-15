"""Standard-library job state, save boundaries, and generation audit helpers."""
import hashlib
import json
import time
import uuid
from pathlib import Path

from evaluation_records import encoded, identity, publish


class SoftStop(Exception):
    """The current operation has not started; release after saving job progress."""


def atomic(path, value):
    path = Path(path)
    data = encoded(value)
    pending = path.with_name(path.name + '.pending_' + uuid.uuid4().hex)
    pending.write_bytes(data)
    pending.replace(path)


def progress(state, phase, saving=False):
    path = Path(state)
    record = json.loads(path.read_text())
    if time.time() + 10 >= record['deadline_wall']:
        raise TimeoutError('Absolute job deadline reached during ' + phase)
    record.update(status='running', phase=phase, progress_wall=time.time(),
                  progress_timeout=1800 if saving else 900)
    atomic(path, record)


def boundary(config, phase):
    if time.time() >= config['soft_wall']:
        raise SoftStop('Saved-work boundary reached before ' + phase)
    progress(config['state'], phase)


def wait_for_parent(state_path):
    import os
    until = time.monotonic() + 10
    while time.monotonic() < until:
        if json.loads(Path(state_path).read_text()).get('worker_pid') == os.getpid():
            return
        time.sleep(.02)
    raise RuntimeError('Parent did not publish the worker process identity')


def publish_bytes(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink(): raise RuntimeError('Unexpected linked result')
    if path.exists():
        if path.read_bytes() != data: raise RuntimeError('Conflicting preserved bytes: ' + str(path))
    else:
        tmp = path.with_name(path.name + '.pending_' + uuid.uuid4().hex)
        tmp.write_bytes(data)
        if tmp.read_bytes() != data: raise IOError('Pending byte readback differs')
        if path.exists(): raise RuntimeError('Concurrent writer')
        tmp.rename(path)
    if path.read_bytes() != data: raise IOError('Final byte readback differs')
    return hashlib.sha256(data).hexdigest()


def termination(new_ids, eos_ids, stop_strings, decoded_with_specials, cap):
    if len(new_ids) > cap: raise ValueError('Generation exceeded the fixed token budget')
    ended_eos = bool(new_ids and new_ids[-1] in set(eos_ids))
    stop_present = any(stop and stop in decoded_with_specials for stop in stop_strings)
    at_cap = len(new_ids) == cap
    return {'generated_tokens': len(new_ids), 'max_new_tokens': cap,
        'at_token_cap': at_cap, 'ended_with_eos': ended_eos,
        'stock_stop_string_present': bool(stop_present),
        'truncated': bool(at_cap and not ended_eos and not stop_present)}


def load_generation(path, contract):
    path = Path(path)
    if not path.exists(): return None
    record = json.loads(path.read_text())
    checksum = record.pop('content_sha256')
    if identity(record) != checksum or record['contract'] != contract:
        raise RuntimeError('Raw generation identity/content differs')
    sequence = record['sequence_ids']
    if len(sequence) != 1 or sequence[0][:len(contract['input_ids'])] != contract['input_ids']:
        raise RuntimeError('Raw generation prompt does not match')
    if any(type(t) is not int or t < 0 for t in sequence[0]):
        raise RuntimeError('Invalid saved token IDs')
    return record


def save_generation(path, contract, sequence_ids, metadata):
    record = {'contract': contract, 'sequence_ids': sequence_ids, 'metadata': metadata}
    publish(path, {**record, 'content_sha256': identity(record)})
    return load_generation(path, contract)
