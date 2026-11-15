"""Verified immutable evaluation records; a missing answer is never a cache hit."""
import hashlib
import json
import math
import uuid
from pathlib import Path


COUNTS = {'gsm8k_cot': 1319, 'hellaswag': 10042, 'boolq': 3270, 'arc_easy': 2376}
CONDITIONS = ('original', 'full_mid', 'full_final', 'l2sp_final',
              'lora4_final', 'lora16_final', 'oplora_final')


def encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False,
                       separators=(',', ':'), allow_nan=False) + '\n').encode()


def identity(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def publish(path, value):
    """Read back both pending and final bytes; preserve conflicting prior evidence."""
    path = Path(path)
    data = encoded(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise RuntimeError('Linked result path')
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError('Conflicting saved result; preserving it: ' + str(path))
        return
    pending = path.with_name(path.name + '.pending_' + uuid.uuid4().hex)
    pending.write_bytes(data)
    if pending.read_bytes() != data:
        raise IOError('Pending result readback differs')
    if path.exists():
        raise RuntimeError('Concurrent writer; run one Task 1 job at a time')
    pending.rename(path)
    if path.read_bytes() != data:
        raise IOError('Final result readback differs')


class RecordStore:
    def __init__(self, directory, condition, model_identity, protocol_identity):
        if condition not in CONDITIONS:
            raise ValueError('Unknown evaluation condition')
        self.directory = Path(directory) / condition
        self.contract = {'condition': condition, 'model_identity': model_identity,
                         'protocol_identity': protocol_identity}
        publish(self.directory / 'CONTRACT.json', self.contract)

    def path(self, task, source_index):
        if task not in COUNTS or not isinstance(source_index, int) or not 0 <= source_index < COUNTS[task]:
            raise ValueError('Invalid canonical evaluation ID')
        return self.directory / task / (f'{source_index:05}.json')

    def get(self, task, source_index, request):
        path = self.path(task, source_index)
        if not path.exists():
            return None
        record = json.loads(path.read_bytes())
        if (record['contract'] != self.contract or record['task'] != task
                or record['source_index'] != source_index or record['request_identity'] != identity(request)):
            raise RuntimeError('Saved result identity differs')
        if record['status'] != 'complete':
            raise RuntimeError('Incomplete record cannot count as an answer')
        return record

    def put(self, task, source_index, request, response, scores, metadata):
        # Empty generated strings ARE completed answers. Errors/timeouts are not.
        for key, value in scores.items():
            if not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError('Nonfinite or nonscalar score: ' + key)
        record = {'contract': self.contract, 'task': task, 'source_index': source_index,
                  'request_identity': identity(request), 'status': 'complete',
                  'response': response, 'scores': scores, 'metadata': metadata}
        publish(self.path(task, source_index), record)
        return self.get(task, source_index, request)


def aggregate(records, task):
    """Only a complete, unique, canonical split can produce the final headline."""
    expected = COUNTS[task]
    ids = [record['source_index'] for record in records]
    if len(ids) != expected or set(ids) != set(range(expected)):
        raise ValueError('Split is incomplete or has duplicated IDs')
    if any(r['status'] != 'complete' or r['task'] != task for r in records):
        raise ValueError('Invalid completion/task')
    if len({identity(r['contract']) for r in records}) != 1:
        raise ValueError('Mixed model/protocol conditions')
    keys = set(records[0]['scores'])
    if not keys or any(set(r['scores']) != keys for r in records):
        raise ValueError('Metric coverage differs')
    return {key: sum(r['scores'][key] for r in records) / expected for key in sorted(keys)}


def cpu_checks(directory):
    directory = Path(directory)
    store = RecordStore(directory, 'original', 'model-A', 'protocol-A')
    request = {'prompt': 'fixture', 'cap': 1024}
    assert store.get('gsm8k_cot', 0, request) is None
    record = store.put('gsm8k_cot', 0, request, '', {'exact_match,flexible-extract': 0.},
                       {'at_token_cap': True, 'fixture_only': True})
    assert store.get('gsm8k_cot', 0, request) == record
    store.put('gsm8k_cot', 0, request, '', record['scores'], record['metadata'])
    checks = {'empty_completed_answer_retained': True, 'completed_record_reused': True}
    attempts = {
        'different_prompt_rejected': lambda: store.get('gsm8k_cot', 0, {'prompt': 'changed'}),
        'different_model_rejected': lambda: RecordStore(directory, 'original', 'model-B', 'protocol-A'),
        'different_protocol_rejected': lambda: RecordStore(directory, 'original', 'model-A', 'protocol-B'),
        'conflicting_answer_preserved': lambda: store.put('gsm8k_cot', 0, request, '42', record['scores'], {}),
        'nonfinite_score_rejected': lambda: store.put('gsm8k_cot', 1, request, '42', {'x': float('nan')}, {}),
        'incomplete_denominator_rejected': lambda: aggregate([record], 'gsm8k_cot'),
    }
    for name, invoke in attempts.items():
        try:
            invoke()
        except (ValueError, RuntimeError):
            checks[name] = True
        else:
            raise AssertionError(name)
    other = RecordStore(directory, 'full_final', 'model-C', 'protocol-A')
    assert other.get('gsm8k_cot', 0, request) is None
    checks['conditions_do_not_share_answers'] = True
    return checks
