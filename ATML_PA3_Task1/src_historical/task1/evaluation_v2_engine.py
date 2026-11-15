"""Resume entire fixed physical chunks; never regroup a partially scored chunk."""
import json
import time
from pathlib import Path

from evaluation_records import encoded, identity, publish
from stock_evaluation import score, serialized_requests


def run_chunk(task, name, prepared, store, infer, journal, audit,
              instance_factory=None, scorer=score):
    if instance_factory is None:
        from lm_eval.api.instance import Instance
        instance_factory = Instance
    completed, groups, missing = {}, [], []
    for row in prepared:
        request_contract = {'source_index': row['source_index'], 'requests': row['requests']}
        previous = store.get(name, row['source_index'], request_contract)
        if previous is not None: completed[row['source_index']] = previous
        else: missing.append(row['source_index'])
        requests = [instance_factory(request_type=req['request_type'], doc=row['document'],
            arguments=tuple(req['arguments']), idx=req['idx'], metadata=(name, row['source_index'], 1))
            for req in row['requests']]
        assert identity(serialized_requests(requests)) == identity(row['requests'])
        groups.append((row, requests, request_contract))
    if not missing:
        return [completed[row['source_index']] for row in prepared]
    flat = [req for _, requests, _ in groups for req in requests]
    kinds = {r.request_type for r in flat}; assert len(kinds) == 1
    kind = kinds.pop()
    # Generation's native physical batches are journaled by BatchJournalHFLM.
    # Likelihood has its own atomic complete-chunk raw-response journal.
    contract = {'audit': audit, 'task': name,
                'source_ids': [row['source_index'] for row in prepared],
                'requests': serialized_requests(flat)}
    # Native Instance.args are tuples; JSON journals restore them as lists.
    # Normalize before both the initial write and resumed equality check.
    contract = json.loads(encoded(contract))
    path = Path(journal) / (identity(contract) + '.json')
    if kind == 'loglikelihood' and path.exists():
        cached = json.loads(path.read_text()); checksum = cached.pop('content_sha256')
        assert identity(cached) == checksum and cached['contract'] == contract
        responses, metadata = cached['responses'], cached['metadata']
    else:
        responses, metadata = infer(kind, flat)
        if kind == 'loglikelihood':
            cached = {'contract': contract, 'responses': responses, 'metadata': metadata}
            publish(path, {**cached, 'content_sha256': identity(cached)})
    assert len(responses) == len(metadata) == len(flat)
    offset = 0
    for row, requests, request_contract in groups:
        end = offset + len(requests)
        if row['source_index'] not in completed:
            metrics, filtered = scorer(task, requests, responses[offset:end])
            completed[row['source_index']] = store.put(name, row['source_index'], request_contract,
                responses[offset:end], metrics, {'requests': metadata[offset:end], 'filtered': filtered})
        offset = end
    return [completed[row['source_index']] for row in prepared]
