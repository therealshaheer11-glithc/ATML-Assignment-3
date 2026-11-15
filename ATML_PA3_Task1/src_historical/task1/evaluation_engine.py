"""Resumable chunks delegated to native harness inference and scoring functions."""
from evaluation_records import identity
from stock_evaluation import score, serialized_requests


def run_chunk(task, name, prepared, store, infer):
    """infer(type, native Instances) returns responses and per-request metadata.

    The caller supplies the unchanged HFLM functions in a GPU job. A CPU fixture
    supplies known answers to test bookkeeping; those are never official results.
    """
    from lm_eval.api.instance import Instance
    completed, missing = [], []
    for row in prepared:
        request_contract = {'source_index': row['source_index'], 'requests': row['requests']}
        previous = store.get(name, row['source_index'], request_contract)
        if previous is not None:
            completed.append(previous)
            continue
        requests = [Instance(request_type=r['request_type'], doc=row['document'],
            arguments=tuple(r['arguments']), idx=r['idx'],
            metadata=(name, row['source_index'], 1)) for r in row['requests']]
        assert identity(serialized_requests(requests)) == identity(row['requests'])
        missing.append((row, requests, request_contract))
    if missing:
        flat = [request for _, requests, _ in missing for request in requests]
        kinds = {r.request_type for r in flat}
        assert len(kinds) == 1
        responses, metadata = infer(kinds.pop(), flat)
        assert len(responses) == len(metadata) == len(flat)
        start = 0
        for row, requests, request_contract in missing:
            end = start + len(requests)
            values = responses[start:end]
            metrics, filtered = score(task, requests, values)
            completed.append(store.put(name, row['source_index'], request_contract,
                values, metrics, {'requests': metadata[start:end], 'filtered': filtered}))
            start = end
    return sorted(completed, key=lambda r: r['source_index'])
