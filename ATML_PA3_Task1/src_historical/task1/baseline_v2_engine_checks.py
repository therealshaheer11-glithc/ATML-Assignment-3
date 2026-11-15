"""CPU bookkeeping tests; native inference/scoring are separately CPU/GPU checked."""
import json
from pathlib import Path

from evaluation_records import RecordStore, identity
from evaluation_v2_engine import run_chunk


class Request:
    def __init__(self, request_type, doc, arguments, idx, metadata):
        self.request_type, self.doc, self.args, self.idx = request_type, doc, arguments, idx
        self.doc_id = metadata[1]


class InterruptedStore(RecordStore):
    remaining = None

    def put(self, *args, **kwargs):
        if self.remaining is not None:
            if self.remaining == 0: raise RuntimeError('INTENTIONAL_SCORE_SAVE_INTERRUPTION')
            self.remaining -= 1
        return super().put(*args, **kwargs)


def fixture_score(task, requests, responses):
    return {'fixture': float(requests[0].doc_id % 2)}, {'fixture': responses}


def checks(directory):
    directory = Path(directory); results = {}
    for name, kind in (('gsm8k_cot', 'generate_until'), ('hellaswag', 'loglikelihood')):
        root = directory / name
        rows = [{'source_index': i, 'document': {'CPU_FIXTURE_ONLY': True},
            'requests': [{'request_type': kind, 'arguments': ['fixture ' + str(i), {}] if kind == 'generate_until'
                         else ['fixture ' + str(i), ' choice'], 'idx': 0}]} for i in range(25)]
        store = InterruptedStore(root / 'records', 'original', 'CPU_FIXTURE_MODEL', 'CPU_FIXTURE_PROTOCOL')
        source_calls, physical_calls, physical_cache = [], [], {}
        def infer(request_kind, requests):
            ids = [r.doc_id for r in requests]; source_calls.append(ids)
            assert ids == list(range(25)), 'A partial chunk was regrouped'
            key = identity(ids)
            if key not in physical_cache:
                physical_calls.append(ids)
                physical_cache[key] = ([('' if i == 0 else 'fixture') for i in ids]
                    if kind == 'generate_until' else [(-float(i + 1), False) for i in ids],
                    [{'fixture_only': True} for _ in ids])
            return physical_cache[key]
        store.remaining = 6
        try: run_chunk(None, name, rows, store, infer, root / 'journal', {'fixture': True}, Request, fixture_score)
        except RuntimeError as error: assert str(error) == 'INTENTIONAL_SCORE_SAVE_INTERRUPTION'
        else: raise AssertionError('Interruption was not triggered')
        store.remaining = None
        recovered = run_chunk(None, name, rows, store, infer, root / 'journal', {'fixture': True}, Request, fixture_score)
        assert len(recovered) == 25 and [r['source_index'] for r in recovered] == list(range(25))
        assert len(physical_calls) == 1
        again = run_chunk(None, name, rows, store, infer, root / 'journal', {'fixture': True}, Request, fixture_score)
        assert recovered == again
        if kind == 'generate_until': assert source_calls == [list(range(25)), list(range(25))]
        else:
            assert len(source_calls) == 1
            journal = next((root / 'journal').glob('*.json'))
            value = json.loads(journal.read_text()); value['responses'][0][0] -= 1
            journal.write_text(json.dumps(value))
            store.path(name, 24).unlink()  # Only this temporary CPU fixture's file.
            try: run_chunk(None, name, rows, store, infer, root / 'journal', {'fixture': True}, Request, fixture_score)
            except AssertionError: pass
            else: raise AssertionError('Edited raw likelihood journal was accepted')
        results[name] = {'partial_score_writes_recover_all_25_IDs': True,
            'fixed_full_chunk_membership_preserved': True,
            'no_new_physical_inference_after_cache_commit': True,
            'complete_chunk_reused': True}
    results['likelihood_chunk_corruption_rejected'] = True
    results['scope'] = 'CPU fixture control flow only; no original-model inference or metric fabrication'
    return results
