"""Transport tests with explicit synthetic data; no experimental model outputs."""
import copy
import gzip
import json
import tempfile
import zipfile
from pathlib import Path

from evaluation_records import COUNTS, encoded, identity, aggregate
from resume_snapshot_cpu import pack_records, validate_answer


def checks(directory):
    root = Path(directory) / 'root'; base = root / 'results_batch8_V2/original'
    base.mkdir(parents=True)
    def save(rel, value):
        path = root / rel; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(encoded(value)); return path
    contract = {'condition': 'original', 'model_identity': 'CPU_FIXTURE_ONLY', 'protocol_identity': 'CPU_FIXTURE_ONLY'}
    implementation = 'CPU_FIXTURE_ONLY'; audit = {**contract, 'gpu_implementation_identity': implementation}
    save('results_batch8_V2/original/CONTRACT.json', contract)
    manifest, rows_by_task, originals = {}, {}, {}
    minimum = {**COUNTS, 'gsm8k_cot': 850}
    for task, count in COUNTS.items():
        rows = [{'source_index': i, 'prompt_input_ids': [100 + i],
                 'requests': [{'request_type': 'generate_until' if task == 'gsm8k_cot' else 'loglikelihood',
                               'arguments': ['CPU_FIXTURE '+str(i), {} if task == 'gsm8k_cot' else ' choice'], 'idx': 0}]}
                for i in range(count)]
        rows_by_task[task] = rows
        rel = 'inputs/' + task + '.jsonl.gz'; path = root / rel; path.parent.mkdir(exist_ok=True)
        with gzip.open(path, 'wb') as stream:
            for row in rows: stream.write(encoded(row))
        import hashlib
        manifest[task] = {'path': rel, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        records = []
        for i in range(minimum[task]):
            row = rows[i]
            record = {'contract': contract, 'status': 'complete', 'task': task, 'source_index': i,
                      'request_identity': identity({'source_index': i, 'requests': row['requests']}),
                      'response': ['CPU_FIXTURE_ONLY'] if task == 'gsm8k_cot' else [[-1., False]],
                      'scores': {'fixture_metric': float(i % 2)}, 'metadata': {'CPU_FIXTURE_ONLY': True}}
            path = save(f'results_batch8_V2/original/{task}/{i:05}.json', record)
            originals[str(path.relative_to(root))] = path.read_bytes(); records.append(record)
        if len(records) == count:
            save(f'results_batch8_V2/original/{task}_COMPLETE.json',
                 {'audit': audit, 'count': count, 'metrics': aggregate(records, task)})
    save('configs/BASELINE_CPU_READY_V1.json', {'benchmarks': manifest})
    c = {'audit': audit, 'padded_input_ids': [[950], [951]], 'max_length': 1025,
         'generation_kwargs': {'do_sample': False}}
    meta = [{'input_ids': prefix, 'physical_batch_identity': identity(c), 'physical_batch_size': 2,
             'max_new_tokens': 1024, 'semantic_new_ids': [1, 2], 'raw_generated_tokens': 2,
             'semantic_generated_tokens': 2, 'at_token_cap': False} for prefix in c['padded_input_ids']]
    value = {'contract': c, 'sequence_ids': [[950, 1, 2], [951, 1, 2]], 'row_metadata': meta,
             'shared_model_inference_seconds': .1}
    save('results_batch8_V2/original/raw_generation_batches/'+identity(c)+'.json',
         {**value, 'content_sha256': identity(value)})
    c = {'audit': audit, 'task': 'hellaswag', 'source_ids': list(range(100)),
         'requests': [r for row in rows_by_task['hellaswag'][:100] for r in row['requests']]}
    value = {'contract': c, 'responses': [[-1., False] for _ in range(100)], 'metadata': [{} for _ in range(100)]}
    save('results_batch8_V2/original/raw_likelihood_chunks/'+identity(c)+'.json',
         {**value, 'content_sha256': identity(value)})
    protocol = {'primary_metrics': {t: 'fixture_metric' for t in COUNTS}, 'secondary_metrics': {}}
    archive, index = pack_records(root, Path(directory)/'local', contract, implementation, protocol, minimum)
    assert index['counts'] == minimum and index['raw_generation_batches'] == index['raw_likelihood_chunks'] == 1
    assert set(index['complete_metrics']) == {'hellaswag', 'boolq', 'arc_easy'}
    with zipfile.ZipFile(archive) as z:
        assert json.loads(z.read('SNAPSHOT_INDEX.json')) == index
        for rel, payload in originals.items(): assert z.read(rel) == payload == (root/rel).read_bytes()
    sample = json.loads(next(iter(originals.values())))
    changed = copy.deepcopy(sample); changed['request_identity'] = 'changed'
    try: validate_answer(changed, sample['task'], sample['source_index'], rows_by_task[sample['task']][sample['source_index']], contract)
    except AssertionError: pass
    else: raise AssertionError('Changed request accepted')
    changed = copy.deepcopy(sample); changed['scores']['fixture_metric'] = float('nan')
    try: validate_answer(changed, sample['task'], sample['source_index'], rows_by_task[sample['task']][sample['source_index']], contract)
    except AssertionError: pass
    else: raise AssertionError('Nonfinite score accepted')
    return {'all_16538_synthetic_records_checked_and_packed': True,
            'all_source_answer_bytes_preserved': True, 'three_complete_means_reaggregated': True,
            'partial_GSM_not_declared_complete': True, 'unfinished_chunk_raw_batch_retained': True,
            'raw_likelihood_checksum_and_group_checked': True, 'request_change_rejected': True,
            'nonfinite_score_rejected': True, 'ZIP_all_members_hash_readback_passed': True,
            'scope': 'Synthetic CPU transport tests only; no actual Drive audit or original-model inference'}
