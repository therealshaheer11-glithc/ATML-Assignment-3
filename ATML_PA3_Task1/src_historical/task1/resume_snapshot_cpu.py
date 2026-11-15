"""CPU-only audit and packed transport of existing V2 answers. No inference."""
import concurrent.futures
import gzip
import hashlib
import json
import math
import time
import zipfile
from pathlib import Path

from archive_io import digest, safe_relative
from evaluation_records import COUNTS, aggregate, encoded, identity
from batch8_records import load_batch


def read_parallel(paths, action, label):
    paths = list(paths)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        # Bounded outstanding work; no thousands of decoded responses in a queue.
        for start in range(0, len(paths), 128):
            for result in pool.map(lambda p: (p, p.read_bytes()), paths[start:start + 128]):
                action(*result)
            if start == 0 or (start + 128) // 1024 != start // 1024 or start + 128 >= len(paths):
                print(f'{label}: {min(start + 128, len(paths))}/{len(paths)} files checked', flush=True)


def validate_answer(record, task, source_index, row, contract):
    assert record['contract'] == contract and record['status'] == 'complete'
    assert record['task'] == task and record['source_index'] == source_index
    assert record['request_identity'] == identity({'source_index': source_index, 'requests': row['requests']})
    assert record['scores'] and all(type(v) in (int, float) and math.isfinite(v) for v in record['scores'].values())
    assert len(record['response']) == len(row['requests'])


def pack_records(root, local, contract, implementation, protocol, minimum_counts):
    root, local = Path(root), Path(local)
    local.mkdir(parents=True, exist_ok=True)
    namespace = 'results_batch8_V2/original'
    base = root / namespace
    audit = {**contract, 'gpu_implementation_identity': implementation}
    ready = json.loads((root / 'configs/BASELINE_CPU_READY_V1.json').read_text())
    archive_path = local / 'verified_resume.zip'
    index = {'schema_version': 1, 'contract': contract, 'numerical_implementation_identity': implementation,
             'created_utc_unix': time.time(), 'files': {}, 'counts': {}, 'complete_metrics': {},
             'raw_generation_batches': 0, 'raw_likelihood_chunks': 0,
             'scope': 'Verified existing bytes only; no inference, training or per-answer score recomputation'}
    rows_by_task = {}
    with zipfile.ZipFile(archive_path, 'x', zipfile.ZIP_DEFLATED) as archive:
        def add(rel, data):
            safe_relative(rel)
            assert rel not in index['files']
            archive.writestr(rel, data)
            index['files'][rel] = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}

        assert json.loads((base / 'CONTRACT.json').read_text()) == contract
        add(namespace + '/CONTRACT.json', (base / 'CONTRACT.json').read_bytes())
        for task, count in COUNTS.items():
            entry = ready['benchmarks'][task]
            path = root / str(safe_relative(entry['path']))
            assert digest(path) == entry['sha256']
            with gzip.open(path, 'rt') as stream: rows = [json.loads(line) for line in stream]
            assert [r['source_index'] for r in rows] == list(range(count))
            rows_by_task[task] = rows
            records = {}

            def accept(path, data):
                i = int(path.stem)
                assert path.name == f'{i:05}.json' and 0 <= i < count and i not in records
                record = json.loads(data)
                validate_answer(record, task, i, rows[i], contract)
                expected_metrics = {protocol['primary_metrics'][task]}
                if task in protocol['secondary_metrics']: expected_metrics.add(protocol['secondary_metrics'][task])
                assert expected_metrics.issubset(record['scores'])
                records[i] = record
                add(str(path.relative_to(root)), data)

            started = time.monotonic()
            read_parallel(sorted((base / task).glob('*.json')), accept, task)
            index.setdefault('record_read_and_check_seconds', {})[task] = time.monotonic() - started
            index['counts'][task] = len(records)
            assert len(records) >= minimum_counts[task], f'{task}: fewer saved records than previously reported'
            if len(records) == count:
                metrics = aggregate([records[i] for i in range(count)], task)
                completion = base / f'{task}_COMPLETE.json'
                saved = json.loads(completion.read_text())
                assert saved == {'audit': audit, 'count': count, 'metrics': metrics}
                index['complete_metrics'][task] = metrics
                add(str(completion.relative_to(root)), completion.read_bytes())
            else:
                assert not (base / f'{task}_COMPLETE.json').exists(), 'Incomplete split has a completion receipt'
                index.setdefault('partial_score_totals', {})[task] = {
                    k: sum(r['scores'][k] for r in records.values()) for k in next(iter(records.values()))['scores']
                } if records else {}

        # The raw cache is preserved too, including work from any unfinished chunk.
        prompt_ids = {tuple(r['prompt_input_ids']) for r in rows_by_task['gsm8k_cot']}
        def raw_generation(path, data):
            value = json.loads(data); c = value['contract']
            assert path.stem == identity(c) and c['audit'] == audit
            # Production reader verifies checksum, tokens, row coverage and prefixes.
            checked = load_batch(path, c)
            prompts = c['padded_input_ids']; width = len(prompts[0])
            assert 1 <= len(prompts) <= 8 and all(len(p) == width for p in prompts)
            assert c['max_length'] - width == 1024 and c['generation_kwargs']['do_sample'] is False
            for sequence, meta in zip(checked['sequence_ids'], checked['row_metadata']):
                assert tuple(meta['input_ids']) in prompt_ids and meta['physical_batch_identity'] == path.stem
                assert meta['physical_batch_size'] == len(prompts) and meta['max_new_tokens'] == 1024
                new = sequence[width:]; semantic = meta['semantic_new_ids']
                assert semantic == new[:len(semantic)] and meta['raw_generated_tokens'] == len(new)
                assert meta['semantic_generated_tokens'] == len(semantic)
                assert meta['at_token_cap'] == (len(semantic) == 1024)
            index['raw_generation_batches'] += 1
            add(str(path.relative_to(root)), data)
        read_parallel(sorted((base / 'raw_generation_batches').glob('*.json')), raw_generation, 'raw generation batches')

        def raw_likelihood(path, data):
            value = json.loads(data); checksum = value.pop('content_sha256')
            assert identity(value) == checksum
            c = value['contract']; assert path.stem == identity(c) and c['audit'] == audit
            name = c['task']; assert name in ('hellaswag', 'boolq', 'arc_easy')
            ids = c['source_ids']; assert ids and len(ids) <= 100 and ids[0] % 100 == 0
            assert ids == list(range(ids[0], min(ids[0] + 100, COUNTS[name])))
            expected = [r for i in ids for r in rows_by_task[name][i]['requests']]
            assert c['requests'] == expected
            assert len(value['responses']) == len(value['metadata']) == len(expected)
            assert all(len(r) == 2 and type(r[1]) is bool and math.isfinite(r[0]) for r in value['responses'])
            index['raw_likelihood_chunks'] += 1
            add(str(path.relative_to(root)), data)
        read_parallel(sorted((base / 'raw_likelihood_chunks').glob('*.json')), raw_likelihood, 'raw likelihood chunks')
        index_bytes = encoded(index)
        archive.writestr('SNAPSHOT_INDEX.json', index_bytes)
    # Independent ZIP readback checks every member against its captured byte hash.
    with zipfile.ZipFile(archive_path) as archive:
        assert len(archive.namelist()) == len(index['files']) + 1
        assert archive.read('SNAPSHOT_INDEX.json') == index_bytes
        for rel, entry in index['files'].items():
            data = archive.read(rel)
            assert len(data) == entry['bytes'] and hashlib.sha256(data).hexdigest() == entry['sha256']
    return archive_path, index
