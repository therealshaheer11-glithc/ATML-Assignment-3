"""Replay frozen stock scoring and export a complete baseline without inference."""
import gzip
import hashlib
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from archive_io import digest, safe_relative
from evaluation_records import COUNTS, encoded, identity, publish


def replay_rows(task, name, rows, read_record, factory, scorer):
    assert [r['source_index'] for r in rows] == list(range(COUNTS[name]))
    totals = {}
    for row in rows:
        i = row['source_index']; record = read_record(i)
        requests = [factory(request_type=r['request_type'], doc=row['document'],
            arguments=tuple(r['arguments']), idx=r['idx'], metadata=(name, i, 1)) for r in row['requests']]
        metrics, filtered = scorer(task, requests, record['response'])
        assert metrics == record['scores'], f'Stock score mismatch: {name}:{i}'
        assert identity(filtered) == identity(record['metadata']['filtered']), f'Stock filter mismatch: {name}:{i}'
        if totals: assert set(totals) == set(metrics)
        for metric, value in metrics.items(): totals[metric] = totals.get(metric, 0.) + value
        if (i + 1) % 1000 == 0: print(f'Stock CPU replay {name}: {i+1}/{len(rows)} match', flush=True)
    return {metric: value / len(rows) for metric, value in totals.items()}


def native_main(inputs, archive_path, output):
    import importlib.metadata
    import lm_eval
    from lm_eval.api.instance import Instance
    from stock_evaluation import load_task, score
    started = time.monotonic(); inputs = Path(inputs)
    assert importlib.metadata.version('lm_eval') == '0.4.9.2'
    hashes = json.loads((inputs / 'configs/HARNESS_SOURCE_HASHES.json').read_text())
    for rel, expected in hashes.items(): assert digest(Path(lm_eval.__file__).parent / rel) == expected
    ready = json.loads((inputs / 'configs/BASELINE_CPU_READY_V1.json').read_text())
    report = {'status': 'TASK1_ORIGINAL_BASELINE_STOCK_REPLAY_OK', 'counts': COUNTS,
        'metrics': {}, 'per_answer_scores_and_filtered_responses_all_match': True,
        'new_model_inference_run': False, 'GPU_work_run': False, 'training_updates': 0}
    with zipfile.ZipFile(archive_path) as archive:
        index = json.loads(archive.read('SNAPSHOT_INDEX.json')); assert index['counts'] == COUNTS
        for name, entry in ready['benchmarks'].items():
            path = inputs / str(safe_relative(entry['path'])); assert digest(path) == entry['sha256']
            with gzip.open(path, 'rt') as stream: rows = [json.loads(line) for line in stream]
            task, _, yaml_path = load_task(inputs, name)
            assert digest(yaml_path) == entry['stock_yaml_sha256'] and task.config.num_fewshot == 0
            def reader(i):
                rel = f'results_batch8_V2/original/{name}/{i:05}.json'
                data = archive.read(rel); assert hashlib.sha256(data).hexdigest() == index['files'][rel]['sha256']
                return json.loads(data)
            report['metrics'][name] = replay_rows(task, name, rows, reader, Instance, score)
            assert report['metrics'][name] == index['complete_metrics'][name]
            print(f'Stock CPU replay {name}: all {len(rows)} saved answers match.', flush=True)
    report['elapsed_seconds'] = time.monotonic() - started
    publish(output, report)
    print(json.dumps(report, indent=2), flush=True)


def make_export(base_archive, target, extras, summary, replay):
    """Add checked sources/references/receipts to already verified answer bytes."""
    assert replay['status'] == 'TASK1_ORIGINAL_BASELINE_STOCK_REPLAY_OK'
    assert replay['metrics'] == summary['metrics'] and replay['counts'] == summary['benchmark_counts'] == COUNTS
    target = Path(target); assert not target.exists(); shutil.copyfile(base_archive, target)
    members = {}
    with zipfile.ZipFile(target, 'a', zipfile.ZIP_DEFLATED) as archive:
        assert len(archive.namelist()) == len(set(archive.namelist()))
        for rel in archive.namelist():
            data = archive.read(rel); members[rel] = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
        for rel, data in extras.items():
            safe_relative(rel); assert rel not in members
            archive.writestr(rel, data)
            members[rel] = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
        manifest = {'status': 'TASK1_ORIGINAL_BASELINE_VERIFIED_EXPORT', 'benchmark_counts': COUNTS,
            'metrics': summary['metrics'], 'audit': summary['audit'], 'files': members,
            'stock_scores_and_filters_replayed_for_all_answers': True,
            'human_qualitative_annotation_pending': True, 'official_training_updates': 0,
            'no_original_model_weights_included': True}
        archive.writestr('EXPORT_MANIFEST.json', encoded(manifest))
    with zipfile.ZipFile(target) as archive:
        assert set(archive.namelist()) == set(members) | {'EXPORT_MANIFEST.json'}
        assert json.loads(archive.read('EXPORT_MANIFEST.json')) == manifest
        for rel, entry in members.items():
            data = archive.read(rel)
            assert len(data) == entry['bytes'] and hashlib.sha256(data).hexdigest() == entry['sha256']
    return manifest


if __name__ == '__main__': native_main(*sys.argv[1:])
