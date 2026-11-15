"""Synthetic CPU checks of scoring replay and full export member integrity."""
import copy
import json
import zipfile
from pathlib import Path

from baseline_completion_cpu import replay_rows, make_export
from baseline_v2_engine_checks import Request
from evaluation_records import COUNTS, encoded


def checks(directory):
    directory = Path(directory); directory.mkdir()
    rows = [{'source_index': i, 'document': {'CPU_FIXTURE_ONLY': True},
             'requests': [{'request_type': 'generate_until', 'arguments': ['fixture', {}], 'idx': 0}]}
            for i in range(COUNTS['gsm8k_cot'])]
    records = [{'response': ['CPU_FIXTURE_ONLY'], 'scores': {'fixture': float(i % 2)},
                'metadata': {'filtered': {'fixture': ['CPU_FIXTURE_ONLY']}}} for i in range(len(rows))]
    def scorer(task, requests, responses):
        return {'fixture': float(requests[0].doc_id % 2)}, {'fixture': responses}
    metrics = replay_rows(None, 'gsm8k_cot', rows, records.__getitem__, Request, scorer)
    assert metrics == {'fixture': sum(i % 2 for i in range(1319)) / 1319}
    bad = copy.deepcopy(records); bad[-1]['scores']['fixture'] = .5
    try: replay_rows(None, 'gsm8k_cot', rows, bad.__getitem__, Request, scorer)
    except AssertionError: pass
    else: raise AssertionError('Changed score accepted')
    bad = copy.deepcopy(records); bad[0]['metadata']['filtered']['fixture'] = ['changed']
    try: replay_rows(None, 'gsm8k_cot', rows, bad.__getitem__, Request, scorer)
    except AssertionError: pass
    else: raise AssertionError('Changed filtered answer accepted')
    try: replay_rows(None, 'gsm8k_cot', rows[:-1], records.__getitem__, Request, scorer)
    except AssertionError: pass
    else: raise AssertionError('Incomplete split accepted')
    archive = directory / 'base.zip'
    with zipfile.ZipFile(archive, 'w') as z: z.writestr('SNAPSHOT_INDEX.json', encoded({'CPU_FIXTURE_ONLY': True}))
    summary = {'metrics': {'fixture': metrics}, 'benchmark_counts': COUNTS, 'audit': {'CPU_FIXTURE_ONLY': True}}
    replay = {'status': 'TASK1_ORIGINAL_BASELINE_STOCK_REPLAY_OK', 'counts': COUNTS, 'metrics': summary['metrics']}
    manifest = make_export(archive, directory / 'export.zip', {'proof.json': b'CPU_FIXTURE_ONLY'}, summary, replay)
    with zipfile.ZipFile(directory / 'export.zip') as z:
        assert z.read('proof.json') == b'CPU_FIXTURE_ONLY'
        assert json.loads(z.read('EXPORT_MANIFEST.json')) == manifest
    return {'full_1319_synthetic_GSM_replay_coverage_checked': True,
        'changed_score_and_filtered_answer_rejected': True, 'incomplete_split_rejected': True,
        'export_all_members_hash_readback_passed': True,
        'scope': 'Synthetic CPU control tests; actual pinned stock replay will run in Colab'}
