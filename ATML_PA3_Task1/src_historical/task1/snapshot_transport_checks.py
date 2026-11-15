"""Synthetic CPU tests of local reads, original Drive writes and completion order."""
import gzip
import hashlib
import json
import os
import time
import zipfile
from pathlib import Path
from unittest.mock import patch

from baseline_runtime import atomic, SoftStop
from evaluation_records import COUNTS, encoded, identity, publish
from resume_snapshot_checks import checks as fixture_checks
from snapshot_transport import restore_snapshot, PackedRecordStore, configure_worker
from launch_snapshot_resume import wrapped_command


def checks(directory):
    directory = Path(directory)
    fixture_checks(directory)
    root = directory / 'root'; archive = directory / 'local/verified_resume.zip'
    with zipfile.ZipFile(archive) as z: index = json.loads(z.read('SNAPSHOT_INDEX.json'))
    ready = {'archive_path': 'packed.zip', 'archive_bytes': archive.stat().st_size,
        'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
        'index_sha256': hashlib.sha256(encoded(index)).hexdigest(), 'snapshot_identity': identity(index),
        'contract': index['contract'], 'counts': index['counts'],
        'numerical_implementation_identity': index['numerical_implementation_identity']}
    (root / 'packed.zip').write_bytes(archive.read_bytes())
    job = directory / 'job'; job.mkdir(); state = job / 'state.json'
    atomic(state, {'deadline_wall': time.time() + 300, 'progress_wall': time.time(), 'status': 'setup'})
    snapshot, restored = restore_snapshot(root, job, ready, state)
    assert restored == index
    stats = {'snapshot_record_reads': 0, 'remote_record_checks_for_unindexed_IDs': 0,
             'new_records_saved_with_original_Drive_readback': 0}
    contract = index['contract']
    store = PackedRecordStore(root / 'results_batch8_V2', 'original', contract['model_identity'],
        contract['protocol_identity'], snapshot=snapshot, index=index, stats=stats)
    rows_by_task = {}
    exists = Path.exists
    prefixes = [str(root / 'results_batch8_V2/original' / t) + '/' for t in COUNTS]
    def forbid_remote_old_checks(path):
        if any(str(path).startswith(p) for p in prefixes):
            raise AssertionError('Unexpected per-record Drive operation for an indexed answer')
        return exists(path)
    with patch.object(Path, 'exists', forbid_remote_old_checks):
        for task, count in index['counts'].items():
            with gzip.open(root / 'inputs' / (task + '.jsonl.gz'), 'rt') as stream:
                rows = [json.loads(line) for line in stream]
            rows_by_task[task] = rows
            for i in range(count):
                request = {'source_index': i, 'requests': rows[i]['requests']}
                value = store.get(task, i, request)
                assert value == json.loads((snapshot / store.key(task, i)).read_bytes())
    assert stats['snapshot_record_reads'] == 16538 and stats['remote_record_checks_for_unindexed_IDs'] == 0
    row = rows_by_task['gsm8k_cot'][0]; request = {'source_index': 0, 'requests': row['requests']}
    path = snapshot / store.key('gsm8k_cot', 0); original = path.read_bytes(); path.unlink()
    try: store.get('gsm8k_cot', 0, request)
    except AssertionError: pass
    else: raise AssertionError('Missing indexed local answer accepted')
    path.write_bytes(original + b' ')
    try: store.get('gsm8k_cot', 0, request)
    except AssertionError: pass
    else: raise AssertionError('Changed indexed local answer accepted')
    path.write_bytes(original)
    row = rows_by_task['gsm8k_cot'][850]; request = {'source_index': 850, 'requests': row['requests']}
    assert store.get('gsm8k_cot', 850, request) is None
    value = store.put('gsm8k_cot', 850, request, ['CPU_FIXTURE_ONLY'], {'fixture_metric': 0.}, {'CPU_FIXTURE_ONLY': True})
    assert value == json.loads(store.path('gsm8k_cot', 850).read_bytes())
    assert stats['new_records_saved_with_original_Drive_readback'] == 1
    assert stats['remote_record_checks_for_unindexed_IDs'] == 2
    # Real original worker main/finalization, with only its numerical run replaced by a CPU fixture.
    import baseline_v2_worker as worker
    saved = worker.run, worker.RecordStore, worker.publish
    config = {'root': str(root), 'job': str(job), 'state': str(state), 'run_id': 'CPU_FIXTURE_ONLY',
        'result': str(root / 'logs/result.json'), 'model_identity': contract['model_identity'],
        'protocol_identity': contract['protocol_identity'], 'implementation_identity': index['numerical_implementation_identity']}
    atomic(state, {'deadline_wall': time.time() + 300, 'progress_wall': time.time(),
                  'status': 'running', 'worker_pid': os.getpid()})
    config_path = job / 'config.json'; atomic(config_path, config)
    events = []
    def spy_publish(path, value):
        if str(path).endswith('_transport_result.json'): events.append('transport_receipt')
        if Path(path) == Path(config['result']): events.append('worker_result')
        publish(path, value)
    worker.publish = spy_publish
    def fixture_run(c, result):
        s = worker.RecordStore(root / 'results_batch8_V2', 'original', contract['model_identity'], contract['protocol_identity'])
        s.get('gsm8k_cot', 0, {'source_index': 0, 'requests': rows_by_task['gsm8k_cot'][0]['requests']})
        result['benchmark_completed'] = {**COUNTS, 'gsm8k_cot': 851}
        raise SoftStop('CPU_FIXTURE_ONLY normal boundary')
    worker.run = fixture_run
    try:
        configure_worker(worker, config, snapshot, index, 'CPU_FIXTURE_TRANSPORT')
        assert worker.main(config_path) == 0
        assert events == ['transport_receipt', 'worker_result']
        result = json.loads(Path(config['result']).read_bytes())
        assert result['status'] == 'TASK1_BASELINE_V2_PARTIAL_SAVED'
        assert result['packed_resume_transport']['record_IO']['snapshot_record_reads'] == 1
        assert json.loads(state.read_bytes())['status'] == 'worker_complete'
    finally: worker.run, worker.RecordStore, worker.publish = saved
    command = ['private_python', '-I', '-u', '/core/baseline_v2_worker.py', '/job/config.json']
    assert wrapped_command(command, '/core', '/transport', '/snapshot', 'ID') == [
        'private_python', '-I', '-u', '/transport/snapshot_worker_entry.py', '/job/config.json', '/core', '/snapshot', 'ID']
    bad = {**ready, 'archive_sha256': '0' * 64}
    badjob = directory / 'badjob'; badjob.mkdir()
    try: restore_snapshot(root, badjob, bad, state)
    except AssertionError: pass
    else: raise AssertionError('Changed archive accepted')
    return {'all_16538_cached_answer_reads_avoid_per_record_Drive_checks': True,
        'snapshot_members_individually_hash_verified_on_local_restore': True,
        'missing_or_changed_indexed_local_answer_rejected': True,
        'new_answers_use_original_Drive_save_and_readback': True,
        'original_worker_receipt_and_completion_handshake_preserved': True,
        'transport_receipt_saved_before_worker_completion': True,
        'changed_archive_rejected': True, 'isolated_worker_argument_routing_checked': True,
        'scope': 'CPU synthetic storage/control fixtures only; no original-model inference'}
