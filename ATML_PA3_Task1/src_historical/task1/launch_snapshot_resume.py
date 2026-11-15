"""Stage one CPU-audited snapshot, then use the unchanged bounded parent."""
import json
from pathlib import Path

from archive_io import digest
from baseline_runtime import progress, publish_bytes
from evaluation_records import identity, publish
from snapshot_transport import restore_snapshot

SNAPSHOT = '3f3902149ce3cfb9c84ae62814e26bc366a5b7f7588826fc44268a8876225411'
READINESS_SHA = 'f31bf79e4fb265b405f03d57576ac9de138231df7560dddb26dba447e3cae5c5'
NUMERICAL_ID = '75d7cc8f78c938313aba6703b0095cafbb7c83adf4b57519eff83ff6800dbc09'


def wrapped_command(command, core, transport, snapshot, transport_id):
    assert Path(command[-2]) == Path(core) / 'baseline_v2_worker.py'
    return command[:-2] + [str(Path(transport) / 'snapshot_worker_entry.py'), command[-1],
                           str(core), str(snapshot), transport_id]


def run(context, bundle):
    import launch_baseline_v2 as numerical_parent
    bundle, root, job = Path(bundle), Path(context['root']), Path(context['job'])
    core, transport = bundle / 'core', bundle / 'transport'
    progress(context['state'], 'verify frozen numerical code and CPU-audited snapshot readiness')
    original = json.loads((bundle / 'CORE_FILES.json').read_text())
    for name, expected in original.items():
        assert '/' not in name and digest(core / name) == expected
    py_files = {n: h for n, h in original.items() if n.endswith('.py')}
    assert identity(py_files) == NUMERICAL_ID
    readiness_path = root / 'configs' / ('BASELINE_V2_RESUME_READY_' + SNAPSHOT + '.json')
    assert digest(readiness_path) == READINESS_SHA
    ready = json.loads(readiness_path.read_text())
    assert ready['status'] == 'TASK1_BASELINE_V2_RESUME_SNAPSHOT_READY'
    assert ready['snapshot_identity'] == SNAPSHOT and ready['original_numerical_files'] == py_files
    assert ready['numerical_implementation_identity'] == NUMERICAL_ID
    assert ready['archive_bytes'] == 17328785
    assert ready['counts'] == {'gsm8k_cot': 850, 'hellaswag': 10042, 'boolq': 3270, 'arc_easy': 2376}
    snapshot, index = restore_snapshot(root, job, ready, context['state'])
    files = {p.name: digest(p) for p in transport.glob('*.py')}
    transport_id = identity(files)
    publish(job / 'transport_manifest.json', {'files': files})
    backup = root / 'execution' / ('baseline_snapshot_transport_' + transport_id)
    for name in files: publish_bytes(backup / name, (transport / name).read_bytes())
    publish(backup / 'FILES.json', files)
    proof = {'run_id': job.name, 'snapshot_identity': SNAPSHOT, 'readiness_sha256': READINESS_SHA,
        'transport_implementation_identity': transport_id, 'numerical_implementation_identity': NUMERICAL_ID,
        'snapshot_counts': index['counts'], 'local_snapshot_verified': True, 'experiment_settings_changed': False}
    publish(root / 'logs' / (job.name + '_transport_started.json'), proof)
    original_execute = numerical_parent.execute
    def execute_with_snapshot(command, env, ctx, session):
        assert Path(command[-2]) == core / 'baseline_v2_worker.py'
        assert session['implementation_identity'] == NUMERICAL_ID
        print('Snapshot restored. Reusing all 16,538 verified scored answers; 469 GSM scored answers remain.', flush=True)
        revised = wrapped_command(command, core, transport, snapshot, transport_id)
        return original_execute(revised, env, ctx, session)
    numerical_parent.execute = execute_with_snapshot
    try:
        return numerical_parent.run(context, core)
    finally:
        numerical_parent.execute = original_execute
