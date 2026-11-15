"""Read last shutdown evidence, then restore frozen software and run CPU checks."""
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from archive_io import digest
from restore_baseline_cpu import ensure_archive
from evaluation_records import publish


def main(root, bundle):
    root, bundle = Path(root), Path(bundle)
    sessions = [json.loads(path.read_text()) for path in (root / 'logs').glob('task1_baseline_*_session_started.json')]
    sessions = [s for s in sessions if s.get('implementation_identity') == 'd5c5b51f6107bd08adb362f878b528d6b1c92738e07e432405ddfeb7fc3ff56e']
    assert sessions, 'Expected interrupted baseline session not found'
    session = max(sessions, key=lambda s: s['started_wall'])
    run = session['run_id']; assert '/' not in run and run.startswith('task1_baseline_')
    evidence = {'run_id': run, 'files': {}}
    print('CPU SHUTDOWN EVIDENCE READBACK —', run, flush=True)
    for suffix in ('_worker_console.txt', '_bootstrap_error.txt', '_result.json', '_session_final.json'):
        path = root / 'logs' / (run + suffix)
        print('\nFILE: ' + path.name, flush=True)
        value = {'exists': path.exists()}
        if path.exists():
            value.update(sha256=digest(path), tail=path.read_text()[-16000:])
            print(value['tail'], flush=True)
        else: print('NOT FOUND', flush=True)
        evidence['files'][suffix] = value
    print('\nEND CPU SHUTDOWN READBACK — continuing CPU checks, NOT starting GPU.', flush=True)
    publish(root / 'logs' / ('block13_shutdown_readback_' + uuid.uuid4().hex + '.json'), evidence)
    local = Path('/content/atml_pa3_task1'); local.mkdir(exist_ok=True)
    job = local / ('batch8_cpu_restore_' + uuid.uuid4().hex); job.mkdir()
    old = json.loads((root / 'configs/GPU_PREFLIGHT_KIT_READY.json').read_text())
    runtime = old['portable_runtime']
    inventory_path = root / 'runtime' / ('inventory_' + runtime['inventory_sha256'] + '.json')
    assert digest(inventory_path) == runtime['inventory_sha256']
    inventory = json.loads(inventory_path.read_text())
    assert inventory['base_directory'] == str(local) and inventory['python_version'] == '3.12.12'
    ensure_archive(root, local, job, runtime, inventory['files'], local)
    env = os.environ.copy()
    for key in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'UV_SYSTEM_PYTHON'): env.pop(key, None)
    env.update(PYTHONNOUSERSITE='1', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
        HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false',
        OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    subprocess.run([str(local / 'venv312/bin/python'), '-I', '-u', str(bundle / 'prepare_batch8_cpu.py'),
                   str(root), str(local), str(bundle)], env=env, check=True)


if __name__ == '__main__': main(*sys.argv[1:])
