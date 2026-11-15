"""CPU restoration and isolated execution; no installations and no GPU work."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from archive_io import digest, restore, safe_relative


def inventory_complete(destination, entries):
    destination = Path(destination)
    missing = False
    for i, entry in enumerate(entries, 1):
        path = destination / str(safe_relative(entry['path']))
        kind = entry.get('kind', 'file')
        if not path.exists() and not path.is_symlink():
            missing = True
            continue
        if kind == 'file':
            if path.is_symlink() or not path.is_file() or path.stat().st_size != entry['bytes'] or digest(path) != entry['sha256']:
                raise RuntimeError('Existing private runtime/input differs: ' + str(path))
        elif kind == 'symlink':
            if not path.is_symlink() or os.readlink(path) != entry['target']:
                raise RuntimeError('Existing software link differs')
            if not path.resolve().is_relative_to(destination.resolve()):
                raise RuntimeError('Software link escapes private runtime')
        elif kind == 'directory':
            if path.is_symlink() or not path.is_dir(): raise RuntimeError('Directory differs')
        else: raise RuntimeError('Unknown inventory type')
        if i % 10000 == 0: print('Verified ' + str(i) + ' software/input entries...', flush=True)
    return not missing


def ensure_archive(root, local, job, record, entries, target):
    if inventory_complete(target, entries):
        print('Reusing checked local files: ' + str(target), flush=True)
        return
    source = root / str(safe_relative(record['archive_path']))
    if source.is_symlink(): raise RuntimeError('Linked archive')
    if shutil.disk_usage(local).free < source.stat().st_size + sum(e.get('bytes', 0) for e in entries) + 2 * 2**30:
        raise RuntimeError('Insufficient CPU disk space to restore frozen environment')
    target_copy = job / source.name
    print('Restoring the frozen archive on CPU: ' + source.name, flush=True)
    with source.open('rb') as src, target_copy.open('xb') as out:
        hasher = hashlib.sha256()
        for block in iter(lambda: src.read(8 * 1024**2), b''):
            hasher.update(block); out.write(block)
    if hasher.hexdigest() != record['archive_sha256']: raise IOError('Archive fingerprint mismatch')
    last_print = [0.]
    def progress(message):
        if time.monotonic() - last_print[0] > 10:
            print(message, flush=True); last_print[0] = time.monotonic()
    restore(target_copy, target, entries, progress)
    target_copy.unlink()


def main(root, bundle):
    root, bundle = Path(root), Path(bundle)
    local = Path('/content/atml_pa3_task1'); local.mkdir(exist_ok=True)
    if local.is_symlink(): raise RuntimeError('Unexpected linked runtime')
    job = local / ('baseline_CPU_restore_' + uuid.uuid4().hex); job.mkdir()
    kit = json.loads((root / 'configs/GPU_PREFLIGHT_KIT_READY.json').read_text())
    runtime = kit['portable_runtime']
    inventory_path = root / 'runtime' / ('inventory_' + runtime['inventory_sha256'] + '.json')
    assert digest(inventory_path) == runtime['inventory_sha256']
    inventory = json.loads(inventory_path.read_text())
    assert inventory['base_directory'] == str(local) and inventory['python_version'] == '3.12.12'
    print('Checking the existing private Python environment on CPU...', flush=True)
    ensure_archive(root, local, job, runtime, inventory['files'], local)
    # Evaluation uses the already frozen original assets on Drive. Do not copy
    # the full training/input archive merely to build text requests on CPU.
    python = local / 'venv312/bin/python'
    env = os.environ.copy()
    for key in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'UV_SYSTEM_PYTHON'): env.pop(key, None)
    env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1',
        HF_DATASETS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false',
        OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    result = subprocess.run([str(python), '-I', '-u', str(bundle / 'prepare_baseline_cpu.py'),
        str(root), str(local), str(bundle)], env=env, check=False)
    if result.returncode: raise RuntimeError('CPU baseline preparation stopped; send the saved error output')


if __name__ == '__main__': main(*sys.argv[1:])
