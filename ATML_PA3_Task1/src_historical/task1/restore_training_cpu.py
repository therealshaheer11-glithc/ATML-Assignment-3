"""Reuse the frozen Python stack on CPU and stream the trainer preparation output."""
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent))
from archive_io import digest
from restore_baseline_cpu import ensure_archive


def main(root,bundle,job):
    root,bundle,job=Path(root),Path(bundle),Path(job)
    local=Path('/content/atml_pa3_task1');local.mkdir(exist_ok=True)
    assert not local.is_symlink()
    runtime=json.loads((root/'configs/PORTABLE_RUNTIME_READY.json').read_text())
    path=root/'runtime'/('inventory_'+runtime['inventory_sha256']+'.json')
    assert digest(path)==runtime['inventory_sha256']
    inventory=json.loads(path.read_text())
    assert inventory['base_directory']==str(local) and inventory['python_version']=='3.12.12'
    print('Checking/restoring the frozen private Python on CPU; no package installs or downloads.',flush=True)
    ensure_archive(root,local,job,runtime,inventory['files'],local)
    env=os.environ.copy()
    for key in ('PYTHONPATH','PYTHONHOME','VIRTUAL_ENV','UV_SYSTEM_PYTHON'):env.pop(key,None)
    env.update(PYTHONNOUSERSITE='1',CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',
        HF_HUB_DISABLE_TELEMETRY='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    process=None
    try:
        with (root/'logs'/(job.name+'_console.txt')).open('x') as log:
            process=subprocess.Popen([str(local/'venv312/bin/python'),'-I','-u',str(bundle/'prepare_training_cpu.py'),
                str(root),str(job),str(bundle)],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                text=True,bufsize=1,start_new_session=True)
            for line in process.stdout:print(line,end='',flush=True);log.write(line);log.flush()
            if process.wait()!=0:raise RuntimeError('CPU training preparation failed; keep GPU off and send the complete output.')
    finally:
        if process is not None and process.poll() is None:
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                process.wait(timeout=5)
