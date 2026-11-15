"""Colab GPU controller. Release the allocation after the isolated cache worker exits."""
import os
import sys
import io
import json
import hashlib
import subprocess
import time
import traceback
import uuid
import zipfile
from pathlib import Path
from datetime import datetime,timezone
from google.colab import drive,runtime

if os.environ.get('COLAB_GPU','0') in ('','0'):
    raise RuntimeError('Choose a T4 GPU runtime, reconnect, then run this Block 11 cell.')

package_bytes = globals().get('task2_features_package_bytes')
if not isinstance(package_bytes,bytes):
    try:
        raise RuntimeError('Run the supplied Block 11 package loader.')
    finally:
        runtime.unassign()
package_sha = hashlib.sha256(package_bytes).hexdigest()
LOCAL_ROOT = Path('/content/atml_pa3_task2/run_20261031_150949_1cff148a')
work = LOCAL_ROOT/'feature_code'/package_sha[:16]
try:
    with zipfile.ZipFile(io.BytesIO(package_bytes)) as archive:
        assert archive.testzip() is None
        manifest = json.loads(archive.read('package_manifest.json'))
        assert set(archive.namelist()) == {row['path'] for row in manifest['files']} | {'package_manifest.json'}
        for row in manifest['files']:
            relative = Path(row['path'])
            assert not relative.is_absolute() and '..' not in relative.parts
            content = archive.read(row['path'])
            assert len(content) == row['bytes'] and hashlib.sha256(content).hexdigest() == row['sha256']
            destination = work/relative
            destination.parent.mkdir(parents=True,exist_ok=True)
            if destination.exists():
                assert destination.read_bytes() == content
            else:
                destination.write_bytes(content)
    sys.path.insert(0,str(work))
    for name in list(sys.modules):
        if name == 'task2' or name.startswith('task2.'):
            del sys.modules[name]
    from task2.gpu_lifecycle import GpuSessionGuard
    from task2.worker_console import run_logged_worker
    from task2.session_io import bounded_io,save_bytes,save_json
except BaseException:
    runtime.unassign()
    raise

def release(reason):
    print('Requesting GPU runtime release after '+reason+'. Return the complete output from this cell.',flush=True)
    runtime.unassign()

guard = GpuSessionGuard(release,hard_seconds=1200)
guard.start()
RUN_ROOT=Path('/content/drive/MyDrive/ATML_PA3/task2/run_20261031_150949_1cff148a')
attempt=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_')+uuid.uuid4().hex[:8]
local_console=LOCAL_ROOT/'logs'/('block_11_console_'+attempt+'.log')
console_receipt=None
console_save_error=None
worker_code=None
controller_error=None

def save_console(path):
    global console_receipt,console_save_error
    try:
        with bounded_io(15,'Worker console snapshot'):
            content=Path(path).read_bytes()
            digest=hashlib.sha256(content).hexdigest()
            destination=RUN_ROOT/'logs'/f'block_11_console_{attempt}_{digest[:16]}.log'
            save_bytes(destination,content)
            console_receipt={'path':str(destination),'sha256':digest,'bytes':len(content)}
        console_save_error=None
    except BaseException as error:
        console_save_error=repr(error)
        print('Console snapshot is not yet saved:',console_save_error,flush=True)

try:
    if not Path('/content/drive/MyDrive').exists():
        drive.mount('/content/drive')
    environment = dict(os.environ)
    environment['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    environment['PYTHONPATH'] = str(work)+os.pathsep+environment.get('PYTHONPATH','')
    command = [sys.executable,'-u',str(work/'cache_worker.py'),
        '--soft-deadline',str(guard.started+900),'--package-sha',package_sha,'--attempt',attempt]
    print('Starting bounded GPU feature caching. Completed archives are verified and reused on rerun.',flush=True)
    worker_code=run_logged_worker(command,environment,guard,local_console,snapshot=save_console)
    print('GPU worker exit code:',worker_code,flush=True)
    if worker_code != 0:
        raise RuntimeError(f'Feature worker exited with code {worker_code}. Its traceback is printed above and captured in the console log.')
    text=local_console.read_text()
    if 'BLOCK 11 PASSED' not in text and 'BLOCK 11 CHECKPOINTED' not in text:
        raise RuntimeError('Worker exited without a feature-cache completion or checkpoint result.')
except BaseException:
    controller_error=traceback.format_exc()
    print('CONTROLLER DIAGNOSTIC\n'+controller_error,flush=True)
    raise
finally:
    try:
        if local_console.exists():
            save_console(local_console)
        diagnostic={'schema':'atml-pa3-task2-feature-controller-v2','attempt':attempt,
            'cache_package_sha256':package_sha,'worker_exit_code':worker_code,
            'controller_traceback':controller_error,'console':console_receipt,
            'console_save_error':console_save_error,'release_request_follows':True}
        try:
            with bounded_io(15,'Controller diagnostic save'):
                save_json(RUN_ROOT/'logs'/f'block_11_controller_{attempt}.json',diagnostic)
        except BaseException as error:
            print('Controller diagnostic could not be saved:',repr(error),flush=True)
        print(json.dumps(diagnostic,indent=2),flush=True)
    finally:
        guard.finish('cache worker completion or handled failure')
