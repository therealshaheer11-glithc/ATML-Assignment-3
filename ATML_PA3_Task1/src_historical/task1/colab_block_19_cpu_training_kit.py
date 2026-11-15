# Task 1 - Block 19: CPU trainer/data/checkpoint readiness; NO official training or GPU work.
# Download the ZIP, keep CPU selected, paste this ENTIRE file into one new Colab cell.
import hashlib, importlib.util, io, json, os, shutil, subprocess, sys, threading, traceback, uuid, zipfile
from pathlib import Path
PACKAGE_SHA = '6783caa2f1522412099d4b3f13c8643ec1ba0837d93a795316c5329591013b95'
EXPECTED_NAMES = {'evaluation_records.py', 'training_runtime_checks.py', 'restore_baseline_cpu.py', 'training_ops.py', 'training_checkpoints.py', 'APPROVED_PLAN.json', 'training_core.py', 'baseline_runtime.py', 'prepare_training_cpu.py', 'restore_training_cpu.py', 'gpu_guardian_R3.py', 'training_loop_checks.py', 'watchdog_evidence.py', 'release_coordination.py', 'training_loop.py', 'official_training.py', 'archive_io.py'}

def cpu_guard():
    evidence = []
    if os.environ.get('COLAB_GPU', '0').lower().strip() not in ('', '0', 'false', 'none'):
        evidence.append('GPU allocation flag')
    if list(Path('/dev').glob('nvidia[0-9]*')):
        evidence.append('GPU device')
    if any(os.environ.get(k) for k in ('COLAB_TPU_ADDR', 'TPU_NAME', 'TPU_WORKER_ID')) or os.environ.get('PJRT_DEVICE', '').upper() == 'TPU':
        evidence.append('TPU allocation')
    if shutil.which('nvidia-smi'):
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                                    capture_output=True, text=True, timeout=10)
            if result.returncode or result.stdout.strip(): evidence.append('GPU present or status uncertain')
        except Exception:
            evidence.append('GPU status uncertain')
    if evidence:
        print('CPU required; requesting release:', evidence, flush=True)
        def release():
            try:
                from google.colab import runtime
                runtime.unassign()
            except Exception as error:
                print('Release request:', type(error).__name__, flush=True)
        thread = threading.Thread(target=release, daemon=True); thread.start(); thread.join(10)
        raise RuntimeError('Select CPU; end this notebook\'s remaining L4 session through Manage sessions if active.')

cpu_guard()
from google.colab import files,drive
print('Upload Task_1_Block_19_CPU_Training_Kit.zip only. Keep CPU selected.',flush=True)
uploaded=files.upload()
assert len(uploaded)==1,'Upload exactly the supplied CPU training kit ZIP.'
payload=next(iter(uploaded.values()))
assert hashlib.sha256(payload).hexdigest()==PACKAGE_SHA,'CPU training kit fingerprint differs.'
JOB=Path('/content')/('task1_training_cpu_'+uuid.uuid4().hex);JOB.mkdir()
BUNDLE=JOB/'code';BUNDLE.mkdir()
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    assert len(archive.namelist())==len(EXPECTED_NAMES) and set(archive.namelist())==EXPECTED_NAMES
    for name in EXPECTED_NAMES:(BUNDLE/name).write_bytes(archive.read(name))
drive.mount('/content/drive')
ROOT=Path('/content/drive/MyDrive/ATML-PA-3/Task1')
for name in EXPECTED_NAMES:
    if name.endswith('.py'):sys.modules.pop(name[:-3],None)
sys.path.insert(0,str(BUNDLE))
spec=importlib.util.spec_from_file_location('task1_training_cpu_restore',BUNDLE/'restore_training_cpu.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
try:
    runner.main(ROOT,BUNDLE,JOB)
except BaseException:
    error=traceback.format_exc();print(error,flush=True)
    try:(ROOT/'logs'/(JOB.name+'_bootstrap_error.txt')).write_text(error)
    except Exception:pass
    raise
print('Keep GPU off and send the complete TASK1_BLOCK19_TRAINING_CPU_READY JSON or error.',flush=True)
