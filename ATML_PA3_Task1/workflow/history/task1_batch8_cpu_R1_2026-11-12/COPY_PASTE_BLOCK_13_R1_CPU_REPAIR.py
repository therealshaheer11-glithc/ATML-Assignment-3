# Task 1 - Block 13 R1: repair readiness-file publication on current CPU runtime.
# Standalone: run after selecting CPU; earlier notebook variables are not needed.
import os, sys, io, json, uuid, hashlib, subprocess, threading, zipfile
from pathlib import Path

GPU_EVIDENCE = []
if os.environ.get("COLAB_GPU", "0").strip().lower() not in ("", "0", "false", "none"):
    GPU_EVIDENCE.append("GPU allocation flag")
if any(os.environ.get(k) for k in ("COLAB_TPU_ADDR", "TPU_NAME", "TPU_WORKER_ID")):
    GPU_EVIDENCE.append("TPU allocation flag")
if os.environ.get("PJRT_DEVICE", "").upper() == "TPU":
    GPU_EVIDENCE.append("TPU runtime")
if list(Path("/dev").glob("nvidia[0-9]*")):
    GPU_EVIDENCE.append("GPU device present")
import shutil
executable = shutil.which("nvidia-smi")
if executable:
    try:
        probe = subprocess.run([executable, "--query-gpu=name", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=10)
        if probe.returncode == 0 and probe.stdout.strip():
            GPU_EVIDENCE.append(probe.stdout.strip())
        elif probe.returncode != 0:
            GPU_EVIDENCE.append("GPU allocation is uncertain")
    except (OSError, subprocess.TimeoutExpired):
        GPU_EVIDENCE.append("GPU allocation could not be checked")
if GPU_EVIDENCE:
    print("CPU required. Requesting runtime release:", GPU_EVIDENCE, flush=True)
    def release():
        try:
            from google.colab import runtime
            runtime.unassign()
        except Exception as error:
            print("Release request failed:", type(error).__name__, flush=True)
    thread = threading.Thread(target=release, daemon=True)
    thread.start(); thread.join(timeout=10)
    raise RuntimeError("Select CPU. If still connected to a GPU, disconnect/delete it first.")

ORIGINAL_ZIP_SHA = '47f22423b6487e98e46908db691b3aeba72b2f8f8e40e868b1804990dbfbf027'
FIXED_PREPARE_SHA = '69a00eb778cdcb31a50e3908c87ce619aa34b3ca5e8407260e1b55cc5aeabcf9'
EXPECTED_FILES = {'batch8_records.py': '6738fcd2146a4057e54c7728129411700524e5f78a6efdfeccfd120f2589c4aa', 'batch8_hflm.py': '1b8b3810028123d6f6623398e028628cc3d9fc72070aad359d92fdc52023969e', 'batch8_pilot.py': '76171f667464327df7c3235cf7f9209dd18efaca8ff56b715c9493af8e156ec9', 'batch8_cpu_checks.py': 'ff5bfe4acc12b2e8ef0809f98c0d0b2d624f3e821567c32b06042043d1cdb905', 'prepare_batch8_cpu.py': '586497508764595e8fc7598c900c6e61e7f8e6ff03481137478bc8868b46cb2d', 'restore_batch8_cpu.py': '32ebce8ea449913d28c8a6681bed8773b222c88babb07cbc09189d30207d8f53', 'archive_io.py': 'd5d059306f34a09eee8b4469d3f47597b07ec182f797d40e4a4958d2a1bf8f57', 'restore_baseline_cpu.py': 'f0e2d61a261c5ee8bd8b3ebc10250dd56defe9906ff139d21ef9ccb2745c7106', 'baseline_runtime.py': 'a2c420ed090689c1849d65d7de4717707a713937b1060bb9c65b6adf44a8723a', 'evaluation_records.py': '841fe3750ba33fb63d1a7e56a3ebd5248c013f3eada114e3690c4b5a8250c607', 'APPROVAL_T1_A003.json': 'a4498249b3036bb5a6b25f1d3cb7b7ea44f5b2873c5d555a21361464ec3a8b56', 'PROPOSED_D10B.md': 'aa251f073c0d994ad1f5364978bb4b65c6c67f8ecf8377d73a3e8f91470ed8b6'}

# Reuse the verified original upload; no new upload or environment restoration.
ROOT = Path('/content/drive/MyDrive/ATML-PA-3/Task1')
PRIVATE_PYTHON = Path('/content/atml_pa3_task1/venv312/bin/python')
if not (ROOT / 'configs/GPU_PREFLIGHT_KIT_READY.json').exists() or not PRIVATE_PYTHON.exists():
    raise RuntimeError('Run this repair in the same CPU runtime where Block 13 just failed.')
payload = None
for candidate in Path('/content').glob('Task_1_Block_13_CPU_Batch8_Checks*.zip'):
    data = candidate.read_bytes()
    if hashlib.sha256(data).hexdigest() == ORIGINAL_ZIP_SHA:
        payload = data
        break
if payload is None:
    for data in globals().get('uploaded', {}).values():
        if isinstance(data, bytes) and hashlib.sha256(data).hexdigest() == ORIGINAL_ZIP_SHA:
            payload = data
            break
if payload is None:
    raise RuntimeError('The verified original ZIP was not found in this runtime. Keep CPU selected and send this message.')
WORK = Path('/content') / ('task1_batch8_CPU_R1_' + uuid.uuid4().hex)
WORK.mkdir()
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    if len(archive.namelist()) != len(EXPECTED_FILES) or set(archive.namelist()) != set(EXPECTED_FILES):
        raise RuntimeError('Original package entries differ')
    for name, expected in EXPECTED_FILES.items():
        data = archive.read(name)
        if hashlib.sha256(data).hexdigest() != expected:
            raise RuntimeError('Original source fingerprint differs: ' + name)
        if name == 'prepare_batch8_cpu.py':
            text = data.decode('utf-8')
            old = "publish(root / 'configs/BATCH8_CPU_CHECKS_' + implementation + '.json', ready)"
            new = "publish(root / 'configs' / ('BATCH8_CPU_CHECKS_' + implementation + '.json'), ready)"
            if text.count(old) != 1:
                raise RuntimeError('Expected single-line repair target differs')
            data = text.replace(old, new).encode('utf-8')
            if hashlib.sha256(data).hexdigest() != FIXED_PREPARE_SHA:
                raise RuntimeError('Repaired source fingerprint differs')
        (WORK / name).write_bytes(data)
print('Verified original package reused; one save-path expression corrected.', flush=True)
env = os.environ.copy()
for key in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'UV_SYSTEM_PYTHON'):
    env.pop(key, None)
env.update(PYTHONNOUSERSITE='1', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
           HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false',
           OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
log_path = ROOT / 'logs' / ('block13_R1_CPU_console_' + uuid.uuid4().hex + '.txt')
process = None
try:
    with log_path.open('x') as log:
        process = subprocess.Popen([str(PRIVATE_PYTHON), '-I', '-u',
            str(WORK / 'prepare_batch8_cpu.py'), str(ROOT), '/content/atml_pa3_task1', str(WORK)],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            bufsize=1, start_new_session=True)
        for line in process.stdout:
            print(line, end='', flush=True); log.write(line); log.flush()
        if process.wait() != 0:
            raise RuntimeError('CPU readiness repair stopped; keep CPU selected and send the output.')
finally:
    if process is not None and process.poll() is None:
        import signal
        try: os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError: pass
        try: process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try: os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            process.wait()
print('Saved CPU repair console:', log_path, flush=True)
print('Keep CPU selected and send the final TASK1_BLOCK13_CPU_BATCH8_OK summary.', flush=True)
