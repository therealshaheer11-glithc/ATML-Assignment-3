# Task 1 - Block 10: complete CPU evaluation preparation and resume checks.
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

from google.colab import drive, files
drive.mount("/content/drive")
ROOT = Path("/content/drive/MyDrive/ATML-PA-3/Task1")
print("Upload Task_1_Block_10_CPU_Baseline_Kit.zip only.", flush=True)
uploaded = files.upload()
if len(uploaded) != 1:
    raise RuntimeError("Upload exactly the single supplied ZIP.")
payload = next(iter(uploaded.values()))
if hashlib.sha256(payload).hexdigest() != "6d7449380fb29b78d57f14f229cd19c7b0dd4ce94651d837c7206f5beb204d17":
    raise RuntimeError("Package fingerprint differs.")
WORK = Path("/content") / ("task1_baseline_CPU_" + uuid.uuid4().hex)
WORK.mkdir()
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    names = {'archive_io.py', 'stock_evaluation.py', 'evaluation_records.py', 'restore_baseline_cpu.py', 'evaluation_engine.py', 'prepare_baseline_cpu.py'}
    if len(archive.namelist()) != len(names) or set(archive.namelist()) != names:
        raise RuntimeError("Unexpected package contents")
    for name in names:
        (WORK / name).write_bytes(archive.read(name))
log_path = ROOT / "logs" / ("block10_console_" + uuid.uuid4().hex + ".txt")
process = None
try:
    with log_path.open("x") as log:
        process = subprocess.Popen(
            [sys.executable, "-I", "-u", str(WORK / "restore_baseline_cpu.py"), str(ROOT), str(WORK)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True,
        )
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line); log.flush()
        if process.wait() != 0:
            raise RuntimeError("CPU preparation stopped. Send the error output; earlier evidence is preserved.")
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
print("Saved console log:", log_path, flush=True)
