# Task 1 - Block 07: CPU checks for the L4 preflight launch kit.
import io
import zipfile
from google.colab import files

check_cpu_only()
expected_sha = "d76d168d5079738718cf0c278db1491bd9b73e9e9ab911691302923e2ac41d55"
print("Upload Task_1_Block_07_CPU_Preflight_Kit.zip only.", flush=True)
uploaded = files.upload()
if len(uploaded) != 1:
    raise RuntimeError("Upload exactly the single supplied ZIP.")
payload = next(iter(uploaded.values()))
if hashlib.sha256(payload).hexdigest() != expected_sha:
    raise RuntimeError("Package fingerprint differs. Stop here.")
bundle = LOCAL / "block07_code"
bundle.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    names = {"prepare_preflight.py", "gpu_guardian.py", "guardian_selftest.py", "preflight_worker.py", "training_ops.py", "training_core.py"}
    if len(archive.namelist()) != 6 or set(archive.namelist()) != names:
        raise RuntimeError("Unexpected package contents.")
    for name in sorted(names):
        save_bytes(bundle / name, archive.read(name))
job_env = env.copy()
job_env.update({"HF_HUB_DISABLE_TELEMETRY":"1", "TOKENIZERS_PARALLELISM":"false", "OMP_NUM_THREADS":"2", "MKL_NUM_THREADS":"2", "OPENBLAS_NUM_THREADS":"2"})
job_env.pop("HF_HUB_OFFLINE", None)
job_env.pop("HF_DATASETS_OFFLINE", None)
log_path = ROOT / "logs" / ("block07_console_" + uuid.uuid4().hex + ".txt")
process = None
try:
    with log_path.open("x") as log:
        process = subprocess.Popen(
            [str(PYTHON), "-I", "-u", str(bundle / "prepare_preflight.py"),
             str(ROOT), str(LOCAL), str(bundle)],
            env=job_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        if process.wait() != 0:
            raise RuntimeError("Preflight-kit preparation stopped. Send the error output; earlier results are preserved.")
finally:
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
print("Saved console log:", log_path, flush=True)
