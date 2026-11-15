# Task 1 - Block 04 R1: repaired CPU data preparation stage.
import io
import zipfile
from google.colab import files

check_cpu_only()
expected_sha = "e9c790754c2cf0141c6e8f4b8e0077247fc795365a0c80502f7a791f834312e5"
print("Upload Task_1_Block_04_R1_CPU_Data.zip only.", flush=True)
uploaded = files.upload()
if len(uploaded) != 1:
    raise RuntimeError("Upload exactly the single supplied ZIP.")
payload = next(iter(uploaded.values()))
if hashlib.sha256(payload).hexdigest() != expected_sha:
    raise RuntimeError("Package fingerprint differs. Stop here.")
bundle = LOCAL / "block04_r1_code"
bundle.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    names = {"prepare_data_R1.py", "APPROVED_PLAN.json", "proposal.md"}
    if len(archive.namelist()) != 3 or set(archive.namelist()) != names:
        raise RuntimeError("Unexpected package contents.")
    for name in sorted(names):
        save_bytes(bundle / name, archive.read(name))
job_env = env.copy()
job_env.update({"HF_HUB_DISABLE_TELEMETRY":"1", "TOKENIZERS_PARALLELISM":"false"})
job_env.pop("HF_HUB_OFFLINE", None)
job_env.pop("HF_DATASETS_OFFLINE", None)
log_path = ROOT / "logs" / ("block04_r1_console_" + uuid.uuid4().hex + ".txt")
process = None
try:
    with log_path.open("x") as log:
        process = subprocess.Popen(
            [str(PYTHON), "-I", "-u", str(bundle / "prepare_data_R1.py"),
             str(ROOT), str(LOCAL), str(bundle)],
            env=job_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        if process.wait() != 0:
            raise RuntimeError("Data preparation stopped. Send the error output; do not change settings.")
finally:
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
print("Saved console log:", log_path, flush=True)
