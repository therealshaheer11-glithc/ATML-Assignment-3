# Task 1 - Block 02: isolated Python 3.12, CPU only.
# Run below Block 01 in the SAME notebook. No ML packages or models yet.
import re
import time

if not all(name in globals() for name in ("check_cpu_only", "write_once", "APPROVAL")):
    raise RuntimeError("Run Block 01 successfully in this notebook first.")
check_cpu_only()

ROOT = Path("/content/drive/MyDrive/ATML-PA-3/Task1")
expected = {
    "schema_version": 1, "assignment": "ATML PA3 Task 1",
    "approval_id": "T1-A001",
    "proposal_sha256": "d13f6172406705a258d7319d20c38ebed78309a997c87ed065e84388fe0bf0a4",
}
if json.loads((ROOT / "WORKSPACE.json").read_text()) != expected:
    raise RuntimeError("Workspace identity differs. Stop and share the output.")
if json.loads((ROOT / "provenance" / "APPROVAL_REFERENCE.json").read_text()) != APPROVAL:
    raise RuntimeError("Approval reference differs. Stop and share the output.")

LOCAL = Path("/content/atml_pa3_task1")
if LOCAL.is_symlink() or (LOCAL.exists() and not LOCAL.is_dir()):
    raise RuntimeError("Unexpected local Task 1 path; preserving it.")
owner = LOCAL / "OWNER.json"
if LOCAL.exists() and not owner.exists() and any(LOCAL.iterdir()):
    raise RuntimeError("Local Task 1 folder has unrecognized content; preserving it.")
LOCAL.mkdir(exist_ok=True)
write_once(owner, expected)

UV_VERSION = "0.9.5"
BOOT = LOCAL / "bootstrap_uv"
VENV = LOCAL / "venv312"
UV = BOOT / "bin" / "uv"
PYTHON = VENV / "bin" / "python"
snapshot_path = ROOT / "configs" / "PYTHON_ENVIRONMENT.json"
prior = json.loads(snapshot_path.read_text()) if snapshot_path.exists() else None
python_request = prior["python_version"] if prior else "3.12"
if prior and not re.fullmatch(r"3\.12\.\d+", python_request):
    raise RuntimeError("Saved Python version is outside the approved 3.12 series.")

env = os.environ.copy()
for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
    env.pop(name, None)
env.update({
    "PYTHONNOUSERSITE": "1", "PIP_CONFIG_FILE": os.devnull,
    "PIP_DISABLE_PIP_VERSION_CHECK": "1", "UV_NO_CONFIG": "1",
    "PIP_CACHE_DIR": str(LOCAL / "cache" / "pip"),
    "UV_CACHE_DIR": str(LOCAL / "cache" / "uv"),
    "UV_PYTHON_INSTALL_DIR": str(LOCAL / "managed_pythons"),
    "UV_PYTHON_BIN_DIR": str(LOCAL / "managed_bin"),
})


def run_setup(args, timeout=600):
    result = subprocess.run(
        [str(arg) for arg in args], env=env, cwd=str(LOCAL),
        capture_output=True, text=True, timeout=timeout, check=False,
    )
    if result.stdout.strip():
        print(result.stdout.strip(), flush=True)
    if result.stderr.strip():
        print(result.stderr.strip(), flush=True)
    if result.returncode:
        raise RuntimeError("Setup command failed. Stop and send the complete output.")
    return result.stdout.strip()


started = time.monotonic()
if not UV.is_file():
    if BOOT.exists() and (not BOOT.is_dir() or any(BOOT.iterdir())):
        raise RuntimeError("Partial/unrecognized bootstrap folder; preserving it.")
    print("Installing the private Python setup helper...", flush=True)
    run_setup([
        sys.executable, "-m", "pip", "install", "--index-url", "https://pypi.org/simple",
        "--only-binary=:all:", "--no-deps", "--target", BOOT, f"uv=={UV_VERSION}",
    ])
uv_banner = run_setup([UV, "--version"], timeout=30)
if not uv_banner.startswith(f"uv {UV_VERSION} ") and uv_banner != f"uv {UV_VERSION}":
    raise RuntimeError("Unexpected setup-helper version. Stop here.")

if not VENV.exists():
    print(f"Preparing managed Python {python_request}...", flush=True)
    run_setup([UV, "python", "install", python_request])
    base_python = run_setup([UV, "python", "find", "--managed-python", python_request], 60)
    run_setup([UV, "venv", "--python", base_python, VENV], 120)
elif VENV.is_symlink() or not (VENV / "pyvenv.cfg").is_file() or not PYTHON.is_file():
    raise RuntimeError("Partial/unrecognized Python environment; preserving it.")

probe_code = '''
import importlib.metadata as metadata
import json, platform, site, sys
assert sys.version_info[:2] == (3, 12), sys.version
assert sys.prefix != sys.base_prefix, "Not an isolated environment"
assert site.ENABLE_USER_SITE is False, "User site packages are enabled"
packages = {}
for name in ("torch", "transformers", "lm_eval", "datasets", "numpy"):
    try:
        packages[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        packages[name] = None
print(json.dumps({
    "python_version": platform.python_version(), "executable": sys.executable,
    "prefix": sys.prefix, "base_prefix": sys.base_prefix,
    "user_site_enabled": site.ENABLE_USER_SITE, "ml_packages": packages,
}))
'''
print("Checking the isolated interpreter...", flush=True)
probe = json.loads(run_setup([PYTHON, "-I", "-c", probe_code], 60))
if "include-system-site-packages = false" not in (VENV / "pyvenv.cfg").read_text().lower():
    raise RuntimeError("The environment includes system packages. Stop here.")
snapshot = {key: value for key, value in probe.items() if key != "ml_packages"}
snapshot.update({"uv_version": UV_VERSION, "proposal_sha256": expected["proposal_sha256"]})
write_once(snapshot_path, snapshot)
receipt = {
    "status": "TASK1_BLOCK02_OK", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "notebook_python": platform.python_version(), "task1_python": probe["python_version"],
    "task1_interpreter": str(PYTHON), "uv_version": UV_VERSION,
    "isolation_check_passed": True, "elapsed_seconds": round(time.monotonic() - started, 1),
    "ml_packages_currently_installed": probe["ml_packages"],
    "models_or_datasets_downloaded": False, "gpu_work_run": False,
    "next_step": "Resolve, install, and validate the approved ML package versions on CPU.",
}
write_once(ROOT / "logs" / ("block02_" + uuid.uuid4().hex + ".json"), receipt)
print(json.dumps(receipt, indent=2), flush=True)
print("TASK1_BLOCK02_OK - keep CPU selected and send the complete output.", flush=True)
