"""CPU-only Colab workspace setup; no dependencies, model loads or data downloads."""
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import uuid
from datetime import datetime
from importlib import metadata
from pathlib import Path
from zoneinfo import ZoneInfo


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def encode_json(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def accelerator_evidence(environ=None, which=shutil.which, run=subprocess.run, dev=Path("/dev")):
    """Detect allocated accelerators without importing a model or GPU framework."""
    environ = os.environ if environ is None else environ
    evidence = []
    value = environ.get("COLAB_GPU", "0").strip().lower()
    if value not in ("", "0", "false", "none"):
        evidence.append("COLAB_GPU indicates an accelerator allocation")
    if any(environ.get(key) for key in ("COLAB_TPU_ADDR", "TPU_NAME", "TPU_WORKER_ID")):
        evidence.append("TPU allocation variable present")
    if environ.get("PJRT_DEVICE", "").upper() == "TPU":
        evidence.append("PJRT_DEVICE is TPU")
    if list(dev.glob("nvidia[0-9]*")):
        evidence.append("NVIDIA device nodes present")
    executable = which("nvidia-smi")
    if executable:
        try:
            probe = run([executable, "--query-gpu=name", "--format=csv,noheader"],
                        capture_output=True, text=True, timeout=10, check=False)
        except (OSError, subprocess.TimeoutExpired):
            evidence.append("GPU probe failed; CPU-only allocation could not be established")
        else:
            if probe.returncode == 0 and probe.stdout.strip():
                evidence.append("nvidia-smi reports: " + probe.stdout.strip())
            elif probe.returncode != 0:
                evidence.append("GPU utility present but probe failed; allocation is uncertain")
    return evidence


def request_release_bounded(callback, seconds=20):
    """A request result is never independent evidence of physical deallocation."""
    result = {"requested": True, "confirmed_deallocation": False}

    def invoke():
        try:
            callback()
            result["api_returned"] = True
        except Exception as error:
            result["error_type"] = type(error).__name__

    thread = threading.Thread(target=invoke, daemon=True)
    thread.start()
    thread.join(seconds)
    snapshot = dict(result)
    snapshot["request_still_running"] = thread.is_alive()
    return snapshot


def save_immutable(path, payload):
    """Write once/read back; refuse changed existing bytes. No remote-durability claim."""
    path = Path(path)
    if path.is_symlink():
        raise RuntimeError(f"Refusing a symlink: {path}")
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise RuntimeError(f"Existing content differs; preserving it: {path}")
        return {"name": path.name, "sha256": digest(payload), "bytes": len(payload), "reused": True}
    pending = path.with_name(path.name + ".pending_" + uuid.uuid4().hex)
    with pending.open("xb") as stream:
        stream.write(payload)
        stream.flush()
    if pending.read_bytes() != payload:
        raise IOError(f"Readback failed: {pending}")
    # Sequential notebook use; never run two copies of this setup simultaneously.
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"Concurrent setup detected; preserving both files: {path}")
    pending.rename(path)
    if path.read_bytes() != payload:
        raise IOError(f"Final readback failed: {path}")
    return {"name": path.name, "sha256": digest(payload), "bytes": len(payload), "reused": False}


def load_protocol(bundle):
    plan_bytes = (bundle / "APPROVED_PLAN.json").read_bytes()
    requirements_bytes = (bundle / "MANUAL_REQUIREMENTS.json").read_bytes()
    plan = json.loads(plan_bytes)
    requirements = json.loads(requirements_bytes)
    if plan["assignment"] != "ATML PA3 Task 3" or [x["id"] for x in plan["choices"]] != list(range(1, 33)):
        raise ValueError("Wrong or incomplete Task 3 approval register")
    if requirements["analysis_seed"] != 6304 or requirements["live"]["model_revision"] != "aa8e72537993ba99e69dfaafa59ed015b17504d1":
        raise ValueError("Unexpected manual protocol")
    return plan, requirements, plan_bytes, requirements_bytes


def setup_workspace(bundle, my_drive, local_base):
    """Pure filesystem portion, separately exercised locally with temporary folders."""
    bundle, my_drive, local_base = Path(bundle), Path(my_drive), Path(local_base)
    plan, requirements, plan_bytes, requirements_bytes = load_protocol(bundle)
    if not my_drive.is_dir():
        raise RuntimeError("MyDrive is unavailable. Stop and share the complete output.")
    root = my_drive / "ATML_PA3_Task3"
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise RuntimeError("Task 3 path is not an ordinary directory; preserving it.")
    marker = root / "WORKSPACE.json"
    identity = {"schema_version": 1, "task": "ATML PA3 Task 3",
                "plan_sha256": digest(plan_bytes), "requirements_sha256": digest(requirements_bytes)}
    if root.exists() and not marker.exists() and any(root.iterdir()):
        raise RuntimeError("Task 3 folder already has unrecognized content. Preserving it; share output.")
    if marker.exists() and (marker.is_symlink() or marker.read_bytes() != encode_json(identity)):
        raise RuntimeError("Workspace identity differs. Preserving it; share output before continuing.")
    folders = ("provenance", "configs", "sources", "metadata", "manifests", "logs", "results", "progress", "live", "exports")
    for name in folders:
        target = root / name
        if target.is_symlink() or (target.exists() and not target.is_dir()):
            raise RuntimeError(f"Unexpected workspace subfolder: {name}")
    root.mkdir(exist_ok=True)
    saved = [save_immutable(marker, encode_json(identity))]
    for name in folders:
        (root / name).mkdir(exist_ok=True)
    assets = {
        "provenance/APPROVED_PLAN.json": plan_bytes,
        "configs/MANUAL_REQUIREMENTS.json": requirements_bytes,
        "provenance/DECISION_REVIEW.md": (bundle / "DECISION_REVIEW.md").read_bytes(),
        "provenance/block_01.py": (bundle / "block_01.py").read_bytes(),
    }
    # Check all existing immutable targets before publishing any new asset.
    for relative, payload in assets.items():
        path = root / relative
        if path.is_symlink() or (path.exists() and (not path.is_file() or path.read_bytes() != payload)):
            raise RuntimeError(f"Existing artifact differs; preserving it: {relative}")
    for relative, payload in assets.items():
        entry = save_immutable(root / relative, payload)
        entry["relative_path"] = relative
        saved.append(entry)
    local_base.mkdir(parents=True, exist_ok=True)
    started = datetime.now(ZoneInfo("Asia/Karachi"))
    attempt = started.strftime("setup_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
    versions = {}
    for name in ("numpy", "pandas", "pyarrow", "math-verify", "torch", "transformers", "accelerate", "huggingface-hub", "gdown"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    receipt = {
        "status": "CPU SETUP COMPLETE", "attempt": attempt, "timestamp": started.isoformat(),
        "workspace_drive": str(root), "workspace_local": str(local_base),
        "approval": "A001; all 32 choices reviewed and approved as proposed",
        "plan_sha256": digest(plan_bytes), "requirements_sha256": digest(requirements_bytes),
        "python": sys.version, "platform": platform.platform(), "observed_versions_not_yet_pinned": versions,
        "local_disk_free_GiB": round(shutil.disk_usage(local_base).free / 2**30, 2),
        "drive_quota": "Not checked; mounted filesystem capacity is not account quota",
        "readback_scope": "Mounted-filesystem bytes verified; remote durability not independently verified",
        "gpu_work_executed": False, "gpu_watchdog_tested": False, "packages_installed": False,
        "data_downloaded": False, "models_downloaded_or_loaded": False,
        "splits_constructed": False, "scoring_executed": False, "held_out_scores_inspected": False,
        "files": saved,
        "next": "Share full output. Next block will acquire and validate supplied data on CPU.",
    }
    receipt_path = root / "logs" / (attempt + ".json")
    save_immutable(receipt_path, encode_json(receipt))
    receipt["receipt_path"] = str(receipt_path)
    return receipt


def main(bundle_dir):
    from google.colab import drive, runtime

    evidence = accelerator_evidence()
    if evidence:
        print("CPU task: accelerator detected or allocation uncertain. Requesting runtime release.", flush=True)
        print(json.dumps({"evidence": evidence, "release": request_release_bounded(runtime.unassign)}, indent=2), flush=True)
        raise RuntimeError("Select Runtime > Change runtime type > Hardware accelerator > None. If still connected, use Disconnect and delete runtime. Then rerun this block.")
    print("CPU-only allocation check passed. Mounting Drive for Task 3 setup.", flush=True)
    drive.mount("/content/drive", timeout_ms=120000)
    receipt = setup_workspace(Path(bundle_dir), Path("/content/drive/MyDrive"), Path("/content/atml_pa3_task3"))
    print(json.dumps(receipt, indent=2), flush=True)
    print("CPU SETUP COMPLETE — keep accelerator None and share the full output.", flush=True)
    return receipt
