# Task 1 - Block 01. Run in a NEW Task 1 notebook with CPU selected.
# Standard library + Colab only: no installations, downloads, or model loading.
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

APPROVAL = {
    "assignment": "ATML PA3 Task 1",
    "approval_id": "T1-A001",
    "approval_date": "2026-11-12",
    "approved_decisions": [f"D{i:02}" for i in range(1, 23)],
    "condition": "Faithful to the manual; meaningful comparisons; no manipulation or silent deviations.",
    "proposal_sha256": "d13f6172406705a258d7319d20c38ebed78309a997c87ed065e84388fe0bf0a4",
    "manual_sha256": "ce74ac092dbdb83b44afe75c3f5a22e1ef3f2f2bb6048f8175e21d63e76659c9",
    "task1_compute_unit_limit": 55.0,
    "gpu_first_choice": "L4, only after CPU preparation and checks",
}


def check_cpu_only():
    evidence = []
    if os.environ.get("COLAB_GPU", "0").lower() not in ("", "0", "false", "none"):
        evidence.append("COLAB_GPU indicates an accelerator")
    if any(os.environ.get(k) for k in ("COLAB_TPU_ADDR", "TPU_NAME", "TPU_WORKER_ID")):
        evidence.append("TPU allocation variable present")
    if os.environ.get("PJRT_DEVICE", "").upper() == "TPU":
        evidence.append("TPU runtime detected")
    if list(Path("/dev").glob("nvidia[0-9]*")):
        evidence.append("NVIDIA GPU device present")
    executable = shutil.which("nvidia-smi")
    if executable:
        try:
            probe = subprocess.run(
                [executable, "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10, check=False,
            )
            if probe.returncode == 0 and probe.stdout.strip():
                evidence.append(probe.stdout.strip())
            elif probe.returncode != 0:
                evidence.append("GPU probe failed; CPU allocation is uncertain")
        except (OSError, subprocess.TimeoutExpired):
            evidence.append("GPU probe unavailable; CPU allocation is uncertain")
    if evidence:
        print("STOP: CPU required. " + "; ".join(evidence), flush=True)
        print("Requesting runtime release. Select CPU before running again.", flush=True)

        def release():
            try:
                from google.colab import runtime
                runtime.unassign()
            except Exception as error:
                print("Release request failed:", type(error).__name__, flush=True)

        thread = threading.Thread(target=release, daemon=True)
        thread.start()
        thread.join(timeout=15)
        raise RuntimeError("CPU check failed. If still connected, disconnect/delete the runtime.")


def write_once(path, value):
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if path.is_symlink():
        raise RuntimeError(f"Unexpected linked file; preserving it: {path}")
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise RuntimeError(f"Existing file differs; preserving it: {path}")
    else:
        pending = path.with_name(path.name + ".pending_" + uuid.uuid4().hex)
        with pending.open("xb") as stream:
            stream.write(payload)
            stream.flush()
        if pending.read_bytes() != payload:
            raise IOError(f"Read-back verification failed: {pending}")
        if path.exists() or path.is_symlink():
            raise RuntimeError("Concurrent setup detected. Run only one copy of this cell.")
        pending.rename(path)
    if path.read_bytes() != payload:
        raise IOError(f"Final read-back verification failed: {path}")
    return hashlib.sha256(payload).hexdigest()


def setup_workspace(my_drive):
    if not my_drive.is_dir():
        raise RuntimeError("MyDrive is unavailable. Stop and share the output.")
    root = my_drive / "ATML-PA-3" / "Task1"
    if root.parent.is_symlink() or root.is_symlink():
        raise RuntimeError("Unexpected linked folder; preserving it.")
    marker = root / "WORKSPACE.json"
    identity = {"schema_version": 1, "assignment": APPROVAL["assignment"],
                "approval_id": APPROVAL["approval_id"],
                "proposal_sha256": APPROVAL["proposal_sha256"]}
    if root.exists() and (not root.is_dir() or (not marker.exists() and any(root.iterdir()))):
        raise RuntimeError("Task1 folder has unrecognized content. Preserving it; share the output.")
    root.mkdir(parents=True, exist_ok=True)
    write_once(marker, identity)
    for name in ("configs", "provenance", "sources", "manifests", "checkpoints",
                 "logs", "results", "diagnostics", "qualitative"):
        target = root / name
        if target.is_symlink() or (target.exists() and not target.is_dir()):
            raise RuntimeError(f"Unexpected folder content; preserving it: {target}")
        target.mkdir(exist_ok=True)
    write_once(root / "provenance" / "APPROVAL_REFERENCE.json", APPROVAL)
    receipt = {
        "status": "TASK1_BLOCK01_OK",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "workspace": str(root), "python": sys.version,
        "platform": platform.platform(),
        "runtime_disk_free_GiB": round(shutil.disk_usage("/content").free / 2**30, 2),
        "packages_installed": False, "models_or_data_downloaded": False,
        "gpu_work_run": False, "training_started": False,
        "full_approval_plan_synchronized": False,
        "qualitative_file_at_expected_path": (root / "sources" / "first_plot_questions.yaml").is_file(),
        "drive_quota": "Not measured; runtime disk is separate from Drive quota",
        "readback": "Mounted-filesystem bytes checked; not independent proof of remote durability",
    }
    write_once(root / "logs" / ("block01_" + uuid.uuid4().hex + ".json"), receipt)
    return receipt


if __name__ == "__main__":
    check_cpu_only()  # Must happen before mounting Drive or writing files.
    print("CPU check passed. Mounting Drive for Task 1 setup.", flush=True)
    from google.colab import drive
    drive.mount("/content/drive")
    print(json.dumps(setup_workspace(Path("/content/drive/MyDrive")), indent=2), flush=True)
    print("TASK1_BLOCK01_OK - keep CPU selected and send the complete output.", flush=True)
