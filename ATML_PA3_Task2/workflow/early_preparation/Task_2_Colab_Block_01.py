# Block 1: CPU-only Colab setup. No installs, model loads, or dataset downloads.
import os, sys, json, uuid, shutil, hashlib, platform, subprocess
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from importlib import metadata
from google.colab import drive, runtime

# Refuse to keep an accidentally allocated accelerator for this CPU task.
gpu_detected = os.environ.get("COLAB_GPU", "0") not in ("", "0")
if shutil.which("nvidia-smi"):
    probe = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=10,
    )
    gpu_detected = gpu_detected or (probe.returncode == 0 and bool(probe.stdout.strip()))
if gpu_detected or os.environ.get("COLAB_TPU_ADDR"):
    print("Accelerator detected. This setup needs a CPU runtime. Requesting release.", flush=True)
    try:
        runtime.unassign()
    finally:
        raise RuntimeError("Select Runtime > Change runtime type > Hardware accelerator > None, then rerun.")

# Colab will ask you to authorize mounting your Google Drive.
drive.mount("/content/drive")
my_drive = Path("/content/drive/MyDrive")
if not my_drive.is_dir():
    raise RuntimeError("MyDrive was not found. Stop here and share the output.")

# A unique run folder preserves any existing work, including on a cell rerun.
started = datetime.now(ZoneInfo("Asia/Karachi"))
run_id = started.strftime("run_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
TASK2_ROOT = my_drive / "ATML_PA3" / "task2" / run_id
TASK2_ROOT.mkdir(parents=True, exist_ok=False)
for folder in ("provenance", "configs", "manifests", "logs", "checkpoints", "results"):
    (TASK2_ROOT / folder).mkdir(exist_ok=False)
TASK2_LOCAL = Path("/content/atml_pa3_task2") / run_id
TASK2_LOCAL.mkdir(parents=True, exist_ok=False)

# Full approved register, including conditions and details still to specify.
APPROVALS = json.loads(r'''
{
  "assignment": "ATML PA3 Task 2",
  "date": "2026-10-31",
  "timezone": "Asia/Karachi",
  "approval_scope": "All 22 proposed user-facing choices, with stated conditions",
  "supersedes": "Earlier assistant interpretation limiting approval to choices 1\u20134",
  "fresh_start": true,
  "experiment_started": false,
  "choices": [
    {"id": 1, "subject": "Scope", "status": "approved_as_proposed", "proposal_and_conditions": "Complete the mandatory 196-token run first; defer optional compression."},
    {"id": 2, "subject": "Storage", "status": "approved_as_proposed", "proposal_and_conditions": "Fresh Drive workspace for durable checkpoints/results; local Colab storage for active reads."},
    {"id": 3, "subject": "GPU segments", "status": "approved_as_proposed", "proposal_and_conditions": "At most 250 training updates per block; scheduled evaluation and verified durable saving before runtime release."},
    {"id": 4, "subject": "Error handling", "status": "approved_as_proposed", "proposal_and_conditions": "Stop on handled errors; preserve prior verified checkpoint and diagnostics; bounded save attempt followed by release."},
    {"id": 5, "subject": "Data selection", "status": "approved_as_proposed", "proposal_and_conditions": "Sort IDs before seeded selection; jointly allocate image-disjoint evaluation suites; preserve exact IDs and orders."},
    {"id": 6, "subject": "Caption selection and order", "status": "approved_as_proposed", "proposal_and_conditions": "Three distinct eligible original training captions per image, one development caption; two seeded shuffled training passes."},
    {"id": 7, "subject": "Grounding evaluation allocation", "status": "approved_as_proposed", "proposal_and_conditions": "Development: 12 presence and 12 counting pairs; final test: 32 presence and 32 counting pairs."},
    {"id": 8, "subject": "Grounding image universe", "status": "approved_as_proposed", "proposal_and_conditions": "Initially use union of selected caption/natural-VQA training images, subject to category/count feasibility; report and approve a revision if insufficient."},
    {"id": 9, "subject": "Grounding sampling", "status": "approved_as_proposed", "proposal_and_conditions": "Uniform eligible category selection, then eligible pairs; sample differing count combinations from 1 to 4."},
    {"id": 10, "subject": "Annotation quality", "status": "approved_as_proposed", "proposal_and_conditions": "Exclude crowd-ambiguous counting candidates; inspect all grounding evaluation pairs and a seeded training sample independently of model predictions. Exact remaining audit details are open."},
    {"id": 11, "subject": "Natural answer targets", "status": "approved_as_proposed", "proposal_and_conditions": "Official VQA normalization; deterministic original consensus answer using lowest annotation-answer ID; retain all ten references."},
    {"id": 12, "subject": "Content caps", "status": "approved_as_proposed", "proposal_and_conditions": "Prefer complete eligible examples within prescribed caps over truncation; verify every required quota remains feasible."},
    {"id": 13, "subject": "Prompts and chat format", "status": "approved_as_proposed", "proposal_and_conditions": "Fixed caption and short-answer instructions within native Qwen chat structure; exact text to be shown before tokenization; no additional trainable image markers or unapproved instructions."},
    {"id": 14, "subject": "Connector initialization and calibration", "status": "approved_as_proposed", "proposal_and_conditions": "Seeded standard Linear initialization including biases; training-only fixed RMS-based calibration. Exact sample/formula/scale placement remain to specify."},
    {"id": 15, "subject": "LoRA initialization", "status": "approved_as_proposed", "proposal_and_conditions": "Standard random A / zero B; keep all manual-prescribed adapter settings."},
    {"id": 16, "subject": "AdamW details", "status": "approved_as_proposed", "proposal_and_conditions": "Betas (0.9, 0.999), epsilon 1e-8, required weight decay 0.01 on all trainable parameters."},
    {"id": 17, "subject": "Stage transitions", "status": "approved_as_proposed", "proposal_and_conditions": "Retain connector moments across stages and LoRA moments from Stages 2 to 3; restart per-stage schedules; preserve scaler state."},
    {"id": 18, "subject": "Warmup rounding", "status": "approved_as_proposed", "proposal_and_conditions": "113, 75, 38 updates; exact first/warmup-end/final learning-rate conventions to be made reviewable before implementation."},
    {"id": 19, "subject": "Numerical failures", "status": "approved_as_proposed", "proposal_and_conditions": "Stop/report nonfinite loss/gradients or skipped updates; no silent change to dtype, clipping, batch, sampler, or budget; recovery details need agreement if failure occurs."},
    {"id": 20, "subject": "Evaluation conventions", "status": "approved_as_proposed", "proposal_and_conditions": "Seeded frozen mismatched mapping; generation timing excludes model loading; percentile pair-bootstrap intervals with seed 6304; additional development generation requires its own decision."},
    {"id": 21, "subject": "Qualitative selection", "status": "approved_as_proposed", "proposal_and_conditions": "Reproducible coverage of successes, failures, and image-swap disagreements; exact categories/counts to specify before final analysis."},
    {"id": 22, "subject": "Software and publication", "status": "approved_as_proposed", "proposal_and_conditions": "Pin compatible libraries and model revisions after checks; separate authorization for publication; no silent recipe or acceleration changes."}
  ]
}
''')
APPROVALS["recorded_in_colab_at"] = started.isoformat()
APPROVALS["approval_source"] = (
    "User clarified that approval covered all 22 choices, not only the first batch. "
    "Apply sensibly and preserve decisions for the report."
)

# This checks bytes read back from mounted Drive; it does not test GPU recovery.
def save_verified_json(relative_path, value):
    path = TASK2_ROOT / relative_path
    payload = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    temp = path.with_name(path.name + ".pending")
    with temp.open("xb") as stream:
        stream.write(payload)
        stream.flush()
    if temp.read_bytes() != payload:
        raise IOError(f"Readback mismatch: {relative_path}")
    if path.exists():
        raise FileExistsError(f"Refusing to replace {path}")
    temp.rename(path)
    if path.read_bytes() != payload:
        raise IOError(f"Final readback mismatch: {relative_path}")
    return {"path": str(path), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}

packages = {}
for name in ("torch", "transformers", "peft", "accelerate", "numpy", "Pillow", "huggingface-hub"):
    try:
        packages[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        packages[name] = None

local_disk = shutil.disk_usage("/content")
report = {
    "timestamp": started.isoformat(), "run_id": run_id,
    "workspace_drive": str(TASK2_ROOT), "workspace_local": str(TASK2_LOCAL),
    "python": sys.version, "platform": platform.platform(), "packages_observed": packages,
    "gpu_detected": False, "models_loaded": False, "training_started": False,
    "dataset_downloads_started": False,
    "local_disk_free_GiB": round(local_disk.free / 2**30, 2),
    "drive_account_quota": "NOT VERIFIED: mounted filesystem capacity is not account quota",
    "gpu_safety_status": "No GPU task executed; training save/resume/release safeguards not yet tested",
    "next_unit": "CPU metadata and tokenizer feasibility checks, after reviewing this output",
}
files = [
    save_verified_json("provenance/approved_choices.json", APPROVALS),
    save_verified_json("logs/block_01_preflight.json", report),
]
save_verified_json("provenance/block_01_saved_files.json", {"files": files})
print("\nBLOCK 1 COMPLETE — CPU setup; 22 approved choices saved and read back.")
print(json.dumps(report, indent=2))
print("\nSaved records:")
for item in files:
    print(item["path"], "\n  SHA256:", item["sha256"])
print("\nKeep hardware accelerator set to None. Paste this complete output into the chat.")
print("Also tell me the free space shown in your Google Drive storage page; account quota is not verified above.")
