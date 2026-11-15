# Task 1 - Block 03: complete package setup and CPU smoke checks.
# Run below Blocks 01-02, with CPU selected. Several GB may be downloaded.
import textwrap
import traceback

if not all(k in globals() for k in ("check_cpu_only", "write_once", "ROOT", "LOCAL", "UV", "PYTHON", "env")):
    raise RuntimeError("Run Blocks 01 and 02 successfully in this notebook first.")
check_cpu_only()
env = env.copy()
env.pop("UV_SYSTEM_PYTHON", None)  # Never let an inherited system-install flag select Colab's environment.
if json.loads((ROOT / "provenance" / "APPROVAL_REFERENCE.json").read_text()) != APPROVAL:
    raise RuntimeError("Approval reference differs. Stop here.")
if shutil.disk_usage(LOCAL).free < 30 * 2**30:
    raise RuntimeError("Less than 30 GiB runtime disk free. Stop and share the output.")

PINS = {
    "torch": "2.8.0+cu126", "transformers": "4.57.1", "lm_eval": "0.4.9.2",
    "datasets": "4.3.0", "accelerate": "1.11.0", "tokenizers": "0.22.1",
    "safetensors": "0.6.2", "huggingface-hub": "0.36.0", "numpy": "2.2.6",
    "peft": "0.17.1", "evaluate": "0.4.6",
}
TORCH_URL = (
    "https://download.pytorch.org/whl/cu126/"
    "torch-2.8.0%2Bcu126-cp312-cp312-manylinux_2_28_x86_64.whl"
)
requirements = "torch @ " + TORCH_URL + "\n" + "\n".join(
    f"{name}=={version}" for name, version in PINS.items() if name != "torch"
) + "\n"
WORK = LOCAL / "environment_setup"
WORK.mkdir(exist_ok=True)
attempt = "block03_" + uuid.uuid4().hex
logs = []
started = time.monotonic()
phase = "initialization"


def save_bytes(path, data):
    if path.is_symlink():
        raise RuntimeError(f"Unexpected linked file: {path}")
    if path.exists():
        if not path.is_file() or path.read_bytes() != data:
            raise RuntimeError(f"Existing frozen file differs; preserving it: {path}")
    else:
        temporary = path.with_name(path.name + ".pending_" + uuid.uuid4().hex)
        temporary.write_bytes(data)
        if temporary.read_bytes() != data:
            raise IOError(f"Readback failed: {temporary}")
        if path.exists() or path.is_symlink():
            raise RuntimeError("Concurrent setup detected. Run only one copy.")
        temporary.rename(path)
    if path.read_bytes() != data:
        raise IOError(f"Final readback failed: {path}")


def command(args, timeout=1200, child_env=None):
    result = subprocess.run(
        [str(x) for x in args], cwd=str(WORK), env=child_env or env,
        capture_output=True, text=True, timeout=timeout, check=False,
    )
    output = result.stdout + "\n" + result.stderr
    logs.append(f"\nPHASE: {phase}\n{output}")
    if output.strip():
        print("\n".join(output.strip().splitlines()[-35:]), flush=True)
    if result.returncode:
        raise RuntimeError(f"{phase} failed (exit {result.returncode}); approved pins were not changed.")
    return result.stdout


try:
    save_bytes(ROOT / "configs" / "requirements.direct.in", requirements.encode())
    write_once(ROOT / "configs" / "APPROVED_PACKAGE_PINS.json", PINS)
    direct = WORK / "requirements.direct.in"
    direct.write_text(requirements)
    frozen = ROOT / "configs" / "requirements.lock.txt"
    lock = WORK / "requirements.lock.txt"

    phase = "resolve and freeze dependencies"
    print("1/4 Resolving approved packages and freezing dependency hashes...", flush=True)
    if frozen.exists():
        lock.write_bytes(frozen.read_bytes())
        print("Reusing the saved dependency lock.", flush=True)
    else:
        candidate = WORK / (attempt + ".lock")
        command([UV, "pip", "compile", direct, "--python", PYTHON,
                 "--index-url", "https://pypi.org/simple", "--generate-hashes",
                 "--no-header", "--no-annotate", "--quiet", "--output-file", candidate])
        save_bytes(frozen, candidate.read_bytes())
        lock.write_bytes(candidate.read_bytes())
    lock_sha = hashlib.sha256(lock.read_bytes()).hexdigest()
    write_once(ROOT / "configs" / "DEPENDENCY_LOCK.json", {"sha256": lock_sha, "pins": PINS})

    phase = "install frozen environment"
    print("2/4 Installing the complete stack into the private Python 3.12 environment...", flush=True)
    command([UV, "pip", "sync", "--python", PYTHON, "--require-hashes",
             "--index-url", "https://pypi.org/simple", lock])
    command([UV, "pip", "check", "--python", PYTHON], timeout=120)

    phase = "CPU imports, Qwen forward/backward, and stock task discovery"
    print("3/4 Checking imports, a tiny random Qwen model, and all four stock task definitions...", flush=True)
    probe_source = textwrap.dedent(r'''
        import hashlib, importlib.metadata as md, json, pathlib, sys
        import torch, transformers, datasets, accelerate, peft, evaluate, tokenizers, safetensors
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        from lm_eval.tasks import TaskManager
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from lm_eval.utils import load_yaml_config
        pins = json.loads(sys.argv[1])
        versions = {name: md.version(name) for name in pins}
        assert versions == pins, (versions, pins)
        assert sys.version_info[:2] == (3,12)
        assert torch.version.cuda == "12.6", torch.version.cuda
        torch.set_num_threads(2)
        torch.manual_seed(6304)
        config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            head_dim=16, max_position_embeddings=64, use_cache=False)
        config._attn_implementation = "sdpa"
        model = Qwen3ForCausalLM(config).float().cpu()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
        model.train()
        ids = torch.randint(0,64,(2,16))
        output = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids)
        assert torch.isfinite(output.loss)
        output.loss.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
        assert all(p.device.type == "cpu" for p in model.parameters())
        manager = TaskManager()
        definitions = {}
        for name in ("gsm8k_cot", "hellaswag", "boolq", "arc_easy"):
            entry = manager.task_index[name]
            path = pathlib.Path(entry["yaml_path"])
            cfg = load_yaml_config(str(path), mode="simple")
            definitions[name] = {"filename":path.name, "sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                "stock_num_fewshot":cfg.get("num_fewshot"), "output_type":cfg.get("output_type")}
        package_root = pathlib.Path(lm_eval.__file__).parent
        hashes = {str(p.relative_to(package_root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(package_root.rglob("*")) if p.is_file() and p.suffix in (".py",".yaml")}
        versions_all = {d.metadata["Name"]:d.version for d in md.distributions() if d.metadata["Name"]}
        result = {"versions":versions,"all_installed_versions":versions_all,
            "torch_cuda_build":torch.version.cuda,"tiny_cpu_loss":float(output.loss.detach()),
            "cpu_forward_backward_passed":True,"stock_tasks":definitions,
            "harness_source_hashes":hashes,"task_datasets_loaded":False,
            "pretrained_weights_downloaded":False,"gpu_work_run":False}
        pathlib.Path(sys.argv[2]).write_text(json.dumps(result,indent=2)+"\n")
        print("All direct pins match; tiny CPU forward/backward passed; four stock tasks found.")
    ''')
    probe = WORK / "cpu_environment_probe.py"
    probe.write_text(probe_source)
    checked = WORK / (attempt + "_checks.json")
    offline = env.copy()
    offline.update({"HF_HUB_OFFLINE":"1", "HF_DATASETS_OFFLINE":"1",
                    "HF_HUB_DISABLE_TELEMETRY":"1", "TOKENIZERS_PARALLELISM":"false"})
    command([PYTHON, "-I", probe, json.dumps(PINS), checked], timeout=300, child_env=offline)
    checks = json.loads(checked.read_text())

    phase = "save verified environment records"
    print("4/4 Saving the environment, task fingerprints, checks, and setup logs to Drive...", flush=True)
    write_once(ROOT / "configs" / "INSTALLED_PACKAGE_VERSIONS.json", checks["all_installed_versions"])
    write_once(ROOT / "configs" / "HARNESS_SOURCE_HASHES.json", checks["harness_source_hashes"])
    save_bytes(ROOT / "provenance" / "cpu_environment_probe.py", probe.read_bytes())
    save_bytes(ROOT / "logs" / (attempt + "_checks.json"), checked.read_bytes())
    write_once(ROOT / "configs" / "ENVIRONMENT_READY.json", {
        "approval_id":"T1-A001", "lock_sha256":lock_sha, "versions":checks["versions"],
        "python_version":json.loads((ROOT / "configs" / "PYTHON_ENVIRONMENT.json").read_text())["python_version"],
        "cpu_smoke_checks_passed":True, "gpu_compatibility_tested":False})
    receipt = {"status":"TASK1_BLOCK03_OK", "versions":checks["versions"],
        "lock_sha256":lock_sha, "cpu_forward_backward_passed":True,
        "stock_tasks_found":list(checks["stock_tasks"]),
        "elapsed_seconds":round(time.monotonic()-started,1),
        "runtime_disk_free_GiB":round(shutil.disk_usage(LOCAL).free/2**30,2),
        "models_or_datasets_downloaded":False, "gpu_work_run":False,
        "next_stage":"Acquire sources, freeze IDs, and prepare/audit the training data on CPU."}
    write_once(ROOT / "logs" / (attempt + ".json"), receipt)
    print(json.dumps(receipt,indent=2),flush=True)
    print("TASK1_BLOCK03_OK - keep CPU selected and send the output.",flush=True)
except BaseException:
    logs.append(traceback.format_exc())
    raise
finally:
    save_bytes(ROOT / "logs" / (attempt + "_setup.txt"), "\n".join(logs).encode())
