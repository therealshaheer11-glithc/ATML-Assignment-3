# Block 2: CPU compatibility, revision pinning, and tokenizer/config files only.
import os, sys, json, uuid, shutil, hashlib, subprocess, traceback
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from importlib import metadata
from google.colab import runtime

ROOT = Path('/content/drive/MyDrive/ATML_PA3/task2/run_20261031_150949_1cff148a')
LOCAL = Path('/content/atml_pa3_task2/run_20261031_150949_1cff148a')
EXPECTED = {
    'provenance/approved_choices.json': 'ec1e97439744d8321fedc8256e5c0f48e095518e4d17386271b3b2bda7965c33',
    'logs/block_01_preflight.json': '110ff8763f7aef0e2a1b2d75fcb4e55c11b1836babd3a4d265ca67ba312cbe0c',
}
# Keep preparation CPU-only, even if the runtime was changed accidentally.
accelerator = os.environ.get('COLAB_GPU', '0') not in ('', '0')
if shutil.which('nvidia-smi'):
    p = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                       capture_output=True, text=True, timeout=10)
    accelerator |= p.returncode == 0 and bool(p.stdout.strip())
if accelerator or os.environ.get('COLAB_TPU_ADDR'):
    print('CPU task: requesting release of accidentally allocated accelerator.', flush=True)
    try:
        runtime.unassign()
    finally:
        raise RuntimeError('Reconnect with hardware accelerator None and rerun.')

for relative, expected in EXPECTED.items():
    path = ROOT / relative
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError(f'Missing or changed Block 1 record: {relative}. Stop and share output.')
LOCAL.mkdir(parents=True, exist_ok=True)
attempt = datetime.now(ZoneInfo('Asia/Karachi')).strftime('%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:8]

def save_record(relative, value):
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, ensure_ascii=False) + '\n').encode()
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f'Refusing to change existing record: {relative}')
    else:
        temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.pending')
        temporary.write_bytes(data)
        if temporary.read_bytes() != data:
            raise IOError(f'Readback failed: {relative}')
        temporary.rename(path)
    if path.read_bytes() != data:
        raise IOError(f'Final readback failed: {relative}')
    return hashlib.sha256(data).hexdigest()

report = {
    'attempt': attempt, 'status': 'started', 'gpu_used': False,
    'weights_loaded': False, 'image_downloads_started': False,
    'storage_evidence': {'source': 'User screenshot, 31 October 2026',
                         'Google_total_GB': 400, 'Google_used_GB': 36.18,
                         'Google_free_GB_approx': 363.82},
    'packages_observed': {},
}
try:
    # No installation or upgrade: test the environment the user actually has.
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
    from transformers.models.siglip.modeling_siglip import SiglipVisionModel
    from peft import LoraConfig

    for name in ('torch', 'transformers', 'peft', 'accelerate', 'numpy',
                 'Pillow', 'huggingface-hub', 'tokenizers'):
        report['packages_observed'][name] = metadata.version(name)

    repos = {'qwen': 'Qwen/Qwen3-0.6B', 'siglip': 'google/siglip-base-patch16-224'}
    lock_path = ROOT / 'configs/model_revisions.json'
    api = HfApi(token=False)
    if lock_path.exists():
        pins = json.loads(lock_path.read_text())
        if set(pins) != set(repos):
            raise RuntimeError('Unexpected model revision lock.')
    else:
        pins = {}
        for name, repo in repos.items():
            info = api.model_info(repo, revision='main', timeout=30)
            pins[name] = {'repo_id': repo, 'revision': info.sha}
        save_record('configs/model_revisions.json', pins)
    for name, repo in repos.items():
        if pins[name]['repo_id'] != repo or len(pins[name]['revision']) != 40:
            raise RuntimeError('Invalid pinned model identity.')
    report['model_revisions'] = pins

    # Explicit whitelist: these downloads cannot include model weight files.
    allowed = ['config.json', 'generation_config.json', 'tokenizer_config.json',
               'tokenizer.json', 'tokenizer.model', 'special_tokens_map.json',
               'vocab.json', 'merges.txt', 'added_tokens.json',
               'chat_template.jinja', 'preprocessor_config.json']
    model_paths, inventory = {}, []
    for name, pin in pins.items():
        source = Path(snapshot_download(
            pin['repo_id'], revision=pin['revision'], allow_patterns=allowed,
            cache_dir=str(LOCAL / 'hub'), token=False, max_workers=2,
        ))
        target = ROOT / 'provenance/model_assets' / name / pin['revision']
        target.mkdir(parents=True, exist_ok=True)
        for filename in allowed:
            src = source / filename
            if not src.is_file():
                continue
            data = src.read_bytes()
            dst = target / filename
            if dst.exists() and dst.read_bytes() != data:
                raise RuntimeError(f'Conflicting saved asset: {filename}')
            if not dst.exists():
                dst.write_bytes(data)
            if dst.read_bytes() != data:
                raise IOError(f'Asset readback failed: {filename}')
            inventory.append({'model': name, 'filename': filename, 'bytes': len(data),
                              'sha256': hashlib.sha256(data).hexdigest()})
        model_paths[name] = source

    qconfig = AutoConfig.from_pretrained(model_paths['qwen'], local_files_only=True,
                                         trust_remote_code=False)
    sconfig = AutoConfig.from_pretrained(model_paths['siglip'], local_files_only=True,
                                         trust_remote_code=False)
    vision = sconfig.vision_config
    assert qconfig.model_type == 'qwen3' and qconfig.hidden_size == 1024
    assert sconfig.model_type == 'siglip' and vision.hidden_size == 768
    assert vision.image_size == 224 and vision.patch_size == 16
    assert (vision.image_size // vision.patch_size) ** 2 == 196

    tokenizer = AutoTokenizer.from_pretrained(
        model_paths['qwen'], local_files_only=True, trust_remote_code=False)
    smoke = [{'role': 'user', 'content': 'Interface smoke test.'}]
    off = tokenizer.apply_chat_template(smoke, tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
    on = tokenizer.apply_chat_template(smoke, tokenize=False,
                                      add_generation_prompt=True, enable_thinking=True)
    assert 'enable_thinking' in tokenizer.get_chat_template()
    assert off != on, 'Thinking switch did not change the serialized prompt.'
    ids = tokenizer(off, add_special_tokens=False)['input_ids']
    assert tokenizer.decode(ids, skip_special_tokens=False) == off
    assert tokenizer.eos_token_id == qconfig.eos_token_id
    assert tokenizer.encode('<|im_end|>', add_special_tokens=False) == [qconfig.eos_token_id]

    # Store processor settings without choosing a new preprocessing recipe.
    processor = json.loads((model_paths['siglip'] / 'preprocessor_config.json').read_text())
    report.update({
        'status': 'passed',
        'checks': {'model_classes_imported': True, 'required_dimensions': True,
                   'thinking_switch_effective': True, 'tokenizer_roundtrip': True,
                   'end_of_turn_token_verified': True},
        'qwen_hidden_size': qconfig.hidden_size,
        'vision_hidden_size': vision.hidden_size, 'visual_tokens': 196,
        'siglip_processor_settings': processor,
        'nonthinking_smoke_serialization': off,
        'smoke_test_is_experiment_prompt': False,
        'tokenizer_eos_id': tokenizer.eos_token_id, 'tokenizer_pad_id': tokenizer.pad_token_id,
        'asset_file_count': len(inventory),
        'local_disk_free_GiB': round(shutil.disk_usage('/content').free / 2**30, 2),
        'gpu_environment_validated': False,
    })
    save_record('provenance/model_assets_manifest.json', inventory)
except Exception:
    report['status'] = 'failed'
    report['traceback'] = traceback.format_exc()
    failure_path = f'logs/block_02_{attempt}.json'
    save_record(failure_path, report)
    print(f'BLOCK 2 FAILED; diagnostics saved to {ROOT / failure_path}', flush=True)
    raise

log_path = f'logs/block_02_{attempt}.json'
digest = save_record(log_path, report)
print('\nBLOCK 2 COMPLETE — CPU config/tokenizer checks passed; no weights loaded.')
print(json.dumps(report, indent=2))
print('Saved log:', ROOT / log_path)
print('SHA256:', digest)
print('Keep accelerator None. Paste the complete output into the chat.')
