"""Block 10. CPU download and integrity staging of both pinned original models."""
import os
import sys
import io
import json
import hashlib
import importlib.metadata
import platform
import shutil
import time
import traceback
import uuid
import zipfile
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter

if os.environ.get('COLAB_GPU', '0') not in ('', '0'):
    from google.colab import runtime
    try:
        runtime.unassign()
    finally:
        raise RuntimeError('Reconnect with accelerator None for CPU model preparation.')
from google.colab import drive
if not Path('/content/drive/MyDrive').exists():
    drive.mount('/content/drive')

RUN_ROOT = Path('/content/drive/MyDrive/ATML_PA3/task2/run_20261031_150949_1cff148a')
LOCAL_ROOT = Path('/content/atml_pa3_task2/run_20261031_150949_1cff148a')
DATASET_SHA = '82ad5006bd1d68ee54e639e50e7d835d8f6d97673537c38dab8d8eebdd8ac1b2'
FREEZE_SHA = 'fe7c9d4efa5d92d5b809eadb3695adcdd19f2421c6d43a2cb1593a7b86c3245b'
SHARDS_SHA = 'd3642cd07bf0f1ccd82cf6bd8c5e947d619237b8cd5860499d6f9972bbff3ad7'
SHARDS_RECIPE = 'e5683db7637de6712b8fc790f99ae71497c3c1e89ed5af4dcdd0d9b5b552b09a'
LOCKS = {'siglip': {'repo_id': 'google/siglip-base-patch16-224',
        'revision': '7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed'},
    'qwen': {'repo_id': 'Qwen/Qwen3-0.6B', 'revision': 'c1899de289a04d12100db370d81485cdf75e47ca'}}
package_bytes = globals().get('task2_models_package_bytes')
if not isinstance(package_bytes, bytes):
    raise RuntimeError('Run the supplied Block 10 package loader.')
package_sha = hashlib.sha256(package_bytes).hexdigest()
work = LOCAL_ROOT / 'model_prep_code' / package_sha[:16]
with zipfile.ZipFile(io.BytesIO(package_bytes)) as archive:
    assert archive.testzip() is None
    package_manifest = json.loads(archive.read('package_manifest.json'))
    assert set(archive.namelist()) == {r['path'] for r in package_manifest['files']} | {'package_manifest.json'}
    for record in package_manifest['files']:
        relative = Path(record['path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        content = archive.read(record['path'])
        assert len(content) == record['bytes'] and hashlib.sha256(content).hexdigest() == record['sha256']
        destination = work / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            assert destination.read_bytes() == content
        else:
            destination.write_bytes(content)
sys.path.insert(0, str(work))
for name in list(sys.modules):
    if name == 'task2' or name.startswith('task2.'):
        del sys.modules[name]
from task2.images import encoded, file_sha, immutable_bytes
from task2.model_assets import make_plan, ModelAssetStore, verify_model_structure

attempt = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:8]
started = time.monotonic()
models, activities = {}, {}
try:
    dataset_dir = RUN_ROOT / 'datasets' / ('task2_A012_' + DATASET_SHA[:16])
    dataset_pointer = json.loads((RUN_ROOT / 'manifests/active_task2_dataset.json').read_bytes())
    assert dataset_pointer['dataset_identity'] == DATASET_SHA and Path(dataset_pointer['dataset_dir']) == dataset_dir
    assert dataset_pointer['freeze_receipt_sha256'] == FREEZE_SHA
    assert file_sha(dataset_dir / 'freeze_receipt.json') == FREEZE_SHA
    assert json.loads((dataset_dir / 'configs/model_revisions.json').read_bytes()) == LOCKS
    pointer = json.loads((RUN_ROOT / 'manifests/active_task2_input_shards.json').read_bytes())
    assert pointer['dataset_identity'] == DATASET_SHA and pointer['recipe_sha256'] == SHARDS_RECIPE
    manifest_path = dataset_dir / 'inputs/image_shards' / SHARDS_RECIPE[:16] / 'input_shards_complete.json'
    assert Path(pointer['manifest_path']) == manifest_path and pointer['manifest_sha256'] == SHARDS_SHA
    assert file_sha(manifest_path) == SHARDS_SHA
    shard_manifest = json.loads(manifest_path.read_bytes())
    assert shard_manifest['status'] == 'passed' and shard_manifest['image_count'] == 9260
    assert shard_manifest['shard_count'] == len(shard_manifest['shards']) == 37
    LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
    assert shutil.disk_usage(LOCAL_ROOT).free >= 12 * 1024**3, 'At least 12 GiB of local free space is required.'
    print('Frozen dataset, model revisions and completed 37-archive manifest verified. Accelerator None.', flush=True)
    print('Downloading original pinned model files and checking their published source hashes.', flush=True)
    import huggingface_hub
    from huggingface_hub import HfApi, hf_hub_download, constants
    constants.HF_HUB_DOWNLOAD_TIMEOUT = 60
    constants.HF_HUB_ETAG_TIMEOUT = 30
    api = HfApi(token=False)

    def fetch(lock, name, retry):
        return hf_hub_download(repo_id=lock['repo_id'], revision=lock['revision'], filename=name,
            cache_dir=str(LOCAL_ROOT / 'hf_cache'), token=False, force_download=retry > 0)

    for key, lock in LOCKS.items():
        store = ModelAssetStore(RUN_ROOT, LOCAL_ROOT, key, lock, fetch)
        plan_path = store.durable_dir / 'source_plan.json'
        if plan_path.exists():
            plan = json.loads(plan_path.read_bytes())
            assert plan['model_key'] == key and all(plan[k] == v for k, v in lock.items())
        else:
            for retry in range(3):
                try:
                    info = api.model_info(repo_id=lock['repo_id'], revision=lock['revision'],
                        files_metadata=True, timeout=30)
                    plan = make_plan(key, lock, info)
                    break
                except Exception:
                    if retry == 2:
                        raise
                    time.sleep(retry + 1)
            immutable_bytes(plan_path, encoded(plan))
        records, counts = [], Counter()
        for number, source_record in enumerate(plan['files'], 1):
            record, activity = store.prepare_file(plan, source_record)
            records.append(record)
            counts[activity] += 1
            print(f'{key} verified {number}/{len(plan["files"])} | {record["file_name"]} | {activity}', flush=True)
        structure = verify_model_structure(key, store.local_dir, records)
        print(f'{key} checkpoint headers and dimensions verified. Saving and reading back its archive.', flush=True)
        receipt, activity = store.pack(plan, records, structure)
        receipt_path = store.durable_dir / 'model_complete.json'
        models[key] = {'receipt_path': str(receipt_path), 'receipt_sha256': file_sha(receipt_path), **receipt}
        activities[key] = {'files': dict(counts), 'archive': activity}
        print(f'{key} original model archive verified and saved.', flush=True)
    manifest = {'schema': 'atml-pa3-task2-model-assets-complete-v1', 'status': 'passed',
        'dataset_identity': DATASET_SHA, 'freeze_receipt_sha256': FREEZE_SHA,
        'input_shards_manifest_sha256': SHARDS_SHA, 'model_revisions': LOCKS, 'models': models}
    content = encoded(manifest)
    digest = hashlib.sha256(content).hexdigest()
    manifest_path = dataset_dir / 'inputs/model_assets' / ('models_complete_' + digest[:16] + '.json')
    immutable_bytes(manifest_path, content)
    immutable_bytes(RUN_ROOT / 'manifests/active_task2_model_assets.json', encoded({
        'schema': 'atml-pa3-task2-active-model-assets-v1', 'dataset_identity': DATASET_SHA,
        'manifest_path': str(manifest_path), 'manifest_sha256': digest, 'model_revisions': LOCKS}))
    environment = {'python': platform.python_version()}
    for name in ('huggingface-hub', 'transformers', 'torch', 'Pillow', 'numpy', 'safetensors'):
        try:
            environment[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            environment[name] = None
    result = {'block': '10', 'status': 'passed', 'dataset_identity': DATASET_SHA,
        'models_prepared': list(models), 'model_revisions': LOCKS, 'run_activities': activities,
        'model_manifest_path': str(manifest_path), 'model_manifest_sha256': digest,
        'original_model_file_bytes': sum(r['bytes'] for m in models.values() for r in m['files']),
        'models': {key: {'archive_sha256': row['archive_sha256'], 'archive_bytes': row['archive_bytes'],
            'file_count': len(row['files']), 'structure': row['structure']} for key, row in models.items()},
        'elapsed_seconds': round(time.monotonic()-started, 3), 'environment': environment,
        'gpu_used': False, 'model_weights_downloaded': True, 'model_weights_loaded': False,
        'feature_cache_created': False, 'training_started': False}
    immutable_bytes(RUN_ROOT / f'logs/block_10_{attempt}.json', encoded(result))
    print('BLOCK 10 PASSED', flush=True)
    print(json.dumps(result, indent=2), flush=True)
    print('Return the complete output before switching to GPU for feature caching.', flush=True)
except BaseException:
    failure = {'block': '10', 'status': 'failed', 'traceback': traceback.format_exc(),
        'dataset_identity': DATASET_SHA, 'model_package_sha256': package_sha,
        'completed_models': list(models), 'gpu_used': False, 'training_started': False,
        'resume': 'Rerun the same package to reuse source-verified files and completed archives.'}
    immutable_bytes(RUN_ROOT / f'logs/block_10_failed_{attempt}.json', encoded(failure))
    raise
