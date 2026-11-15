"""Block 9. Pack verified original images for efficient GPU feature-cache reads. CPU only."""
import os
import sys
import io
import json
import hashlib
import time
import traceback
import uuid
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

if os.environ.get('COLAB_GPU', '0') not in ('', '0'):
    from google.colab import runtime
    try:
        runtime.unassign()
    finally:
        raise RuntimeError('Reconnect with accelerator None for original-image packing.')
from google.colab import drive
if not Path('/content/drive/MyDrive').exists():
    drive.mount('/content/drive')

RUN_ROOT = Path('/content/drive/MyDrive/ATML_PA3/task2/run_20261031_150949_1cff148a')
LOCAL_ROOT = Path('/content/atml_pa3_task2/run_20261031_150949_1cff148a')
DATASET_SHA = '82ad5006bd1d68ee54e639e50e7d835d8f6d97673537c38dab8d8eebdd8ac1b2'
FREEZE_SHA = 'fe7c9d4efa5d92d5b809eadb3695adcdd19f2421c6d43a2cb1593a7b86c3245b'
IMAGES_SHA = '076a35e8cd6e3e47346059fa6899af803a919b95f7c98bb4f5da0e097cb30697'
INDEX_SHA = '55aaabf72e8635aa3897cf54f06e69c024e13a2cff33aac8d7fb47c48a6d8cdd'
package_bytes = globals().get('task2_staging_package_bytes')
if not isinstance(package_bytes, bytes):
    raise RuntimeError('Run the supplied Block 9 package loader.')
package_sha = hashlib.sha256(package_bytes).hexdigest()
work = LOCAL_ROOT / 'staging_code' / package_sha[:16]
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
from task2.staging import InputArchiveStore

attempt = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:8]
started = time.monotonic()
executor = None
receipts, activities = {}, Counter()
try:
    dataset_dir = RUN_ROOT / 'datasets' / ('task2_A012_' + DATASET_SHA[:16])
    pointer = json.loads((RUN_ROOT / 'manifests/active_task2_dataset.json').read_bytes())
    assert pointer['dataset_identity'] == DATASET_SHA and Path(pointer['dataset_dir']) == dataset_dir
    assert pointer['freeze_receipt_sha256'] == FREEZE_SHA
    assert file_sha(dataset_dir / 'freeze_receipt.json') == FREEZE_SHA
    image_pointer = json.loads((RUN_ROOT / 'manifests/active_task2_images.json').read_bytes())
    assert image_pointer['dataset_identity'] == DATASET_SHA and image_pointer['completion_sha256'] == IMAGES_SHA
    assert image_pointer['image_index_sha256'] == INDEX_SHA
    complete_path = dataset_dir / 'inputs/images_complete.json'
    assert file_sha(complete_path) == IMAGES_SHA
    complete = json.loads(complete_path.read_bytes())
    assert complete['status'] == 'passed' and complete['image_count'] == 9260
    index_path = dataset_dir / 'inputs' / ('original_images_' + INDEX_SHA[:16] + '.jsonl')
    assert Path(image_pointer['image_index_path']) == index_path and file_sha(index_path) == INDEX_SHA
    images = [json.loads(line) for line in index_path.read_text().splitlines()]
    assert len(images) == len({(r['source_split'], r['image_id']) for r in images}) == 9260
    assert all(r['dataset_identity'] == DATASET_SHA for r in images)
    assert Counter(r['source_split'] for r in images) == {'train2014': 8733, 'val2014': 527}
    recipe = {'schema': 'atml-pa3-task2-original-input-shards-v1', 'dataset_identity': DATASET_SHA,
        'freeze_receipt_sha256': FREEZE_SHA, 'image_completion_sha256': IMAGES_SHA,
        'image_index_sha256': INDEX_SHA, 'images_per_shard': 256,
        'ordering': 'Exact completed image-index order', 'archive_format': 'Deterministic uncompressed USTAR',
        'image_encoding': 'Original JPEG bytes unchanged', 'image_count': 9260}
    store = InputArchiveStore(RUN_ROOT, LOCAL_ROOT, dataset_dir, recipe)
    immutable_bytes(store.durable_dir / 'recipe.json', encoded(recipe))
    groups = [images[start:start + 256] for start in range(0, len(images), 256)]
    assert len(groups) == 37
    print('Verified completed original-image index. Packing 9260 unchanged JPEGs into 37 archives.', flush=True)
    executor = ThreadPoolExecutor(max_workers=4)
    futures = {executor.submit(store.pack, index, rows): index for index, rows in enumerate(groups)}
    for future in as_completed(futures):
        index = futures[future]
        receipt, activity = future.result()
        receipts[index] = receipt
        activities[activity] += 1
        print(f'Archives verified and saved: {len(receipts)}/37 | elapsed {time.monotonic()-started:.1f}s', flush=True)
    executor.shutdown(wait=True)
    executor = None
    ordered = [receipts[index] for index in range(len(groups))]
    assert [r for receipt in ordered for r in receipt['images']] == images
    manifest = {'schema': 'atml-pa3-task2-original-input-shards-complete-v1', 'status': 'passed',
        'dataset_identity': DATASET_SHA, 'recipe': recipe, 'recipe_sha256': store.recipe_sha,
        'image_count': len(images), 'shard_count': len(ordered), 'shards': ordered}
    manifest_content = encoded(manifest)
    manifest_path = store.durable_dir / 'input_shards_complete.json'
    immutable_bytes(manifest_path, manifest_content)
    manifest_sha = hashlib.sha256(manifest_content).hexdigest()
    pointer = {'schema': 'atml-pa3-task2-active-input-shards-v1', 'dataset_identity': DATASET_SHA,
        'recipe_sha256': store.recipe_sha, 'manifest_path': str(manifest_path), 'manifest_sha256': manifest_sha,
        'image_count': 9260, 'shard_count': 37}
    immutable_bytes(RUN_ROOT / 'manifests/active_task2_input_shards.json', encoded(pointer))
    result = {'block': '09', 'status': 'passed', 'dataset_identity': DATASET_SHA,
        'image_count': 9260, 'shard_count': 37, 'images_per_shard': 256,
        'run_activities': dict(activities), 'recipe_sha256': store.recipe_sha,
        'input_shards_manifest_path': str(manifest_path), 'input_shards_manifest_sha256': manifest_sha,
        'archive_bytes': sum(r['tar_bytes'] for r in ordered),
        'elapsed_seconds': round(time.monotonic()-started, 3),
        'gpu_used': False, 'feature_cache_created': False, 'training_started': False}
    immutable_bytes(RUN_ROOT / f'logs/block_09_{attempt}.json', encoded(result))
    print('BLOCK 9 PASSED', flush=True)
    print(json.dumps(result, indent=2), flush=True)
    print('Return the complete output. The next block will configure and check GPU feature extraction.', flush=True)
except BaseException:
    if executor is not None:
        executor.shutdown(wait=True, cancel_futures=True)
    failure = {'block': '09', 'status': 'failed', 'traceback': traceback.format_exc(),
        'dataset_identity': DATASET_SHA, 'staging_package_sha256': package_sha,
        'completed_archive_results': len(receipts), 'gpu_used': False, 'training_started': False,
        'resume': 'Rerun this same package to reuse verified completed archives.'}
    immutable_bytes(RUN_ROOT / f'logs/block_09_failed_{attempt}.json', encoded(failure))
    raise
