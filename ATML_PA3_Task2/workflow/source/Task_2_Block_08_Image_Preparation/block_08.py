"""Block 8. Prepare all frozen-dataset original images on CPU with durable receipts."""
import os
import sys
import io
import json
import hashlib
import importlib.util
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
        raise RuntimeError('Reconnect with accelerator None for original-image preparation.')
from google.colab import drive
if not Path('/content/drive/MyDrive').exists():
    drive.mount('/content/drive')

RUN_ROOT = Path('/content/drive/MyDrive/ATML_PA3/task2/run_20261031_150949_1cff148a')
LOCAL_ROOT = Path('/content/atml_pa3_task2/run_20261031_150949_1cff148a')
EXPECTED_DATASET = '82ad5006bd1d68ee54e639e50e7d835d8f6d97673537c38dab8d8eebdd8ac1b2'
EXPECTED_FREEZE = 'fe7c9d4efa5d92d5b809eadb3695adcdd19f2421c6d43a2cb1593a7b86c3245b'
EXPECTED_PREFLIGHT = 'd5d2ad21658d452673d6a549f725e2f78949b97b3890ee9d93e170d6eb50ec95'
package_bytes = globals().get('task2_image_package_bytes')
if not isinstance(package_bytes, bytes):
    raise RuntimeError('Run the supplied Block 8 package loader.')
package_sha = hashlib.sha256(package_bytes).hexdigest()
work = LOCAL_ROOT / 'image_preparation_code' / package_sha[:16]
with zipfile.ZipFile(io.BytesIO(package_bytes)) as archive:
    assert archive.testzip() is None
    package_manifest = json.loads(archive.read('package_manifest.json'))
    assert set(archive.namelist()) == {r['path'] for r in package_manifest['files']} | {'package_manifest.json'}
    for record in package_manifest['files']:
        path = Path(record['path'])
        assert not path.is_absolute() and '..' not in path.parts
        content = archive.read(record['path'])
        assert len(content) == record['bytes'] and hashlib.sha256(content).hexdigest() == record['sha256']
        destination = work / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            assert destination.read_bytes() == content
        else:
            destination.write_bytes(content)

spec = importlib.util.spec_from_file_location('task2_image_preparation', work / 'image_preparation.py')
image_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(image_module)
hash_file = image_module.file_sha
save_bytes = image_module.immutable_bytes
encode = image_module.encoded
attempt = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:8]
preparer, executor = None, None
activities, completed = Counter(), {}
current_job = None
started = time.monotonic()
try:
    pointer = json.loads((RUN_ROOT / 'manifests/active_task2_dataset.json').read_bytes())
    dataset_dir = RUN_ROOT / 'datasets' / ('task2_A012_' + EXPECTED_DATASET[:16])
    assert pointer['dataset_identity'] == EXPECTED_DATASET and Path(pointer['dataset_dir']) == dataset_dir
    assert pointer['freeze_receipt_sha256'] == EXPECTED_FREEZE and pointer['preflight_package_sha256'] == EXPECTED_PREFLIGHT
    freeze_path = dataset_dir / 'freeze_receipt.json'
    assert hash_file(freeze_path) == EXPECTED_FREEZE
    freeze = json.loads(freeze_path.read_bytes())
    assert freeze['dataset_frozen'] is True and freeze['status'] == 'passed'
    assert freeze['dataset_identity'] == EXPECTED_DATASET
    assert hash_file(dataset_dir / 'dataset_manifest.json') == EXPECTED_DATASET
    dataset_manifest = json.loads((dataset_dir / 'dataset_manifest.json').read_bytes())
    for entry in dataset_manifest['files']:
        path = dataset_dir / entry['path']
        assert path.stat().st_size == entry['bytes'] and hash_file(path) == entry['sha256']
    jobs_path = dataset_dir / 'inputs/image_jobs.jsonl'
    assert hash_file(jobs_path) == freeze['image_jobs_sha256']
    jobs = [json.loads(line) for line in jobs_path.read_text().splitlines()]
    assert len(jobs) == len({(j['source_split'], j['image_id']) for j in jobs}) == freeze['image_jobs'] == 9260
    assert Counter(j['source_split'] for j in jobs) == {'train2014': 8733, 'val2014': 527}
    manual = json.loads((dataset_dir / 'provenance/manual_tv_image_receipt.json').read_bytes())
    seed_path = dataset_dir / 'provenance/manual_tv_source.jpg'
    assert hash_file(seed_path) == manual['sha256']
    seeds = {('train2014', 153620): {'path': seed_path, 'sha256': manual['sha256']}}
    preparer = image_module.ImagePreparer(RUN_ROOT, LOCAL_ROOT, dataset_dir, EXPECTED_DATASET, seeds)
    print('Frozen dataset and 9260-image input list verified. Accelerator None.', flush=True)
    print('Preparing unchanged original JPEGs with four workers. Verified images are reused on rerun.', flush=True)
    executor = ThreadPoolExecutor(max_workers=4)
    futures = {executor.submit(preparer.prepare, job): job for job in jobs}
    for future in as_completed(futures):
        current_job = futures[future]
        record, activity = future.result()
        completed[(record['source_split'], record['image_id'])] = record
        activities[activity] += 1
        if len(completed) % 100 == 0 or len(completed) == len(jobs):
            print(f"Verified and saved {len(completed)}/{len(jobs)} | downloaded {activities['downloaded']} | reused {activities['reused']} | elapsed {time.monotonic()-started:.1f}s", flush=True)
    executor.shutdown(wait=True)
    executor = None
    ordered = []
    for job in jobs:
        key = (job['source_split'], job['image_id'])
        record = preparer.load_receipt(job)
        assert record == completed[key]
        ordered.append(record)
    index_bytes = ''.join(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n' for record in ordered).encode()
    index_sha = hashlib.sha256(index_bytes).hexdigest()
    index_path = dataset_dir / 'inputs' / ('original_images_' + index_sha[:16] + '.jsonl')
    save_bytes(index_path, index_bytes)
    assert hash_file(index_path) == index_sha
    modes = Counter(r['decoded_metadata']['mode'] for r in ordered)
    orientation = Counter(str(r['decoded_metadata']['exif_orientation']) for r in ordered)
    completion = {'schema': 'atml-pa3-task2-images-complete-v1', 'status': 'passed',
        'dataset_identity': EXPECTED_DATASET, 'freeze_receipt_sha256': EXPECTED_FREEZE,
        'image_jobs_sha256': freeze['image_jobs_sha256'], 'image_index_sha256': index_sha,
        'image_index_path': str(index_path), 'image_count': len(ordered),
        'source_splits': dict(Counter(r['source_split'] for r in ordered)),
        'original_image_bytes': sum(r['bytes'] for r in ordered), 'decoded_modes': dict(modes),
        'exif_orientations': dict(orientation), 'image_preparation_package_sha256': package_sha}
    completion_path = dataset_dir / 'inputs/images_complete.json'
    save_bytes(completion_path, encode(completion))
    image_pointer = {'schema': 'atml-pa3-task2-active-images-v1', 'dataset_identity': EXPECTED_DATASET,
        'completion_path': str(completion_path), 'completion_sha256': hash_file(completion_path),
        'image_index_path': str(index_path), 'image_index_sha256': index_sha}
    save_bytes(RUN_ROOT / 'manifests/active_task2_images.json', encode(image_pointer))
    result = {'block': '08', 'status': 'passed', 'original_images_complete': True,
        'dataset_identity': EXPECTED_DATASET, 'images_verified': len(ordered),
        'run_activities': dict(activities), 'original_image_bytes': completion['original_image_bytes'],
        'decoded_modes': dict(modes), 'exif_orientations': dict(orientation),
        'elapsed_seconds': round(time.monotonic()-started, 3),
        'image_index_path': str(index_path), 'image_index_sha256': index_sha,
        'completion_sha256': hash_file(completion_path), 'gpu_used': False,
        'feature_cache_created': False, 'training_started': False}
    save_bytes(RUN_ROOT / f'logs/block_08_{attempt}.json', encode(result))
    print('BLOCK 8 PASSED', flush=True)
    print(json.dumps(result, indent=2), flush=True)
    print('Return the complete output before the next block.', flush=True)
except BaseException:
    if preparer is not None:
        preparer.stop.set()
    if executor is not None:
        executor.shutdown(wait=True, cancel_futures=True)
    failure = {'block': '08', 'status': 'failed', 'traceback': traceback.format_exc(),
        'dataset_identity': EXPECTED_DATASET, 'image_preparation_package_sha256': package_sha,
        'verified_results_returned': len(completed), 'run_activities': dict(activities),
        'current_job': current_job, 'gpu_used': False, 'training_started': False,
        'resume': 'Rerun this same block to reuse verified original images and durable receipts.'}
    save_bytes(RUN_ROOT / f'logs/block_08_failed_{attempt}.json', encode(failure))
    raise
