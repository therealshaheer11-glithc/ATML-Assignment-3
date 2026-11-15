# Block 6B: original audit images and annotation-box index. CPU only.
# No label judgments, image resizing, model weights, or GPU work.
import os, json, uuid, shutil, traceback, gc, time, urllib.request
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

needed = ('ROOT', 'LOCAL', 'save_record', 'file_sha', 'copy_checked')
if any(n not in globals() for n in needed):
    raise RuntimeError('Notebook state missing. Stop; do not rerun Block 1.')
old_path = ROOT / 'logs/block_06a_20261031_163930_3cb996af.json'
assert file_sha(old_path) == '44a121f200cfd59ec38771010438328e8aaa35482989c5843ddd54c42ddbf13a'
old6 = json.loads(old_path.read_text())
if os.environ.get('COLAB_GPU', '0') not in ('', '0'):
    print('CPU image preparation: requesting GPU release.', flush=True)
    try:
        runtime.unassign()
    finally:
        raise RuntimeError('Reconnect with accelerator None.')
stamp = datetime.now(ZoneInfo('Asia/Karachi')).strftime('%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:8]
started = time.monotonic()
report6b = {'attempt': stamp, 'status': 'started', 'gpu_used': False,
            'visual_reviews_imported': 0, 'complete_dataset_frozen': False,
            'model_weights_loaded': False, 'images_verified': 0}

# Inspection decodes the JPEG but never rewrites its pixels.
def inspect_original(path, expected):
    with Image.open(path) as image:
        image.load()
        if image.format != 'JPEG' or image.size != (expected['width'], expected['height']):
            raise IOError(f'Image format/dimensions mismatch: {path}')
        return {'width': image.width, 'height': image.height, 'mode': image.mode,
                'exif_orientation': int(image.getexif().get(274, 1))}

def ensure_audit_image(job):
    local = LOCAL / 'images' / job['source_split'] / job['file_name']
    durable = ROOT / 'raw_images' / job['source_split'] / job['file_name']
    receipt_rel = f'manifests/images/{job["source_split"]}/{job["image_id"]:012d}.json'
    receipt_path = ROOT / receipt_rel
    local.parent.mkdir(parents=True, exist_ok=True)
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt['url'] != job['url'] or receipt['image_id'] != job['image_id']:
            raise IOError('Conflicting saved image identity.')
        if durable.exists():
            if file_sha(durable) != receipt['sha256']:
                raise IOError(f'Drive image differs from its receipt: {durable}')
            copy_checked(durable, local)
        if not local.exists() or file_sha(local) != receipt['sha256']:
            raise IOError('Recorded image is missing/changed; preserve it and report the error.')
        assert inspect_original(local, job) == receipt['decoded_metadata']
        copy_checked(local, durable)
        return receipt, 'reused'

    temporary = local.with_name(local.name + '.download.part')
    last_error = None
    for retry in range(3):
        try:
            request = urllib.request.Request(job['url'], headers={'User-Agent': 'ATML-PA3-CPU-data-preparation'})
            with urllib.request.urlopen(request, timeout=30) as response:
                if not response.geturl().startswith('https://'):
                    raise IOError('Unexpected non-HTTPS redirect.')
                expected_bytes = response.headers.get('Content-Length')
                with temporary.open('wb') as out:
                    shutil.copyfileobj(response, out)
                if expected_bytes is not None and temporary.stat().st_size != int(expected_bytes):
                    raise IOError('Incomplete image response.')
            decoded = inspect_original(temporary, job)
            last_error = None
            break
        except Exception as exc:
            last_error = exc
            if retry < 2:
                time.sleep(retry + 1)
    if last_error is not None:
        raise last_error
    digest = file_sha(temporary)
    if local.exists():
        if file_sha(local) != digest:
            raise IOError(f'Existing unrecorded image differs; preserved: {local}')
        temporary.unlink()
    else:
        temporary.rename(local)
    copy_checked(local, durable)
    receipt = {'source_split': job['source_split'], 'image_id': job['image_id'],
               'file_name': job['file_name'], 'url': job['url'],
               'original_coco_url': job.get('original_coco_url'),
               'sha256': digest, 'bytes': local.stat().st_size,
               'decoded_metadata': decoded,
               'hash_provenance': 'Observed original-image SHA256; HTTPS COCO object download'}
    save_record(receipt_rel, receipt)
    return receipt, 'downloaded'

try:
    ids_path = ROOT / 'audits/annotation_audit_image_ids.json'
    assert file_sha(ids_path) == old6['audit_image_ids_sha256']
    ids = json.loads(ids_path.read_text())
    assert len(ids['train2014']) == 314 and len(ids['val2014']) == 174
    pairs = []
    for name in ('grounding_training_audit.jsonl', 'grounding_dev.jsonl', 'grounding_test.jsonl'):
        item = next(r for r in old6['outputs'] if r['path'].endswith('/' + name))
        path = Path(item['path'])
        assert file_sha(path) == item['sha256']
        with path.open() as stream:
            pairs.extend(json.loads(line) for line in stream)
    assert len(pairs) == 246
    requested = {(r['source_split'], int(ex['image_id']), int(r['category_id']))
                 for r in pairs for ex in r['examples']}
    coco = json.loads((ROOT / 'manifests/annotation_downloads/coco2014.json').read_text())
    hashes = {r['filename']: r['sha256'] for r in coco['json_files']}
    jobs, box_records, image_metadata, licenses = [], {}, {}, {}
    for split in ('train2014', 'val2014'):
        print('Preparing audit image metadata:', split, flush=True)
        path = LOCAL / 'annotations' / f'instances_{split}.json'
        assert file_sha(path) == hashes[path.name]
        with path.open() as stream:
            payload = json.load(stream)
        wanted = set(ids[split])
        chosen = {int(im['id']): im for im in payload['images'] if int(im['id']) in wanted}
        assert set(chosen) == wanted
        licenses[split] = payload.get('licenses', [])
        boxes = defaultdict(list)
        for annotation in payload['annotations']:
            key = (split, int(annotation['image_id']), int(annotation['category_id']))
            if key in requested:
                boxes[key].append({'annotation_id': int(annotation['id']),
                                   'bbox': annotation['bbox'], 'iscrowd': bool(annotation.get('iscrowd', 0))})
        for key in sorted(k for k in requested if k[0] == split):
            box_records['|'.join(map(str, key))] = sorted(boxes[key], key=lambda a: a['annotation_id'])
        for image_id, im in sorted(chosen.items()):
            filename = im['file_name']
            assert Path(filename).name == filename and filename.endswith(f'{image_id:012d}.jpg')
            job = {'source_split': split, 'image_id': image_id, 'file_name': filename,
                   'width': int(im['width']), 'height': int(im['height']),
                   'original_coco_url': im.get('coco_url'),
                   'url': f'https://s3.amazonaws.com/images.cocodataset.org/{split}/{filename}'}
            jobs.append(job)
            image_metadata[f'{split}|{image_id}'] = {
                **job, 'flickr_url': im.get('flickr_url'), 'license_id': im.get('license')}
        del payload, chosen, boxes
        gc.collect()

    # This checks consistency with source annotations, not visual correctness.
    for row in pairs:
        for ex in row['examples']:
            key = f'{row["source_split"]}|{ex["image_id"]}|{row["category_id"]}'
            source_boxes = box_records[key]
            if row['question_type'] == 'presence':
                assert bool(source_boxes) == (ex['target'] == 'yes')
            else:
                assert not any(a['iscrowd'] for a in source_boxes)
                assert len(source_boxes) == int(ex['target'])
    report6b['annotation_consistency_check_passed'] = True
    report6b['audit_box_index_sha256'] = save_record('audits/annotation_box_index.json', {
        'boxes_by_split_image_category': box_records, 'images': image_metadata, 'licenses': licenses,
        'coordinate_system': 'Original encoded JPEG dimensions; boxes are COCO x,y,width,height',
        'visual_audit_complete': False})

    results, activities = [], Counter()
    # Check one real image from each split before requesting the remaining images.
    first = [next(j for j in jobs if j['source_split'] == split) for split in ('train2014', 'val2014')]
    first_keys = {(j['source_split'], j['image_id']) for j in first}
    for job in first:
        print('Checking image source:', job['url'], flush=True)
        receipt, activity = ensure_audit_image(job)
        results.append(receipt)
        activities[activity] += 1
        report6b['images_verified'] = len(results)
    remaining = [j for j in jobs if (j['source_split'], j['image_id']) not in first_keys]
    # Small batches limit outstanding requests; completed images persist individually.
    for start in range(0, len(remaining), 32):
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_map = {executor.submit(ensure_audit_image, j): j for j in remaining[start:start + 32]}
            for future in as_completed(future_map):
                receipt, activity = future.result()
                results.append(receipt)
                activities[activity] += 1
        report6b['images_verified'] = len(results)
        save_record(f'logs/image_download_progress/{stamp}_{len(results):04d}.json', {
            'verified_images': len(results), 'activities': dict(activities),
            'elapsed_seconds': round(time.monotonic() - started, 2)})
        print(f'Verified and saved {len(results)}/488 original audit images', flush=True)
    results.sort(key=lambda r: (r['source_split'], r['image_id']))
    assert len(results) == len({(r['source_split'], r['image_id']) for r in results}) == 488
    report6b['image_manifest_sha256'] = save_record('audits/annotation_audit_image_manifest.json', results)
    report6b.update({'status': 'passed', 'images_verified': len(results),
                     'image_counts': dict(Counter(r['source_split'] for r in results)),
                     'total_original_image_bytes': sum(r['bytes'] for r in results),
                     'activities_this_attempt': dict(activities),
                     'nonstandard_exif_orientation_images': [f'{r["source_split"]}|{r["image_id"]}'
                         for r in results if r['decoded_metadata']['exif_orientation'] != 1],
                     'image_mode_counts': dict(Counter(r['decoded_metadata']['mode'] for r in results)),
                     'elapsed_seconds': round(time.monotonic() - started, 2),
                     'local_disk_free_GiB': round(shutil.disk_usage('/content').free / 2**30, 2)})
except Exception:
    report6b['status'] = 'failed'
    report6b['traceback'] = traceback.format_exc()
    save_record(f'logs/block_06b_{stamp}.json', report6b)
    print('BLOCK 6B FAILED. Completed image files/receipts retained; no image IDs replaced.', flush=True)
    raise

log_path = f'logs/block_06b_{stamp}.json'
digest = save_record(log_path, report6b)
print('\nBLOCK 6B COMPLETE — original audit images and box metadata verified; visual review not performed.')
print(json.dumps(report6b, indent=2))
print('Saved log:', ROOT / log_path)
print('SHA256:', digest)
print('Keep accelerator None. Paste the complete output; review viewer comes next.')
