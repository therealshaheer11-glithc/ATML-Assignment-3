# Block 3B: finish annotation downloads and inventory after the COCO repair.
# Run in the SAME notebook; keep accelerator None.
import os, json, uuid, hashlib, shutil, traceback, subprocess
import gc, time, urllib.request, zipfile
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import Counter

needed = ('ROOT', 'LOCAL', 'save_record', 'file_sha', 'copy_checked')
if any(name not in globals() for name in needed):
    raise RuntimeError('Notebook state is missing. Stop; do not rerun Block 1.')
repair_log = ROOT / 'logs/block_03_repair_20261031_152911_9193bda5.json'
assert file_sha(repair_log) == '2bed1d7779467a12748a872cbb90cb47d875893c1265c16199ab1d9ea71f8104'
source = json.loads((ROOT / 'configs/coco2014_source.json').read_text())
assert source['url'] == 'https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2014.zip'
assert source['expected_sha256'] == '031296bbc80c45a1d1f76bf9a90ead27e94e99ec629208449507a4917a3bf009'

accelerator = os.environ.get('COLAB_GPU', '0') not in ('', '0')
if shutil.which('nvidia-smi'):
    probe = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                           capture_output=True, text=True, timeout=10)
    accelerator |= probe.returncode == 0 and bool(probe.stdout.strip())
if accelerator or os.environ.get('COLAB_TPU_ADDR'):
    print('CPU preparation: requesting accelerator release.', flush=True)
    try:
        runtime.unassign()
    finally:
        raise RuntimeError('Reconnect with accelerator None.')

stamp = datetime.now(ZoneInfo('Asia/Karachi')).strftime('%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:8]
report3b = {'attempt': stamp, 'status': 'started', 'gpu_used': False,
            'images_downloaded': False, 'model_weights_loaded': False,
            'examples_selected': False, 'archives': [], 'json_inventory': []}
archives = LOCAL / 'annotation_archives'
json_dir = LOCAL / 'annotations'
archives.mkdir(parents=True, exist_ok=True)
json_dir.mkdir(parents=True, exist_ok=True)
base = 'https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/'
plan3b = [
    ('coco2014', source['url']),
    ('vqa_train_annotations', base + 'v2_Annotations_Train_mscoco.zip'),
    ('vqa_val_annotations', base + 'v2_Annotations_Val_mscoco.zip'),
    ('vqa_train_questions', base + 'v2_Questions_Train_mscoco.zip'),
    ('vqa_val_questions', base + 'v2_Questions_Val_mscoco.zip'),
    ('vqa_train_pairs', base + 'v2_Complementary_Pairs_Train_mscoco.zip'),
    ('vqa_val_pairs', base + 'v2_Complementary_Pairs_Val_mscoco.zip'),
]
coco_needed = {'captions_train2014.json', 'captions_val2014.json',
               'instances_train2014.json', 'instances_val2014.json'}

try:
    for key, url in plan3b:
        local_zip = archives / f'{key}.zip'
        durable_zip = ROOT / 'raw_annotation_archives' / f'{key}.zip'
        receipt_rel = f'manifests/annotation_downloads/{key}.json'
        receipt_path = ROOT / receipt_rel
        if durable_zip.exists():
            copy_checked(durable_zip, local_zip)
        if not local_zip.exists():
            partial = local_zip.with_suffix('.zip.part')
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {'Range': f'bytes={offset}-'} if offset else {}
            print(f'Downloading {key}; resume offset {offset:,} bytes...', flush=True)
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                if not response.geturl().startswith('https://'):
                    raise RuntimeError('Unexpected non-HTTPS redirect.')
                resumed = offset > 0 and response.status == 206
                if resumed and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                    raise IOError('Unexpected download resume position.')
                received, last_update = 0, time.monotonic()
                expected = response.headers.get('Content-Length')
                with partial.open('ab' if resumed else 'wb') as out:
                    while True:
                        chunk = response.read(4 * 1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
                        received += len(chunk)
                        if time.monotonic() - last_update > 15:
                            print(f'  {partial.stat().st_size / 2**20:.1f} MiB', flush=True)
                            last_update = time.monotonic()
                if expected is not None and received != int(expected):
                    raise IOError('Incomplete response; partial retained.')
            with zipfile.ZipFile(partial) as z:
                if z.testzip() is not None:
                    raise IOError(f'Archive CRC failed: {key}')
            partial.rename(local_zip)

        digest = file_sha(local_zip)
        if key == 'coco2014':
            assert local_zip.stat().st_size == source['expected_bytes']
            assert digest == source['expected_sha256']
        if receipt_path.exists():
            old = json.loads(receipt_path.read_text())
            if old['url'] != url or old['sha256'] != digest:
                raise IOError(f'Archive differs from its saved receipt: {key}')
        extracted = []
        with zipfile.ZipFile(local_zip) as z:
            if z.testzip() is not None:
                raise IOError(f'Archive CRC failed: {key}')
            members = [m for m in z.infolist() if not m.is_dir()
                       and m.filename.endswith('.json')
                       and (key != 'coco2014' or Path(m.filename).name in coco_needed)]
            if len(members) != (4 if key == 'coco2014' else 1):
                raise RuntimeError(f'Unexpected JSON members: {key}: {[m.filename for m in members]}')
            for member in members:
                dest = json_dir / Path(member.filename).name
                temp = dest.with_name(dest.name + '.extracting')
                with z.open(member) as src, temp.open('wb') as out:
                    shutil.copyfileobj(src, out)
                if dest.exists():
                    if file_sha(dest) != file_sha(temp):
                        raise IOError(f'Existing JSON differs; preserved: {dest.name}')
                    temp.unlink()
                else:
                    temp.rename(dest)
                extracted.append({'filename': dest.name, 'bytes': dest.stat().st_size,
                                  'sha256': file_sha(dest)})
        copy_checked(local_zip, durable_zip)
        receipt = {'key': key, 'url': url, 'bytes': local_zip.stat().st_size,
                   'sha256': digest, 'zip_crc_passed': True, 'json_files': extracted,
                   'hash_provenance': ('Published TensorFlow Datasets SHA256 verified'
                                       if key == 'coco2014' else
                                       'Observed download SHA256; not a published reference checksum')}
        save_record(receipt_rel, receipt)
        report3b['archives'].append(receipt)
        print('Verified, extracted, and saved:', key, flush=True)

    files = sorted(json_dir.glob('*.json'))
    if len(files) != 10:
        raise RuntimeError(f'Expected 10 annotation JSON files, found {len(files)}.')
    for path in files:
        print('Inspecting', path.name, flush=True)
        with path.open() as f:
            data = json.load(f)
        split = 'train2014' if 'train2014' in path.name else 'val2014'
        row = {'filename': path.name, 'split': split, 'bytes': path.stat().st_size}
        if isinstance(data, list):
            assert all(isinstance(p, list) and len(p) == 2 and p[0] != p[1] for p in data)
            row.update({'kind': 'complementary_pairs', 'pairs': len(data)})
        elif 'images' in data:
            row.update({'kind': 'coco_annotations', 'images': len(data['images']),
                        'annotations': len(data['annotations'])})
            assert row['images'] == (82783 if split == 'train2014' else 40504)
            if 'instances_' in path.name:
                row['categories'] = len(data['categories'])
                row['crowd_annotations'] = sum(bool(a.get('iscrowd', 0)) for a in data['annotations'])
                assert row['categories'] == 80
        elif 'questions' in data:
            row.update({'kind': 'vqa_questions', 'questions': len(data['questions']),
                        'data_subtype': data.get('data_subtype')})
            assert row['questions'] == (443757 if split == 'train2014' else 214354)
            assert row['data_subtype'] == split
        elif 'annotations' in data:
            row.update({'kind': 'vqa_answers', 'questions': len(data['annotations']),
                        'answer_types': dict(Counter(a['answer_type'] for a in data['annotations'])),
                        'all_examples_have_10_answers': all(len(a['answers']) == 10 for a in data['annotations'])})
            assert row['questions'] == (443757 if split == 'train2014' else 214354)
            assert row['all_examples_have_10_answers']
        else:
            raise RuntimeError(f'Unexpected JSON structure: {path.name}')
        report3b['json_inventory'].append(row)
        del data
        gc.collect()
    save_record('manifests/annotation_inventory.json', report3b['json_inventory'])
    report3b.update({'status': 'passed', 'local_annotation_directory': str(json_dir),
                     'local_disk_free_GiB': round(shutil.disk_usage('/content').free / 2**30, 2)})
except Exception:
    report3b['status'] = 'failed'
    report3b['traceback'] = traceback.format_exc()
    save_record(f'logs/block_03b_{stamp}.json', report3b)
    print('BLOCK 3B FAILED. Verified archives and diagnostics retained; no GPU used.', flush=True)
    raise

log_name = f'logs/block_03b_{stamp}.json'
digest = save_record(log_name, report3b)
print('\nBLOCK 3B COMPLETE — 7 archives and 10 annotation JSON files checked.')
print(json.dumps({'status': report3b['status'], 'json_inventory': report3b['json_inventory'],
                  'local_disk_free_GiB': report3b['local_disk_free_GiB'],
                  'images_downloaded': False, 'model_weights_loaded': False,
                  'examples_selected': False}, indent=2))
print('Saved log:', ROOT / log_name)
print('SHA256:', digest)
print('Keep accelerator None. Paste the complete output.')
