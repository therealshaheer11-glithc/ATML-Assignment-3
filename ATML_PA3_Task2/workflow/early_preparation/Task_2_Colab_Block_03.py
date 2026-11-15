# Block 3: official annotation archives and structure inventory, CPU only.
# Continue in the SAME notebook after successful Block 2.
import gc, inspect, time, urllib.request, zipfile
from collections import Counter
from pathlib import Path

required_state = ('ROOT', 'LOCAL', 'save_record', 'sconfig', 'qconfig')
if any(name not in globals() for name in required_state):
    raise RuntimeError('Block 2 state is missing. Stop and share this message; do not rerun Block 1.')
previous = ROOT / 'logs/block_02_20261031_151449_9ecb9e8b.json'
assert hashlib.sha256(previous.read_bytes()).hexdigest() == 'e33edb522e01d7ef475949aa6e56692356ddbbf524769307abdc45fe0d4171aa'
if os.environ.get('COLAB_GPU', '0') not in ('', '0'):
    print('This is CPU preparation; requesting runtime release.', flush=True)
    try:
        runtime.unassign()
    finally:
        raise RuntimeError('Select accelerator None before continuing.')

stamp = datetime.now(ZoneInfo('Asia/Karachi')).strftime('%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:8]
report3 = {'attempt': stamp, 'status': 'started', 'gpu_used': False,
           'images_downloaded': False, 'model_weights_loaded': False,
           'examples_selected': False, 'archives': [], 'json_inventory': []}
archives = LOCAL / 'annotation_archives'
json_dir = LOCAL / 'annotations'
archives.mkdir(parents=True, exist_ok=True)
json_dir.mkdir(parents=True, exist_ok=True)

def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def copy_checked(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    wanted = file_sha(source)
    if target.exists():
        if file_sha(target) != wanted:
            raise IOError(f'Existing file differs; preserving it: {target}')
        return wanted
    temporary = target.with_name(target.name + '.' + uuid.uuid4().hex + '.pending')
    shutil.copyfile(source, temporary)
    if file_sha(temporary) != wanted:
        raise IOError(f'Copy readback failed: {target.name}')
    temporary.rename(target)
    return wanted

# Official COCO public GCS bucket, and current links on visualqa.org/download.html.
plan = [
    ('coco2014', 'https://storage.googleapis.com/images.cocodataset.org/annotations/annotations_trainval2014.zip'),
    ('vqa_train_annotations', 'https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip'),
    ('vqa_val_annotations', 'https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip'),
    ('vqa_train_questions', 'https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip'),
    ('vqa_val_questions', 'https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip'),
    ('vqa_train_pairs', 'https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Complementary_Pairs_Train_mscoco.zip'),
    ('vqa_val_pairs', 'https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Complementary_Pairs_Val_mscoco.zip'),
]
coco_needed = {'captions_train2014.json', 'captions_val2014.json',
               'instances_train2014.json', 'instances_val2014.json'}

try:
    # Record the actual installed-config origin of the warnings; change no fields.
    from transformers.models.siglip.configuration_siglip import SiglipTextConfig
    from transformers.models.siglip.modeling_siglip import SiglipVisionModel
    text_source = inspect.getsource(SiglipTextConfig)
    vision_source = inspect.getsource(SiglipVisionModel)
    text_cfg = sconfig.text_config
    warning_record = {
        'text_vocab_size': text_cfg.vocab_size,
        'text_bos_id': text_cfg.bos_token_id,
        'text_eos_id': text_cfg.eos_token_id,
        'qwen_eos_id_unchanged': qconfig.eos_token_id,
        'planned_encoder_class': 'SiglipVisionModel',
        'siglip_text_branch_used_in_task2': False,
        'checkpoint_configuration_modified': False,
        'text_config_source_sha256': hashlib.sha256(text_source.encode()).hexdigest(),
        'vision_class_source_sha256': hashlib.sha256(vision_source.encode()).hexdigest(),
        'text_default_source_lines': [line.strip() for line in text_source.splitlines()
                                      if 'bos_token_id' in line or 'eos_token_id' in line],
        'vision_normalization_source_lines': [line.strip() for line in vision_source.splitlines()
                                             if 'post_layernorm' in line],
        'vision_forward_numerically_tested': False,
    }
    assert warning_record['qwen_eos_id_unchanged'] == 151645
    save_record('provenance/siglip_warning_analysis.json', warning_record)
    report3['siglip_warning_analysis'] = warning_record

    for key, url in plan:
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
            print(f'Downloading {key} (resume offset {offset:,} bytes)...', flush=True)
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
                resumed = offset > 0 and response.status == 206
                if resumed and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                    raise IOError('Server returned an unexpected resume position.')
                mode = 'ab' if resumed else 'wb'
                expected = response.headers.get('Content-Length')
                received, last_print = 0, time.monotonic()
                with partial.open(mode) as out:
                    while True:
                        chunk = response.read(4 * 1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
                        received += len(chunk)
                        if time.monotonic() - last_print > 20:
                            print(f'  {partial.stat().st_size / 2**20:.1f} MiB received', flush=True)
                            last_print = time.monotonic()
                if expected is not None and received != int(expected):
                    raise IOError(f'Incomplete download: {key}; partial retained for retry.')
            with zipfile.ZipFile(partial) as z:
                bad = z.testzip()
                if bad:
                    raise IOError(f'Archive CRC failed: {bad}')
            partial.rename(local_zip)

        digest = file_sha(local_zip)
        if receipt_path.exists():
            old = json.loads(receipt_path.read_text())
            if old['sha256'] != digest or old['url'] != url:
                raise IOError(f'Archive conflicts with its saved receipt: {key}')
        with zipfile.ZipFile(local_zip) as z:
            bad = z.testzip()
            if bad:
                raise IOError(f'Archive CRC failed: {bad}')
            members = [m for m in z.infolist() if not m.is_dir() and m.filename.endswith('.json')
                       and (key != 'coco2014' or Path(m.filename).name in coco_needed)]
            expected_count = 4 if key == 'coco2014' else 1
            if len(members) != expected_count:
                raise RuntimeError(f'Unexpected annotation members in {key}: {[m.filename for m in members]}')
            extracted = []
            for member in members:
                # Use basenames, not paths supplied inside an archive.
                dest = json_dir / Path(member.filename).name
                temp = dest.with_name(dest.name + '.extracting')
                with z.open(member) as src, temp.open('wb') as out:
                    shutil.copyfileobj(src, out)
                if dest.exists():
                    if file_sha(dest) != file_sha(temp):
                        raise IOError(f'Existing annotation differs: {dest.name}')
                    temp.unlink()
                else:
                    temp.rename(dest)
                extracted.append({'filename': dest.name, 'bytes': dest.stat().st_size,
                                  'sha256': file_sha(dest)})
        copy_checked(local_zip, durable_zip)
        receipt = {'key': key, 'url': url, 'bytes': local_zip.stat().st_size,
                   'sha256': digest, 'zip_crc_passed': True, 'json_files': extracted,
                   'hash_provenance': 'Observed download SHA256; not an official published checksum'}
        save_record(receipt_rel, receipt)
        report3['archives'].append(receipt)
        print(f'Verified and saved: {key}', flush=True)

    # Parse one file at a time to limit CPU memory. No selections or splits yet.
    for path in sorted(json_dir.glob('*.json')):
        print('Inspecting', path.name, flush=True)
        with path.open() as f:
            data = json.load(f)
        row = {'filename': path.name, 'bytes': path.stat().st_size}
        if isinstance(data, list):
            assert all(isinstance(pair, list) and len(pair) == 2 and pair[0] != pair[1] for pair in data)
            row.update({'kind': 'complementary_pairs', 'pairs': len(data)})
        elif 'images' in data:
            row.update({'kind': 'coco_annotations', 'images': len(data['images']),
                        'annotations': len(data['annotations'])})
            if 'instances_' in path.name:
                row['categories'] = len(data['categories'])
                row['crowd_annotations'] = sum(bool(a.get('iscrowd', 0)) for a in data['annotations'])
        elif 'questions' in data:
            row.update({'kind': 'vqa_questions', 'questions': len(data['questions']),
                        'data_subtype': data.get('data_subtype')})
        elif 'annotations' in data:
            row.update({'kind': 'vqa_answers', 'questions': len(data['annotations']),
                        'answer_types': dict(Counter(a['answer_type'] for a in data['annotations'])),
                        'all_examples_have_10_answers': all(len(a['answers']) == 10 for a in data['annotations'])})
            assert row['all_examples_have_10_answers']
        else:
            raise RuntimeError(f'Unexpected JSON structure: {path.name}')
        report3['json_inventory'].append(row)
        del data
        gc.collect()

    report3.update({'status': 'passed', 'local_annotation_directory': str(json_dir),
                    'local_disk_free_GiB': round(shutil.disk_usage('/content').free / 2**30, 2)})
    save_record('manifests/annotation_inventory.json', report3['json_inventory'])
except Exception:
    report3['status'] = 'failed'
    report3['traceback'] = traceback.format_exc()
    save_record(f'logs/block_03_{stamp}.json', report3)
    print('BLOCK 3 FAILED. Progress/diagnostics saved; keep accelerator None.', flush=True)
    raise

log_name = f'logs/block_03_{stamp}.json'
digest = save_record(log_name, report3)
print('\nBLOCK 3 COMPLETE — annotation archives saved and checked; no images or weights downloaded.')
print(json.dumps({'status': report3['status'], 'archives': len(report3['archives']),
                  'json_inventory': report3['json_inventory'],
                  'siglip_warning_analysis': warning_record,
                  'local_disk_free_GiB': report3['local_disk_free_GiB']}, indent=2))
print('Saved log:', ROOT / log_name)
print('SHA256:', digest)
print('Paste the complete output. Keep accelerator None.')
