"""Block 7. CPU source verification, final dataset freeze and cache input preparation."""
import os
import sys
import gc
import io
import json
import hashlib
import importlib.util
import importlib.metadata
import shutil
import traceback
import uuid
import zipfile
from pathlib import Path
from datetime import datetime, timezone

if os.environ.get('COLAB_GPU', '0') not in ('', '0'):
    from google.colab import runtime
    try:
        runtime.unassign()
    finally:
        raise RuntimeError('Reconnect with accelerator None for CPU preparation.')
from google.colab import drive
if not Path('/content/drive/MyDrive').exists():
    drive.mount('/content/drive')
for previous_name in ('A007_REPLACEMENT_APP', 'A008_REPLACEMENT_APP', 'A009_REPLACEMENT_APP',
                      'A010_REPLACEMENT_APP', 'A011_REPLACEMENT_APP'):
    previous = globals().get(previous_name)
    if previous is not None:
        previous.close()
        globals()[previous_name] = None

RUN_ROOT = Path('/content/drive/MyDrive/ATML_PA3/task2/run_20261031_150949_1cff148a')
LOCAL_ROOT = Path('/content/atml_pa3_task2/run_20261031_150949_1cff148a')


def hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()


def write_checked(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_bytes() == content, 'Different existing artifact preserved: ' + str(path)
        return
    temporary = path.with_name(path.name + '.part_' + uuid.uuid4().hex[:8])
    temporary.write_bytes(content)
    assert temporary.read_bytes() == content
    temporary.replace(path)
    assert path.read_bytes() == content


preflight_package = globals().get('task2_preflight_package_bytes')
if not isinstance(preflight_package, bytes):
    raise RuntimeError('Run the supplied Block 7 package loader.')
PACKAGE_SHA = hashlib.sha256(preflight_package).hexdigest()
WORK = LOCAL_ROOT / 'final_preflight' / PACKAGE_SHA[:16]
with zipfile.ZipFile(io.BytesIO(preflight_package)) as archive:
    assert archive.testzip() is None
    package_manifest = json.loads(archive.read('package_manifest.json'))
    assert set(archive.namelist()) == {r['path'] for r in package_manifest['files']} | {'package_manifest.json'}
    for record in package_manifest['files']:
        relative = Path(record['path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        content = archive.read(record['path'])
        assert len(content) == record['bytes'] and hashlib.sha256(content).hexdigest() == record['sha256']
        write_checked(WORK / relative, content)
sys.path.insert(0, str(WORK))
for module_name in list(sys.modules):
    if module_name == 'task2' or module_name.startswith('task2.'):
        del sys.modules[module_name]
from task2.dataset import load_dataset, validate_dataset, pair_images
from task2.checks import run_checks
from task2.sourcecheck import catalog_from_instances, verify_grounding_pairs, verify_training_pool
from task2.protocol import MODEL_REVISIONS

attempt = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:8]
try:
    data = load_dataset(WORK / 'dataset')
    DATASET_SHA = data['identity']
    manifest = data['manifest']
    assert manifest['run_id'] == RUN_ROOT.name
    for relative, expected in manifest['original_source_locks'].items():
        assert hash_file(RUN_ROOT / relative) == expected, 'Original source lock changed: ' + relative
    normalizer_path = RUN_ROOT / 'code/vqa_normalization.py'
    assert hash_file(normalizer_path) == manifest['normalizer_sha256']
    spec = importlib.util.spec_from_file_location('task2_official_normalizer', normalizer_path)
    normalizer_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(normalizer_module)
    normalize = normalizer_module.normalize_answer
    checks = run_checks(data, normalize)
    summary = validate_dataset(data, normalize)
    print('Human review complete: 246/246 accepted; 245 Colab decisions and one explicit chat decision.', flush=True)
    print('Sampler, schedules, scoring, official normalization and resume checks passed.', flush=True)

    receipt_path = RUN_ROOT / 'manifests/annotation_downloads/coco2014.json'
    expected_receipt = (WORK / 'dataset/provenance/annotation_receipt.json').read_bytes()
    assert receipt_path.read_bytes() == expected_receipt
    receipt = json.loads(expected_receipt)
    annotation_hashes = {record['filename']: record['sha256'] for record in receipt['json_files']}

    def load_original_annotation(filename):
        local = LOCAL_ROOT / 'annotations' / filename
        if local.exists():
            assert hash_file(local) == annotation_hashes[filename], filename
            return json.loads(local.read_bytes())
        source_zip = RUN_ROOT / 'raw_annotation_archives/coco2014.zip'
        assert hash_file(source_zip) == '031296bbc80c45a1d1f76bf9a90ead27e94e99ec629208449507a4917a3bf009'
        with zipfile.ZipFile(source_zip) as source:
            matches = [name for name in source.namelist() if Path(name).name == filename]
            assert len(matches) == 1
            content = source.read(matches[0])
        assert hashlib.sha256(content).hexdigest() == annotation_hashes[filename]
        return json.loads(content)

    groups = summary['image_groups']
    cache_ids = {'train2014': set(groups['training_union']),
        'val2014': set().union(*(set(ids) for name, ids in groups.items() if name != 'training_union'))}
    jobs = []
    source_proof = {}
    for split in ('train2014', 'val2014'):
        filename = f'instances_{split}.json'
        print('Verifying original instance annotations:', split, flush=True)
        payload = load_original_annotation(filename)
        catalog, images = catalog_from_instances(payload, cache_ids[split])
        pairs = data['grounding']['training_audit'] if split == 'train2014' else data['grounding']['dev'] + data['grounding']['test']
        verify_grounding_pairs(pairs, catalog)
        if split == 'train2014':
            verify_training_pool(data['pool'], catalog)
        for image_id, metadata in sorted(images.items()):
            filename = metadata['file_name']
            assert Path(filename).name == filename == f'COCO_{split}_{image_id:012d}.jpg'
            jobs.append({'source_split': split, 'image_id': image_id, 'file_name': filename,
                'width': int(metadata['width']), 'height': int(metadata['height']),
                'url': f'https://s3.amazonaws.com/images.cocodataset.org/{split}/{filename}',
                'original_coco_url': metadata.get('coco_url'), 'license_id': metadata.get('license')})
        source_proof[f'instances_{split}'] = {'sha256': annotation_hashes[f'instances_{split}.json'],
            'verified_cache_images': len(images), 'grounding_pairs_verified': len(pairs)}
        del payload, catalog, images
        gc.collect()
        print('Verifying original caption annotations:', split, flush=True)
        captions_payload = load_original_annotation(f'captions_{split}.json')
        selected = data['captions']['train' if split == 'train2014' else 'dev']
        selected_by_id = {r['caption_id']: r for r in selected}
        found = set()
        for row in captions_payload['annotations']:
            caption_id = int(row['id'])
            if caption_id in selected_by_id:
                expected = selected_by_id[caption_id]
                assert int(row['image_id']) == expected['image_id'] and row['caption'] == expected['caption']
                found.add(caption_id)
        assert found == set(selected_by_id)
        source_proof[f'captions_{split}'] = {'sha256': annotation_hashes[f'captions_{split}.json'],
            'verified_original_captions': len(found)}
        del captions_payload
        gc.collect()
    assert len(jobs) == len({(r['source_split'], r['image_id']) for r in jobs}) == 9260
    manual_receipt = json.loads((WORK / 'dataset/provenance/manual_tv_image_receipt.json').read_bytes())
    assert hash_file(WORK / 'manual_tv_source.jpg') == manual_receipt['sha256']
    manual_job = next(r for r in jobs if r['source_split'] == 'train2014' and r['image_id'] == 153620)
    assert [manual_job['width'], manual_job['height']] == manual_receipt['decoded_size']
    print('Original COCO instances, complete grounding pools, caption references and TV source identity verified.', flush=True)

    from transformers import AutoConfig, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(**{'pretrained_model_name_or_path': MODEL_REVISIONS['qwen']['repo_id'],
        'revision': MODEL_REVISIONS['qwen']['revision']}, trust_remote_code=False)
    qwen_config = AutoConfig.from_pretrained(MODEL_REVISIONS['qwen']['repo_id'], revision=MODEL_REVISIONS['qwen']['revision'], trust_remote_code=False)
    vision_config = AutoConfig.from_pretrained(MODEL_REVISIONS['siglip']['repo_id'], revision=MODEL_REVISIONS['siglip']['revision'], trust_remote_code=False)
    assert qwen_config.hidden_size == 1024
    assert vision_config.vision_config.hidden_size == 768 and vision_config.vision_config.patch_size == 16
    assert vision_config.vision_config.image_size == 224
    assert tokenizer.eos_token_id == 151645 and tokenizer.pad_token_id == 151643
    maxima = {'question': 0, 'user_text': 0, 'answer': 0, 'caption': 0}

    def cap(text, field, maximum):
        count = len(tokenizer.encode(text, add_special_tokens=False))
        maxima[field] = max(maxima[field], count)
        assert count <= maximum, (field, count, text)

    for pair in data['question_bank']:
        cap(pair['question'], 'question', 80)
        cap(pair['user_text'], 'user_text', 80)
    for pairs in list(data['grounding'].values()) + list(data['natural'].values()):
        for pair in pairs:
            cap(pair['question'], 'question', 80)
            cap(pair['user_text'], 'user_text', 80)
            for example in pair['examples']:
                cap(example['target'], 'answer', 16)
    for rows in data['captions'].values():
        for row in rows:
            cap(row['caption'], 'caption', 64)
    probe = tokenizer.apply_chat_template([{'role': 'user', 'content': 'Answer with a short phrase.\nHow many TVs can you see?'}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert 'How many TVs can you see?' in probe
    versions = {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'numpy', 'Pillow', 'tokenizers', 'huggingface-hub')}
    print('Pinned model configurations, native chat template and all content-token caps verified.', flush=True)

    destination = RUN_ROOT / 'datasets' / ('task2_A012_' + DATASET_SHA[:16])
    for entry in manifest['files']:
        write_checked(destination / entry['path'], (WORK / 'dataset' / entry['path']).read_bytes())
    write_checked(destination / 'dataset_manifest.json', (WORK / 'dataset/dataset_manifest.json').read_bytes())
    loaded_back = load_dataset(destination)
    assert loaded_back['identity'] == DATASET_SHA
    validate_dataset(loaded_back, normalize)
    jobs_bytes = ''.join(json.dumps(job, sort_keys=True) + '\n' for job in jobs).encode()
    write_checked(destination / 'inputs/image_jobs.jsonl', jobs_bytes)
    write_checked(destination / 'provenance/manual_tv_source.jpg', (WORK / 'manual_tv_source.jpg').read_bytes())
    code_dir = RUN_ROOT / 'code/task2_versions' / PACKAGE_SHA[:16]
    code_hashes = {}
    for path in sorted((WORK / 'task2').glob('*.py')):
        write_checked(code_dir / 'task2' / path.name, path.read_bytes())
        code_hashes[path.name] = hash_file(code_dir / 'task2' / path.name)
    freeze_path = destination / 'freeze_receipt.json'
    freeze = {'schema': 'atml-pa3-task2-dataset-freeze-v1', 'status': 'passed', 'dataset_frozen': True,
        'dataset_identity': DATASET_SHA, 'preflight_package_sha256': PACKAGE_SHA,
        'image_jobs_sha256': hashlib.sha256(jobs_bytes).hexdigest(), 'image_jobs': len(jobs),
        'source_proof': source_proof, 'checks': checks, 'content_token_maxima': maxima,
        'model_revisions': MODEL_REVISIONS, 'environment': versions, 'code_hashes': code_hashes,
        'native_chat_probe': probe, 'human_audit_accepts': 246,
        'completed_at_utc': datetime.now(timezone.utc).isoformat(), 'gpu_used': False,
        'model_weights_loaded': False, 'feature_cache_created': False, 'training_started': False}
    if freeze_path.exists():
        existing = json.loads(freeze_path.read_bytes())
        assert existing['dataset_identity'] == DATASET_SHA and existing['preflight_package_sha256'] == PACKAGE_SHA
        assert existing['image_jobs_sha256'] == freeze['image_jobs_sha256'] and existing['code_hashes'] == code_hashes
        assert existing['status'] == 'passed' and existing['dataset_frozen'] is True
    else:
        write_checked(freeze_path, json_bytes(freeze))
    pointer = {'schema': 'atml-pa3-task2-active-dataset-v1', 'dataset_dir': str(destination),
        'dataset_identity': DATASET_SHA, 'freeze_receipt_sha256': hash_file(freeze_path),
        'code_dir': str(code_dir), 'preflight_package_sha256': PACKAGE_SHA}
    write_checked(RUN_ROOT / 'manifests/active_task2_dataset.json', json_bytes(pointer))
    result = {'status': 'passed', 'block': '07', 'dataset_frozen': True, 'human_audit_accepts': 246,
        'dataset_identity': DATASET_SHA, 'dataset_dir': str(destination), 'image_jobs': len(jobs),
        'training_images': 8733, 'evaluation_images': 527, 'environment': versions,
        'content_token_maxima': maxima, 'freeze_receipt_sha256': hash_file(freeze_path),
        'gpu_used': False, 'training_started': False}
    write_checked(RUN_ROOT / f'logs/block_07_{attempt}.json', json_bytes(result))
    print('BLOCK 7 PASSED', flush=True)
    print(json.dumps(result, indent=2), flush=True)
    print('Return this complete output. Keep accelerator None for the next image-preparation block.', flush=True)
except Exception:
    failure = {'block': '07', 'status': 'failed', 'traceback': traceback.format_exc(),
        'preflight_package_sha256': PACKAGE_SHA, 'gpu_used': False, 'training_started': False}
    write_checked(RUN_ROOT / f'logs/block_07_failed_{attempt}.json', json_bytes(failure))
    raise
