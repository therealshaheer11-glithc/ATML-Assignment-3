# Block 4B: inspect flagged natural answers and caption eligibility. CPU only.
# No exclusions are added, and no experiment subsets are selected.
import os, json, uuid, shutil, traceback, gc, warnings, importlib.util
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import Counter, defaultdict

needed = ('ROOT', 'LOCAL', 'save_record', 'file_sha', 'copy_checked', 'tokenizer')
if any(n not in globals() for n in needed):
    raise RuntimeError('Notebook state missing. Stop; do not rerun Block 1.')
previous = ROOT / 'logs/block_04a_20261031_154343_2e9b562d.json'
assert file_sha(previous) == '78d19d8f125aade489c94c1ca23fb82212d4168153cc40f09af3193be1bacd1e'
if os.environ.get('COLAB_GPU', '0') not in ('', '0'):
    print('CPU task: requesting GPU release.', flush=True)
    try:
        runtime.unassign()
    finally:
        raise RuntimeError('Reconnect with accelerator None.')

stamp = datetime.now(ZoneInfo('Asia/Karachi')).strftime('%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:8]
report4b = {'attempt': stamp, 'status': 'started', 'gpu_used': False,
            'subsets_selected': False, 'new_exclusions_applied': False,
            'natural_audits': {}, 'caption_eligibility': {}}
annotations = LOCAL / 'annotations'
scratch = LOCAL / 'eligibility'
scratch.mkdir(parents=True, exist_ok=True)

try:
    module_path = ROOT / 'code/vqa_normalization.py'
    assert file_sha(module_path) == '56251cdcddbad50cc5e31cb83744cf1752267390b7d4e315dfe00e9edf905d0d'
    spec = importlib.util.spec_from_file_location('task2_checked_vqa_normalizer', module_path)
    module = importlib.util.module_from_spec(spec)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', SyntaxWarning)
        spec.loader.exec_module(module)
    norm = module.normalize_answer
    report4b['normalizer_import_warnings'] = [str(w.message) for w in caught]

    for split in ('train2014', 'val2014'):
        summary = json.loads((ROOT / f'manifests/eligibility/natural_{split}_summary.json').read_text())
        paths = {}
        for item in summary['outputs']:
            durable = Path(item['path'])
            local = scratch / durable.name
            copy_checked(durable, local)
            assert file_sha(local) == item['sha256']
            paths['rejected' if 'rejected_' in local.name else 'eligible'] = local

        flagged = []
        with paths['eligible'].open() as stream:
            for line in stream:
                row = json.loads(line)
                if any(not ex['canonical'] for ex in row['examples']):
                    for ex in row['examples']:
                        assert norm(ex['target']) == ex['canonical']
                        assert sum(norm(a) == ex['canonical'] for a in ex['human_answers']) == ex['agreement']
                    flagged.append(row)
        assert len(flagged) == summary['review_flags'].get('empty_normalized_canonical_pairs', 0)
        empty_targets = Counter(ex['target'] for row in flagged for ex in row['examples'] if not ex['canonical'])

        mixed_ids = []
        with paths['rejected'].open() as stream:
            for line in stream:
                row = json.loads(line)
                if row['reason'] == 'mixed_answer_types_unassigned_to_quota':
                    mixed_ids.append(tuple(row['question_ids']))
        ids = {qid for pair in mixed_ids for qid in pair}
        with (annotations / f'v2_OpenEnded_mscoco_{split}_questions.json').open() as stream:
            payload = json.load(stream)
        qmap = {int(q['question_id']): q for q in payload['questions'] if int(q['question_id']) in ids}
        del payload
        with (annotations / f'v2_mscoco_{split}_annotations.json').open() as stream:
            payload = json.load(stream)
        amap = {}
        for annotation in payload['annotations']:
            qid = int(annotation['question_id'])
            if qid in ids:
                refs = sorted(annotation['answers'], key=lambda a: int(a['answer_id']))
                canonical, agreement = Counter(norm(a['answer']) for a in refs).most_common(1)[0]
                amap[qid] = {'question_id': qid, 'image_id': int(annotation['image_id']),
                              'question': qmap[qid]['question'], 'answer_type': annotation['answer_type'],
                              'canonical': canonical, 'agreement': agreement,
                              'human_answers': [a['answer'] for a in refs]}
        del payload
        gc.collect()
        assert set(amap) == ids
        mixed_cases, mixed_types = [], Counter()
        for pair in sorted(mixed_ids):
            examples = [amap[qid] for qid in pair]
            key = ' | '.join(sorted(ex['answer_type'] for ex in examples))
            mixed_types[key] += 1
            mixed_cases.append({'question_ids': pair, 'examples': examples})
        audit = {'source_split': split, 'empty_normalization_pairs': flagged,
                 'empty_normalization_original_targets': dict(empty_targets),
                 'mixed_answer_type_pairs': mixed_cases, 'mixed_type_counts': dict(mixed_types),
                 'no_new_exclusion_policy_applied': True}
        audit_rel = f'manifests/eligibility/natural_edge_case_audit_{split}.json'
        audit_hash = save_record(audit_rel, audit)
        display_cases = [
            {'pair_id': row['pair_id'], 'question': row['question'], 'answer_type': row['answer_type'],
             'examples': [{'image_id': ex['image_id'], 'target': ex['target'], 'canonical': ex['canonical'],
                           'agreement': ex['agreement'], 'raw_answer_counts': dict(Counter(ex['human_answers']))}
                          for ex in row['examples']]}
            for row in flagged
        ]
        report4b['natural_audits'][split] = {
            'empty_normalization_pair_count': len(flagged),
            'empty_normalization_original_targets': dict(empty_targets),
            'empty_normalization_cases': display_cases,
            'mixed_answer_type_pair_count': len(mixed_cases), 'mixed_type_counts': dict(mixed_types),
            'mixed_diagnostic_examples_first_3_by_pair_ids': mixed_cases[:3],
            'audit_path': str(ROOT / audit_rel), 'audit_sha256': audit_hash}
        del flagged, mixed_cases, qmap, amap, ids
        gc.collect()

    # Caption caps apply to original caption content, without chat/termination tokens.
    coco_receipt = json.loads((ROOT / 'manifests/annotation_downloads/coco2014.json').read_text())
    caption_hashes = {r['filename']: r['sha256'] for r in coco_receipt['json_files']}
    assert tokenizer.eos_token_id == 151645
    for split in ('train2014', 'val2014'):
        print(f'\nChecking {split} captions...', flush=True)
        raw_path = annotations / f'captions_{split}.json'
        assert file_sha(raw_path) == caption_hashes[raw_path.name]
        with raw_path.open() as stream:
            payload = json.load(stream)
        image_ids = {int(im['id']) for im in payload['images']}
        records = sorted(payload['annotations'], key=lambda a: int(a['id']))
        assert len({int(a['id']) for a in records}) == len(records)
        per_image = Counter()
        distinct_text = defaultdict(set)
        totals = Counter()
        local_output = scratch / f'caption_candidates_{split}.jsonl'
        temp = local_output.with_suffix('.jsonl.pending')
        with temp.open('w') as stream:
            for start in range(0, len(records), 512):
                batch = records[start:start + 512]
                encoded = tokenizer([a['caption'] for a in batch], add_special_tokens=False,
                                    padding=False, truncation=False, return_attention_mask=False)['input_ids']
                assert len(encoded) == len(batch)
                for annotation, tokens in zip(batch, encoded):
                    image_id, caption_id = int(annotation['image_id']), int(annotation['id'])
                    assert image_id in image_ids
                    text, length = annotation['caption'], len(tokens)
                    within_cap = length <= 64
                    empty = not text.strip()
                    totals['all_captions'] += 1
                    totals['within_cap' if within_cap else 'above_cap'] += 1
                    totals['empty_or_whitespace'] += int(empty)
                    if within_cap:
                        per_image[image_id] += 1
                        totals['duplicate_eligible_caption_texts'] += int(text in distinct_text[image_id])
                        distinct_text[image_id].add(text)
                    stream.write(json.dumps({'caption_id': caption_id, 'image_id': image_id,
                                             'caption': text, 'content_tokens': length,
                                             'within_content_cap': within_cap,
                                             'empty_or_whitespace': empty}, ensure_ascii=False) + '\n')
                if start and start % (512 * 100) == 0:
                    print(f'  checked {min(start + 512, len(records)):,} captions', flush=True)
        if local_output.exists():
            if file_sha(local_output) != file_sha(temp):
                raise IOError('Caption index differs from previous output; preserving it.')
            temp.unlink()
        else:
            temp.rename(local_output)
        target = ROOT / 'manifests/eligibility' / local_output.name
        digest = copy_checked(local_output, target)
        needed_per_image = 3 if split == 'train2014' else 1
        qualifying = sum(len(distinct_text[image_id]) >= needed_per_image for image_id in image_ids)
        caption_summary = {
            'source_split': split, 'images': len(image_ids), 'counts': dict(totals),
            'distinct_caption_texts_required_per_image': needed_per_image,
            'images_with_enough_distinct_caption_texts_within_64_tokens': qualifying,
            'required_image_count_before_evaluation_image_exclusions': 6000 if split == 'train2014' else 96,
            'image_disjoint_allocation_tested': False, 'subsets_selected': False,
            'index_path': str(target), 'index_sha256': digest, 'index_bytes': target.stat().st_size}
        report4b['caption_eligibility'][split] = caption_summary
        save_record(f'manifests/eligibility/caption_{split}_summary.json', caption_summary)
        print(json.dumps(caption_summary, indent=2), flush=True)
        del payload, records, image_ids, per_image, distinct_text
        gc.collect()
    report4b.update({'status': 'passed',
                     'local_disk_free_GiB': round(shutil.disk_usage('/content').free / 2**30, 2)})
except Exception:
    report4b['status'] = 'failed'
    report4b['traceback'] = traceback.format_exc()
    save_record(f'logs/block_04b_{stamp}.json', report4b)
    print('BLOCK 4B FAILED; diagnostics saved. Keep accelerator None.', flush=True)
    raise

log_path = f'logs/block_04b_{stamp}.json'
digest = save_record(log_path, report4b)
print('\nBLOCK 4B COMPLETE — edge-case audit and caption counts; no subsets selected.')
print(json.dumps(report4b, indent=2))
print('Saved log:', ROOT / log_path)
print('SHA256:', digest)
print('Keep accelerator None. Paste the complete output.')
