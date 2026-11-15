# Block 5A: approved exclusions, seeded natural-VQA subsets, training captions.
# CPU only. Grounding evaluation and caption development are allocated later.
import os, json, uuid, shutil, traceback, gc
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import Counter, defaultdict
import numpy as np

required = ('ROOT', 'LOCAL', 'save_record', 'file_sha', 'copy_checked')
if any(n not in globals() for n in required):
    raise RuntimeError('Notebook state missing. Stop; do not rerun Block 1.')
prior_path = ROOT / 'logs/block_04b_20261031_155026_82d3640b.json'
assert file_sha(prior_path) == '35c8314906b0f5cdab93f998159f43b493fc2e111cc058cea957b2053dd23945'
prior = json.loads(prior_path.read_text())
if os.environ.get('COLAB_GPU', '0') not in ('', '0'):
    print('CPU selection: requesting GPU release.', flush=True)
    try:
        runtime.unassign()
    finally:
        raise RuntimeError('Reconnect with accelerator None.')
stamp = datetime.now(ZoneInfo('Asia/Karachi')).strftime('%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:8]
report5 = {'attempt': stamp, 'status': 'started', 'gpu_used': False,
           'complete_dataset_frozen': False, 'allocation': {}, 'outputs': []}
local5 = LOCAL / 'selection'
local5.mkdir(parents=True, exist_ok=True)

# Seeded sampling is without replacement over pair IDs, with a fixed type order.
# Image separation is between evaluation sets; no extra within-set restriction.
def choose_pairs(rows, quotas, rng, blocked_images=()):
    blocked_images = set(blocked_images)
    selected = []
    for answer_type in ('yes/no', 'number', 'other'):
        candidates = sorted((r for r in rows if r['answer_type'] == answer_type),
                            key=lambda r: tuple(int(ex['question_id']) for ex in r['examples']))
        count = 0
        for index in rng.permutation(len(candidates)):
            row = candidates[int(index)]
            images = {int(ex['image_id']) for ex in row['examples']}
            if images & blocked_images:
                continue
            selected.append(row)
            count += 1
            if count == quotas[answer_type]:
                break
        if count != quotas[answer_type]:
            raise RuntimeError(f'Insufficient {answer_type} pairs after image exclusions: {count}/{quotas[answer_type]}')
    return selected

def pair_images(rows):
    return {int(ex['image_id']) for row in rows for ex in row['examples']}

def save_rows(filename, rows):
    path = local5 / filename
    temp = path.with_name(path.name + '.pending')
    with temp.open('w') as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + '\n')
    if path.exists():
        if file_sha(path) != file_sha(temp):
            raise IOError(f'Existing selection differs; preserving it: {filename}')
        temp.unlink()
    else:
        temp.rename(path)
    durable = ROOT / 'manifests/selected' / filename
    digest = copy_checked(path, durable)
    return {'path': str(durable), 'sha256': digest, 'rows': len(rows), 'bytes': path.stat().st_size}

try:
    # Persist explicit approval of the two newly discovered eligibility rules.
    policy = {
        'decision_id': 'A004', 'date': '2026-10-31',
        'user_response': 'Approve excluding both groups (recommended)',
        'exclude_empty_normalized_canonical_pairs': True,
        'exclude_mixed_official_answer_type_pairs': True,
        'normalization_changed': False,
        'basis': 'Prediction-independent reference/type audit before subset selection',
    }
    save_record('configs/additional_eligibility_A004.json', policy)
    exclusions = []
    excluded_ids = set()
    for split in ('train2014', 'val2014'):
        audit_info = prior['natural_audits'][split]
        audit_path = Path(audit_info['audit_path'])
        assert file_sha(audit_path) == audit_info['audit_sha256']
        audit = json.loads(audit_path.read_text())
        for row in audit['empty_normalization_pairs']:
            exclusions.append({'pair_id': row['pair_id'], 'source_split': split,
                               'question_ids': [int(ex['question_id']) for ex in row['examples']],
                               'image_ids': [int(ex['image_id']) for ex in row['examples']],
                               'reason': 'empty_normalized_canonical_answer', 'decision_id': 'A004'})
        for row in audit['mixed_answer_type_pairs']:
            qids = sorted(map(int, row['question_ids']))
            exclusions.append({'pair_id': f'{split}:{qids[0]}:{qids[1]}', 'source_split': split,
                               'question_ids': qids,
                               'image_ids': [int(ex['image_id']) for ex in row['examples']],
                               'reason': 'mixed_official_answer_types', 'decision_id': 'A004'})
    exclusions.sort(key=lambda r: (r['source_split'], tuple(r['question_ids'])))
    excluded_ids = {r['pair_id'] for r in exclusions}
    assert len(exclusions) == len(excluded_ids) == 161
    report5['outputs'].append(save_rows('additional_exclusions_A004.jsonl', exclusions))
    report5['approved_exclusion_counts'] = dict(Counter(r['reason'] for r in exclusions))

    # Verify and load the previously saved natural eligibility pools.
    old_log = ROOT / 'logs/block_04a_20261031_154343_2e9b562d.json'
    assert file_sha(old_log) == '78d19d8f125aade489c94c1ca23fb82212d4168153cc40f09af3193be1bacd1e'
    old = json.loads(old_log.read_text())
    pools = {}
    for split in ('train2014', 'val2014'):
        item = next(r for r in old['splits'][split]['outputs'] if '/natural_eligible_' in r['path'])
        durable = Path(item['path'])
        local = LOCAL / 'eligibility' / durable.name
        copy_checked(durable, local)
        assert file_sha(local) == item['sha256']
        rows, removed = [], 0
        with local.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row['pair_id'] in excluded_ids:
                    removed += 1
                    continue
                assert row['source_split'] == split
                assert len(row['examples']) == 2
                assert len({ex['image_id'] for ex in row['examples']}) == 2
                assert all(ex['agreement'] >= 8 and ex['canonical'] for ex in row['examples'])
                assert all(ex['answer_type'] == row['answer_type'] for ex in row['examples'])
                assert row['examples'][0]['canonical'] != row['examples'][1]['canonical']
                rows.append(row)
        assert removed == (7 if split == 'train2014' else 6)
        pools[split] = rows
        report5.setdefault('post_policy_pool_counts', {})[split] = dict(Counter(r['answer_type'] for r in rows))

    quotas = {'train': {'yes/no': 640, 'number': 256, 'other': 640},
              'dev': {'yes/no': 12, 'number': 8, 'other': 12},
              'test': {'yes/no': 36, 'number': 24, 'other': 36}}
    selection_recipe = {
        'seed': 6304, 'rng': 'numpy.random.default_rng', 'numpy_version': np.__version__,
        'allocation_order': ['natural_train', 'natural_dev', 'natural_test',
                             'caption_train_images', 'caption_train_captions'],
        'pair_type_order': ['yes/no', 'number', 'other'],
        'pair_sort': 'Ascending integer question-ID tuple before seeded permutation',
        'pair_sampling': 'Without replacement by pair ID',
        'evaluation_separation': 'Natural test excludes all natural development images',
        'within_evaluation_suite_image_reuse': 'Allowed; unique-image counts are recorded',
        'caption_text_deduplication': 'Exact text within image; retain lowest original caption ID',
        'caption_epoch_order_rng': 'Separate default_rng(6304), two successive permutations',
        'quotas': quotas,
        'remaining': ['grounding evaluation/pools', 'caption development',
                      'global split/label audit', 'complete dataset freeze'],
    }
    save_record('configs/data_selection_recipe.json', selection_recipe)
    rng = np.random.default_rng(6304)
    selected = {}
    selected['train'] = choose_pairs(pools['train2014'], quotas['train'], rng)
    selected['dev'] = choose_pairs(pools['val2014'], quotas['dev'], rng)
    selected['test'] = choose_pairs(pools['val2014'], quotas['test'], rng,
                                    blocked_images=pair_images(selected['dev']))
    dev_images, test_images = pair_images(selected['dev']), pair_images(selected['test'])
    natural_train_images = pair_images(selected['train'])
    assert dev_images.isdisjoint(test_images)
    assert natural_train_images.isdisjoint(dev_images | test_images)
    for split in ('train', 'dev', 'test'):
        rows = [{**r, 'experiment_split': split} for r in selected[split]]
        assert dict(Counter(r['answer_type'] for r in rows)) == quotas[split]
        assert len({r['pair_id'] for r in rows}) == len(rows)
        assert not ({r['pair_id'] for r in rows} & excluded_ids)
        report5['outputs'].append(save_rows(f'natural_{split}.jsonl', rows))
        report5['allocation'][f'natural_{split}'] = {
            'pairs': len(rows), 'questions': 2 * len(rows),
            'answer_type_counts': dict(Counter(r['answer_type'] for r in rows)),
            'unique_images': len(pair_images(rows))}
    del pools
    gc.collect()

    # Choose 6,000 training images, then three distinct original captions each.
    cap = prior['caption_eligibility']['train2014']
    caption_source = Path(cap['index_path'])
    local_caps = LOCAL / 'eligibility' / caption_source.name
    copy_checked(caption_source, local_caps)
    assert file_sha(local_caps) == cap['index_sha256']
    by_image = defaultdict(dict)
    all_train_images = set()
    with local_caps.open() as stream:
        for line in stream:
            row = json.loads(line)
            image_id = int(row['image_id'])
            all_train_images.add(image_id)
            if row['within_content_cap']:
                assert not row['empty_or_whitespace']
                old_row = by_image[image_id].get(row['caption'])
                if old_row is None or int(row['caption_id']) < int(old_row['caption_id']):
                    by_image[image_id][row['caption']] = row
    available = sorted(im for im, captions in by_image.items() if len(captions) >= 3)
    assert len(available) >= 6000
    assert natural_train_images <= all_train_images
    assert (dev_images | test_images).isdisjoint(all_train_images)
    caption_images = [available[int(i)] for i in rng.permutation(len(available))[:6000]]
    caption_rows = []
    for image_id in sorted(caption_images):
        options = sorted(by_image[image_id].values(), key=lambda r: int(r['caption_id']))
        for index in rng.permutation(len(options))[:3]:
            caption_rows.append({**options[int(index)], 'experiment_split': 'train'})
    caption_rows.sort(key=lambda r: (int(r['image_id']), int(r['caption_id'])))
    assert len(caption_rows) == len({r['caption_id'] for r in caption_rows}) == 18000
    assert len({r['image_id'] for r in caption_rows}) == 6000
    assert all(r['within_content_cap'] and r['content_tokens'] <= 64 for r in caption_rows)
    per_image = Counter(r['image_id'] for r in caption_rows)
    assert set(per_image.values()) == {3}
    report5['outputs'].append(save_rows('captions_train.jsonl', caption_rows))
    epoch_rng = np.random.default_rng(6304)
    epoch_orders = {'seed': 6304, 'epoch_caption_ids': [
        [caption_rows[int(i)]['caption_id'] for i in epoch_rng.permutation(len(caption_rows))]
        for _ in range(2)]}
    for order in epoch_orders['epoch_caption_ids']:
        assert len(order) == len(set(order)) == 18000
    epoch_hash = save_record('manifests/selected/stage1_caption_epoch_orders.json', epoch_orders)
    report5['stage1_order_sha256'] = epoch_hash
    report5['allocation']['captions_train'] = {'images': 6000, 'captions': 18000,
                                              'passes': 2, 'planned_presentations': 36000}
    union = sorted(set(caption_images) | natural_train_images)
    union_record = {'source_split': 'train2014', 'image_ids': union,
                    'caption_image_ids': sorted(caption_images),
                    'natural_vqa_image_ids': sorted(natural_train_images),
                    'purpose': 'Approved candidate image universe for grounding training pools',
                    'grounding_pool_feasibility_verified': False}
    report5['training_union_sha256'] = save_record('manifests/selected/training_image_union.json', union_record)
    reservation = {'natural_dev_image_ids': sorted(dev_images),
                   'natural_test_image_ids': sorted(test_images),
                   'exclude_from_all_grounding_evaluation_and_caption_dev': sorted(dev_images | test_images),
                   'caption_dev_not_selected': True, 'grounding_eval_not_selected': True}
    save_record('manifests/selected/evaluation_image_reservations.json', reservation)
    report5.update({'status': 'passed', 'training_image_union_count': len(union),
                     'estimated_training_union_feature_bytes_without_headers': len(union) * 196 * 768 * 2,
                     'natural_dev_test_image_disjoint': True,
                     'local_disk_free_GiB': round(shutil.disk_usage('/content').free / 2**30, 2)})
except Exception:
    report5['status'] = 'failed'
    report5['traceback'] = traceback.format_exc()
    save_record(f'logs/block_05a_{stamp}.json', report5)
    print('BLOCK 5A FAILED; diagnostics and existing manifests preserved.', flush=True)
    raise

log_path = f'logs/block_05a_{stamp}.json'
digest = save_record(log_path, report5)
print('\nBLOCK 5A COMPLETE — natural subsets and training captions selected; complete dataset not frozen.')
print(json.dumps(report5, indent=2))
print('Saved log:', ROOT / log_path)
print('SHA256:', digest)
print('Keep accelerator None. Paste the complete output.')
