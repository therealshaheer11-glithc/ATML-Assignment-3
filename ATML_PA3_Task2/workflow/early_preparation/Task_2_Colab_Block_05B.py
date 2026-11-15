# Block 5B: grounding pool feasibility, evaluation pair metadata, caption dev.
# CPU only. Question wording and visual label audits are still pending.
import os, json, uuid, shutil, traceback, gc
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import Counter, defaultdict
import numpy as np

needed = ('ROOT', 'LOCAL', 'save_record', 'file_sha', 'copy_checked', 'save_rows')
if any(n not in globals() for n in needed):
    raise RuntimeError('Notebook state missing. Stop; do not rerun Block 1.')
old_path = ROOT / 'logs/block_05a_20261031_155959_c7d3095c.json'
assert file_sha(old_path) == 'a84f2436281e657665850dfad2750840b55678b271f4c1dffddc7f8e195973fc'
old5 = json.loads(old_path.read_text())
if os.environ.get('COLAB_GPU', '0') not in ('', '0'):
    print('CPU preparation: requesting GPU release.', flush=True)
    try:
        runtime.unassign()
    finally:
        raise RuntimeError('Reconnect with accelerator None.')
stamp = datetime.now(ZoneInfo('Asia/Karachi')).strftime('%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:8]
report5b = {'attempt': stamp, 'status': 'started', 'gpu_used': False,
            'complete_dataset_frozen': False, 'questions_rendered': False,
            'visual_label_audit_complete': False, 'outputs': [], 'pool_summary': {}}

# Compact implicit Cartesian pools: no need to enumerate every possible pair.
def counting_combinations(category):
    return [(a, b) for a in range(1, 5) for b in range(a + 1, 5)
            if category['count_images'][str(a)] and category['count_images'][str(b)]]

def capacity(category, kind):
    if kind == 'presence':
        return len(category['present']) * len(category['absent'])
    return sum(len(category['count_images'][str(a)]) * len(category['count_images'][str(b)])
               for a, b in counting_combinations(category))

def filter_catalog(catalog, banned):
    banned = set(banned)
    return [{**c, 'present': [i for i in c['present'] if i not in banned],
             'absent': [i for i in c['absent'] if i not in banned],
             'crowd_images': [i for i in c['crowd_images'] if i not in banned],
             'count_images': {k: [i for i in v if i not in banned] for k, v in c['count_images'].items()}}
            for c in catalog]

def allocate_grounding(catalog, suite, quotas, rng):
    used, usage, rows = set(), Counter(), []
    for kind in ('presence', 'counting'):
        for _ in range(quotas[kind]):
            available = [c for c in catalog if capacity(c, kind) > usage[(kind, c['category_id'])]]
            if not available:
                raise RuntimeError(f'Insufficient unique {kind} pairs for {suite}.')
            category = available[int(rng.integers(len(available)))]
            for retry in range(10000):
                if kind == 'presence':
                    im1 = int(rng.choice(category['present']))
                    im2 = int(rng.choice(category['absent']))
                    targets, counts = ('yes', 'no'), (None, None)
                else:
                    combos = counting_combinations(category)
                    a, b = combos[int(rng.integers(len(combos)))]
                    im1 = int(rng.choice(category['count_images'][str(a)]))
                    im2 = int(rng.choice(category['count_images'][str(b)]))
                    targets, counts = (str(a), str(b)), (a, b)
                key = (kind, category['category_id'], min(im1, im2), max(im1, im2))
                if key not in used:
                    break
            else:
                raise RuntimeError('Unique-pair sampling retry limit reached; no seed/settings changed.')
            assert im1 != im2
            used.add(key)
            usage[(kind, category['category_id'])] += 1
            pair_id = f'grounding:{suite}:{kind}:{category["category_id"]}:{min(im1, im2)}:{max(im1, im2)}'
            examples = [{'question_id': f'{pair_id}:{j}', 'image_id': im,
                         'target': target, 'canonical': target, 'annotated_count': count}
                        for j, (im, target, count) in enumerate(zip((im1, im2), targets, counts))]
            rows.append({'pair_id': pair_id, 'source_split': 'val2014', 'experiment_split': suite,
                         'question_type': kind, 'category_id': category['category_id'],
                         'category_name': category['category_name'],
                         'question_template': ('Does this image contain [object]?' if kind == 'presence'
                                               else 'How many [objects] can you see?'),
                         'question_rendered': False, 'examples': examples})
    assert dict(Counter(r['question_type'] for r in rows)) == quotas
    return rows

def image_set(rows):
    return {int(ex['image_id']) for row in rows for ex in row['examples']}

try:
    union_path = ROOT / 'manifests/selected/training_image_union.json'
    assert file_sha(union_path) == old5['training_union_sha256']
    training_ids = set(json.loads(union_path.read_text())['image_ids'])
    assert len(training_ids) == 8733
    natural = {}
    for suite in ('dev', 'test'):
        item = next(r for r in old5['outputs'] if r['path'].endswith(f'/natural_{suite}.jsonl'))
        path = Path(item['path'])
        assert file_sha(path) == item['sha256']
        with path.open() as stream:
            natural[suite] = [json.loads(line) for line in stream]
    reserved = image_set(natural['dev']) | image_set(natural['test'])
    assert len(reserved) == 256 and training_ids.isdisjoint(reserved)
    coco = json.loads((ROOT / 'manifests/annotation_downloads/coco2014.json').read_text())
    raw_hashes = {r['filename']: r['sha256'] for r in coco['json_files']}
    catalogs, source_ids = {}, {}
    categories_reference = None
    for split in ('train2014', 'val2014'):
        print(f'Indexing {split} object annotations...', flush=True)
        path = LOCAL / 'annotations' / f'instances_{split}.json'
        assert file_sha(path) == raw_hashes[path.name]
        with path.open() as stream:
            payload = json.load(stream)
        source_ids[split] = {int(im['id']) for im in payload['images']}
        wanted = training_ids if split == 'train2014' else source_ids[split] - reserved
        assert wanted <= source_ids[split]
        categories = sorted(payload['categories'], key=lambda c: int(c['id']))
        category_signature = [(int(c['id']), c['name']) for c in categories]
        if categories_reference is None:
            categories_reference = category_signature
        assert category_signature == categories_reference and len(categories) == 80
        present, crowds, counts = defaultdict(set), defaultdict(set), defaultdict(Counter)
        for annotation in payload['annotations']:
            image_id, category_id = int(annotation['image_id']), int(annotation['category_id'])
            if image_id not in wanted:
                continue
            present[category_id].add(image_id)
            if annotation.get('iscrowd', 0):
                crowds[category_id].add(image_id)
            else:
                counts[category_id][image_id] += 1
        catalog = []
        for category in categories:
            cid = int(category['id'])
            catalog.append({'category_id': cid, 'category_name': category['name'],
                            'present': sorted(present[cid]), 'absent': sorted(wanted - present[cid]),
                            'crowd_images': sorted(crowds[cid]),
                            'count_images': {str(k): sorted(im for im, n in counts[cid].items()
                                                          if n == k and im not in crowds[cid])
                                             for k in range(1, 5)}})
        catalogs[split] = catalog
        details = [{'category_id': c['category_id'], 'category_name': c['category_name'],
                    'positive_images': len(c['present']), 'negative_images': len(c['absent']),
                    'crowd_image_categories_excluded_from_counting': len(c['crowd_images']),
                    'count_group_sizes': {k: len(v) for k, v in c['count_images'].items()},
                    'possible_presence_pairs': capacity(c, 'presence'),
                    'possible_counting_pairs': capacity(c, 'counting')} for c in catalog]
        detail_rel = f'manifests/eligibility/grounding_pool_support_{split}.json'
        detail_hash = save_record(detail_rel, details)
        pool_summary = {'candidate_images': len(wanted),
                        'presence_eligible_categories': sum(capacity(c, 'presence') > 0 for c in catalog),
                        'counting_eligible_categories': sum(capacity(c, 'counting') > 0 for c in catalog),
                        'presence_unavailable_categories': [c['category_name'] for c in catalog if not capacity(c, 'presence')],
                        'counting_unavailable_categories': [c['category_name'] for c in catalog if not capacity(c, 'counting')],
                        'category_details_path': str(ROOT / detail_rel), 'category_details_sha256': detail_hash}
        assert pool_summary['presence_eligible_categories'] and pool_summary['counting_eligible_categories']
        report5b['pool_summary'][split] = pool_summary
        print(json.dumps({split: pool_summary}, indent=2), flush=True)
        del payload, present, crowds, counts
        gc.collect()
    assert source_ids['train2014'].isdisjoint(source_ids['val2014'])
    train_catalog = catalogs['train2014']
    training_pool_record = {'source_split': 'train2014', 'image_ids': sorted(training_ids),
                            'categories': train_catalog,
                            'presence_policy': 'Any category annotation is positive; no category annotation is negative',
                            'counting_policy': 'Non-crowd instance counts 1-4; exclude image-category if any crowd annotation exists'}
    report5b['grounding_training_pools_sha256'] = save_record(
        'manifests/selected/grounding_training_pools.json', training_pool_record)
    recipe = {'seed': 6304, 'rng': 'Independent numpy.random.default_rng(6304) for this allocation',
              'order': ['grounding_dev_presence', 'grounding_dev_counting',
                        'grounding_test_presence', 'grounding_test_counting',
                        'caption_dev_images', 'caption_dev_caption'],
              'category_sampling': 'Uniform among categories with remaining feasible pairs',
              'count_combinations': 'Uniform among available unordered combinations of different counts 1-4',
              'image_sampling': 'Uniform within selected category/label group',
              'pair_uniqueness': 'No duplicate question-type/category/unordered-image pair within a suite',
              'pair_order': 'Presence positive then negative; counting lower then higher count',
              'image_separation': 'All evaluation sets mutually image-disjoint; within-set reuse allowed',
              'train_pool_image_universe_changed': False,
              'question_rendering_and_visual_label_audit_pending': True}
    save_record('configs/grounding_pool_and_allocation_recipe.json', recipe)
    rng = np.random.default_rng(6304)
    ground = {}
    ground['dev'] = allocate_grounding(catalogs['val2014'], 'dev', {'presence': 12, 'counting': 12}, rng)
    ground['test'] = allocate_grounding(filter_catalog(catalogs['val2014'], image_set(ground['dev'])),
                                         'test', {'presence': 32, 'counting': 32}, rng)
    for suite in ('dev', 'test'):
        report5b['outputs'].append(save_rows(f'grounding_{suite}_pair_metadata.jsonl', ground[suite]))
    blocked = reserved | image_set(ground['dev']) | image_set(ground['test'])
    prior4_path = ROOT / 'logs/block_04b_20261031_155026_82d3640b.json'
    assert file_sha(prior4_path) == '35c8314906b0f5cdab93f998159f43b493fc2e111cc058cea957b2053dd23945'
    cap_info = json.loads(prior4_path.read_text())['caption_eligibility']['val2014']
    durable = Path(cap_info['index_path'])
    local = LOCAL / 'eligibility' / durable.name
    copy_checked(durable, local)
    assert file_sha(local) == cap_info['index_sha256']
    by_image = defaultdict(dict)
    with local.open() as stream:
        for line in stream:
            row = json.loads(line)
            image_id = int(row['image_id'])
            if image_id not in blocked and row['within_content_cap']:
                assert not row['empty_or_whitespace']
                old_row = by_image[image_id].get(row['caption'])
                if old_row is None or int(row['caption_id']) < int(old_row['caption_id']):
                    by_image[image_id][row['caption']] = row
    available = sorted(by_image)
    if len(available) < 96:
        raise RuntimeError('Insufficient caption-dev images after all evaluation exclusions.')
    chosen_ids = [available[int(i)] for i in rng.permutation(len(available))[:96]]
    caption_dev = []
    for image_id in sorted(chosen_ids):
        options = sorted(by_image[image_id].values(), key=lambda r: int(r['caption_id']))
        caption_dev.append({**options[int(rng.integers(len(options)))], 'experiment_split': 'dev'})
    report5b['outputs'].append(save_rows('captions_dev.jsonl', caption_dev))
    groups = {'training_union': training_ids,
              'natural_dev': image_set(natural['dev']), 'natural_test': image_set(natural['test']),
              'grounding_dev': image_set(ground['dev']), 'grounding_test': image_set(ground['test']),
              'captions_dev': set(chosen_ids)}
    names = list(groups)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            assert groups[left].isdisjoint(groups[right]), f'Image overlap: {left}, {right}'
    assert all(groups[n] <= source_ids['val2014'] for n in names if n != 'training_union')
    assert len(caption_dev) == len(set(chosen_ids)) == 96
    check_record = {'image_ids_by_set': {n: sorted(ids) for n, ids in groups.items()},
                    'all_sets_mutually_image_disjoint': True,
                    'grounding_pair_metadata_not_visually_audited': True,
                    'complete_dataset_frozen': False}
    report5b['global_image_allocation_sha256'] = save_record(
        'manifests/selected/global_image_allocation.json', check_record)
    evaluation_ids = sorted(set().union(*(groups[n] for n in names if n != 'training_union')))
    report5b['candidate_cache_image_ids_sha256'] = save_record(
        'manifests/selected/candidate_cache_image_ids.json',
        {'train2014': sorted(training_ids), 'val2014': evaluation_ids,
         'status': 'Candidate IDs only; label audit and question rendering pending'})
    report5b.update({'status': 'passed', 'all_sets_mutually_image_disjoint': True,
                     'allocation': {suite: {'pairs': len(rows), 'questions': 2 * len(rows),
                                            'type_counts': dict(Counter(r['question_type'] for r in rows)),
                                            'unique_images': len(image_set(rows))}
                                    for suite, rows in ground.items()},
                     'caption_dev_images': 96,
                     'candidate_cache_images': {'train2014': len(training_ids), 'val2014': len(evaluation_ids)},
                     'local_disk_free_GiB': round(shutil.disk_usage('/content').free / 2**30, 2)})
except Exception:
    report5b['status'] = 'failed'
    report5b['traceback'] = traceback.format_exc()
    save_record(f'logs/block_05b_{stamp}.json', report5b)
    print('BLOCK 5B FAILED; diagnostics and earlier allocations retained.', flush=True)
    raise

log_path = f'logs/block_05b_{stamp}.json'
digest = save_record(log_path, report5b)
print('\nBLOCK 5B COMPLETE — grounding pools and evaluation image allocation; label audit still pending.')
print(json.dumps(report5b, indent=2))
print('Saved log:', ROOT / log_path)
print('SHA256:', digest)
print('Keep accelerator None. Paste the complete output.')
