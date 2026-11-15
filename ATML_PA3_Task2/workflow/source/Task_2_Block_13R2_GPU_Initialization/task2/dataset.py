"""Read and validate the final versioned dataset without changing original selections."""
from collections import Counter
from pathlib import Path
import hashlib
import json
from .protocol import vqa_user_text


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_bytes())


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def pair_images(rows):
    return {example['image_id'] for row in rows for example in row['examples']}


def load_dataset(folder):
    folder = Path(folder)
    manifest = read_json(folder / 'dataset_manifest.json')
    for entry in manifest['files']:
        relative = Path(entry['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Invalid dataset member path.')
        path = folder / relative
        if path.stat().st_size != entry['bytes'] or file_sha(path) != entry['sha256']:
            raise ValueError('Dataset member differs from its manifest: ' + str(relative))
    selected = folder / 'manifests/selected'
    return {
        'manifest': manifest, 'identity': file_sha(folder / 'dataset_manifest.json'),
        'pool': read_json(selected / 'grounding_training_pools.json'),
        'grounding': {split: read_rows(selected / f'grounding_{split}.jsonl') for split in ('training_audit', 'dev', 'test')},
        'natural': {split: read_rows(selected / f'natural_{split}_formatted.jsonl') for split in ('train', 'dev', 'test')},
        'captions': {split: read_rows(selected / f'captions_{split}.jsonl') for split in ('train', 'dev')},
        'caption_orders': read_json(selected / 'stage1_caption_epoch_orders.json')['epoch_caption_ids'],
        'calibration': read_json(folder / 'configs/calibration_sample_ids.json'),
        'question_bank': read_json(folder / 'configs/grounding_question_bank.json'),
        'reviews': read_rows(folder / 'provenance/human_review_ledger.jsonl'),
    }


def validate_dataset(data, normalize=None):
    pool, ground, natural, captions = data['pool'], data['grounding'], data['natural'], data['captions']
    universe = set(pool['image_ids'])
    assert len(universe) == len(pool['image_ids']) == 8733
    assert len(pool['categories']) == 80
    assert sum(bool(c['present'] and c['absent']) for c in pool['categories']) == 80
    assert sum(sum(bool(v) for v in c['count_images'].values()) >= 2 for c in pool['categories']) == 78
    blocked = {(r['image_id'], r['category_id']) for r in pool['eligibility_revision']['quarantined_image_categories']}
    assert len(blocked) == 110
    for category in pool['categories']:
        groups = [category[role] for role in ('present', 'absent', 'crowd_images')] + list(category['count_images'].values())
        assert all(set(group) <= universe and len(group) == len(set(group)) for group in groups)
        assert all((image, category['category_id']) not in blocked for group in groups for image in group)
    assert len(captions['train']) == 18000 and len(captions['dev']) == 96
    caption_ids = [r['caption_id'] for r in captions['train']]
    assert len(set(caption_ids)) == 18000
    per_image = Counter(r['image_id'] for r in captions['train'])
    assert len(per_image) == 6000 and set(per_image.values()) == {3}
    assert len({(r['image_id'], r['caption']) for r in captions['train']}) == 18000
    assert len(data['caption_orders']) == 2
    assert all(sorted(order) == sorted(caption_ids) for order in data['caption_orders'])
    assert len({r['image_id'] for r in captions['dev']}) == 96
    for rows in captions.values():
        assert all(r['caption'].strip() and r['within_content_cap'] and r['content_tokens'] <= 64 for r in rows)
    assert universe == set(per_image) | pair_images(natural['train'])
    quotas = {'train': {'yes/no': 640, 'number': 256, 'other': 640},
        'dev': {'yes/no': 12, 'number': 8, 'other': 12}, 'test': {'yes/no': 36, 'number': 24, 'other': 36}}
    for split, rows in natural.items():
        assert Counter(r['answer_type'] for r in rows) == quotas[split]
        assert len({r['pair_id'] for r in rows}) == len(rows)
        for pair in rows:
            assert pair['user_text'] == vqa_user_text(pair['question'])
            assert pair['question_content_tokens'] <= 80 and pair['user_content_tokens'] <= 80
            assert len(pair['examples']) == 2
            assert pair['examples'][0]['image_id'] != pair['examples'][1]['image_id']
            assert pair['examples'][0]['canonical'] != pair['examples'][1]['canonical']
            for example in pair['examples']:
                assert example['agreement'] >= 8 and len(example['human_answers']) == 10
                assert example['answer_ids'] == list(range(1, 11)) and example['target_content_tokens'] <= 16
                target_index = example['target_answer_id'] - 1
                assert example['target'] == example['human_answers'][target_index]
                if normalize is not None:
                    normalized = [normalize(answer) for answer in example['human_answers']]
                    assert Counter(normalized).most_common(1)[0] == (example['canonical'], example['agreement'])
                    assert normalize(example['target']) == example['canonical']
                    assert normalized.index(example['canonical']) == target_index
    expected_sizes = {'training_audit': 158, 'dev': 24, 'test': 64}
    for split, rows in ground.items():
        assert len(rows) == len({r['pair_id'] for r in rows}) == expected_sizes[split]
        quota = {'presence': 80, 'counting': 78} if split == 'training_audit' else {'presence': expected_sizes[split] // 2, 'counting': expected_sizes[split] // 2}
        assert Counter(r['question_type'] for r in rows) == quota
        for pair in rows:
            assert pair['user_text'] == vqa_user_text(pair['question'])
            assert pair['examples'][0]['image_id'] != pair['examples'][1]['image_id']
            assert pair['examples'][0]['target'] != pair['examples'][1]['target']
            assert pair['question_content_tokens'] <= 80 and pair['user_content_tokens'] <= 80
    assert pair_images(ground['training_audit']) <= universe
    categories = {c['category_id']: c for c in pool['categories']}
    for pair in ground['training_audit']:
        category = categories[pair['category_id']]
        for example in pair['examples']:
            target = example['target']
            allowed = category['present'] if target == 'yes' else category['absent'] if target == 'no' else category['count_images'][target]
            assert example['image_id'] in allowed
    groups = {'training_union': universe, 'captions_dev': {r['image_id'] for r in captions['dev']},
        'natural_dev': pair_images(natural['dev']), 'natural_test': pair_images(natural['test']),
        'grounding_dev': pair_images(ground['dev']), 'grounding_test': pair_images(ground['test'])}
    names = list(groups)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            assert groups[left].isdisjoint(groups[right]), (left, right)
    reviews = data['reviews']
    assert len(reviews) == len({r['original_pair_id'] for r in reviews}) == 246
    assert all(r['human_judgment']['label_status'] == 'accept' for r in reviews)
    assert {r['final_pair_id'] for r in reviews} == {p['pair_id'] for rows in ground.values() for p in rows}
    calibration = data['calibration']
    assert len(calibration['image_ids']) == len(set(calibration['image_ids'])) == 256
    assert set(calibration['image_ids']) <= universe
    assert len(calibration['caption_ids']) == len(set(calibration['caption_ids'])) == 256
    assert set(calibration['caption_ids']) <= set(caption_ids)
    return {'audit_accepts': 246, 'training_images': len(universe),
        'evaluation_unique_images': sum(len(groups[name]) for name in names if name != 'training_union'),
        'cache_images': len(set().union(*groups.values())), 'image_groups': {name: sorted(ids) for name, ids in groups.items()},
        'grounding_unique_images': {split: len(pair_images(rows)) for split, rows in ground.items()},
        'human_review_origins': dict(Counter(r['human_judgment']['review_origin'] for r in reviews)),
        'presence_categories': 80, 'counting_categories': 78}
