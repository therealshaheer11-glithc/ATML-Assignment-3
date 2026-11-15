"""Reconstruct grounding eligibility from unchanged original COCO instances."""
from collections import Counter, defaultdict


def catalog_from_instances(payload, image_ids):
    wanted = set(image_ids)
    images = {int(image['id']): image for image in payload['images'] if int(image['id']) in wanted}
    assert set(images) == wanted
    present, crowd, counts = defaultdict(set), defaultdict(set), defaultdict(Counter)
    for annotation in payload['annotations']:
        image_id, category_id = int(annotation['image_id']), int(annotation['category_id'])
        if image_id not in wanted:
            continue
        present[category_id].add(image_id)
        if annotation.get('iscrowd', 0):
            crowd[category_id].add(image_id)
        else:
            counts[category_id][image_id] += 1
    categories = []
    for category in sorted(payload['categories'], key=lambda c: int(c['id'])):
        cid = int(category['id'])
        categories.append({'category_id': cid, 'category_name': category['name'],
            'present': sorted(present[cid]), 'absent': sorted(wanted - present[cid]),
            'crowd_images': sorted(crowd[cid]),
            'count_images': {str(count): sorted(image for image, n in counts[cid].items()
                if n == count and image not in crowd[cid]) for count in range(1, 5)}})
    return categories, images


def verify_grounding_pairs(pairs, categories):
    by_category = {c['category_id']: c for c in categories}
    for pair in pairs:
        category = by_category[pair['category_id']]
        assert pair['category_name'] == category['category_name']
        for example in pair['examples']:
            target = example['target']
            allowed = category['present'] if target == 'yes' else category['absent'] if target == 'no' else category['count_images'][target]
            assert example['image_id'] in allowed, (pair['pair_id'], example['image_id'], target)


def verify_training_pool(pool, reconstructed):
    blocked = defaultdict(set)
    for record in pool['eligibility_revision']['quarantined_image_categories']:
        blocked[record['category_id']].add(record['image_id'])
    assert len(reconstructed) == len(pool['categories']) == 80
    for source, final in zip(reconstructed, pool['categories']):
        assert source['category_id'] == final['category_id'] and source['category_name'] == final['category_name']
        excluded = blocked[source['category_id']]
        for role in ('present', 'absent', 'crowd_images'):
            assert final[role] == [image for image in source[role] if image not in excluded]
        for count in source['count_images']:
            assert final['count_images'][count] == [image for image in source['count_images'][count] if image not in excluded]
