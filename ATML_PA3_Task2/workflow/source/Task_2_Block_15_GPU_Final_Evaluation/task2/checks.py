"""Regression checks for the approved curriculum, scoring and final dataset."""
import copy
import json
import math
import sys
from .dataset import load_dataset, validate_dataset
from .protocol import STAGES, learning_rate
from .sampler import CurriculumSampler
from .scoring import paired_bootstrap_difference, score_pair, vqa_soft_score
from .sourcecheck import catalog_from_instances


def make_sampler(data):
    return CurriculumSampler(data['pool'], data['natural']['train'], data['captions']['train'],
        data['caption_orders'], data['question_bank'], data['identity'])


def run_checks(data, normalize=None):
    summary = validate_dataset(data, normalize)
    checks = ['Final data sizes, quotas, all 246 accepted positions, 110 exclusions, calibration and split isolation pass']
    for stage, config in STAGES.items():
        for peak in [config['connector_lr']] + ([config['lora_lr']] if config['lora_lr'] is not None else []):
            assert math.isclose(learning_rate(peak, stage, 1), peak / config['warmup'])
            assert math.isclose(learning_rate(peak, stage, config['warmup']), peak)
            assert math.isclose(learning_rate(peak, stage, config['updates']), peak * 0.1)
    checks.append('All connector and LoRA schedule endpoints match the approved one-based convention')
    sampler = make_sampler(data)
    original_rng = sampler.state_dict()['rng_state']
    observed = []
    for _ in range(18000):
        draw = sampler.draw(1)
        assert draw['kind'] == 'caption' and len(draw['examples']) == 2
        observed.extend(example['caption_id'] for example in draw['examples'])
    assert observed == data['caption_orders'][0] + data['caption_orders'][1]
    assert sampler.state_dict()['rng_state'] == original_rng
    try:
        sampler.draw(1)
    except StopIteration:
        pass
    else:
        raise AssertionError('Stage one exceeded two complete caption passes')
    checks.append('Stage one emits exactly the two saved caption passes and consumes no curriculum RNG')
    kinds = set()
    for _ in range(128):
        draw = sampler.draw(2)
        kinds.add(draw['kind'])
        assert len(draw['examples']) == 2
        if draw['kind'] != 'caption':
            assert draw['examples'][0]['target'] != draw['examples'][1]['target']
    assert kinds == {'presence', 'counting', 'caption'}
    checkpoint = json.loads(json.dumps(sampler.state_dict()))
    expected = [sampler.draw(2) for _ in range(32)] + [sampler.draw(3) for _ in range(128)]
    assert {draw['kind'] for draw in expected[32:]} == {'natural', 'presence', 'counting', 'caption'}
    restored = make_sampler(data)
    restored.load_state_dict(checkpoint)
    actual = [restored.draw(2) for _ in range(32)] + [restored.draw(3) for _ in range(128)]
    assert actual == expected and restored.state_dict() == sampler.state_dict()
    checks.append('Complete-pair stochastic draws and exact JSON checkpoint/resume persist across stages two and three')
    before = restored.state_dict()
    invalid = copy.deepcopy(before)
    invalid['dataset_identity'] = 'another-dataset'
    try:
        restored.load_state_dict(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError('Sampler accepted another dataset identity')
    assert restored.state_dict() == before
    checks.append('Sampler rejects another dataset identity without changing its live state')
    normalize_test = normalize if normalize is not None else lambda text: text.strip().lower()
    for matches, expected_score in [(0, 0.0), (1, 0.3), (2, 0.6), (3, 0.9), (4, 1.0), (10, 1.0)]:
        references = ['yes'] * matches + ['no'] * (10 - matches)
        assert math.isclose(vqa_soft_score('yes', references, normalize_test), expected_score)
    natural_pair = {'examples': [
        {'canonical': 'yes', 'human_answers': ['yes'] * 8 + ['no'] * 2},
        {'canonical': 'no', 'human_answers': ['no'] * 8 + ['yes'] * 2}]}
    result = score_pair(natural_pair, ['no', ''], normalize_test, natural=True)
    assert math.isclose(result['mean_score'], 0.3) and result['both_canonical_correct'] == 0
    assert result['empty_answers'] == 1 and result['answer_count'] == 2
    checks.append('VQAv2 uses ten leave-one-out references, with canonical pair accuracy and empty outputs handled separately')
    left, right = {'a': 0.5, 'b': 1.0}, {'a': 0.0, 'b': 0.5}
    boot = paired_bootstrap_difference(left, right)
    assert boot['difference'] == boot['ci95_low'] == boot['ci95_high'] == 0.5
    assert boot == paired_bootstrap_difference(left, right) and boot['bootstrap_samples'] == 4000
    checks.append('Paired bootstrap resamples complete matching pair IDs reproducibly at seed 6304')
    fixture = {'images': [{'id': i} for i in (1, 2, 3)], 'categories': [{'id': 1, 'name': 'person'}],
        'annotations': [{'image_id': 1, 'category_id': 1, 'iscrowd': 1},
            {'image_id': 1, 'category_id': 1, 'iscrowd': 0},
            {'image_id': 2, 'category_id': 1, 'iscrowd': 0},
            {'image_id': 2, 'category_id': 1, 'iscrowd': 0}]}
    catalog, _ = catalog_from_instances(fixture, (1, 2, 3))
    assert catalog[0]['present'] == [1, 2] and catalog[0]['absent'] == [3]
    assert catalog[0]['crowd_images'] == [1] and catalog[0]['count_images']['2'] == [2]
    assert catalog[0]['count_images']['1'] == []
    checks.append('Independent COCO reconstruction preserves category absence and excludes any crowd-bearing image/category from counting')
    if normalize is not None:
        assert normalize('The two cats.') == '2 cats' and normalize('1,000') == '1000'
        assert normalize('3.14') == '3.14' and normalize('dont') == "don't"
        checks.append('Checksum-locked official normalizer passes punctuation, article, numeric and contraction checks')
    summary.pop('image_groups')
    return {'status': 'passed', 'checks': checks, 'dataset': summary,
        'official_normalizer_checked': normalize is not None, 'gpu_used': False, 'training_started': False}


if __name__ == '__main__':
    print(json.dumps(run_checks(load_dataset(sys.argv[1])), indent=2))
