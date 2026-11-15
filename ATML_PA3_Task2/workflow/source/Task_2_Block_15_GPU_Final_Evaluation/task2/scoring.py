"""Grounding and VQAv2 scores using the separately checksum-locked official normalizer."""
import numpy as np
from .protocol import SEED


def exact_match(prediction, target, normalize):
    return float(normalize(prediction) == normalize(target))


def vqa_soft_score(prediction, references, normalize):
    if len(references) != 10:
        raise ValueError('Natural VQA requires all ten original human references.')
    prediction = normalize(prediction)
    matches = [normalize(reference) == prediction for reference in references]
    total = sum(matches)
    return sum(min(1.0, (total - int(match)) / 3.0) for match in matches) / 10.0


def score_pair(pair, predictions, normalize, natural=False):
    if len(predictions) != 2 or len(pair['examples']) != 2:
        raise ValueError('Pair scoring requires both examples and predictions.')
    normalized = [normalize(prediction) for prediction in predictions]
    correct = [exact_match(prediction, example['canonical'], normalize)
        for prediction, example in zip(predictions, pair['examples'])]
    scores = correct
    if natural:
        scores = [vqa_soft_score(prediction, example['human_answers'], normalize)
            for prediction, example in zip(predictions, pair['examples'])]
    return {'example_scores': scores, 'mean_score': sum(scores) / 2,
        'both_canonical_correct': float(all(correct)), 'identical_answers': float(normalized[0] == normalized[1]),
        'empty_answers': sum(not text for text in normalized), 'answer_count': 2}


def paired_bootstrap_difference(left_by_pair, right_by_pair, samples=4000, seed=SEED):
    if not left_by_pair or set(left_by_pair) != set(right_by_pair):
        raise ValueError('Conditions must contain the same nonempty set of complete pair IDs.')
    if type(samples) is not int or samples < 1:
        raise ValueError('Bootstrap sample count must be positive.')
    ids = sorted(left_by_pair)
    differences = np.asarray([left_by_pair[pid] - right_by_pair[pid] for pid in ids], dtype=np.float64)
    if not np.isfinite(differences).all():
        raise ValueError('Bootstrap scores must be finite.')
    indices = np.random.default_rng(seed).integers(len(ids), size=(samples, len(ids)))
    draws = differences[indices].mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return {'difference': float(differences.mean()), 'ci95_low': float(low), 'ci95_high': float(high),
        'pairs': len(ids), 'bootstrap_samples': samples, 'seed': seed}
