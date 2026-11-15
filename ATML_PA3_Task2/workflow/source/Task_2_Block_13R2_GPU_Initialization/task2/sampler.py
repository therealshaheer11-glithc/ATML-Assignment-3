"""Complete-pair curriculum draws with independently serializable training RNG state."""
import copy
from itertools import combinations
import numpy as np
from .protocol import SEED, STAGES


class CurriculumSampler:
    def __init__(self, pool, natural_pairs, captions, caption_orders, question_bank, dataset_identity):
        self.pool = pool
        self.natural = natural_pairs
        self.captions = captions
        self.by_caption = {r['caption_id']: r for r in captions}
        self.caption_orders = [caption_id for order in caption_orders for caption_id in order]
        self.questions = {(r['category_id'], r['question_type']): r for r in question_bank}
        self.dataset_identity = dataset_identity
        self.rng = np.random.default_rng(SEED)
        self.caption_cursor = 0
        self.draw_counts = {stage: 0 for stage in STAGES}
        self.count_combinations = {
            c['category_id']: list(combinations([k for k in range(1, 5) if c['count_images'][str(k)]], 2))
            for c in pool['categories']
        }
        self.categories = {
            'presence': [c for c in pool['categories'] if c['present'] and c['absent']],
            'counting': [c for c in pool['categories'] if self.count_combinations[c['category_id']]],
        }
        if len(self.by_caption) != len(captions) or not natural_pairs:
            raise ValueError('Caption identities must be unique and natural pairs must be available.')
        if len(caption_orders) != 2 or any(sorted(order) != sorted(self.by_caption) for order in caption_orders):
            raise ValueError('Stage one requires the two saved complete caption orders.')
        if any(not self.categories[kind] for kind in self.categories):
            raise ValueError('Grounding pools do not support the curriculum.')

    def choose(self, items):
        return items[int(self.rng.integers(len(items)))]

    @staticmethod
    def caption_example(row):
        return {'source_split': 'train2014', 'image_id': row['image_id'],
            'question_id': f"caption:{row['caption_id']}", 'caption_id': row['caption_id'],
            'user_text': 'Describe this image.', 'target': row['caption']}

    def grounding_pair(self, kind):
        category = self.choose(self.categories[kind])
        if kind == 'presence':
            targets = ('yes', 'no')
            images = (self.choose(category['present']), self.choose(category['absent']))
        else:
            counts = self.choose(self.count_combinations[category['category_id']])
            targets = tuple(str(count) for count in counts)
            images = tuple(self.choose(category['count_images'][target]) for target in targets)
        question = self.questions[(category['category_id'], kind)]
        pair_id = f"training:{kind}:{category['category_id']}:{images[0]}:{images[1]}"
        return {'kind': kind, 'pair_id': pair_id, 'category_id': category['category_id'],
            'examples': [{'source_split': 'train2014', 'image_id': image_id, 'question_id': f'{pair_id}:{side}',
                'user_text': question['user_text'], 'target': target, 'canonical': target}
                for side, (image_id, target) in enumerate(zip(images, targets))]}

    def draw(self, stage):
        if stage not in STAGES:
            raise ValueError('Unknown curriculum stage.')
        if stage == 1:
            if self.caption_cursor + 2 > len(self.caption_orders):
                raise StopIteration('Both saved caption passes are complete.')
            ids = self.caption_orders[self.caption_cursor:self.caption_cursor + 2]
            result = {'kind': 'caption', 'examples': [self.caption_example(self.by_caption[cid]) for cid in ids]}
            self.caption_cursor += 2
        else:
            value = self.rng.random()
            cumulative = 0.0
            kind = STAGES[stage]['mixture'][-1][0]
            for component, probability in STAGES[stage]['mixture']:
                cumulative += probability
                if value < cumulative:
                    kind = component
                    break
            if kind in self.categories:
                result = self.grounding_pair(kind)
            elif kind == 'natural':
                pair = self.choose(self.natural)
                result = {'kind': kind, 'pair_id': pair['pair_id'],
                    'examples': [{**copy.deepcopy(example), 'source_split': 'train2014', 'user_text': pair['user_text']}
                        for example in pair['examples']]}
            else:
                result = {'kind': 'caption', 'examples': [self.caption_example(self.choose(self.captions)) for _ in range(2)]}
        self.draw_counts[stage] += 1
        return result

    def state_dict(self):
        return {'schema': 'atml-pa3-task2-curriculum-state-v1', 'dataset_identity': self.dataset_identity,
            'caption_cursor': self.caption_cursor, 'draw_counts': dict(self.draw_counts),
            'rng_state': copy.deepcopy(self.rng.bit_generator.state)}

    def load_state_dict(self, state):
        if state['schema'] != 'atml-pa3-task2-curriculum-state-v1' or state['dataset_identity'] != self.dataset_identity:
            raise ValueError('Sampler checkpoint belongs to another dataset or implementation.')
        cursor = state['caption_cursor']
        counts = {int(stage): count for stage, count in state['draw_counts'].items()}
        if type(cursor) is not int or cursor % 2 or not 0 <= cursor <= len(self.caption_orders):
            raise ValueError('Invalid caption cursor.')
        if set(counts) != set(STAGES) or any(type(count) is not int or count < 0 for count in counts.values()):
            raise ValueError('Invalid curriculum draw counters.')
        if counts[1] * 2 != cursor:
            raise ValueError('Caption cursor and stage-one draws disagree.')
        rng = np.random.default_rng()
        rng.bit_generator.state = copy.deepcopy(state['rng_state'])
        self.rng, self.caption_cursor, self.draw_counts = rng, cursor, counts
