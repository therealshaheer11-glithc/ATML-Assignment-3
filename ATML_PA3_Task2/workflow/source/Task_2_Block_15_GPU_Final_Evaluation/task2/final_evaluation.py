"""Frozen final-test controls and prompt-only greedy evaluation; no training."""
import hashlib
import json
import math

CONDITIONS = ('correct', 'complementary', 'mismatched', 'no_image')


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()


def build_plan(data):
    """Seeded shuffled cycle of OTHER PAIRS, independent of answers/predictions."""
    import numpy as np
    rng = np.random.default_rng(6304)
    rows = []
    for suite, count in (('grounding', 64), ('natural', 96)):
        pairs = sorted(data[suite]['test'], key=lambda p: p['pair_id'])
        if len(pairs) != count or len({p['pair_id'] for p in pairs}) != count:
            raise ValueError('Final suite pair quota differs.')
        order = rng.permutation(len(pairs)).tolist()
        partner = {order[i]: order[(i + 1) % len(order)] for i in range(len(order))}
        for index, pair in enumerate(pairs):
            other = pairs[partner[index]]
            original_images = {e['image_id'] for e in pair['examples']}
            for side, example in enumerate(pair['examples']):
                mismatch = other['examples'][side]['image_id']
                if mismatch in original_images or other['pair_id'] == pair['pair_id']:
                    raise ValueError('Mismatched mapping contains an original pair image.')
                row = {'suite': suite, 'pair_id': pair['pair_id'], 'side': side,
                    'source_split': pair['source_split'], 'image_id': example['image_id'],
                    'partner_image_id': pair['examples'][1-side]['image_id'],
                    'mismatched_image_id': mismatch, 'mismatched_pair_id': other['pair_id'],
                    'question': pair['question'], 'user_text': pair['user_text'],
                    'question_id': example['question_id'], 'target': example['target'],
                    'canonical': example['canonical'],
                    'partner_canonical': pair['examples'][1-side]['canonical'],
                    'question_type': pair.get('question_type', pair.get('answer_type'))}
                if suite == 'natural':
                    row.update(human_answers=example['human_answers'],
                        partner_human_answers=pair['examples'][1-side]['human_answers'])
                rows.append(row)
    return {'schema': 'atml-pa3-task2-final-test-plan-v1', 'dataset_identity': data['identity'],
        'seed': 6304, 'conditions': list(CONDITIONS), 'rows': rows,
        'mismatch_rule': 'Sort pairs by ID within grounding then natural; independent default_rng(6304); shuffled pair cycle to next pair, same side. Excludes both original images, bijective within suite; no label/prediction selection.',
        'original_references_retained_in_all_conditions': True,
        'generation': {'batch_size': 8, 'max_new_tokens': 16, 'do_sample': False,
            'num_beams': 1, 'enable_thinking': False, 'eos_token_id': 151645,
            'pad_token_id': 151643, 'use_cache': True},
        'reference_NLL': 'Original answer content tokens only, EOT excluded; equal example weighting',
        'generation_timing': 'Synchronized wall time for connector, prompt construction and batched greedy generation; excludes feature staging, model loading, teacher-forced NLL, decoding strings and saving; divided by batch answers; no warmup excluded.'}


def batches(plan):
    for suite in ('grounding', 'natural'):
        rows = [r for r in plan['rows'] if r['suite'] == suite]
        for condition in CONDITIONS:
            for start in range(0, len(rows), 8):
                batch = rows[start:start+8]
                if len(batch) != 8:
                    raise ValueError('Every generation batch must contain eight complete-pair examples.')
                yield f'{suite}_{condition}_{start:04d}', condition, batch


def image_keys(rows, condition):
    if condition not in CONDITIONS:
        raise ValueError('Unknown image condition.')
    if condition == 'no_image':
        return None
    field = {'correct': 'image_id', 'complementary': 'partner_image_id',
        'mismatched': 'mismatched_image_id'}[condition]
    return [(row['source_split'], row[field]) for row in rows]


def generate_batch(model, connector, records, features, tokenizer):
    import torch
    from transformers import GenerationConfig, LogitsProcessor, LogitsProcessorList
    from .text import splice_batch
    base = model.get_base_model()
    class FiniteLogits(LogitsProcessor):
        def __call__(self, input_ids, scores):
            if not torch.isfinite(scores).all():
                raise FloatingPointError('Nonfinite logits during final greedy generation.')
            return scores
    device = base.get_input_embeddings().weight.device
    if model.training or connector.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Evaluation model must be frozen and in eval mode.')
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    import time
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
        visual = None if features is None else connector(features, base.get_input_embeddings().weight.dtype)
        batch = splice_batch(records, base.get_input_embeddings(), visual, include_targets=False, padding_side='left')
        config = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=16,
            eos_token_id=151645, pad_token_id=151643, bos_token_id=None, use_cache=True,
            return_dict_in_generate=True, output_scores=False)
        generated = base.generate(inputs_embeds=batch['inputs_embeds'], attention_mask=batch['attention_mask'],
            position_ids=batch['position_ids'], generation_config=config, logits_to_keep=1,
            logits_processor=LogitsProcessorList([FiniteLogits()]))
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    ids = generated.sequences.cpu().tolist()
    if len(ids) != len(records) or any(not 1 <= len(row) <= 16 for row in ids):
        raise ValueError('Generation must return only 1–16 new token IDs, without prompt tokens.')
    answers = []
    for row in ids:
        ended = 151645 in row
        end = row.index(151645) if ended else len(row)
        content = row[:end]
        kept = row[:end+1] if ended else row
        answers.append({'prediction': tokenizer.decode(content, skip_special_tokens=True,
                clean_up_tokenization_spaces=False),
            'raw_generation': tokenizer.decode(kept, skip_special_tokens=False, clean_up_tokenization_spaces=False),
            'generated_ids': kept, 'generated_content_ids': content,
            'finish_reason': 'eot' if ended else 'length', 'truncated': not ended,
            'generated_tokens': len(kept), 'generation_seconds_per_answer': seconds/len(records)})
    return answers, seconds


def validate_chunk(chunk, recipe_sha, key, condition, expected):
    if chunk['recipe_sha256'] != recipe_sha or chunk['batch_key'] != key or chunk['condition'] != condition:
        raise ValueError('Saved batch belongs to another evaluation recipe or batch.')
    rows = chunk['rows']
    if len(rows) != len(expected):
        raise ValueError('Saved batch row count differs.')
    for row, source in zip(rows, expected):
        if any(row.get(k) != value for k, value in source.items()):
            raise ValueError('Saved batch changed original row metadata/references.')
        keys = image_keys([source], condition)
        if row['input_image_id'] != (None if keys is None else keys[0][1]):
            raise ValueError('Saved batch used another image condition.')
        if row['condition'] != condition or not isinstance(row['prediction'], str):
            raise ValueError('Saved prediction or condition is invalid.')
        if not math.isfinite(row['reference_NLL']) or row['reference_NLL'] < 0 or row['reference_tokens'] < 1:
            raise ValueError('Saved reference NLL is invalid.')
        if not math.isfinite(row['generation_seconds_per_answer']) or row['generation_seconds_per_answer'] <= 0:
            raise ValueError('Saved generation timing is invalid.')
        ids = row['generated_ids']
        if not 1 <= len(ids) <= 16 or any(type(i) is not int or not 0 <= i < 151936 for i in ids):
            raise ValueError('Saved generated token IDs differ from the approved bounds.')
        if row['generated_tokens'] != len(ids):
            raise ValueError('Saved generated token count differs.')
        ended = ids[-1] == 151645
        if 151645 in ids[:-1] or row['finish_reason'] != ('eot' if ended else 'length') or row['truncated'] != (not ended):
            raise ValueError('Saved termination flags disagree with token IDs.')
        if not ended and len(ids) != 16:
            raise ValueError('A nonterminated generation ended before the fixed cap.')
        if row['generated_content_ids'] != (ids[:-1] if ended else ids):
            raise ValueError('Saved content IDs disagree with EOT exclusion.')
    if not math.isfinite(chunk['generation_seconds']) or chunk['generation_seconds'] <= 0:
        raise ValueError('Saved batch generation timing is invalid.')
    if not math.isclose(sum(r['generation_seconds_per_answer'] for r in rows), chunk['generation_seconds'], rel_tol=1e-9):
        raise ValueError('Batch and per-answer timing disagree.')
    return rows
