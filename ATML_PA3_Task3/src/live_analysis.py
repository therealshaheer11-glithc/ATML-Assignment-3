"""Frozen CPU scoring and question-paired analysis of the 250 saved live calls.

Requires NumPy 2.4.6 and the exact packages in task3_scorer_requirements.lock.
No inference, package changes, gold-based selection, or GPU imports.
"""
import argparse
import csv
import hashlib
import json
import logging
from collections import Counter
from importlib import metadata
from pathlib import Path

import numpy as np
try:
    from .scoring import MathScorer, group_revealed
except ImportError:
    from scoring import MathScorer, group_revealed

METHODS = {
    'long': ['long'],
    'parallel': ['parallel_1', 'parallel_2', 'parallel_3'],
    'refinement': ['parallel_1', 'revision'],
}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_text())

def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def csvwrite(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

class ScoringLog(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.context = None
        self.events = []
    def emit(self, record):
        self.events.append({'context': self.context, 'logger': record.name,
                            'level': record.levelname, 'message': record.getMessage()})

def main(snapshot, offline, out):
    assert np.__version__ == '2.4.6'
    out.mkdir(parents=True, exist_ok=True)
    frozen = read(offline / 'configs/task3_scorer_frozen.json')
    assert digest(offline / 'configs/task3_scorer_frozen.json') == '6e97a2379450ba34714d0dce027ce337364dc2ce438a49ec46c9b75b0f219467'
    assert digest(Path(__file__).with_name('scoring.py')) == frozen['specification']['module_sha256']
    versions = {name: metadata.version(name) for name in frozen['specification']['dependencies']}
    assert versions == frozen['specification']['dependencies']
    inputs = read(snapshot / 'live/task3_live_inputs_v1.json')
    rows = inputs['questions']
    splits = read(offline / 'manifests/task3_math_splits.json')
    assert [r['sample_id'] for r in rows] == splits['evaluation_ids'][:50]
    questions = read(offline / 'metadata/task3_math_questions.json')
    gold = read(offline / 'metadata/task3_math_reference_answers.json')
    protocol = read(snapshot / 'configs/MANUAL_REQUIREMENTS.json')['live']
    call_paths = list((snapshot / 'results/task3_live_generation_A002/calls').glob('*.json'))
    assert len(call_paths) == 250
    calls = {}
    for path in call_paths:
        envelope = read(path)
        record = envelope['record']
        canonical = (json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()
        assert hashlib.sha256(canonical).hexdigest() == envelope['record_sha256']
        key = (record['sample_id'], record['call_name'])
        assert key not in calls
        calls[key] = record
    logger = ScoringLog()
    library_logger = logging.getLogger('math_verify')
    library_logger.addHandler(logger)
    scorer = MathScorer()
    records = []
    flat_calls = []
    for row in rows:
        q = row['sample_id']
        assert row['initial_messages'][1]['content'] == questions[q]['question'] + protocol['problem_suffix']
        answers = {name: calls[q, name]['extraction']['answer'] for name in row['call_plan']['call_order']}
        # Final selector is completed before the reference is accessed for this question.
        logger.context = {'sample_id': q, 'phase': 'gold_free_parallel_grouping', 'answers': [answers[n] for n in METHODS['parallel']]}
        parallel_answers = [scorer.prepare(answers[n]) for n in METHODS['parallel']]
        groups, assignments = group_revealed(parallel_answers, scorer.equivalent)
        selected_index = max(groups, key=len)[0]
        selected = {'long': 'long', 'parallel': METHODS['parallel'][selected_index], 'refinement': 'revision'}
        reference = gold[q]['export_reference']
        assert reference == gold[q]['official_answer']
        logger.context = {'sample_id': q, 'phase': 'reference_parse', 'reference': reference}
        assert scorer.prepare(reference).kind == 'math_parsed', 'Invalid reference; stop instead of changing scorer'
        graded = {}
        call_info = {}
        for name in row['call_plan']['call_order']:
            r = calls[q, name]
            logger.context = {'sample_id': q, 'phase': 'grade', 'call': name, 'prediction': answers[name], 'reference': reference}
            correct = bool(scorer.grade(answers[name], reference))
            prepared = scorer.prepare(answers[name])
            graded[name] = correct
            info = {'sample_id': q, 'call': name, 'answer': answers[name], 'correct': correct,
                    'parse_kind': prepared.kind, 'parse_reason': prepared.reason,
                    'input_tokens': r['input_tokens'], **r['termination'],
                    'inference_seconds': r['inference_seconds'], 'call_wall_seconds': r['call_wall_seconds'],
                    'peak_allocated_bytes': r['peak_allocated_bytes'], 'peak_reserved_bytes': r['peak_reserved_bytes'],
                    'seed': r['seed'], 'session_id': r['session_id']}
            call_info[name] = info
            flat_calls.append(info)
        strategies = {}
        for method, names in METHODS.items():
            cc = [call_info[n] for n in names]
            final = call_info[selected[method]]
            strategies[method] = {
                'selected_call': selected[method], 'selected_answer': final['answer'], 'correct': final['correct'],
                'final_extraction_missing': final['answer'] is None,
                'any_component_missing': any(c['answer'] is None for c in cc),
                'any_component_capped': any(c['truncated_by_cap'] for c in cc),
                'component_missing_count': sum(c['answer'] is None for c in cc),
                'component_capped_count': sum(c['truncated_by_cap'] for c in cc),
                'input_tokens': sum(c['input_tokens'] for c in cc),
                'output_tokens': sum(c['output_tokens'] for c in cc),
                'retained_nontermination_tokens': sum(c['retained_nontermination_tokens'] for c in cc),
                'inference_seconds': sum(c['inference_seconds'] for c in cc),
                'call_wall_seconds': sum(c['call_wall_seconds'] for c in cc),
                'peak_allocated_bytes': max(c['peak_allocated_bytes'] for c in cc),
                'peak_reserved_bytes': max(c['peak_reserved_bytes'] for c in cc),
            }
            assert strategies[method]['output_tokens'] <= 3072
        transition = ('correct' if graded['parallel_1'] else 'wrong') + '_to_' + ('correct' if graded['revision'] else 'wrong')
        records.append({'sample_id': q, 'question': questions[q]['question'], 'reference': reference,
                        'parallel_groups': groups, 'parallel_assignments': assignments,
                        'parallel_group_call_order': METHODS['parallel'],
                        'parallel_oracle': any(graded[n] for n in METHODS['parallel']),
                        'initial_correct': graded['parallel_1'], 'revision_transition': transition,
                        'calls': call_info, 'strategies': strategies})
        print(json.dumps({'status': 'CPU QUESTION SCORED', 'sample_id': q, 'of': 50, 'completed': len(records)}), flush=True)
    library_logger.removeHandler(logger)
    assert len(records) == 50
    # Independent stream for the live question bootstrap; all contrasts share draws.
    ids = np.random.default_rng(np.random.SeedSequence([6304, 3, 2])).integers(0, 50, size=(1000, 50), dtype=np.int64)
    np.save(out / 'bootstrap_question_indices.npy', ids.astype(np.int32))
    write(out / 'bootstrap_question_ids.json', [r['sample_id'] for r in records])
    def stats(values):
        values = np.array(values, dtype=np.float64)
        low, high = np.percentile(values[ids].mean(axis=1), [2.5, 97.5], method='linear')
        return {'mean': float(values.mean()), 'low': float(low), 'high': float(high)}
    metrics = []
    comparisons = []
    for method, names in METHODS.items():
        values = [r['strategies'][method] for r in records]
        result = {'method': method, 'questions': 50, 'correct': sum(v['correct'] for v in values),
                  'accuracy_percent': stats([100 * v['correct'] for v in values]),
                  'final_extraction_missing_count': sum(v['final_extraction_missing'] for v in values),
                  'any_component_missing_questions': sum(v['any_component_missing'] for v in values),
                  'any_component_capped_questions': sum(v['any_component_capped'] for v in values),
                  'component_call_count': 50 * len(names),
                  'component_missing_calls': sum(v['component_missing_count'] for v in values),
                  'component_capped_calls': sum(v['component_capped_count'] for v in values)}
        for field in ['input_tokens', 'output_tokens', 'retained_nontermination_tokens', 'inference_seconds', 'call_wall_seconds', 'peak_allocated_bytes', 'peak_reserved_bytes']:
            result[field] = stats([v[field] for v in values])
            result[field]['total' if not field.startswith('peak') else 'maximum'] = sum(v[field] for v in values) if not field.startswith('peak') else max(v[field] for v in values)
        metrics.append(result)
    for a, b in [('parallel', 'long'), ('refinement', 'long'), ('refinement', 'parallel')]:
        for field in ['correct', 'input_tokens', 'output_tokens', 'inference_seconds']:
            scale = 100 if field == 'correct' else 1
            delta = [scale * (int(r['strategies'][a][field]) - int(r['strategies'][b][field])) if field == 'correct' else r['strategies'][a][field] - r['strategies'][b][field] for r in records]
            comparisons.append({'A': a, 'B': b, 'measure': 'accuracy_pp' if field == 'correct' else field, **stats(delta)})
    transitions = dict(Counter(r['revision_transition'] for r in records))
    for key in ['wrong_to_wrong', 'wrong_to_correct', 'correct_to_wrong', 'correct_to_correct']:
        transitions.setdefault(key, 0)
    assert sum(transitions.values()) == 50
    assert transitions['correct_to_correct'] + transitions['wrong_to_correct'] == next(m['correct'] for m in metrics if m['method'] == 'refinement')
    assert transitions['correct_to_correct'] + transitions['correct_to_wrong'] == sum(r['initial_correct'] for r in records)
    revision_delta = stats([100 * (int(r['strategies']['refinement']['correct']) - int(r['initial_correct'])) for r in records])
    examples = {}
    for category, transition in [('beneficial_revision', 'wrong_to_correct'), ('harmful_revision', 'correct_to_wrong')]:
        examples[category] = [r['sample_id'] for r in sorted(records, key=lambda r: r['sample_id']) if r['revision_transition'] == transition][:2]
    session_results = [read(p) for p in sorted((snapshot / 'logs').glob('gpu_*_result.json'))]
    unique_output = sum(c['output_tokens'] for c in flat_calls)
    charged_output = sum(m['output_tokens']['total'] for m in metrics)
    shared_output = sum(r['calls']['parallel_1']['output_tokens'] for r in records)
    assert charged_output == unique_output + shared_output
    assert len(flat_calls) == 250 and len(session_results) == 5 and sum(r['new_calls'] for r in session_results) == 250
    # Explicit accounting identity holds for tokens, input prefill, and timed inference.
    for field in ['input_tokens', 'inference_seconds', 'call_wall_seconds']:
        actual = sum(c[field] for c in flat_calls) + sum(r['calls']['parallel_1'][field] for r in records)
        assert np.isclose(sum(m[field]['total'] for m in metrics), actual, atol=1e-7)
    result = {
        'status': 'LIVE CPU ANALYSIS COMPLETE', 'questions': 50, 'unique_calls': 250,
        'scorer_id': frozen['scorer_id'], 'scorer_module_sha256': frozen['specification']['module_sha256'], 'versions': versions,
        'bootstrap': {'draws': 1000, 'unit': 'question', 'stream': 'SeedSequence([6304,3,2])', 'seed': 6304,
                      'paired': True, 'confidence': .95, 'method': 'pointwise percentile with linear interpolation', 'indices_saved': True},
        'metrics': metrics, 'paired_differences': comparisons, 'revision_transitions': transitions,
        'initial_correct': sum(r['initial_correct'] for r in records), 'revision_minus_initial_accuracy_pp': revision_delta,
        'beneficial_revision_initial_missing': sum(r['revision_transition'] == 'wrong_to_correct' and r['calls']['parallel_1']['answer'] is None for r in records),
        'beneficial_revision_initial_capped': sum(r['revision_transition'] == 'wrong_to_correct' and r['calls']['parallel_1']['truncated_by_cap'] for r in records),
        'parallel_oracle_correct': sum(r['parallel_oracle'] for r in records),
        'examples': examples, 'example_rule': 'First two distinct lexicographic question IDs per beneficial/harmful revision category; fewer if absent.',
        'scoring_warning_count': len(logger.events),
        'parse_kind_counts': dict(Counter(c['parse_kind'] for c in flat_calls)),
        'unique_generated_tokens': unique_output, 'strategy_charged_output_tokens': charged_output,
        'shared_initial_output_tokens_charged_to_both': shared_output,
        'unique_inference_seconds': sum(c['inference_seconds'] for c in flat_calls),
        'successful_session_elapsed_seconds': sum(r['elapsed_seconds'] for r in session_results),
        'session_results': session_results,
        'limits': [
            'Only 50 fixed questions and one set of independently seeded calls; pointwise intervals are broad and not corrected for multiple comparisons.',
            'Parallel denotes independent candidates generated serially at batch size 1. Runtime is summed measured inference, not concurrent serving latency.',
            'Shared initial input/output tokens and runtime charged to both parallel and refinement; revision input includes the complete initial answer and review instruction.',
            'Peak memory per strategy/question is the maximum of participating calls, not their sum; global maximum is also recorded.',
            'Inference includes synchronized prompt/tokenization, prefill, generation, transfer and decoding; excludes model loading, Drive saving and CPU grading.',
            'Session elapsed totals cover five successful generation sessions, including setup/staging/saving. Earlier failed integrity check and diagnostic allocations are preserved separately; total billed GPU time is not independently available.',
            'All missing and capped responses retained; final revision returned unconditionally. No gold-based choice between initial and revised answers.',
        ],
        'GPU_used_for_analysis': False, 'new_generations': 0,
    }
    write(out / 'analysis.json', result)
    write(out / 'question_results.json', records)
    write(out / 'scoring_warnings.json', logger.events)
    csvwrite(out / 'calls.csv', flat_calls)
    csvwrite(out / 'strategy_question_metrics.csv', [{'sample_id': r['sample_id'], 'method': method, **r['strategies'][method]} for r in records for method in METHODS])
    csvwrite(out / 'method_metrics.csv', [{'method': m['method'], 'correct': m['correct'], 'questions': 50, 'accuracy_percent': m['accuracy_percent']['mean'], 'accuracy_low': m['accuracy_percent']['low'], 'accuracy_high': m['accuracy_percent']['high'], **{field: m[field]['mean'] for field in ['input_tokens', 'output_tokens', 'inference_seconds', 'peak_allocated_bytes']}, 'missing_final': m['final_extraction_missing_count'], 'any_component_capped_questions': m['any_component_capped_questions']} for m in metrics])
    csvwrite(out / 'paired_differences.csv', comparisons)
    csvwrite(out / 'revision_transitions.csv', [{'sample_id': r['sample_id'], 'initial_correct': r['initial_correct'], 'revision_correct': r['strategies']['refinement']['correct'], 'transition': r['revision_transition']} for r in records])
    print(json.dumps({k: result[k] for k in ['status', 'metrics', 'revision_transitions', 'paired_differences', 'scoring_warning_count']}, indent=2), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('offline', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    main(args.snapshot, args.offline, args.out)
