"""Stock lm-eval tasks on frozen local splits; no rewritten prompts or scorers."""
import copy
import gzip
import hashlib
import json
from pathlib import Path

from evaluation_records import COUNTS


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_task(root, name):
    from datasets import Dataset, DatasetDict
    from lm_eval.api.task import ConfigurableTask
    from lm_eval.tasks import TaskManager
    from lm_eval.utils import load_yaml_config
    root = Path(root)
    manifest = json.loads((root / 'manifests/BENCHMARK_FILES.json').read_text())[name]
    path = root / manifest['path']
    assert digest(path) == manifest['sha256']
    with gzip.open(path, 'rt') as stream:
        rows = [json.loads(line) for line in stream]
    assert len(rows) == manifest['count'] == COUNTS[name]
    assert [row['source_index'] for row in rows] == list(range(COUNTS[name]))
    manager = TaskManager()
    yaml_path = Path(manager.task_index[name]['yaml_path'])
    config = load_yaml_config(str(yaml_path), mode='full')
    source_config = copy.deepcopy(config)
    config['num_fewshot'] = 0
    if name == 'gsm8k_cot':
        config['generation_kwargs'] = copy.deepcopy(config['generation_kwargs'])
        config['generation_kwargs'].update(do_sample=False, max_gen_toks=1024)
    split = 'test' if name in ('gsm8k_cot', 'arc_easy') else 'validation'
    assert (config.get('test_split') or config.get('validation_split')) == split

    class FrozenLocalTask(ConfigurableTask):
        def download(self, *args, **kwargs):
            # Only data location changes. Zero-shot sampling cannot read a train split.
            self.dataset = DatasetDict({split: Dataset.from_list([r['document'] for r in rows])})

        def fewshot_docs(self):
            if self.config.num_fewshot != 0:
                raise RuntimeError('The PA requires zero-shot evaluation')
            return None

    task = FrozenLocalTask(config=config)
    task.set_fewshot_seed(6304)
    assert len(task.eval_docs) == COUNTS[name]
    return task, source_config, yaml_path


def chat_formatter(tokenizer):
    def render(messages, **kwargs):
        return tokenizer.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
    return render


def request_for(task, name, source_index, document, tokenizer):
    chat = name == 'gsm8k_cot'
    formatter = chat_formatter(tokenizer) if chat else None
    context = task.fewshot_context(document, num_fewshot=0, system_instruction=None,
        apply_chat_template=chat, fewshot_as_multiturn=False,
        chat_template=formatter, gen_prefix=task.doc_to_prefix(document))
    requests = task.construct_requests(doc=document, ctx=context,
        metadata=(name, source_index, 1), apply_chat_template=chat, chat_template=formatter)
    if not isinstance(requests, list):
        requests = [requests]
    assert requests and all(req.doc_id == source_index for req in requests)
    return context, requests


def score(task, requests, responses):
    """Use the unmodified stock filters AND stock per-document metric functions."""
    assert len(requests) == len(responses)
    for request, response in zip(requests, responses):
        request.resps = [response]
        request.filtered_resps = {}
    task._instances = requests
    task.apply_filters()
    filters = list(requests[0].filtered_resps)
    assert all(set(r.filtered_resps) == set(filters) for r in requests)
    scores = {}
    filtered = {}
    for filter_name in filters:
        values = [req.filtered_resps[filter_name] for req in requests]
        filtered[filter_name] = values
        values_scored = task.process_results(requests[0].doc, values)
        for metric, value in values_scored.items():
            scores[metric + ',' + filter_name] = float(value)
    return scores, filtered


def serialized_requests(requests):
    return [{'request_type': r.request_type, 'arguments': r.args, 'idx': r.idx}
            for r in requests]


def scoring_checks(task, name, tokenizer):
    import numpy as np
    document = task.eval_docs[0]
    _, requests = request_for(task, name, 0, document, tokenizer)
    target = task.doc_to_target(document)
    if name == 'gsm8k_cot':
        good, _ = score(task, requests, ['The answer is ' + str(target) + '.'])
        empty, _ = score(task, requests, [''])
        assert good['exact_match,flexible-extract'] == 1.
        assert empty['exact_match,flexible-extract'] == 0.
        # Characteristic stock behavior: flexible-extract selects the final number.
        final, extracted = score(task, requests, ['Earlier 987654321; final ' + str(target)])
        assert final['exact_match,flexible-extract'] == 1.
        return {'stock_correct_answer_scores_one': True, 'empty_answer_scores_zero': True,
                'stock_flexible_filter_uses_final_number': True}
    choices = task.doc_to_choice(document)
    gold = int(target) if not isinstance(target, str) else choices.index(target)
    good_responses = [(-1000., False)] * len(choices)
    good_responses[gold] = (0., True)
    good, _ = score(task, requests, good_responses)
    assert good['acc,none'] == 1.
    wrong = (gold + 1) % len(choices)
    wrong_responses = [(-1000., False)] * len(choices)
    wrong_responses[wrong] = (0., True)
    bad, _ = score(task, requests, wrong_responses)
    assert bad['acc,none'] == 0.
    checks = {'stock_correct_choice_scores_one': True, 'stock_wrong_choice_scores_zero': True}
    if name in ('hellaswag', 'arc_easy'):
        ll = np.arange(-len(choices), 0, dtype=float)
        actual, _ = score(task, requests, [(float(v), False) for v in ll])
        expected = float(int(np.argmax(ll / np.array([len(c) for c in choices]))) == gold)
        assert actual['acc_norm,none'] == expected
        checks['stock_character_length_normalization_retained'] = True
    return checks
