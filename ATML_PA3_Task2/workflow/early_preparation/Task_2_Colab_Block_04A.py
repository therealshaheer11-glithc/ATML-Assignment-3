# Block 4A: official answer normalization and natural-pair eligibility. CPU only.
# No experiment subset selection or image/model-weight downloads.
import os, re, ast, json, uuid, hashlib, shutil, traceback, urllib.request, warnings, gc
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import Counter
from types import SimpleNamespace
from functools import lru_cache

required = ('ROOT', 'LOCAL', 'save_record', 'file_sha', 'copy_checked', 'tokenizer')
if any(name not in globals() for name in required):
    raise RuntimeError('Required notebook state missing. Stop; do not restart Block 1.')
previous = ROOT / 'logs/block_03b_20261031_153258_5997ff44.json'
assert file_sha(previous) == 'c43bd46506db8a534fe478e6f72fb385bbc6b2370c1efec5f4b48164f3fbdf5d'
if os.environ.get('COLAB_GPU', '0') not in ('', '0'):
    print('CPU task: requesting release of allocated GPU.', flush=True)
    try:
        runtime.unassign()
    finally:
        raise RuntimeError('Reconnect with accelerator None.')

stamp = datetime.now(ZoneInfo('Asia/Karachi')).strftime('%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:8]
report4 = {'attempt': stamp, 'status': 'started', 'gpu_used': False,
           'experiment_subsets_selected': False, 'final_test_frozen': False, 'splits': {}}
annotation_dir = LOCAL / 'annotations'
work4 = LOCAL / 'eligibility'
work4.mkdir(parents=True, exist_ok=True)

try:
    # Verify the exact raw JSON files against the saved archive receipts.
    for receipt_path in sorted((ROOT / 'manifests/annotation_downloads').glob('*.json')):
        receipt = json.loads(receipt_path.read_text())
        for item in receipt['json_files']:
            path = annotation_dir / item['filename']
            if not path.is_file() or file_sha(path) != item['sha256']:
                raise IOError(f'Annotation missing or changed: {item["filename"]}')

    # Snapshot the official source once; reuse its checksum-locked bytes later.
    upstream_url = ('https://raw.githubusercontent.com/GT-Vision-Lab/VQA/master/'
                    'PythonEvaluationTools/vqaEvaluation/vqaEval.py')
    snapshot = ROOT / 'provenance/external/vqaEval_upstream.py'
    source_receipt = ROOT / 'provenance/vqa_source_snapshot.json'
    if source_receipt.exists():
        info = json.loads(source_receipt.read_text())
        assert info['url'] == upstream_url and file_sha(snapshot) == info['sha256']
        raw = snapshot.read_bytes()
    else:
        print('Retrieving official VQA normalization source...', flush=True)
        with urllib.request.urlopen(upstream_url, timeout=30) as response:
            raw = response.read()
        local_source = work4 / 'vqaEval_upstream.py'
        local_source.write_bytes(raw)
        copy_checked(local_source, snapshot)
        info = {'url': upstream_url, 'sha256': hashlib.sha256(raw).hexdigest(),
                'attribution': 'GT-Vision-Lab/VQA; original author aagrawal; original source retained'}
        save_record('provenance/vqa_source_snapshot.json', info)
    upstream = raw.decode('utf-8')

    # Only the three reviewed, Python-3-compatible methods are reused.
    # The legacy evaluate() routine (and its conditional normalization) is excluded.
    method_names = ('__init__', 'processPunctuation', 'processDigitArticle')
    def extract_method(name):
        match = re.search(r'(?m)^([\t ]+)def ' + re.escape(name) + r'\(', upstream)
        if not match:
            raise RuntimeError(f'Expected official method missing: {name}')
        next_method = re.search(r'(?m)^' + re.escape(match.group(1)) + r'def \w+\(', upstream[match.end():])
        end = match.end() + next_method.start() if next_method else len(upstream)
        return upstream[match.start():end].rstrip() + '\n\n'
    selected_class = 'class VQAEval:\n' + ''.join(extract_method(n) for n in method_names)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', SyntaxWarning)
        tree = ast.parse(selected_class)
    assert len(tree.body) == 1 and isinstance(tree.body[0], ast.ClassDef)
    assert [n.name for n in tree.body[0].body] == list(method_names)
    allowed_calls = {'getQuesIds', 'compile', 'search', 'replace', 'sub', 'lower',
                     'split', 'setdefault', 'append', 'join'}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal)):
            raise RuntimeError('Unexpected construct in selected upstream methods.')
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                assert node.func.id == 'enumerate'
            else:
                assert isinstance(node.func, ast.Attribute) and node.func.attr in allowed_calls
    namespace = {'re': re, '__name__': 'task2_vqa_normalization'}
    exec(compile(tree, '<selected official VQA methods>', 'exec'), namespace)
    official = namespace['VQAEval'](SimpleNamespace(getQuesIds=lambda: []), None)

    @lru_cache(maxsize=200000)
    def normalize_answer(text):
        text = text.replace('\n', ' ').replace('\t', ' ').strip()
        return official.processDigitArticle(official.processPunctuation(text))

    tests = {'The two cats.': '2 cats', '1,000': '1000', '3.14': '3.14',
             'dont': "don't", ' YES!\n': 'yes', '': ''}
    for text, expected in tests.items():
        assert normalize_answer(text) == expected, (text, normalize_answer(text), expected)
    wrapper = '''\nfrom types import SimpleNamespace
from functools import lru_cache
_normalizer = VQAEval(SimpleNamespace(getQuesIds=lambda: []), None)
@lru_cache(maxsize=200000)
def normalize_answer(text):
    text = text.replace("\\n", " ").replace("\\t", " ").strip()
    return _normalizer.processDigitArticle(_normalizer.processPunctuation(text))
'''
    module_text = ('# Normalization methods reused from GT-Vision-Lab/VQA (aagrawal).\n'
                   '# Full upstream source and SHA256 are saved under provenance/.\n'
                   'import re\n\n' + selected_class + wrapper)
    module_path = work4 / 'vqa_normalization.py'
    module_path.write_text(module_text)
    copy_checked(module_path, ROOT / 'code/vqa_normalization.py')
    normalizer_info = {'upstream': info, 'module_sha256': file_sha(module_path),
                       'methods_reused': list(method_names), 'normalization_smoke_tests': len(tests),
                       'normalization_policy': 'Always apply the official punctuation and digit/article helpers to each answer',
                       'legacy_evaluate_not_used': True,
                       'syntax_warnings': [str(w.message) for w in caught]}
    save_record('provenance/vqa_normalizer.json', normalizer_info)
    report4['normalizer'] = normalizer_info

    @lru_cache(maxsize=100000)
    def content_length(text):
        return len(tokenizer.encode(text, add_special_tokens=False))

    for split in ('train2014', 'val2014'):
        print(f'\nPreparing {split} candidate pairs...', flush=True)
        with (annotation_dir / f'v2_mscoco_{split}_complementary_pairs.json').open() as f:
            raw_pairs = json.load(f)
        pairs = sorted(tuple(sorted(map(int, p))) for p in raw_pairs)
        needed_ids = {qid for pair in pairs for qid in pair}
        del raw_pairs
        with (annotation_dir / f'v2_OpenEnded_mscoco_{split}_questions.json').open() as f:
            payload = json.load(f)
        questions = {int(q['question_id']): (int(q['image_id']), q['question'])
                     for q in payload['questions'] if int(q['question_id']) in needed_ids}
        del payload
        assert set(questions) == needed_ids

        with (annotation_dir / f'v2_mscoco_{split}_annotations.json').open() as f:
            payload = json.load(f)
        answers = {}
        for index, annotation in enumerate(payload['annotations']):
            qid = int(annotation['question_id'])
            if qid not in needed_ids:
                continue
            assert qid not in answers
            assert int(annotation['image_id']) == questions[qid][0]
            refs = sorted(annotation['answers'], key=lambda a: int(a['answer_id']))
            assert len(refs) == 10 and len({int(a['answer_id']) for a in refs}) == 10
            normalized = [normalize_answer(a['answer']) for a in refs]
            canonical, agreement = Counter(normalized).most_common(1)[0]
            target_ref = next(a for a, norm in zip(refs, normalized) if norm == canonical)
            answers[qid] = {
                'answer_type': annotation['answer_type'], 'canonical': canonical,
                'agreement': agreement, 'target': target_ref['answer'],
                'target_answer_id': int(target_ref['answer_id']),
                'human_answers': [a['answer'] for a in refs],
                'answer_ids': [int(a['answer_id']) for a in refs],
            }
            if (index + 1) % 50000 == 0:
                print(f'  normalized {index + 1:,} source questions', flush=True)
        del payload
        gc.collect()
        assert set(answers) == needed_ids

        counts, rejected_counts, flags = Counter(), Counter(), Counter()
        seen = set()
        eligible_path = work4 / f'natural_eligible_{split}.jsonl'
        rejected_path = work4 / f'natural_rejected_{split}.jsonl'
        eligible_temp = eligible_path.with_suffix('.jsonl.pending')
        rejected_temp = rejected_path.with_suffix('.jsonl.pending')
        with eligible_temp.open('w') as good, rejected_temp.open('w') as bad:
            for pair in pairs:
                q1, q2 = pair
                a, b = answers[q1], answers[q2]
                im1, question1 = questions[q1]
                im2, question2 = questions[q2]
                reason = None
                if pair in seen:
                    reason = 'duplicate_unordered_question_pair'
                elif im1 == im2:
                    reason = 'same_image'
                elif question1 != question2:
                    reason = 'question_strings_differ'
                elif min(a['agreement'], b['agreement']) < 8:
                    reason = 'consensus_below_8_of_10'
                elif a['canonical'] == b['canonical']:
                    reason = 'normalized_consensus_answers_equal'
                elif a['answer_type'] != b['answer_type']:
                    reason = 'mixed_answer_types_unassigned_to_quota'
                elif content_length(question1) > 80:
                    reason = 'question_content_above_80_tokens'
                elif max(content_length(a['target']), content_length(b['target'])) > 16:
                    reason = 'chosen_original_target_above_16_tokens'
                seen.add(pair)
                if reason:
                    rejected_counts[reason] += 1
                    bad.write(json.dumps({'question_ids': pair, 'reason': reason}, ensure_ascii=False) + '\n')
                    continue
                assert a['answer_type'] in ('yes/no', 'number', 'other')
                examples = []
                for qid, image_id, answer in ((q1, im1, a), (q2, im2, b)):
                    examples.append(dict(question_id=qid, image_id=image_id, **answer,
                                         target_content_tokens=content_length(answer['target'])))
                row = {'pair_id': f'{split}:{q1}:{q2}', 'source_split': split,
                       'answer_type': a['answer_type'], 'question': question1,
                       'question_content_tokens': content_length(question1), 'examples': examples}
                if not question1.strip():
                    flags['empty_question_pairs'] += 1
                if not a['canonical'] or not b['canonical']:
                    flags['empty_normalized_canonical_pairs'] += 1
                if not a['target'].strip() or not b['target'].strip():
                    flags['empty_original_target_pairs'] += 1
                counts[a['answer_type']] += 1
                good.write(json.dumps(row, ensure_ascii=False) + '\n')
        for temp, final in ((eligible_temp, eligible_path), (rejected_temp, rejected_path)):
            if final.exists():
                if file_sha(temp) != file_sha(final):
                    raise IOError(f'Existing eligibility output differs: {final.name}')
                temp.unlink()
            else:
                temp.rename(final)
        outputs = []
        for path in (eligible_path, rejected_path):
            destination = ROOT / 'manifests/eligibility' / path.name
            digest = copy_checked(path, destination)
            outputs.append({'path': str(destination), 'sha256': digest, 'bytes': path.stat().st_size})
        minimums = ({'yes/no': 640, 'number': 256, 'other': 640} if split == 'train2014'
                    else {'yes/no': 48, 'number': 32, 'other': 48})
        summary = {'supplied_pairs': len(pairs), 'eligible_by_answer_type': dict(counts),
                   'rejected_first_reason_counts': dict(rejected_counts), 'review_flags': dict(flags),
                   'minimum_pair_counts_before_image_disjoint_allocation': minimums,
                   'basic_count_feasible': all(counts[t] >= n for t, n in minimums.items()),
                   'image_disjoint_allocation_tested': False, 'outputs': outputs}
        assert sum(counts.values()) + sum(rejected_counts.values()) == len(pairs)
        report4['splits'][split] = summary
        save_record(f'manifests/eligibility/natural_{split}_summary.json', summary)
        print(json.dumps({split: summary}, indent=2), flush=True)
        del questions, answers, pairs, needed_ids, seen
        gc.collect()
    report4['status'] = 'passed'
    report4['local_disk_free_GiB'] = round(shutil.disk_usage('/content').free / 2**30, 2)
except Exception:
    report4['status'] = 'failed'
    report4['traceback'] = traceback.format_exc()
    save_record(f'logs/block_04a_{stamp}.json', report4)
    print('BLOCK 4A FAILED; diagnostics saved. Keep accelerator None.', flush=True)
    raise

log_path = f'logs/block_04a_{stamp}.json'
digest = save_record(log_path, report4)
print('\nBLOCK 4A COMPLETE — normalization and eligibility analysis; no experiment subsets selected.')
print(json.dumps(report4, indent=2))
print('Saved log:', ROOT / log_path)
print('SHA256:', digest)
print('Keep accelerator None. Paste the complete output.')
