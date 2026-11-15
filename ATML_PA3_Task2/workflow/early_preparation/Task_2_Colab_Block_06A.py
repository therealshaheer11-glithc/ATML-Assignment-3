# Block 6A: approved exact protocol, rendered wording, and audit preparation.
# CPU only. No images, model weights, calibration computation, or training.
import os, json, uuid, shutil, traceback, math, csv, io
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import Counter
import numpy as np

needed = ('ROOT', 'LOCAL', 'save_record', 'file_sha', 'copy_checked', 'save_rows',
          'tokenizer', 'capacity', 'counting_combinations')
if any(n not in globals() for n in needed):
    raise RuntimeError('Notebook state missing. Stop; do not rerun Block 1.')
previous = ROOT / 'logs/block_05b_20261031_161050_c9dc05e9.json'
assert file_sha(previous) == '21ccdac5191bebd4caeba690adf481ecb47ef5ae476a7e03127731bb9ca5499a'
old5b = json.loads(previous.read_text())
if os.environ.get('COLAB_GPU', '0') not in ('', '0'):
    print('CPU preparation: requesting GPU release.', flush=True)
    try:
        runtime.unassign()
    finally:
        raise RuntimeError('Reconnect with accelerator None.')
stamp = datetime.now(ZoneInfo('Asia/Karachi')).strftime('%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:8]
report6 = {'attempt': stamp, 'status': 'started', 'gpu_used': False,
           'complete_dataset_frozen': False, 'visual_labels_reviewed': False,
           'calibration_computed': False, 'outputs': [], 'token_maxima': {}}
SPEC = json.loads(r'''{"status":"approved","date":"2026-10-31","scope":"Exact values previously left unspecified; extends approved directions without replacing earlier approvals","wording":{"caption_user_text":"Describe this image.","vqa_user_text":"Answer with a short phrase.\n{original_question}","system_message":null,"thinking_enabled":false,"presence_question":"Does this image contain {presence_phrase}?","counting_question":"How many {counting_phrase} can you see?","same_vqa_text_for_all_image_conditions":true,"preserve_natural_questions_and_original_targets":true,"cap_check":"Check original question and complete rendered VQA user text each <=80 tokens. Stop without truncation or reselection if any chosen example fails."},"audit":{"seed":6304,"evaluation_grounding_pairs":88,"training_presence_pairs":80,"training_counting_pairs":78,"total_pairs":246,"training_sample_rule":"One complete pair for every eligible category/task, using an independent default_rng(6304); uniform eligible count combination and images","review":"Full images plus category boxes; review annotation absence, omitted objects, count/group ambiguity, occlusion and crowd handling independently of model predictions","dispositions":["accept","ambiguous","annotation_error"],"automatic_replacement_or_relabeling":false,"training_budget_affected":false},"calibration":{"seed":6304,"training_images":256,"training_caption_examples":256,"selection":"Sorted IDs, independent default_rng(6304), without replacement; image sample from 8733-image training union; captions from selected 18000","numerator":"Global RMS of frozen Qwen embeddings of all content tokens in sampled original captions; exclude chat, termination and padding tokens","denominator":"Global RMS of initial seeded connector outputs for all 196 patches of sampled cached training images","arithmetic":"FP32 streaming sum of squares and element counts","scale":"numerator / denominator; error if either RMS is zero or nonfinite","placement":"One fixed non-trainable scalar after connector MLP, before conversion to Qwen FP16 input embeddings; retain throughout all stages","saved_evidence":["sample IDs","both RMS values","scale","initial connector state and hash"],"evaluation_inputs_used":false,"additional_normalization_or_clamping":false},"learning_rate_endpoints":{"warmup_updates":[113,75,38],"update_index":"One-based successful optimizer update u=1..U within stage","warmup":"LR(u)=peak*u/w for 1<=u<=w","decay":"LR(u)=peak*(0.1+0.9*(1+cos(pi*(u-w)/(U-w)))/2) for w<u<=U","first":"peak/w","warmup_end":"peak","stage_end":"0.1*peak","applied":"Set the LR for update u before its optimizer step; only completed updates advance state"},"decision_id":"A005","approval_date":"2026-10-31","user_approval":"approved and document and let's continue"}''')
# Literal, approved noun forms; no automatic inflection or unit changes.
NOUN_FORMS = """person|a person|people
bicycle|a bicycle|bicycles
car|a car|cars
motorcycle|a motorcycle|motorcycles
airplane|an airplane|airplanes
bus|a bus|buses
train|a train|trains
truck|a truck|trucks
boat|a boat|boats
traffic light|a traffic light|traffic lights
fire hydrant|a fire hydrant|fire hydrants
stop sign|a stop sign|stop signs
parking meter|a parking meter|parking meters
bench|a bench|benches
bird|a bird|birds
cat|a cat|cats
dog|a dog|dogs
horse|a horse|horses
sheep|a sheep|sheep
cow|a cow|cows
elephant|an elephant|elephants
bear|a bear|bears
zebra|a zebra|zebras
giraffe|a giraffe|giraffes
backpack|a backpack|backpacks
umbrella|an umbrella|umbrellas
handbag|a handbag|handbags
tie|a tie|ties
suitcase|a suitcase|suitcases
frisbee|a frisbee|frisbees
skis|skis|skis
snowboard|a snowboard|snowboards
sports ball|a sports ball|sports balls
kite|a kite|kites
baseball bat|a baseball bat|baseball bats
baseball glove|a baseball glove|baseball gloves
skateboard|a skateboard|skateboards
surfboard|a surfboard|surfboards
tennis racket|a tennis racket|tennis rackets
bottle|a bottle|bottles
wine glass|a wine glass|wine glasses
cup|a cup|cups
fork|a fork|forks
knife|a knife|knives
spoon|a spoon|spoons
bowl|a bowl|bowls
banana|a banana|bananas
apple|an apple|apples
sandwich|a sandwich|sandwiches
orange|an orange|oranges
broccoli|broccoli|broccoli
carrot|a carrot|carrots
hot dog|a hot dog|hot dogs
pizza|pizza|pizzas
donut|a donut|donuts
cake|cake|cakes
chair|a chair|chairs
couch|a couch|couches
potted plant|a potted plant|potted plants
bed|a bed|beds
dining table|a dining table|dining tables
toilet|a toilet|toilets
tv|a TV|TVs
laptop|a laptop|laptops
mouse|a mouse|mice
remote|a remote|remotes
keyboard|a keyboard|keyboards
cell phone|a cell phone|cell phones
microwave|a microwave|microwaves
oven|an oven|ovens
toaster|a toaster|toasters
sink|a sink|sinks
refrigerator|a refrigerator|refrigerators
book|a book|books
clock|a clock|clocks
vase|a vase|vases
scissors|scissors|scissors
teddy bear|a teddy bear|teddy bears
hair drier|a hair drier|hair driers
toothbrush|a toothbrush|toothbrushes"""
SPEC['noun_forms'] = {}
for line in NOUN_FORMS.splitlines():
    name, presence, counting = line.split('|')
    assert name not in SPEC['noun_forms']
    SPEC['noun_forms'][name] = {'presence_phrase': presence, 'counting_phrase': counting}
assert len(SPEC['noun_forms']) == 80 and SPEC['status'] == 'approved'

def load_rows(item):
    durable = Path(item['path'])
    local = LOCAL / 'protocol_inputs' / durable.name
    copy_checked(durable, local)
    assert file_sha(local) == item['sha256']
    with local.open() as stream:
        return [json.loads(line) for line in stream]

def user_text(question):
    return SPEC['wording']['vqa_user_text'].format(original_question=question)

violations = []
def length(text):
    return len(tokenizer.encode(text, add_special_tokens=False))

def checked_question(question, identity):
    text = user_text(question)
    qn, un = length(question), length(text)
    report6['token_maxima']['original_question'] = max(report6['token_maxima'].get('original_question', 0), qn)
    report6['token_maxima']['complete_vqa_user_text'] = max(report6['token_maxima'].get('complete_vqa_user_text', 0), un)
    if qn > 80 or un > 80:
        violations.append({'id': identity, 'question_tokens': qn, 'user_text_tokens': un})
    return {'question': question, 'user_text': text,
            'question_content_tokens': qn, 'user_content_tokens': un}

def format_pair(row, question):
    formatted = {**row, **checked_question(question, row['pair_id'])}
    for ex in row['examples']:
        n = length(ex['target'])
        report6['token_maxima']['short_answer_target'] = max(report6['token_maxima'].get('short_answer_target', 0), n)
        if n > 16:
            violations.append({'id': ex['question_id'], 'answer_tokens': n})
    return formatted

def ground_question(category_name, kind):
    forms = SPEC['noun_forms'][category_name]
    if kind == 'presence':
        return SPEC['wording']['presence_question'].format(presence_phrase=forms['presence_phrase'])
    if kind == 'counting':
        return SPEC['wording']['counting_question'].format(counting_phrase=forms['counting_phrase'])
    raise ValueError(kind)

def draw_training_audit_pair(category, kind, rng):
    if kind == 'presence':
        images = (int(rng.choice(category['present'])), int(rng.choice(category['absent'])))
        targets, counts = ('yes', 'no'), (None, None)
    else:
        combos = counting_combinations(category)
        a, b = combos[int(rng.integers(len(combos)))]
        images = (int(rng.choice(category['count_images'][str(a)])),
                  int(rng.choice(category['count_images'][str(b)])))
        targets, counts = (str(a), str(b)), (a, b)
    pair_id = f'grounding:train_audit:{kind}:{category["category_id"]}:{min(images)}:{max(images)}'
    return {'pair_id': pair_id, 'source_split': 'train2014', 'experiment_split': 'train_audit',
            'audit_only': True, 'question_type': kind, 'category_id': category['category_id'],
            'category_name': category['category_name'], 'question_rendered': True,
            'examples': [{'question_id': f'{pair_id}:{j}', 'image_id': im, 'target': target,
                          'canonical': target, 'annotated_count': count}
                         for j, (im, target, count) in enumerate(zip(images, targets, counts))]}

def learning_rate_for_update(peak, updates, warmup, update):
    if not (peak > 0 and 0 < warmup < updates and 1 <= update <= updates):
        raise ValueError('Invalid schedule input.')
    if update <= warmup:
        return peak * update / warmup
    return peak * (0.1 + 0.9 * (1 + math.cos(math.pi * (update - warmup) / (updates - warmup))) / 2)

try:
    protocol_hash = save_record('configs/exact_protocol_A005.json', SPEC)
    report6['approved_protocol_sha256'] = protocol_hash
    pool_path = ROOT / 'manifests/selected/grounding_training_pools.json'
    assert file_sha(pool_path) == old5b['grounding_training_pools_sha256']
    pool = json.loads(pool_path.read_text())
    categories = sorted(pool['categories'], key=lambda c: c['category_id'])
    assert {c['category_name'] for c in categories} == set(SPEC['noun_forms'])
    assert sum(capacity(c, 'presence') > 0 for c in categories) == 80
    assert sum(capacity(c, 'counting') > 0 for c in categories) == 78

    # One fixed question bank supports training and every image condition.
    bank = []
    for category in categories:
        for kind in ('presence', 'counting'):
            bank.append({'category_id': category['category_id'], 'category_name': category['category_name'],
                         'question_type': kind, 'available_in_training_pool': capacity(category, kind) > 0,
                         **checked_question(ground_question(category['category_name'], kind),
                                            f'bank:{category["category_id"]}:{kind}')})
    output_sets = {}
    for suite in ('dev', 'test'):
        item = next(r for r in old5b['outputs'] if r['path'].endswith(f'/grounding_{suite}_pair_metadata.jsonl'))
        metadata_rows = load_rows(item)
        output_sets[f'grounding_{suite}.jsonl'] = [
            {**format_pair(row, ground_question(row['category_name'], row['question_type'])),
             'question_rendered': True} for row in metadata_rows]

    old5a_path = ROOT / 'logs/block_05a_20261031_155959_c7d3095c.json'
    assert file_sha(old5a_path) == 'a84f2436281e657665850dfad2750840b55678b271f4c1dffddc7f8e195973fc'
    old5a = json.loads(old5a_path.read_text())
    for suite in ('train', 'dev', 'test'):
        item = next(r for r in old5a['outputs'] if r['path'].endswith(f'/natural_{suite}.jsonl'))
        rows = load_rows(item)
        output_sets[f'natural_{suite}_formatted.jsonl'] = [format_pair(row, row['question']) for row in rows]

    # Audit RNG is independent of curriculum/data-allocation RNG state.
    audit_rng = np.random.default_rng(6304)
    train_audit = []
    for kind in ('presence', 'counting'):
        for category in categories:
            if capacity(category, kind):
                row = draw_training_audit_pair(category, kind, audit_rng)
                train_audit.append(format_pair(row, ground_question(category['category_name'], kind)))
    assert Counter(r['question_type'] for r in train_audit) == {'presence': 80, 'counting': 78}
    output_sets['grounding_training_audit.jsonl'] = train_audit
    audit_pairs = train_audit + output_sets['grounding_dev.jsonl'] + output_sets['grounding_test.jsonl']
    assert len(audit_pairs) == len({r['pair_id'] for r in audit_pairs}) == 246
    if violations:
        report6['cap_violations'] = violations
        raise RuntimeError(f'{len(violations)} cap violations. No truncation or reselection performed.')
    report6['grounding_question_bank_sha256'] = save_record('configs/grounding_question_bank.json', bank)
    for filename, rows in output_sets.items():
        report6['outputs'].append(save_rows(filename, rows))

    # Save the 246-pair review checklist with genuinely blank review fields.
    review_local = LOCAL / 'selection' / 'annotation_audit_review_template.csv'
    columns = ['pair_id', 'source_split', 'experiment_split', 'question_type', 'category_id',
               'category_name', 'question', 'image_0', 'target_0', 'image_1', 'target_1',
               'label_status', 'notes', 'reviewer', 'reviewed_at']
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator='\n')
    writer.writeheader()
    for row in audit_pairs:
        a, b = row['examples']
        writer.writerow({'pair_id': row['pair_id'], 'source_split': row['source_split'],
                         'experiment_split': row['experiment_split'], 'question_type': row['question_type'],
                         'category_id': row['category_id'], 'category_name': row['category_name'],
                         'question': row['question'], 'image_0': a['image_id'], 'target_0': a['target'],
                         'image_1': b['image_id'], 'target_1': b['target'],
                         'label_status': '', 'notes': '', 'reviewer': '', 'reviewed_at': ''})
    if review_local.exists():
        assert review_local.read_text() == buffer.getvalue(), 'Existing review template differs; preserved.'
    else:
        review_local.write_text(buffer.getvalue())
    review_target = ROOT / 'audits/annotation_audit_review_template.csv'
    report6['audit_review_template_sha256'] = copy_checked(review_local, review_target)
    cache_path = ROOT / 'manifests/selected/candidate_cache_image_ids.json'
    assert file_sha(cache_path) == old5b['candidate_cache_image_ids_sha256']
    cache_ids = json.loads(cache_path.read_text())
    audit_ids = {split: sorted({ex['image_id'] for row in audit_pairs if row['source_split'] == split
                               for ex in row['examples']}) for split in ('train2014', 'val2014')}
    assert all(set(audit_ids[s]) <= set(cache_ids[s]) for s in audit_ids)
    report6['audit_image_ids_sha256'] = save_record('audits/annotation_audit_image_ids.json', audit_ids)
    report6['audit_plan_sha256'] = save_record('audits/annotation_audit_plan.json', {
        'decision_id': 'A005', 'seed': 6304, 'training_pairs': 158, 'evaluation_pairs': 88,
        'total_pairs': 246, 'image_presentations': 492, 'image_ids': audit_ids,
        'review_template': str(review_target), 'completed_reviews': 0,
        'allowed_statuses': ['accept', 'ambiguous', 'annotation_error'],
        'automatic_changes_allowed': False, 'model_predictions_used': False,
        'audit_views_do_not_count_as_training_presentations': True})

    # Prepare calibration IDs only; do not load weights or compute a scale.
    union_path = ROOT / 'manifests/selected/training_image_union.json'
    assert file_sha(union_path) == old5a['training_union_sha256']
    training_images = sorted(json.loads(union_path.read_text())['image_ids'])
    item = next(r for r in old5a['outputs'] if r['path'].endswith('/captions_train.jsonl'))
    caption_rows = load_rows(item)
    caption_ids = sorted(r['caption_id'] for r in caption_rows)
    assert len(training_images) == 8733 and len(caption_ids) == len(set(caption_ids)) == 18000
    calibration_rng = np.random.default_rng(6304)
    calibration_ids = {
        'decision_id': 'A005', 'seed': 6304, 'sampling_order': ['training_images', 'training_caption_examples'],
        'image_ids': [training_images[int(i)] for i in calibration_rng.permutation(len(training_images))[:256]],
        'caption_ids': [caption_ids[int(i)] for i in calibration_rng.permutation(len(caption_ids))[:256]],
        'scale_computed': False}
    assert len(set(calibration_ids['image_ids'])) == len(set(calibration_ids['caption_ids'])) == 256
    report6['calibration_sample_sha256'] = save_record('configs/calibration_sample_ids.json', calibration_ids)
    caption_prompt_tokens = length(SPEC['wording']['caption_user_text'])
    assert caption_prompt_tokens <= 80
    report6['caption_user_text_tokens'] = caption_prompt_tokens

    # Verify exact approved LR endpoints without performing optimizer updates.
    schedule_cases = [(1, 'connector', 2250, 113, 1e-3),
                      (2, 'connector', 1500, 75, 1e-4), (2, 'lora', 1500, 75, 2e-5),
                      (3, 'connector', 750, 38, 5e-5), (3, 'lora', 750, 38, 1e-5)]
    endpoints = []
    for stage, component, updates, warmup, peak in schedule_cases:
        values = [learning_rate_for_update(peak, updates, warmup, u) for u in range(1, updates + 1)]
        assert math.isclose(values[0], peak / warmup, rel_tol=1e-12)
        assert math.isclose(values[warmup - 1], peak, rel_tol=1e-12)
        assert math.isclose(values[-1], 0.1 * peak, rel_tol=1e-12)
        assert all(a <= b for a, b in zip(values[:warmup - 1], values[1:warmup]))
        assert all(a >= b for a, b in zip(values[warmup - 1:-1], values[warmup:]))
        endpoints.append({'stage': stage, 'component': component, 'updates': updates,
                          'warmup': warmup, 'peak': peak, 'first': values[0],
                          'warmup_end': values[warmup - 1], 'last': values[-1]})
    report6['lr_endpoints_sha256'] = save_record('configs/training_lr_endpoints.json', endpoints)
    report6.update({'status': 'passed', 'audit_pairs': {'training': 158, 'evaluation': 88, 'total': 246},
                    'audit_unique_images': {s: len(ids) for s, ids in audit_ids.items()},
                    'audit_reviews_completed': 0, 'calibration_images_selected': 256,
                    'calibration_captions_selected': 256, 'lr_endpoint_checks_passed': True,
                    'chat_serialization_example': tokenizer.apply_chat_template(
                        [{'role': 'user', 'content': output_sets['grounding_dev.jsonl'][0]['user_text']}],
                        tokenize=False, add_generation_prompt=True, enable_thinking=False),
                    'local_disk_free_GiB': round(shutil.disk_usage('/content').free / 2**30, 2)})
except Exception:
    report6['status'] = 'failed'
    report6['traceback'] = traceback.format_exc()
    save_record(f'logs/block_06a_{stamp}.json', report6)
    print('BLOCK 6A FAILED; diagnostics saved; no examples truncated or reselected.', flush=True)
    raise

log_path = f'logs/block_06a_{stamp}.json'
digest = save_record(log_path, report6)
print('\nBLOCK 6A COMPLETE — approved protocol saved; wording checked; audits prepared, not completed.')
print(json.dumps(report6, indent=2))
print('Saved log:', ROOT / log_path)
print('SHA256:', digest)
print('Keep accelerator None. Paste the complete output.')
