"""CPU source freeze, first-eligible selection, fixed probes, and full training audit."""
import gzip
import hashlib
import json
import re
import shutil
import sys
import time
import unicodedata
import uuid
from collections import Counter
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def publish(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise RuntimeError(f'Unexpected linked file: {path}')
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f'Frozen content differs; preserving it: {path}')
        return
    temporary = path.with_name(path.name + '.pending_' + uuid.uuid4().hex)
    temporary.write_bytes(data)
    if digest(temporary) != hashlib.sha256(data).hexdigest():
        raise IOError(f'Readback failed: {temporary}')
    if path.exists():
        raise RuntimeError('Concurrent execution detected')
    temporary.rename(path)
    if digest(path) != hashlib.sha256(data).hexdigest():
        raise IOError(f'Final readback failed: {path}')


def dump(path, value):
    publish(path, (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n').encode())


def normalized(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def encode_record(tokenizer, row, index, forbidden):
    question, answer = row['query'], row['response']
    original = row.get('original_question') or ''
    if not all(isinstance(t, str) for t in (question, answer, original)):
        raise ValueError(f'Non-text source fields at row {index}')
    if not question.strip() or not answer.strip():
        return None, 'empty'
    if normalized(question) in forbidden or normalized(original) in forbidden:
        return None, 'gsm8k_match'
    prefix = tokenizer.apply_chat_template([{'role':'user','content':question}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    # Define the supervision boundary explicitly. BPE may otherwise merge the
    # prompt's final whitespace with leading response whitespace/content.
    prompt_ids = tokenizer.encode(prefix, add_special_tokens=False)
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    ids = prompt_ids + answer_ids + [tokenizer.eos_token_id]
    decoded_prompt = tokenizer.decode(prompt_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    decoded_answer = tokenizer.decode(answer_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    decoded_all = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if decoded_prompt != prefix or decoded_answer != answer or decoded_all != prefix + answer + tokenizer.eos_token:
        raise RuntimeError(f'Exact text roundtrip failed at row {index}')
    if len(ids) > 2048:
        return None, 'overlength'
    if ids[-1] != tokenizer.eos_token_id or len(ids) <= len(prompt_ids):
        raise RuntimeError('Response/end-of-turn supervision is invalid')
    return {'source_index':index,'query':question,'response':answer,
            'original_question':original,'input_ids':ids,'response_start':len(prompt_ids),
            'n_supervised_tokens':len(ids)-len(prompt_ids)}, None


def main(root, local, bundle):
    import numpy as np
    from datasets import load_dataset
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoTokenizer
    from lm_eval.tasks import TaskManager
    from lm_eval.utils import load_yaml_config
    root, local, bundle = Path(root), Path(local), Path(bundle)
    started = time.monotonic()
    plan = json.loads((bundle/'APPROVED_PLAN.json').read_text())
    reference = json.loads((root/'provenance/APPROVAL_REFERENCE.json').read_text())
    assert plan['source_proposal_sha256'] == reference['proposal_sha256']
    assert digest(bundle/'proposal.md') == reference['proposal_sha256']
    assert [v['id'] for v in plan['choices']] == [f'D{i:02}' for i in range(1,23)]
    ready_env = json.loads((root/'configs/ENVIRONMENT_READY.json').read_text())
    assert ready_env['lock_sha256'] == '54ad919ba1a099e6327014d51b7f441ceee5df7c93330278e4f85ad82e43cc06'
    publish(root/'provenance/APPROVED_PLAN.json', (bundle/'APPROVED_PLAN.json').read_bytes())
    publish(root/'provenance/APPROVED_PROPOSAL.md', (bundle/'proposal.md').read_bytes())
    has_training = ((root/'manifests/TRAINING_DATA_READY.json').exists()
        or (root/'configs/TRAIN_SOURCE_IDS.json').exists()
        or ((root/'data').exists() and any((root/'data').glob('train_*.jsonl.gz'))))
    if has_training and not (root/'configs/TOKENIZATION_PROTOCOL_R1.json').exists():
        raise RuntimeError('Training state predates R1. Preserve it and report before applying the repair.')
    publish(root/'provenance/prepare_data_R1.py', Path(__file__).read_bytes())
    dump(root/'configs/TOKENIZATION_PROTOCOL_R1.json', {
        'version':'R1', 'construction':'encode(native_non_thinking_prompt) + encode(verbatim_response) + [im_end]',
        'labels':'Mask every prompt token; supervise every response token and the single appended im_end',
        'length':'Filter the constructed token sequence at 2048 tokens; never truncate',
        'verification':'Decode prompt, response, and combined IDs exactly, with cleanup and special-token removal disabled',
        'reason':'Fix whole-string BPE crossing the mask boundary, observed at source row 91 before any training shards were written',
        'scope':'Implementation correction under D04; preserve frozen sources, source order, supplied text, and manual eligibility rules'})
    api = HfApi()
    freeze_path = root/'configs/SOURCE_REVISIONS.json'
    manager = TaskManager()
    specs = {'gsm8k_cot':('test',1319),'hellaswag':('validation',10042),
             'boolq':('validation',3270),'arc_easy':('test',2376)}
    print('1/5 Freezing original model/tokenizer and dataset revisions...', flush=True)
    if freeze_path.exists():
        freeze = json.loads(freeze_path.read_text())
    else:
        freeze = {'model':{'repo':'Qwen/Qwen3-0.6B','revision':api.model_info('Qwen/Qwen3-0.6B').sha},
                  'metamath':{'repo':'meta-math/MetaMathQA','revision':api.dataset_info('meta-math/MetaMathQA').sha},
                  'benchmarks':{}}
        for name,(split,count) in specs.items():
            cfg = load_yaml_config(manager.task_index[name]['yaml_path'], mode='full')
            info = api.dataset_info(cfg['dataset_path'])
            freeze['benchmarks'][name] = {'repo':info.id,'revision':info.sha,
                'config':cfg.get('dataset_name'),'split':split,'expected_count':count,
                'stock_dataset_path':cfg['dataset_path']}
        dump(freeze_path, freeze)
    assert all(re.fullmatch('[0-9a-f]{40}', v['revision']) for v in
               [freeze['model'],freeze['metamath'],*freeze['benchmarks'].values()])
    cache = local/'hf_cache'; cache.mkdir(exist_ok=True)
    model = freeze['model']
    tokenizer = AutoTokenizer.from_pretrained(model['repo'], revision=model['revision'],
        cache_dir=str(cache), trust_remote_code=False)
    assert tokenizer.eos_token == '<|im_end|>' and tokenizer.pad_token_id is not None
    token_assets = {}
    for filename in ('config.json','tokenizer_config.json','tokenizer.json'):
        p = Path(hf_hub_download(model['repo'], filename, revision=model['revision'], cache_dir=cache))
        target = root/'sources/tokenizer'/filename
        publish(target,p.read_bytes()); token_assets[filename]=digest(target)
    dump(root/'manifests/TOKENIZER_FILES.json',token_assets)
    print('2/5 Downloading complete benchmark splits and freezing probe IDs...', flush=True)
    benchmarks = {}
    benchmark_files = {}
    for name,spec in freeze['benchmarks'].items():
        data = load_dataset(spec['repo'], spec['config'], revision=spec['revision'],
            split=spec['split'], cache_dir=str(cache))
        assert len(data)==spec['expected_count'], (name,len(data))
        benchmarks[name]=data
        rows=[{'source_index':i,'document':row} for i,row in enumerate(data)]
        payload=''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in rows).encode()
        path=root/'sources/benchmarks'/(name+'.jsonl.gz')
        publish(path,gzip.compress(payload,mtime=0))
        benchmark_files[name]={'count':len(data),'sha256':digest(path),'path':str(path.relative_to(root))}
        print(f'  {name}: {len(data):,} examples saved',flush=True)
    rng=np.random.default_rng(6304)
    probes={}
    for name in ('gsm8k_cot','hellaswag'):
        indices=rng.choice(len(benchmarks[name]),size=32,replace=False).tolist()
        probes[name]=[{'source_index':i,'source_id':benchmarks[name][i].get('idx',
            benchmarks[name][i].get('ind',i))} for i in indices]
    dump(root/'configs/ACTIVATION_PROBE_IDS.json',probes)
    dump(root/'manifests/BENCHMARK_FILES.json',benchmark_files)
    forbidden={normalized(q) for q in benchmarks['gsm8k_cot']['question']}
    assert '' not in forbidden
    print('3/5 Acquiring the original MetaMath JSON in source order...',flush=True)
    spec=freeze['metamath']
    original=Path(hf_hub_download(spec['repo'],'MetaMathQA-395K.json',repo_type='dataset',
        revision=spec['revision'],cache_dir=cache))
    source_hash=digest(original)
    target=root/'sources/metamath/MetaMathQA-395K.json'
    # Streaming copy avoids an extra full source-file byte copy in host RAM.
    if not target.exists():
        target.parent.mkdir(parents=True,exist_ok=True)
        pending=target.with_name(target.name+'.pending_'+uuid.uuid4().hex)
        shutil.copyfile(original,pending)
        assert digest(pending)==source_hash
        if target.exists(): raise RuntimeError('Concurrent source copy')
        pending.rename(target)
    assert digest(target)==source_hash
    dump(root/'manifests/METAMATH_SOURCE.json',{'sha256':source_hash,'revision':spec['revision']})
    with original.open() as stream: source=json.load(stream)
    assert isinstance(source,list) and len(source)>=40000
    print('Checking exact boundary/text preservation on the first 128 source rows, including failing row 91...',flush=True)
    boundary_cases = 0
    for check_index,check_row in enumerate(source[:128]):
        record,reason = encode_record(tokenizer,check_row,check_index,forbidden)
        if record is not None:
            boundary_cases += 1
    print(f'  Boundary regression check passed: {boundary_cases} eligible rows, no discarded boundary errors.',flush=True)
    print('4/5 Selecting and saving the first 40,000 eligible examples...',flush=True)
    selected=[]; rejected=[]; reasons=Counter(); shards=[]; chunk=[]
    for index,row in enumerate(source):
        record,reason=encode_record(tokenizer,row,index,forbidden)
        if reason:
            reasons[reason]+=1
            rejected.append({'source_index':index,'reason':reason})
        else:
            selected.append(index);chunk.append(record)
            if len(chunk)==1000:
                payload=''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in chunk).encode()
                path=root/'data'/f'train_{len(shards):03}.jsonl.gz'
                publish(path,gzip.compress(payload,mtime=0))
                shards.append({'path':str(path.relative_to(root)),'sha256':digest(path),'count':len(chunk)})
                chunk=[]
                print(f'  {len(selected):,} selected; scanned {index+1:,} source rows',flush=True)
        if len(selected)==40000:break
    assert len(selected)==40000 and not chunk and len(shards)==40
    dump(root/'configs/TRAIN_SOURCE_IDS.json',selected)
    dump(root/'manifests/TRAIN_REJECTIONS.json',rejected)
    print('5/5 Auditing all saved examples, masking boundaries, order, and exclusion rules...',flush=True)
    seen=[]; tokens=0; maximum=0
    for shard in shards:
        path=root/shard['path']; assert digest(path)==shard['sha256']
        with gzip.open(path,'rt') as stream:
            count=0
            for line in stream:
                record=json.loads(line); idx=record['source_index']
                replay,reason=encode_record(tokenizer,source[idx],idx,forbidden)
                assert reason is None and replay==record
                assert 0<record['response_start']<len(record['input_ids'])<=2048
                assert len(record['input_ids'])-record['response_start']==record['n_supervised_tokens']
                seen.append(idx); tokens+=record['n_supervised_tokens']
                maximum=max(maximum,len(record['input_ids']));count+=1
            assert count==shard['count']
    assert seen==selected and selected==sorted(set(selected))
    assert set(selected).isdisjoint(r['source_index'] for r in rejected)
    assert len(selected)+len(rejected)==selected[-1]+1
    assert sum(reasons.values())==len(rejected)
    manifest={'schema_version':1,'selected_count':40000,'updates':1250,'effective_batch':32,
        'source_rows':len(source),'scanned_rows':selected[-1]+1,'rejections':dict(reasons),
        'source_sha256':source_hash,'source_revision':spec['revision'],'shards':shards,
        'total_supervised_tokens':tokens,'maximum_formatted_length':maximum,
        'normalization':'NFKC; casefold; collapse whitespace; strip; preserve punctuation/numbers/operators',
        'format':'Native non-thinking user/assistant prefix masked; verbatim response + one supervised im_end',
        'tokenization_protocol':'R1 boundary-separated encoding; exact decoded text verified',
        'full_saved_training_audit_passed':True,'benchmark_scores_inspected':False}
    dump(root/'manifests/TRAINING_DATA_READY.json',manifest)
    yaml_present=(root/'sources/first_plot_questions.yaml').is_file()
    receipt={'status':'TASK1_BLOCK04_R1_OK','selected_examples':40000,'training_shards':40,
        'scanned_source_rows':selected[-1]+1,'rejections':dict(reasons),
        'maximum_formatted_length':maximum,'total_supervised_tokens':tokens,
        'benchmarks':{k:len(v) for k,v in benchmarks.items()},'probe_questions':64,
        'full_saved_training_audit_passed':True,'full_approval_plan_synchronized':True,
        'qualitative_file_present_not_yet_validated':yaml_present,'gpu_work_run':False,
        'model_weights_downloaded':False,'elapsed_seconds':round(time.monotonic()-started,1)}
    dump(root/'logs'/('block04_'+uuid.uuid4().hex+'.json'),receipt)
    print(json.dumps(receipt,indent=2),flush=True)
    print('TASK1_BLOCK04_R1_OK — keep CPU selected and send the complete output.',flush=True)


if __name__=='__main__':
    main(*sys.argv[1:])
