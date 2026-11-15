"""CPU acquisition and structural validation. Never grade or select an answer."""
import hashlib
import json
import math
import shutil
import subprocess
import sys
import uuid
from datetime import datetime
from importlib import metadata
from pathlib import Path
from zoneinfo import ZoneInfo

SOURCE_URL = 'https://drive.google.com/drive/folders/1utX-Gfi9vvaoi1mi0pzNpDGL1o1fAh23'
PLAN_SHA = 'f6c48fc63de87286cd87d1029b823670ce085af2cc41a3f175f4d76ea3f90c02'
REQ_SHA = '60904f8e38bfe213cc2ed2c3b59af9be4ff01be42c27b223dda3588282d7744d'
EXPECTED = {'README.MD', 'qwen25-math/generations.parquet', 'qwen25-math/probability_metrics.parquet', 'qwen25vl-mathvista/generations.parquet', 'qwen25vl-mathvista/probability_metrics.parquet', 'qwen25vl-mathvista/attention_metrics.parquet'}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(4*1024*1024), b''): h.update(data)
    return h.hexdigest()

def require(condition, message):
    if not condition: raise ValueError(message)

def key(row):
    require(isinstance(row['sample_id'], str), 'Question IDs must be strings')
    require(isinstance(row['rollout_id'], int) and not isinstance(row['rollout_id'], bool) and 0 <= row['rollout_id'] < 32, 'Invalid rollout ID')
    return row['dataset'], row['sample_id'], row['rollout_id']

def finite_values(values, name, lower=None, upper=None):
    require(isinstance(values, list), f'{name}: expected token list')
    for v in values:
        require(isinstance(v, (int, float)) and math.isfinite(v), f'{name}: nonfinite/nonnumeric value')
        require(lower is None or v >= lower, f'{name}: below valid range')
        require(upper is None or v <= upper, f'{name}: above valid range')

def validate_dataset(rows, expected_questions=1000, reference=None):
    """rows(kind) yields dicts from independent streams; joins never rely on row order."""
    index, questions, models, datasets = {}, {}, set(), set()
    tokens = caps = missing = 0
    for row in rows('generations'):
        k = key(row)
        require(k not in index, 'Duplicate generation join key')
        ids = row['generated_token_ids']; t = row['n_generated_text_tokens']
        require(isinstance(t, int) and not isinstance(t, bool) and t > 0 and isinstance(ids, list) and t == len(ids), 'Invalid scalar generation length/alignment')
        require(all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in ids), 'Invalid generated token IDs')
        require(isinstance(row['hit_max_new_tokens'], bool), 'Cap flag must be boolean')
        require(isinstance(row['generated_text'], str), 'Missing generated text')
        require(isinstance(row['question_text'], str) and row['question_text'].strip(), 'Invalid question')
        require(isinstance(row['gold_answer'], str) and row['gold_answer'].strip(), 'Invalid reference')
        require(row['extracted_answer'] is None or isinstance(row['extracted_answer'], str), 'Invalid extracted-answer type')
        signature = (row['question_text'], row['gold_answer'])
        require(k[1] not in questions or questions[k[1]] == signature, 'Question/reference changes across rollouts')
        questions[k[1]] = signature
        datasets.add(k[0]); models.add(row['model']); index[k] = t
        tokens += t; caps += row['hit_max_new_tokens']; missing += row['extracted_answer'] is None
    require(len(questions) == expected_questions and len(index) == expected_questions*32, 'Wrong question/rollout count')
    require(len(datasets) == len(models) == 1, 'Mixed dataset/model identifiers')
    for q in questions:
        require({k[2] for k in index if k[1] == q} == set(range(32)), 'Question lacks exactly rollout IDs 0..31')
    seen = set()
    for row in rows('probability_metrics'):
        k = key(row)
        require(k in index and k not in seen, 'Probability join key missing/extra/duplicate')
        seen.add(k)
        for name, lower, upper in [('logp', None, 1e-5), ('kl_u_p', -1e-5, None)]:
            values = row[name]
            require(isinstance(values, list) and len(values) == index[k], f'{name}: token alignment mismatch')
            finite_values(values, name, lower, upper)
    require(seen == set(index), 'Probability join is incomplete')
    attention = rows('attention_metrics')
    attention_count = 0
    if attention is not None:
        seen = set()
        for row in attention:
            k = key(row)
            require(k in index and k not in seen, 'Attention join key missing/extra/duplicate')
            seen.add(k); t = index[k]
            for name in ['n_image_tokens', 'n_prompt_text_tokens']:
                require(isinstance(row[name], int) and row[name] > 0, 'Invalid attention group size')
            require(row['n_generated_text_tokens'] == list(range(t)), 'Attention preceding-token counts misaligned')
            groups = []
            for name in ['img_attn_total', 'prompt_text_attn_total', 'generated_text_attn_total']:
                values = row[name]
                require(isinstance(values, list) and len(values) == t, 'Attention list length mismatch')
                finite_values(values, name, 0, 1.00001); groups.append(values)
            require(groups[2][0] == 0, 'First prediction has generated-text attention')
            require(all(sum(v) > 0 for v in zip(*groups)), 'Zero eligible-attention denominator')
        require(seen == set(index), 'Attention join is incomplete'); attention_count = len(seen)
    result = {'questions':len(questions), 'rollouts':len(index), 'retained_tokens':tokens, 'capped_rollouts':caps, 'null_extracted_answers':missing, 'attention_rows':attention_count, 'dataset_identifier':next(iter(datasets)), 'model_identifier':next(iter(models))}
    if reference:
        for name in ['retained_tokens','capped_rollouts','null_extracted_answers']:
            require(result[name] == reference[name], f'README total mismatch: {name}')
    return result

def parquet_rows(path, schema_log):
    import pyarrow.parquet as pq
    file = pq.ParquetFile(path)
    schema_log[str(path.name)] = str(file.schema_arrow)
    for batch in file.iter_batches(batch_size=128):
        yield from batch.to_pylist()

def run(bundle):
    # Reuse the exact previously checked CPU guard and immutable writer.
    import importlib.util
    spec = importlib.util.spec_from_file_location('task3_setup_guard', bundle/'block_01.py')
    guard = importlib.util.module_from_spec(spec); spec.loader.exec_module(guard)
    from google.colab import drive, runtime
    evidence = guard.accelerator_evidence()
    if evidence:
        print('Accelerator detected; releasing this CPU-only session.', flush=True)
        print(guard.request_release_bounded(runtime.unassign), flush=True)
        raise RuntimeError('Reconnect with Hardware accelerator None before rerunning.')
    drive.mount('/content/drive', timeout_ms=120000)
    root = Path('/content/drive/MyDrive/ATML_PA3_Task3')
    require(sha(root/'provenance/APPROVED_PLAN.json') == PLAN_SHA, 'Approval register mismatch')
    require(sha(root/'configs/MANUAL_REQUIREMENTS.json') == REQ_SHA, 'Manual configuration mismatch')
    marker = json.loads((root/'WORKSPACE.json').read_text())
    require(marker['plan_sha256']==PLAN_SHA and marker['requirements_sha256']==REQ_SHA, 'Workspace mismatch')
    req = json.loads((root/'configs/MANUAL_REQUIREMENTS.json').read_text())
    for package, version in [('gdown','5.2.2'), ('pyarrow','23.0.1')]:
        try: current=metadata.version(package)
        except metadata.PackageNotFoundError: current=None
        if current != version:
            subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check',f'{package}=={version}'],check=True,timeout=300)
        require(metadata.version(package)==version, 'Dependency version check failed')
    import gdown
    attempt = datetime.now(ZoneInfo('Asia/Karachi')).strftime('data_%Y%m%d_%H%M%S_')+uuid.uuid4().hex[:8]
    local = Path('/content/atml_pa3_task3')/attempt; local.mkdir(parents=True)
    require(shutil.disk_usage(local).free > 2*1024**3, 'Need at least 2 GiB local free space')
    source = root/'sources/supplied'; source.mkdir(exist_ok=True)
    receipts = root/'manifests/source_files'; receipts.mkdir(exist_ok=True)
    print('Listing the public supplied folder; no model or image downloads.',flush=True)
    listing = gdown.download_folder(url=SOURCE_URL,output=str(local/'listing'),skip_download=True,quiet=False,use_cookies=False)
    require(listing is not None, 'Could not list the supplied folder')
    items = {x.path.replace('\\','/'):x.id for x in listing if Path(x.path).name != '.DS_Store'}
    require(set(items)==EXPECTED and len(items)==len([x for x in listing if Path(x.path).name!='.DS_Store']), 'Unexpected source inventory; stop for review')
    guard.save_immutable(root/'manifests/supplied_inventory.json',guard.encode_json({'folder':SOURCE_URL,'files':items}))
    files=[]
    for relative in sorted(EXPECTED):
        dest=source/relative; receipt_path=receipts/(relative.replace('/','__')+'.json')
        if receipt_path.exists():
            receipt=json.loads(receipt_path.read_text())
            require(dest.is_file() and sha(dest)==receipt['sha256'] and receipt['drive_file_id']==items[relative], 'Saved source hash/identity mismatch')
            print('Verified existing:',relative,flush=True)
        else:
            require(not dest.exists(), 'Unreceipted existing source; preserving it for review')
            download=local/relative; download.parent.mkdir(parents=True,exist_ok=True)
            print('Downloading:',relative,flush=True)
            require(gdown.download(id=items[relative],output=str(download),quiet=False,use_cookies=False) is not None,'Download failed')
            require(download.stat().st_size>0,'Empty download')
            dest.parent.mkdir(parents=True,exist_ok=True)
            pending=dest.with_name(dest.name+'.pending_'+attempt)
            shutil.copyfile(download,pending)
            require(sha(download)==sha(pending),'Source copy readback failed')
            require(not dest.exists(),'Concurrent source publication detected')
            pending.rename(dest)
            require(sha(download)==sha(dest),'Final source hash mismatch')
            receipt={'relative_path':relative,'drive_file_id':items[relative],'bytes':dest.stat().st_size,'sha256':sha(dest),'origin':'first retrieval from supplied public folder; not instructor-signed checksum'}
            guard.save_immutable(receipt_path,guard.encode_json(receipt))
        files.append(receipt)
    readme=(source/'README.MD').read_text()
    compact=readme.replace(',','')
    for value in ['20742094','8961703','aa8e72537993ba99e69dfaafa59ed015b17504d1']:
        require(value in compact,'README content differs from reviewed package')
    results={}; schemas={}
    for dataset,folder in [('MATH','qwen25-math'),('MathVista','qwen25vl-mathvista')]:
        schema={}
        def rows(kind):
            path=source/folder/(kind+'.parquet')
            return parquet_rows(path,schema) if path.exists() else None
        print('Validating every row and token list:',dataset,flush=True)
        results[dataset]=validate_dataset(rows,reference=req['readme_reference_totals_unverified_against_files'][dataset])
        expected_model=req['datasets'][dataset]['model']
        # Exact supplied export labels observed in the checksum-verified Block 2D output.
        # Canonical identities come from the supplied README, not model inference.
        export_labels = {'MATH': ('qwen2_5_3b_instruct', 'math'),
                         'MathVista': ('qwen2_5_vl_3b', 'mathvista')}
        observed = (results[dataset]['model_identifier'], results[dataset]['dataset_identifier'])
        require(observed == export_labels[dataset], 'Unexpected supplied model/dataset export label')
        results[dataset]['canonical_model_from_readme'] = expected_model
        results[dataset]['identity_mapping_source'] = 'Supplied folder/README and verified Block 2D model/dataset labels; original fields preserved'
        schemas[dataset]=schema
    for receipt in files: require(sha(source/receipt['relative_path'])==receipt['sha256'],'Source changed during validation')
    report={'status':'CPU DATA VALIDATION COMPLETE','attempt':attempt,'workspace_drive':str(root),'files':files,'datasets':results,'schemas':schemas,'versions':{x:metadata.version(x) for x in ['gdown','pyarrow']},'correctness_computed':False,'held_out_scores_inspected':False,'splits_constructed':False,'models_loaded':False,'gpu_work_executed':False,'durability_scope':'Mounted-filesystem readback; no independent remote durability verification','next':'Share full output. Next: CPU metadata, scorer checks and deterministic split/order preparation.'}
    guard.save_immutable(root/'logs'/(attempt+'.json'),guard.encode_json(report))
    guard.save_immutable(root/'provenance'/(attempt+'_block_02.py'),(bundle/'block_02.py').read_bytes())
    print(json.dumps(report,indent=2),flush=True)
    print('CPU DATA VALIDATION COMPLETE — keep accelerator None and share the full output.',flush=True)
    return report
