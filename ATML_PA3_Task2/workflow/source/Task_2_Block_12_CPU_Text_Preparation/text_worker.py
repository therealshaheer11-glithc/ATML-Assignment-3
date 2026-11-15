"""CPU worker. Freeze verbatim Qwen layouts and collect cache-session evidence."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import argparse
import gzip
import hashlib
import importlib.metadata
import io
import json
import platform
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--attempt',required=True)
parser.add_argument('--package-sha',required=True)
args=parser.parse_args()
RUN_ROOT=Path('/content/drive/MyDrive/ATML_PA3/task2/run_20261031_150949_1cff148a')
LOCAL_ROOT=Path('/content/atml_pa3_task2/run_20261031_150949_1cff148a')
DATASET_SHA='82ad5006bd1d68ee54e639e50e7d835d8f6d97673537c38dab8d8eebdd8ac1b2'
FREEZE_SHA='fe7c9d4efa5d92d5b809eadb3695adcdd19f2421c6d43a2cb1593a7b86c3245b'
MODELS_SHA='66f14f21d911fb57ae379df700ed3161cf348a997922235d8ab8b56491604f7f'
FEATURE_SHA='1a015e4a07f2406937d63832dae1d648148a243a7931b39358d68e2a2ad8bea6'
FEATURE_RECIPE='380f19811df42cc56d3dfc3ecdf6496bae3557d11c27f1b2f64fd4b2a33bc2c6'
PINS={'transformers':'5.22.1','huggingface-hub':'1.33.0','tokenizers':'0.23.3',
    'safetensors':'0.8.0','Pillow':'11.3.0','numpy':'2.4.6'}
started=time.monotonic()
phase='startup'
from task2.session_io import bounded_io,save_json


def phase_message(text):
    global phase
    phase=text
    print('CPU TEXT PHASE:',text,flush=True)


def prepare_libraries():
    versions={}
    for name in PINS:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    if versions==PINS:
        return 'Exact existing library versions reused'
    tag=hashlib.sha256(json.dumps(PINS,sort_keys=True).encode()).hexdigest()[:16]
    target=LOCAL_ROOT/'text_python_packages'/tag
    marker=target/'complete.json'
    if not marker.exists():
        target.mkdir(parents=True,exist_ok=True)
        subprocess.run([sys.executable,'-m','pip','install','--quiet','--no-deps','--upgrade',
            '--target',str(target),*[f'{name}=={version}' for name,version in PINS.items()]],check=True,timeout=240)
        marker.write_text(json.dumps(PINS,sort_keys=True))
    else:
        assert json.loads(marker.read_text())==PINS
    sys.path.insert(0,str(target))
    assert {name:importlib.metadata.version(name) for name in PINS}==PINS
    return 'Pinned libraries loaded from an isolated local directory; PyTorch retained'


try:
    if os.environ.get('COLAB_GPU','0') not in ('','0'):
        raise RuntimeError('This worker requires a CPU runtime.')
    phase_message('preparing CPU libraries')
    package_setup=prepare_libraries()
    from transformers import AutoTokenizer
    from task2.dataset import load_dataset,validate_dataset
    from task2.images import encoded,file_sha,immutable_bytes,verified_copy
    from task2.text_data import build_text_data,build_feature_lookup,jsonl,load_text_tables
    from task2.interface_checks import run_interface_checks
    dataset_dir=RUN_ROOT/'datasets'/('task2_A012_'+DATASET_SHA[:16])
    phase_message('verifying the frozen dataset and completed feature manifest')
    with bounded_io(120,'Prerequisite verification'):
        pointer=json.loads((RUN_ROOT/'manifests/active_task2_dataset.json').read_bytes())
        assert pointer['dataset_identity']==DATASET_SHA and Path(pointer['dataset_dir'])==dataset_dir
        assert pointer['freeze_receipt_sha256']==FREEZE_SHA and file_sha(dataset_dir/'freeze_receipt.json')==FREEZE_SHA
        data=load_dataset(dataset_dir)
        assert data['identity']==DATASET_SHA
        data_summary=validate_dataset(data)
        fp=json.loads((RUN_ROOT/'manifests/active_task2_features.json').read_bytes())
        feature_path=dataset_dir/'features'/('siglip196_'+FEATURE_RECIPE[:16])/'features_complete.json'
        assert fp['dataset_identity']==DATASET_SHA and fp['recipe_sha256']==FEATURE_RECIPE
        assert Path(fp['manifest_path'])==feature_path and fp['manifest_sha256']==FEATURE_SHA
        assert file_sha(feature_path)==FEATURE_SHA
        features=json.loads(feature_path.read_bytes())
        assert features['status']=='passed' and features['image_count']==9260 and features['shard_count']==37
        assert features['recipe_sha256']==FEATURE_RECIPE and features['dataset_identity']==DATASET_SHA
        lookup=build_feature_lookup(features,data)
        assert len(lookup)==data_summary['cache_images']==9260
        mp=json.loads((RUN_ROOT/'manifests/active_task2_model_assets.json').read_bytes())
        model_path=dataset_dir/'inputs/model_assets'/('models_complete_'+MODELS_SHA[:16]+'.json')
        assert Path(mp['manifest_path'])==model_path and mp['manifest_sha256']==MODELS_SHA
        assert file_sha(model_path)==MODELS_SHA
        asset=json.loads(model_path.read_bytes())['models']['qwen']
        assert asset['repo_id']=='Qwen/Qwen3-0.6B' and asset['revision']=='c1899de289a04d12100db370d81485cdf75e47ca'
    phase_message('staging the existing pinned tokenizer files')
    tokenizer_dir=LOCAL_ROOT/'text_tokenizer'/asset['revision']
    names=('config.json','tokenizer_config.json','tokenizer.json','vocab.json','merges.txt')
    originals={row['file_name']:row for row in asset['files']}
    tokenizer_hashes={}
    with bounded_io(120,'Original tokenizer staging'):
        for name in names:
            record=originals[name]
            source=RUN_ROOT/'model_assets/qwen'/asset['revision']/'originals'/name
            verified_copy(source,tokenizer_dir/name,record['sha256'])
            tokenizer_hashes[name]=file_sha(tokenizer_dir/name)
    tokenizer=AutoTokenizer.from_pretrained(str(tokenizer_dir),local_files_only=True,trust_remote_code=False)
    assert tokenizer.is_fast and tokenizer.eos_token_id==151645 and tokenizer.pad_token_id==151643
    phase_message('preparing all original text layouts without truncation')
    built=build_text_data(data,tokenizer)
    expected=json.loads((Path(__file__).parent/'expected_text_hashes.json').read_bytes())
    for name,content in built['files'].items():
        assert hashlib.sha256(content).hexdigest()==expected['logical_file_hashes'][name], 'Text differs from locally verified tokenization: '+name
    assert tokenizer_hashes==expected['tokenizer_files']
    available={(row['source_split'],row['image_id']) for row in lookup}
    required={(row['source_split'],row['image_id']) for row in built['examples'] if 'image_id' in row}
    assert required==available
    assert all(('train2014',image_id) in available for image_id in built['calibration']['image_ids'])
    built['files']['feature_lookup.jsonl']=jsonl(lookup)
    print('Prepared',built['summary']['examples'],'records and',built['summary']['unique_text_layouts'],'unique layouts.',flush=True)
    phase_message('checking native Qwen masks, loss weighting and frozen-backbone gradients on CPU')
    interface=run_interface_checks(tokenizer)
    assert interface['status']=='passed'

    phase_message('collecting saved cache-session measurements')
    sessions=[]
    raw_sessions=[]
    with bounded_io(120,'Cache session evidence collection'):
        for path in sorted((RUN_ROOT/'logs').glob('block_11_[0-9]*.json')):
            if not re.fullmatch(r'block_11_\d{8}_\d{6}_[0-9a-f]{8}\.json',path.name):
                continue
            raw=path.read_bytes()
            row=json.loads(raw)
            if row.get('dataset_identity')!=DATASET_SHA or row.get('recipe_sha256')!=FEATURE_RECIPE:
                continue
            if row.get('status') not in ('passed','checkpointed'):
                continue
            fields=('status','images_verified','shards_verified','computed_images_this_run','run_activities',
                'elapsed_seconds','compute_seconds_cuda_events','preprocessing_and_transfer_seconds',
                'input_archive_staging_seconds','save_and_verification_seconds','peak_gpu_allocated_bytes',
                'peak_gpu_reserved_bytes','cache_package_sha256','environment')
            sessions.append({'file_name':path.name,'source_sha256':hashlib.sha256(raw).hexdigest(),
                **{key:row[key] for key in fields if key in row}})
            raw_sessions.append((path.name,raw))
    audit={'schema':'atml-pa3-task2-cache-session-audit-v1','dataset_identity':DATASET_SHA,
        'feature_recipe_sha256':FEATURE_RECIPE,'sessions':sessions,
        'creation_metrics_available':any(row.get('computed_images_this_run',0)>0 for row in sessions),
        'scope':'Saved worker timings and measured peaks only; verification runs do not establish creation throughput or billed allocation time.'}
    phase_message('saving verified text tables and evidence')
    recipe={'schema':'atml-pa3-task2-text-preparation-v1','dataset_identity':DATASET_SHA,
        'freeze_receipt_sha256':FREEZE_SHA,'feature_manifest_sha256':FEATURE_SHA,
        'model_manifest_sha256':MODELS_SHA,'tokenizer_files':tokenizer_hashes,
        'chat_template_sha256':built['protocol']['chat_template_sha256'],
        'logical_files':{name:hashlib.sha256(content).hexdigest() for name,content in sorted(built['files'].items())},
        'text_code_sha256':file_sha(Path(__file__).parent/'task2/text.py'),
        'text_data_code_sha256':file_sha(Path(__file__).parent/'task2/text_data.py'),
        'transformers':PINS['transformers'],'tokenizers':PINS['tokenizers']}
    recipe_sha=hashlib.sha256(encoded(recipe)).hexdigest()
    destination=dataset_dir/'inputs'/('text_'+recipe_sha[:16])
    members=[]
    with bounded_io(120,'Prepared text save and readback'):
        immutable_bytes(destination/'recipe.json',encoded(recipe))
        for name,raw in sorted(built['files'].items()):
            if name.endswith('.jsonl'):
                buffer=io.BytesIO()
                with gzip.GzipFile(fileobj=buffer,mode='wb',filename='',mtime=0,compresslevel=6) as stream:
                    stream.write(raw)
                content=buffer.getvalue()
                stored=name+'.gz'
            else:
                content,stored=raw,name
            immutable_bytes(destination/stored,content)
            saved=(destination/stored).read_bytes()
            assert (gzip.decompress(saved) if stored.endswith('.gz') else saved)==raw
            members.append({'path':stored,'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest(),
                'content_bytes':len(raw),'content_sha256':hashlib.sha256(raw).hexdigest()})
        audit_content=encoded(audit)
        audit_sha=hashlib.sha256(audit_content).hexdigest()
        audit_path=destination/('cache_session_audit_'+audit_sha[:16]+'.json')
        immutable_bytes(audit_path,audit_content)
        for name,raw in raw_sessions:
            immutable_bytes(destination/'cache_session_original_logs'/name,raw)
        complete={'schema':'atml-pa3-task2-text-complete-v1','status':'passed','dataset_identity':DATASET_SHA,
            'recipe_sha256':recipe_sha,'recipe':recipe,'files':members,'summary':built['summary']}
        complete_path=destination/'text_complete.json'
        immutable_bytes(complete_path,encoded(complete))
        complete_sha=file_sha(complete_path)
        readback=load_text_tables(destination,complete_sha)
        assert readback['calibration']==built['calibration'] and len(readback['by_layout'])==built['summary']['unique_text_layouts']
        immutable_bytes(RUN_ROOT/'manifests/active_task2_text.json',encoded({
            'schema':'atml-pa3-task2-active-text-v1','dataset_identity':DATASET_SHA,'recipe_sha256':recipe_sha,
            'manifest_path':str(complete_path),'manifest_sha256':complete_sha}))
    result={'block':'12','status':'passed','dataset_identity':DATASET_SHA,'feature_manifest_sha256':FEATURE_SHA,
        'text_manifest_path':str(complete_path),'text_manifest_sha256':complete_sha,'text_recipe_sha256':recipe_sha,
        'summary':built['summary'],'feature_lookup_images':len(lookup),'interface_checks':interface,
        'cache_sessions_found':len(sessions),'cache_creation_sessions_found':sum(row.get('computed_images_this_run',0)>0 for row in sessions),
        'cache_creation_metrics_available':audit['creation_metrics_available'],'cache_sessions':sessions,
        'cache_session_audit_path':str(audit_path),'cache_session_audit_sha256':audit_sha,
        'package_setup':package_setup,'text_package_sha256':args.package_sha,
        'elapsed_seconds':round(time.monotonic()-started,3),
        'environment':{'python':platform.python_version(),**{name:importlib.metadata.version(name) for name in (*PINS,'torch')}},
        'gpu_used':False,'full_Qwen_weights_loaded':False,'calibration_computed':False,'training_started':False}
    with bounded_io(30,'CPU result save'):
        immutable_bytes(RUN_ROOT/f'logs/block_12_{args.attempt}.json',encoded(result))
        immutable_bytes(destination/f'interface_checks_{args.attempt}.json',encoded(interface))
    print('BLOCK 12 PASSED',flush=True)
    print(json.dumps(result,indent=2),flush=True)
    print('Return the complete output before Qwen initialization and calibration on GPU.',flush=True)
except BaseException:
    failure={'block':'12','status':'failed','phase':phase,'traceback':traceback.format_exc(),
        'dataset_identity':DATASET_SHA,'text_package_sha256':args.package_sha,'gpu_used':False,'training_started':False}
    print(json.dumps(failure,indent=2),flush=True)
    try:
        with bounded_io(30,'CPU failure diagnostic save'):
            save_json(RUN_ROOT/f'logs/block_12_failed_{args.attempt}.json',failure)
    except BaseException as error:
        print('Failure diagnostic could not be saved:',repr(error),flush=True)
    raise
