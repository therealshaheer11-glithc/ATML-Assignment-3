"""Acquire and byte-verify pinned live-model assets while accelerator=None."""
import hashlib,importlib.util,json,platform,sys,uuid
from datetime import datetime
from importlib import metadata
from pathlib import Path
from zoneinfo import ZoneInfo

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def require(c,message):
    if not c:raise ValueError(message)

def prepare(root,bundle,guard,fetch=None):
    spec=importlib.util.spec_from_file_location('task3_download_assets',bundle/'download_assets.py');assets=importlib.util.module_from_spec(spec);spec.loader.exec_module(assets)
    manifest_bytes=(bundle/'MODEL_ASSET_MANIFEST.json').read_bytes();manifest=json.loads(manifest_bytes)
    require(manifest['repo']=='Qwen/Qwen2.5-3B-Instruct' and manifest['revision']=='aa8e72537993ba99e69dfaafa59ed015b17504d1','Unexpected model revision')
    inputs=json.loads((bundle/'EXISTING_INPUTS.json').read_text())
    for name,expected in inputs.items():require(sha(root/name)==expected,'Previously verified input changed: '+name)
    plan=json.loads((root/'configs/task3_live_call_plan.json').read_text());splits=json.loads((root/'manifests/task3_math_splits.json').read_text());questions=json.loads((root/'metadata/task3_math_questions.json').read_text());protocol=json.loads((root/'configs/MANUAL_REQUIREMENTS.json').read_text())['live']
    rows=plan['questions'];require(plan['unique_calls']==250 and len(rows)==50,'Wrong call plan')
    ids=[r['sample_id'] for r in rows];require(ids==splits['evaluation_ids'][:50] and len(set(ids))==50,'Live subset changed')
    live_inputs=[];all_seeds=[]
    for i,r in enumerate(rows):
        require(r['index']==i and r['caps']=={'parallel_1':1024,'parallel_2':1024,'parallel_3':1024,'long':3072,'revision':2048},'Call cap/index changed')
        order=['long','parallel_1','parallel_2','parallel_3','revision'] if i%2==0 else ['parallel_1','parallel_2','parallel_3','revision','long']
        require(r['call_order']==order and r['revision_initial_call']=='parallel_1','Scheduled ordering changed');all_seeds.extend(r['seeds'].values())
        q=questions[r['sample_id']]['question'];require(isinstance(q,str) and q.strip(),'Question text missing')
        live_inputs.append({'sample_id':r['sample_id'],'index':i,'initial_messages':[{'role':'system','content':protocol['system']},{'role':'user','content':q+protocol['problem_suffix']}],'revision_instruction':protocol['revision_instruction'],'call_plan':r})
    require(len(all_seeds)==250 and len(set(all_seeds))==250 and all(type(s)==int and 0<=s<2**32 for s in all_seeds),'Invalid/duplicated saved call seeds')
    folder=root/'live'/('model_'+manifest['revision']);require(not folder.is_symlink() and (not folder.exists() or folder.is_dir()),'Unexpected model folder');folder.mkdir(exist_ok=True)
    # Check existing final assets before beginning network work; preserve changed files.
    for e in manifest['files']:
        require(Path(e['name']).name==e['name'] and e['bytes']>0 and len(e['sha256'])==64,'Unsafe manifest member')
        p=folder/e['name']
        if p.exists() or p.is_symlink():assets.verify_file(p,e)
    guard.save_immutable(root/'configs/task3_live_model_asset_manifest.json',manifest_bytes)
    live_payload=guard.encode_json({'schema_version':1,'model_revision':manifest['revision'],'source_hashes':inputs,'question_count':50,'unique_call_count':250,'questions':live_inputs,'references_included':False,'native_template_application':'Pending CPU tokenizer preflight; messages are not yet tokenized','generation_performed':False})
    guard.save_immutable(root/'live/task3_live_inputs_v1.json',live_payload)
    completed=[]
    for e in sorted(manifest['files'],key=lambda e:(e['name'].endswith('.safetensors'),e['name'])):
        url=f"https://huggingface.co/{manifest['repo']}/resolve/{manifest['revision']}/{e['name']}"
        completed.append((fetch or assets.download)(e,folder/e['name'],url))
    index=assets.validate_small_assets(folder,manifest);tensor_audit=assets.inspect_safetensors(folder,index)
    # All final members read back again before publishing completion.
    for e in manifest['files']:assets.verify_file(folder/e['name'],e)
    versions={}
    for name in ['torch','transformers','accelerate','huggingface-hub','tokenizers','safetensors']:
        try:versions[name]=metadata.version(name)
        except metadata.PackageNotFoundError:versions[name]=None
    attempt=datetime.now(ZoneInfo('Asia/Karachi')).strftime('live_assets_%Y%m%d_%H%M%S_')+uuid.uuid4().hex[:8]
    receipt={'status':'CPU LIVE ASSETS COMPLETE','attempt':attempt,'repo':manifest['repo'],'revision':manifest['revision'],'asset_manifest_sha256':hashlib.sha256(manifest_bytes).hexdigest(),'asset_folder':str(folder),'verified_files':len(completed),'verified_bytes':manifest['total_bytes'],'live_inputs_sha256':hashlib.sha256(live_payload).hexdigest(),'scheduled_questions':50,'scheduled_unique_calls':250,'tensor_audit':tensor_audit,'python':sys.version,'platform':platform.platform(),'observed_versions_not_yet_frozen':versions,'gpu_work_executed':False,'model_loaded':False,'generation_performed':False,'software_installed':False,'gpu_watchdog_tested':False,'scoring_rerun':False,'policies_changed':False,'durability_scope':'Mounted-filesystem byte readback; remote durability not independently observed','next':'Share full output; keep accelerator None. Next CPU stage checks tokenizer/software and GPU watchdogs before any T4 session.'}
    for name in ['block_10.py','download_assets.py','MODEL_ASSET_MANIFEST.json','EXISTING_INPUTS.json','model_api.json']:guard.save_immutable(root/'provenance'/(attempt+'_'+name),(bundle/name).read_bytes())
    guard.save_immutable(root/'logs'/(attempt+'.json'),guard.encode_json(receipt));return receipt

def run(bundle):
    spec=importlib.util.spec_from_file_location('guard',bundle/'block_01.py');guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
    from google.colab import drive,runtime
    evidence=guard.accelerator_evidence()
    if evidence:
        print(guard.request_release_bounded(runtime.unassign),flush=True);raise RuntimeError('Reconnect with accelerator None')
    drive.mount('/content/drive',timeout_ms=120000)
    receipt=prepare(Path('/content/drive/MyDrive/ATML_PA3_Task3'),bundle,guard)
    print(json.dumps(receipt,indent=2),flush=True);print('CPU LIVE ASSETS COMPLETE — keep accelerator None and share full output.',flush=True);return receipt
