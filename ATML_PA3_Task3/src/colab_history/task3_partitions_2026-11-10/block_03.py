"""Save outcome-independent split/order/live-call plans and untruncated source metadata."""
import base64
import hashlib
import importlib.util
import json
import uuid
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PLAN_SHA='f6c48fc63de87286cd87d1029b823670ce085af2cc41a3f175f4d76ea3f90c02'
REQ_SHA='60904f8e38bfe213cc2ed2c3b59af9be4ff01be42c27b223dda3588282d7744d'
RECEIPT='data_20261110_102549_0f278770.json'

def require(condition,message):
    if not condition:raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def make_plan(question_ids,dataset_index,n_development=200,n_live=50):
    """This function accepts IDs only: answers, scores and lengths cannot affect the plan."""
    import numpy as np
    ids=sorted(question_ids)
    require(len(ids)==len(set(ids)) and all(isinstance(x,str) for x in ids),'Unique string IDs required')
    require(n_development>0 and len(ids)>n_development,'Invalid split size')
    perm=np.random.default_rng(6304).permutation(len(ids)).tolist()
    ordered=[ids[i] for i in perm]
    split={'lexicographic_ids':ids,'permutation_indices':perm,'development_ids':ordered[:n_development],'evaluation_ids':ordered[n_development:]}
    orders={}
    # Distinct deterministic stream per dataset, sorted question position and order.
    for q_index,q in enumerate(ids):
        orders[q]=[np.random.default_rng(np.random.SeedSequence([6304,1,dataset_index,q_index,o])).permutation(32).tolist() for o in range(10)]
    require(not set(split['development_ids']) & set(split['evaluation_ids']),'Split overlap')
    live=[]
    if dataset_index==0:
        require(len(split['evaluation_ids'])>=n_live,'Not enough evaluation questions')
        seen_seeds=set()
        slots=['parallel_1','parallel_2','parallel_3','long','revision']
        for index,q in enumerate(split['evaluation_ids'][:n_live]):
            seeds={name:int(np.random.SeedSequence([6304,2,index,slot]).generate_state(1,dtype=np.uint32)[0]) for slot,name in enumerate(slots)}
            require(not seen_seeds & set(seeds.values()) and len(set(seeds.values()))==5,'Live seed collision; stop for review')
            seen_seeds.update(seeds.values())
            parallel_revision=['parallel_1','parallel_2','parallel_3','revision']
            schedule=['long']+parallel_revision if index%2==0 else parallel_revision+['long']
            live.append({'sample_id':q,'index':index,'seeds':seeds,'call_order':schedule,'caps':{'parallel_1':1024,'parallel_2':1024,'parallel_3':1024,'long':3072,'revision':2048},'revision_initial_call':'parallel_1'})
    return split,orders,live

def export_metadata(path):
    import pyarrow.parquet as pq
    file=pq.ParquetFile(path)
    metadata={}
    for k,v in (file.schema_arrow.metadata or {}).items():
        # Raw base64 preserves every metadata byte; UTF-8 text is for inspection.
        try:text=v.decode('utf-8')
        except UnicodeDecodeError:text=None
        metadata[k.decode('utf-8')]={'utf8':text,'base64':base64.b64encode(v).decode('ascii')}
    return {'rows':file.metadata.num_rows,'fields':[{'name':f.name,'type':str(f.type),'nullable':f.nullable} for f in file.schema_arrow],'metadata':metadata}

def prepare(root,bundle,guard,required_numpy='2.4.6'):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from importlib import metadata
    require(np.__version__==required_numpy,'Unexpected NumPy version: preserve reproducibility; share output before continuing')
    reference=json.loads((bundle/'VALIDATED_SOURCE_REFERENCE.json').read_text())
    require(json.loads((root/'logs'/RECEIPT).read_text())==reference,'Successful source-validation receipt differs from observed reference')
    require(sha(root/'provenance/APPROVED_PLAN.json')==PLAN_SHA,'Approval mismatch')
    require(sha(root/'configs/MANUAL_REQUIREMENTS.json')==REQ_SHA,'Manual requirements mismatch')
    metadata_all={}
    for entry in reference['files']:
        relative=entry['relative_path'];path=root/'sources/supplied'/relative
        require(path.is_file() and path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256'],'Source bytes differ: '+relative)
        receipt_path=root/'manifests/source_files'/(relative.replace('/','__')+'.json')
        require(json.loads(receipt_path.read_text())==entry,'Acquisition receipt differs: '+relative)
        if relative.endswith('.parquet'):metadata_all[relative]=export_metadata(path)
    sources=root/'sources/supplied'
    outputs={};summaries={};live_plan=[];trajectories=[]
    for dataset_index,(dataset,folder,label) in enumerate([('MATH','qwen25-math','math'),('MathVista','qwen25vl-mathvista','mathvista')]):
        # Only identifying columns are read. No answers, text, lengths or scores.
        groups={}
        for batch in pq.ParquetFile(sources/folder/'generations.parquet').iter_batches(batch_size=512,columns=['dataset','sample_id','rollout_id']):
            for row in batch.to_pylist():
                require(row['dataset']==label,'Dataset label mismatch')
                q=row['sample_id'];r=row['rollout_id']
                require(isinstance(q,str) and isinstance(r,int) and r not in groups.setdefault(q,set()),'Duplicate/invalid question key')
                groups[q].add(r)
        require(len(groups)==1000 and all(v==set(range(32)) for v in groups.values()),'Question/rollout universe changed')
        if dataset=='MATH':require(all(q.startswith('test:') and q[5:].isdigit() for q in groups),'Unexpected MATH ID format')
        split,orders,live=make_plan(list(groups),dataset_index)
        outputs['manifests/task3_'+label+'_splits.json']=split
        outputs['manifests/task3_'+label+'_orders.json']=orders
        summaries[dataset]={'development':len(split['development_ids']),'evaluation':len(split['evaluation_ids']),'questions_with_ten_orders':len(orders),'first_five_evaluation_ids':split['evaluation_ids'][:5]}
        if live:live_plan=live
        if dataset=='MathVista':trajectories=[{'dataset':'mathvista','sample_id':q,'rollout_id':r} for q in split['evaluation_ids'][:4] for r in [0,1]]
    outputs['configs/task3_live_call_plan.json']={'questions':live_plan,'unique_calls':250,'shared_initial_response':'parallel_1','status':'IDs/seeds/schedule fixed; no generation performed'}
    outputs['configs/task3_fixed_trajectories.json']=trajectories
    outputs['provenance/task3_full_export_metadata.json']=metadata_all
    outputs['configs/task3_randomness.json']={'seed':6304,'numpy':np.__version__,'bit_generator':np.random.default_rng(6304).bit_generator.__class__.__name__,'split':'Fresh default_rng(6304) per dataset; permute lexically sorted string IDs','dataset_indices':{'MATH':0,'MathVista':1},'orders':'default_rng(SeedSequence([6304,1,dataset_index,sorted_question_index,order_index])).permutation(32), order_index=0..9','live_seeds':'SeedSequence([6304,2,live_question_index,call_slot]).generate_state(1,uint32); slots parallel1,parallel2,parallel3,long,revision','bootstrap_reserved_streams':{'offline':'default_rng(SeedSequence([6304,3,dataset_index]))','live':'default_rng(SeedSequence([6304,4]))'},'budgets':[1,2,4,8,16,32],'prefix_rule':'Consume nested prefixes of the same stored order for every selector/controller','outcome_independent':True}
    outputs['configs/task3_preparation_status.json']={'status':'Split/order/seed plans fixed','scorer_frozen':False,'adaptive_thresholds_frozen':False,'held_out_results_inspected':False,'next':'CPU official question metadata acquisition and scorer verification; development selection before held-out analysis'}
    # Preflight every immutable target first; a rerun never replaces an existing decision.
    payloads={name:guard.encode_json(value) for name,value in outputs.items()}
    for name,payload in payloads.items():
        path=root/name
        require(not path.is_symlink() and (not path.exists() or path.read_bytes()==payload),'Existing plan differs; preserving it: '+name)
    saved=[]
    for name,payload in payloads.items():
        entry=guard.save_immutable(root/name,payload);entry['relative_path']=name;saved.append(entry)
    attempt=datetime.now(ZoneInfo('Asia/Karachi')).strftime('partitions_%Y%m%d_%H%M%S_')+uuid.uuid4().hex[:8]
    guard.save_immutable(root/'provenance'/(attempt+'_block_03.py'),(bundle/'block_03.py').read_bytes())
    guard.save_immutable(root/'provenance/task3_validated_source_reference.json',(bundle/'VALIDATED_SOURCE_REFERENCE.json').read_bytes())
    report={'status':'CPU PARTITION PREPARATION COMPLETE','attempt':attempt,'workspace_drive':str(root),'source_validation_attempt':reference['attempt'],'datasets':summaries,'saved_artifacts':saved,'live_questions':len(live_plan),'live_unique_calls_planned':sum(len(x['call_order']) for x in live_plan),'fixed_illustrative_trajectories':len(trajectories),'versions':{'numpy':np.__version__,'pyarrow':pa.__version__},'metadata_scope':'Untruncated UTF-8 and raw bytes preserved; supplied VLM local revision retained without claiming a public model commit','scoring_executed':False,'scorer_frozen':False,'adaptive_thresholds_selected':False,'held_out_scores_inspected':False,'data_or_model_downloads':False,'models_loaded':False,'gpu_work_executed':False,'durability_scope':'Mounted-filesystem byte readback; no independent remote durability verification','next':'Share full output; next CPU block obtains official metadata and verifies the frozen scorer specification.'}
    guard.save_immutable(root/'logs'/(attempt+'.json'),guard.encode_json(report))
    return report

def run(bundle):
    spec=importlib.util.spec_from_file_location('task3_guard',bundle/'block_01.py')
    guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
    from google.colab import drive,runtime
    if guard.accelerator_evidence():
        print('CPU-only task: requesting accelerator release.',flush=True)
        print(guard.request_release_bounded(runtime.unassign),flush=True)
        raise RuntimeError('Reconnect with accelerator None; share output if release fails.')
    drive.mount('/content/drive',timeout_ms=120000)
    report=prepare(Path('/content/drive/MyDrive/ATML_PA3_Task3'),bundle,guard)
    print(json.dumps(report,indent=2),flush=True)
    print('CPU PARTITION PREPARATION COMPLETE — keep accelerator None and share full output.',flush=True)
    return report
