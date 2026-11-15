"""CPU-only verified export of saved results, source-backed examples and scorer audit."""
import hashlib,importlib.util,json,shutil,tempfile,uuid,zipfile
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

def run(bundle):
    spec=importlib.util.spec_from_file_location('guard',bundle/'block_01.py');guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
    from google.colab import drive,runtime,files
    if guard.accelerator_evidence():
        print(guard.request_release_bounded(runtime.unassign),flush=True);raise RuntimeError('Reconnect with accelerator None')
    drive.mount('/content/drive',timeout_ms=120000)
    import sys
    sys.path.insert(0,str(bundle))
    from export_results import file_hash,sha_bytes,require,timeout_audit,select_examples
    import pyarrow.dataset as ds
    root=Path('/content/drive/MyDrive/ATML_PA3_Task3')
    ref=json.loads((bundle/'HELDOUT_REFERENCE.json').read_text());dev=json.loads((bundle/'DEVELOPMENT_REFERENCE.json').read_text())
    require(json.loads((root/'logs'/(ref['attempt']+'.json')).read_text())==ref,'Held-out receipt differs')
    require(file_hash(root/'results/task3_heldout_v1/heldout_summary.json')==ref['summary_sha256'],'Held-out summary differs')
    require(file_hash(root/'results/task3_heldout_v1/question_manifest.json')==ref['manifest_sha256'],'Held-out manifest differs')
    require(file_hash(root/'configs/task3_adaptive_selected.json')==dev['thresholds_sha256'],'Frozen thresholds differ')
    require(file_hash(root/'results/task3_development_v1/development_summary.json')==dev['summary_sha256'],'Development summary differs')
    selected=json.loads((root/'configs/task3_adaptive_selected.json').read_text())
    manifest=json.loads((root/'results/task3_heldout_v1/question_manifest.json').read_text())
    members={};heldout={'MATH':[],'MathVista':[]}
    def include(relative,expected=None):
        path=root/relative;actual=file_hash(path)
        require(expected is None or actual==expected,'Export source hash differs: '+relative)
        members[relative]={'path':path,'sha256':actual,'bytes':path.stat().st_size}
    for split,files_map in [('heldout',manifest['question_files']),('development',selected['development_question_manifest'])]:
        require(len(files_map)==(1600 if split=='heldout' else 400),'Incomplete question manifest')
        recipe_path=f'results/task3_{split}_v1/recipe.json';recipe_sha=file_hash(root/recipe_path);include(recipe_path)
        for key,expected in files_map.items():
            dataset,q=key.split('/',1);label='math' if dataset=='MATH' else 'mathvista';relative=f'results/task3_{split}_v1/{label}/'+sha_bytes(q.encode())+'.json'
            include(relative,expected);envelope=json.loads((root/relative).read_text());record=envelope['record']
            require(sha_bytes(guard.encode_json(record))==envelope['record_sha256'] and record['recipe_sha256']==recipe_sha and record['sample_id']==q and record['dataset']==dataset,'Saved question envelope differs')
            if split=='heldout':heldout[dataset].append(record)
        print('Verified '+split+' question records: '+str(len(files_map)),flush=True)
    worker_log='logs/'+ref['attempt']+'_worker.txt';include(worker_log)
    audit=timeout_audit((root/worker_log).read_text())
    examples={d:select_examples(records,selected['selected'][d],d) for d,records in heldout.items()}
    needed={d:set() for d in heldout}
    for d,example in examples.items():
        for cases in example['categories'].values():needed[d].update(c['sample_id'] for c in cases)
    for case in audit['question_intervals']:needed[case['dataset']].add(case['sample_id'])
    for item in json.loads((root/'configs/task3_fixed_trajectories.json').read_text()):
        needed['MathVista' if item['dataset']=='mathvista' else 'MATH'].add(item['sample_id'])
    text_examples={}
    source_ref=json.loads((root/'provenance/task3_validated_source_reference.json').read_text())
    source_hashes={e['relative_path']:e['sha256'] for e in source_ref['files']}
    for dataset,folder in [('MATH','qwen25-math'),('MathVista','qwen25vl-mathvista')]:
        relative=folder+'/generations.parquet';path=root/'sources/supplied'/relative
        require(file_hash(path)==source_hashes[relative],'Original generation source differs')
        scan=ds.dataset(path,format='parquet').scanner(filter=ds.field('sample_id').isin(sorted(needed[dataset])),columns=['dataset','model','sample_id','rollout_id','question_text','generated_text','extracted_answer','gold_answer','hit_max_new_tokens','n_generated_text_tokens'],batch_size=128)
        rows=[]
        for batch in scan.to_batches():rows.extend(batch.to_pylist())
        require(len(rows)==32*len(needed[dataset]),'Missing example rollouts')
        text_examples[dataset]=rows
    for directory in ['configs','manifests','metadata','provenance']:
        for path in (root/directory).iterdir():
            if path.is_file() and '.pending_' not in path.name:include(str(path.relative_to(root)))
    for relative in ['results/task3_heldout_v1/heldout_summary.json','results/task3_heldout_v1/question_manifest.json','results/task3_development_v1/development_summary.json','logs/'+ref['attempt']+'.json','logs/'+dev['attempt']+'.json']:include(relative)
    attempt=datetime.now(ZoneInfo('Asia/Karachi')).strftime('offline_export_%Y%m%d_%H%M%S_')+uuid.uuid4().hex[:8]
    local=Path(tempfile.mkdtemp(prefix='task3_offline_export_'));archive_path=local/(attempt+'.zip')
    extra={'audit/comparison_timeouts.json':guard.encode_json(audit),'audit/example_selection.json':guard.encode_json(examples),'audit/source_backed_example_rollouts.json':guard.encode_json(text_examples),'audit/HELDOUT_REFERENCE.json':(bundle/'HELDOUT_REFERENCE.json').read_bytes(),'audit/DEVELOPMENT_REFERENCE.json':(bundle/'DEVELOPMENT_REFERENCE.json').read_bytes(),'audit/block_09.py':(bundle/'block_09.py').read_bytes(),'audit/export_results.py':(bundle/'export_results.py').read_bytes()}
    inventory={name:{'bytes':r['bytes'],'sha256':r['sha256']} for name,r in members.items()}
    inventory.update({name:{'bytes':len(data),'sha256':sha_bytes(data)} for name,data in extra.items()})
    with zipfile.ZipFile(archive_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for name,item in members.items():archive.write(item['path'],name)
        for name,data in extra.items():archive.writestr(name,data)
        archive.writestr('export_manifest.json',guard.encode_json({'schema_version':1,'attempt':attempt,'members':inventory,'purpose':'Read-only offline evidence export for analysis/visual review; no scoring reruns or threshold changes','missing_scope':'Live-generation experiment not yet run'}))
    # Independently decode/check every archive member before publishing/downloading.
    with zipfile.ZipFile(archive_path) as archive:
        require(set(archive.namelist())==set(inventory)|{'export_manifest.json'},'Archive inventory differs')
        for name,entry in inventory.items():require(sha_bytes(archive.read(name))==entry['sha256'],'Archive member differs: '+name)
    target=root/'exports'/archive_path.name
    pending=target.with_name(target.name+'.pending');shutil.copyfile(archive_path,pending)
    require(file_hash(pending)==file_hash(archive_path),'Drive ZIP copy differs');pending.rename(target)
    require(file_hash(target)==file_hash(archive_path),'Final Drive ZIP readback differs')
    report={'status':'CPU OFFLINE EXPORT COMPLETE','attempt':attempt,'drive_zip':str(target),'zip_bytes':target.stat().st_size,'zip_sha256':file_hash(target),'verified_question_records':2000,'timeout_audit':audit,'example_question_counts':{d:len(q) for d,q in needed.items()},'scoring_rerun':False,'threshold_changes':False,'gpu_work_executed':False,'durability_scope':'Mounted-filesystem byte readback; remote durability not independently observed','next':'Attach the downloaded ZIP in chat for audit, question-bootstrap intervals, plots and actual visual review. Live GPU generation remains pending.'}
    guard.save_immutable(root/'logs'/(attempt+'.json'),guard.encode_json(report))
    print(json.dumps(report,indent=2),flush=True)
    print('CPU OFFLINE EXPORT COMPLETE — download the ZIP and attach it in chat.',flush=True)
    try:
        files.download(str(archive_path))
    except Exception as error:
        print('Download request failed ('+type(error).__name__+'). Verified ZIP remains at: '+str(target),flush=True)
    return report
