"""CPU held-out execution with frozen policies, per-question recovery and signal diagnostics."""
import hashlib,importlib.util,json,subprocess,sys,threading,uuid
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(4*1024*1024),b''):h.update(part)
    return h.hexdigest()
def require(c,message):
    if not c:raise ValueError(message)

def prepare(root,bundle,guard):
    development_reference=json.loads((bundle/'DEVELOPMENT_REFERENCE.json').read_text())
    require(json.loads((root/'logs'/(development_reference['attempt']+'.json')).read_text())==development_reference,'Development completion receipt differs')
    require(sha(root/'configs/task3_adaptive_selected.json')==development_reference['thresholds_sha256']=='1ae6c94a10fba0d3ee2e4eab192e7f131244ca8c3bb1f7afc4e3c97427e7ce20','Frozen policies changed')
    require(sha(root/'results/task3_development_v1/development_summary.json')==development_reference['summary_sha256'],'Development summary changed')
    thresholds=json.loads((root/'configs/task3_adaptive_selected.json').read_text())
    from hashlib import sha256
    for key,h in thresholds['development_question_manifest'].items():
        dataset,q=key.split('/',1);label='math' if dataset=='MATH' else 'mathvista'
        require(sha(root/'results/task3_development_v1'/label/(sha256(q.encode()).hexdigest()+'.json'))==h,'Development question changed')
    reference=json.loads((bundle/'SELECTION_REFERENCE.json').read_text())
    require(json.loads((root/'logs'/(reference['attempt']+'.json')).read_text())==reference,'Selection receipt differs')
    for e in reference['saved_artifacts']:require(sha(root/e['relative_path'])==e['sha256'],'Verified selector differs')
    core=json.loads((root/'configs/task3_selection_verified.json').read_text());scorer=json.loads((root/'configs/task3_scorer_frozen.json').read_text())
    require(core['selection_sha256']==sha(bundle/'selection.py') and scorer['specification']['module_sha256']==sha(bundle/'scoring.py')==core['scorer_sha256'],'Packaged numerical source differs')
    inputs={}
    for e in json.loads((bundle/'PARTITION_REFERENCE.json').read_text())['saved_artifacts']:
        inputs[e['relative_path']]=e['sha256']
    inputs.update(scorer['reference_metadata_sha256'])
    for e in json.loads((bundle/'VALIDATED_SOURCE_REFERENCE.json').read_text())['files']:inputs['sources/supplied/'+e['relative_path']]=e['sha256']
    inputs['configs/task3_scorer_frozen.json']='6e97a2379450ba34714d0dce027ce337364dc2ce438a49ec46c9b75b0f219467'
    for name,h in inputs.items():require(sha(root/name)==h,'Input bytes differ: '+name)
    inputs['configs/task3_adaptive_selected.json']=development_reference['thresholds_sha256']
    recipe={'implementation_id':'task3_heldout_v1','inputs':inputs,'code':{n:sha(bundle/n) for n in ['block_08.py','development.py','dev_worker.py','evaluation.py','eval_worker.py','scoring.py','selection.py','scorer_worker.py']},'split':'held-out only,800 per dataset,10 orders each; policies already frozen','environment':scorer['installed_source_fingerprints'],'missing_and_capped':'Retained; no filtering','held_out_evaluation':True,'threshold_search':False}
    work=root/'results/task3_heldout_v1';work.mkdir(exist_ok=True)
    guard.save_immutable(work/'recipe.json',guard.encode_json(recipe))
    lock=root/'configs/task3_scorer_requirements.lock';lock_sha=sha(lock)
    envbase=Path('/content/atml_pa3_task3');envbase.mkdir(parents=True,exist_ok=True);env=envbase/('scorer_env_'+lock_sha[:16])
    if not env.exists():
        pending=envbase/('scorer_pending_'+uuid.uuid4().hex)
        subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','--only-binary=:all:','--require-hashes','--no-deps','--no-compile','--target',str(pending),'-r',str(lock)],check=True,timeout=600)
        (pending/'.task3_lock_sha256').write_text(lock_sha);pending.rename(env)
    require((env/'.task3_lock_sha256').read_text()==lock_sha,'Scorer environment is incomplete')
    attempt=datetime.now(ZoneInfo('Asia/Karachi')).strftime('heldout_%Y%m%d_%H%M%S_')+uuid.uuid4().hex[:8]
    command=[sys.executable,'-I',str(bundle/'eval_worker.py'),'--env',str(env),'--root',str(root),'--bundle',str(bundle)]
    proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    timer=threading.Timer(1800,proc.kill);timer.daemon=True;timer.start();lines=[]
    try:
        for line in proc.stdout:print(line,end='',flush=True);lines.append(line)
        code=proc.wait()
    finally:timer.cancel()
    guard.save_immutable(root/'logs'/(attempt+'_worker.txt'),''.join(lines).encode())
    require(code==0,'Held-out worker failed/timed out; completed question saves preserved. Share full output.')
    # Parent independently verifies every persisted question; thresholds remain unchanged.
    sys.path.insert(0,str(bundle))
    from development import filename
    from evaluation import aggregate_fixed
    from dev_worker import read_question
    summaries={};manifest={};done=0
    recipe_sha=sha(work/'recipe.json')
    for dataset,label in [('MATH','math'),('MathVista','mathvista')]:
        ids=json.loads((root/f'manifests/task3_{label}_splits.json').read_text())['evaluation_ids']
        records=[]
        for q in ids:
            path=work/label/filename(q)
            if path.exists():records.append(read_question(path,recipe_sha,q,guard));manifest[dataset+'/'+q]=sha(path)
        done+=len(records)
        if len(records)==800:summaries[dataset]=aggregate_fixed(records)
    report={'attempt':attempt,'questions_verified':done,'total_questions':1600,'held_out_scores_inspected':True,'threshold_selection_performed':False,'gpu_work_executed':False}
    if done!=1600:
        report.update(status='CPU HELD-OUT PARTIAL',next='Repeat this same Block8 ZIP/loader with accelerator None after reviewing output; verified questions are reused.')
    else:
        require(sha(root/'configs/task3_adaptive_selected.json')==development_reference['thresholds_sha256'],'Frozen policies changed during evaluation')
        guard.save_immutable(work/'heldout_summary.json',guard.encode_json(summaries))
        manifest_record={'recipe_sha256':recipe_sha,'thresholds_sha256':development_reference['thresholds_sha256'],'question_files':manifest,'threshold_selection_performed':False}
        guard.save_immutable(work/'question_manifest.json',guard.encode_json(manifest_record))
        report.update(status='CPU HELD-OUT COMPLETE',results={d:{'fixed32_SC':next(r for r in x['fixed_budget_results'] if r['N']==32 and r['method']=='SC'),'adaptive':x['adaptive_results']} for d,x in summaries.items()},summary_sha256=sha(work/'heldout_summary.json'),manifest_sha256=sha(work/'question_manifest.json'),next='Share full output; next CPU stage computes question-bootstrap intervals, signal summaries and plots. GPU live experiment remains pending.')
    for name in ['block_08.py','evaluation.py','eval_worker.py']:guard.save_immutable(root/'provenance'/(attempt+'_'+name),(bundle/name).read_bytes())
    guard.save_immutable(root/'logs'/(attempt+'.json'),guard.encode_json(report));return report

def run(bundle):
    spec=importlib.util.spec_from_file_location('guard',bundle/'block_01.py');guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
    from google.colab import drive,runtime
    if guard.accelerator_evidence():
        print(guard.request_release_bounded(runtime.unassign),flush=True);raise RuntimeError('Reconnect with accelerator None')
    drive.mount('/content/drive',timeout_ms=120000)
    report=prepare(Path('/content/drive/MyDrive/ATML_PA3_Task3'),bundle,guard)
    print(json.dumps(report,indent=2),flush=True);print(report['status']+' — keep accelerator None and share full output.',flush=True);return report
