"""CPU development execution, per-question recovery and threshold freeze."""
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
    recipe={'implementation_id':'task3_development_v1','inputs':inputs,'code':{n:sha(bundle/n) for n in ['block_07.py','development.py','dev_worker.py','scoring.py','selection.py','scorer_worker.py']},'split':'development only,200 per dataset,10 orders each','environment':scorer['installed_source_fingerprints'],'missing_and_capped':'Retained; no filtering','held_out_evaluation':False}
    work=root/'results/task3_development_v1';work.mkdir(exist_ok=True)
    guard.save_immutable(work/'recipe.json',guard.encode_json(recipe))
    lock=root/'configs/task3_scorer_requirements.lock';lock_sha=sha(lock)
    envbase=Path('/content/atml_pa3_task3');envbase.mkdir(parents=True,exist_ok=True);env=envbase/('scorer_env_'+lock_sha[:16])
    if not env.exists():
        pending=envbase/('scorer_pending_'+uuid.uuid4().hex)
        subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','--only-binary=:all:','--require-hashes','--no-deps','--no-compile','--target',str(pending),'-r',str(lock)],check=True,timeout=600)
        (pending/'.task3_lock_sha256').write_text(lock_sha);pending.rename(env)
    require((env/'.task3_lock_sha256').read_text()==lock_sha,'Scorer environment is incomplete')
    attempt=datetime.now(ZoneInfo('Asia/Karachi')).strftime('development_%Y%m%d_%H%M%S_')+uuid.uuid4().hex[:8]
    command=[sys.executable,'-I',str(bundle/'dev_worker.py'),'--env',str(env),'--root',str(root),'--bundle',str(bundle)]
    proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    timer=threading.Timer(1800,proc.kill);timer.daemon=True;timer.start();lines=[]
    try:
        for line in proc.stdout:print(line,end='',flush=True);lines.append(line)
        code=proc.wait()
    finally:timer.cancel()
    guard.save_immutable(root/'logs'/(attempt+'_worker.txt'),''.join(lines).encode())
    require(code==0,'Development worker failed/timed out; completed question saves preserved. Share full output.')
    # Parent independently verifies every persisted question before publishing thresholds.
    sys.path.insert(0,str(bundle))
    from development import aggregate,filename
    from dev_worker import read_question
    summaries={};manifest={};done=0
    recipe_sha=sha(work/'recipe.json')
    for dataset,label in [('MATH','math'),('MathVista','mathvista')]:
        ids=json.loads((root/f'manifests/task3_{label}_splits.json').read_text())['development_ids']
        records=[]
        for q in ids:
            path=work/label/filename(q)
            if path.exists():records.append(read_question(path,recipe_sha,q,guard));manifest[dataset+'/'+q]=sha(path)
        done+=len(records)
        if len(records)==200:summaries[dataset]=aggregate(records)
    report={'attempt':attempt,'questions_verified':done,'total_questions':400,'held_out_scores_inspected':False,'gpu_work_executed':False}
    if done!=400:
        report.update(status='CPU DEVELOPMENT PARTIAL',next='Repeat this same Block7 ZIP/loader with accelerator None after reviewing output; verified questions are reused.')
    else:
        thresholds={'recipe_sha256':recipe_sha,'development_question_manifest':manifest,'selected':{d:s['selected_policies'] for d,s in summaries.items()},'selection_rule':'Lowest token cost within1pp of fixed32 SC; higher accuracy, smaller tau/gamma; always32 fallback; exact2000-observation counts','held_out_evaluation_started':False}
        guard.save_immutable(work/'development_summary.json',guard.encode_json(summaries))
        guard.save_immutable(root/'configs/task3_adaptive_selected.json',guard.encode_json(thresholds))
        report.update(status='CPU DEVELOPMENT COMPLETE',selected_policies=thresholds['selected'],thresholds_sha256=sha(root/'configs/task3_adaptive_selected.json'),summary_sha256=sha(work/'development_summary.json'),next='Share full output; held-out evaluation uses these frozen settings without tuning.')
    for name in ['block_07.py','development.py','dev_worker.py']:guard.save_immutable(root/'provenance'/(attempt+'_'+name),(bundle/name).read_bytes())
    guard.save_immutable(root/'logs'/(attempt+'.json'),guard.encode_json(report));return report

def run(bundle):
    spec=importlib.util.spec_from_file_location('guard',bundle/'block_01.py');guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
    from google.colab import drive,runtime
    if guard.accelerator_evidence():
        print(guard.request_release_bounded(runtime.unassign),flush=True);raise RuntimeError('Reconnect with accelerator None')
    drive.mount('/content/drive',timeout_ms=120000)
    report=prepare(Path('/content/drive/MyDrive/ATML_PA3_Task3'),bundle,guard)
    print(json.dumps(report,indent=2),flush=True);print(report['status']+' — keep accelerator None and share full output.',flush=True);return report
