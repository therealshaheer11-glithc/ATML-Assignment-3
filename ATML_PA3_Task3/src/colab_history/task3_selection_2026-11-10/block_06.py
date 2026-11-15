"""CPU verification/publication of quality, selector and stopping implementations."""
import hashlib,importlib.util,json,subprocess,sys,tempfile,uuid
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(condition,message):
    if not condition:raise ValueError(message)

def prepare(root,bundle,guard,required_numpy='2.4.6'):
    reference=json.loads((bundle/'SCORER_REFERENCE.json').read_text())
    require(json.loads((root/'logs'/(reference['attempt']+'.json')).read_text())==reference,'Scorer receipt differs')
    for entry in reference['saved_artifacts']:require(sha(root/entry['relative_path'])==entry['sha256'],'Frozen scorer artifact differs: '+entry['relative_path'])
    frozen=json.loads((root/'configs/task3_scorer_frozen.json').read_text())
    require(frozen['scorer_frozen'] and frozen['specification']['module_sha256']==sha(bundle/'scoring.py')==reference['scorer_sha256'],'Wrong answer scorer source')
    local=Path(tempfile.mkdtemp(prefix='task3_selection_checks_'));out=local/'result.json'
    process=subprocess.run([sys.executable,'-I','-c',"import runpy,sys;sys.path.insert(0,sys.argv[1]);sys.argv=[sys.argv[1]+'/selection_checks.py',sys.argv[2]];runpy.run_path(sys.argv[0],run_name='__main__')",str(bundle),str(out)],capture_output=True,text=True,timeout=120)
    print(process.stdout+process.stderr,flush=True)
    require(process.returncode==0 and out.is_file(),'Selection checks failed; share output')
    result=json.loads(out.read_text())
    require(result['checks_passed']>=52 and result['numpy_version']==required_numpy,'Checks incomplete or NumPy differs from frozen protocol')
    core={'implementation_id':'task3_selection_v1','selection_sha256':sha(bundle/'selection.py'),'scorer_sha256':sha(bundle/'scoring.py'),'checks_sha256':sha(bundle/'selection_checks.py'),'numpy_version':result['numpy_version'],'checks_passed':result['checks_passed'],
          'scores':'Raw per-rollout mean logp/KL; eligible-mass normalized vision; pooled prefix token KL Q05/Q95 linear calibration; Z=mean(V)*mean(C), Ztoken=mean(V*C). No future calibration.',
          'votes':'Exact score ties get descending average ranks; RWC=(N-r)/(N-1), N1 sole candidate. Common integer rescaling of rank weights used for stable identical margins/winners. Final vote ties use earliest current supporting rollout.',
          'cost':'FirstAns only first retained length; other selectors all consumed retained lengths; no filtering missing/capped records.',
          'stopping':'4,8,16,otherwise32; unique nonmissing SC leader >=tau; quality-aware adds same SC/RWC winner and margin >=gamma; returns SC. Lazy state stream never advances after stop.',
          'development_selection':'Per family/dataset: exact integer correct/token totals over200 questions x10 orders; include always32; require (fixed32_correct-policy_correct)*100<=2000. Min tokens, then max correct, smaller tau/gamma.',
          'tau_grid':[.5,.625,.75,.875,1.],'gamma_grid':[0.,.1,.25,.5],
          'real_development_thresholds_selected':False,'held_out_scores_inspected':False}
    source_name='task3_selection_v1_'+core['selection_sha256'][:16]+'.py'
    stable={'provenance/'+source_name:(bundle/'selection.py').read_bytes(),'configs/task3_selection_verified.json':guard.encode_json(core)}
    for name,data in stable.items():require(not (root/name).exists() or (root/name).read_bytes()==data,'Existing selection implementation differs; preserving it: '+name)
    attempt=datetime.now(ZoneInfo('Asia/Karachi')).strftime('selection_checks_%Y%m%d_%H%M%S_')+uuid.uuid4().hex[:8]
    guard.save_immutable(root/'logs'/(attempt+'_checks.json'),guard.encode_json(result))
    for name in ['block_06.py','selection_checks.py']:guard.save_immutable(root/'provenance'/(attempt+'_'+name),(bundle/name).read_bytes())
    saved=[]
    for name,data in stable.items():entry=guard.save_immutable(root/name,data);entry['relative_path']=name;saved.append(entry)
    report={'status':'CPU SELECTION CHECKS COMPLETE','attempt':attempt,'workspace_drive':str(root),'scorer_attempt':reference['attempt'],'checks_passed':result['checks_passed'],'numpy_version':result['numpy_version'],'saved_artifacts':saved,'supplied_candidate_records_read':False,'gold_answers_read':False,'real_development_thresholds_selected':False,'held_out_scores_inspected':False,'gpu_work_executed':False,'next':'Share full output; next CPU block runs development-only fixed-budget analysis and adaptive-threshold selection with per-question verified saves.'}
    guard.save_immutable(root/'logs'/(attempt+'.json'),guard.encode_json(report));return report

def run(bundle):
    spec=importlib.util.spec_from_file_location('guard',bundle/'block_01.py');guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
    from google.colab import drive,runtime
    if guard.accelerator_evidence():
        print(guard.request_release_bounded(runtime.unassign),flush=True)
        raise RuntimeError('Reconnect with accelerator None; manually release if necessary')
    drive.mount('/content/drive',timeout_ms=120000)
    report=prepare(Path('/content/drive/MyDrive/ATML_PA3_Task3'),bundle,guard)
    print(json.dumps(report,indent=2),flush=True)
    print('CPU SELECTION CHECKS COMPLETE — keep accelerator None and share full output.',flush=True)
    return report
