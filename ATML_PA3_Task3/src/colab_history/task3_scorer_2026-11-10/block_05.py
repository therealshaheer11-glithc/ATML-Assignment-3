"""CPU install/verification and immutable answer-scorer freeze. No candidate scores."""
import hashlib,importlib.util,json,subprocess,sys,tempfile,uuid
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(condition,message):
    if not condition:raise ValueError(message)

def prepare(root,bundle,guard,environment_base=Path('/content/atml_pa3_task3')):
    reference=json.loads((bundle/'METADATA_REFERENCE.json').read_text())
    require(json.loads((root/'logs'/(reference['attempt']+'.json')).read_text())==reference,'Metadata receipt differs')
    for entry in reference['saved_artifacts']:require(sha(root/entry['relative_path'])==entry['sha256'],'Metadata bytes differ: '+entry['relative_path'])
    for name,entry in reference['official_sources'].items():require(sha(root/'sources/official'/name)==entry['sha256'],'Official reference source differs: '+name)
    # Verify the approved settings as well as immutable upstream partition identities.
    require(sha(root/'provenance/APPROVED_PLAN.json')=='f6c48fc63de87286cd87d1029b823670ce085af2cc41a3f175f4d76ea3f90c02','Approval register differs')
    require(sha(root/'configs/MANUAL_REQUIREMENTS.json')=='60904f8e38bfe213cc2ed2c3b59af9be4ff01be42c27b223dda3588282d7744d','Manual requirements differ')
    lock_sha=sha(bundle/'requirements.lock')
    environment_base.mkdir(parents=True,exist_ok=True)
    env=environment_base/('scorer_env_'+lock_sha[:16]);marker=env/'.task3_lock_sha256'
    if env.exists():require(marker.is_file() and marker.read_text()==lock_sha,'Existing scorer environment is incomplete or different; preserving it')
    else:
        pending=environment_base/('scorer_env_pending_'+uuid.uuid4().hex)
        print('Installing five checksum-locked CPU scorer packages in an isolated folder.',flush=True)
        subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','--only-binary=:all:','--require-hashes','--no-deps','--no-compile','--target',str(pending),'-r',str(bundle/'requirements.lock')],check=True,timeout=600)
        (pending/'.task3_lock_sha256').write_text(lock_sha)
        require(not env.exists(),'Concurrent scorer environment setup detected')
        pending.rename(env)
    attempt=datetime.now(ZoneInfo('Asia/Karachi')).strftime('scorer_%Y%m%d_%H%M%S_')+uuid.uuid4().hex[:8]
    local=Path(tempfile.mkdtemp(prefix='task3_scorer_checks_'));output=local/'checks.json'
    command=[sys.executable,'-I',str(bundle/'scorer_worker.py'),'--env',str(env),'--root',str(root),'--bundle',str(bundle),'--output',str(output)]
    print('Checking answer rules and all official references; no candidate predictions are read.',flush=True)
    try:
        process=subprocess.run(command,capture_output=True,text=True,timeout=900)
    except subprocess.TimeoutExpired as error:
        text=error.stdout or b'';text=text.decode(errors='replace') if isinstance(text,bytes) else text
        guard.save_immutable(root/'logs'/(attempt+'_TIMEOUT.txt'),text.encode())
        raise RuntimeError('CPU scorer checks timed out; no scorer freeze was published. Share full output.') from error
    console=process.stdout+process.stderr
    guard.save_immutable(root/'logs'/(attempt+'_worker.txt'),console.encode())
    print(console,flush=True)
    require(process.returncode==0 and output.is_file(),'Scorer checks failed; no scorer freeze published. Share full output.')
    result=json.loads(output.read_text())
    require(result['checks']['checks_passed']>=96 and result['official_question_checks']['questions']==1000 and not result['candidate_records_read'],'Scorer verification incomplete')
    source_name='task3_answers_v1_'+sha(bundle/'scoring.py')[:16]+'.py'
    frozen={'scorer_frozen':True,'scorer_id':'task3_answers_v1','specification':result['specification'],'installed_source_fingerprints':result['environment_fingerprint'],'metadata_preparation_attempt':reference['attempt'],'reference_metadata_sha256':{e['relative_path']:e['sha256'] for e in reference['saved_artifacts']},'source_relative_path':'provenance/'+source_name}
    stable={'configs/task3_scorer_frozen.json':guard.encode_json(frozen),'configs/task3_scorer_requirements.lock':(bundle/'requirements.lock').read_bytes(),'provenance/'+source_name:(bundle/'scoring.py').read_bytes()}
    # All targets preflighted before publishing any change to freeze state.
    for relative,payload in stable.items():require(not (root/relative).exists() or (root/relative).read_bytes()==payload,'Existing scorer differs; preserving it: '+relative)
    # Save verification evidence before publishing the authoritative frozen scorer.
    guard.save_immutable(root/'provenance'/(attempt+'_block_05.py'),(bundle/'block_05.py').read_bytes())
    guard.save_immutable(root/'provenance'/(attempt+'_scorer_checks.py'),(bundle/'scorer_checks.py').read_bytes())
    guard.save_immutable(root/'provenance'/(attempt+'_scorer_worker.py'),(bundle/'scorer_worker.py').read_bytes())
    guard.save_immutable(root/'logs'/(attempt+'_checks.json'),guard.encode_json(result))
    saved=[]
    # Publish source and dependency lock first, authoritative freeze artifact last.
    for relative in ['provenance/'+source_name,'configs/task3_scorer_requirements.lock','configs/task3_scorer_frozen.json']:
        entry=guard.save_immutable(root/relative,stable[relative]);entry['relative_path']=relative;saved.append(entry)
    report={'status':'CPU SCORER FREEZE COMPLETE','attempt':attempt,'workspace_drive':str(root),'scorer_id':'task3_answers_v1','scorer_sha256':sha(bundle/'scoring.py'),'checks_passed':result['checks']['checks_passed'],'mathvista_differential_cases':result['checks']['mathvista_differential_cases'],'reference_checks':result['reference_checks'],'official_question_checks':result['official_question_checks'],'library_edge_cases':result['checks']['library_edge_cases'],'versions':result['specification']['dependencies'],'worker_python':result['python'],'saved_artifacts':saved,'candidate_records_read':False,'correctness_computed':False,'held_out_scores_inspected':False,'adaptive_thresholds_selected':False,'gpu_work_executed':False,'next':'Share full output; next CPU block implements and verifies fixed-budget selectors/quality calculations, then development-only adaptive policy selection.'}
    guard.save_immutable(root/'logs'/(attempt+'.json'),guard.encode_json(report))
    return report

def run(bundle):
    spec=importlib.util.spec_from_file_location('task3_guard',bundle/'block_01.py');guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
    from google.colab import drive,runtime
    if guard.accelerator_evidence():
        print(guard.request_release_bounded(runtime.unassign),flush=True)
        raise RuntimeError('CPU-only task: reconnect with accelerator None and manually release if needed.')
    drive.mount('/content/drive',timeout_ms=120000)
    report=prepare(Path('/content/drive/MyDrive/ATML_PA3_Task3'),bundle,guard)
    print(json.dumps(report,indent=2),flush=True)
    print('CPU SCORER FREEZE COMPLETE — keep accelerator None and share full output.',flush=True)
    return report
