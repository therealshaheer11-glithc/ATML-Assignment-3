"""CPU test/stage finished GPU kit; no model load, GPU or runtime-release call."""
import ast,hashlib,importlib.util,io,json,os,signal,subprocess,sys,tempfile,threading,uuid,zipfile
from datetime import datetime
from importlib import metadata
from pathlib import Path
from zoneinfo import ZoneInfo

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(c,message):
    if not c:raise ValueError(message)

def prepare(root,bundle,guard):
    reference=json.loads((bundle/'PREFLIGHT_REFERENCE.json').read_text())
    require(json.loads((root/'logs'/(reference['attempt']+'.json')).read_text())==reference,'Block11 receipt differs')
    require(sha(root/'configs/task3_live_runtime_preflight.json')==reference['frozen_preflight_sha256'],'Frozen CPU preflight differs')
    require(sha(root/'logs'/(reference['attempt']+'_checks.json'))==reference['checks_sha256'],'CPU checks artifact differs')
    require({n:metadata.version(n) for n in reference['versions']}==reference['versions'],'CPU core software changed since Block11')
    # Compare all active captured GPU Python sources with the frozen CPU fingerprints first.
    preflight=json.loads((root/'configs/task3_live_runtime_preflight.json').read_text())
    diagnostic=json.loads((bundle/'GPU_DIAGNOSTIC_REFERENCE.json').read_text())
    comparisons=[]
    for name,cpu in preflight['cpu_source_fingerprints'].items():
        if name=='torch':continue
        matches=[r for r in diagnostic['distributions'] if r['name']==name and r['version']==cpu['version'] and r['checked_files']>0]
        matched=False;actual=None
        if len(matches)==1:
            row=matches[0];entries=sorted(e['file']+':'+e['sha256'] for e in row['source_entries']);actual=hashlib.sha256('\n'.join(entries).encode()).hexdigest()
            matched=row['all_checked_sources_match_RECORD'] and len(entries)==cpu['python_source_files'] and actual==cpu['source_sha256']
        comparisons.append({'package':name,'CPU_version':cpu['version'],'captured_GPU_candidates':[r['version'] for r in diagnostic['distributions'] if r['name']==name and r['checked_files']>0],'Python_sources_match':matched,'captured_source_sha256':actual})
    print(json.dumps({'CPU_vs_captured_GPU_sources':comparisons,'binary_assets':'Checked again in actual GPU session; diagnostic captured Python sources only'},indent=2),flush=True)
    require(all(r['Python_sources_match'] for r in comparisons),'Additional CPU/GPU source difference exists; keep GPU off and share this comparison')
    # No earlier generated responses may be silently mixed with the repaired recipe.
    prior=root/'results/task3_live_generation_v1/calls'
    require(not prior.exists() or not any(prior.iterdir()),'Earlier live call files exist; review before starting A002')
    artifacts=json.loads((bundle/'GPU_ARTIFACTS.json').read_text())
    for n,h in artifacts['outer_hashes'].items():require(sha(bundle/n)==h,'Execution-kit artifact differs')
    base=Path('/content/atml_pa3_task3');base.mkdir(exist_ok=True);work=Path(tempfile.mkdtemp(prefix='gpu_integrity_A002_cpu_',dir=base))
    with zipfile.ZipFile(io.BytesIO((bundle/'GPU_CODE.zip').read_bytes())) as archive:
        require(len(archive.infolist())==len(artifacts['code_members']) and set(archive.namelist())==set(artifacts['code_members']),'Unexpected GPU code members')
        for n,h in artifacts['code_members'].items():
            require(Path(n).name==n,'Unsafe code member');raw=archive.read(n);require(hashlib.sha256(raw).hexdigest()==h,'Code member differs');(work/n).write_bytes(raw)
    for n in ['gpu_kit_checks.py','kit_checks_entry.py','integrity_checks.py']:(work/n).write_bytes((bundle/n).read_bytes())
    # Fixtures come from the user's verified diagnostic archive, not from an installed-file rewrite.
    diagnostic_zip=root/'exports/task3_GPU_package_diagnostic_20261110_155701_c0fada98.zip'
    require(sha(diagnostic_zip)=='e6bb94ab4eb1681cfc325b821ab3f18fb7bb81a0d5f9c7a6b8d1c9a965ce5e7e','Diagnostic archive bytes differ')
    policy=json.loads((work/'TORCH_INTEGRITY_POLICY_A002.json').read_text())
    gpu_torch=next(r for r in diagnostic['distributions'] if r['name']=='torch' and r['version']==policy['version'])
    member=next(e['member'] for e in gpu_torch['captured_sources'] if e['file']==policy['file'])
    with zipfile.ZipFile(diagnostic_zip) as z:observed=z.read(member)
    require(hashlib.sha256(observed).hexdigest()==policy['accepted_modified_sha256'],'Captured approved initializer differs')
    # Invert only the reviewed movement of the generic-path glob; verify the complete official hash.
    tree=ast.parse(observed.decode());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_get_cuda_dep_paths')
    assignment=next(n for n in fn.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='nvidia_lib_paths' for t in n.targets))
    addition=next(n for n in fn.body if isinstance(n,ast.AugAssign) and isinstance(n.target,ast.Name) and n.target.id=='nvidia_lib_paths')
    lines=observed.decode().splitlines(keepends=True);block=lines[addition.lineno-1:addition.end_lineno];block[0]=block[0].replace('+=','=');del lines[addition.lineno-1:addition.end_lineno];lines[assignment.lineno-1:assignment.end_lineno]=block;official=''.join(lines).encode()
    require(hashlib.sha256(official).hexdigest()==policy['official_RECORD_sha256'],'Reconstructed official fixture differs')
    (work/'TEST_torch_init_observed.txt').write_bytes(observed);(work/'TEST_torch_init_official.txt').write_bytes(official)
    live=(root/'live/task3_live_inputs_v1.json').read_bytes();require(hashlib.sha256(live).hexdigest()=='6a01498bf05d468fb575182b31deb6086ba5ed137f3a34de662297bcea813c5d','Live questions/seeds changed');(work/'TEST_LIVE_INPUTS.json').write_bytes(live)
    protocol=json.loads((root/'configs/MANUAL_REQUIREMENTS.json').read_text())['live'];(work/'TEST_PROTOCOL.json').write_bytes(guard.encode_json(protocol))
    tokenizer=root/'live'/('model_'+reference['model_revision'])/'tokenizer.json';model_manifest=json.loads((work/'MODEL_ASSET_MANIFEST.json').read_text());tokenizer_hash=next(e['sha256'] for e in model_manifest['files'] if e['name']=='tokenizer.json');require(sha(tokenizer)==tokenizer_hash,'Tokenizer bytes changed');(work/'TEST_tokenizer.json').write_bytes(tokenizer.read_bytes())
    output=work/'kit_checks.json';print('CPU A002 verification only: exact approved exception and rejection tests, fake model/CUDA API and local release service.',flush=True)
    proc=subprocess.Popen([sys.executable,'-I',str(work/'kit_checks_entry.py'),'--output',str(output)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True)
    def stop():
        try:os.killpg(proc.pid,signal.SIGKILL)
        except ProcessLookupError:pass
    timer=threading.Timer(180,stop);timer.daemon=True;timer.start();lines=[]
    try:
        for line in proc.stdout:print(line,end='',flush=True);lines.append(line)
        code=proc.wait()
    except BaseException:stop();raise
    finally:timer.cancel()
    attempt=datetime.now(ZoneInfo('Asia/Karachi')).strftime('gpu_integrity_A002_cpu_%Y%m%d_%H%M%S_')+uuid.uuid4().hex[:8]
    guard.save_immutable(root/'logs'/(attempt+'_worker.txt'),''.join(lines).encode());require(code==0 and output.exists(),'CPU simulation failed; keep GPU off and share full output')
    checks=json.loads(output.read_text());checks['CPU_vs_captured_GPU_sources']=comparisons;require(checks['execution_kit']['tests_passed']==9 and len(checks['execution_kit']['bootstrap_real_process_release_scenarios'])==2 and checks['integrity']['passed']==8 and checks['GPU_used'] is False and checks['native_runtime_release_executed'] is False,'Incomplete CPU A002 execution-kit checks')
    for relative,payload in {'provenance/task3_gpu_code_A002.zip':(bundle/'GPU_CODE.zip').read_bytes(),'provenance/task3_gpu_starter_A002.py':(bundle/'GPU_STARTER.py').read_bytes(),'configs/task3_live_generation_recipe_A002.json':(work/'GENERATION_RECIPE.json').read_bytes(),'configs/task3_gpu_artifacts_A002.json':(bundle/'GPU_ARTIFACTS.json').read_bytes()}.items():guard.save_immutable(root/relative,payload)
    for n in ['block_14.py','gpu_kit_checks.py','kit_checks_entry.py','integrity_checks.py','APPROVAL_A002.json','TORCH_INTEGRITY_POLICY_A002.json']:guard.save_immutable(root/'provenance'/(attempt+'_'+n),(bundle/n).read_bytes())
    guard.save_immutable(root/'logs'/(attempt+'_checks.json'),guard.encode_json(checks))
    report={'status':'CPU APPROVED INTEGRITY FIX READY','approval_id':'A002','attempt':attempt,'frozen_preflight_sha256':reference['frozen_preflight_sha256'],'GPU_code_zip_sha256':artifacts['outer_hashes']['GPU_CODE.zip'],'GPU_starter_sha256':artifacts['outer_hashes']['GPU_STARTER.py'],'generation_recipe_sha256':sha(work/'GENERATION_RECIPE.json'),'CPU_execution_tests_passed':checks['execution_kit']['tests_passed'],'CPU_integrity_tests_passed':checks['integrity']['passed'],'other_GPU_Python_sources_match_CPU':True,'backup_release_test_scenarios':len(checks['execution_kit']['bootstrap_real_process_release_scenarios']),'gpu_work_executed':False,'model_loaded':False,'actual_model_generations':0,'native_runtime_release_executed':False,'software_installed':False,'package_files_modified':False,'scorer_or_stopping_policy_changes':False,'integrity_rule_change':'A002 exact approved file only','durability_scope':'Mounted-filesystem byte readback; remote durability not independently observed','next':'Share full output. Keep accelerator None until the separate T4 starter is supplied after receipt review.'}
    guard.save_immutable(root/'logs'/(attempt+'.json'),guard.encode_json(report));return report

def run(bundle):
    s=importlib.util.spec_from_file_location('guard',bundle/'block_01.py');guard=importlib.util.module_from_spec(s);s.loader.exec_module(guard)
    from google.colab import drive,runtime
    if guard.accelerator_evidence():
        print(guard.request_release_bounded(runtime.unassign),flush=True);raise RuntimeError('Reconnect with accelerator None')
    drive.mount('/content/drive',timeout_ms=120000);report=prepare(Path('/content/drive/MyDrive/ATML_PA3_Task3'),bundle,guard);print(json.dumps(report,indent=2),flush=True);print(report['status']+' — keep accelerator None and share full output.',flush=True);return report
