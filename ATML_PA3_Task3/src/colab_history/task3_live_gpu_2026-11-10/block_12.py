"""CPU test/stage finished GPU kit; no model load, GPU or runtime-release call."""
import hashlib,importlib.util,io,json,os,signal,subprocess,sys,tempfile,threading,uuid,zipfile
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
    artifacts=json.loads((bundle/'GPU_ARTIFACTS.json').read_text())
    for n,h in artifacts['outer_hashes'].items():require(sha(bundle/n)==h,'Execution-kit artifact differs')
    base=Path('/content/atml_pa3_task3');base.mkdir(exist_ok=True);work=Path(tempfile.mkdtemp(prefix='gpu_kit_cpu_',dir=base))
    with zipfile.ZipFile(io.BytesIO((bundle/'GPU_CODE.zip').read_bytes())) as archive:
        require(len(archive.infolist())==len(artifacts['code_members']) and set(archive.namelist())==set(artifacts['code_members']),'Unexpected GPU code members')
        for n,h in artifacts['code_members'].items():
            require(Path(n).name==n,'Unsafe code member');raw=archive.read(n);require(hashlib.sha256(raw).hexdigest()==h,'Code member differs');(work/n).write_bytes(raw)
    for n in ['gpu_kit_checks.py','kit_checks_entry.py']:(work/n).write_bytes((bundle/n).read_bytes())
    live=(root/'live/task3_live_inputs_v1.json').read_bytes();require(hashlib.sha256(live).hexdigest()=='6a01498bf05d468fb575182b31deb6086ba5ed137f3a34de662297bcea813c5d','Live questions/seeds changed');(work/'TEST_LIVE_INPUTS.json').write_bytes(live)
    protocol=json.loads((root/'configs/MANUAL_REQUIREMENTS.json').read_text())['live'];(work/'TEST_PROTOCOL.json').write_bytes(guard.encode_json(protocol))
    tokenizer=root/'live'/('model_'+reference['model_revision'])/'tokenizer.json';model_manifest=json.loads((work/'MODEL_ASSET_MANIFEST.json').read_text());tokenizer_hash=next(e['sha256'] for e in model_manifest['files'] if e['name']=='tokenizer.json');require(sha(tokenizer)==tokenizer_hash,'Tokenizer bytes changed');(work/'TEST_tokenizer.json').write_bytes(tokenizer.read_bytes())
    output=work/'kit_checks.json';print('CPU simulation only: disposable call files, fake model/CUDA API and local release service.',flush=True)
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
    attempt=datetime.now(ZoneInfo('Asia/Karachi')).strftime('gpu_kit_cpu_%Y%m%d_%H%M%S_')+uuid.uuid4().hex[:8]
    guard.save_immutable(root/'logs'/(attempt+'_worker.txt'),''.join(lines).encode());require(code==0 and output.exists(),'CPU simulation failed; keep GPU off and share full output')
    checks=json.loads(output.read_text());require(checks['tests_passed']==9 and len(checks['bootstrap_real_process_release_scenarios'])==2 and checks['GPU_used'] is False and checks['native_runtime_release_executed'] is False,'Incomplete CPU execution-kit checks')
    for relative,payload in {'provenance/task3_gpu_code_v1.zip':(bundle/'GPU_CODE.zip').read_bytes(),'provenance/task3_gpu_starter_v1.py':(bundle/'GPU_STARTER.py').read_bytes(),'configs/task3_live_generation_recipe.json':(work/'GENERATION_RECIPE.json').read_bytes(),'configs/task3_gpu_artifacts_v1.json':(bundle/'GPU_ARTIFACTS.json').read_bytes()}.items():guard.save_immutable(root/relative,payload)
    for n in ['block_12.py','gpu_kit_checks.py','kit_checks_entry.py']:guard.save_immutable(root/'provenance'/(attempt+'_'+n),(bundle/n).read_bytes())
    guard.save_immutable(root/'logs'/(attempt+'_checks.json'),guard.encode_json(checks))
    report={'status':'CPU EXECUTION KIT READY','attempt':attempt,'frozen_preflight_sha256':reference['frozen_preflight_sha256'],'GPU_code_zip_sha256':artifacts['outer_hashes']['GPU_CODE.zip'],'GPU_starter_sha256':artifacts['outer_hashes']['GPU_STARTER.py'],'generation_recipe_sha256':sha(work/'GENERATION_RECIPE.json'),'CPU_tests_passed':checks['tests_passed'],'backup_release_test_scenarios':len(checks['bootstrap_real_process_release_scenarios']),'gpu_work_executed':False,'model_loaded':False,'actual_model_generations':0,'native_runtime_release_executed':False,'software_installed':False,'scorer_or_policy_changes':False,'durability_scope':'Mounted-filesystem byte readback; remote durability not independently observed','next':'Share full output. Keep accelerator None until the separate T4 starter is supplied after receipt review.'}
    guard.save_immutable(root/'logs'/(attempt+'.json'),guard.encode_json(report));return report

def run(bundle):
    s=importlib.util.spec_from_file_location('guard',bundle/'block_01.py');guard=importlib.util.module_from_spec(s);s.loader.exec_module(guard)
    from google.colab import drive,runtime
    if guard.accelerator_evidence():
        print(guard.request_release_bounded(runtime.unassign),flush=True);raise RuntimeError('Reconnect with accelerator None')
    drive.mount('/content/drive',timeout_ms=120000);report=prepare(Path('/content/drive/MyDrive/ATML_PA3_Task3'),bundle,guard);print(json.dumps(report,indent=2),flush=True);print(report['status']+' — keep accelerator None and share full output.',flush=True);return report
