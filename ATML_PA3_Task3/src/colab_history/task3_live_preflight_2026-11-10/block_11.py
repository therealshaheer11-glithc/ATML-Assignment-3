"""CPU preflight of frozen live prompts, installed software and independent shutdown."""
import hashlib,importlib.util,json,os,signal,subprocess,sys,tempfile,threading,uuid
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def require(c,message):
    if not c:raise ValueError(message)

def validate_inputs(root,bundle):
    reference=json.loads((bundle/'ASSET_REFERENCE.json').read_text())
    require(json.loads((root/'logs'/(reference['attempt']+'.json')).read_text())==reference,'Block10 receipt differs')
    for name,h in json.loads((bundle/'EXISTING_INPUTS.json').read_text()).items():require(sha(root/name)==h,'Frozen input changed: '+name)
    manifest=json.loads((bundle/'MODEL_ASSET_MANIFEST.json').read_text())
    require(sha(root/'configs/task3_live_model_asset_manifest.json')==sha(bundle/'MODEL_ASSET_MANIFEST.json')==reference['asset_manifest_sha256'],'Model manifest differs')
    require(sha(root/'live/task3_live_inputs_v1.json')==reference['live_inputs_sha256'],'Frozen live messages/seeds differ')
    require(json.loads((bundle/'EXPECTED_TOKENIZATION.json').read_text())['live_inputs_sha256']==reference['live_inputs_sha256'],'Wrong independent tokenization fixture')
    folder=root/'live'/('model_'+reference['revision']);require(not folder.is_symlink(),'Model directory is a symlink')
    for e in manifest['files']:
        path=folder/e['name'];require(not path.is_symlink() and path.is_file() and path.stat().st_size==e['bytes'],'Asset size/type changed: '+e['name'])
        print('CPU verifying '+e['name'],flush=True);require(sha(path)==e['sha256'],'Asset checksum changed; preserve it: '+e['name'])
    return reference

def publish_checks(root,bundle,guard,reference,result,attempt):
    require(result['status']=='CPU LIVE PREFLIGHT CHECKS PASSED' and result['actual_model_generations']==0 and result['gpu_work_executed'] is False and result['native_runtime_release_executed'] is False,'Incomplete or invalid CPU preflight')
    require(result['tokenizer']['questions']==50 and result['tokenizer']['native_render_and_independent_ids_matched'] is True,'Tokenizer checks incomplete')
    require(result['synthetic_tiny_model']['seeded_repeat_equal'] and result['synthetic_tiny_model']['nonfinite_guard_raised'],'Framework checks incomplete')
    require(len(result['watchdog']['integration_scenarios'])==8 and result['watchdog']['native_runtime_release_executed'] is False and result['runtime_release_endpoint_present'] is True,'Watchdog checks incomplete')
    code={n:sha(bundle/n) for n in ['block_11.py','cpu_preflight.py','watchdog.py','supervision_checks.py','live_protocol.py','EXPECTED_TOKENIZATION.json']}
    spec=importlib.util.spec_from_file_location('task3_watchdog_constants',bundle/'watchdog.py');watch=importlib.util.module_from_spec(spec);spec.loader.exec_module(watch)
    frozen={'schema_version':1,'implementation_id':'task3_live_runtime_preflight_v1','approval':'A001','model_revision':reference['revision'],'asset_manifest_sha256':reference['asset_manifest_sha256'],'live_inputs_sha256':reference['live_inputs_sha256'],'source_code':code,'cpu_python':result['python'],'cpu_versions':result['versions'],'cpu_source_fingerprints':result['source_fingerprints'],'runtime_release_source_sha256':result['runtime_release_source_sha256'],'generation_configurations':result['generation_configurations'],'watchdog_limits_seconds':watch.LIMITS,'watchdog_worker_registration_required_before_GPU_import':True,'GPU_requirement':'NVIDIA T4; FP16; batch1; no quantization; CUDA build/hardware frozen at first actual session and kept identical thereafter','torch_variant_rule':'Same public2.14.1 as CPU preflight; actual CUDA variant and binary/hardware details checked and frozen during first session','actual_revision_context':'Check actual tokenized revision prompt plus2048 before each call; no silent truncation','source_generation_defaults':'Explicit approved repetition_penalty1.0 overrides source1.05','GPU_used':False,'physical_runtime_deallocation_verified':False}
    payload=guard.encode_json(frozen);guard.save_immutable(root/'configs/task3_live_runtime_preflight.json',payload)
    artifacts=[]
    for n in ['block_11.py','cpu_preflight.py','watchdog.py','supervision_checks.py','live_protocol.py','EXPECTED_TOKENIZATION.json','ASSET_REFERENCE.json']:
        item=guard.save_immutable(root/'provenance'/(attempt+'_'+n),(bundle/n).read_bytes());artifacts.append(item)
    checks_path=root/'logs'/(attempt+'_checks.json');guard.save_immutable(checks_path,guard.encode_json(result))
    return {'status':'CPU LIVE PREFLIGHT COMPLETE','attempt':attempt,'frozen_preflight_sha256':hashlib.sha256(payload).hexdigest(),'checks_sha256':sha(checks_path),'model_revision':reference['revision'],'asset_files_reverified':11,'prompt_questions_checked':50,'initial_prompt_tokens_min':result['tokenizer']['initial_tokens_min'],'initial_prompt_tokens_max':result['tokenizer']['initial_tokens_max'],'CPU_tiny_model_checks':result['synthetic_tiny_model'],'watchdog_assertions':result['watchdog']['passed_assertions'],'watchdog_integration_scenarios':len(result['watchdog']['integration_scenarios']),'runtime_release_endpoint_present':True,'gpu_work_executed':False,'actual_model_loaded':False,'actual_model_generations':0,'native_runtime_release_executed':False,'physical_runtime_deallocation_verified':False,'software_installed':False,'scoring_rerun':False,'policies_changed':False,'versions':result['versions'],'durability_scope':'Mounted-filesystem byte readback; remote durability not independently observed','next':'Share full output. Keep accelerator None until this receipt is reviewed and the separate bounded T4 execution block is supplied.'}

def prepare(root,bundle,guard):
    reference=validate_inputs(root,bundle);local=Path('/content/atml_pa3_task3');local.mkdir(exist_ok=True)
    work=Path(tempfile.mkdtemp(prefix='cpu_live_preflight_',dir=local));output=work/'checks.json'
    attempt=datetime.now(ZoneInfo('Asia/Karachi')).strftime('live_preflight_%Y%m%d_%H%M%S_')+uuid.uuid4().hex[:8]
    command=[sys.executable,'-I',str(bundle/'cpu_preflight.py'),'--root',str(root),'--bundle',str(bundle),'--output',str(output)]
    proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True)
    def stop():
        try:os.killpg(proc.pid,signal.SIGKILL)
        except ProcessLookupError:pass
    timer=threading.Timer(600,stop);timer.daemon=True;timer.start();lines=[]
    try:
        for line in proc.stdout:print(line,end='',flush=True);lines.append(line)
        code=proc.wait()
    except BaseException:
        stop();raise
    finally:timer.cancel()
    guard.save_immutable(root/'logs'/(attempt+'_worker.txt'),''.join(lines).encode())
    if code!=0 or not output.exists():
        receipt={'status':'CPU LIVE PREFLIGHT FAILED','attempt':attempt,'worker_exit_code':code,'gpu_work_executed':False,'next':'Keep accelerator None. Share the entire output; do not start GPU work or change frozen files.'}
    else:receipt=publish_checks(root,bundle,guard,reference,json.loads(output.read_text()),attempt)
    guard.save_immutable(root/'logs'/(attempt+'.json'),guard.encode_json(receipt));return receipt

def run(bundle):
    spec=importlib.util.spec_from_file_location('guard',bundle/'block_01.py');guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
    from google.colab import drive,runtime
    if guard.accelerator_evidence():
        print(guard.request_release_bounded(runtime.unassign),flush=True);raise RuntimeError('Reconnect with accelerator None')
    drive.mount('/content/drive',timeout_ms=120000)
    receipt=prepare(Path('/content/drive/MyDrive/ATML_PA3_Task3'),bundle,guard)
    print(json.dumps(receipt,indent=2),flush=True);print(receipt['status']+' — keep accelerator None and share full output.',flush=True);return receipt
