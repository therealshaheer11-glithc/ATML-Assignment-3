"""One bounded live session. No correctness labels or reference answers are read."""
import argparse,hashlib,json,os,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
import live_protocol,watchdog
import live_store as store

def utc():return datetime.now(timezone.utc).isoformat()
class Control:
    def __init__(self,config):self.config=config;self.directory=Path(config['directory']);self.pending=None;self.current_call=None;self.last_phase=None;self.last_phase_started=None
    def phase(self,name,**extra):
        value={'session_id':self.config['session_id'],'phase':name,'phase_started':time.monotonic(),'worker':watchdog.process_info(os.getpid()),'current_call':self.current_call,'completed_call_envelope':self.pending,**extra}
        watchdog.write_json(self.directory/'state.json',value);self.last_phase=name;self.last_phase_started=value['phase_started'];return value['phase_started']
    def handshake(self):
        self.phase('setup');deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            path=self.directory/'worker_ack.json'
            if path.exists():
                ack=json.loads(path.read_text());store.require(ack['session_id']==self.config['session_id'] and ack['worker']==watchdog.process_info(os.getpid()),'Watchdog acknowledged wrong worker');return
            time.sleep(.05)
        raise RuntimeError('Independent watchdog did not acknowledge worker; no GPU imports allowed')

def schedule(work,rows,protocol,recipe_sha,environment_sha,session_sha,control,engine,previous,clock=time.monotonic):
    total_calls=sum(len(r['call_plan']['call_order']) for r in rows);cfg=control.config;attempts=work/'attempts'/cfg['session_id'];attempts.mkdir(parents=True,exist_ok=True);new=0
    for row in rows:
        for name in row['call_plan']['call_order']:
            key=store.call_key(row['sample_id'],name)
            if key in previous:continue
            if not watchdog.can_start_call(cfg['started'],clock()):return {'reason':'soft_limit','new_calls':new,'verified_calls':len(previous),'total_calls':total_calls}
            control.current_call={'sample_id':row['sample_id'],'call_name':name,'call_key':key,'seed':row['call_plan']['seeds'][name],'cap':row['call_plan']['caps'][name]}
            control.phase('saving');intent={'status':'CALL_INTENT','session_id':cfg['session_id'],'timestamp':utc(),'recipe_sha256':recipe_sha,**control.current_call}
            store.save(attempts/(key+'_intent.json'),store.encode(intent))
            if not watchdog.can_start_call(cfg['started'],clock()):return {'reason':'soft_limit','new_calls':new,'verified_calls':len(previous),'total_calls':total_calls}
            messages,shared=store.messages_for(row,name,previous,protocol);began=control.phase('generating')
            try:answer=engine.infer(messages,row['call_plan']['caps'][name],row['call_plan']['seeds'][name],began)
            except store.SoftLimitReached:
                control.phase('saving');store.save(attempts/(key+'_not_started.json'),store.encode({'status':'SOFT_LIMIT_BEFORE_GENERATION','call_key':key,'preparation_elapsed_seconds':time.monotonic()-began,'model_generate_started':False}))
                return {'reason':'soft_limit','new_calls':new,'verified_calls':len(previous),'total_calls':total_calls}
            record={'schema_version':1,'status':'COMPLETE','recipe_sha256':recipe_sha,'GPU_environment_sha256':environment_sha,'physical_session_manifest_sha256':session_sha,'session_id':cfg['session_id'],'sample_id':row['sample_id'],'call_name':name,'call_key':key,'seed':row['call_plan']['seeds'][name],'max_new_tokens':row['call_plan']['caps'][name],'messages':messages,'shared_initial':shared,'completed_timestamp':utc(),**answer}
            store.validate(record,row,name,recipe_sha,environment_sha,previous,protocol,engine.tokenizer)
            packaged=store.envelope(record);raw=store.encode(packaged);control.pending=packaged
            # Local spool and control-state copy permit bounded best-effort salvage if Drive fails.
            (control.directory/(key+'_completed.json')).write_bytes(raw);control.phase('saving')
            sha=store.save(work/'calls'/(key+'.json'),raw);previous[key]={'record':record,'envelope_sha256':sha};new+=1
            store.save(attempts/(key+'_verified.json'),store.encode({'status':'CALL_VERIFIED','call_key':key,'envelope_sha256':sha,'inference_seconds':record['inference_seconds'],'output_tokens':record['termination']['output_tokens']}))
            control.pending=None;control.phase('idle',verified_calls=len(previous))
            print(json.dumps({'status':'CALL SAVED','sample_id':row['sample_id'],'call':name,'verified_calls':len(previous),'of':250,'output_tokens':record['termination']['output_tokens'],'seconds':round(record['inference_seconds'],3),'capped':record['termination']['truncated_by_cap']}),flush=True)
    return {'reason':'all_calls_complete','new_calls':new,'verified_calls':len(previous),'total_calls':total_calls}

def execute(root,bundle,config,control):
    reference=json.loads((bundle/'PREFLIGHT_REFERENCE.json').read_text());store.require(json.loads((root/'logs'/(reference['attempt']+'.json')).read_text())==reference,'CPU preflight receipt differs')
    store.require(store.file_sha(root/'configs/task3_live_runtime_preflight.json')==reference['frozen_preflight_sha256'],'Frozen CPU preflight differs')
    preflight=json.loads((root/'configs/task3_live_runtime_preflight.json').read_text())
    for n in ['watchdog.py','live_protocol.py','cpu_preflight.py']:store.require(store.file_sha(bundle/n)==preflight['source_code'][n],'CPU-checked code changed: '+n)
    for name,h in json.loads((bundle/'EXISTING_INPUTS.json').read_text()).items():store.require(store.file_sha(root/name)==h,'Frozen input changed: '+name)
    recipe_raw=(bundle/'GENERATION_RECIPE.json').read_bytes();store.require((root/'configs/task3_live_generation_recipe_A002.json').read_bytes()==recipe_raw,'CPU-staged generation recipe differs');recipe=json.loads(recipe_raw)
    for name,h in {**recipe['code_sha256'],**recipe['static_input_sha256']}.items():store.require(store.file_sha(bundle/name)==h,'Generation code/input differs: '+name)
    store.require(store.file_sha(root/'live/task3_live_inputs_v1.json')==preflight['live_inputs_sha256'],'Frozen live inputs differ')
    rows=json.loads((root/'live/task3_live_inputs_v1.json').read_text())['questions'];protocol=json.loads((root/'configs/MANUAL_REQUIREMENTS.json').read_text())['live'];manifest=json.loads((bundle/'MODEL_ASSET_MANIFEST.json').read_text())
    store.require(len(rows)==50 and sum(len(r['call_plan']['call_order']) for r in rows)==250,'Incomplete fixed call plan')
    work=root/'results/task3_live_generation_A002';work.mkdir(exist_ok=True)
    for name in ['calls','sessions','attempts']:(work/name).mkdir(exist_ok=True)
    store.save(work/'recipe.json',recipe_raw);recipe_sha=store.digest(recipe_raw)
    import gpu_engine
    import torch
    from transformers import AutoTokenizer
    signature,physical=gpu_engine.environment(preflight);environment_raw=store.encode(signature);env_sha=store.save(root/'configs/task3_live_GPU_environment_A002.json',environment_raw)
    metadata={'session_id':config['session_id'],'session_started_timestamp':config['started_utc'],'GPU_environment_sha256':env_sha,'recipe_sha256':recipe_sha,'physical_hardware':physical,'limits_seconds':watchdog.LIMITS,'prior_verified_state_is_recovery_point':True}
    session_sha=store.save(work/'sessions'/(config['session_id']+'.json'),store.encode(metadata))
    source=root/'live'/('model_'+preflight['model_revision']);tokenizer=AutoTokenizer.from_pretrained(str(source),local_files_only=True,trust_remote_code=False,use_fast=True)
    previous=store.scan(work,rows,recipe_sha,env_sha,protocol,tokenizer);print('Verified saved calls: '+str(len(previous))+'/250',flush=True)
    if len(previous)==250:return {'reason':'all_calls_already_complete','new_calls':0,'verified_calls':250,'total_calls':250}
    store.require(watchdog.can_start_call(config['started'],time.monotonic()),'Soft limit reached before model load')
    control.phase('loading');folder=store.stage_assets(source,Path(config['directory'])/'model',manifest)
    engine=gpu_engine.Engine(folder,tokenizer,protocol,config['started']);print('Approved T4 FP16 model loaded; first scheduled call is the execution check.',flush=True)
    return schedule(work,rows,protocol,recipe_sha,env_sha,session_sha,control,engine,previous)

def main(root,bundle,config_path):
    config=json.loads(Path(config_path).read_text());control=Control(config)
    try:
        control.handshake()
        for k,v in {'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}.items():os.environ[k]=v
        result=execute(root,bundle,config,control);control.phase('saving')
        report={'status':'GPU SESSION SAVED','session_id':config['session_id'],**result,'finished_timestamp':utc(),'elapsed_seconds':time.monotonic()-config['started'],'no_automatic_retry':True,'scoring_executed':False,'physical_runtime_deallocation_verified':False}
        store.save(root/'logs'/(config['session_id']+'_result.json'),store.encode(report));print(json.dumps(report,indent=2),flush=True)
        control.phase('finished',result=report)
        # Remain alive until watchdog stops the registered group and requests release.
        while True:time.sleep(1)
    except BaseException as error:
        traceback.print_exc();print('SESSION STOPPED — prior verified calls preserved; requesting release. No automatic retry.',flush=True)
        try:control.phase('error',error_type=type(error).__name__,error_message=str(error)[:2000],failed_phase=control.last_phase,failed_phase_elapsed_seconds=None if control.last_phase_started is None else time.monotonic()-control.last_phase_started,incomplete_call_costs_unknown=control.pending is None and control.current_call is not None)
        except Exception:pass
        try:watchdog.write_json(control.directory/'abort.json',{'release_after_monotonic':time.monotonic()+8})
        except Exception:pass
        while True:time.sleep(1)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--bundle',required=True);p.add_argument('--config',required=True);a=p.parse_args();main(Path(a.root),Path(a.bundle),Path(a.config))
