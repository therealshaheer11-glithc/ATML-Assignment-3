"""L4 worker for the first approved Full FT boundary. Core code is unchanged."""
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'core'))
from archive_io import digest
from baseline_runtime import atomic,progress,wait_for_parent
from evaluation_records import identity,publish
from training_launch_support import checkpoint_evidence,latest_checkpoint


def run(config,result):
    import torch
    from transformers import AutoTokenizer
    from official_training import build_method,SourceOrderData,require_training_gpu
    from training_checkpoints import restore_checkpoint
    from training_loop import run_updates
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True,warn_only=False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    require_training_gpu();torch.cuda.reset_peak_memory_stats()
    root,inputs,core=Path(config['root']),Path(config['inputs']),Path(config['core'])
    protocol=json.loads(Path(config['protocol_local']).read_text());contract=config['training_contract']
    assert identity(protocol)==contract['protocol_identity']==config['protocol_identity']
    assert contract['method']=='full' and contract['peak_lr']==2e-5 and contract['lambda']==0.
    for name,expected in protocol['numerical_source_files'].items():assert digest(core/name)==expected
    assert identity(protocol['numerical_source_files'])==config['numerical_implementation_identity']
    assert digest(inputs/'manifests/TRAINING_DATA_READY.json')==protocol['training_data_manifest_sha256']
    assert digest(inputs/'manifests/CPU_ASSETS_READY.json')==protocol['CPU_assets_manifest_sha256']
    assert digest(inputs/'configs/CPU_MODEL_CONTRACT.json')==protocol['CPU_model_contract_sha256']
    assert digest(inputs/'manifests/ORIGINAL_MODEL_FILES.json')==protocol['original_model_manifest_sha256']
    progress(config['state'],'load fresh original Full FT FP32 masters')
    print('Loading original FP32 masters for official Full FT; no scratch optimizer updates.',flush=True)
    begin=time.monotonic()
    model,named,optimizer,scaler,reference,peak=build_method(inputs,'full','cuda')
    assert reference is None and peak==contract['peak_lr'] and sum(p.numel() for _,p in named)==596049920
    tokenizer=AutoTokenizer.from_pretrained(inputs/'sources/original_model',local_files_only=True)
    data=SourceOrderData(inputs);restored=None
    if config.get('resume_checkpoint'):
        progress(config['state'],'restore verified Full FT tensors, optimizer, scaler and RNG',saving=True)
        restored=restore_checkpoint(root/config['resume_checkpoint'],contract,model,named,optimizer,scaler,
            progress=lambda phase:progress(config['state'],phase,saving=True))
        assert restored['successful_updates']==config['start_update']
    else:assert config['start_update']==0
    result['model_load_and_resume_seconds']=time.monotonic()-begin
    outcome=run_updates(config,model,named,optimizer,scaler,data,peak,tokenizer.pad_token_id,restored=restored)
    result['training_outcome']=outcome
    batch=json.loads(Path(config['batch_plan_local']).read_text())
    assert identity(batch)==protocol['batch_plan_identity']
    evidence=checkpoint_evidence(root,outcome['checkpoint'],contract,596049920,batch)
    u=evidence['successful_updates'];assert config['start_update']<=u<=config['target_update']==312
    result.update(status='TASK1_FULL_FT_BOUNDARY_SAVED' if u==312 else 'TASK1_FULL_FT_PARTIAL_SAVED',
        official_training_updates=u,examples_seen=u*32,updates_this_job=u-config['start_update'],checkpoint=evidence)
    result['GPU_memory']={'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'total_gpu_bytes':torch.cuda.get_device_properties(0).total_memory}


def main(config_path):
    config=json.loads(Path(config_path).read_text());wait_for_parent(config['state']);start=time.monotonic();code=0
    result={'status':'TASK1_FULL_FT_FAILED','run_id':config['run_id'],'method':'full',
        'training_contract_identity':identity(config['training_contract']),
        'numerical_implementation_identity':config['numerical_implementation_identity'],
        'job_target_update':312,'start_update':config['start_update']}
    try:run(config,result)
    except BaseException as error:
        code=1;result.update(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        try:
            result['latest_durable_checkpoint']=latest_checkpoint(config['root'],config['training_contract'],596049920)
        except Exception as recovery_error:result['checkpoint_scan_error']=str(recovery_error)
        result['durable_count_caveat']='Only verified checkpoint counts establish recoverable updates; a failed optimizer may have lost uncheckpointed work.'
    result['elapsed_worker_seconds']=time.monotonic()-start
    publish(config['result'],result);print(json.dumps(result,indent=2),flush=True)
    state=json.loads(Path(config['state']).read_text())
    state.update(status='worker_complete',worker_completed_wall=time.time(),progress_wall=time.time());atomic(config['state'],state)
    return code


if __name__=='__main__':sys.exit(main(sys.argv[1]))
