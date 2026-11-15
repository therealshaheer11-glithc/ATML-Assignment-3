"""Bounded first Full FT job using the unchanged, actual-Colab-validated trainer."""
import json
import os
import shutil
import time
from pathlib import Path

from archive_io import digest,safe_relative
from baseline_runtime import atomic,progress,publish_bytes
from evaluation_records import identity,publish
from launch_baseline import restore_archive
from training_launch_support import latest_checkpoint,execute

KIT='fe4fccd0592d4e5e634af6df831aa3c72cacf0c236f34e6cae2d9921e5419323'
READY_SHA='2018adb0429923fef6de37e144d0eefc03fb0957f44d1958a9653e8e0abcf33a'
NUMERICAL='654543e33f45ac0ef7eaa46f5d51ffc87bbd773d5174730b815f103a9ef89e26'
PROTOCOL='8c4c86260d5102fb5545efacce1a10d48af801c41eeff376dd7e27e0d836102c'
FULL_CONTRACT='6ed4a55411318958c5d853334b7069e0cdf752c628732b756b6b2c971ac0343b'


def run(context,bundle):
    root,job,bundle=Path(context['root']),Path(context['job']),Path(bundle)
    core,launch=bundle/'core',bundle/'launch';state=context['state'];run_id=job.name
    progress(state,'verify successful CPU training gate and frozen Full FT contract')
    ready_path=root/'configs'/('TRAINING_CPU_READY_'+KIT+'.json')
    assert digest(ready_path)==READY_SHA
    ready=json.loads(ready_path.read_text())
    assert ready['status']=='TASK1_BLOCK19_TRAINING_CPU_READY' and ready['numerical_implementation_identity']==NUMERICAL
    assert identity(ready['files'])==KIT and ready['runtime_checks']['fixture_torch_version']=='2.8.0+cu126'
    assert all(ready['runtime_checks']['token_loss_and_projection_checks'].values())
    # Explicitly check all required Boolean acceptance gates, excluding descriptive zero-update fields.
    for section in ('runtime_checks','bounded_loop_checks','native_tiny_Qwen_checks'):
        for key,value in ready[section].items():
            if isinstance(value,bool) and key not in ('GPU_work_run','original_model_used_for_fixture'):
                assert value is True,(section,key)
    assert ready['all_40000_source_order_examples_checked'] and ready['effective_batches']==1250 and ready['total_supervised_tokens']==7100056
    assert ready['runtime_checks']['chunked_head_gradient_checks']['chunked_response_loss_matches_dense'] is True
    assert ready['runtime_checks']['chunked_head_gradient_checks']['all_parameter_gradients_match_dense'] is True
    assert ready['protocol_identity']==PROTOCOL and ready['methods']['full']['identity']==FULL_CONTRACT
    assert context['Drive_free_GB']*1e9/2**30>=ready['conservative_remaining_checkpoint_storage_GiB']
    original=json.loads((bundle/'CORE_FILES.json').read_text())
    assert {n:h for n,h in original.items() if n.endswith('.py')}==ready['files']
    for name,expected in original.items():assert '/' not in name and digest(core/name)==expected
    assert digest(job/'guardian.py')==ready['files']['gpu_guardian_R3.py']
    assert digest(job/'watchdog_evidence.py')==ready['files']['watchdog_evidence.py']
    assert digest(job/'release_coordination.py')==ready['files']['release_coordination.py']
    protocol_path=root/str(safe_relative(ready['protocol_path']));protocol=json.loads(protocol_path.read_text())
    assert identity(protocol)==PROTOCOL and identity(protocol['numerical_source_files'])==NUMERICAL
    contract_path=root/str(safe_relative(ready['methods']['full']['path']));contract=json.loads(contract_path.read_text())
    assert identity(contract)==FULL_CONTRACT and contract['method']=='full' and contract['protocol_identity']==PROTOCOL
    lock=root/'locks'/('full_'+FULL_CONTRACT+'_active.json')
    assert not lock.exists(),'A previous Full FT job still owns its active-job record. Release and review it on CPU before another training run.'
    batch_path=root/str(safe_relative(protocol['batch_plan_path']));batch_plan=json.loads(batch_path.read_text())
    assert identity(batch_plan)==protocol['batch_plan_identity']
    previous=latest_checkpoint(root,contract,596049920,batch_plan)
    start_update=previous['successful_updates'] if previous else 0
    assert start_update<312,'Full FT update 312 is already saved. Release and return the output; do not advance another boundary with Block 20.'
    launch_files={p.name:digest(p) for p in launch.glob('*.py')};launch_id=identity(launch_files)
    backup=root/'execution'/('training_full_launcher_'+launch_id)
    for name in launch_files:publish_bytes(backup/name,(launch/name).read_bytes())
    publish(backup/'FILES.json',launch_files)
    session={**context,'run_id':run_id,'method':'full','start_update':start_update,'target_update':312,
        'training_contract_identity':FULL_CONTRACT,'protocol_identity':PROTOCOL,'numerical_implementation_identity':NUMERICAL,
        'launch_implementation_identity':launch_id,'CPU_readiness_sha256':READY_SHA,
        'resume_checkpoint':previous,'status':'STARTED','Drive_free_source':'User reported 400 GB total minus 67.15 GB used on 2026-11-13',
        'conservative_remaining_checkpoint_storage_GiB':ready['conservative_remaining_checkpoint_storage_GiB'],
        'physical_release_verified':False}
    publish(root/'logs'/(run_id+'_session_started.json'),session)
    print('Official Full FT job: start update %d; target boundary 312 (9,984 examples).'%start_update,flush=True)
    local=Path('/content/atml_pa3_task1');local.mkdir(exist_ok=True);assert not local.is_symlink()
    old=json.loads((root/'configs/GPU_PREFLIGHT_KIT_READY.json').read_text())
    runtime,fixed=old['portable_runtime'],old['fixed_inputs']
    inventory_path=root/'runtime'/('inventory_'+runtime['inventory_sha256']+'.json')
    assert digest(inventory_path)==runtime['inventory_sha256'];inventory=json.loads(inventory_path.read_text())
    assert inventory['python_version']=='3.12.12' and inventory['base_directory']==str(local)
    input_path=root/str(safe_relative(old['input_inventory_path']))
    assert digest(input_path)==fixed['inventory_sha256'];input_entries=json.loads(input_path.read_text())
    assert len(input_entries)==fixed['files'] and len({e['path'] for e in input_entries})==len(input_entries)
    required=runtime['archive_bytes']+runtime['uncompressed_file_bytes']+fixed['archive_bytes']+sum(e['bytes'] for e in input_entries)+596049920*12+2*2**30
    assert shutil.disk_usage(local).free>=max(30*2**30,required),'Insufficient measured runtime disk for restoration/checkpoint staging'
    print('Restoring/verifying the frozen private Python software...',flush=True)
    restore_archive(root,local,runtime,inventory['files'],job,state)
    inputs=local/'training_inputs_V1';inputs.mkdir(exist_ok=True)
    print('Restoring/verifying the frozen original model and training inputs...',flush=True)
    restore_archive(root,inputs,fixed,input_entries,job,state)
    for rel in (ready['protocol_path'],ready['methods']['full']['path'],protocol['batch_plan_path']):
        safe_relative(rel);publish_bytes(inputs/rel,(root/rel).read_bytes())
    assert digest(inputs/'manifests/TRAINING_DATA_READY.json')==protocol['training_data_manifest_sha256']
    assert digest(inputs/'manifests/CPU_ASSETS_READY.json')==protocol['CPU_assets_manifest_sha256']
    assert time.time()+120< context['soft_wall'],'Setup exhausted the productive work allowance; no official updates started. Release and return this output.'
    config={**context,'run_id':run_id,'core':str(core),'inputs':str(inputs),
        'result':str(root/'logs'/(run_id+'_result.json')),'training_contract':contract,
        'protocol_identity':PROTOCOL,'protocol_local':str(inputs/ready['protocol_path']),
        'batch_plan_local':str(inputs/protocol['batch_plan_path']),'numerical_implementation_identity':NUMERICAL,
        'start_update':start_update,'target_update':312,'resume_checkpoint':previous['path'] if previous else None}
    config_path=job/'worker_config.json';atomic(config_path,config)
    env=os.environ.copy()
    for name in ('PYTHONPATH','PYTHONHOME','VIRTUAL_ENV','UV_SYSTEM_PYTHON','CUDA_VISIBLE_DEVICES'):env.pop(name,None)
    env.update(PYTHONNOUSERSITE='1',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',
        TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    lock.parent.mkdir(parents=True,exist_ok=True)
    from evaluation_records import encoded
    with lock.open('xb') as handle:handle.write(encoded({'run_id':run_id,'contract_identity':FULL_CONTRACT}))
    assert json.loads(lock.read_text())=={'run_id':run_id,'contract_identity':FULL_CONTRACT}
    # Preserve the ownership record on failure for CPU review; remove it only after
    # successful worker exit and verified final checkpoint/session acknowledgement.
    value=execute([str(local/'venv312/bin/python'),'-I','-u',str(launch/'full_training_worker.py'),str(config_path)],
        env,context,session,contract,596049920,batch_plan)
    assert json.loads(lock.read_text())['run_id']==run_id;lock.unlink()
    return value
