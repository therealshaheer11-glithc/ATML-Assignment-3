"""Explicit AMP128 initialization recovery and rolled-back full preflight."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import argparse,hashlib,importlib.metadata,json,platform,random,shutil,subprocess,sys,time,traceback
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('--soft-deadline',type=float,required=True)
parser.add_argument('--package-sha',required=True)
parser.add_argument('--attempt',required=True)
args=parser.parse_args()
RUN_ROOT=Path('/content/drive/MyDrive/ATML_PA3/task2/run_20261031_150949_1cff148a')
LOCAL_ROOT=Path('/content/atml_pa3_task2/run_20261031_150949_1cff148a')
DATASET_SHA='82ad5006bd1d68ee54e639e50e7d835d8f6d97673537c38dab8d8eebdd8ac1b2'
FREEZE_SHA='fe7c9d4efa5d92d5b809eadb3695adcdd19f2421c6d43a2cb1593a7b86c3245b'
MODELS_SHA='66f14f21d911fb57ae379df700ed3161cf348a997922235d8ab8b56491604f7f'
FEATURE_SHA='1a015e4a07f2406937d63832dae1d648148a243a7931b39358d68e2a2ad8bea6'
FEATURE_RECIPE='380f19811df42cc56d3dfc3ecdf6496bae3557d11c27f1b2f64fd4b2a33bc2c6'
TEXT_SHA='8d10d1daea771af72d58441615a99e9b4fcd155fa36adde8b08cfcc7cacb2490'
TEXT_RECIPE='e14ef6fd5200cb4d58ad0b61ca5b4669b0ffb5af6cde9384f7fa225f1147bc6c'
QWEN_ARCHIVE='a3c43cd415968956b078c4463b4e112866344c0c20235a2a9049db15f0d0d0e9'
PINS={'transformers':'5.22.1','huggingface-hub':'1.33.0','tokenizers':'0.23.3',
    'safetensors':'0.8.0','Pillow':'11.3.0','numpy':'2.4.6','peft':'0.21.2','accelerate':'1.15.0'}
from task2.session_io import bounded_io,save_json,save_bytes
started=time.monotonic()
phase='startup'
phase_number=0
validation_steps=0

def report_phase(name):
    global phase,phase_number
    phase=name;phase_number+=1
    print('INITIALIZATION PHASE:',name,flush=True)
    if time.monotonic()>args.soft_deadline:
        raise TimeoutError('Initialization soft deadline reached. No production state has advanced.')
    try:
        with bounded_io(15,'Initialization phase diagnostic'):
            save_json(RUN_ROOT/'logs'/f'block_13_phase_{args.attempt}_{phase_number:02d}.json',
                {'phase':phase,'elapsed_seconds':round(time.monotonic()-started,3),
                    'production_updates':0,'temporary_validation_updates':validation_steps})
    except TimeoutError:raise
    except Exception as error:print('Phase diagnostic not saved:',repr(error),flush=True)


def prepare_libraries():
    installed={}
    for name in PINS:
        try:installed[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:installed[name]=None
    if installed==PINS:return 'Exact existing pinned libraries reused'
    tag=hashlib.sha256(json.dumps(PINS,sort_keys=True).encode()).hexdigest()[:16]
    target=LOCAL_ROOT/'training_python_packages'/tag
    marker=target/'complete.json'
    if not marker.exists():
        target.mkdir(parents=True,exist_ok=True)
        subprocess.run([sys.executable,'-m','pip','install','--quiet','--no-deps','--upgrade',
            '--target',str(target),*[f'{k}=={v}' for k,v in PINS.items()]],check=True,timeout=240)
        marker.write_text(json.dumps(PINS,sort_keys=True))
    else:assert json.loads(marker.read_text())==PINS
    sys.path.insert(0,str(target))
    assert {k:importlib.metadata.version(k) for k in PINS}==PINS
    return 'Pinned libraries isolated locally; existing CUDA PyTorch retained'


try:
    report_phase('preparing pinned libraries')
    package_setup=prepare_libraries()
    from task2.runtime_policy import isolate_unused_torchao
    optional_backend_policy=isolate_unused_torchao()
    print('Optional backend policy:',json.dumps(optional_backend_policy,sort_keys=True),flush=True)
    import numpy as np
    import torch
    from safetensors.torch import save as save_tensors
    from transformers import Qwen3ForCausalLM
    from task2.images import encoded,file_sha,immutable_bytes
    from task2.features import stage_file,unpack_model,verify_features
    from task2.dataset import load_dataset,validate_dataset
    from task2.text_data import load_text_tables,build_feature_lookup
    from task2.sampler import CurriculumSampler
    from task2.protocol import STAGES,DEV_UPDATES
    from task2.modeling import Connector,add_standard_lora,set_stage,assert_precision,calibrate,is_lora
    from task2.training_state import TrainingSession,assert_equal,save_checkpoint,load_checkpoint,rng_witness
    if not torch.cuda.is_available() or torch.version.cuda is None:
        raise RuntimeError('Working CUDA PyTorch is required. No model was loaded.')
    capability=list(torch.cuda.get_device_capability(0))
    if capability[0]<7:raise RuntimeError('Use a T4 or newer CUDA GPU.')
    if shutil.disk_usage(LOCAL_ROOT).free<8*1024**3:raise RuntimeError('At least 8 GiB local free disk space is required.')
    device=torch.device('cuda:0')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    torch.set_float32_matmul_precision('highest')
    environment={'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,
        'cudnn':torch.backends.cudnn.version(),'gpu':torch.cuda.get_device_name(0),'capability':capability,
        'gpu_total_memory_bytes':torch.cuda.get_device_properties(0).total_memory,
        'nvidia_driver':subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True,timeout=10).strip(),
        **{k:importlib.metadata.version(k) for k in PINS},'optional_backend_policy':optional_backend_policy}
    print('CUDA environment:',json.dumps(environment,sort_keys=True),flush=True)
    report_phase('verifying frozen data, model, feature and text identities')
    dataset_dir=RUN_ROOT/'datasets'/('task2_A012_'+DATASET_SHA[:16])
    with bounded_io(150,'All prerequisite identities'):
        pointer=json.loads((RUN_ROOT/'manifests/active_task2_dataset.json').read_bytes())
        assert pointer['dataset_identity']==DATASET_SHA and Path(pointer['dataset_dir'])==dataset_dir
        assert pointer['freeze_receipt_sha256']==FREEZE_SHA and file_sha(dataset_dir/'freeze_receipt.json')==FREEZE_SHA
        data=load_dataset(dataset_dir)
        assert data['identity']==DATASET_SHA
        validate_dataset(data)
        feature_path=dataset_dir/'features'/('siglip196_'+FEATURE_RECIPE[:16])/'features_complete.json'
        text_dir=dataset_dir/'inputs'/('text_'+TEXT_RECIPE[:16])
        model_path=dataset_dir/'inputs/model_assets'/('models_complete_'+MODELS_SHA[:16]+'.json')
        for pointer_name,path,sha in [('features',feature_path,FEATURE_SHA),('text',text_dir/'text_complete.json',TEXT_SHA),('model_assets',model_path,MODELS_SHA)]:
            active=json.loads((RUN_ROOT/'manifests'/('active_task2_'+pointer_name+'.json')).read_bytes())
            assert active['dataset_identity']==DATASET_SHA and active['manifest_sha256']==sha
            assert Path(active['manifest_path'])==path and file_sha(path)==sha
        features=json.loads(feature_path.read_bytes())
        assert features['status']=='passed' and features['image_count']==9260 and features['shard_count']==37
        assert features['recipe_sha256']==FEATURE_RECIPE
        tables=load_text_tables(text_dir,TEXT_SHA)
        expected_lookup=build_feature_lookup(features,data)
        assert { (r['source_split'],r['image_id']):r for r in expected_lookup}==tables['feature_lookup']
        assert tables['calibration']['image_ids']==data['calibration']['image_ids']
        assert tables['calibration']['caption_ids']==data['calibration']['caption_ids']
        asset=json.loads(model_path.read_bytes())['models']['qwen']
        assert asset['repo_id']=='Qwen/Qwen3-0.6B' and asset['revision']=='c1899de289a04d12100db370d81485cdf75e47ca'
        assert asset['archive_sha256']==QWEN_ARCHIVE
        relative=Path(asset['archive_relative_path'])
        assert relative.name==str(relative)
    report_phase('verifying the saved untrained candidate and calibration evidence')
    reference=json.loads((Path(__file__).parent/'initialization_reference.json').read_bytes())
    previous=reference['failed_package_receipt']
    for name,digest in previous['modules'].items():
        assert file_sha(Path(__file__).parent/name)==digest, 'Diagnostic must use the exact failed-run computational code: '+name
    matching=[]
    with bounded_io(120,'Untrained candidate discovery and verification'):
        for recipe_path in sorted((dataset_dir/'initialization').glob('*/recipe.json')):
            recipe_raw=recipe_path.read_bytes()
            recipe=json.loads(recipe_raw)
            if recipe.get('initialization_worker_sha256')!=previous['worker_sha256']:continue
            recipe_sha=hashlib.sha256(recipe_raw).hexdigest()
            assert recipe_path.parent.name==recipe_sha[:16]
            assert recipe['dataset_identity']==DATASET_SHA and recipe['feature_manifest_sha256']==FEATURE_SHA and recipe['text_manifest_sha256']==TEXT_SHA
            for cal_path in sorted(recipe_path.parent.glob('calibration_*.json')):
                cal_raw=cal_path.read_bytes();cal=json.loads(cal_raw)
                cal_sha=hashlib.sha256(cal_raw).hexdigest()
                assert cal_path.name=='calibration_'+cal_sha[:16]+'.json'
                if any(cal.get(key)!=value for key,value in reference['observed_calibration'].items()):continue
                assert cal['image_ids']==data['calibration']['image_ids'] and cal['caption_ids']==data['calibration']['caption_ids']
                assert file_sha(cal['initial_connector_path'])==cal['initial_connector_sha256']
                for candidate in sorted((recipe_path.parent/'checkpoints').glob('initial_candidate_*.pt')):
                    content=candidate.read_bytes();digest=hashlib.sha256(content).hexdigest()
                    assert candidate.name=='initial_candidate_'+digest[:16]+'.pt'
                    state=torch.load(__import__('io').BytesIO(content),map_location='cpu',weights_only=False)
                    identity=state['identity']
                    if identity['recipe_sha256']!=recipe_sha or identity['calibration_sha256']!=cal_sha:continue
                    assert identity['dataset_identity']==DATASET_SHA and identity['qwen_archive_sha256']==QWEN_ARCHIVE
                    assert identity['feature_manifest_sha256']==FEATURE_SHA and identity['text_manifest_sha256']==TEXT_SHA
                    assert identity['configuration']==recipe
                    assert json.loads(json.dumps(identity['environment']))==reference['observed_environment']==json.loads(json.dumps(environment))
                    assert state['global_updates']==0 and state['stage']==1 and state['stage_updates']=={1:0,2:0,3:0}
                    assert state['sampler']['caption_cursor']==0 and not state['optimizer']['state']
                    assert state['scaler']['scale']==65536.0
                    matching.append((state,{'path':str(candidate),'sha256':digest,'bytes':len(content)},cal))
        assert len(matching)==1, 'Expected one exact saved untrained candidate from failed 13R1, found '+str(len(matching))
    canonical,candidate_receipt,calibration=matching[0]
    report_phase('staging the verified original Qwen archive')
    model_folder=LOCAL_ROOT/'gpu_qwen_originals'/asset['revision']
    local_archive=model_folder.parent/('original_'+QWEN_ARCHIVE[:16]+'.tar')
    with bounded_io(180,'Original Qwen archive staging'):
        stage_file(RUN_ROOT/'model_assets/qwen'/asset['revision']/relative,local_archive,QWEN_ARCHIVE)
    with bounded_io(150,'Original Qwen member verification'):
        unpack_model(local_archive,model_folder,asset['files'])
    report_phase('loading full original Qwen in frozen FP16')
    base,loading=Qwen3ForCausalLM.from_pretrained(str(model_folder),dtype=torch.float16,
        attn_implementation='eager',local_files_only=True,use_safetensors=True,output_loading_info=True)
    if any(loading.get(key) for key in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
        raise ValueError('Original Qwen loading is not exact: '+repr(loading))
    assert (base.config.hidden_size,base.config.num_hidden_layers,base.config.vocab_size)==(1024,28,151936)
    assert base.config.tie_word_embeddings and base.config._attn_implementation=='eager'
    base.requires_grad_(False).to(device=device,dtype=torch.float16)
    assert base.get_input_embeddings().weight.data_ptr()==base.lm_head.weight.data_ptr()
    assert all(p.dtype==torch.float16 and not p.requires_grad for p in base.parameters())
    base.config.use_cache=False
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    report_phase('verifying the completed gradient diagnostic and exact source candidate')
    from task2.recovery import amp128_initial_state
    from task2.training_state import cpu_copy
    diagnostic_reference=json.loads((Path(__file__).parent/'gradient_diagnostic_reference.json').read_bytes())
    diagnosed=diagnostic_reference['observed_result']
    assert diagnosed['status']=='diagnosed' and diagnosed['optimizer_steps']==0
    assert diagnosed['all_canonical_state_rollbacks_exact'] and diagnosed['same_first_sixteen_examples_and_starting_RNG_each_probe']
    assert diagnosed['candidate_checkpoint']==candidate_receipt
    assert diagnosed['gradient_diagnostic_package_sha256']=='593c67f036a8aafe784b372cb07ee55c20126d00221c337ec5dbebebd970a7f1'
    by_scale={row['diagnostic_loss_scale']:row for row in diagnosed['probes']}
    assert not by_scale[65536.0]['connector_gradients_finite_FP32']
    assert by_scale[128.0]['connector_gradients_finite_FP32'] and by_scale[128.0]['connector_gradient_nonzero']
    assert by_scale[65536.0]['micro_losses']==by_scale[128.0]['micro_losses']
    source_identity=canonical['identity']
    recipe=json.loads(json.dumps(source_identity['configuration']))
    recipe['schema']='atml-pa3-task2-initialization-amp128-v1'
    recipe['scaler']['init_scale']=128.0
    recipe['initialization_worker_sha256']=file_sha(__file__)
    recipe['numerical_recovery']={'kind':'Explicit update-zero AMP initialization',
        'previous_initial_scale':65536.0,'new_initial_scale':128.0,
        'source_candidate':candidate_receipt,'source_initialization_recipe_sha256':source_identity['recipe_sha256'],
        'diagnostic_source_sha256':diagnostic_reference['source_attachment_sha256'],
        'diagnostic_package_sha256':diagnosed['gradient_diagnostic_package_sha256'],
        'recovery_module_sha256':file_sha(Path(__file__).parent/'task2/recovery.py'),
        'scope':'Change only initial dynamic-AMP scale at update zero. Preserve weights, fixed RMS scalar, sampler/RNGs, optimizer settings, clipping, dtypes, batch and curriculum.',
        'later_nonfinite_or_skipped_steps':'Stop and report; no automatic retries or dtype/batch/budget changes'}
    recipe_sha=hashlib.sha256(encoded(recipe)).hexdigest()
    output_dir=dataset_dir/'initialization'/recipe_sha[:16]
    identity={**source_identity,'recipe_sha256':recipe_sha,'configuration':recipe}
    recovered=amp128_initial_state(canonical,identity)
    connector=Connector().to(device)
    assert not getattr(base,'is_quantized',False) and getattr(base,'quantization_method',None) is None and getattr(base.config,'quantization_config',None) is None
    assert all(type(p) is torch.nn.Parameter for p in base.parameters())
    model=add_standard_lora(base)
    from peft.tuners.lora.layer import Linear as StandardLoraLinear
    adapter_layers=[module for name,module in model.named_modules() if name.rsplit('.',1)[-1] in ('q_proj','k_proj','v_proj','o_proj')]
    assert len(adapter_layers)==112 and all(type(module) is StandardLoraLinear for module in adapter_layers)
    arrays={}
    feature_staging=[]
    shard_by_index={row['index']:row for row in features['shards']}
    def get_features(keys):
        values=[]
        for key in keys:
            if key[0]!='train2014' or key[1] not in data['pool']['image_ids']:
                raise ValueError('Initialization and validation may only consume training images.')
            row=tables['feature_lookup'][key]
            index=row['feature_shard_index']
            if index not in arrays:
                report_phase('staging calibration/training feature shard '+str(index))
                shard=shard_by_index[index]
                path=LOCAL_ROOT/'training_features'/Path(shard['feature_relative_path']).name
                before=time.monotonic()
                with bounded_io(120,'Feature shard staging and local verification'):
                    activity=stage_file(dataset_dir/shard['feature_relative_path'],path,shard['feature_sha256'])
                    verify_features(path,shard['image_count'])
                arrays[index]=np.load(path,mmap_mode='r',allow_pickle=False)
                feature_staging.append({'index':index,'activity':activity,'seconds':round(time.monotonic()-before,3)})
                assert row['feature_sha256']==shard['feature_sha256']
            values.append(arrays[index][row['feature_row']])
        return torch.from_numpy(np.stack(values)).to(device=device,dtype=torch.float16)
    def sampler_factory():
        return CurriculumSampler(data['pool'],data['natural']['train'],data['captions']['train'],data['caption_orders'],data['question_bank'],DATASET_SHA)
    session=TrainingSession(model,connector,sampler_factory,tables,get_features,identity,device,initial_scale=128.0)
    session.restore(recovered)
    assert_equal(recovered,session.snapshot())
    canonical=recovered
    counts=assert_precision(model,connector,1)
    assert counts['connector']==1837056 and counts['lora']==4587520 and counts['frozen_base']==596049920
    assert session.scaler.get_scale()==128.0
    raw_connector_sha=calibration['initial_connector_sha256']
    calibration_sha=source_identity['calibration_sha256']
    with bounded_io(60,'Recovery recipe save'):
        immutable_bytes(output_dir/'recipe.json',encoded(recipe))
    print('RMS calibration reused without change:',json.dumps({key:calibration[key] for key in ['text_rms','visual_rms','scale_fp32']},sort_keys=True),flush=True)
    print('AMP initial scale explicitly set to 128.0 at update zero; dynamic scaling remains enabled.',flush=True)
    report_phase('saving the untrained canonical full mutable state')
    initial_receipt=save_checkpoint(output_dir/'checkpoints','initial_candidate',canonical)
    report_phase('temporary Stage 1 effective-batch-16 precision/gradient update')
    start=time.monotonic()
    stage1=session.update();torch.cuda.synchronize();validation_steps+=1
    stage1_seconds=time.monotonic()-start
    assert not stage1['lora_A_gradient_nonzero'] and not stage1['lora_B_gradient_nonzero']
    session.restore(canonical)
    assert_equal(canonical,session.snapshot())
    report_phase('temporary Stage 2 update with active LoRA and dropout')
    session.select_stage(2)
    first=session.update();torch.cuda.synchronize();validation_steps+=1
    report_phase('saving a full temporary checkpoint after a complete LoRA update')
    interrupted=session.snapshot()
    interrupted_receipt=save_checkpoint(output_dir/'validation'/args.attempt,'after_stage2_update1',interrupted)
    report_phase('reference second LoRA update')
    witness=rng_witness(device)
    start=time.monotonic()
    reference_step=session.update();torch.cuda.synchronize();validation_steps+=1
    stage2_seconds=time.monotonic()-start
    reference=session.snapshot()
    report_phase('restoring optimizer/scaler/sampler/all RNGs and repeating the second update')
    session.restore(load_checkpoint(interrupted_receipt))
    assert_equal(interrupted,session.snapshot())
    assert_equal(witness,rng_witness(device),'RNG witnesses')
    resumed_step=session.update();torch.cuda.synchronize();validation_steps+=1
    assert_equal(reference_step,resumed_step,'next update including losses, LR and sixteen selected examples')
    assert_equal(reference,session.snapshot(),'full state after the resumed update')
    assert resumed_step['lora_A_gradient_nonzero'] and resumed_step['lora_B_gradient_nonzero']
    # Stage transitions retain all moments and scaler state.
    moments=reference['optimizer'];scaler_state=reference['scaler']
    session.select_stage(3)
    assert_equal(moments,session.snapshot()['optimizer'],'stage2-to3 moments')
    assert_equal(scaler_state,session.scaler.state_dict(),'stage2-to3 scaler')
    report_phase('restoring and verifying the canonical untrained state')
    session.restore(load_checkpoint(initial_receipt))
    assert_equal(canonical,session.snapshot())
    assert session.global_updates==0 and session.stage_updates=={1:0,2:0,3:0}
    assert not session.optimizer.state and session.sampler.caption_cursor==0
    assert validation_steps==4
    preflight={'status':'passed','full_original_Qwen_loaded':True,'exact_source_loading':True,
        'tied_embeddings':True,'FP16_frozen_base_FP32_trainables_and_moments':True,
        'effective16_complete_pair_gradients_verified':True,'gradient_checkpointing':True,
        'LoRA_A_and_B_nonzero_gradients_after_initial_B_update':True,
        'full_state_disk_save_and_bit_exact_next_update_resume':True,
        'resume_checks':['connector','adapters','optimizer moments and groups','scaler','sampler and next complete pairs',
            'Python/NumPy/Torch CPU/CUDA RNGs','dropout','LRs','losses','stage and successful counters'],
        'stage2_to3_moments_and_scaler_preserved':True,
        'canonical_state_restored_exactly':True,'production_updates_committed':0,
        'temporary_validation_updates':4,'temporary_example_presentations':64,
        'stage1_validation_update_seconds':round(stage1_seconds,3),'stage2_validation_update_seconds':round(stage2_seconds,3),
        'validation_checkpoint':interrupted_receipt,'stage1_loss':stage1['loss'],'stage2_reference_loss':reference_step['loss'],
        'stage1_gradient_norm':stage1['unclipped_gradient_norm'],'stage2_gradient_norm':reference_step['unclipped_gradient_norm'],
        'AMP_scale':session.scaler.get_scale(),'AMP_initial_scale':128.0,'original_candidate':candidate_receipt,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),
        'peak_gpu_reserved_bytes':torch.cuda.max_memory_reserved()}
    manifest={'schema':'atml-pa3-task2-initial-state-v1','status':'passed','dataset_identity':DATASET_SHA,
        'recipe_sha256':recipe_sha,'recipe_path':str(output_dir/'recipe.json'),
        'calibration_sha256':calibration_sha,'calibration_path':str(Path(candidate_receipt['path']).parent.parent/('calibration_'+calibration_sha[:16]+'.json')),
        'initial_connector_sha256':raw_connector_sha,'checkpoint':initial_receipt,
        'production_updates_committed':0,'temporary_validation_updates_rolled_back':4,
        'precision_gradient_and_bit_exact_resume_preflight_passed':True,
        'dev_update_zero_completed':False,'parameter_counts':counts}
    manifest_sha=hashlib.sha256(encoded(manifest)).hexdigest()
    manifest_path=output_dir/('initial_state_'+manifest_sha[:16]+'.json')
    report_phase('publishing the verified initial-state pointer')
    with bounded_io(120,'Initial manifest and active pointer save'):
        immutable_bytes(manifest_path,encoded(manifest))
        assert file_sha(manifest_path)==manifest_sha
        active={'dataset_identity':DATASET_SHA,'recipe_sha256':recipe_sha,
            'manifest_path':str(manifest_path),'manifest_sha256':manifest_sha,'production_updates':0}
        # Preserve an already-published different run rather than overwrite it.
        immutable_bytes(RUN_ROOT/'manifests/active_task2_initial_state.json',encoded(active))
    result={'block':'13','status':'passed','dataset_identity':DATASET_SHA,
        'initial_manifest_path':str(manifest_path),'initial_manifest_sha256':manifest_sha,
        'recipe_sha256':recipe_sha,'calibration':{key:calibration[key] for key in ['text_rms','visual_rms','scale_fp32','text_elements','visual_elements','initial_connector_sha256']},
        'initial_checkpoint':initial_receipt,'preflight':preflight,'parameter_counts':counts,
        'feature_staging':feature_staging,'initialization_package_sha256':args.package_sha,
        'package_setup':package_setup,'environment':environment,'elapsed_seconds':round(time.monotonic()-started,3),
        'production_training_started':False,'next_step':'Dev update-zero baseline, then the first bounded training segment'}
    with bounded_io(60,'Initialization result save'):
        save_json(RUN_ROOT/'logs'/f'block_13_{args.attempt}.json',result)
    print('BLOCK 13 PASSED\n'+json.dumps(result,indent=2),flush=True)
    print('Return the complete output before the baseline and first production training segment.',flush=True)
except BaseException:
    diagnostic={'block':'13','status':'failed','phase':phase,'dataset_identity':DATASET_SHA,
        'initialization_package_sha256':args.package_sha,'production_updates_committed':0,
        'temporary_validation_updates_completed':validation_steps,'traceback':traceback.format_exc(),
        'elapsed_seconds':round(time.monotonic()-started,3),'gpu_release_requested_by_controller_on_exit':True}
    try:
        if 'session' in globals():
            from task2.gradient_probe import tensor_summary
            diagnostic['AMP_scale_at_failure']=session.scaler.get_scale()
            diagnostic['gradient_details']={}
            for name,p in [('connector.'+n,p) for n,p in session.connector.named_parameters()]+list(session.model.named_parameters()):
                if not p.requires_grad:continue
                if name.startswith('connector.') or p.grad is None or p.grad.dtype!=torch.float32 or not torch.isfinite(p.grad).all():
                    diagnostic['gradient_details'][name]=tensor_summary(p.grad)
                    if len(diagnostic['gradient_details'])>=12:break
    except Exception as error:
        diagnostic['gradient_detail_error']=repr(error)
    print('BLOCK 13 DIAGNOSTIC\n'+json.dumps(diagnostic,indent=2),flush=True)
    try:
        with bounded_io(30,'Failure diagnostic save'):
            save_json(RUN_ROOT/'logs'/f'block_13_failed_{args.attempt}.json',diagnostic)
    except BaseException as error:print('Failure diagnostic not saved:',repr(error),flush=True)
    raise
