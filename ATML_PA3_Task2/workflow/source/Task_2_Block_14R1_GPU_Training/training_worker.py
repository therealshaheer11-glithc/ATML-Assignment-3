"""Bounded production training with exact checkpoint resume and scheduled development."""
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
    print('TRAINING PHASE:',name,flush=True)
    if time.monotonic()>args.soft_deadline:
        raise TimeoutError('Session soft deadline reached. The last verified training progress remains authoritative.')
    try:
        with bounded_io(15,'Initialization phase diagnostic'):
            save_json(RUN_ROOT/'logs'/f'block_14_phase_{args.attempt}_{phase_number:02d}.json',
                {'phase':phase,'elapsed_seconds':round(time.monotonic()-started,3),
                    'production_updates':globals()['session'].global_updates if 'session' in globals() else 0})
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
    report_phase('verifying the passed initialization and untrained checkpoint')
    from task2.training_state import cpu_copy
    from task2.development import evaluate_development
    from task2.training_progress import validate_counters,save_artifact,read_artifact,publish_progress
    from collections import Counter
    initial_reference=json.loads((Path(__file__).parent/'initial_state_reference.json').read_bytes())
    with bounded_io(120,'Passed initialization verification'):
        active_initial=json.loads((RUN_ROOT/'manifests/active_task2_initial_state.json').read_bytes())
        assert initial_reference['dataset_identity']==active_initial['dataset_identity']==DATASET_SHA
        assert active_initial['recipe_sha256']==initial_reference['recipe_sha256']
        assert active_initial['manifest_sha256']==initial_reference['initial_manifest_sha256']
        assert Path(active_initial['manifest_path'])==Path(initial_reference['initial_manifest_path'])
        initial_path=Path(initial_reference['initial_manifest_path'])
        assert file_sha(initial_path)==initial_reference['initial_manifest_sha256']
        initial=json.loads(initial_path.read_bytes())
        assert initial['dataset_identity']==DATASET_SHA
        assert initial['status']=='passed' and initial['precision_gradient_and_bit_exact_resume_preflight_passed']
        assert initial['production_updates_committed']==0 and initial['checkpoint']==initial_reference['checkpoint']
        recipe_path=Path(initial['recipe_path'])
        assert file_sha(recipe_path)==initial_reference['recipe_sha256']==initial['recipe_sha256']
        initial_recipe=json.loads(recipe_path.read_bytes())
        assert initial_recipe['scaler']['init_scale']==128.0
        for name,digest in initial_recipe['training_code_sha256'].items():
            assert file_sha(Path(__file__).parent/name)==digest, 'Preflighted numerical module differs: '+name
        assert file_sha(Path(__file__).parent/'task2/recovery.py')==initial_recipe['numerical_recovery']['recovery_module_sha256']
        assert json.loads(json.dumps(environment))==initial_reference['environment']
        original_state=load_checkpoint(initial['checkpoint'])
        validate_counters(original_state)
        assert original_state['global_updates']==0 and original_state['stage']==1 and original_state['scaler']['scale']==128.0
        assert not original_state['optimizer']['state']
        assert original_state['identity']['dataset_identity']==DATASET_SHA
        assert original_state['identity']['recipe_sha256']==initial['recipe_sha256']
        assert original_state['identity']['calibration_sha256']==initial['calibration_sha256']
        assert original_state['identity']['qwen_archive_sha256']==QWEN_ARCHIVE
        assert original_state['identity']['feature_manifest_sha256']==FEATURE_SHA and original_state['identity']['text_manifest_sha256']==TEXT_SHA
        assert original_state['identity']['configuration']==initial_recipe
    qwen_loading_started=time.monotonic()
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
    qwen_stage_load_seconds=time.monotonic()-qwen_loading_started
    report_phase('restoring Qwen, connector, adapters and complete training progress')
    connector=Connector().to(device)
    model=add_standard_lora(base)
    arrays={}
    feature_staging=[]
    shard_by_index={row['index']:row for row in features['shards']}
    def get_features(keys):
        values=[]
        for key in keys:
            row=tables['feature_lookup'][key];index=row['feature_shard_index']
            if index not in arrays:
                shard=shard_by_index[index]
                path=LOCAL_ROOT/'training_features'/Path(shard['feature_relative_path']).name
                before=time.monotonic()
                with bounded_io(120,'Feature shard staging and verification'):
                    activity=stage_file(dataset_dir/shard['feature_relative_path'],path,shard['feature_sha256'])
                    verify_features(path,shard['image_count'])
                arrays[index]=np.load(path,mmap_mode='r',allow_pickle=False)
                feature_staging.append({'index':index,'activity':activity,'seconds':round(time.monotonic()-before,3)})
            values.append(arrays[index][row['feature_row']])
        return torch.from_numpy(np.stack(values)).to(device=device,dtype=torch.float16)
    def sampler_factory():
        return CurriculumSampler(data['pool'],data['natural']['train'],data['captions']['train'],data['caption_orders'],data['question_bank'],DATASET_SHA)
    session=TrainingSession(model,connector,sampler_factory,tables,get_features,original_state['identity'],device,initial_scale=128.0)
    session.restore(original_state)
    assert_equal(original_state,session.snapshot())
    execution_recipe={'schema':'atml-pa3-task2-training-execution-v1','dataset_identity':DATASET_SHA,
        'initial_manifest_sha256':initial_reference['initial_manifest_sha256'],
        'initialization_recipe_sha256':initial_reference['recipe_sha256'],
        'source_preflighted_numerical_code':initial_recipe['training_code_sha256'],
        'development_code_sha256':file_sha(Path(__file__).parent/'task2/development.py'),
        'progress_code_sha256':file_sha(Path(__file__).parent/'task2/training_progress.py'),
        'training_worker_sha256':file_sha(__file__),
        'maximum_segment_updates':250,'periodic_complete_checkpoint_updates':50,
        'worker_soft_seconds':900,'save_margin_seconds':90,'additional_margin_before_scheduled_dev_seconds':120,'controller_hard_seconds':1200,
        'development_global_updates':list(DEV_UPDATES),'development':'Teacher-forced original reference losses only; no generation or test predictions',
        'caption_dev':'96 correct-image examples, content plus EOT token mean per example',
        'VQA_dev':'Grounding48 and natural64; correct, complementary and omitted-image; original reference content only',
        'training_loss':'Original preflighted effective16, complete pairs and equal-example loss',
        'RNG':'Development restores all mutable training state exactly',
        'scheduler':'Deterministic approved learning_rate(config, stage, one-based successful stage update); state is configuration plus saved stage/global counters and optimizer group LRs',
        'numerical_failure':'Stop/report; do not save a partially attempted update as completed; retain previous verified progress'}
    execution_sha=hashlib.sha256(encoded(execution_recipe)).hexdigest()
    training_dir=dataset_dir/'training'/initial_reference['recipe_sha256'][:16]
    with bounded_io(60,'Execution recipe save'):
        immutable_bytes(training_dir/('execution_'+execution_sha[:16]+'.json'),encoded(execution_recipe))
    pointer_path=RUN_ROOT/'manifests/active_task2_training.json'
    expected_pointer=pointer_path.read_bytes() if pointer_path.exists() else None
    dev_receipts={};log_receipts=[];last_progress=None
    last_checkpoint=initial['checkpoint']
    if expected_pointer is not None:
        pointer=json.loads(expected_pointer)
        assert pointer['dataset_identity']==DATASET_SHA and pointer['initial_manifest_sha256']==initial_reference['initial_manifest_sha256']
        assert pointer['execution_recipe_sha256']==execution_sha
        last_progress=read_artifact(pointer['progress'])
        assert last_progress['execution_recipe_sha256']==execution_sha
        state=load_checkpoint(last_progress['checkpoint'])
        validate_counters(state)
        assert state['global_updates']==last_progress['global_updates']==pointer['global_updates']
        session.restore(state)
        assert_equal(state,session.snapshot())
        last_checkpoint=last_progress['checkpoint']
        dev_receipts=last_progress['dev_receipts'];log_receipts=last_progress['training_log_receipts']
    start_update=session.global_updates
    segment_limit=min(4500,start_update+250)
    complete_update_boundary=True
    pending_records=[]
    completed_devs=[]
    update_times=[]
    segment_pair_counts=Counter()
    report_phase('prefetching completed feature shards to local storage')
    for index,shard in sorted(shard_by_index.items()):
        report_phase('verifying local feature shard '+str(index))
        first=shard['images'][0]
        get_features([(first['source_split'],first['image_id'])])
    def deadline_check():
        if time.monotonic()>args.soft_deadline-90:
            raise TimeoutError('Development reached the reserved checkpoint-saving window.')
    def weights_sha():
        state=session.snapshot()
        weights={**{'connector.'+key:value.contiguous() for key,value in state['connector'].items()},**{key:value.contiguous() for key,value in state['lora'].items()}}
        return hashlib.sha256(save_tensors(weights)).hexdigest()
    def development_if_due():
        key=str(session.global_updates)
        if session.global_updates not in DEV_UPDATES:return
        if key in dev_receipts:
            saved=read_artifact(dev_receipts[key])
            assert saved['global_update']==session.global_updates and saved['initial_manifest_sha256']==initial_reference['initial_manifest_sha256']
            assert saved['stage']==session.stage and saved['trainable_weights_sha256']==weights_sha()
            return
        report_phase('scheduled development reference losses at update '+key)
        metrics=evaluate_development(session,data,check_deadline=deadline_check,
            progress=lambda message:print('DEV:',message,flush=True))
        metrics.update({'dataset_identity':DATASET_SHA,'initial_manifest_sha256':initial_reference['initial_manifest_sha256'],
            'execution_recipe_sha256':execution_sha,'environment':environment,
            'stage':session.stage,'trainable_weights_sha256':weights_sha()})
        receipt=save_artifact(training_dir/'development','update_'+f'{session.global_updates:06d}',metrics)
        dev_receipts[key]=receipt;completed_devs.append(session.global_updates)
        print('DEVELOPMENT SUMMARY:',json.dumps({'global_update':session.global_updates,
            'caption_CE':metrics['caption']['correct_image_CE'],
            'VQA':{kind:{'conditions':{name:row['mean_reference_NLL'] for name,row in suite['conditions'].items()},
                'swap_penalty_NLL':suite['swap_penalty_NLL']} for kind,suite in metrics['suites'].items()},
            'receipt':receipt},sort_keys=True),flush=True)
    def persist(reason):
        global expected_pointer,last_progress,last_checkpoint,pending_records,log_receipts
        if not complete_update_boundary:raise RuntimeError('Refusing to save a partially attempted update.')
        state=session.snapshot();validate_counters(state)
        if session.global_updates in DEV_UPDATES and str(session.global_updates) not in dev_receipts:
            raise RuntimeError('Required development evaluation is incomplete at this checkpoint.')
        if pending_records:
            log_receipt=save_artifact(training_dir/'training_logs',
                f"updates_{pending_records[0]['global_update']:06d}_{pending_records[-1]['global_update']:06d}",pending_records)
            log_receipts=log_receipts+[log_receipt]
        checkpoint=save_checkpoint(training_dir/'checkpoints',f'update_{session.global_updates:06d}',state)
        endpoint=bool(session.stage_updates[session.stage]==STAGES[session.stage]['updates'])
        stage_weights=None
        if endpoint:
            weights={**{'connector.'+key:value.contiguous() for key,value in state['connector'].items()},
                **{key:value.contiguous() for key,value in state['lora'].items()}}
            raw=save_tensors(weights);digest=hashlib.sha256(raw).hexdigest()
            path=training_dir/'stage_weights'/f'stage{session.stage}_{digest[:16]}.safetensors'
            with bounded_io(90,'Stage-end connector and adapter weights save'):save_bytes(path,raw)
            stage_weights={'path':str(path),'sha256':digest,'bytes':len(raw)}
        progress={'schema':'atml-pa3-task2-training-progress-v1','dataset_identity':DATASET_SHA,
            'initial_manifest_sha256':initial_reference['initial_manifest_sha256'],'execution_recipe_sha256':execution_sha,
            'global_updates':session.global_updates,'stage':session.stage,
            'stage_updates':{str(k):v for k,v in session.stage_updates.items()},'checkpoint':checkpoint,
            'dev_receipts':dev_receipts,'training_log_receipts':log_receipts,
            'stage_endpoint':endpoint,'stage_weights':stage_weights,'reason':reason}
        receipt,expected_pointer=publish_progress(training_dir,pointer_path,progress,expected_pointer)
        last_checkpoint=checkpoint;last_progress=progress;pending_records=[]
        print('CHECKPOINT VERIFIED:',json.dumps({'global_updates':session.global_updates,'checkpoint':checkpoint,'progress':receipt},sort_keys=True),flush=True)
    development_if_due()
    if last_progress is None:persist('Development update-zero baseline completed before production training')
    report_phase('production training from update '+str(session.global_updates))
    stop_reason='segment update limit'
    while session.global_updates<segment_limit:
        estimated=max(update_times[-5:],default=15.0)
        evaluation_margin=120.0 if session.global_updates+1 in DEV_UPDATES else 0.0
        if time.monotonic()+estimated+90+evaluation_margin>args.soft_deadline:
            stop_reason='reserved save window';break
        development_if_due()
        if session.stage_updates[session.stage]==STAGES[session.stage]['updates']:
            if session.stage==3:break
            session.select_stage(session.stage+1)
        complete_update_boundary=False
        before=time.monotonic()
        step=session.update();torch.cuda.synchronize()
        seconds=time.monotonic()-before;update_times.append(seconds)
        complete_update_boundary=True
        kinds=[pair['kind'] for pair in step.pop('pairs')]
        segment_pair_counts.update(kinds)
        step.update({'global_update':session.global_updates,'pair_kinds':kinds,
            'pair_counts':dict(Counter(kinds)),'example_counts':dict(Counter({kind:2*count for kind,count in Counter(kinds).items()})),
            'wall_seconds':round(seconds,6)})
        pending_records.append(step)
        if session.global_updates%10==0 or session.global_updates==start_update+1:
            print('TRAIN:',json.dumps({key:step[key] for key in ['global_update','stage','stage_update','loss','learning_rates','scale_after','wall_seconds']},sort_keys=True),flush=True)
        development_if_due()
        if session.global_updates%50==0 or session.stage_updates[session.stage]==STAGES[session.stage]['updates']:
            persist('Periodic complete-update or stage-end checkpoint')
    if pending_records or last_progress['global_updates']!=session.global_updates:
        persist('Segment ended: '+stop_reason)
    assert last_progress['global_updates']==session.global_updates and not pending_records
    result={'block':'14','status':'passed' if session.global_updates==segment_limit else 'checkpointed',
        'dataset_identity':DATASET_SHA,'initial_manifest_sha256':initial_reference['initial_manifest_sha256'],
        'execution_recipe_sha256':execution_sha,'start_global_update':start_update,'end_global_update':session.global_updates,
        'successful_updates_this_segment':session.global_updates-start_update,'segment_maximum_updates':250,
        'total_required_updates':4500,'stop_reason':stop_reason,'stage':session.stage,'stage_updates':session.stage_updates,
        'AMP_scale':session.scaler.get_scale(),'parameter_counts':assert_precision(model,connector,session.stage),
        'realized_pairs_this_segment':dict(segment_pair_counts),'realized_examples_this_segment':{key:2*value for key,value in segment_pair_counts.items()},
        'checkpoint':last_checkpoint,'active_progress_pointer':str(pointer_path),
        'dev_updates_completed_this_segment':completed_devs,'development_receipts':dev_receipts,
        'training_log_receipts':log_receipts,'feature_staging':feature_staging,
        'training_update_wall_seconds':sum(update_times),'qwen_source_staging_and_loading_seconds':round(qwen_stage_load_seconds,3),
        'feature_staging_and_verification_seconds':sum(row['seconds'] for row in feature_staging),
        'elapsed_seconds':round(time.monotonic()-started,3),
        'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_gpu_reserved_bytes':torch.cuda.max_memory_reserved(),
        'environment':environment,'training_package_sha256':args.package_sha,'package_setup':package_setup,
        'final_test_evaluated':False,'next_step':'Final four-condition evaluation of completed Stage3' if session.global_updates==4500 else 'Resume from verified active training progress with the remaining approved curriculum budget.'}
    with bounded_io(60,'Training segment result save'):
        save_json(RUN_ROOT/'logs'/f'block_14_{args.attempt}.json',result)
    print(('BLOCK 14 PASSED' if result['status']=='passed' else 'BLOCK 14 CHECKPOINTED')+'\n'+json.dumps(result,indent=2),flush=True)
    print('Return the complete output before the next training segment.',flush=True)
except BaseException:
    diagnostic={'block':'14','status':'failed','phase':phase,'dataset_identity':DATASET_SHA,
        'training_package_sha256':args.package_sha,'traceback':traceback.format_exc(),
        'elapsed_seconds':round(time.monotonic()-started,3),'gpu_release_requested_by_controller_on_exit':True}
    try:
        if 'session' in globals():
            diagnostic.update({'successful_updates_in_memory':session.global_updates,'AMP_scale':session.scaler.get_scale(),
                'last_verified_checkpoint':globals().get('last_checkpoint'),
                'complete_update_boundary':globals().get('complete_update_boundary',False)})
            from task2.gradient_probe import tensor_summary
            diagnostic['gradient_details']={}
            for name,p in [('connector.'+n,p) for n,p in session.connector.named_parameters()]+list(session.model.named_parameters()):
                if not p.requires_grad:continue
                if name.startswith('connector.') or p.grad is None or p.grad.dtype!=torch.float32 or not torch.isfinite(p.grad).all():
                    diagnostic['gradient_details'][name]=tensor_summary(p.grad)
                    if len(diagnostic['gradient_details'])>=12:break
    except Exception as error:diagnostic['gradient_detail_error']=repr(error)
    print('BLOCK 14 DIAGNOSTIC\n'+json.dumps(diagnostic,indent=2),flush=True)
    try:
        with bounded_io(30,'Training failure diagnostic save'):
            save_json(RUN_ROOT/'logs'/f'block_14_failed_{args.attempt}.json',diagnostic)
    except BaseException as error:print('Failure diagnostic not saved:',repr(error),flush=True)
    raise
