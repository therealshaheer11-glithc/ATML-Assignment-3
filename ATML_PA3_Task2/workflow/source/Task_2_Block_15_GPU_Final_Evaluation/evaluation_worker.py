"""Final evaluation of the fixed completed model; no optimizer updates."""
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
    print('EVALUATION PHASE:',name,flush=True)
    if time.monotonic()>args.soft_deadline:
        raise TimeoutError('Session soft deadline reached. The last verified training progress remains authoritative.')
    try:
        with bounded_io(15,'Initialization phase diagnostic'):
            save_json(RUN_ROOT/'logs'/f'block_15_phase_{args.attempt}_{phase_number:02d}.json',
                {'phase':phase,'elapsed_seconds':round(time.monotonic()-started,3),
                    'training_performed':False,'source_training_updates':4500})
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
    from task2.final_evaluation import build_plan,batches,image_keys,generate_batch,validate_chunk
    from task2.evaluation_artifacts import write_artifact,read_receipt,verify_manifest
    from task2.text import native_prompt
    from transformers import AutoTokenizer
    from task2.training_progress import read_artifact
    from task2.modeling import set_stage
    import uuid

    report_phase('verifying the completed 4500-update checkpoint')
    final_reference=json.loads((Path(__file__).parent/'final_state_reference.json').read_bytes())
    pointer_path=RUN_ROOT/'manifests/active_task2_training.json'
    with bounded_io(120,'Final completed checkpoint verification'):
        source_pointer_raw=pointer_path.read_bytes()
        source_pointer=json.loads(source_pointer_raw)
        assert source_pointer['global_updates']==4500
        assert source_pointer['progress']==final_reference['progress']
        for key in ('dataset_identity','initial_manifest_sha256','execution_recipe_sha256'):
            assert source_pointer[key]==final_reference[key]
        progress=read_receipt(source_pointer['progress'],RUN_ROOT)
        assert progress['checkpoint']==final_reference['checkpoint']
        assert progress['stage_updates']=={'1':2250,'2':1500,'3':750} and progress['stage']==3
        completed=json.loads((RUN_ROOT/'logs'/('block_14_status_'+final_reference['completion_attempt']+'.json')).read_bytes())
        assert completed['code']=='COMPLETE' and completed['global_updates']==4500
        assert completed['checkpoint_sha256']==final_reference['checkpoint']['sha256']
        state=load_checkpoint(final_reference['checkpoint'])
        assert_equal(original_state['identity'],state['identity'],'unchanged training identity')
        assert state['global_updates']==4500 and state['stage']==3 and state['stage_updates']=={1:2250,2:1500,3:750}
        validate_counters(state)
    connector.load_state_dict(state['connector'],strict=True)
    adapters={name:p for name,p in model.named_parameters() if is_lora(name)}
    assert adapters.keys()==state['lora'].keys()
    with torch.no_grad():
        for name,p in adapters.items():
            saved=state['lora'][name]
            assert p.shape==saved.shape and p.dtype==saved.dtype
            p.copy_(saved)
    set_stage(model,connector,3,training=False)
    counts=assert_precision(model,connector,3)
    assert connector.scale.item()==original_state['connector']['scale'].item()
    model.requires_grad_(False);connector.requires_grad_(False)
    model.eval();connector.eval()
    base.gradient_checkpointing_disable()
    tokenizer=AutoTokenizer.from_pretrained(str(model_folder),local_files_only=True,trust_remote_code=False)

    # The mapping is saved BEFORE any final-test generation; it never uses predictions.
    report_phase('freezing the final-test plan and fixed image-control mapping')
    plan=build_plan(data)
    recipe={'schema':'atml-pa3-task2-final-evaluation-recipe-v1','source_reference':final_reference,
        'feature_manifest_sha256':FEATURE_SHA,'text_manifest_sha256':TEXT_SHA,
        'plan_sha256':hashlib.sha256(encoded(plan)).hexdigest(),
        'generation':plan['generation'],'reference_NLL':plan['reference_NLL'],
        'generation_timing':plan['generation_timing'],
        'worker_sha256':file_sha(__file__),
        'evaluation_code_sha256':file_sha(Path(__file__).parent/'task2/final_evaluation.py'),
        'artifact_code_sha256':file_sha(Path(__file__).parent/'task2/evaluation_artifacts.py'),
        'modeling_code_sha256':file_sha(Path(__file__).parent/'task2/modeling.py'),
        'text_code_sha256':file_sha(Path(__file__).parent/'task2/text.py'),
        'development_NLL_code_sha256':file_sha(Path(__file__).parent/'task2/development.py'),
        'environment':environment,'training_performed':False}
    recipe_sha=hashlib.sha256(encoded(recipe)).hexdigest()
    evaluation_dir=dataset_dir/'evaluation'/('stage3_'+recipe_sha[:16])
    plan_receipt=write_artifact(evaluation_dir/'plan.json',plan)
    recipe_receipt=write_artifact(evaluation_dir/'recipe.json',recipe)
    assert recipe_receipt['sha256']==recipe_sha
    eval_pointer_path=RUN_ROOT/'manifests/active_task2_evaluation.json'
    expected_eval_pointer=eval_pointer_path.read_bytes() if eval_pointer_path.exists() else None
    manifest={'schema':'atml-pa3-task2-final-evaluation-progress-v1',
        'recipe_sha256':recipe_sha,'plan_sha256':plan_receipt['sha256'],
        'batches':{},'verified_answers':0,'complete':False}
    if expected_eval_pointer is not None:
        ep=json.loads(expected_eval_pointer)
        assert ep['recipe_sha256']==recipe_sha,'An existing final evaluation has another recipe; do not silently rerun or replace it.'
        manifest=read_receipt(ep['manifest'],RUN_ROOT)
        with bounded_io(120,'Saved evaluation batches verification'):
            verify_manifest(manifest,plan,recipe_sha,RUN_ROOT)
    source_eval_count=manifest['verified_answers']
    all_batches=list(batches(plan))
    def publish():
        global manifest_receipt,expected_eval_pointer
        manifest['verified_answers']=8*len(manifest['batches'])
        manifest['complete']=len(manifest['batches'])==len(all_batches)
        raw=encoded(manifest);digest=hashlib.sha256(raw).hexdigest()
        manifest_receipt=write_artifact(evaluation_dir/'progress'/('answers_'+str(manifest['verified_answers']).zfill(4)+'_'+digest[:16]+'.json'),manifest)
        pointer={'schema':'atml-pa3-task2-active-final-evaluation-v1',
            'recipe_sha256':recipe_sha,'dataset_identity':DATASET_SHA,
            'checkpoint_sha256':final_reference['checkpoint']['sha256'],
            'verified_answers':manifest['verified_answers'],'complete':manifest['complete'],
            'manifest':manifest_receipt,'recipe':recipe_receipt,'plan':plan_receipt}
        with bounded_io(45,'Evaluation pointer publication'):
            current=eval_pointer_path.read_bytes() if eval_pointer_path.exists() else None
            if current!=expected_eval_pointer:raise RuntimeError('Another evaluation changed active progress.')
            if current is not None:
                save_bytes(evaluation_dir/'pointer_history'/(hashlib.sha256(current).hexdigest()+'.json'),current)
            raw=encoded(pointer);temporary=eval_pointer_path.with_name(eval_pointer_path.name+'.part_'+uuid.uuid4().hex)
            try:
                temporary.write_bytes(raw)
                assert temporary.read_bytes()==raw
                temporary.replace(eval_pointer_path)
                assert eval_pointer_path.read_bytes()==raw
            finally:temporary.unlink(missing_ok=True)
            expected_eval_pointer=raw
    publish()

    needed=sorted({key for _,condition,rows in all_batches for key in (image_keys(rows,condition) or [])})
    for start in range(0,len(needed),8):
        report_phase('staging final-test cached features '+str(start)+'/'+str(len(needed)))
        get_features(needed[start:start+8])
    torch.cuda.reset_peak_memory_stats()
    generation_seconds=0.0;nll_seconds=0.0;batch_seconds=[]
    for key,condition,rows in all_batches:
        if key in manifest['batches']:continue
        if time.monotonic()+max(batch_seconds[-5:],default=20)+90>args.soft_deadline:break
        phase='final evaluation '+key
        batch_path=evaluation_dir/'batches'/(key+'.json')
        if batch_path.exists():
            raw=batch_path.read_bytes()
            receipt={'path':str(batch_path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'readback_verified':True}
            validate_chunk(read_receipt(receipt,RUN_ROOT),recipe_sha,key,condition,rows)
        else:
            before=time.monotonic()
            records=[tables['by_text'][(row['user_text'],row['target'])] for row in rows]
            for row,record in zip(rows,records):
                prompt=native_prompt(tokenizer,row['user_text'])
                assert record['prompt_ids']==prompt['prompt_ids'] and record['user_prefix_length']==prompt['user_prefix_length']
            keys=image_keys(rows,condition)
            batch_features=None if keys is None else get_features(keys)
            predictions,seconds=generate_batch(model,connector,records,batch_features,tokenizer)
            generation_seconds+=seconds
            torch.cuda.synchronize();nll_started=time.perf_counter()
            from task2.development import example_losses
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                losses=example_losses(model,connector,records,batch_features,True)
            torch.cuda.synchronize();loss_seconds=time.perf_counter()-nll_started;nll_seconds+=loss_seconds
            output=[{**row,'condition':condition,'input_image_id':None if keys is None else keys[i][1],
                **prediction,'reference_NLL':loss['loss'],'reference_tokens':loss['tokens']}
                for i,(row,prediction,loss) in enumerate(zip(rows,predictions,losses))]
            chunk={'schema':'atml-pa3-task2-final-evaluation-batch-v1','recipe_sha256':recipe_sha,
                'batch_key':key,'condition':condition,'rows':output,'generation_seconds':seconds,
                'NLL_seconds':loss_seconds,'attempt':args.attempt}
            validate_chunk(chunk,recipe_sha,key,condition,rows)
            receipt=write_artifact(batch_path,chunk)
            batch_seconds.append(time.monotonic()-before)
        manifest['batches'][key]=receipt
        publish()
        print('EVALUATION SAVED:',manifest['verified_answers'],'/1280',key,flush=True)
    report_phase('verifying final model weights and saved evaluation results')
    for name,value in connector.state_dict().items():assert_equal(value,state['connector'][name],'connector/'+name)
    for name,p in adapters.items():assert_equal(p,state['lora'][name],name)
    assert not model.training and not connector.training and not any(p.requires_grad for p in model.parameters())
    assert all(p.grad is None for p in list(model.parameters())+list(connector.parameters()))
    assert pointer_path.read_bytes()==source_pointer_raw,'Training progress changed during evaluation.'
    with bounded_io(120,'Final evaluation manifest verification'):
        verify_manifest(manifest,plan,recipe_sha,RUN_ROOT)
    result={'block':'15','status':'passed' if manifest['complete'] else 'checkpointed',
        'package_sha256':args.package_sha,'source_reference':final_reference,
        'recipe':recipe_receipt,'plan':plan_receipt,'manifest':manifest_receipt,
        'verified_answers':manifest['verified_answers'],'total_answers':1280,
        'new_answers_this_run':manifest['verified_answers']-source_eval_count,
        'source_verified_answers':source_eval_count,'training_state_modified':False,'training_performed':False,
        'generation_seconds_this_run':generation_seconds,'NLL_seconds_this_run':nll_seconds,
        'elapsed_seconds':round(time.monotonic()-started,3),
        'qwen_source_staging_and_loading_seconds':qwen_stage_load_seconds,
        'feature_staging':feature_staging,'environment':environment,'parameter_counts':counts,
        'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),
        'peak_gpu_reserved_bytes':torch.cuda.max_memory_reserved(),
        'next_step':'CPU scoring and plots' if manifest['complete'] else 'Repeat same Block15 loader/ZIP to resume verified batches.'}
    with bounded_io(45,'Evaluation result save'):save_json(RUN_ROOT/'logs'/f'block_15_{args.attempt}.json',result)
    print('BLOCK 15 '+('PASSED' if manifest['complete'] else 'CHECKPOINTED')+'\n'+json.dumps(result,indent=2),flush=True)
except BaseException:
    diagnostic={'block':'15','status':'failed','phase':phase,'traceback':traceback.format_exc(),
        'training_performed':False,'package_sha256':args.package_sha,
        'elapsed_seconds':round(time.monotonic()-started,3)}
    print('BLOCK 15 DIAGNOSTIC\n'+json.dumps(diagnostic,indent=2),flush=True)
    try:
        with bounded_io(30,'Evaluation failure diagnostic save'):
            save_json(RUN_ROOT/'logs'/f'block_15_failed_{args.attempt}.json',diagnostic)
    except BaseException as error:print('Diagnostic could not be saved:',repr(error),flush=True)
    raise
