"""Validate the approved trainer in pinned CPU software, freeze all five run contracts."""
import json
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from archive_io import digest
from evaluation_records import identity, publish
from baseline_runtime import publish_bytes

CORE_FILES = ('training_core.py','training_ops.py','official_training.py','training_checkpoints.py','training_loop.py')


def native_qwen_check(directory):
    import torch
    from transformers import Qwen3Config,Qwen3ForCausalLM
    from official_training import optimizer_for,effective_step,seed_training
    from training_checkpoints import save_checkpoint,restore_checkpoint
    from training_runtime_checks import rows
    config=Qwen3Config(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=2048,
        tie_word_embeddings=True,attention_dropout=0.,use_cache=False)
    seed_training();model=Qwen3ForCausalLM(config).train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False,'preserve_rng_state':True})
    named,opt=optimizer_for(model,2e-5);scaler=torch.amp.GradScaler('cpu')
    seed_training();history=[]
    for update in (1,2):
        coherence={}
        history.append(effective_step(model,named,opt,scaler,rows(),update,2e-5,0,'cpu',coherence))
    contract={'method':'full','scope':'RANDOM_TINY_QWEN_CPU_FIXTURE_ONLY'}
    path,marker=save_checkpoint(Path(directory)/'remote',directory,'full',contract,2,history,named,opt,scaler,False,'exit')
    recovered=Qwen3ForCausalLM(config).train()
    recovered.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False,'preserve_rng_state':True})
    named2,opt2=optimizer_for(recovered,2e-5);scaler2=torch.amp.GradScaler('cpu')
    restored=restore_checkpoint(path,contract,recovered,named2,opt2,scaler2)
    assert restored['successful_updates']==2 and recovered.lm_head.weight is recovered.model.embed_tokens.weight
    for (_,p),(_,q) in zip(named,named2):assert torch.equal(p,q)
    return {'actual_pinned_Qwen3_nonreentrant_checkpoint_backward_passed':True,
        'real_32_example_16_microbatch_token_normalized_updates_passed':True,
        'tiny_Qwen_tied_master_weights_and_Adam_states_reloaded':True,
        'original_model_used_for_fixture':False,'official_training_updates':0}


def main(root,job,bundle):
    import torch
    import importlib.metadata
    from safetensors import safe_open
    from transformers import AutoTokenizer
    from official_training import SourceOrderData,collate,METHODS
    from training_runtime_checks import checks
    from training_loop_checks import checks as loop_checks
    root,job,bundle=Path(root),Path(job),Path(bundle);start=time.monotonic()
    assert torch.__version__=='2.8.0+cu126' and importlib.metadata.version('transformers')=='4.57.1'
    assert not torch.cuda.is_available(),'CPU runtime required for this preparation'
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True,warn_only=False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    print('1/4 Verifying completed baseline, approved source and existing training assets...',flush=True)
    export_id='2062ae3a93f76b5f15651e8ea04c62c7555e626a78321d6949bd67b8d6a877ae'
    baseline_path=root/'configs'/('ORIGINAL_BASELINE_V2_AUDITED_'+export_id+'.json')
    baseline=json.loads(baseline_path.read_text())
    assert baseline['status']=='TASK1_BLOCK18_BASELINE_AUDIT_EXPORT_OK'
    assert baseline['all_17007_stock_scores_and_filtered_responses_replayed'] and baseline['official_training_updates']==0
    assert baseline['export_sha256']=='51802b3cb2d6ce20306b68cccb61fa8c452c07354073de8528204e5c085024a2'
    assert digest(root/baseline['export_path'])==baseline['export_sha256']
    approval=json.loads((root/'provenance/APPROVED_PLAN.json').read_text())
    assert approval['approval_id']=='T1-A001' and len(approval['choices'])==22
    assert digest(bundle/'APPROVED_PLAN.json')==digest(root/'provenance/APPROVED_PLAN.json')
    assets=json.loads((root/'manifests/CPU_ASSETS_READY.json').read_text())
    assert assets['original_unique_parameters']==596049920 and assets['target_matrices']==196 and assets['protection_rank']==128
    assert digest(bundle/'training_core.py')==assets['training_core_sha256']
    old=json.loads((root/'configs/GPU_PREFLIGHT_KIT_READY.json').read_text())
    assert digest(bundle/'training_ops.py')==old['files']['training_ops.py']
    model_contract=json.loads((root/'configs/CPU_MODEL_CONTRACT.json').read_text())
    assert model_contract['fp32_master_state_sha256']=='43f910d3e84e52f5214474dca7c74748e1701aec10440d8660ec1332a1ee1a3f'
    for item in assets['bases']:
        path=root/'bases'/(item['key']+'.safetensors');assert digest(path)==item['basis_sha256']
        with safe_open(path,framework='pt',device='cpu') as f:
            U,V=f.get_tensor('U'),f.get_tensor('V')
            assert U.shape==(item['shape'][0],128) and V.shape==(item['shape'][1],128)
            assert U.dtype==V.dtype==torch.float32
    for rank in (4,16):
        entry=assets['initializer_files'][str(rank)];path=root/entry['path'];assert digest(path)==entry['sha256']
        count=0
        with safe_open(path,framework='pt',device='cpu') as f:
            for item in assets['bases']:
                n=item['key'][:-len('.weight')];A=f.get_tensor(n+'.A');B=f.get_tensor(n+'.B')
                assert A.shape==(rank,item['shape'][1]) and B.shape==(item['shape'][0],rank)
                assert A.dtype==B.dtype==torch.float32 and not bool(B.ne(0).any())
                count+=A.numel()+B.numel()
            assert len(f.keys())==392 and count==assets['adapter_parameter_counts'][str(rank)]
    tokenizer_hashes=json.loads((root/'manifests/TOKENIZER_FILES.json').read_text())
    for name,expected in tokenizer_hashes.items():assert digest(root/'sources/original_model'/name)==expected
    tokenizer=AutoTokenizer.from_pretrained(root/'sources/original_model',local_files_only=True)
    data=SourceOrderData(root);model_config=json.loads((root/'sources/original_model/config.json').read_text())
    print('2/4 Auditing all 40,000 source-order examples and all 1,250 effective batches...',flush=True)
    batch_plan=[];seen=[];total=0
    for update in range(1,1251):
        batch=data.batch(update);denominator=0
        for row in batch:
            ids=row['input_ids'];assert ids[-1]==tokenizer.eos_token_id and 0<row['response_start']<len(ids)<=2048
            assert len(ids)-row['response_start']==row['n_supervised_tokens']
            assert all(type(t) is int and 0<=t<model_config['vocab_size'] for t in ids)
            denominator+=row['n_supervised_tokens'];seen.append(row['source_index'])
        if update in (1,32,63,312,625,937,1250):
            for i in range(0,32,2):
                ids,labels,mask=collate(batch[i:i+2],tokenizer.pad_token_id,'cpu')
                assert bool(labels[mask==0].eq(-100).all())
        batch_plan.append({'update':update,'source_ids':[r['source_index'] for r in batch],'supervised_tokens':denominator})
        total+=denominator
        if update%250==0:print(f'Source-order batches checked: {update}/1250',flush=True)
    assert seen==data.ids and len(seen)==40000 and total==7100056
    print('3/4 Testing actual training/overflow/checkpoint recovery code on CPU fixtures...',flush=True)
    runtime_checks=checks(job/'runtime_checks');loop_results=loop_checks(job/'loop_checks')
    qwen_checks=native_qwen_check(job)
    print('All training arithmetic, retry, checkpoint and bounded-loop CPU checks passed.',flush=True)
    print('4/4 Freezing the five approved run contracts and CPU readiness records...',flush=True)
    files={p.name:digest(p) for p in bundle.glob('*.py')}
    numerical={n:files[n] for n in CORE_FILES};numerical_id=identity(numerical)
    batch_id=identity(batch_plan);batch_rel='configs/TRAINING_EFFECTIVE_BATCHES_'+batch_id+'.json'
    publish(root/batch_rel,batch_plan)
    protocol={'schema_version':1,'approval_id':'T1-A001','seed':6304,'examples':40000,'successful_updates':1250,
        'source_order':True,'shuffle':False,'truncate':False,'microbatch':2,'accumulation':16,'effective_batch':32,
        'max_sequence_length':2048,'loss':'sum response/EOT CE divided by total effective-batch supervised tokens',
        'optimizer':'torch.optim.AdamW','betas':[.9,.999],'eps':1e-8,'weight_decay':.01,
        'decay_exclusion':'biases and parameters with ndim < 2','foreach':False,'fused':False,
        'warmup_updates':38,'schedule':'approved cosine to zero at update 1250, no floor','gradient_clip_norm':1.,
        'compute_dtype':'float16','trainable_and_optimizer_dtype':'float32','quantization':False,
        'deterministic_algorithms':True,'deterministic_warn_only':False,'TF32':False,'CUBLAS_WORKSPACE_CONFIG':':4096:8',
        'gradient_checkpointing':{'use_reentrant':False,'preserve_rng_state':True},'training_KV_cache':False,
        'GradScaler':{'initial_scale':65536.,'growth_factor':2.,'backoff_factor':.5,'growth_interval':2000},
        'overflow_max_retries_after_initial_attempt':8,'retry_same_batch_and_pre_attempt_RNG':True,
        'mandatory_updates':[312,625,937,1250],'rolling_interval':50,'retain_last_two_owned_rolling':True,
        'one_method_per_job':True,'stop_at_next_mandatory_checkpoint':True,'soft_minutes':90,'hard_minutes':110,
        'checkpoint_format':'lossless FP32 shards in one verified tar, COMPLETE marker last','maximum_CPU_shard_bytes':128*2**20,
        'learning_rate_function_source':numerical['training_core.py'],'numerical_implementation_identity':numerical_id,
        'numerical_source_files':numerical,'training_data_manifest_sha256':digest(root/'manifests/TRAINING_DATA_READY.json'),
        'batch_plan_identity':batch_id,'batch_plan_path':batch_rel,'original_model_manifest_sha256':digest(root/'manifests/ORIGINAL_MODEL_FILES.json'),
        'CPU_model_contract_sha256':digest(root/'configs/CPU_MODEL_CONTRACT.json'),'CPU_assets_manifest_sha256':digest(root/'manifests/CPU_ASSETS_READY.json'),
        'baseline_audit_path':str(baseline_path.relative_to(root)),'baseline_audit_sha256':digest(baseline_path),
        'method_order':['full','l2sp','lora4','lora16','oplora'],'FullFT_midpoint_evaluation_update':625}
    protocol_id=identity(protocol);protocol_rel='configs/TRAINING_PROTOCOL_'+protocol_id+'.json';publish(root/protocol_rel,protocol)
    method_entries={}
    for method in METHODS:
        adapter=method in ('lora4','lora16','oplora');rank=4 if method=='lora4' else 16 if adapter else None
        method_contract={'protocol_identity':protocol_id,'method':method,'peak_lr':2e-4 if adapter else 2e-5,
            'lambda':1e-3 if method=='l2sp' else 0.,'rank':rank,'alpha':8 if rank==4 else 32 if adapter else None,
            'adapter_dropout':.05 if adapter else None,'protection_rank':128 if method=='oplora' else None,
            'adapter_targets':list(__import__('training_core').TARGETS) if adapter else [],
            'initializer_sha256':assets['initializer_files'][str(rank)]['sha256'] if adapter else None,
            'numerical_implementation_identity':numerical_id}
        cid=identity(method_contract);rel='configs/TRAINING_METHOD_'+method+'_'+cid+'.json';publish(root/rel,method_contract)
        method_entries[method]={'path':rel,'identity':cid,'trainable_parameters':assets['adapter_parameter_counts'][str(rank)] if adapter else 596049920}
    kit_id=identity(files);dest=root/'execution'/('training_cpu_'+kit_id)
    for name in files:publish_bytes(dest/name,(bundle/name).read_bytes())
    publish(dest/'FILES.json',files)
    full_bytes=596049920*12
    adapter_mandatory_bytes=4*12*(2523136+2*10092544)
    future_peak_durable_bytes=int((11*full_bytes+adapter_mandatory_bytes)*1.1)+2*2**30
    result={'status':'TASK1_BLOCK19_TRAINING_CPU_READY','kit_identity':kit_id,'kit_path':str(dest.relative_to(root)),
        'files':files,'numerical_implementation_identity':numerical_id,'protocol_identity':protocol_id,'protocol_path':protocol_rel,
        'methods':method_entries,'all_40000_source_order_examples_checked':True,'effective_batches':1250,'total_supervised_tokens':total,
        'runtime_checks':runtime_checks,'bounded_loop_checks':loop_results,'native_tiny_Qwen_checks':qwen_checks,
        'full_checkpoint_tensor_payload_GiB':full_bytes/2**30,'conservative_remaining_checkpoint_storage_GiB':future_peak_durable_bytes/2**30,
        'Drive_free_space_not_measured':True,'Drive_free_space_must_be_recorded_before_GPU_launch':True,
        'checkpoint_shard_tensor_payload_limit_bytes':128*2**20,
        'conservative_checkpoint_transient_CPU_bytes':5*128*2**20,'GPU_launch_wrapper_preparation_pending':True,
        'GPU_work_run':False,'original_model_training_started':False,'official_training_updates':0,
        'elapsed_seconds':time.monotonic()-start,
        'next':'Keep GPU off and return the complete JSON. Next is bounded Full FT launch, after current units/rate and free Drive storage checks.'}
    rel='configs/TRAINING_CPU_READY_'+kit_id+'.json';publish(root/rel,result)
    publish(root/'logs'/('block19_training_cpu_'+uuid.uuid4().hex+'.json'),result)
    print(json.dumps(result,indent=2),flush=True)
    print('TASK1_BLOCK19_OK — keep GPU off and send the complete JSON.',flush=True)


if __name__=='__main__':main(*sys.argv[1:])
