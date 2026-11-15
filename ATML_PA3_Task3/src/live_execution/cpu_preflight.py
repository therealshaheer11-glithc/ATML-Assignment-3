"""Isolated CPU software/tokenizer preflight. Never loads supplied model weights."""
import argparse,base64,hashlib,inspect,json,os,sys,tempfile
from importlib import metadata
from pathlib import Path

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(c,message):
    if not c:raise ValueError(message)
def fingerprint(names):
    result={}
    for name in names:
        dist=metadata.distribution(name);entries=[];binaries=[];version_file=None
        for entry in dist.files or []:
            label=str(entry)
            is_source=label.endswith('.py');is_binary=name in ['tokenizers','safetensors'] and (label.endswith('.so') or label.endswith('.pyd'))
            if not (is_source or is_binary):continue
            raw=dist.locate_file(entry).read_bytes();h=hashlib.sha256(raw).hexdigest()
            require(entry.hash and entry.hash.mode=='sha256','Installed asset lacks expected RECORD hash: '+label)
            require(base64.urlsafe_b64encode(bytes.fromhex(h)).decode().rstrip('=')==entry.hash.value,'Installed package modified: '+label)
            if name=='torch' and label=='torch/version.py':version_file=h;continue
            (entries if is_source else binaries).append(label+':'+h)
        require(entries,'No source inventory: '+name)
        result[name]={'version':dist.version,'python_source_files':len(entries),'source_sha256':hashlib.sha256('\n'.join(sorted(entries)).encode()).hexdigest(),'compiled_assets':binaries,'torch_build_version_file_sha256':version_file,'torch_variant_rule':'Require public2.14.1; first actual CUDA session freezes CUDA build separately' if name=='torch' else None}
    return result

def execute(root,bundle):
    # Set before framework imports; refuse all model/code downloads in this worker.
    for k,v in {'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}.items():os.environ[k]=v
    sys.path.insert(0,str(bundle))
    import live_protocol
    from supervision_checks import run_checks
    from google.colab import runtime
    observed=json.loads((bundle/'ASSET_REFERENCE.json').read_text());expected=observed['observed_versions_not_yet_frozen']
    versions={n:metadata.version(n) for n in expected}
    require(versions==expected,'CPU software changed since Block10; stop and review versions')
    import torch
    import transformers
    from transformers import AutoConfig,AutoTokenizer,AutoModelForCausalLM,GenerationConfig,Qwen2Config,Qwen2ForCausalLM
    torch.set_num_threads(1)
    require(not torch.cuda.is_available(),'CPU preflight unexpectedly has CUDA access')
    model_folder=root/'live'/('model_'+observed['revision']);live=json.loads((root/'live/task3_live_inputs_v1.json').read_text());protocol=json.loads((root/'configs/MANUAL_REQUIREMENTS.json').read_text())['live'];expected_tokens=json.loads((bundle/'EXPECTED_TOKENIZATION.json').read_text())
    tokenizer=AutoTokenizer.from_pretrained(str(model_folder),local_files_only=True,trust_remote_code=False,use_fast=True)
    require(tokenizer.is_fast and tokenizer.eos_token_id==151645 and tokenizer.pad_token_id==151643,'Tokenizer configuration differs')
    require(tokenizer.convert_tokens_to_ids('<|im_start|>')==151644 and tokenizer.convert_tokens_to_ids('<|im_end|>')==151645,'Chat control IDs differ')
    config=AutoConfig.from_pretrained(str(model_folder),local_files_only=True,trust_remote_code=False)
    require(config.model_type=='qwen2' and config.max_position_embeddings==32768 and config.num_hidden_layers==36,'Model config differs')
    token_rows=[]
    require(len(live['questions'])==len(expected_tokens['questions'])==50,'Wrong prompt count')
    for record,reference in zip(live['questions'],expected_tokens['questions']):
        require(record['sample_id']==reference['sample_id'],'Question order changed')
        results={}
        for kind,messages in [('initial',record['initial_messages']),('revision_placeholder',live_protocol.revision_messages(record['initial_messages'],'CPU preflight placeholder. \\boxed{0}',protocol))]:
            encoded=live_protocol.encode_messages(tokenizer,messages);text=encoded['rendered_prompt'];ids=encoded['input_ids']
            require(ids==reference[kind]['input_ids'] and hashlib.sha256(text.encode()).hexdigest()==reference[kind]['text_sha256'],'Independent tokenizer fixture differs: '+record['sample_id']+'/'+kind)
            if kind=='initial':live_protocol.check_context(ids,3072,config.max_position_embeddings)
            results[kind]={'tokens':len(ids),'input_ids_sha256':hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),'text_sha256':hashlib.sha256(text.encode()).hexdigest()}
        token_rows.append({'sample_id':record['sample_id'],**results})
    print('CPU prompt checks passed:50 initial and50 placeholder revision prompts; no actual responses generated.',flush=True)
    # Test the actual framework classes with a tiny synthetic model only.
    torch.manual_seed(6304)
    tiny=Qwen2Config(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=128,pad_token_id=0,bos_token_id=1,eos_token_id=2,tie_word_embeddings=True)
    tiny._attn_implementation='sdpa';model=Qwen2ForCausalLM(tiny).to(device='cpu',dtype=torch.float16).eval()
    with tempfile.TemporaryDirectory(prefix='task3_tiny_checkpoint_') as checkpoint:
        model.save_pretrained(checkpoint,push_to_hub=False)
        require(sum(p.stat().st_size for p in Path(checkpoint).rglob('*') if p.is_file())<1024*1024,'Synthetic checkpoint unexpectedly large')
        del model
        model=AutoModelForCausalLM.from_pretrained(checkpoint,device_map={'':'cpu'},dtype=torch.float16,local_files_only=True,trust_remote_code=False,use_safetensors=True,attn_implementation='sdpa').eval()
    require(all(p.device.type=='cpu' and p.dtype==torch.float16 for p in model.parameters()),'Tiny checkpoint load/device/dtype mismatch')
    handle=live_protocol.install_finite_guard(model,torch)
    source=torch.tensor([[1,5,6]],dtype=torch.long);mask=torch.ones_like(source);opts=live_protocol.generation_options(protocol,4);opts.update(pad_token_id=0,bos_token_id=1,eos_token_id=None);toy_config=GenerationConfig(**opts)
    with torch.inference_mode():
        output=model(input_ids=source,attention_mask=mask,use_cache=True,logits_to_keep=1)
        require(output.logits.shape==(1,1,32) and torch.isfinite(output.logits).all(),'Tiny FP16 forward failed')
        torch.manual_seed(12345);first=model.generate(input_ids=source,attention_mask=mask,generation_config=toy_config)
        torch.manual_seed(12345);second=model.generate(input_ids=source,attention_mask=mask,generation_config=toy_config)
        require(first.shape==(1,7) and torch.equal(first,second),'Tiny capped seeded sampling failed')
        model.lm_head.weight[0,0]=float('nan')
        caught=False
        try:model(input_ids=source,attention_mask=mask,logits_to_keep=1)
        except FloatingPointError:caught=True
        require(caught,'Nonfinite-logit guard failed')
    tiny_info={'parameter_values':sum(p.numel() for p in model.parameters()),'dtype':'float16','device':'cpu','forward_last_logits_shape':[1,1,32],'capped_sampling_tokens':4,'seeded_repeat_equal':True,'nonfinite_guard_raised':True,'actual_Qwen_3B_loaded':False,'tiny_safetensors_save_and_AutoModel_reload_passed':True}
    handle.remove();del model,output,first,second
    print('Tiny CPU FP16/SDPA/sampling/nonfinite checks passed.',flush=True)
    supervision=run_checks();print('Independent CPU watchdog checks passed.',flush=True)
    release_source=inspect.getsource(runtime.unassign)
    # Inspect without invoking Colab's real unassignment service.
    require('TBE_RUNTIME_ADDR' in release_source and '/unassign' in release_source and 'POST' in release_source,'Installed Colab release implementation changed; review before GPU use')
    require(bool(os.environ.get('TBE_RUNTIME_ADDR')),'Native Colab runtime management address is absent')
    core=list(expected)+['numpy','jinja2','packaging','regex','requests']
    fps=fingerprint(core)
    full_environment=sorted({(d.metadata['Name'],d.version) for d in metadata.distributions() if d.metadata['Name']})
    generations={str(cap):GenerationConfig(**live_protocol.generation_options(protocol,cap)).to_dict() for cap in [1024,2048,3072]}
    return {'status':'CPU LIVE PREFLIGHT CHECKS PASSED','python':sys.version,'versions':versions,'source_fingerprints':fps,'full_environment':[{'name':n,'version':v} for n,v in full_environment],'tokenizer':{'class':type(tokenizer).__name__,'fast':tokenizer.is_fast,'questions':50,'initial_tokens_min':min(r['initial']['tokens'] for r in token_rows),'initial_tokens_max':max(r['initial']['tokens'] for r in token_rows),'native_render_and_independent_ids_matched':True,'prompt_checks':token_rows,'actual_revision_context_must_be_checked_at_execution':True},'synthetic_tiny_model':tiny_info,'watchdog':supervision,'runtime_release_source':release_source,'runtime_release_source_sha256':hashlib.sha256(release_source.encode()).hexdigest(),'runtime_release_endpoint_present':True,'generation_configurations':generations,'gpu_work_executed':False,'actual_model_loaded':False,'actual_model_generations':0,'native_runtime_release_executed':False,'physical_runtime_deallocation_verified':False}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--bundle',required=True);p.add_argument('--output',required=True);a=p.parse_args();result=execute(Path(a.root),Path(a.bundle));Path(a.output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(result['status'],flush=True)
