"""FP16 batch-one engine. Imports torch only after independent worker acknowledgement."""
import hashlib,json,platform,random,subprocess,sys,time
from importlib import metadata
from pathlib import Path
import live_protocol
from live_store import require,termination,file_sha,SoftLimitReached

def environment(preflight):
    import torch
    from gpu_integrity import fingerprint,load_policy,cuda_dependency_inventory
    versions={n:metadata.version(n) for n in preflight['cpu_versions']}
    for n,wanted in preflight['cpu_versions'].items():
        require(versions[n].split('+')[0]==wanted.split('+')[0] if n=='torch' else versions[n]==wanted,'GPU image software differs: '+n+'='+versions[n]+'; stop and review on CPU')
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,'Exactly one CUDA GPU is required')
    props=torch.cuda.get_device_properties(0);cap=list(torch.cuda.get_device_capability(0))
    require('T4' in props.name and cap==[7,5],'Approved T4 unavailable; no silent GPU substitution')
    require('sm_75' in torch.cuda.get_arch_list(),'Installed PyTorch build does not include T4 support')
    require(props.total_memory>=14*2**30,'T4 memory capacity below requirement')
    free,total=torch.cuda.mem_get_info();needed=6171877376+32768*36*2*2*128*2+2*2**30
    require(free>=needed,'Insufficient free memory for approved FP16 setup; no quantization fallback')
    # Match CPU-checked core code; CUDA-specific torch build is frozen separately.
    policy=load_policy(Path(__file__).with_name('TORCH_INTEGRITY_POLICY_A002.json'))
    fps=fingerprint(list(preflight['cpu_source_fingerprints']),policy)
    for name,cpu in preflight['cpu_source_fingerprints'].items():
        if name=='torch':continue
        require(fps[name]==cpu,'CPU-checked installed source differs: '+name)
    probe=subprocess.run(['nvidia-smi','--query-gpu=uuid,driver_version','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10,check=True)
    fields=[v.strip() for v in probe.stdout.strip().split(',')];require(len(fields)==2,'Cannot identify physical GPU/driver')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    record=metadata.distribution('torch').read_text('RECORD');require(record is not None,'Torch wheel inventory absent')
    signature={'python':sys.version,'platform':platform.platform(),'versions':versions,'source_fingerprints':fps,'torch_wheel_RECORD_sha256':hashlib.sha256(record.encode()).hexdigest(),'torch_C_extension_sha256':file_sha(torch._C.__file__),'CUDA_version':torch.version.cuda,'cuDNN_version':torch.backends.cudnn.version(),'driver_version':fields[1],'GPU_family':'T4','compute_capability':cap,'attention':'sdpa','weight_dtype':'float16','batch_size':1,'quantization':False,'TF32':False,'compile':False,'cache':'dynamic','binary_hash_scope':'Torch wheel RECORD and loaded _C facade hashed; large CUDA libraries not separately rehashed','approved_integrity_policy':'A002','CUDA_dependencies':cuda_dependency_inventory()}
    physical={'GPU_name':props.name,'GPU_uuid':fields[0],'total_memory_bytes':props.total_memory,'free_memory_before_model_bytes':free,'required_free_bytes':needed,'CUDA_arch_list':torch.cuda.get_arch_list()}
    return signature,physical

class Engine:
    def __init__(self,folder,tokenizer,protocol,started):
        import torch
        from transformers import AutoModelForCausalLM
        self.torch=torch;self.tokenizer=tokenizer;self.protocol=protocol;self.started=started
        self.model=AutoModelForCausalLM.from_pretrained(str(folder),local_files_only=True,trust_remote_code=False,use_safetensors=True,dtype=torch.float16,device_map={'':'cuda:0'},attn_implementation='sdpa').eval()
        self.model.requires_grad_(False)
        require(not getattr(self.model,'is_quantized',False),'Unexpected quantized model')
        require(all(p.dtype==torch.float16 and p.device.type=='cuda' and p.device.index==0 for p in self.model.parameters()),'Model device/dtype differs')
        self.handle=live_protocol.install_finite_guard(self.model,torch)
    def infer(self,messages,cap,seed,phase_started):
        import numpy as np
        from transformers import GenerationConfig
        from watchdog import can_start_call
        t=self.torch;random.seed(seed);np.random.seed(seed);t.manual_seed(seed);t.cuda.manual_seed_all(seed)
        t.cuda.synchronize()
        if not can_start_call(self.started,time.monotonic()):raise SoftLimitReached('Soft limit before inference')
        t.cuda.reset_peak_memory_stats();began=time.perf_counter()
        encoded=live_protocol.encode_messages(self.tokenizer,messages);live_protocol.check_context(encoded['input_ids'],cap)
        ids=t.tensor([encoded['input_ids']],dtype=t.long,device='cuda:0');mask=t.ones_like(ids)
        if not can_start_call(self.started,time.monotonic()):raise SoftLimitReached('Soft limit during prompt preparation')
        options=live_protocol.generation_options(self.protocol,cap)
        with t.inference_mode():output=self.model.generate(input_ids=ids,attention_mask=mask,generation_config=GenerationConfig(**options),logits_to_keep=1)
        generated=output[0,ids.shape[1]:].detach().cpu().tolist()
        raw=self.tokenizer.decode(generated,skip_special_tokens=False,clean_up_tokenization_spaces=False)
        text=self.tokenizer.decode(generated,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        t.cuda.synchronize();elapsed=time.perf_counter()-began;wall=time.monotonic()-phase_started
        require(wall<600 and elapsed<600,'Call exceeded ten minutes; output is incomplete execution, not scored response')
        return {**encoded,'input_tokens':len(encoded['input_ids']),'output_ids':generated,'raw_decoded_text':raw,'generated_text':text,'termination':termination(generated,cap),'extraction':live_protocol.extract_last_box(text),'inference_seconds':elapsed,'call_wall_seconds':wall,'peak_allocated_bytes':int(t.cuda.max_memory_allocated()),'peak_reserved_bytes':int(t.cuda.max_memory_reserved()),'started_before_soft_limit':True,'completed_before_call_deadline':True,'generation_options':options}
