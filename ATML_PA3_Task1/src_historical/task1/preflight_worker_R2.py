"""Bounded L4-only scratch capacity/AMP check. Never trains official checkpoints."""
import gc
import importlib.util
import json
import os
import sys
import time
import traceback
import uuid
from pathlib import Path


def atomic(path,value):
    path=Path(path);pending=path.with_name(path.name+'.pending_'+uuid.uuid4().hex)
    pending.write_text(json.dumps(value,indent=2)+'\n');pending.replace(path)


def progress(state_path,phase):
    state=json.loads(Path(state_path).read_text())
    if time.time()+10>=state['deadline_wall']:raise TimeoutError('Job deadline reached; preflight is incomplete')
    state.update({'phase':phase,'progress_wall':time.time(),'status':'running','progress_timeout':900})
    atomic(state_path,state)


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj


def run_variant(kind,inputs,kit,state_path):
    import torch
    from transformers import AutoModelForCausalLM
    from safetensors.torch import load_file
    core=module(kit/'training_core.py','core_'+kind)
    ops=module(kit/'training_ops.py','ops_'+kind)
    assets=json.loads((inputs/'manifests/CPU_ASSETS_READY.json').read_text())
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    progress(state_path,kind+': load original weights')
    started=time.monotonic()
    dtype=torch.float16 if kind=='oplora' else torch.float32
    model=AutoModelForCausalLM.from_pretrained(inputs/'sources/original_model',local_files_only=True,
        dtype=dtype,attn_implementation='sdpa',low_cpu_mem_usage=True)
    model.config.use_cache=False
    if kind=='oplora':
        model.requires_grad_(False)
        initial=load_file(inputs/assets['initializer_files']['16']['path'],device='cpu')
        for item in assets['bases']:
            name=item['key'][:-len('.weight')];base=model.get_submodule(name)
            basis=load_file(inputs/'bases'/(item['key']+'.safetensors'),device='cpu')
            adapter=core.AdapterLinear(base,initial[name+'.A'],initial[name+'.B'],basis['U'],basis['V'],alpha=32,dropout=.05)
            parent,leaf=name.rsplit('.',1);setattr(model.get_submodule(parent),leaf,adapter)
        del initial,basis,adapter,base
    model.to('cuda');model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    named=[(name,p) for name,p in model.named_parameters() if p.requires_grad]
    assert all(p.dtype==torch.float32 for _,p in named)
    count=sum(p.numel() for _,p in named)
    assert count==(10092544 if kind=='oplora' else 596049920)
    if kind=='oplora':
        assert all(p.dtype==torch.float16 for p in model.parameters() if not p.requires_grad)
        assert all(m.U.dtype==m.V.dtype==torch.float32 for m in model.modules() if isinstance(m,core.AdapterLinear))
    reference={name:p.detach().clone() for name,p in model.named_parameters()} if kind=='l2sp' else None
    groups=[{'params':[p for name,p in named if p.ndim>=2 and not name.endswith('bias')],'weight_decay':.01},
            {'params':[p for name,p in named if p.ndim<2 or name.endswith('bias')],'weight_decay':0.}]
    optimizer=torch.optim.AdamW(groups,lr=2e-4 if kind=='oplora' else 2e-5,
        betas=(.9,.999),eps=1e-8,foreach=False,fused=False)
    scaler=torch.amp.GradScaler('cuda',init_scale=65536.,growth_factor=2.,backoff_factor=.5,growth_interval=2000)
    torch.manual_seed(6304);torch.cuda.manual_seed_all(6304)
    ids=torch.randint(0,151000,(2,2048),device='cuda');ids[:,-1]=151645
    labels=ids.clone();labels[:,:64]=-100;mask=torch.ones_like(ids)
    denominator=int(labels[:,1:].ne(-100).sum())*16
    loading_seconds=time.monotonic()-started;steps=[];skips=[]
    for update in (1,2):
        seed_cpu=torch.get_rng_state();seed_gpu=torch.cuda.get_rng_state()
        for retry in range(9):
            torch.set_rng_state(seed_cpu);torch.cuda.set_rng_state(seed_gpu)
            optimizer.zero_grad(set_to_none=True)
            peak=2e-4 if kind=='oplora' else 2e-5
            for group in optimizer.param_groups:group['lr']=core.learning_rate(update,peak)
            torch.cuda.synchronize();t=time.monotonic();sft=0.;omega_value=0.
            for micro in range(16):
                progress(state_path,f'{kind}: scratch update {update}, microbatch {micro+1}/16, retry {retry}')
                with torch.autocast('cuda',dtype=torch.float16):
                    loss,n=ops.chunked_response_sum(model,ids,labels,mask)
                    assert n*16==denominator
                    objective=loss/denominator
                sft+=float(objective.detach())
                if kind=='l2sp' and micro==15:
                    omega=core.l2_sp(model,reference)
                    omega_value=float(omega.detach());objective=objective+1e-3*omega
                if not torch.isfinite(objective):raise FloatingPointError('Nonfinite forward objective')
                scaler.scale(objective).backward()
                del loss,objective
                if kind=='l2sp' and micro==15:del omega
            scaler.unscale_(optimizer)
            finite=bool(torch.stack([torch.isfinite(p.grad).all() for _,p in named if p.grad is not None]).all())
            if not finite:
                old=scaler.get_scale();scaler.step(optimizer);scaler.update()
                assert scaler.get_scale()<old
                skips.append({'update':update,'retry':retry,'old_scale':old,'new_scale':scaler.get_scale()})
                if retry==8:raise FloatingPointError('Gradient overflow persisted after eight retries')
                continue
            norm=torch.nn.utils.clip_grad_norm_([p for _,p in named],1.,error_if_nonfinite=True)
            scaler.step(optimizer);scaler.update();torch.cuda.synchronize()
            assert all(value.dtype==torch.float32 for state in optimizer.state.values()
                       for value in state.values() if isinstance(value,torch.Tensor))
            steps.append({'update':update,'seconds':time.monotonic()-t,'sft_loss':sft,
                'omega':omega_value,'gradient_norm_before_clip':float(norm),'scale':scaler.get_scale()})
            break
    allocated=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved()
    total=torch.cuda.get_device_properties(0).total_memory
    result={'variant':kind,'scratch_successful_updates':len(steps),'microbatch':2,'accumulation':16,
        'sequence_length':2048,'response_head_chunk_tokens':128,'trainable_parameters':count,
        'loading_seconds':loading_seconds,'steps':steps,'overflow_attempts':skips,
        'peak_allocated_bytes':allocated,'peak_reserved_bytes':reserved,'total_gpu_bytes':total,
        'headroom_fraction_using_peak_reserved':1-reserved/total,'optimizer_states_fp32':True,
        'deterministic_algorithms_enabled':torch.are_deterministic_algorithms_enabled(),
        'deterministic_warn_only':torch.is_deterministic_algorithms_warn_only_enabled()}
    if reserved>.9*total:raise MemoryError('Less than 10% GPU memory headroom under scratch workload: '+json.dumps(result))
    return result


def main(config_path):
    import torch
    config=json.loads(Path(config_path).read_text());root=Path(config['root']);state_path=Path(config['state'])
    kit=Path(config['kit']);inputs=Path(config['inputs']);result_path=Path(config['result'])
    started=time.monotonic();results=[];kind=None
    torch.use_deterministic_algorithms(True,warn_only=False)
    try:
        assert torch.cuda.is_available() and 'L4' in torch.cuda.get_device_name(0)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        for kind in ('full','l2sp','oplora'):
            print('Scratch check:',kind,flush=True)
            results.append(run_variant(kind,inputs,kit,state_path))
            print(json.dumps(results[-1],indent=2),flush=True)
            gc.collect();torch.cuda.empty_cache()
        result={'status':'TASK1_L4_PREFLIGHT_OK','results':results,'elapsed_seconds':time.monotonic()-started,
            'gpu_name':torch.cuda.get_device_name(0),'official_training_updates':0,'scratch_weights_discarded':True}
    except BaseException as error:
        result={'status':'TASK1_L4_PREFLIGHT_FAILED','completed_variants':results,
            'error_type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc(),
            'elapsed_seconds':time.monotonic()-started,'official_training_updates':0,'failed_variant':kind}
        try:
            result['gpu_memory_on_failure']={'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'total_gpu_bytes':torch.cuda.get_device_properties(0).total_memory}
        except Exception:
            result['gpu_memory_on_failure']='unavailable'
        atomic(result_path,result);print(json.dumps(result,indent=2),flush=True)
        state=json.loads(state_path.read_text());state.update({'status':'failed','progress_wall':time.time()});atomic(state_path,state)
        raise
    atomic(result_path,result);print(json.dumps(result,indent=2),flush=True)
    state=json.loads(state_path.read_text());state.update({'status':'worker_complete','progress_wall':time.time(),'worker_completed_wall':time.time()});atomic(state_path,state)


if __name__=='__main__':main(sys.argv[1])
