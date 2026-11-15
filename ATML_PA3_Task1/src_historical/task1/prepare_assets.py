"""Prepare original weights, FP32 protected bases, initial factors and CPU checks."""
import hashlib
import importlib.util
import json
import math
import shutil
import sys
import time
import uuid
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def copy_verified(source,target):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    fingerprint=digest(source)
    if target.is_symlink():raise RuntimeError(f'Unexpected symlink: {target}')
    if target.exists():
        if digest(target)!=fingerprint:raise RuntimeError(f'Existing content differs: {target}')
    else:
        pending=target.with_name(target.name+'.pending_'+uuid.uuid4().hex)
        shutil.copyfile(source,pending)
        assert digest(pending)==fingerprint
        if target.exists():raise RuntimeError('Concurrent writes detected')
        pending.rename(target)
    assert digest(target)==fingerprint
    return fingerprint


def write_json(path,value):
    payload=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=payload:raise RuntimeError(f'Frozen content differs: {path}')
    else:
        pending=path.with_name(path.name+'.pending_'+uuid.uuid4().hex);pending.write_bytes(payload)
        assert pending.read_bytes()==payload
        if path.exists():raise RuntimeError('Concurrent writes detected')
        pending.rename(path)
    assert path.read_bytes()==payload


def main(root,local,bundle):
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file,save_file
    from huggingface_hub import HfApi,hf_hub_download
    root,local,bundle=Path(root),Path(local),Path(bundle)
    started=time.monotonic()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    env=json.loads((root/'configs/ENVIRONMENT_READY.json').read_text())
    assert env['versions']['torch']==torch.__version__=='2.8.0+cu126'
    data=json.loads((root/'manifests/TRAINING_DATA_READY.json').read_text())
    assert data['selected_count']==40000 and data['full_saved_training_audit_passed']
    assert data['total_supervised_tokens']==7100056 and data['scanned_rows']==40002
    frozen=json.loads((root/'configs/SOURCE_REVISIONS.json').read_text())['model']
    for name in ('prepare_assets.py','training_core.py'):
        copy_verified(bundle/name,root/'provenance'/name)
    # Import from the verified package explicitly; -I intentionally excludes CWD.
    spec=importlib.util.spec_from_file_location('task1_training_core',bundle/'training_core.py')
    core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
    print('1/4 Checking token-normalized gradients, L2-SP, projection, merging and frozen parameters...',flush=True)
    checks=core.cpu_math_checks()
    write_json(root/'configs/CPU_MATH_CHECKS.json',checks)
    print('  All six CPU mathematical checks passed.',flush=True)
    print('2/4 Downloading and preserving the original frozen-revision model files on CPU...',flush=True)
    info=HfApi().model_info(frozen['repo'],revision=frozen['revision'],files_metadata=True)
    assert info.sha==frozen['revision']
    wanted=[f for f in info.siblings if f.rfilename.endswith('.safetensors')
            or f.rfilename in ('config.json','generation_config.json','model.safetensors.index.json')]
    assert any(f.rfilename.endswith('.safetensors') for f in wanted)
    model_dir=root/'sources/original_model';files={}
    local_files={}
    for entry in wanted:
        name=entry.rfilename
        if '/' in name or name.startswith('.'):raise RuntimeError('Unexpected original model layout')
        cached=Path(hf_hub_download(frozen['repo'],name,revision=frozen['revision'],cache_dir=local/'hf_cache'))
        fingerprint=copy_verified(cached,model_dir/name)
        lfs=entry.lfs
        expected=(lfs.get('sha256') if isinstance(lfs,dict) else getattr(lfs,'sha256',None)) if lfs else None
        if expected and fingerprint!=expected:raise RuntimeError(f'Official weight hash mismatch: {name}')
        files[name]={'sha256':fingerprint,'bytes':cached.stat().st_size,'official_lfs_hash_checked':bool(expected)}
        local_files[name]=cached
        print(f'  Saved and checked {name}: {cached.stat().st_size/2**20:.1f} MiB',flush=True)
    for path in (root/'sources/tokenizer').iterdir():
        if path.is_file():copy_verified(path,model_dir/path.name)
    write_json(root/'manifests/ORIGINAL_MODEL_FILES.json',{'model':frozen,'files':files})
    config=json.loads((model_dir/'config.json').read_text())
    assert config['num_hidden_layers']==28 and config['hidden_size']==1024
    keys={}
    original_parameters=0
    for name,cached in local_files.items():
        if not name.endswith('.safetensors'):continue
        with safe_open(cached,framework='pt',device='cpu') as f:
            for key in f.keys():
                if key in keys:raise RuntimeError('Duplicate tensor across model shards')
                keys[key]=cached
                if not (config.get('tie_word_embeddings') and key=='lm_head.weight'):
                    original_parameters+=math.prod(f.get_slice(key).get_shape())
    targets=[]
    for layer in range(28):
        for suffix in core.TARGETS:
            block='self_attn' if suffix in ('q_proj','k_proj','v_proj','o_proj') else 'mlp'
            key=f'model.layers.{layer}.{block}.{suffix}.weight'
            if key not in keys:raise RuntimeError(f'Missing prescribed target: {key}')
            targets.append(key)
    assert len(targets)==196
    print('3/4 Computing/verifying 196 exact FP32 CPU SVD bases; saving each matrix separately...',flush=True)
    entries=[]
    for number,key in enumerate(targets,1):
        with safe_open(keys[key],framework='pt',device='cpu') as f:
            original=f.get_tensor(key)
        original_dtype=str(original.dtype);W=original.float();del original
        assert W.ndim==2 and min(W.shape)>=128 and torch.isfinite(W).all()
        weight_sha=hashlib.sha256(W.numpy().tobytes()).hexdigest()
        path=root/'bases'/(key+'.safetensors');meta_path=path.with_suffix('.json')
        if path.exists() and meta_path.exists():
            meta=json.loads(meta_path.read_text())
            assert meta['model_revision']==frozen['revision'] and meta['weight_fp32_sha256']==weight_sha
            assert digest(path)==meta['basis_sha256']
            tensors=load_file(path,device='cpu');U,V,S=tensors['U'],tensors['V'],tensors['S']
            reused=True
        else:
            if path.exists() or meta_path.exists():raise RuntimeError(f'Incomplete basis pair; preserving it: {key}')
            t=time.monotonic();u,s,vh=torch.linalg.svd(W,full_matrices=False)
            elapsed=time.monotonic()-t
            U=u[:,:128].clone().contiguous();V=vh[:128].T.clone().contiguous();S=s[:128].clone()
            del u,s,vh
            meta={'key':key,'shape':list(W.shape),'model_revision':frozen['revision'],
                  'weight_fp32_sha256':weight_sha,'original_dtype':original_dtype,
                  'svd_seconds':elapsed,'precision':'float32','algorithm':'full reduced CPU SVD, leading 128'}
            reused=False
        assert U.shape==(W.shape[0],128) and V.shape==(W.shape[1],128) and S.shape==(128,)
        assert all(t.dtype==torch.float32 and torch.isfinite(t).all() for t in (U,V,S))
        eye=torch.eye(128)
        orth=max(float((U.T@U-eye).abs().max()),float((V.T@V-eye).abs().max()))
        residual=max(float(torch.linalg.norm(W@V-U*S)/torch.linalg.norm(U*S)),
                     float(torch.linalg.norm(W.T@U-V*S)/torch.linalg.norm(V*S)))
        assert orth<=1e-5 and residual<=1e-5,(key,orth,residual)
        assert bool((S[:-1]>=S[1:]).all()) and float(S[-1])>0
        half=W.half();assert torch.isfinite(half).all()
        half_relative=float(torch.linalg.norm(half.float()-W)/torch.linalg.norm(W));del half
        # Check projection of nonzero factors independently of a zero-B initialization.
        g=torch.Generator().manual_seed(6304+number)
        A=torch.randn((16,W.shape[1]),generator=g)*.01;B=torch.randn((W.shape[0],16),generator=g)*.01
        ae,be=core.projected_factors(A,B,U,V)
        leakage=max(float(torch.linalg.norm(ae@V)/torch.linalg.norm(ae)),
                    float(torch.linalg.norm(U.T@be)/torch.linalg.norm(be)))
        assert leakage<=1e-5,(key,leakage)
        if not reused:
            path.parent.mkdir(parents=True,exist_ok=True)
            pending=path.with_name(path.name+'.pending_'+uuid.uuid4().hex)
            save_file({'U':U,'V':V,'S':S},str(pending))
            loaded=load_file(pending,device='cpu')
            assert torch.equal(loaded['U'],U) and torch.equal(loaded['V'],V) and torch.equal(loaded['S'],S)
            assert not path.exists();pending.rename(path)
            meta.update({'basis_sha256':digest(path),'bytes':path.stat().st_size,
                'orthonormality_max_error':orth,'singular_pair_relative_residual':residual,
                'factor_projection_relative_leakage':leakage,'fp16_conversion_relative_error':half_relative})
            write_json(meta_path,meta)
        entries.append(meta)
        print(f'  {number:03}/196 {key}: '+('reused' if reused else f"SVD {meta['svd_seconds']:.1f}s")+f', orth={orth:.2e}, residual={residual:.2e}',flush=True)
        del W,U,V,S,A,B,ae,be
    print('4/4 Saving exact shared adapter initializations and final asset manifest...',flush=True)
    initial_counts={};initializer_files={}
    for rank in (4,16):
        generator=torch.Generator().manual_seed(6304);factors={};count=0
        for entry in entries:
            dout,din=entry['shape'];key=entry['key'][:-len('.weight')]
            A=torch.empty((rank,din));torch.nn.init.kaiming_uniform_(A,a=math.sqrt(5),generator=generator)
            B=torch.zeros((dout,rank));factors[key+'.A']=A;factors[key+'.B']=B;count+=A.numel()+B.numel()
        path=root/'initializers'/f'rank_{rank}.safetensors'
        if path.exists():
            saved=load_file(path,device='cpu')
            assert saved.keys()==factors.keys() and all(torch.equal(saved[k],factors[k]) for k in saved)
        else:
            path.parent.mkdir(parents=True,exist_ok=True)
            pending=path.with_name(path.name+'.pending_'+uuid.uuid4().hex);save_file(factors,str(pending))
            saved=load_file(pending,device='cpu');assert all(torch.equal(saved[k],factors[k]) for k in saved)
            assert not path.exists();pending.rename(path)
        initializer_files[str(rank)]={'sha256':digest(path),'bytes':path.stat().st_size,'path':str(path.relative_to(root))}
        initial_counts[str(rank)]=count
    assert initial_counts['16']==4*initial_counts['4']
    manifest={'model':frozen,'original_unique_parameters':original_parameters,'target_matrices':196,
        'protection_rank':128,'bases':entries,'initializer_files':initializer_files,
        'adapter_parameter_counts':initial_counts,'shared_rank16_initializer_for_lora_and_oplora':True,
        'basis_storage_bytes':sum(e['bytes'] for e in entries),'svd_preprocessing_seconds':sum(e['svd_seconds'] for e in entries),
        'training_core_sha256':digest(bundle/'training_core.py'),'cpu_math_checks':checks}
    write_json(root/'manifests/CPU_ASSETS_READY.json',manifest)
    receipt={'status':'TASK1_BLOCK05_OK','original_unique_parameters':original_parameters,
        'svd_matrices':196,'protection_rank':128,'adapter_parameter_counts':initial_counts,
        'basis_storage_MiB':round(manifest['basis_storage_bytes']/2**20,2),
        'svd_preprocessing_seconds':round(manifest['svd_preprocessing_seconds'],1),
        'elapsed_seconds':round(time.monotonic()-started,1),'cpu_math_checks':checks,
        'gpu_work_run':False,'official_training_started':False,
        'qualitative_yaml_present':(root/'sources/first_plot_questions.yaml').is_file()}
    write_json(root/'logs'/('block05_'+uuid.uuid4().hex+'.json'),receipt)
    print(json.dumps(receipt,indent=2),flush=True)
    print('TASK1_BLOCK05_OK — keep CPU selected and send the final summary.',flush=True)


if __name__=='__main__':main(*sys.argv[1:])
