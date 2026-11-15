"""CPU model/dtype contract check and portable verified runtime archive."""
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tarfile
import time
import uuid
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def write_once(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.is_symlink():raise RuntimeError(f'Unexpected linked output: {path}')
    if path.exists():
        if path.read_bytes()!=data:raise RuntimeError(f'Frozen record differs: {path}')
        return
    tmp=path.with_name(path.name+'.pending_'+uuid.uuid4().hex);tmp.write_bytes(data)
    assert tmp.read_bytes()==data
    if path.exists():raise RuntimeError('Concurrent writes detected')
    tmp.rename(path);assert path.read_bytes()==data


def dump(path,value):
    write_once(path,(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode())


def software_files(local):
    """Only private software; exclude caches/bytecode and preserve internal links."""
    for name in ('OWNER.json','bootstrap_uv','managed_pythons','venv312'):
        start=local/name
        if not start.exists():raise RuntimeError(f'Missing portable runtime component: {name}')
        paths=[start]+sorted(start.rglob('*')) if start.is_dir() else [start]
        for p in paths:
            relative=p.relative_to(local)
            if '__pycache__' in relative.parts or p.suffix in ('.pyc','.pyo'):continue
            if p.is_symlink():
                resolved=p.resolve()
                if not resolved.is_relative_to(local.resolve()):
                    raise RuntimeError(f'Software link escapes private runtime: {relative}')
                yield p,{'path':str(relative),'kind':'symlink','target':os.readlink(p)}
            elif p.is_dir():yield p,{'path':str(relative),'kind':'directory'}
            elif p.is_file():yield p,{'path':str(relative),'kind':'file','bytes':p.stat().st_size,'sha256':digest(p)}
            else:raise RuntimeError(f'Unexpected software file type: {relative}')


def main(root,local,bundle):
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM,AutoTokenizer
    root,local,bundle=Path(root),Path(local),Path(bundle)
    started=time.monotonic();torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ready=json.loads((root/'configs/ENVIRONMENT_READY.json').read_text())
    assert ready['versions']['torch']==torch.__version__=='2.8.0+cu126'
    assert ready['python_version']=='3.12.12'
    assets=json.loads((root/'manifests/CPU_ASSETS_READY.json').read_text())
    assert assets['target_matrices']==196 and assets['protection_rank']==128
    core_path=bundle/'training_core.py'
    assert digest(core_path)==assets['training_core_sha256']
    for name in ('prepare_execution.py','training_core.py'):
        write_once(root/'provenance'/name,(bundle/name).read_bytes())
    spec=importlib.util.spec_from_file_location('task1_core',core_path)
    core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
    print('1/4 Loading the actual original model on CPU and checking its FP32 starting state...',flush=True)
    model_dir=root/'sources/original_model'
    source=json.loads((root/'manifests/ORIGINAL_MODEL_FILES.json').read_text())
    for name,entry in source['files'].items():assert digest(model_dir/name)==entry['sha256'],name
    model=AutoModelForCausalLM.from_pretrained(model_dir,local_files_only=True,
        torch_dtype=torch.float32,attn_implementation='sdpa',low_cpu_mem_usage=True)
    model.config.use_cache=False
    assert sum(p.numel() for p in model.parameters())==assets['original_unique_parameters']==596049920
    assert all(p.dtype==torch.float32 and p.device.type=='cpu' for p in model.parameters())
    tied=model.get_input_embeddings().weight.data_ptr()==model.get_output_embeddings().weight.data_ptr()
    assert tied,'Loaded model must preserve its configured tied embeddings'
    master_hash=hashlib.sha256()
    for name,p in model.named_parameters():
        master_hash.update(name.encode()+b'\0')
        master_hash.update(memoryview(p.detach().contiguous().numpy()).cast('B'))
    master_sha=master_hash.hexdigest()
    print('2/4 Checking all target shapes, saved initial factors, and actual OPLoRA parameter dtypes...',flush=True)
    initializers={}
    for rank in (4,16):
        entry=assets['initializer_files'][str(rank)];path=root/entry['path']
        assert digest(path)==entry['sha256']
        tensors=load_file(path,device='cpu');count=0
        for e in assets['bases']:
            name=e['key'][:-len('.weight')];module=model.get_submodule(name)
            A=tensors[name+'.A'];B=tensors[name+'.B']
            assert list(module.weight.shape)==e['shape']
            assert A.shape==(rank,e['shape'][1]) and B.shape==(e['shape'][0],rank)
            assert A.dtype==B.dtype==torch.float32 and torch.count_nonzero(B)==0
            assert torch.isfinite(A).all();count+=A.numel()+B.numel()
        assert count==assets['adapter_parameter_counts'][str(rank)]
        initializers[rank]=tensors
    # Convert only the backbone BEFORE adding FP32 adapters and protected bases.
    model.requires_grad_(False);model.half()
    for e in assets['bases']:
        name=e['key'][:-len('.weight')];base=model.get_submodule(name)
        path=root/'bases'/(e['key']+'.safetensors');assert digest(path)==e['basis_sha256']
        basis=load_file(path,device='cpu');initial=initializers[16]
        layer=core.AdapterLinear(base,initial[name+'.A'],initial[name+'.B'],
            basis['U'],basis['V'],alpha=32,dropout=.05)
        assert torch.equal(layer.A,initial[name+'.A']) and torch.equal(layer.B,initial[name+'.B'])
        parent,leaf=name.rsplit('.',1);setattr(model.get_submodule(parent),leaf,layer)
    trainable=[(name,p) for name,p in model.named_parameters() if p.requires_grad]
    assert len(trainable)==392 and sum(p.numel() for _,p in trainable)==10092544
    assert all(p.dtype==torch.float32 and (name.endswith('.A') or name.endswith('.B')) for name,p in trainable)
    assert all(p.dtype==torch.float16 for name,p in model.named_parameters() if not p.requires_grad)
    for module in model.modules():
        if isinstance(module,core.AdapterLinear):assert module.U.dtype==module.V.dtype==torch.float32
    tokenizer=AutoTokenizer.from_pretrained(model_dir,local_files_only=True)
    prefix=tokenizer.apply_chat_template([{'role':'user','content':'CPU setup check.'}],
        tokenize=False,add_generation_prompt=True,enable_thinking=False)
    assert '<think>\n\n</think>\n\n' in prefix
    contract={'original_unique_parameters':596049920,'fp32_master_state_sha256':master_sha,
        'original_embeddings_tied':tied,'target_shapes_checked':196,'rank4_initializer_checked':True,
        'rank16_initializer_checked':True,'actual_oplora_trainable_parameters':10092544,
        'actual_oplora_trainable_tensors':392,'frozen_backbone_fp16':True,
        'adapters_and_protected_bases_fp32':True,'native_non_thinking_prefix_verified':True,
        'gpu_or_forward_capacity_tested':False}
    dump(root/'configs/CPU_MODEL_CONTRACT.json',contract)
    del model,initializers,trainable
    import gc;gc.collect()
    print('3/4 Hashing the private software runtime and creating a portable archive...',flush=True)
    inventory=list(software_files(local))
    inventory_data={'schema_version':1,'base_directory':'/content/atml_pa3_task1',
        'python_version':ready['python_version'],'lock_sha256':ready['lock_sha256'],
        'files':[entry for _,entry in inventory]}
    payload=(json.dumps(inventory_data,sort_keys=True,separators=(',',':'))+'\n').encode()
    inventory_sha=hashlib.sha256(payload).hexdigest()
    target=root/'runtime'/('python312_'+inventory_sha[:16]+'.tar.gz')
    index_path=target.with_suffix('.json')
    total_bytes=sum(e.get('bytes',0) for _,e in inventory)
    if shutil.disk_usage(local).free < total_bytes+2*2**30:raise RuntimeError('Insufficient local space for runtime archive')
    if target.exists() and index_path.exists():
        archive_record=json.loads(index_path.read_text())
        assert archive_record['inventory_sha256']==inventory_sha
        assert digest(target)==archive_record['archive_sha256']
        print('  Reusing the checked portable runtime archive.',flush=True)
    else:
        if target.exists() or index_path.exists():raise RuntimeError('Incomplete runtime archive pair; preserving it for review')
        temporary=local/('portable_runtime_'+uuid.uuid4().hex+'.tar.gz')
        with tarfile.open(temporary,'w:gz',compresslevel=1,dereference=False) as archive:
            for i,(path,entry) in enumerate(inventory,1):
                if entry['kind']=='file':
                    # Explicit regular-file entries avoid hidden hardlink dependencies.
                    metadata=tarfile.TarInfo(entry['path']);metadata.size=entry['bytes']
                    metadata.mode=path.stat().st_mode & 0o7777;metadata.mtime=int(path.stat().st_mtime)
                    with path.open('rb') as stream:archive.addfile(metadata,stream)
                else:
                    archive.add(path,arcname=entry['path'],recursive=False)
                if i%5000==0:print(f'  Archived {i:,}/{len(inventory):,} software entries',flush=True)
        archive_sha=digest(temporary)
        print('4/4 Verifying every archived file and copying the checked archive to Drive...',flush=True)
        expected={entry['path']:entry for _,entry in inventory};seen=set()
        with tarfile.open(temporary,'r:gz') as archive:
            for member in archive:
                assert member.name in expected and member.name not in seen
                entry=expected[member.name];seen.add(member.name)
                if entry['kind']=='file':
                    assert member.isfile() and member.size==entry['bytes']
                    h=hashlib.sha256()
                    with archive.extractfile(member) as f:
                        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
                    assert h.hexdigest()==entry['sha256']
                elif entry['kind']=='symlink':assert member.issym() and member.linkname==entry['target']
                else:assert member.isdir()
        assert seen==set(expected)
        target.parent.mkdir(exist_ok=True)
        pending=target.with_name(target.name+'.pending_'+uuid.uuid4().hex)
        shutil.copyfile(temporary,pending);assert digest(pending)==archive_sha
        assert not target.exists();pending.rename(target);assert digest(target)==archive_sha
        archive_record={'archive_path':str(target.relative_to(root)),'archive_sha256':archive_sha,
            'archive_bytes':target.stat().st_size,'inventory_sha256':inventory_sha,
            'uncompressed_file_bytes':total_bytes,'entries':len(inventory),
            'python_version':ready['python_version'],'lock_sha256':ready['lock_sha256'],
            'every_archived_file_hash_verified':True}
        dump(index_path,archive_record);temporary.unlink()
    write_once(root/'runtime'/('inventory_'+inventory_sha+'.json'),payload)
    dump(root/'configs/PORTABLE_RUNTIME_READY.json',archive_record)
    receipt={'status':'TASK1_BLOCK06_OK','cpu_model_contract':contract,
        'portable_archive_GiB':round(archive_record['archive_bytes']/2**30,2),
        'runtime_entries':archive_record['entries'],'every_archived_file_hash_verified':True,
        'elapsed_seconds':round(time.monotonic()-started,1),'gpu_work_run':False,
        'official_training_started':False,'qualitative_yaml_present':(root/'sources/first_plot_questions.yaml').is_file()}
    dump(root/'logs'/('block06_'+uuid.uuid4().hex+'.json'),receipt)
    print(json.dumps(receipt,indent=2),flush=True)
    print('TASK1_BLOCK06_OK — keep CPU selected and send the final summary.',flush=True)


if __name__=='__main__':main(*sys.argv[1:])
