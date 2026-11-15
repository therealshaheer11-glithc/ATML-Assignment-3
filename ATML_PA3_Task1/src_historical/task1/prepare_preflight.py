"""CPU-only preflight kit validation, real process safety tests and input transport."""
import hashlib
import importlib.util
import json
import shutil
import sys
import tarfile
import time
import uuid
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def dump(path,value):
    data=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=data:raise RuntimeError('Frozen artifact differs: '+str(path))
        return
    tmp=path.with_name(path.name+'.pending_'+uuid.uuid4().hex);tmp.write_bytes(data)
    assert tmp.read_bytes()==data
    if path.exists():raise RuntimeError('Concurrent writes')
    tmp.rename(path);assert path.read_bytes()==data


def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj


def main(root,local,bundle):
    root,local,bundle=Path(root),Path(local),Path(bundle);started=time.monotonic()
    contract=json.loads((root/'configs/CPU_MODEL_CONTRACT.json').read_text())
    assert contract['fp32_master_state_sha256']=='43f910d3e84e52f5214474dca7c74748e1701aec10440d8660ec1332a1ee1a3f'
    runtime=json.loads((root/'configs/PORTABLE_RUNTIME_READY.json').read_text())
    assert runtime['every_archived_file_hash_verified']
    assets=json.loads((root/'manifests/CPU_ASSETS_READY.json').read_text())
    assert digest(bundle/'training_core.py')==assets['training_core_sha256']
    print('1/4 Checking bounded response-loss computation against dense loss and gradients on CPU...',flush=True)
    ops=load_module(bundle/'training_ops.py','preflight_ops')
    numerical=ops.check_chunked_gradients()
    print(json.dumps(numerical,indent=2),flush=True)
    print('2/4 Testing independent supervisor on six real CPU child-process scenarios...',flush=True)
    test=load_module(bundle/'guardian_selftest.py','selftest')
    outcomes=test.run_tests(bundle/'gpu_guardian.py',local/('safety_tests_'+uuid.uuid4().hex))
    assert len(outcomes)==6 and all(outcomes.values())
    print(json.dumps(outcomes,indent=2),flush=True)
    print('3/4 Saving the checked GPU worker/supervisor and packing fixed inputs for fast GPU restoration...',flush=True)
    names=('gpu_guardian.py','preflight_worker.py','training_ops.py','training_core.py','guardian_selftest.py','prepare_preflight.py')
    hashes={name:digest(bundle/name) for name in names}
    kit_id=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    destination=root/'execution'/kit_id
    destination.mkdir(parents=True,exist_ok=True)
    for name in names:
        path=destination/name
        if path.exists():assert digest(path)==hashes[name]
        else:
            tmp=path.with_name(path.name+'.pending_'+uuid.uuid4().hex)
            shutil.copyfile(bundle/name,tmp);assert digest(tmp)==hashes[name]
            assert not path.exists();tmp.rename(path)
    config_names=('SOURCE_REVISIONS.json','APPROVED_PACKAGE_PINS.json','PYTHON_ENVIRONMENT.json',
        'ENVIRONMENT_READY.json','CPU_MODEL_CONTRACT.json','TOKENIZATION_PROTOCOL_R1.json',
        'ACTIVATION_PROBE_IDS.json','TRAIN_SOURCE_IDS.json','DEPENDENCY_LOCK.json')
    paths=[root/'configs'/name for name in config_names]
    for name in ('sources/original_model','sources/benchmarks','bases','initializers','data','manifests'):
        paths.extend(sorted(p for p in (root/name).rglob('*') if p.is_file()))
    expected={}
    data=json.loads((root/'manifests/TRAINING_DATA_READY.json').read_text())
    for entry in data['shards']:expected[entry['path']]=entry['sha256']
    for entry in assets['bases']:expected['bases/'+entry['key']+'.safetensors']=entry['basis_sha256']
    for entry in assets['initializer_files'].values():expected[entry['path']]=entry['sha256']
    originals=json.loads((root/'manifests/ORIGINAL_MODEL_FILES.json').read_text())
    for name,entry in originals['files'].items():expected['sources/original_model/'+name]=entry['sha256']
    inventory=[]
    for path in paths:
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):raise RuntimeError('Unexpected input link')
        rel=str(path.relative_to(root));h=digest(path)
        if rel in expected:assert h==expected[rel],rel
        inventory.append({'path':rel,'sha256':h,'bytes':path.stat().st_size})
    assert len({entry['path'] for entry in inventory})==len(inventory)
    inventory_bytes=(json.dumps(inventory,sort_keys=True,separators=(',',':'))+'\n').encode()
    inv_sha=hashlib.sha256(inventory_bytes).hexdigest()
    transport=root/'runtime'/('fixed_inputs_'+inv_sha[:16]+'.tar')
    index=transport.with_suffix('.json')
    if transport.exists() and index.exists():
        record=json.loads(index.read_text());assert record['inventory_sha256']==inv_sha
        assert digest(transport)==record['archive_sha256']
    else:
        if transport.exists() or index.exists():raise RuntimeError('Incomplete input archive pair; preserving it')
        temporary=local/('fixed_inputs_'+uuid.uuid4().hex+'.tar')
        needed=sum(entry['bytes'] for entry in inventory)
        if shutil.disk_usage(local).free<needed+2*2**30:raise RuntimeError('Insufficient input archive disk space')
        with tarfile.open(temporary,'w') as archive:
            for path,entry in zip(paths,inventory):
                info=tarfile.TarInfo(entry['path']);info.size=entry['bytes'];info.mode=0o644
                with path.open('rb') as stream:archive.addfile(info,stream)
        print('4/4 Verifying the input archive, copying it to Drive, and saving preflight readiness...',flush=True)
        with tarfile.open(temporary,'r:') as archive:
            members=list(archive);assert len(members)==len(inventory)
            for member,entry in zip(members,inventory):
                assert member.name==entry['path'] and member.isfile() and member.size==entry['bytes']
                h=hashlib.sha256()
                with archive.extractfile(member) as stream:
                    for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
                assert h.hexdigest()==entry['sha256']
        sha=digest(temporary);pending=transport.with_name(transport.name+'.pending_'+uuid.uuid4().hex)
        shutil.copyfile(temporary,pending);assert digest(pending)==sha
        assert not transport.exists();pending.rename(transport);assert digest(transport)==sha
        record={'archive_path':str(transport.relative_to(root)),'archive_sha256':sha,
            'archive_bytes':transport.stat().st_size,'inventory_sha256':inv_sha,'files':len(inventory),
            'every_archived_file_verified':True}
        dump(index,record);temporary.unlink()
    inventory_path=root/'runtime'/('inputs_inventory_'+inv_sha+'.json')
    if inventory_path.exists():assert inventory_path.read_bytes()==inventory_bytes
    else:inventory_path.write_bytes(inventory_bytes);assert inventory_path.read_bytes()==inventory_bytes
    readiness={'kit_id':kit_id,'kit_path':str(destination.relative_to(root)),'files':hashes,
        'portable_runtime':runtime,'fixed_inputs':record,'input_inventory_path':str(inventory_path.relative_to(root)),
        'numerical_checks':numerical,'supervisor_cpu_tests':outcomes,
        'gpu_preflight_limits':{'minutes':20,'estimated_compute_units':1,'headroom_fraction':.1,'release_reserve_seconds':45},
        'official_training_updates':0,'gpu_work_run':False}
    dump(root/'configs/GPU_PREFLIGHT_KIT_READY.json',readiness)
    receipt={'status':'TASK1_BLOCK07_OK','kit_id':kit_id,'numerical_checks':numerical,
        'supervisor_cpu_tests':outcomes,'fixed_input_archive_GiB':round(record['archive_bytes']/2**30,2),
        'input_files':record['files'],'elapsed_seconds':round(time.monotonic()-started,1),
        'gpu_work_run':False,'official_training_started':False,
        'next':'Return output before selecting L4; the GPU launch block must start its independent guardian first.'}
    dump(root/'logs'/('block07_'+uuid.uuid4().hex+'.json'),receipt)
    print(json.dumps(receipt,indent=2),flush=True)
    print('TASK1_BLOCK07_OK — keep CPU selected and send the final summary.',flush=True)


if __name__=='__main__':main(*sys.argv[1:])
