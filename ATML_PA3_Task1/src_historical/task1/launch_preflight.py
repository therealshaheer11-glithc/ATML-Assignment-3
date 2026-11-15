"""Restore verified CPU-prepared assets and run the already-checked scratch worker."""
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import time
import uuid
from pathlib import Path,PurePosixPath


KIT_ID='20e10b447813341eed5b79026e35a28a1508e48bb1980b09b815e1298cbd97cd'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def safe_relative(value):
    p=PurePosixPath(value)
    if p.is_absolute() or '..' in p.parts or not p.parts or str(p)!=value:
        raise RuntimeError('Unsafe archive/manifest path: '+value)
    return p


def atomic(path,value):
    path=Path(path);pending=path.with_name(path.name+'.pending_'+uuid.uuid4().hex)
    pending.write_text(json.dumps(value,indent=2)+'\n');pending.replace(path)


def pulse(state_path,phase):
    state=json.loads(Path(state_path).read_text())
    if time.time()+10>=state['deadline_wall']:raise TimeoutError('Preflight budget/deadline reached during '+phase)
    state.update({'phase':phase,'progress_wall':time.time(),'progress_timeout':900})
    atomic(state_path,state)


def copy_archive(source,target,expected_sha,state_path):
    h=hashlib.sha256();last=time.monotonic()
    print('Copying',source.name,'from Drive...',flush=True)
    with Path(source).open('rb') as source_stream,Path(target).open('xb') as target_stream:
        for block in iter(lambda:source_stream.read(8*1024**2),b''):
            h.update(block);target_stream.write(block)
            if time.monotonic()-last>2:
                pulse(state_path,'copy '+Path(source).name);last=time.monotonic()
    if h.hexdigest()!=expected_sha:raise IOError('Archive hash mismatch: '+str(source))
    assert Path(target).stat().st_size==Path(source).stat().st_size


def restore(archive_path,destination,entries,heartbeat=lambda phase:None):
    """No extractall: validate each header, path, link and streamed file hash."""
    destination=Path(destination)
    if destination.is_symlink():raise RuntimeError('Linked extraction root')
    destination.mkdir(parents=True,exist_ok=True)
    expected={entry['path']:entry for entry in entries}
    if len(expected)!=len(entries):raise RuntimeError('Duplicate inventory paths')
    for rel,entry in expected.items():
        safe_relative(rel)
        if entry.get('kind','file')=='symlink':
            target=Path(entry['target'])
            resolved=(target if target.is_absolute() else (destination/rel).parent/target).resolve()
            if not resolved.is_relative_to(destination.resolve()):raise RuntimeError('External software symlink: '+rel)
    seen=set()
    with tarfile.open(archive_path,'r:*') as archive:
        for number,member in enumerate(archive,1):
            safe_relative(member.name)
            if member.name not in expected or member.name in seen:raise RuntimeError('Unexpected/duplicate archive entry')
            seen.add(member.name);entry=expected[member.name];kind=entry.get('kind','file')
            target=destination/member.name
            if not target.parent.resolve().is_relative_to(destination.resolve()):raise RuntimeError('Archive parent escapes root')
            target.parent.mkdir(parents=True,exist_ok=True)
            if kind=='directory':
                if not member.isdir() or target.is_symlink() or (target.exists() and not target.is_dir()):raise RuntimeError('Directory mismatch')
                target.mkdir(exist_ok=True)
            elif kind=='symlink':
                if not member.issym() or member.linkname!=entry['target']:raise RuntimeError('Symlink mismatch')
                if target.is_symlink():
                    if os.readlink(target)!=entry['target']:raise RuntimeError('Existing software link differs')
                elif target.exists():raise RuntimeError('Existing non-link at software symlink path')
                else:target.symlink_to(entry['target'])
            elif kind=='file':
                if not member.isfile() or member.size!=entry['bytes']:raise RuntimeError('File type/size mismatch')
                if target.is_symlink():raise RuntimeError('Unexpected linked output')
                if target.exists():
                    if not target.is_file() or digest(target)!=entry['sha256']:raise RuntimeError('Existing file differs: '+str(target))
                else:
                    pending=target.with_name(target.name+'.pending_'+uuid.uuid4().hex);h=hashlib.sha256()
                    with archive.extractfile(member) as stream,pending.open('xb') as out:
                        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block);out.write(block)
                    if h.hexdigest()!=entry['sha256']:raise IOError('Extracted content hash mismatch')
                    pending.chmod(member.mode&0o777)
                    if target.exists():raise RuntimeError('Concurrent extraction detected')
                    pending.rename(target)
            else:raise RuntimeError('Unsupported inventory type')
            if number%250==0:heartbeat('restore '+str(number)+'/'+str(len(entries)))
    if seen!=set(expected):raise RuntimeError('Incomplete archive coverage')
    heartbeat('archive fully restored and checked')


def run(context):
    from google.colab import drive
    state_path=Path(context['state']);job=Path(context['job']);started=context['started_wall']
    print('Mount Drive when prompted. The independent GPU timer is already running.',flush=True)
    drive.mount('/content/drive')
    root=Path('/content/drive/MyDrive/ATML-PA-3/Task1')
    state=json.loads(state_path.read_text());state['progress_wall']=time.time();atomic(state_path,state)
    ready=json.loads((root/'configs/GPU_PREFLIGHT_KIT_READY.json').read_text())
    assert ready['kit_id']==KIT_ID
    assert all(ready['supervisor_cpu_tests'].values()) and all(ready['numerical_checks'][k] for k in ('chunked_response_loss_matches_dense','all_parameter_gradients_match_dense'))
    assert hashlib.sha256(json.dumps(ready['files'],sort_keys=True).encode()).hexdigest()==KIT_ID
    assert ready['files']['gpu_guardian.py']==digest(job/'guardian.py')
    run_id=job.name
    record={'run_id':run_id,'status':'STARTED','started_wall':started,'balance_at_start':context['balance'],
        'displayed_units_per_hour':context['rate'],'estimated_unit_cap':1.,'maximum_allocation_minutes':20,
        'budget_rate_margin':1.1,'release_reserve_seconds':45,'official_training_updates':0,'kit_id':KIT_ID}
    record_path=root/'logs'/(run_id+'_session.json');atomic(record_path,record)
    budget=json.loads(state_path.read_text())['deadline_wall']+45
    if budget-time.time()<120:raise TimeoutError('Too little budget remains to restore the preflight environment')
    local=Path('/content/atml_pa3_task1');local.mkdir(exist_ok=True)
    if local.is_symlink():raise RuntimeError('Unexpected private runtime link')
    # Runtime inventory is frozen and was checked before this GPU session.
    runtime=ready['portable_runtime'];inv_sha=runtime['inventory_sha256']
    inv_path=root/'runtime'/('inventory_'+inv_sha+'.json')
    inventory_bytes=inv_path.read_bytes();assert hashlib.sha256(inventory_bytes).hexdigest()==inv_sha
    software_inventory=json.loads(inventory_bytes)
    assert software_inventory['base_directory']==str(local) and software_inventory['python_version']=='3.12.12'
    runtime_source=root/str(safe_relative(runtime['archive_path']))
    runtime_copy=job/'runtime.tar.gz'
    pulse(state_path,'software restoration begins')
    copy_archive(runtime_source,runtime_copy,runtime['archive_sha256'],state_path)
    restore(runtime_copy,local,software_inventory['files'],lambda phase:pulse(state_path,phase))
    runtime_copy.unlink()
    print('Verified Python environment restored; no packages installed or downloaded.',flush=True)
    inputs=ready['fixed_inputs'];inventory_path=root/str(safe_relative(ready['input_inventory_path']))
    inventory_bytes=inventory_path.read_bytes();assert hashlib.sha256(inventory_bytes).hexdigest()==inputs['inventory_sha256']
    input_copy=job/'inputs.tar';input_source=root/str(safe_relative(inputs['archive_path']))
    copy_archive(input_source,input_copy,inputs['archive_sha256'],state_path)
    input_root=local/'gpu_inputs'
    restore(input_copy,input_root,json.loads(inventory_bytes),lambda phase:pulse(state_path,phase))
    input_copy.unlink()
    kit=job/'kit';kit.mkdir()
    for name,h in ready['files'].items():
        safe_relative(name)
        if '/' in name:raise RuntimeError('Unexpected nested kit path')
        payload=(root/str(safe_relative(ready['kit_path']))/name).read_bytes()
        assert hashlib.sha256(payload).hexdigest()==h
        (kit/name).write_bytes(payload)
    result_path=root/'logs'/(run_id+'_result.json')
    config={'root':str(root),'state':str(state_path),'kit':str(kit),'inputs':str(input_root),'result':str(result_path)}
    config_path=job/'worker_config.json';atomic(config_path,config)
    python=local/'venv312/bin/python'
    env=os.environ.copy()
    for key in ('PYTHONPATH','PYTHONHOME','VIRTUAL_ENV','UV_SYSTEM_PYTHON'):env.pop(key,None)
    env.update({'PYTHONNOUSERSITE':'1','HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1',
        'HF_HUB_DISABLE_TELEMETRY':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'2',
        'MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false'})
    version=subprocess.run([str(python),'-I','-c','import platform; print(platform.python_version())'],
                           capture_output=True,text=True,env=env,timeout=30,check=True)
    assert version.stdout.strip()=='3.12.12'
    pulse(state_path,'start original-recipe maximum-length scratch checks')
    print('Starting scratch checks: Full FT, L2-SP, OPLoRA. Official training updates remain zero.',flush=True)
    worker=None
    try:
        with (root/'logs'/(run_id+'_worker.txt')).open('x') as log:
            worker=subprocess.Popen([str(python),'-I','-u',str(kit/'preflight_worker.py'),str(config_path)],
                env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,start_new_session=True)
            state=json.loads(state_path.read_text());state.update({'worker_pid':worker.pid,'progress_wall':time.time(),'status':'running'});atomic(state_path,state)
            for line in worker.stdout:print(line,end='',flush=True);log.write(line);log.flush()
            code=worker.wait()
            if code:raise RuntimeError('Scratch worker stopped; read the saved result for the measured failure')
        result=json.loads(result_path.read_text());assert result['status']=='TASK1_L4_PREFLIGHT_OK'
        record.update({'status':result['status'],'result_path':str(result_path),
            'elapsed_allocation_seconds':time.time()-started,
            'estimated_units_so_far_with_margin':(time.time()-started)/3600*context['rate']*1.1})
        atomic(record_path,record)
        print('Preflight saved:',result_path,flush=True)
        print('Requesting runtime release now. Return the output; do not start training.',flush=True)
    finally:
        if worker is not None and worker.poll() is None:
            try:os.killpg(worker.pid,15)
            except ProcessLookupError:pass
            try:worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:os.killpg(worker.pid,9)
                except ProcessLookupError:pass
