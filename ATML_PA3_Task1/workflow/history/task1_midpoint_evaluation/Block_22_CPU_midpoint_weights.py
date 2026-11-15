# Run the entire cell on CPU.
import os, shutil, subprocess, threading, json, hashlib, io, tarfile, zipfile, tempfile, uuid
from pathlib import Path
def cpu_guard():
    evidence = []
    if os.environ.get('COLAB_GPU', '0').lower().strip() not in ('', '0', 'false', 'none'):
        evidence.append('GPU allocation flag')
    if list(Path('/dev').glob('nvidia[0-9]*')):
        evidence.append('GPU device')
    if any(os.environ.get(k) for k in ('COLAB_TPU_ADDR', 'TPU_NAME', 'TPU_WORKER_ID')) or os.environ.get('PJRT_DEVICE', '').upper() == 'TPU':
        evidence.append('TPU allocation')
    if shutil.which('nvidia-smi'):
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                                    capture_output=True, text=True, timeout=10)
            if result.returncode or result.stdout.strip(): evidence.append('GPU present or status uncertain')
        except Exception:
            evidence.append('GPU status uncertain')
    if evidence:
        print('CPU required; requesting release:', evidence, flush=True)
        def release():
            try:
                from google.colab import runtime
                runtime.unassign()
            except Exception as error:
                print('Release request:', type(error).__name__, flush=True)
        thread = threading.Thread(target=release, daemon=True); thread.start(); thread.join(10)
        raise RuntimeError('Select CPU; end this notebook\'s remaining L4 session through Manage sessions if active.')
cpu_guard()
from google.colab import drive
drive.mount('/content/drive')
import torch
import numpy as np
root=Path('/content/drive/MyDrive/ATML-PA-3/Task1')
source=root/'checkpoints/full/6ed4a55411318958c5d853334b7069e0cdf752c628732b756b6b2c971ac0343b/mandatory_000625_b4ee3aa2f19b'
expected='e1b62bfcd438ca057c3f60028f6bf6c42bbc46538a88262a8795739bbac33119'
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(8*2**20),b''):h.update(part)
    return h.hexdigest()
assert sha(source/'COMPLETE.json')=='64c48635d333b7510eb5da985481d9e4f74a60d03f8c1d2758b73521dfdc2d96'
assert sha(source/'MANIFEST.json')=='f61d709442f8ba736a17e69f841e5200ff011ebc1d3bb74118708335cf188035'
m=json.loads((source/'MANIFEST.json').read_bytes())
assert m['header']['successful_updates']==625 and m['header']['examples_seen']==20000
with tempfile.TemporaryDirectory(dir='/content') as tmp:
    tmp=Path(tmp)
    assert shutil.disk_usage(tmp).free>10*2**30
    print('Copying and verifying checkpoint on CPU...',flush=True)
    archive=tmp/'state.tar';shutil.copyfile(source/'state.tar',archive)
    assert sha(archive)==expected
    specs={s['id']:s for s in m['tensors'] if s['source'][0]=='model'}
    assert sum(s['numel'] for s in specs.values())==596049920
    output=tmp/'full_mid_fp16.zip';index={};active={};offsets={}
    with tarfile.open(archive) as tar,zipfile.ZipFile(output,'x') as z:
        for shard in m['shards']:
            parts=[p for p in shard['pieces'] if p['tensor_id'] in specs]
            if not parts:continue
            raw=tar.extractfile(shard['path']).read()
            assert hashlib.sha256(raw).hexdigest()==shard['sha256']
            tensors=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
            for part in parts:
                i=part['tensor_id'];spec=specs[i];name=spec['source'][1]
                if i not in active:
                    assert name not in index
                    active[i]=np.empty(spec['numel'],np.float16);offsets[i]=0
                pos=offsets[i];v=tensors[part['key']]
                assert pos==part['offset'] and v.dtype==torch.float32 and v.numel()==part['count']
                end=pos+v.numel();active[i][pos:end]=v.view(-1).numpy()
                assert np.isfinite(active[i][pos:end]).all()
                offsets[i]=end
                if end==spec['numel']:
                    member=f'weights/{i}.npy';buf=io.BytesIO()
                    np.save(buf,active.pop(i).reshape(spec['shape']),allow_pickle=False)
                    payload=buf.getvalue();z.writestr(member,payload)
                    index[name]={'member':member,'shape':spec['shape'],'sha256':hashlib.sha256(payload).hexdigest()}
        assert not active and len(index)==len(specs)
        z.writestr('index.json',json.dumps({'condition':'full_mid','update':625,'checkpoint_sha256':expected,'dtype':'float16','unique_parameters':index},sort_keys=True))
    h=sha(output);dest=root/'evaluation_inputs'/('full_mid_625_'+h+'.zip')
    if not dest.exists():
        pending=dest.with_name(dest.name+'.pending_'+uuid.uuid4().hex)
        shutil.copyfile(output,pending);assert sha(pending)==h;pending.rename(dest)
    assert sha(dest)==h
    print(json.dumps({'status':'FULL_MID_EVALUATION_WEIGHTS_READY','path':str(dest.relative_to(root)),'sha256':h,'unique_parameter_count':596049920,'checkpoint_update':625},indent=2))
