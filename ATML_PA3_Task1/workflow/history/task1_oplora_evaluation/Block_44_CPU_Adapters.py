# Block 44 — OPLoRA final adapter export. Run this entire cell on CPU.
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
source=root/'checkpoints/oplora/8cafd5243fcac8c1e3cd615a0497dfb2812b73e4ce46a997b161e85f504165a8/mandatory_001250_3575a06121f0'
expected='69862cbaad0e36fef5f8ffe87028c0b8bb3b5f7b768d9e9cdd662cf8434682b9'
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(8*2**20),b''):h.update(part)
    return h.hexdigest()
assert sha(source/'COMPLETE.json')=='24612f33ef613a066b2e9ba539e36e129c27fe4335ce06147b84bfba5f508a50'
assert hashlib.sha256((json.dumps(json.loads((source/'MANIFEST.json').read_bytes()),sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)+chr(10)).encode()).hexdigest()=='c51b64f04a6e9ebb37c9626b4159da4806eba054255b146cd16a253730168606'
m=json.loads((source/'MANIFEST.json').read_bytes())
assert m['header']['successful_updates']==1250 and m['header']['examples_seen']==40000
assert m['header']['method']=='oplora'
assert m['header']['contract_identity']==source.parent.name
with tempfile.TemporaryDirectory(dir='/content') as tmp:
    tmp=Path(tmp)
    assert shutil.disk_usage(tmp).free>512*2**20
    print('Copying and verifying checkpoint on CPU...',flush=True)
    archive=tmp/'state.tar';shutil.copyfile(source/'state.tar',archive)
    assert archive.stat().st_size==122654720 and sha(archive)==expected
    specs={s['id']:s for s in m['tensors'] if s['source'][0]=='model'}
    assert sum(s['numel'] for s in specs.values())==10092544
    assert len(specs)==392 and all(s['source'][1].endswith(('.A','.B')) for s in specs.values())
    assert len({s['source'][1] for s in specs.values()})==len(specs)
    assert all(s['dtype']=='torch.float32' for s in specs.values())
    output=tmp/'oplora_final_fp32.zip';index={};active={};offsets={}
    with tarfile.open(archive) as tar,zipfile.ZipFile(output,'x') as z:
        assert json.loads(tar.extractfile('MANIFEST.json').read())==m
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
                    active[i]=np.empty(spec['numel'],np.float32);offsets[i]=0
                pos=offsets[i];v=tensors[part['key']]
                assert pos==part['offset'] and v.dtype==torch.float32 and v.numel()==part['count']
                end=pos+v.numel();assert end<=spec['numel']
                active[i][pos:end]=v.view(-1).numpy()
                assert np.isfinite(active[i][pos:end]).all()
                offsets[i]=end
                if end==spec['numel']:
                    member=f'weights/{i}.npy';buf=io.BytesIO()
                    np.save(buf,active.pop(i).reshape(spec['shape']),allow_pickle=False)
                    payload=buf.getvalue();z.writestr(member,payload)
                    index[name]={'member':member,'shape':spec['shape'],'sha256':hashlib.sha256(payload).hexdigest()}
        assert not active and len(index)==len(specs)
        z.writestr('index.json',json.dumps({'condition':'oplora_final','update':1250,'checkpoint_sha256':expected,'dtype':'float32','unique_parameters':index},sort_keys=True))
    with zipfile.ZipFile(output) as checked:
        assert checked.testzip() is None
        assert len(checked.namelist())==len(index)+1
        assert json.loads(checked.read('index.json'))['unique_parameters']==index
        for item in index.values():
            payload=checked.read(item['member'])
            assert hashlib.sha256(payload).hexdigest()==item['sha256']
            array=np.load(io.BytesIO(payload),allow_pickle=False)
            assert array.dtype==np.float32 and list(array.shape)==item['shape']
            assert np.isfinite(array).all()
    (root/'evaluation_inputs').mkdir(exist_ok=True)
    h=sha(output);dest=root/'evaluation_inputs'/('oplora_final_1250_'+h+'.zip')
    if not dest.exists():
        pending=dest.with_name(dest.name+'.pending_'+uuid.uuid4().hex)
        shutil.copyfile(output,pending);assert sha(pending)==h;pending.rename(dest)
    assert sha(dest)==h
    print(json.dumps({'status':'OPLORA_FINAL_EVALUATION_WEIGHTS_READY','path':str(dest.relative_to(root)),'sha256':h,'unique_parameter_count':10092544,'checkpoint_update':1250},indent=2))
