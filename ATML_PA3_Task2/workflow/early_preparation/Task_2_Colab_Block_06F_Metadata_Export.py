# Block 6F: read-only metadata export for remediation feasibility. CPU only.
import os, json, hashlib, zipfile, uuid, io, shutil
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict, Counter

if os.environ.get('COLAB_GPU', '0') not in ('', '0'):
    from google.colab import runtime
    try:
        runtime.unassign()
    finally:
        raise RuntimeError('Reconnect with accelerator None for this CPU-only block.')

from google.colab import drive, files
if not Path('/content/drive/MyDrive').exists():
    drive.mount('/content/drive')
ROOT = Path('/content/drive/MyDrive/ATML_PA3/task2/run_20261031_150949_1cff148a')
LOCAL = Path('/content/atml_pa3_task2/run_20261031_150949_1cff148a')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))+'\n').encode()

LOCKS = {
    'logs/block_05a_20261031_155959_c7d3095c.json': 'a84f2436281e657665850dfad2750840b55678b271f4c1dffddc7f8e195973fc',
    'logs/block_05b_20261031_161050_c9dc05e9.json': '21ccdac5191bebd4caeba690adf481ecb47ef5ae476a7e03127731bb9ca5499a',
    'logs/block_06a_20261031_163930_3cb996af.json': '44a121f200cfd59ec38771010438328e8aaa35482989c5843ddd54c42ddbf13a',
    'manifests/selected/grounding_training_pools.json': '2f4cc89596c4e900d4a90a0b3af946ab7167c5adf4daf4901286a854c3d56e39',
    'manifests/selected/global_image_allocation.json': 'f7562959b948bda4143c3ffd9be975f5110ef7f74907dab9bca1af9eca9afe69',
    'manifests/selected/training_image_union.json': '54c92dfeffd2e8ccdc3f973a97455edd42979efa732cf765f3a9bfcb23506470',
    'manifests/selected/stage1_caption_epoch_orders.json': '4c2f1495fe97eaf05bf6a0975a7cbf6cca1bd91ef609dfac25e6a820bd1f0af8',
    'configs/calibration_sample_ids.json': '88a5f841d6c4feb84f188dcd7a8a2ebb972796316d460eba715d5b8b16162a77',
    'configs/exact_protocol_A005.json': 'fd7411f8ada3b43870f56cf7f12b48ce3a9067c9a04c8b8ecb3ba4ab7e5e7e4b',
    'audits/reconciliation_batch01_a39f40346d6aa4ba/reconciliation_batch01_20261107_085944_6c3c89.json': 'f50b8ad9b306bd08fccf75d5fba267adeb3f97b1e35c7ee10052feb29ad112e9',
}
for relative, expected in LOCKS.items():
    if not (ROOT/relative).is_file() or sha(ROOT/relative) != expected:
        raise IOError(f'Missing or changed source; no selection changes made: {relative}')

# Check every saved selected-output hash in the successful execution logs.
for relative in list(LOCKS)[:3]:
    report = json.loads((ROOT/relative).read_text())
    assert report['status'] == 'passed'
    for item in report['outputs']:
        path = Path(item['path'])
        assert path.is_relative_to(ROOT) and sha(path) == item['sha256'], str(path)
        LOCKS[str(path.relative_to(ROOT))] = item['sha256']
print('Saved selections, pools, allocation, calibration IDs and reconciliation verified.')

# Reconstruct candidate metadata using the original annotation-count rules; no sampling.
receipt = json.loads((ROOT/'manifests/annotation_downloads/coco2014.json').read_text())
val_hash = next(r['sha256'] for r in receipt['json_files'] if r['filename']=='instances_val2014.json')
val_local = LOCAL/'annotations/instances_val2014.json'
if val_local.exists():
    assert sha(val_local) == val_hash
    with val_local.open() as stream:
        val = json.load(stream)
else:
    archive = ROOT/'raw_annotation_archives/coco2014.zip'
    assert sha(archive) == '031296bbc80c45a1d1f76bf9a90ead27e94e99ec629208449507a4917a3bf009'
    with zipfile.ZipFile(archive) as z:
        names = [n for n in z.namelist() if Path(n).name=='instances_val2014.json']
        assert len(names)==1
        raw = z.read(names[0])
    assert hashlib.sha256(raw).hexdigest() == val_hash
    val = json.loads(raw)
    del raw
assert len(val['images'])==40504 and len(val['categories'])==80
present, crowd, counts = defaultdict(set), defaultdict(set), defaultdict(Counter)
for a in val['annotations']:
    cid, iid = int(a['category_id']), int(a['image_id'])
    present[cid].add(iid)
    if a.get('iscrowd',0):
        crowd[cid].add(iid)
    else:
        counts[cid][iid] += 1
catalog = []
for c in sorted(val['categories'], key=lambda x:int(x['id'])):
    cid = int(c['id'])
    catalog.append({'category_id':cid,'category_name':c['name'],
        'present':sorted(present[cid]),'crowd_images':sorted(crowd[cid]),
        'count_images':{str(k):sorted(i for i,n in counts[cid].items() if n==k and i not in crowd[cid])
                        for k in range(1,5)}})
index = {'source_split':'val2014','source_instances_sha256':val_hash,
    'images':[{k:im[k] for k in ('id','file_name','width','height','license')} for im in sorted(val['images'],key=lambda x:int(x['id']))],
    'categories':catalog,'absence_rule':'All source image IDs minus present IDs for that category',
    'reserved_images_not_removed_here':True,'purpose':'Candidate metadata only; no replacement sampled or approved'}
del val
print('Built metadata for 40,504 validation images; no replacement IDs selected.')

sources = {ROOT/r for r in LOCKS}
for folder in ('manifests/selected','configs'):
    sources.update(p for p in (ROOT/folder).glob('*') if p.is_file() and p.suffix in ('.json','.jsonl'))
sources.add(ROOT/'manifests/annotation_downloads/coco2014.json')
stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_')+uuid.uuid4().hex[:8]
local_dir = LOCAL/'remediation_metadata_exports'
local_dir.mkdir(parents=True,exist_ok=True)
target = local_dir/f'remediation_metadata_{stamp}.zip'
manifest = {'schema':'atml-pa3-task2-remediation-metadata-v1','run_id':ROOT.name,
    'created_at_utc':datetime.now(timezone.utc).isoformat(),'files':[],
    'dataset_frozen':False,'data_changes_applied':False,'replacements_selected':False,
    'validated_source_locks':LOCKS}
with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED) as z:
    for path in sorted(sources):
        assert path.is_relative_to(ROOT)
        name='source/'+str(path.relative_to(ROOT));raw=path.read_bytes()
        if str(path.relative_to(ROOT)) in LOCKS:
            assert hashlib.sha256(raw).hexdigest()==LOCKS[str(path.relative_to(ROOT))]
        z.writestr(name,raw)
        manifest['files'].append({'path':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
    raw=encoded(index);name='derived/val2014_candidate_index.json';z.writestr(name,raw)
    manifest['files'].append({'path':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
    source_code=globals().get('metadata_export_cell_bytes')
    if isinstance(source_code,bytes):
        name='code/executed_block_06F.py';z.writestr(name,source_code)
        manifest['files'].append({'path':name,'bytes':len(source_code),'sha256':hashlib.sha256(source_code).hexdigest()})
    z.writestr('bundle_manifest.json',encoded(manifest))
with zipfile.ZipFile(target) as z:
    assert z.testzip() is None
    for item in manifest['files']:
        raw=z.read(item['path'])
        assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256']
durable=ROOT/'audits/remediation_metadata_exports'/target.name
durable.parent.mkdir(parents=True,exist_ok=True)
assert not durable.exists()
partial=durable.with_suffix('.part')
shutil.copyfile(target,partial)
assert sha(partial)==sha(target)
partial.rename(durable)
assert sha(durable)==sha(target)
result={'status':'passed','zip_path':str(durable),'zip_bytes':durable.stat().st_size,
    'zip_sha256':sha(durable),'members':len(manifest['files'])+1,'drive_readback_verified':True,
    'validation_images_indexed':40504,'data_changes_applied':False,'replacements_selected':False,
    'dataset_frozen':False,'gpu_used':False}
log=ROOT/'logs'/f'block_06f_{stamp}.json';log.write_bytes(encoded(result))
assert json.loads(log.read_bytes())==result
print('BLOCK 6F COMPLETE — read-only remediation metadata export.')
print(json.dumps(result,indent=2))
print('Log SHA256:',sha(log))
files.download(str(durable))
print('Return the downloaded ZIP and full printed output to this chat. Keep accelerator None.')
