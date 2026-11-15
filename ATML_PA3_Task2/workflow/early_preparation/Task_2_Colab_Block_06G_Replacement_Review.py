# Block 6G: A007 candidate preparation and human replacement review. CPU only.
import os, json, hashlib, shutil, zipfile, uuid, traceback, base64, html, csv, io, copy, gc, time
import urllib.request
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

if os.environ.get('COLAB_GPU', '0') not in ('', '0'):
    from google.colab import runtime
    try:
        runtime.unassign()
    finally:
        raise RuntimeError('Reconnect with accelerator None for label review.')
from google.colab import drive, output, files
from IPython.display import display, HTML, clear_output
import ipywidgets as W
output.enable_custom_widget_manager()
if not Path('/content/drive/MyDrive').exists():
    drive.mount('/content/drive')
previous = globals().get('A007_REPLACEMENT_APP')
if previous is not None:
    previous.close()
ROOT=Path('/content/drive/MyDrive/ATML_PA3/task2/run_20261031_150949_1cff148a')
LOCAL=Path('/content/atml_pa3_task2/run_20261031_150949_1cff148a')

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def encode_json(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()

def write_verified(path, content, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        if path.read_bytes() != content:
            raise IOError(f'Different existing file preserved: {path}')
        return
    temp = path.with_name(path.name + '.part_' + uuid.uuid4().hex[:8])
    temp.write_bytes(content)
    if temp.read_bytes() != content:
        raise IOError('Save verification failed before rename.')
    temp.replace(path)
    if path.read_bytes() != content:
        raise IOError('Save verification failed after rename.')

def utc_now():
    return datetime.now(timezone.utc).isoformat()


package_bytes=globals().get('remediation_package_bytes')
if not isinstance(package_bytes,bytes):
    raise RuntimeError('Run the supplied one-cell package loader.')
with zipfile.ZipFile(io.BytesIO(package_bytes)) as package:
    manifest=json.loads(package.read('package_manifest.json'))
    assets={}
    for item in manifest['files']:
        value=package.read(item['path'])
        assert len(value)==item['bytes'] and hashlib.sha256(value).hexdigest()==item['sha256']
        assets[item['path']]=value
PLAN_RAW=assets['replacement_plan_A007.json'];PLAN_SHA=hashlib.sha256(PLAN_RAW).hexdigest()
PLAN=json.loads(PLAN_RAW)
assert PLAN['schema']=='atml-pa3-task2-A007-replacement-plan-v1'
assert PLAN['run_id']==ROOT.name and len(PLAN['candidate_pairs'])==86
assert PLAN['dataset_frozen'] is False and PLAN['replacements_accepted'] is False
assert hashlib.sha256(assets['approval_A007.json']).hexdigest()==PLAN['approval_sha256']
assert hashlib.sha256(assets['grounding_training_pools_A007_candidate.json']).hexdigest()==PLAN['candidate_pool_sha256']
BASE_SHA=PLAN['base_export_sha256']
assert hashlib.sha256(assets['source/original_human_review.json']).hexdigest()==BASE_SHA
assert hashlib.sha256(assets['source/reconciliation_batch01.json']).hexdigest()==PLAN['reconciliation_export_sha256']
for relative,expected in PLAN['source_locks'].items():
    if not (ROOT/relative).is_file() or digest(ROOT/relative)!=expected:
        raise IOError('Source changed/missing; originals preserved: '+relative)
assert digest(ROOT/'manifests/annotation_downloads/coco2014.json')=='4aa0cd5afbabb49e16d1ffacb1dce25dbe7b1440728e78f097b4d9c082c101f6'
raw_pool=json.loads((ROOT/'manifests/selected/grounding_training_pools.json').read_text())
mask=defaultdict(set)
for entry in PLAN['quarantine']:mask[entry['category_id']].add(entry['image_id'])
assert len(PLAN['quarantine'])==sum(len(s) for s in mask.values())==66
version_pool=json.loads(assets['grounding_training_pools_A007_candidate.json'])
assert version_pool['image_ids']==raw_pool['image_ids']==PLAN['training_image_union']
for original,pruned in zip(raw_pool['categories'],version_pool['categories']):
    assert original['category_id']==pruned['category_id']
    blocked=mask[original['category_id']]
    for field in ['present','absent','crowd_images']:
        assert pruned[field]==[x for x in original[field] if x not in blocked]
    for count,images in original['count_images'].items():
        assert pruned['count_images'][count]==[x for x in images if x not in blocked]
FOLDER=ROOT/'audits'/('replacement_review_A007_'+PLAN_SHA[:16])
IMAGE_FOLDER=LOCAL/'replacement_A007'/PLAN_SHA[:16]
FOLDER.mkdir(parents=True,exist_ok=True)
for name,value in assets.items():
    assert not Path(name).is_absolute() and '..' not in Path(name).parts
    write_verified(FOLDER/'prepared_sources'/name,value)
write_verified(FOLDER/'prepared_sources'/'uploaded_package.zip',package_bytes)
base_review=json.loads(assets['source/original_human_review.json'])
reconciliation=json.loads(assets['source/reconciliation_batch01.json'])
context={r['pair_id']:r for r in base_review['judgments']}
context.update({r['pair_id']:r for r in reconciliation['judgments']})
ORIGINALS={};ORIGINAL_NUMBER={};QUESTIONS={}
for pair in PLAN['candidate_pairs']:
    pid=pair['pair_id'];old=pair['replacement_of_pair_id'];ORIGINALS[pid]=context[old]
    assert PLAN['original_human_context'][old]=={k:context[old][k] for k in PLAN['original_human_context'][old]}
    ORIGINAL_NUMBER[pid]=pair['original_review_index']
    targets='/'.join(e['target'] for e in pair['examples'])
    QUESTIONS[pair['original_review_index']]=('Inspect BOTH candidate images for targets '+targets+'. '
        +'Targets and source images are unchanged original annotations; no visual acceptance is prefilled. '
        +('This training-audit pair uses the A007-approved count-combination fallback. ' if pair['count_combination_fallback'] else '')
        +('An eligible supported original training partner is retained. ' if any(pair['retained_original_sides']) else ''))
BATCH=[p['original_review_index'] for p in PLAN['candidate_pairs']]
assert len(BATCH)==len(set(BATCH))==86
train=set(PLAN['training_image_union']);reserved=set(PLAN['original_reserved_validation_images']);new_eval=[]
for pair in PLAN['candidate_pairs']:
    ids=[e['image_id'] for e in pair['examples']]
    assert len(set(ids))==2 and pair['question_content_tokens']<=80 and pair['user_content_tokens']<=80
    if pair['source_split']=='train2014':
        assert set(ids)<=train and not (set(ids)&mask[pair['category_id']])
    else:
        assert pair['source_split']=='val2014' and not (set(ids)&(train|reserved));new_eval+=ids
assert len(new_eval)==len(set(new_eval))==60

# Restore/read original annotations; verify exact source bytes before using labels or boxes.
receipt=json.loads((ROOT/'manifests/annotation_downloads/coco2014.json').read_text())
hashes={r['filename']:r['sha256'] for r in receipt['json_files']}
requested={(p['source_split'],e['image_id'],p['category_id']) for p in PLAN['candidate_pairs'] for e in p['examples']}
metadata={};boxes={};licenses={};jobs=[]
for split in ['train2014','val2014']:
    print('Verifying source annotations:',split,flush=True)
    filename=f'instances_{split}.json';local=LOCAL/'annotations'/filename
    if local.exists():
        assert digest(local)==hashes[filename]
        with local.open() as stream:payload=json.load(stream)
    else:
        archive=ROOT/'raw_annotation_archives/coco2014.zip'
        assert digest(archive)=='031296bbc80c45a1d1f76bf9a90ead27e94e99ec629208449507a4917a3bf009'
        with zipfile.ZipFile(archive) as z:
            names=[n for n in z.namelist() if Path(n).name==filename];assert len(names)==1
            raw=z.read(names[0])
        assert hashlib.sha256(raw).hexdigest()==hashes[filename]
        payload=json.loads(raw);del raw
    wanted={iid for s,iid,c in requested if s==split}
    selected={int(im['id']):im for im in payload['images'] if int(im['id']) in wanted}
    assert set(selected)==wanted
    licenses[split]=payload.get('licenses',[])
    category_names={int(c['id']):c['name'] for c in payload['categories']}
    for pair in PLAN['candidate_pairs']:
        if pair['source_split']==split:assert category_names[pair['category_id']]==pair['category_name']
    index=defaultdict(list)
    for a in payload['annotations']:
        key=(split,int(a['image_id']),int(a['category_id']))
        if key in requested:index[key].append({'annotation_id':int(a['id']),'bbox':a['bbox'],'iscrowd':bool(a.get('iscrowd',0))})
    for key in requested:
        if key[0]==split:boxes[key]=sorted(index[key],key=lambda a:a['annotation_id'])
    for iid,im in sorted(selected.items()):
        name=im['file_name'];assert Path(name).name==name and name.endswith(f'{iid:012d}.jpg')
        job={'source_split':split,'image_id':iid,'file_name':name,'width':int(im['width']),'height':int(im['height']),
            'url':f'https://s3.amazonaws.com/images.cocodataset.org/{split}/{name}',
            'original_coco_url':im.get('coco_url')}
        jobs.append(job);metadata[(split,iid)]={**job,'license_id':im.get('license'),'flickr_url':im.get('flickr_url')}
    del payload,index,selected;gc.collect()
for pair in PLAN['candidate_pairs']:
    for e in pair['examples']:
        source=boxes[(pair['source_split'],e['image_id'],pair['category_id'])]
        if pair['question_type']=='presence':assert bool(source)==(e['target']=='yes')
        else:assert not any(a['iscrowd'] for a in source) and len(source)==int(e['target'])
print('All 172 candidate-image labels match original annotations. This is not visual acceptance.',flush=True)


def copy_checked(source,target):
    source,target=Path(source),Path(target)
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if digest(source)!=digest(target):raise IOError('Existing different file preserved: '+str(target))
        return
    temporary=target.with_name(target.name+'.part_'+uuid.uuid4().hex[:8])
    shutil.copyfile(source,temporary)
    assert digest(temporary)==digest(source)
    temporary.replace(target)
    assert digest(target)==digest(source)

def save_record(relative,value):
    target=ROOT/relative
    write_verified(target,encode_json(value))
    return digest(target)

file_sha=digest

def inspect_original(path, expected):
    with Image.open(path) as image:
        image.load()
        if image.format != 'JPEG' or image.size != (expected['width'], expected['height']):
            raise IOError(f'Image format/dimensions mismatch: {path}')
        return {'width': image.width, 'height': image.height, 'mode': image.mode,
                'exif_orientation': int(image.getexif().get(274, 1))}

def ensure_audit_image(job):
    local = LOCAL / 'images' / job['source_split'] / job['file_name']
    durable = ROOT / 'raw_images' / job['source_split'] / job['file_name']
    receipt_rel = f'manifests/images/{job["source_split"]}/{job["image_id"]:012d}.json'
    receipt_path = ROOT / receipt_rel
    local.parent.mkdir(parents=True, exist_ok=True)
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt['url'] != job['url'] or receipt['image_id'] != job['image_id']:
            raise IOError('Conflicting saved image identity.')
        if durable.exists():
            if file_sha(durable) != receipt['sha256']:
                raise IOError(f'Drive image differs from its receipt: {durable}')
            copy_checked(durable, local)
        if not local.exists() or file_sha(local) != receipt['sha256']:
            raise IOError('Recorded image is missing/changed; preserve it and report the error.')
        assert inspect_original(local, job) == receipt['decoded_metadata']
        copy_checked(local, durable)
        return receipt, 'reused'

    temporary = local.with_name(local.name + '.download.part')
    last_error = None
    for retry in range(3):
        try:
            request = urllib.request.Request(job['url'], headers={'User-Agent': 'ATML-PA3-CPU-data-preparation'})
            with urllib.request.urlopen(request, timeout=30) as response:
                if not response.geturl().startswith('https://'):
                    raise IOError('Unexpected non-HTTPS redirect.')
                expected_bytes = response.headers.get('Content-Length')
                with temporary.open('wb') as out:
                    shutil.copyfileobj(response, out)
                if expected_bytes is not None and temporary.stat().st_size != int(expected_bytes):
                    raise IOError('Incomplete image response.')
            decoded = inspect_original(temporary, job)
            last_error = None
            break
        except Exception as exc:
            last_error = exc
            if retry < 2:
                time.sleep(retry + 1)
    if last_error is not None:
        raise last_error
    digest = file_sha(temporary)
    if local.exists():
        if file_sha(local) != digest:
            raise IOError(f'Existing unrecorded image differs; preserved: {local}')
        temporary.unlink()
    else:
        temporary.rename(local)
    copy_checked(local, durable)
    receipt = {'source_split': job['source_split'], 'image_id': job['image_id'],
               'file_name': job['file_name'], 'url': job['url'],
               'original_coco_url': job.get('original_coco_url'),
               'sha256': digest, 'bytes': local.stat().st_size,
               'decoded_metadata': decoded,
               'hash_provenance': 'Observed original-image SHA256; HTTPS COCO object download'}
    save_record(receipt_rel, receipt)
    return receipt, 'downloaded'

start=time.monotonic();image_map={};activities=Counter()
try:
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures={executor.submit(ensure_audit_image,job):job for job in jobs}
        for future in as_completed(futures):
            try:
                record,activity=future.result()
            except Exception:
                for pending in futures: pending.cancel()
                raise
            if record['decoded_metadata']['exif_orientation']!=1:
                raise ValueError('Nonstandard EXIF orientation: stop for viewer-handling review.')
            key=(record['source_split'],record['image_id']);image_map[key]=record;activities[activity]+=1
            source=LOCAL/'images'/record['source_split']/record['file_name']
            destination=IMAGE_FOLDER/'images'/record['source_split']/record['file_name']
            copy_checked(source,destination)
            if len(image_map)%20==0 or len(image_map)==len(jobs):
                print(f'Original images verified and saved: {len(image_map)}/{len(jobs)}',flush=True)
except Exception:
    write_verified(FOLDER/('preparation_failure_'+uuid.uuid4().hex[:8]+'.json'),encode_json({
        'status':'failed','traceback':traceback.format_exc(),'verified_images':len(image_map),
        'plan_sha256':PLAN_SHA,'dataset_frozen':False,'gpu_used':False}))
    raise
fingerprints={**PLAN['source_locks'],'original_human_export':BASE_SHA,
    'reconciliation_export':PLAN['reconciliation_export_sha256'],'replacement_plan_A007':PLAN_SHA,
    'candidate_training_pool':PLAN['candidate_pool_sha256'],
    'candidate_image_receipts':hashlib.sha256(encode_json([image_map[k] for k in sorted(image_map)])).hexdigest()}
BINDING=hashlib.sha256(encode_json({'run_id':ROOT.name,'source_fingerprints':fingerprints})).hexdigest()
view=[]
for pair in PLAN['candidate_pairs']:
    p={k:pair[k] for k in ['pair_id','source_split','experiment_split','question_type','category_id','category_name','question']}
    p['examples']=[]
    for e in pair['examples']:
        record=image_map[(pair['source_split'],e['image_id'])];meta=record['decoded_metadata']
        p['examples'].append({'image_id':e['image_id'],'target':e['target'],
            'src':'images/'+pair['source_split']+'/'+record['file_name'],
            'width':meta['width'],'height':meta['height'],'mode':meta['mode'],
            'boxes':boxes[(pair['source_split'],e['image_id'],pair['category_id'])]})
    view.append(p)
DATA={'schema':'atml-pa3-task2-A007-replacement-viewer-v1','run_id':ROOT.name,'binding_sha256':BINDING,
    'source_fingerprints':fingerprints,'test_fixture':False,'pairs':view}
write_verified(FOLDER/'prepared_sources'/'viewer_cases.json',encode_json(DATA))
write_verified(FOLDER/'prepared_sources'/'image_receipts.json',encode_json([image_map[k] for k in sorted(image_map)]))
write_verified(FOLDER/'prepared_sources'/'image_licenses.json',encode_json(licenses))

class ReplacementReviewStore:
    """Human-only review snapshots; source labels are never edited."""
    def __init__(self, folder, data):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.data = data
        self.ids = [p['pair_id'] for p in data['pairs']]
        self.latest = self.folder / 'latest.json'
        self.pointer_sha = digest(self.latest) if self.latest.exists() else None
        self.previous_snapshot_sha = None
        if self.latest.exists():
            pointer = json.loads(self.latest.read_text())
            if pointer['binding_sha256'] != data['binding_sha256']:
                raise IOError('Saved human review belongs to another audit.')
            name = pointer['snapshot']
            if Path(name).name != name:
                raise IOError('Invalid saved review path.')
            snapshot = self.folder / name
            if digest(snapshot) != pointer['snapshot_sha256']:
                raise IOError('Human review snapshot checksum mismatch.')
            self.state = json.loads(snapshot.read_text())
            if self.state['revision'] != pointer['revision']:
                raise IOError('Review revision mismatch.')
            self.previous_snapshot_sha = pointer['snapshot_sha256']
        else:
            self.state = {'schema': 'atml-pa3-task2-replacement-state-v1',
                'binding_sha256': data['binding_sha256'], 'run_id': data['run_id'],
                'base_export_sha256': BASE_SHA, 'batch_original_numbers': BATCH,
                'revision': 0, 'reviews': {}, 'drafts': {}, 'history': [],
                'last_reviewer': '', 'cursor': self.ids[0], 'dataset_frozen': False}
        self.validate(self.state)

    def validate(self, state):
        if (state['schema'] != 'atml-pa3-task2-replacement-state-v1' or
            state['binding_sha256'] != self.data['binding_sha256'] or
            state['run_id'] != self.data['run_id'] or state['base_export_sha256'] != BASE_SHA or
            state['batch_original_numbers'] != BATCH or state['dataset_frozen'] is not False):
            raise ValueError('Invalid human review state.')
        for pid, record in state['reviews'].items():
            if pid not in self.ids or record.get('pair_id') != pid:
                raise ValueError('Unknown review pair.')
            if record['label_status'] not in ('accept', 'ambiguous', 'annotation_error'):
                raise ValueError('Invalid judgment.')
            if record.get('review_origin') != 'human_colab' or not record['reviewer'].strip():
                raise ValueError('Human reviewer identity missing.')
            if not record.get('reviewed_at'):
                raise ValueError('Review time missing.')
            if not record['notes'].strip():
                raise ValueError('Every replacement decision requires a short explanation.')
            if record.get('affected_sides') not in ('A','B','both','neither','unclear'):
                raise ValueError('Missing affected-image field.')
            if type(record.get('ai_assisted')) is not bool:
                raise ValueError('Missing AI-assistance field.')
        if any(pid not in self.ids for pid in state['drafts']):
            raise ValueError('Unknown draft pair.')
        if state['cursor'] not in self.ids:
            raise ValueError('Unknown review cursor.')

    def check_current(self):
        observed = digest(self.latest) if self.latest.exists() else None
        if observed != self.pointer_sha:
            raise IOError('Review changed in another session. Stop and rerun this cell to reload it.')

    def commit(self, new_state, event):
        self.check_current()
        candidate = copy.deepcopy(new_state)
        candidate['revision'] = self.state['revision'] + 1
        candidate['saved_at_utc'] = utc_now()
        candidate['previous_snapshot_sha256'] = self.previous_snapshot_sha
        candidate['history'].append({'event': event, 'at_utc': candidate['saved_at_utc']})
        self.validate(candidate)
        name = f'review_state_{candidate["revision"]:06d}_{uuid.uuid4().hex[:8]}.json'
        snapshot = self.folder / name
        write_verified(snapshot, encode_json(candidate))
        snapshot_sha = digest(snapshot)
        pointer = {'binding_sha256': self.data['binding_sha256'],
            'revision': candidate['revision'], 'snapshot': name, 'snapshot_sha256': snapshot_sha}
        self.check_current()
        write_verified(self.latest, encode_json(pointer), replace=True)
        # Advance in-memory state only after the durable pointer and snapshot pass readback.
        self.pointer_sha = digest(self.latest)
        self.previous_snapshot_sha = snapshot_sha
        self.state = candidate

    def exported(self):
        self.check_current()
        judgments = []
        for pid in self.ids:
            judgments.append(self.state['reviews'].get(pid, {'pair_id': pid,
                'label_status': '', 'notes': self.state['drafts'].get(pid, {}).get('notes', ''),
                'reviewer': '', 'reviewed_at': '', 'review_origin': 'human_colab'}))
        return {'schema': 'atml-pa3-task2-replacement-review-v1',
            'run_id': self.data['run_id'], 'binding_sha256': self.data['binding_sha256'],
            'source_fingerprints': self.data['source_fingerprints'], 'base_export_sha256': BASE_SHA,
            'batch_original_numbers': BATCH, 'original_judgments': [ORIGINALS[pid] for pid in self.ids],
            'data_changes_applied': False, 'replacements_accepted_as_dataset': False,
            'approval_id': 'A007', 'plan_sha256': PLAN_SHA,
            'candidate_pool_sha256': PLAN['candidate_pool_sha256'],
            'reconciliation_export_sha256': PLAN['reconciliation_export_sha256'],
            'candidate_pairs': PLAN['candidate_pairs'], 'exported_at_utc': utc_now(),
            'reviewer': self.state['last_reviewer'], 'review_origin': 'human_colab',
            'judgments': judgments, 'history': self.state['history'],
            'complete_records': len(self.state['reviews']), 'dataset_frozen': False}

class ReplacementReviewApp:
    def __init__(self, store, data, image_folder):
        self.store, self.data, self.image_folder = store, data, image_folder
        self.pairs = data['pairs']
        self.index = next(i for i,p in enumerate(self.pairs) if p['pair_id'] == store.state['cursor'])
        self.loading = False
        self.reviewer = W.Text(value=store.state['last_reviewer'], description='Your name:',
            placeholder='Name or initials', layout=W.Layout(width='400px'))
        self.suite = W.Dropdown(options=[('All sets','all'),('Training audit','train_audit'),
            ('Development','dev'),('Test','test')], description='Set:')
        self.scope = W.Dropdown(options=[('All records','all'),('Unreviewed','pending'),
            ('Flagged','flagged'),('Accepted','accept')], description='Show:')
        self.position = W.BoundedIntText(value=self.index+1, min=1, max=len(self.pairs),
            description='Pair #:', layout=W.Layout(width='180px'))
        self.boxes = W.Checkbox(value=False, description='Show source annotation boxes', indent=False)
        self.view = W.ToggleButtons(options=[('Both images','both'),('A full size','A'),('B full size','B')])
        self.affected = W.Dropdown(options=[('Unclear / still discussing','unclear'),('Image A','A'),('Image B','B'),('Both images','both'),('Neither: targets supported','neither')], description='Issue in:')
        self.assisted = W.Checkbox(value=True, description='Discussed with assistant during this reconciliation', indent=False)
        self.notes = W.Textarea(description='Notes:', placeholder='For a flag: image A/B, what is unclear/wrong, and why.',
            layout=W.Layout(width='98%', height='90px'))
        self.top, self.message, self.images = W.HTML(), W.HTML(), W.Output()
        self.buttons = []
        def button(text, callback, style=''):
            b = W.Button(description=text, button_style=style, layout=W.Layout(width='auto'))
            b.on_click(lambda _: self.safe(callback))
            self.buttons.append(b)
            return b
        nav = W.HBox([button('Previous',lambda:self.move(-1)),button('Next / skip',lambda:self.move(1)),
            button('Next unreviewed',self.next_pending),button('Go to #',self.jump),button('Apply filters',self.apply_filters)],
            layout=W.Layout(flex_flow='row wrap'))
        decisions = W.HBox([button('Accept both targets',lambda:self.decide('accept'),'success'),
            button('Ambiguous',lambda:self.decide('ambiguous'),'warning'),
            button('Annotation error',lambda:self.decide('annotation_error'),'danger'),
            button('Save draft',self.save_draft),button('Clear judgment',self.clear_judgment)],
            layout=W.Layout(flex_flow='row wrap'))
        exports = W.HBox([button('Export progress JSON',lambda:self.export('json'),'info'),
            button('Export CSV',lambda:self.export('csv'))], layout=W.Layout(flex_flow='row wrap'))
        criteria = W.HTML("""<p><b>Replacement label review, CPU only.</b> Review both originals with boxes off first.
        Read the original saved judgment and clarification below. Discuss one case at a time in chat.
        Record your assessment and affected image(s). Accepted cases need a brief supporting note; flags need a reason.
        <b>Accept both targets</b> means both are supported; <b>Ambiguous</b> means uncertainty remains;
        <b>Annotation error</b> means a definite contradiction. Category conventions can remain unresolved.
        Decisions are saved separately from all original reviews. Nothing automatically accepts, replaces,
        relabels or freezes datasets. Notes are drafts until you press a judgment button.
        Use Save draft before a break and Export progress JSON when finished.</p>""")
        self.widget = W.VBox([criteria,self.reviewer,W.HBox([self.suite,self.scope]),
            W.HBox([self.position]),nav,self.top,W.HBox([self.boxes]),self.view,self.images,
            self.affected,self.assisted,self.notes,decisions,self.message,exports])
        self.boxes.observe(lambda _:self.render_images(), names='value')
        self.view.observe(lambda _:self.render_images(), names='value')
        self.render()

    def safe(self, callback):
        for b in self.buttons: b.disabled = True
        try:
            callback()
        except Exception as exc:
            self.message.value = '<p style="color:#b00020"><b>Action stopped:</b> '+html.escape(str(exc))+'<br>Resolve this before continuing. Previously saved snapshots are retained.</p>'
        finally:
            for b in self.buttons: b.disabled = False

    def current(self):
        return self.pairs[self.index]

    def matching(self):
        selected = []
        for i,p in enumerate(self.pairs):
            r = self.store.state['reviews'].get(p['pair_id'])
            status = r['label_status'] if r else ''
            if self.suite.value != 'all' and p['experiment_split'] != self.suite.value: continue
            if self.scope.value == 'pending' and r: continue
            if self.scope.value == 'flagged' and status not in ('ambiguous','annotation_error'): continue
            if self.scope.value == 'accept' and status != 'accept': continue
            selected.append(i)
        return selected

    def draft_state(self, destination=None):
        state = copy.deepcopy(self.store.state)
        pid = self.current()['pair_id']
        state['drafts'][pid] = {'notes': self.notes.value, 'affected_sides': self.affected.value,
            'ai_assisted': self.assisted.value, 'updated_at': utc_now()}
        state['last_reviewer'] = self.reviewer.value.strip()
        if destination is not None:
            state['cursor'] = self.pairs[destination]['pair_id']
        return state


    def go(self, destination):
        self.store.commit(self.draft_state(destination), 'navigation_and_draft_save')
        self.index = destination
        self.render()
        self.message.value = '<p>Position/notes saved and verified on Drive. Skipped records remain unreviewed.</p>'

    def move(self, direction):
        candidates = self.matching()
        if not candidates: raise ValueError('No records match these filters. Choose All records / All sets.')
        ahead = [i for i in candidates if i > self.index] if direction > 0 else [i for i in candidates if i < self.index]
        destination = (ahead[0] if ahead else candidates[0]) if direction > 0 else (ahead[-1] if ahead else candidates[-1])
        self.go(destination)

    def next_pending(self):
        candidates = [i for i,p in enumerate(self.pairs) if p['pair_id'] not in self.store.state['reviews']
            and (self.suite.value == 'all' or p['experiment_split'] == self.suite.value)]
        if not candidates: raise ValueError('No unreviewed pairs remain in this set. Export your JSON.')
        self.go(next((i for i in candidates if i > self.index), candidates[0]))

    def jump(self):
        self.go(self.position.value-1)

    def apply_filters(self):
        candidates = self.matching()
        if not candidates: raise ValueError('No matching records. Change filters; current judgment is unchanged.')
        self.go(candidates[0])

    def save_draft(self):
        self.store.commit(self.draft_state(), 'explicit_draft_save')
        self.message.value = '<p>Draft saved and verified on Drive. This does not record a judgment.</p>'

    def decide(self, status):
        name, notes = self.reviewer.value.strip(), self.notes.value.strip()
        if not name: raise ValueError('Enter your name or initials first.')
        if not notes: raise ValueError('Write a short review reason, including A/B as needed.')
        state = self.draft_state()
        pid = self.current()['pair_id']
        state['reviews'][pid] = {'pair_id':pid,'label_status':status,'notes':notes,
            'reviewer':name,'reviewed_at':utc_now(),'review_origin':'human_colab',
            'affected_sides':self.affected.value,'ai_assisted':self.assisted.value}
        state['drafts'].pop(pid,None)
        candidates = [i for i,p in enumerate(self.pairs) if p['pair_id'] not in state['reviews']
            and (self.suite.value == 'all' or p['experiment_split'] == self.suite.value)]
        destination = next((i for i in candidates if i > self.index), candidates[0] if candidates else self.index)
        state['cursor'] = self.pairs[destination]['pair_id']
        self.store.commit(state, f'judgment:{pid}:{status}')
        self.index = destination
        self.render()
        self.message.value = '<p><b>Judgment saved and verified on Drive.</b> '+('Next unreviewed pair shown.' if candidates
            else 'No unreviewed pair remains in this set. Export JSON; flags will be discussed before dataset changes.')+'</p>'

    def clear_judgment(self):
        state = self.draft_state()
        state['reviews'].pop(self.current()['pair_id'],None)
        self.store.commit(state,'clear_judgment:'+self.current()['pair_id'])
        self.render()
        self.message.value = '<p>Judgment cleared; earlier snapshots retained. Pair is unreviewed again.</p>'

    def render(self):
        self.loading = True
        p = self.current(); pid = p['pair_id']; state = self.store.state
        record = state['reviews'].get(pid, {})
        draft = state['drafts'].get(pid, record)
        self.position.value = self.index+1
        self.notes.value = draft.get('notes', '')
        self.affected.value = draft.get('affected_sides', 'unclear')
        self.assisted.value = draft.get('ai_assisted', True)
        number = ORIGINAL_NUMBER[pid]
        original = ORIGINALS[pid]
        esc = html.escape
        self.top.value = (f'<h3>Replacement {self.index+1}/{len(self.pairs)} | Original pair #{number}</h3>'
            f'<p><b>{esc(p["question"])}</b> | {esc(p["experiment_split"])}</p>'
            f'<p><b>Replaced case: original human judgment:</b> {esc(original["label_status"])}<br>'
            f'<b>Reason for replacing old case:</b> {esc(original["notes"])}</p>'
            f'<p><b>Clarify:</b> {esc(QUESTIONS[number])}</p>'
            f'<p>Replacement saved assessment: <b>{esc(record.get("label_status", "NOT RECORDED"))}</b>'
            f' | Completed {len(state["reviews"])}/{len(self.pairs)}</p><small>{esc(pid)}</small>')
        self.boxes.value = False
        self.view.value = 'both'
        self.loading = False
        self.render_images()


    def render_images(self):
        if self.loading: return
        panels = []
        for j,ex in enumerate(self.current()['examples']):
            label = 'AB'[j]
            if self.view.value not in ('both',label): continue
            raw = (self.image_folder/ex['src']).read_bytes()
            uri = 'data:image/jpeg;base64,'+base64.b64encode(raw).decode()
            boxes = ''
            if self.boxes.value:
                for b in ex['boxes']:
                    x,y,w,h = b['bbox']
                    boxes += f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="none" stroke="#ffb000" stroke-width="2"/>'
            overlay = f'<svg viewBox="0 0 {ex["width"]} {ex["height"]}" style="position:absolute;inset:0;width:100%;height:100%;pointer-events:none">{boxes}</svg>'
            width = '48%' if self.view.value == 'both' else f'{ex["width"]}px'
            panels.append(f'<div style="width:{width};max-width:100%;flex-shrink:0"><p><b>Image {label} · {ex["image_id"]} · Target: {html.escape(str(ex["target"]))}</b></p>'+
                f'<div style="position:relative"><img src="{uri}" style="display:block;width:100%;height:auto">{overlay}</div></div>')
        with self.images:
            clear_output(wait=True)
            display(HTML('<div style="display:flex;gap:15px;align-items:flex-start;flex-wrap:wrap">'+''.join(panels)+'</div>'))

    def export(self, kind):
        self.save_draft()
        exported = self.store.exported()
        suffix = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_')+uuid.uuid4().hex[:6]
        json_path = self.store.folder/f'replacement_review_A007_{suffix}.json'
        write_verified(json_path, encode_json(exported))
        if kind == 'csv':
            stream = io.StringIO(newline='')
            columns = ['pair_id','label_status','notes','reviewer','reviewed_at','review_origin','affected_sides','ai_assisted']
            writer = csv.DictWriter(stream,fieldnames=columns,extrasaction='ignore')
            writer.writeheader(); writer.writerows(exported['judgments'])
            target = json_path.with_suffix('.csv')
            write_verified(target,stream.getvalue().encode())
        else: target = json_path
        files.download(str(target))
        self.message.value = '<p>Export saved and verified on Drive, then download requested: '+html.escape(str(target)) + \
            '<br>Return the JSON to this chat. Dataset remains unfrozen.</p>'

    def close(self):
        # Preserve any unsent notes before replacing controls on a cell rerun.
        self.store.commit(self.draft_state(),'close_previous_controls_and_save_draft')
        self.widget.close()
        for b in self.buttons: b.disabled = True


STORE_A007=ReplacementReviewStore(FOLDER/'human_review',DATA)
setup_log=FOLDER/('setup_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_')+uuid.uuid4().hex[:8]+'.json')
write_verified(setup_log,encode_json({'status':'interface_prepared','approval_id':'A007','plan_sha256':PLAN_SHA,
    'binding_sha256':BINDING,'candidate_pool_sha256':PLAN['candidate_pool_sha256'],
    'candidate_pairs':86,'images_verified':len(image_map),'image_activities':dict(activities),
    'image_preparation_seconds':round(time.monotonic()-start,3),'restored_human_decisions':len(STORE_A007.state['reviews']),
    'dataset_frozen':False,'replacements_accepted_as_dataset':False,'original_selected_manifests_modified':False,
    'gpu_used':False,'weights_loaded':False}))
A007_REPLACEMENT_APP=ReplacementReviewApp(STORE_A007,DATA,IMAGE_FOLDER)
display(A007_REPLACEMENT_APP.widget)
print('BLOCK 6G READY — review 86 replacement pairs; new decisions start blank or resume only this source-bound store.')
print('Completed decisions:',len(STORE_A007.state['reviews']),'/86')
print('Current original pair:',ORIGINAL_NUMBER[A007_REPLACEMENT_APP.current()['pair_id']])
print('Save folder:',STORE_A007.folder)
print('Plan SHA256:',PLAN_SHA)
print('Binding SHA256:',BINDING)
print('Setup log:',setup_log)
print('Setup log SHA256:',digest(setup_log))
print('Keep accelerator None. Return full output; export progress JSON after review.')
