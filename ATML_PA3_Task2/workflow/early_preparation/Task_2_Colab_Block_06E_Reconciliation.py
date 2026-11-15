# Block 6E: seven-case human reconciliation inside Colab. CPU only.
# Source images/labels stay unchanged. Every saved judgment is verified on Drive.
import os, json, hashlib, shutil, zipfile, uuid, traceback, base64, html, csv, io, copy
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter

if os.environ.get('COLAB_GPU', '0') not in ('', '0'):
    from google.colab import runtime
    try:
        runtime.unassign()
    finally:
        raise RuntimeError('This is CPU review. Reconnect with accelerator None.')

from google.colab import drive, output, files
from IPython.display import display, HTML, clear_output
import ipywidgets as W
output.enable_custom_widget_manager()
if not Path('/content/drive/MyDrive').exists():
    drive.mount('/content/drive')

ROOT = Path('/content/drive/MyDrive/ATML_PA3/task2/run_20261031_150949_1cff148a')
LOCAL = Path('/content/atml_pa3_task2/run_20261031_150949_1cff148a')
BINDING = 'a39f40346d6aa4baba790316810d9be1e99a72bc9891c7ed36391e53e2cb6c49'
ZIP_SHA = 'f45af20228069533c3b37c0169dba7426c281750278ed3726832a05e0ca1b05a'
ZIP_NAME = 'annotation_audit_a39f40346d6aa4ba.zip'

# Close an earlier instance to prevent accidental writes from old controls.
previous = globals().get('RECONCILIATION_APP')
if previous is not None:
    previous.close()


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


# Verify the exact bundle, restore files locally, and check every member hash.
local_zip = LOCAL / 'human_review_source' / ZIP_NAME
if not local_zip.exists():
    local_zip.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / 'audits' / ZIP_NAME, local_zip)
if digest(local_zip) != ZIP_SHA:
    raise IOError('Audit ZIP checksum mismatch; existing files preserved.')
bundle = LOCAL / 'human_review_source' / 'extracted'
bundle.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(local_zip) as archive:
    if archive.testzip() is not None:
        raise IOError('Audit ZIP CRC check failed.')
    for item in archive.infolist():
        target = (bundle / item.filename).resolve()
        if not target.is_relative_to(bundle.resolve()):
            raise IOError('Unsafe ZIP member path.')
        if not item.is_dir():
            write_verified(target, archive.read(item))
manifest = json.loads((bundle / 'bundle_manifest.json').read_text())
for item in manifest['files']:
    path = bundle / item['path']
    if path.stat().st_size != item['bytes'] or digest(path) != item['sha256']:
        raise IOError(f'Bundle file mismatch: {item["path"]}')
DATA = json.loads((bundle / 'data/viewer_cases.json').read_text())
assert DATA['binding_sha256'] == BINDING and DATA['run_id'] == ROOT.name
assert len(DATA['pairs']) == 246
BASE_SHA = '5c48607823e9dc4d319e311e74041d6435a04dd7c4f831d9cc5f3ca0a4e25e86'
BATCH = [66, 91, 110, 117, 229, 238, 245]
FOLDER = ROOT/'audits'/'reconciliation_batch01_a39f40346d6aa4ba'
original_exports = ROOT/'audits'/'human_review_a39f40346d6aa4ba'
raw = None
for candidate in original_exports.glob('human_review_*.json'):
    if digest(candidate) == BASE_SHA:
        raw = candidate.read_bytes()
        break
if raw is None:
    print('Upload your original human_review_20261104_052236_a82460.json export.')
    uploads = files.upload()
    matches = [value for value in uploads.values() if hashlib.sha256(value).hexdigest() == BASE_SHA]
    if len(matches) != 1:
        raise ValueError('Expected exactly one export matching the verified original SHA-256.')
    raw = matches[0]
BASE = json.loads(raw)
assert BASE['schema'] == 'atml-pa3-task2-annotation-review-v1'
for key in ('run_id','binding_sha256','source_fingerprints'):
    assert BASE[key] == DATA[key], key
assert BASE['review_origin'] == 'human_colab' and BASE['dataset_frozen'] is False
assert BASE['complete_records'] == len(BASE['judgments']) == 246
ORIGINALS = {r['pair_id']: r for r in BASE['judgments']}
assert len(ORIGINALS) == 246
assert set(ORIGINALS) == {p['pair_id'] for p in DATA['pairs']}
for r in ORIGINALS.values():
    assert r['label_status'] in ('accept','ambiguous','annotation_error')
    assert r['review_origin'] == 'human_colab' and r['reviewer'].strip() and r['reviewed_at']
    assert r['label_status'] == 'accept' or r['notes'].strip()
write_verified(FOLDER/'source'/'original_review.json', raw)
ORIGINAL_NUMBER = {p['pair_id']: i for i,p in enumerate(DATA['pairs'],1)}
DATA = copy.deepcopy(DATA)
DATA['pairs'] = [DATA['pairs'][i-1] for i in BATCH]
QUESTIONS = {
    66: 'The question is about a remote, but the original note says mouse. What can you identify in A?',
    91: 'Does A definitely show another object, or is the distant object too unclear to identify?',
    110: 'Can you identify two distinct frisbees in A? Explain whether the issue is definite or uncertain.',
    117: 'Targets are A=1 and B=3. The note says B has 2, not 1. Which image/count did you intend?',
    229: 'What counting unit is visually supportable? B target is 3; more than one alone does not contradict it.',
    238: 'Do the original suitcase targets remain supported? Explain the repurposed suitcase bodies in B.',
    245: 'Is the concern the horse sculpture convention, a definite count error, or both? Do not invent a global rule.'
}



class ReviewStore:
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
            self.state = {'schema': 'atml-pa3-task2-reconciliation-state-v1',
                'binding_sha256': data['binding_sha256'], 'run_id': data['run_id'],
                'base_export_sha256': BASE_SHA, 'batch_original_numbers': BATCH,
                'revision': 0, 'reviews': {}, 'drafts': {}, 'history': [],
                'last_reviewer': '', 'cursor': self.ids[0], 'dataset_frozen': False}
        self.validate(self.state)

    def validate(self, state):
        if (state['schema'] != 'atml-pa3-task2-reconciliation-state-v1' or
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
                raise ValueError('Every reconciliation decision requires a short explanation.')
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
        return {'schema': 'atml-pa3-task2-reconciliation-v1',
            'run_id': self.data['run_id'], 'binding_sha256': self.data['binding_sha256'],
            'source_fingerprints': self.data['source_fingerprints'], 'base_export_sha256': BASE_SHA,
            'batch_original_numbers': BATCH, 'original_judgments': [ORIGINALS[pid] for pid in self.ids],
            'data_changes_applied': False, 'exported_at_utc': utc_now(),
            'reviewer': self.state['last_reviewer'], 'review_origin': 'human_colab',
            'judgments': judgments, 'history': self.state['history'],
            'complete_records': len(self.state['reviews']), 'dataset_frozen': False}


class HumanReviewApp:
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
        criteria = W.HTML("""<p><b>Seven-case reconciliation, CPU only.</b> Review both originals with boxes off first.
        Read the original saved judgment and clarification below. Discuss one case at a time in chat.
        Record your final assessment with a short reason and the affected image(s).
        <b>Accept both targets</b> means both are supported; <b>Ambiguous</b> means uncertainty remains;
        <b>Annotation error</b> means a definite contradiction. Category conventions can remain unresolved.
        Decisions are saved separately from the original 246-pair review. Nothing filters, replaces,
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
        if not notes: raise ValueError('Write a short reconciliation reason, including A/B as needed.')
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
        self.top.value = (f'<h3>Reconciliation {self.index+1}/7 | Original pair #{number}</h3>'
            f'<p><b>{esc(p["question"])}</b> | {esc(p["experiment_split"])}</p>'
            f'<p><b>Original human judgment:</b> {esc(original["label_status"])}<br>'
            f'<b>Original note:</b> {esc(original["notes"])}</p>'
            f'<p><b>Clarify:</b> {esc(QUESTIONS[number])}</p>'
            f'<p>New saved assessment: <b>{esc(record.get("label_status", "NOT RECORDED"))}</b>'
            f' | Completed {len(state["reviews"])}/7</p><small>{esc(pid)}</small>')
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
        json_path = self.store.folder/f'reconciliation_batch01_{suffix}.json'
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


# Retain the executed cell and setup evidence for the later report.
provided_source = globals().get('reconciliation_cell_bytes')
cell_sha = hashlib.sha256(provided_source).hexdigest() if isinstance(provided_source, bytes) else None
if cell_sha is not None:
    write_verified(ROOT/'code'/f'reconciliation_batch01_cell_{cell_sha[:16]}.py', provided_source)
STORE = ReviewStore(FOLDER, DATA)
setup_log = ROOT/'logs'/('block_06e_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_')+uuid.uuid4().hex[:8]+'.json')
write_verified(setup_log,encode_json({'status':'interface_prepared','review_origin':'human_colab',
    'binding_sha256':BINDING,'source_bundle_sha256':ZIP_SHA,'cell_sha256':cell_sha,
    'reconciliation_judgments_restored':len(STORE.state['reviews']), 'assistant_judgments_imported':0,
    'gpu_used':False,'model_weights_loaded':False,'dataset_frozen':False,
    'reconciliation_complete':len(STORE.state['reviews'])==7, 'base_export_sha256':BASE_SHA, 'batch_original_numbers':BATCH,'at_utc':utc_now()}))
RECONCILIATION_APP = HumanReviewApp(STORE, DATA, bundle)
display(RECONCILIATION_APP.widget)
print('BLOCK 6E READY — seven-case reconciliation; original review preserved.')
print('Restored saved reconciliation decisions:',len(STORE.state['reviews']),'/7')
print('Separate reconciliation folder:',STORE.folder)
print('Current original pair:',ORIGINAL_NUMBER[RECONCILIATION_APP.current()['pair_id']])
print('Keep accelerator None. Enter your name, inspect both images, then use a judgment button.')
print('Before a break: Save draft. When finished: Export progress JSON and attach it here.')

print('Setup log saved and read back:', setup_log)
print('Setup log SHA256:', digest(setup_log))
