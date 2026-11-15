# Block 6C: portable annotation-audit bundle. CPU only; no judgments imported.
import os, json, uuid, shutil, traceback, hashlib, zipfile, time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

needed = ('ROOT', 'LOCAL', 'save_record', 'file_sha', 'copy_checked')
if any(n not in globals() for n in needed):
    raise RuntimeError('Notebook state missing. Stop; do not rerun Block 1.')
if os.environ.get('COLAB_GPU', '0') not in ('', '0'):
    from google.colab import runtime
    try:
        runtime.unassign()
    finally:
        raise RuntimeError('Reconnect with accelerator None.')
stamp = datetime.now(ZoneInfo('Asia/Karachi')).strftime('%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:8]
started = time.monotonic()
report6c = {'attempt': stamp, 'status': 'started', 'gpu_used': False,
            'visual_reviews_imported': 0, 'complete_dataset_frozen': False,
            'model_weights_loaded': False, 'visual_review_complete': False}

HTML_TEMPLATE = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Task 2 · Annotation audit</title>
<style>
:root{font-family:system-ui,sans-serif;color:#192b3b;background:#eef2f5}*{box-sizing:border-box}body{margin:0}header{position:sticky;top:0;z-index:5;background:#fff;border-bottom:1px solid #cbd5df;padding:14px max(16px,calc((100vw - 1160px)/2))}h1{font-size:22px;margin:0 0 6px}p{line-height:1.5;margin:6px 0}button,select,input,textarea{font:inherit;border:1px solid #9cacba;border-radius:7px;padding:8px;background:white;color:inherit}button{cursor:pointer}button.primary{background:#155b87;color:#fff;border-color:#155b87}button:disabled{opacity:.5;cursor:default}.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-top:9px}.bar input[type=text]{min-width:180px;flex:1}#progress{font-weight:650}#notice{color:#9b310e;font-size:13px}.small{font-size:13px;color:#50616e}main{max-width:1160px;margin:auto;padding:16px}.card{background:#fff;border:1px solid #cbd5df;border-radius:12px;padding:16px;margin:0 0 20px}.card h2{font-size:19px;margin:7px 0 12px}.photos{display:grid;grid-template-columns:1fr 1fr;gap:14px}.photo{min-width:0}.stage{position:relative;background:#ddd}.stage img{display:block;width:100%;height:auto;image-orientation:none}.stage svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none;display:none}.show-boxes .stage svg{display:block}.target{font-weight:700;margin:8px 0}.decision{display:grid;grid-template-columns:220px 1fr;gap:12px;margin-top:15px}.decision label{display:flex;flex-direction:column;gap:5px}.decision textarea{width:100%;min-height:70px}.status-accept{border-left:5px solid #238354}.status-ambiguous,.status-annotation_error{border-left:5px solid #bb7816}.pager{display:flex;justify-content:space-between;align-items:center;gap:12px;margin:14px 0}.pill{display:inline-block;background:#edf2f6;padding:3px 7px;border-radius:5px;font-size:12px}.empty{padding:30px;background:white;border-radius:10px}a{color:#155b87}#fixture{color:#9b310e;font-weight:bold}@media(max-width:720px){header{position:static}.photos,.decision{grid-template-columns:1fr}.bar input[type=text]{min-width:130px}main{padding:10px}.card{padding:12px}}
</style></head><body><header><h1>Task 2 · Annotation audit</h1><div id="fixture"></div><p>Inspect the original images. Boxes show existing annotations and may omit objects. Record what the images support; no model predictions are shown.</p><div class="bar"><label>Reviewer <input id="reviewer" type="text" placeholder="Name or initials"></label><button id="boxes">Show annotation boxes</button><button id="exportJson" class="primary">Export review JSON</button><button id="exportCsv">Export CSV</button><button id="load">Load review JSON</button><button id="undo" disabled>Undo last load</button><input id="loadFile" type="file" accept="application/json,.json" hidden></div><div class="bar"><select id="suite"><option value="all">All sets</option><option value="train_audit">Training audit</option><option value="dev">Development</option><option value="test">Final test</option></select><select id="kind"><option value="all">Both question types</option><option value="presence">Presence</option><option value="counting">Counting</option></select><select id="status"><option value="all">All review states</option><option value="pending">Unreviewed / incomplete</option><option value="accept">Accept</option><option value="ambiguous">Ambiguous</option><option value="annotation_error">Annotation error</option></select><input id="search" type="text" placeholder="Find category, question, or pair ID"></div><p id="progress"></p><p id="notice" aria-live="polite"></p><p class="small">Export JSON before closing or switching browsers. Flagged cases need notes; dataset changes are handled separately after review.</p></header><main><div class="pager"><button id="prev">Previous page</button><span id="page"></span><button id="next">Next page</button></div><div id="cards"></div></main>
<script type="application/json" id="audit-data">__AUDIT_DATA__</script>
<script>
'use strict';
const DATA=JSON.parse(document.getElementById('audit-data').textContent), $=id=>document.getElementById(id), KEY='atml-pa3-audit:'+DATA.binding_sha256, VALID=new Set(['','accept','ambiguous','annotation_error']);
const escapeHtml=value=>String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const blank=()=>({reviewer:'',reviews:Object.fromEntries(DATA.pairs.map(p=>[p.pair_id,{label_status:'',notes:'',reviewer:'',reviewed_at:''}])),history:[]});
let state=blank(),page=0,backup=null,storageOK=true;
function validated(value){const next=blank();if(!value||typeof value!=='object')throw Error('Invalid progress object.');next.reviewer=String(value.reviewer||'');next.history=Array.isArray(value.history)?value.history:[];for(const [id,j] of Object.entries(value.reviews||{})){if(!Object.prototype.hasOwnProperty.call(next.reviews,id)||!j||!VALID.has(j.label_status))throw Error('Unknown pair or judgment.');for(const field of ['notes','reviewer','reviewed_at'])if(typeof j[field]!=='string')throw Error('Invalid review field.');next.reviews[id]={label_status:j.label_status,notes:j.notes,reviewer:j.reviewer,reviewed_at:j.reviewed_at};}return next;}
try{const saved=localStorage.getItem(KEY);if(saved)state=validated(JSON.parse(saved));}catch(e){storageOK=false;$('notice').textContent='Saved browser progress could not be loaded. Use a previously exported JSON file.';}
$('reviewer').value=state.reviewer;
if(DATA.test_fixture)$('fixture').textContent='UI TEST FIXTURE — no experiment data';
function persist(){try{localStorage.setItem(KEY,JSON.stringify(state));}catch(e){storageOK=false;$('notice').textContent='Browser storage unavailable. Export JSON to preserve your progress.';}}
const complete=j=>Boolean(j.label_status&&j.reviewer.trim()&&j.reviewed_at&&(!['ambiguous','annotation_error'].includes(j.label_status)||j.notes.trim()));
function progress(){const js=Object.values(state.reviews),done=js.filter(complete).length,flagged=js.filter(j=>['ambiguous','annotation_error'].includes(j.label_status)).length;$('progress').textContent=`Complete records: ${done}/${DATA.pairs.length} · Flagged: ${flagged} · Pending/incomplete: ${DATA.pairs.length-done}`;}
function filtered(){const term=$('search').value.toLowerCase();return DATA.pairs.filter(p=>($('suite').value==='all'||p.experiment_split===$('suite').value)&&($('kind').value==='all'||p.question_type===$('kind').value)&&($('status').value==='all'||($('status').value==='pending'?!complete(state.reviews[p.pair_id]):state.reviews[p.pair_id].label_status===$('status').value))&&(!term||`${p.category_name} ${p.question} ${p.pair_id}`.toLowerCase().includes(term)));}
function photo(ex){const boxes=ex.boxes.map((b,i)=>{const [x,y,w,h]=b.bbox,sw=Math.max(1.5,Math.min(ex.width,ex.height)/160);return `<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="none" stroke="#ffbd36" stroke-width="${sw}" ${b.iscrowd?'stroke-dasharray="8 4"':''}/><text x="${x+3}" y="${Math.max(14,y+15)}" fill="#ffdf7b" stroke="#111" stroke-width=".6" font-size="15">${i+1}</text>`;}).join('');return `<div class="photo"><div class="stage"><img loading="lazy" src="${escapeHtml(ex.src)}" width="${ex.width}" height="${ex.height}" alt="COCO image ${ex.image_id}"><svg viewBox="0 0 ${ex.width} ${ex.height}" aria-hidden="true">${boxes}</svg></div><p class="target">Annotation target: ${escapeHtml(ex.target)}</p><p class="small">Image ${ex.image_id} · ${ex.width}×${ex.height} · ${escapeHtml(ex.mode)} · ${ex.boxes.length} boxes (${ex.boxes.filter(b=>b.iscrowd).length} crowd)</p><a href="${escapeHtml(ex.src)}" target="_blank" rel="noopener">Open original at full size</a></div>`;}
function render(){const rows=filtered(),size=10,max=Math.max(1,Math.ceil(rows.length/size));page=Math.max(0,Math.min(page,max-1));$('page').textContent=`Page ${page+1}/${max} · ${rows.length} matching pairs`;$('prev').disabled=page===0;$('next').disabled=page>=max-1;$('cards').innerHTML=rows.slice(page*size,(page+1)*size).map(p=>{const j=state.reviews[p.pair_id];return `<article class="card status-${j.label_status||'pending'}" data-id="${escapeHtml(p.pair_id)}"><div class="pill">${escapeHtml(p.experiment_split)} · ${escapeHtml(p.question_type)} · ${escapeHtml(p.category_name)}</div><h2>${escapeHtml(p.question)}</h2><div class="photos">${p.examples.map(photo).join('')}</div><div class="decision"><label>Judgment<select class="judgment"><option value="" ${!j.label_status?'selected':''}>Unreviewed</option>${[['accept','Accept'],['ambiguous','Ambiguous'],['annotation_error','Annotation error']].map(([v,t])=>`<option value="${v}" ${j.label_status===v?'selected':''}>${t}</option>`).join('')}</select></label><label>Notes<textarea class="notes" placeholder="Explain ambiguity or incorrect/missing annotations.">${escapeHtml(j.notes)}</textarea></label></div><p class="small">${escapeHtml(p.pair_id)}</p></article>`;}).join('')||'<p class="empty">No pairs match these filters.</p>';progress();}
function edit(id,patch,recordHistory){const old=state.reviews[id],next={...old,...patch},reviewer=$('reviewer').value.trim();if(next.label_status&&!reviewer){$('notice').textContent='Enter your name or initials before recording a judgment.';$('reviewer').focus();return false;}next.reviewer=next.label_status?reviewer:'';next.reviewed_at=next.label_status?new Date().toISOString():'';state.reviews[id]=next;state.reviewer=$('reviewer').value;if(recordHistory)state.history.push({pair_id:id,...next,event_at:new Date().toISOString()});persist();progress();return true;}
$('cards').addEventListener('input',e=>{if(e.target.classList.contains('notes')){const id=e.target.closest('[data-id]').dataset.id;edit(id,{notes:e.target.value},false);}});
$('cards').addEventListener('change',e=>{const card=e.target.closest('[data-id]');if(!card)return;const id=card.dataset.id;if(e.target.classList.contains('judgment')){if(edit(id,{label_status:e.target.value},true))render();else e.target.value=state.reviews[id].label_status;}if(e.target.classList.contains('notes')){if(edit(id,{notes:e.target.value},true))render();}});
$('reviewer').addEventListener('input',()=>{state.reviewer=$('reviewer').value;persist();});
for(const id of ['suite','kind','status'])$(id).addEventListener('change',()=>{page=0;render();});$('search').addEventListener('input',()=>{page=0;render();});
$('prev').onclick=()=>{page--;render();window.scrollTo(0,0);};$('next').onclick=()=>{page++;render();window.scrollTo(0,0);};
$('boxes').onclick=()=>{const shown=document.body.classList.toggle('show-boxes');$('boxes').textContent=shown?'Hide annotation boxes':'Show annotation boxes';};
function makeExport(data,reviewState,reviewer,now){return {schema:'atml-pa3-task2-annotation-review-v1',run_id:data.run_id,binding_sha256:data.binding_sha256,source_fingerprints:data.source_fingerprints,exported_at_utc:now,reviewer,judgments:data.pairs.map(p=>({pair_id:p.pair_id,...reviewState.reviews[p.pair_id]})),history:reviewState.history,complete_records:Object.values(reviewState.reviews).filter(complete).length,dataset_frozen:false};}
function exported(){return makeExport(DATA,state,$('reviewer').value,new Date().toISOString());}
function buildCSV(data,reviewState){const columns=['pair_id','source_split','experiment_split','question_type','category_id','category_name','question','image_0','target_0','image_1','target_1','label_status','notes','reviewer','reviewed_at'];const quote=v=>'"'+String(v??'').replace(/"/g,'""')+'"';const rows=data.pairs.map(p=>({...p,image_0:p.examples[0].image_id,target_0:p.examples[0].target,image_1:p.examples[1].image_id,target_1:p.examples[1].target,...reviewState.reviews[p.pair_id]}));return [columns.map(quote).join(','),...rows.map(r=>columns.map(k=>quote(r[k])).join(','))].join('\r\n');}
function stateFromExport(data){if(data.schema!=='atml-pa3-task2-annotation-review-v1'||data.binding_sha256!==DATA.binding_sha256||data.run_id!==DATA.run_id)throw Error('This JSON belongs to a different audit.');if(!Array.isArray(data.judgments))throw Error('Judgments are missing.');const reviews=Object.create(null),seen=new Set(),ids=new Set(DATA.pairs.map(p=>p.pair_id));for(const j of data.judgments){if(!ids.has(j.pair_id)||seen.has(j.pair_id))throw Error('Unknown or duplicate pair ID.');seen.add(j.pair_id);reviews[j.pair_id]=j;}return validated({reviewer:data.reviewer,reviews,history:data.history});}

function download(text,name,type){const url=URL.createObjectURL(new Blob([text],{type})),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
const stamp=()=>new Date().toISOString().replace(/[:.]/g,'-');
$('exportJson').onclick=()=>download(JSON.stringify(exported(),null,2),'annotation_review_'+DATA.run_id+'_'+stamp()+'.json','application/json');
$('exportCsv').onclick=()=>download(buildCSV(DATA,state),'annotation_review_'+DATA.run_id+'_'+stamp()+'.csv','text/csv;charset=utf-8');
$('load').onclick=()=>$('loadFile').click();$('loadFile').onchange=async e=>{const file=e.target.files[0];if(!file)return;try{const data=JSON.parse(await file.text()),next=stateFromExport(data);backup=JSON.parse(JSON.stringify(state));state=next;$('reviewer').value=state.reviewer;$('undo').disabled=false;persist();page=0;render();$('notice').textContent='Saved review JSON loaded. Undo last load restores the previous local progress.';}catch(error){$('notice').textContent='Load failed: '+error.message;}e.target.value='';};
$('undo').onclick=()=>{if(backup){state=backup;backup=null;$('reviewer').value=state.reviewer;$('undo').disabled=true;persist();render();$('notice').textContent='Previous local progress restored.';}};
render();
</script></body></html>
'''

def checked_json(relative, expected):
    path = ROOT / relative
    if file_sha(path) != expected:
        raise IOError(f'Fingerprint mismatch: {path}')
    return json.loads(path.read_text())

def put_bytes(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise IOError(f'Existing different file preserved: {path}')
        return
    temporary = path.with_name(path.name + '.part')
    temporary.write_bytes(data)
    if temporary.read_bytes() != data:
        raise IOError('Local write/readback mismatch.')
    temporary.replace(path)

def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + '\n').encode()

def make_zip(destination, entries):
    # Fixed timestamps and ordering make reruns reproducible. JPEGs stay unchanged.
    temporary = destination.with_name(destination.name + '.part')
    with zipfile.ZipFile(temporary, 'w') as archive:
        for name, source in sorted(entries.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED if name.endswith('.jpg') else zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            with source.open('rb') as incoming, archive.open(info, 'w') as outgoing:
                shutil.copyfileobj(incoming, outgoing)
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None or set(archive.namelist()) != set(entries):
            raise IOError('ZIP verification failed.')
        if len(archive.namelist()) != len(entries):
            raise IOError('Duplicate ZIP members.')
    if destination.exists():
        if file_sha(destination) != file_sha(temporary):
            raise IOError('Existing different ZIP preserved.')
        temporary.unlink()
    else:
        temporary.replace(destination)

try:
    old6a = checked_json('logs/block_06a_20261031_163930_3cb996af.json',
        '44a121f200cfd59ec38771010438328e8aaa35482989c5843ddd54c42ddbf13a')
    old6b = checked_json('logs/block_06b_20261031_165402_a8bf6a6a.json',
        '3fef9afee4e36666a7ba80da674fd6b86fb0381a8d019adbee750044b6f81295')
    assert old6a['status'] == old6b['status'] == 'passed'
    assert old6b['images_verified'] == 488 and not old6b['nonstandard_exif_orientation_images']
    sources = {
        'configs/exact_protocol_A005.json': old6a['approved_protocol_sha256'],
        'audits/annotation_audit_plan.json': old6a['audit_plan_sha256'],
        'audits/annotation_audit_review_template.csv': old6a['audit_review_template_sha256'],
        'audits/annotation_box_index.json': old6b['audit_box_index_sha256'],
        'audits/annotation_audit_image_manifest.json': old6b['image_manifest_sha256']}
    for relative, digest in sources.items():
        if file_sha(ROOT / relative) != digest:
            raise IOError(f'Changed audit source: {relative}')
    box_index = json.loads((ROOT / 'audits/annotation_box_index.json').read_text())
    images = json.loads((ROOT / 'audits/annotation_audit_image_manifest.json').read_text())
    image_map = {(r['source_split'], int(r['image_id'])): r for r in images}
    assert len(images) == len(image_map) == 488
    pairs = []
    for name in ('grounding_training_audit.jsonl', 'grounding_dev.jsonl', 'grounding_test.jsonl'):
        item = next(r for r in old6a['outputs'] if r['path'].endswith('/' + name))
        path = ROOT / 'manifests/selected' / name
        if file_sha(path) != item['sha256']:
            raise IOError(f'Changed pair manifest: {name}')
        sources['manifests/selected/' + name] = item['sha256']
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        assert len(rows) == item['rows']
        pairs.extend(rows)
    assert len(pairs) == len({r['pair_id'] for r in pairs}) == 246
    fingerprints = {**sources, 'viewer_template_sha256': hashlib.sha256(HTML_TEMPLATE.encode()).hexdigest()}
    binding = hashlib.sha256(json.dumps({'run_id': ROOT.name, 'sources': fingerprints},
        sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    folder = LOCAL / 'audit_bundles' / binding[:16]
    folder.mkdir(parents=True, exist_ok=True)
    entries, records = {}, []
    for i, r in enumerate(images, 1):
        split, filename = r['source_split'], r['file_name']
        assert split in ('train2014', 'val2014') and Path(filename).name == filename
        local = LOCAL / 'images' / split / filename
        if not local.exists():
            copy_checked(ROOT / 'raw_images' / split / filename, local)
        if local.stat().st_size != r['bytes'] or file_sha(local) != r['sha256']:
            raise IOError(f'Original image mismatch: {local}')
        name = f'images/{split}/{filename}'
        entries[name] = local
        records.append({'path': name, 'sha256': r['sha256'], 'bytes': r['bytes']})
        if i % 100 == 0 or i == len(images):
            print(f'Bundle originals checked: {i}/488', flush=True)
    viewer_pairs = []
    used = set()
    for pair in pairs:
        view = {k: pair[k] for k in ('pair_id', 'source_split', 'experiment_split',
                'question_type', 'category_id', 'category_name', 'question')}
        view['examples'] = []
        assert len(pair['examples']) == 2
        for ex in pair['examples']:
            key = (pair['source_split'], int(ex['image_id']))
            r = image_map[key]
            meta = r['decoded_metadata']
            assert meta['exif_orientation'] == 1
            used.add(key)
            view['examples'].append({'image_id': key[1], 'target': ex['target'],
                'src': f"images/{key[0]}/{r['file_name']}",
                'width': meta['width'], 'height': meta['height'], 'mode': meta['mode'],
                'boxes': box_index['boxes_by_split_image_category'][
                    f"{key[0]}|{key[1]}|{pair['category_id']}"]})
        viewer_pairs.append(view)
    assert used == set(image_map)
    payload = {'schema': 'atml-pa3-task2-audit-viewer-v1', 'run_id': ROOT.name,
               'binding_sha256': binding, 'source_fingerprints': fingerprints,
               'test_fixture': False, 'pairs': viewer_pairs}
    # Escape '<' inside inline JSON, including any accidental closing-script text.
    embedded = json.dumps(payload, ensure_ascii=False).replace('<', '\u003c')
    assert HTML_TEMPLATE.count('__AUDIT_DATA__') == 1
    html = HTML_TEMPLATE.replace('__AUDIT_DATA__', embedded)
    readme = f"""ATML PA3 Task 2: annotation audit (not yet reviewed)
Run: {ROOT.name}
Binding: {binding}

Extract the entire ZIP, then open index.html in your browser.
Keep index.html alongside the images and data folders. No server or internet required.
Enter your name/initials; inspect both original images and the question/targets.
Boxes start hidden. Inspect the images first; toggle boxes to check annotations.
Accept means BOTH target answers are supported by the images.
Ambiguous means either image does not support a confident judgment.
Annotation error means a target is visibly wrong or annotations omit relevant objects.
Use notes to identify the affected image and explain every flagged pair.
Open original at full size when needed. No defaults mark a pair accepted.

Browser progress is convenient but not a durable experiment record.
Export review JSON regularly and before closing, changing browsers, or moving this folder.
Load review JSON restores an export for this exact audit; Undo last load restores the prior state.
CSV is an additional readable export. JSON is the file to return for validation/import.
Exported files go to your browser's download folder, NOT automatically to Google Drive.

246 pairs = 158 training audit +24 dev +64 test; 488 unchanged original JPEGs.
Two JPEGs are grayscale. No image has nonstandard EXIF orientation.
Labels are based on source annotations and have NOT been visually verified yet.
No model predictions appear here. Do not change labels, replace cases, or freeze the dataset here.
Source fingerprints, original pair records, image metadata and COCO licenses are in data/.

Viewer verification: JavaScript syntax and isolated save/import/CSV checks passed locally.
Browser appearance and interactions were not visually verified: local-file preview was blocked.
Actual COCO label review is pending. Preparing this bundle does not complete the audit.
"""
    assets = {'index.html': html.encode(), 'README.txt': readme.encode(),
              'data/viewer_cases.json': json_bytes(payload),
              'data/source_pairs.json': json_bytes(pairs)}
    for relative in sources:
        assets['data/' + Path(relative).name] = (ROOT / relative).read_bytes()
    for name, content in sorted(assets.items()):
        path = folder / name
        put_bytes(path, content)
        entries[name] = path
        records.append({'path': name, 'sha256': file_sha(path), 'bytes': len(content)})
    asset_manifest = {'run_id': ROOT.name, 'binding_sha256': binding,
                      'files': sorted(records, key=lambda r: r['path']),
                      'visual_reviews_completed': 0, 'dataset_frozen': False}
    manifest_path = folder / 'bundle_manifest.json'
    put_bytes(manifest_path, json_bytes(asset_manifest))
    entries['bundle_manifest.json'] = manifest_path
    local_zip = folder / f'annotation_audit_{binding[:16]}.zip'
    make_zip(local_zip, entries)
    durable_zip = ROOT / 'audits' / local_zip.name
    digest = copy_checked(local_zip, durable_zip)
    report6c.update({'status': 'passed', 'binding_sha256': binding,
        'pairs': len(pairs), 'original_images': len(images), 'zip_members': len(entries),
        'bundle_drive_path': str(durable_zip), 'bundle_sha256': digest,
        'bundle_bytes': local_zip.stat().st_size, 'zip_crc_passed': True,
        'drive_readback_verified': True, 'browser_visual_QA_performed': False,
        'elapsed_seconds': round(time.monotonic() - started, 2)})
except Exception:
    report6c['status'] = 'failed'
    report6c['traceback'] = traceback.format_exc()
    save_record(f'logs/block_06c_{stamp}.json', report6c)
    print('BLOCK 6C FAILED. Existing images and records retained; keep accelerator None.')
    raise

log_path = f'logs/block_06c_{stamp}.json'
log_digest = save_record(log_path, report6c)
print('\nBLOCK 6C COMPLETE — review bundle saved; actual visual review remains pending.')
print(json.dumps(report6c, indent=2))
print('Saved log:', ROOT / log_path)
print('Log SHA256:', log_digest)
print('Keep accelerator None. Paste the complete output and attach the downloaded ZIP here.')
try:
    from google.colab import files
    files.download(str(local_zip))
    print('Browser download requested. The verified Drive copy is also available at the path above.')
except Exception as exc:
    print('Browser download request failed; use the verified Drive ZIP:', type(exc).__name__, str(exc))
