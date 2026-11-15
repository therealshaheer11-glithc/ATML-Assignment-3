"""Immutable call evidence, resumption and protocol validation; no torch or gold."""
import hashlib,json,math,uuid
from pathlib import Path
import live_protocol

class SoftLimitReached(RuntimeError):pass

def require(c,message):
    if not c:raise ValueError(message)
def encode(value):return (json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
def digest(raw):return hashlib.sha256(raw).hexdigest()
def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(path,raw):
    path=Path(path);require(not path.is_symlink(),'Refusing symlink: '+path.name)
    if path.exists():require(path.is_file() and path.read_bytes()==raw,'Existing bytes differ; preserving '+path.name);return digest(raw)
    pending=path.with_name(path.name+'.pending_'+uuid.uuid4().hex)
    with pending.open('xb') as f:f.write(raw);f.flush()
    require(pending.read_bytes()==raw,'Pending readback differs')
    require(not path.exists() and not path.is_symlink(),'Concurrent write detected; files preserved')
    pending.rename(path);require(path.read_bytes()==raw,'Final readback differs');return digest(raw)
def call_key(q,name):return hashlib.sha256((q+'\0'+name).encode()).hexdigest()
def envelope(record):return {'record':record,'record_sha256':digest(encode(record))}
def parse_envelope(raw):
    value=json.loads(raw);require(set(value)=={'record','record_sha256'} and value['record_sha256']==digest(encode(value['record'])),'Call envelope checksum differs');return value['record']
def termination(ids,cap):
    require(isinstance(ids,list) and 0<len(ids)<=cap and all(type(i)==int and 0<=i<151936 for i in ids),'Invalid generated token IDs/count')
    eos=[i for i,x in enumerate(ids) if x in [151645,151643]];require(not eos or eos==[len(ids)-1],'Termination token occurs before sequence end')
    ended=bool(eos);require(ended or len(ids)==cap,'Unexpected early termination; incomplete call is not a model response')
    return {'output_tokens':len(ids),'retained_nontermination_tokens':len(ids)-int(ended),'ended_on_eos':ended,'termination_reason':'eos' if ended else 'token_cap','at_output_cap':len(ids)==cap,'truncated_by_cap':len(ids)==cap and not ended}
def messages_for(row,name,previous,protocol):
    if name!='revision':return row['initial_messages'],None
    key=call_key(row['sample_id'],'parallel_1');require(key in previous,'Shared initial response is not verified')
    prior=previous[key]
    return live_protocol.revision_messages(row['initial_messages'],prior['record']['generated_text'],protocol),{'call_key':key,'envelope_sha256':prior['envelope_sha256']}
def validate(record,row,name,recipe_sha,environment_sha,previous,protocol,tokenizer=None):
    plan=row['call_plan'];cap=plan['caps'][name];key=call_key(row['sample_id'],name)
    require(record['schema_version']==1 and record['status']=='COMPLETE' and record['recipe_sha256']==recipe_sha and record['GPU_environment_sha256']==environment_sha,'Call recipe/environment differs')
    require(record['call_key']==key and record['sample_id']==row['sample_id'] and record['call_name']==name and record['seed']==plan['seeds'][name] and record['max_new_tokens']==cap,'Call identity/seed/cap differs')
    messages,shared=messages_for(row,name,previous,protocol)
    require(record['messages']==messages and record['shared_initial']==shared and record['rendered_prompt']==live_protocol.expected_native_text(messages),'Call prompt/shared initial differs')
    require(record['generation_options']==live_protocol.generation_options(protocol,cap),'Generation settings differ')
    ids=record['input_ids'];require(isinstance(ids,list) and ids and all(type(i)==int and 0<=i<151936 for i in ids) and len(ids)==record['input_tokens'],'Invalid prompt IDs/count')
    live_protocol.check_context(ids,cap)
    require(record['termination']==termination(record['output_ids'],cap),'Termination metadata differs')
    require(record['extraction']==live_protocol.extract_last_box(record['generated_text']),'Extraction differs')
    for field in ['inference_seconds','call_wall_seconds']:
        require(isinstance(record[field],(int,float)) and math.isfinite(record[field]) and 0<record[field]<600,'Invalid/over-time inference measurement')
    require(record['call_wall_seconds']>=record['inference_seconds'],'Call wall time shorter than inference')
    require(record['started_before_soft_limit'] is True and record['completed_before_call_deadline'] is True,'Incomplete execution cannot be graded')
    for field in ['peak_allocated_bytes','peak_reserved_bytes']:require(type(record[field])==int and record[field]>0,'Invalid memory measurement')
    require(record['peak_reserved_bytes']>=record['peak_allocated_bytes'],'Reserved memory below allocated peak')
    if tokenizer:
        require(live_protocol.encode_messages(tokenizer,messages)['input_ids']==ids,'Saved prompt tokenization differs')
        require(tokenizer.decode(record['output_ids'],skip_special_tokens=True,clean_up_tokenization_spaces=False)==record['generated_text'],'Saved decoded response differs')
        require(tokenizer.decode(record['output_ids'],skip_special_tokens=False,clean_up_tokenization_spaces=False)==record['raw_decoded_text'],'Saved raw response differs')
    return key

def scan(root,rows,recipe_sha,environment_sha,protocol,tokenizer=None):
    """Never sample over incomplete files. Complete pending files can be promoted only if unique."""
    folder=root/'calls';folder.mkdir(exist_ok=True);previous={};expected={call_key(r['sample_id'],name) for r in rows for name in r['call_plan']['call_order']}
    for p in folder.iterdir():
        require(p.is_file() and not p.is_symlink(),'Unexpected call directory member')
        require(any(p.name==key+'.json' or p.name.startswith(key+'.json.pending_') for key in expected),'Unknown saved call file: '+p.name)
    for row in rows:
        for name in row['call_plan']['call_order']:
            key=call_key(row['sample_id'],name);path=folder/(key+'.json');partials=sorted(folder.glob(key+'.json.pending_*'));candidates=([path] if path.exists() else [])+partials
            if not candidates:continue
            blobs=[]
            for candidate in candidates:
                raw=candidate.read_bytes();record=parse_envelope(raw);validate(record,row,name,recipe_sha,environment_sha,previous,protocol,tokenizer)
                session=root/'sessions'/(record['session_id']+'.json');require(session.is_file() and file_sha(session)==record['physical_session_manifest_sha256'],'Physical-session manifest differs')
                blobs.append(raw)
            require(all(raw==blobs[0] for raw in blobs),'Conflicting completed attempts; preserve and review before retry')
            if not path.exists():save(path,blobs[0])
            previous[key]={'record':parse_envelope(blobs[0]),'envelope_sha256':digest(blobs[0])}
    return previous

def stage_assets(source,destination,manifest,notify=print):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    for e in manifest['files']:
        target=destination/e['name'];require(not target.is_symlink(),'Unexpected local asset symlink')
        if target.exists():require(target.stat().st_size==e['bytes'] and file_sha(target)==e['sha256'],'Cached model differs');continue
        notify('Staging verified '+e['name'],flush=True);pending=target.with_name(target.name+'.pending_'+uuid.uuid4().hex);h=hashlib.sha256();count=0
        with (source/e['name']).open('rb') as src,pending.open('xb') as dst:
            for b in iter(lambda:src.read(8*1024*1024),b''):dst.write(b);h.update(b);count+=len(b)
            dst.flush()
        require(count==e['bytes'] and h.hexdigest()==e['sha256'],'Source asset differs; staged partial preserved')
        require(file_sha(pending)==e['sha256'],'Local staged readback differs');pending.rename(target)
    return destination
