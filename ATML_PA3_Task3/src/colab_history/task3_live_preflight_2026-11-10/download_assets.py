"""CPU-only pinned-byte model preparation. No torch import, model load or generation."""
import hashlib,json,re,struct,time,urllib.request
from pathlib import Path

CHUNK=8*1024*1024

def require(c,message):
    if not c:raise ValueError(message)
def digest_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(CHUNK),b''):h.update(b)
    return h.hexdigest()
def verify_file(path,entry):
    path=Path(path);require(not path.is_symlink() and path.is_file(),'Asset is not an ordinary file: '+path.name)
    require(path.stat().st_size==entry['bytes'],'Asset size differs: '+path.name)
    actual=digest_file(path);require(actual==entry['sha256'],'Asset checksum differs; preserving it: '+path.name)
    return {'name':path.name,'bytes':entry['bytes'],'sha256':actual}

def download(entry,destination,url,open_url=urllib.request.urlopen,clock=time.monotonic,notify=print):
    """One HTTP attempt. Resume a bounded partial only when Content-Range agrees."""
    destination=Path(destination)
    if destination.exists() or destination.is_symlink():
        notify('Verifying saved '+entry['name'],flush=True)
        return {**verify_file(destination,entry),'reused':True}
    partial=destination.with_name(destination.name+'.task3_partial')
    require(not partial.is_symlink() and (not partial.exists() or partial.is_file()),'Unexpected download partial')
    offset=partial.stat().st_size if partial.exists() else 0
    require(offset<=entry['bytes'],'Partial exceeds expected length; preserving it')
    if offset<entry['bytes']:
        headers={'User-Agent':'ATML-PA3-Task3-CPU-Preparation/1','Accept-Encoding':'identity'}
        if offset:headers['Range']=f'bytes={offset}-'
        start=clock();request=urllib.request.Request(url,headers=headers)
        notify(f"Downloading {entry['name']}: {offset:,}/{entry['bytes']:,} bytes",flush=True)
        with open_url(request,timeout=60) as response:
            status=response.status
            if status==206:
                cr=response.headers.get('Content-Range','');m=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',cr)
                require(m is not None and int(m[1])==offset and int(m[2])==entry['bytes']-1 and int(m[3])==entry['bytes'],'Incorrect HTTP resume range; preserving partial')
                mode='ab' if offset else 'wb'
            elif status==200:
                # HTTP server ignored Range. Restart this package-owned partial only.
                mode='wb';offset=0
            else:raise ValueError('Unexpected download HTTP status '+str(status))
            length=response.headers.get('Content-Length')
            if length is not None:require(int(length)==entry['bytes']-offset,'HTTP content length differs')
            next_progress=offset+64*1024*1024
            with partial.open(mode) as stream:
                while True:
                    require(clock()-start<1800,'CPU file download exceeded30 minutes; partial preserved')
                    b=response.read(CHUNK)
                    if not b:break
                    require(offset+len(b)<=entry['bytes'],'Downloaded bytes exceed expected size')
                    stream.write(b);offset+=len(b)
                    if offset>=next_progress:
                        stream.flush();notify(f"  {entry['name']}: {offset:,}/{entry['bytes']:,} bytes",flush=True);next_progress=offset+64*1024*1024
                stream.flush()
    verified=verify_file(partial,entry)
    if destination.exists() or destination.is_symlink():
        # Never overwrite content created by another notebook execution.
        verify_file(destination,entry);raise RuntimeError('Concurrent asset preparation detected; both files preserved')
    partial.rename(destination);verify_file(destination,entry)
    return {**verified,'name':destination.name,'reused':False}

def inspect_safetensors(folder,index):
    """Validate shard structure/dtypes and exact index coverage, without loading tensors."""
    entries={};dtypes={};parameters=0;payload_bytes=0
    for name in sorted(set(index['weight_map'].values())):
        require(Path(name).name==name and name.endswith('.safetensors'),'Unsafe shard name')
        p=folder/name
        with p.open('rb') as f:
            raw=f.read(8);require(len(raw)==8,'Missing safetensor header length');n=struct.unpack('<Q',raw)[0]
            require(0<n<16*1024*1024 and n+8<p.stat().st_size,'Invalid safetensor header length')
            header=json.loads(f.read(n));end=p.stat().st_size-8-n;offsets=[]
            for tensor,info in header.items():
                if tensor=='__metadata__':continue
                shape=info['shape'];require(isinstance(shape,list) and all(isinstance(x,int) and x>0 for x in shape),'Invalid tensor shape')
                count=1
                for dim in shape:count*=dim
                dtype=info['dtype'];require(dtype in ['BF16','F16','F32'],'Unexpected source tensor dtype')
                lo,hi=info['data_offsets'];width=4 if dtype=='F32' else 2
                require(0<=lo<hi<=end and hi-lo==count*width,'Invalid tensor byte range')
                require(tensor not in entries,'Duplicated tensor in shards');entries[tensor]=name;parameters+=count;payload_bytes+=hi-lo;offsets.append((lo,hi));dtypes[dtype]=dtypes.get(dtype,0)+count
            offsets.sort();require(offsets and offsets[0][0]==0 and offsets[-1][1]==end and all(x[1]==y[0] for x,y in zip(offsets,offsets[1:])),'Shard payload gaps/overlaps')
    require(entries==index['weight_map'],'Safetensor keys/shards differ from index')
    require(payload_bytes==index['metadata']['total_size'],'Indexed tensor byte total differs')
    return {'stored_parameter_values':parameters,'tensor_count':len(entries),'source_dtype_parameter_counts':dtypes,'tensor_payload_bytes':payload_bytes,'model_loaded':False}

def validate_small_assets(folder,manifest):
    config=json.loads((folder/'config.json').read_text());generation=json.loads((folder/'generation_config.json').read_text());tokenizer=json.loads((folder/'tokenizer_config.json').read_text());index=json.loads((folder/'model.safetensors.index.json').read_text())
    require(config['architectures']==['Qwen2ForCausalLM'] and config['model_type']=='qwen2' and config['vocab_size']==151936,'Wrong architecture/config')
    require(config['max_position_embeddings']==32768 and config['tie_word_embeddings'] is True,'Wrong context/embedding configuration')
    require(config['eos_token_id']==151645 and 151645 in generation['eos_token_id'],'EOS IDs differ')
    require(isinstance(tokenizer['chat_template'],str) and 'add_generation_prompt' in tokenizer['chat_template'],'Missing native chat template')
    names={e['name'] for e in manifest['files'] if e['name'].endswith('.safetensors')}
    require(set(index['weight_map'].values())==names and len(names)==2,'Model shard index differs')
    return index
