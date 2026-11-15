"""A002: one exact approved PyTorch loader exception; every other RECORD check stays strict."""
import base64,hashlib,json
from importlib import metadata
from pathlib import Path

POLICY_CANONICAL_SHA256='20c33b5b8b0e679431241fec16ec8f22102655ad061b4b8ab881992640a81f0a'
def require(c,message):
    if not c:raise ValueError(message)
def digest(raw):return hashlib.sha256(raw).hexdigest()
def validate_policy(policy):
    require(digest(json.dumps(policy,sort_keys=True,separators=(',',':')).encode())==POLICY_CANONICAL_SHA256,'A002 policy differs from explicit approval')
def load_policy(path):
    policy=json.loads(Path(path).read_text());validate_policy(policy);return policy

def fingerprint(names,policy):
    from cpu_preflight import fingerprint as strict_fingerprint
    validate_policy(policy)
    others=strict_fingerprint([name for name in names if name!='torch']);result={}
    for name in names:
        if name!='torch':result[name]=others[name];continue
        import torch
        dist=metadata.distribution('torch')
        require(dist.version==policy['version'] and str(torch.__version__)==policy['version'],'A002 applies only to the inspected exact CUDA build')
        require(torch.version.cuda==policy['CUDA_build'] and torch.version.git_version==policy['git_version'],'A002 CUDA/Git identity differs')
        require(Path(torch.__file__).resolve()==Path(dist.locate_file(policy['file'])).resolve(),'Imported torch and distribution files differ')
        entries=[];version_file=None;exceptions=[]
        for entry in dist.files or []:
            label=str(entry)
            if not label.endswith('.py'):continue
            raw=dist.locate_file(entry).read_bytes();actual=digest(raw)
            require(entry.hash and entry.hash.mode=='sha256','Installed asset lacks SHA256 RECORD hash: '+label)
            actual_record=base64.urlsafe_b64encode(bytes.fromhex(actual)).decode().rstrip('=')
            if actual_record!=entry.hash.value:
                expected=base64.urlsafe_b64encode(bytes.fromhex(policy['official_RECORD_sha256'])).decode().rstrip('=')
                require(label==policy['file'] and entry.hash.value==expected and actual==policy['accepted_modified_sha256'] and len(raw)==policy['accepted_modified_bytes'],'Unknown installed modification; A002 does not cover: '+label)
                exceptions.append({'approval_id':'A002','file':label,'actual_sha256':actual,'original_RECORD_sha256':policy['official_RECORD_sha256'],'difference':policy['difference']})
            if label=='torch/version.py':version_file=actual;continue
            entries.append(label+':'+actual)
        require(entries,'Torch source inventory absent')
        result[name]={'version':dist.version,'python_source_files':len(entries),'source_sha256':digest('\n'.join(sorted(entries)).encode()),'compiled_assets':[],'torch_build_version_file_sha256':version_file,'torch_variant_rule':'Exact inspected2.14.1+cu130 build; A002 covers one exact initializer hash only','accepted_source_exceptions':exceptions}
    return result

def cuda_dependency_inventory():
    """Record library locations and installed manifests without rehashing gigabytes of binaries."""
    packages=[];seen=set()
    for dist in metadata.distributions():
        name=dist.metadata.get('Name','').lower().replace('_','-')
        if not name.startswith('nvidia-'):continue
        location=str(getattr(dist,'_path',''));key=(name,dist.version,location)
        if key in seen:continue
        seen.add(key);record=dist.read_text('RECORD')
        require(record is not None,'CUDA dependency RECORD missing: '+name)
        packages.append({'name':name,'version':dist.version,'metadata_location':location,'RECORD_sha256':digest(record.encode())})
    maps=Path('/proc/self/maps');libraries=set()
    if maps.is_file():
        for line in maps.read_text().splitlines():
            parts=line.split(maxsplit=5)
            if len(parts)<6:continue
            path=parts[5]
            if '.so' in path and ('/nvidia/' in path or '/torch/lib/' in path or '/libcuda.so' in path):libraries.add(path)
    return {'installed_NVIDIA_distributions':sorted(packages,key=lambda x:(x['name'],x['version'],x['metadata_location'])),'loaded_library_paths_at_setup':sorted(libraries),'process_maps_available':maps.is_file(),'large_library_bytes_rehashed':False,'scope':'Manifest identities and paths at setup; additional libraries may load lazily during inference.'}
