import ast,hashlib,importlib,json,sys,tempfile,types,zipfile,io,contextlib
from pathlib import Path
import torch,numpy as np
out=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
with tempfile.TemporaryDirectory(prefix='oplora45_test_') as tmp:
 tmp=Path(tmp);bundle=tmp/'bundle';bundle.mkdir()
 with zipfile.ZipFile(out/'Block_45_L4_Evaluation.zip') as z:
  assert z.testzip() is None
  z.extractall(bundle);names=set(z.namelist())
  for n in names:
   if n.endswith('.py'):ast.parse(z.read(n))
 tree=ast.parse((out/'Block_45_L4_Cell.py').read_text());values={}
 for n in tree.body:
  if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('PACKAGE_SHA','EXPECTED_NAMES','GUARDIAN_SOURCE','EVIDENCE_SOURCE','COORDINATOR_SOURCE','EMBEDDED_SHAS'):
   values[n.targets[0].id]=ast.literal_eval(n.value)
 assert values['PACKAGE_SHA']==sha((out/'Block_45_L4_Evaluation.zip').read_bytes()) and values['EXPECTED_NAMES']==names
 for key,v in values['EMBEDDED_SHAS'].items():assert sha(values[key].encode())==v
 with zipfile.ZipFile(out.parent/'task1_lora16_evaluation/Block_43_L4_Evaluation.zip') as old:
  changed=[n for n in old.namelist() if old.read(n)!=(bundle/n).read_bytes()]
  assert sorted(changed)==['baseline_v2_worker.py','launch_baseline_v2.py']
 with zipfile.ZipFile(out.parent/'task1_oplora_training/Block_41_L4.zip') as train:
  assert train.read('core/training_core.py')==(bundle/'training_core.py').read_bytes()
 sys.path.insert(0,str(bundle));core=importlib.import_module('training_core');module=importlib.import_module('oplora_evaluation')
 torch.set_num_threads(2);checks=core.cpu_math_checks();assert all(checks.values())
 # Native Torch tensor/projection integration; mock only safetensors deserialization.
 st=types.ModuleType('safetensors');stt=types.ModuleType('safetensors.torch')
 stt.load_file=lambda path,device='cpu':torch.load(io.BytesIO(Path(path).read_bytes()),map_location=device,weights_only=True)
 st.torch=stt;sys.modules['safetensors']=st;sys.modules['safetensors.torch']=stt
 model=torch.nn.Module();model.layers=torch.nn.ModuleList([torch.nn.Module() for _ in range(28)])
 dims={'q_proj':(1024,2048),'k_proj':(1024,1024),'v_proj':(1024,1024),'o_proj':(2048,1024),'gate_proj':(1024,3072),'up_proj':(1024,3072),'down_proj':(3072,1024)}
 bases=tmp/'bases_root';(bases/'bases').mkdir(parents=True);entries={};assets={'bases':[],'training_core_sha256':sha((bundle/'training_core.py').read_bytes())}
 weights=tmp/'weights.zip';state=tmp/'state.json';state.write_text(json.dumps({'deadline_wall':__import__('time').time()+600}))
 with zipfile.ZipFile(weights,'w') as z:
  for i in range(28):
   for target,(ni,no) in dims.items():
    name=f'layers.{i}.{target}';setattr(model.layers[i],target,torch.nn.Linear(ni,no,bias=False,device='meta',dtype=torch.float16))
    for suffix,shape in [('.A',(16,ni)),('.B',(no,16))]:
     arr=np.full(shape,.01 if suffix=='.A' else .02,dtype=np.float32);buf=io.BytesIO();np.save(buf,arr,allow_pickle=False);raw=buf.getvalue();member=name+suffix+'.npy';z.writestr(member,raw)
     entries[name+suffix]={'member':member,'shape':list(shape),'sha256':sha(raw)}
    U=torch.zeros(no,128);V=torch.zeros(ni,128);U[:128]=torch.eye(128);V[:128]=torch.eye(128)
    key=name+'.weight';p=bases/'bases'/(key+'.safetensors');torch.save({'U':U,'V':V},p)
    assets['bases'].append({'key':key,'basis_sha256':sha(p.read_bytes())})
  z.writestr('index.json',json.dumps({'condition':'oplora_final','update':1250,'checkpoint_sha256':module.CHECKPOINT,'dtype':'float32','unique_parameters':entries}))
 ap=bases/'CPU_ASSETS_READY.json';ap.write_text(json.dumps(assets));ah=sha(ap.read_bytes());module.WEIGHTS=sha(weights.read_bytes())
 module.attach_adapters(model,weights,bases,ah,state)
 layers=[m for m in model.modules() if isinstance(m,core.AdapterLinear)]
 assert len(layers)==196
 assert sum(m.A.numel()+m.B.numel() for m in layers)==10092544
 for layer in layers:
  A,B=layer.effective_factors();assert torch.count_nonzero(A[:,:128])==0 and torch.count_nonzero(B[:128])==0
  assert torch.all(A[:,128:]==layer.A[:,128:]) and torch.all(B[128:]==layer.B[128:])
 # Exercise the actual checkpoint -> contract -> protocol -> assets hash chain.
 from evaluation_records import identity
 import shutil
 root=tmp/'root';(root/'manifests').mkdir(parents=True);(root/'configs').mkdir()
 assets.update(protection_rank=128,target_matrices=196)
 assetbytes=json.dumps(assets).encode();(root/'manifests/CPU_ASSETS_READY.json').write_bytes(assetbytes)
 (root/'bases').mkdir()
 for item in assets['bases']:
  name=item['key']+'.safetensors';(root/'bases'/name).hardlink_to(bases/'bases'/name)
 protocol={'CPU_assets_manifest_sha256':sha(assetbytes),'numerical_source_files':{'training_core.py':assets['training_core_sha256']}}
 pid=identity(protocol);(root/'configs'/('TRAINING_PROTOCOL_'+pid+'.json')).write_text(json.dumps(protocol))
 contract={'method':'oplora','rank':16,'alpha':32,'protection_rank':128,'protocol_identity':pid}
 module.CONTRACT=identity(contract)
 manifest={'header':{'method':'oplora','contract':contract,'contract_identity':module.CONTRACT,'successful_updates':1250,'examples_seen':40000}}
 module.MANIFEST=identity(manifest)
 folder=root/'checkpoints/oplora'/module.CONTRACT/'mandatory_001250_3575a06121f0';folder.mkdir(parents=True)
 (folder/'MANIFEST.json').write_text(json.dumps(manifest))
 job=tmp/'job';job.mkdir()
 assert module.prepare_bases(root,job,bundle,state)==sha(assetbytes)
 assert len(list((job/'oplora_bases/bases').glob('*.safetensors')))==196
 (root/'manifests/CPU_ASSETS_READY.json').write_bytes(b'corrupt')
 try:module.prepare_bases(root,job,bundle,state)
 except AssertionError:pass
 else:raise AssertionError('Changed asset manifest accepted')
 # Tampering with a basis fails before attaching that layer.
 p=bases/'bases'/(assets['bases'][0]['key']+'.safetensors');p.write_bytes(b'corrupt')
 fresh=torch.nn.Module();fresh.layers=torch.nn.ModuleList([torch.nn.Module() for _ in range(28)])
 for i in range(28):
  for target,(ni,no) in dims.items():setattr(fresh.layers[i],target,torch.nn.Linear(ni,no,bias=False,device='meta',dtype=torch.float16))
 try:module.attach_adapters(fresh,weights,bases,ah,state)
 except AssertionError:pass
 else:raise AssertionError('Basis corruption accepted')
 launcher=importlib.import_module('launch_baseline_v2');COUNTS=launcher.COUNTS
 session={'run_id':'fixture','protocol_identity':'p','model_identity':'m'}
 val={'run_id':'fixture','official_training_updates':0,'qualitative_completed':24,'FP32_probe_files_reused':64,'benchmark_completed':COUNTS,'status':'TASK1_OPLORA_FINAL_COMPLETE_SAVED','baseline_summary':{'status':'TASK1_OPLORA_FINAL_EVALUATION_COMPLETE','benchmark_counts':COUNTS,'audit':{'protocol_identity':'p','model_identity':'m'}}}
 assert launcher.valid_result(val,0,session)
 assert not launcher.valid_result(val,1,session)
 val['status']='TASK1_LORA16_FINAL_COMPLETE_SAVED';assert not launcher.valid_result(val,0,session)
 report={'status':'PASS','package_sha256':values['PACKAGE_SHA'],'python_modules_parsed':len(list(bundle.glob('*.py'))),'unchanged_existing_modules':len(names)-3,'guard_sources_unchanged':True,'training_core_byte_identical':True,'native_cpu_math_checks':checks,'adapter_modules_restored':196,'FP32_adapter_parameters':10092544,'projection_applied_and_unprotected_components_preserved':True,'corrupt_basis_rejected':True,'OPLoRA_completion_and_wrong_method_rejection':True,'local_torch_version':torch.__version__,'safetensors_deserialization_mocked_for_local_fixture':True,'checkpoint_to_basis_hash_chain_passed':True,'modified_asset_manifest_rejected':True,'GPU_run':False,'real_checkpoint_evaluation_pending':True}
 (out/'BLOCK_45_LOCAL_CHECKS.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
