"""Real CPU child-process/guardian tests of the Full FT launch handoff."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from baseline_runtime import atomic
from evaluation_records import identity,publish
from release_coordination import finalize
from training_launch_support import checkpoint_evidence,latest_checkpoint,execute


def fixture_checkpoint(root,contract,update,kind):
    folder=Path(root)/'checkpoints/full'/identity(contract)/(kind+'_'+str(update))
    folder.mkdir(parents=True)
    header={'contract':contract,'contract_identity':identity(contract),'successful_updates':update,'examples_seen':32*update,
        'history':[{'update':i} for i in range(1,update+1)],
        'scheduler':{'successful_updates':update,'next_update':update+1}}
    tensors=[{'source':['model','p',None],'numel':4,'dtype':'torch.float32'}]
    if update:tensors += [{'source':['optimizer','p',key],'numel':1 if key=='step' else 4,'dtype':'torch.float32'} for key in ('step','exp_avg','exp_avg_sq')]
    manifest={'header':header,'tensors':tensors}
    archive=folder/'state.tar';archive.write_bytes(b'CPU_FIXTURE_NOT_A_REAL_TENSOR_ARCHIVE'+b'.'*1024)
    publish(folder/'MANIFEST.json',manifest)
    publish(folder/'OWNER.json',{'workflow':'ATML_PA3_TASK1_TRAINING_V1','method':'full','contract_identity':identity(contract),'kind':kind,'update':update})
    publish(folder/'COMPLETE.json',{'method':'full','kind':kind,'successful_updates':update,'examples_seen':32*update,
        'contract_identity':identity(contract),'manifest_identity':identity(manifest),'state_signature':identity(header),
        'archive_bytes':archive.stat().st_size,'archive_sha256':'CPU_FIXTURE_NO_NUMERICAL_READBACK_CLAIM','created_wall':float(update)})
    return checkpoint_evidence(root,str(folder.relative_to(root)),contract,4)


def parent_case(bundle,directory,kind):
    root=Path(directory);job=root/'job';job.mkdir(parents=True);(root/'logs').mkdir()
    run='task1_full_parent_fixture_'+kind;state,event,released=job/'state.json',job/'event.json',job/'released.txt';now=time.time()
    atomic(state,{'deadline_wall':now+20,'progress_wall':now,'progress_timeout':900,'worker_pid':None,
        'parent_pid':os.getpid(),'status':'setup','run_id':run,'evidence_directory':str(root/'logs')})
    env=os.environ.copy();env['TASK1_SAFETY_SELFTEST']='1'
    guardian=subprocess.Popen([sys.executable,str(Path(bundle)/'gpu_guardian_R3.py'),str(state),str(event),str(released)],env=env,start_new_session=True)
    contract={'method':'full','scope':'CPU_FIXTURE_ONLY'}
    u=312 if kind=='mandatory' else 50
    ckpt=fixture_checkpoint(root,contract,u,'mandatory' if u==312 else 'rolling')
    session={'run_id':run,'start_update':0,'target_update':312,'training_contract_identity':identity(contract)}
    value={'status':'TASK1_FULL_FT_BOUNDARY_SAVED' if u==312 else 'TASK1_FULL_FT_PARTIAL_SAVED',
        'run_id':run,'method':'full','official_training_updates':u,'updates_this_job':u,'examples_seen':32*u,
        'training_contract_identity':identity(contract),'checkpoint':ckpt}
    if kind=='false_boundary':value['status']='TASK1_FULL_FT_BOUNDARY_SAVED'
    if kind=='wrong_count':value['official_training_updates']=51
    if kind=='marker_changed':
        marker=root/ckpt['path']/'COMPLETE.json';changed=json.loads(marker.read_text());changed['created_wall']+=1;marker.write_text(json.dumps(changed))
    final=root/'logs'/(run+'_session_final.json')
    if kind=='save_error':final.write_text('{"preserved":true}')
    payload=job/'payload.json';payload.write_text(json.dumps(value))
    dummy=job/'dummy.py'
    dummy.write_text('''import json,os,pathlib,sys,time
s=pathlib.Path(sys.argv[1]);r=pathlib.Path(sys.argv[2]);payload=pathlib.Path(sys.argv[3]);kind=sys.argv[4]
while json.loads(s.read_text()).get('worker_pid')!=os.getpid():time.sleep(.01)
print('CPU fixture worker output',flush=True)
if kind!='missing':r.write_bytes(payload.read_bytes())
v=json.loads(s.read_text());v.update(status='worker_complete',worker_completed_wall=time.time(),progress_wall=time.time())
p=s.with_suffix('.pending');p.write_text(json.dumps(v));p.replace(s)
sys.exit(7 if kind=='worker_error' else 0)
''')
    success=False
    try:
        try:
            execute([sys.executable,str(dummy),str(state),str(root/'logs'/(run+'_result.json')),str(payload),kind],env,
                {'root':str(root),'job':str(job),'state':str(state),'started_wall':now,'rate':1.94},session,contract,4)
            success=True
        except Exception:assert kind not in ('partial','mandatory')
        assert success==(kind in ('partial','mandatory'))
        assert not released.exists()
        outcome=finalize(state,event,lambda:(_ for _ in ()).throw(AssertionError('Unexpected fallback')),success,wait_seconds=3)
        guardian.wait(timeout=5)
        assert outcome['primary_request_accepted'] and released.read_text()=='SIMULATED_RELEASE_ONLY\n'
        assert json.loads(event.read_text())['reason']==('finished' if success else 'failed')
        if kind=='save_error':assert json.loads(final.read_text())=={'preserved':True}
        return True
    finally:
        if guardian.poll() is None:guardian.kill();guardian.wait()


def checks(bundle,directory):
    directory=Path(directory)
    cases={kind:parent_case(bundle,directory/kind,kind) for kind in
        ('partial','mandatory','false_boundary','wrong_count','marker_changed','save_error','missing','worker_error')}
    root=directory/'selection';root.mkdir()
    contract={'method':'full','scope':'CPU_FIXTURE_ONLY'}
    fixture_checkpoint(root,contract,0,'rolling');fixture_checkpoint(root,contract,50,'rolling');latest=fixture_checkpoint(root,contract,100,'rolling')
    assert latest_checkpoint(root,contract,4)==latest
    (root/'checkpoints/full'/identity(contract)/'unfinished').mkdir()
    assert latest_checkpoint(root,contract,4)==latest
    try:checkpoint_evidence(root,'../outside',contract,4)
    except RuntimeError:pass
    else:raise AssertionError('Escaping checkpoint path accepted')
    return {'real_child_parent_guardian_cases':cases,'latest_complete_checkpoint_selection_ignores_unfinished_states':True,
        'escaping_checkpoint_path_rejected':True,'no_torch_import_required_by_parent': 'torch' not in sys.modules,
        'scope':'CPU synthetic controller/metadata tests; no GPU allocation or numerical training execution'}
