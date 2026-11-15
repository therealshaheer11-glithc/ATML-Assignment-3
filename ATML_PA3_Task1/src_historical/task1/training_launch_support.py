"""Standard-library launch checks. No torch/model imports in the parent."""
import hashlib
import json
import os
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path

from archive_io import digest, safe_relative
from baseline_runtime import atomic, publish_bytes
from evaluation_records import identity, publish


def checkpoint_evidence(root, relative, contract, expected_parameters, batch_plan=None):
    root = Path(root); safe_relative(relative); folder = root / relative
    assert folder.parent == root / 'checkpoints' / contract['method'] / identity(contract)
    assert not folder.is_symlink() and folder.resolve().is_relative_to(root.resolve())
    marker_path = folder / 'COMPLETE.json'; marker = json.loads(marker_path.read_text())
    owner = json.loads((folder / 'OWNER.json').read_text())
    assert owner['workflow'] == 'ATML_PA3_TASK1_TRAINING_V1'
    assert owner['method'] == marker['method'] == contract['method']
    assert owner['contract_identity'] == marker['contract_identity'] == identity(contract)
    assert owner['kind'] == marker['kind'] and owner['update'] == marker['successful_updates']
    update = marker['successful_updates']; assert type(update) is int and 0 <= update <= 1250
    assert marker['examples_seen'] == update * 32
    if marker['kind'] == 'mandatory': assert update in (312, 625, 937, 1250)
    else: assert marker['kind'] == 'rolling'
    manifest = json.loads((folder / 'MANIFEST.json').read_text()); header = manifest['header']
    assert identity(manifest) == marker['manifest_identity'] and identity(header) == marker['state_signature']
    assert header['contract'] == contract and header['contract_identity'] == identity(contract)
    assert header['successful_updates'] == update and header['examples_seen'] == update * 32
    assert len(header['history']) == update and [r['update'] for r in header['history']] == list(range(1, update + 1))
    assert header['scheduler']['successful_updates'] == update and header['scheduler']['next_update'] == update + 1
    if batch_plan is not None:
        for record, expected in zip(header['history'], batch_plan):
            assert record['source_ids'] == expected['source_ids'] and record['supervised_tokens'] == expected['supervised_tokens']
    tensors = manifest['tensors']; model = [t for t in tensors if t['source'][0] == 'model']
    assert sum(t['numel'] for t in model) == expected_parameters
    assert all(t['dtype'] == 'torch.float32' for t in tensors)
    assert len({tuple(t['source']) for t in tensors}) == len(tensors)
    if update:
        names = {t['source'][1] for t in model}
        opt = {tuple(t['source'][1:]) for t in tensors if t['source'][0] == 'optimizer'}
        assert opt == {(n, key) for n in names for key in ('step','exp_avg','exp_avg_sq')}
    archive = folder / 'state.tar'; assert archive.is_file() and not archive.is_symlink()
    assert archive.stat().st_size == marker['archive_bytes'] >= sum(t['numel'] * 4 for t in tensors)
    # Tensor bytes were verified by the worker before COMPLETE; do not reread a
    # multi-gigabyte archive again merely to acknowledge its completed save.
    return {'path': relative, 'marker_sha256': digest(marker_path), 'manifest_identity': marker['manifest_identity'],
        'archive_sha256': marker['archive_sha256'], 'archive_bytes': marker['archive_bytes'],
        'successful_updates': update, 'examples_seen': update * 32, 'kind': marker['kind'],
        'created_wall': marker['created_wall'], 'worker_verified_tensor_readback': True}


def latest_checkpoint(root, contract, expected_parameters, batch_plan=None):
    root = Path(root); directory = root / 'checkpoints' / contract['method'] / identity(contract)
    values = [checkpoint_evidence(root,str(p.parent.relative_to(root)),contract,expected_parameters,batch_plan)
              for p in directory.glob('*/COMPLETE.json')]
    return max(values,key=lambda v:(v['successful_updates'],v['created_wall'])) if values else None


def valid_result(value, code, session):
    if code != 0 or value.get('run_id') != session['run_id'] or value.get('method') != 'full': return False
    u = value.get('official_training_updates')
    if type(u) is not int or not session['start_update'] <= u <= session['target_update']: return False
    if value.get('examples_seen') != 32*u or value.get('updates_this_job') != u-session['start_update']: return False
    if value.get('training_contract_identity') != session['training_contract_identity']: return False
    ckpt = value.get('checkpoint', {})
    if ckpt.get('successful_updates') != u: return False
    if value.get('status') == 'TASK1_FULL_FT_BOUNDARY_SAVED':
        return u == session['target_update'] and ckpt.get('kind') == 'mandatory'
    return value.get('status') == 'TASK1_FULL_FT_PARTIAL_SAVED' and u < session['target_update']


def execute(command, env, context, session, contract, expected_parameters, batch_plan=None):
    root, job = Path(context['root']), Path(context['job']); run_id = session['run_id']; worker = None
    live = None; last_live = 0.
    monitor_stop=threading.Event()
    def monitor():
        last=0.
        while not monitor_stop.wait(5):
            if time.monotonic()-last<30:continue
            try:
                state=json.loads(Path(context['state']).read_text())
                if state.get('status')!='running':continue
                age=max(0.,time.time()-state.get('progress_wall',time.time()))
                print('Worker progress: %s (last recorded %.0fs ago)'%(state.get('phase','starting'),age),flush=True)
                last=time.monotonic()
            except Exception:pass
    monitor_thread=threading.Thread(target=monitor,daemon=True)
    local_log = job / 'worker_console.txt'
    def mirror(data):
        try:
            target = root / 'logs' / (run_id+'_worker_console_live.txt')
            temporary = target.with_name(target.name+'.pending_'+uuid.uuid4().hex)
            temporary.write_bytes(data); temporary.replace(target)
        except Exception: pass  # Best effort; verified final console is saved separately.
    try:
        with local_log.open('x') as log:
            worker = subprocess.Popen(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                text=True,bufsize=1,start_new_session=True)
            state=json.loads(Path(context['state']).read_text())
            state.update(worker_pid=worker.pid,status='running',progress_wall=time.time());atomic(context['state'],state)
            monitor_thread.start()
            for line in worker.stdout:
                log.write(line);log.flush()
                try: record=json.loads(line)
                except (ValueError,TypeError): record={}
                if isinstance(record,dict) and 'update' in record and 'sft_loss' in record:
                    if record['update']%10==0 or record['update']==session['start_update']+1 or record.get('overflow_attempts'):
                        print('Full FT %d/%d | SFT %.5f | LR %.8g | scale %.0f | %.1fs'%(
                            record['update'],session['target_update'],record['sft_loss'],record['lr'],record['scale'],record['total_update_seconds_including_retries']),flush=True)
                else: print(line,end='',flush=True)
                if time.monotonic()-last_live>=30 and (live is None or not live.is_alive()):
                    last_live=time.monotonic();live=threading.Thread(target=mirror,args=(local_log.read_bytes(),),daemon=True);live.start()
            code=worker.wait()
        publish_bytes(root/'logs'/(run_id+'_worker_console.txt'),local_log.read_bytes())
        path=root/'logs'/(run_id+'_result.json');value=json.loads(path.read_text())
        valid=valid_result(value,code,session)
        if valid:
            actual=checkpoint_evidence(root,value['checkpoint']['path'],contract,expected_parameters,batch_plan)
            valid=actual==value['checkpoint']
        publish(root/'logs'/(run_id+'_session_final.json'),{**session,'status':value.get('status'),
            'worker_exit_code':code,'worker_result_validated':valid,'result_sha256':digest(path),
            'verified_console_sha256':digest(root/'logs'/(run_id+'_worker_console.txt')),
            'elapsed_allocation_seconds_at_final_save':time.time()-context['started_wall'],
            'estimated_units_so_far_with_margin':(time.time()-context['started_wall'])/3600*context['rate']*1.1,
            'release_request_pending':True,'physical_release_verified':False})
        if not valid: raise RuntimeError('Full FT failed or its final recovery receipt was inconsistent; return the preserved output.')
        print('Saved Full FT result:',path,flush=True)
        return value
    finally:
        monitor_stop.set()
        if monitor_thread.is_alive():monitor_thread.join(.25)
        if worker is not None and worker.poll() is None:
            try:os.killpg(worker.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:worker.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:os.killpg(worker.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                worker.wait(timeout=3)
