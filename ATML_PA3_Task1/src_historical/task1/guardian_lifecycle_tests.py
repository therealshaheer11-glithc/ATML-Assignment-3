"""Tests the worker/result/parent-ack handshake using CPU processes only."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def put(path,data):
    pending=path.with_suffix('.pending');pending.write_text(json.dumps(data));pending.replace(path)


def lifecycle(guardian,folder,kind):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    state=folder/'state.json';event=folder/'event.json';released=folder/'release.txt'
    now=time.time()
    data={'deadline_wall':now+(1.5 if kind=='hard_deadline' else 10),
          'progress_wall':now,'progress_timeout':900,'status':'running','worker_pid':None,
          'completion_grace_seconds':30 if kind in ('parent_ack','hard_deadline') else 1.2}
    put(state,data)
    # A successful worker remains alive briefly AFTER publishing its result/flag.
    code=('import json,time,pathlib;time.sleep(.4);p=pathlib.Path('+repr(str(state))+');'
          'd=json.loads(p.read_text());d.update(status="worker_complete",progress_wall=time.time(),worker_completed_wall=time.time());'
          'q=p.with_suffix(".worker_pending");q.write_text(json.dumps(d));q.replace(p);'
          + ('time.sleep(60)' if kind in ('hung_after_result','hard_deadline') else 'time.sleep(.8)'))
    worker=subprocess.Popen([sys.executable,'-c',code],start_new_session=True)
    data['worker_pid']=worker.pid;put(state,data)
    env=os.environ.copy();env['TASK1_SAFETY_SELFTEST']='1'
    guard=subprocess.Popen([sys.executable,str(guardian),str(state),str(event),str(released)],env=env,
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True)
    try:
        if kind=='parent_ack':
            worker.wait(timeout=5)
            assert worker.returncode==0,'Successful worker was prematurely terminated'
            assert not released.exists(),'Release happened before parent finalized the result'
            # Represent parent result verification, durable session record, and then ACK.
            (folder/'parent_finalized.txt').write_text('verified exit 0 and saved result')
            d=json.loads(state.read_text());d.update(status='finished',progress_wall=time.time());put(state,d)
        end=time.monotonic()+12
        while guard.poll() is None and time.monotonic()<end:worker.poll();time.sleep(.05)
        if guard.poll() is None:raise AssertionError('Guardian did not honor bounded completion')
        output=guard.communicate()[0];assert guard.returncode==0,output
        assert released.read_text()=='SIMULATED_RELEASE_ONLY\n'
        worker.wait(timeout=3)
        reason=json.loads(event.read_text())['reason']
        expected={'parent_ack':'finished','parent_missing':'completion_ack_timeout',
                  'hung_after_result':'completion_ack_timeout','hard_deadline':'absolute_deadline'}[kind]
        assert reason==expected,(kind,reason)
        if kind in ('parent_ack','parent_missing'):assert worker.returncode==0
        else:assert worker.returncode!=0
        return True
    finally:
        if worker.poll() is None:worker.kill();worker.wait()
        if guard.poll() is None:guard.kill();guard.wait()


def run_tests(guardian,directory):
    return {kind:lifecycle(guardian,Path(directory)/kind,kind)
            for kind in ('parent_ack','parent_missing','hung_after_result','hard_deadline')}


if __name__=='__main__':print(json.dumps(run_tests(sys.argv[1],sys.argv[2]),indent=2))
