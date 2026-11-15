"""Exercise the real supervisor against disposable CPU child processes."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def run_tests(guardian,directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    outcomes={}
    for kind in ('completed','worker_error','deadline','stalled','corrupt_state','event_save_failure'):
        folder=directory/kind;folder.mkdir(exist_ok=True)
        state=folder/'state.json';event=folder/'event.json';released=folder/'simulated_release.txt'
        t=time.time();limit=1.5 if kind in ('deadline','corrupt_state','event_save_failure') else 8
        value={'deadline_wall':t+limit,'progress_wall':t,'progress_timeout':.7 if kind=='stalled' else 900,
               'worker_pid':None,'status':'running'}
        state.write_text(json.dumps(value))
        child_code='import time; time.sleep(60)'
        if kind=='completed':
            child_code=('import time,json,pathlib;time.sleep(.4);p=pathlib.Path('+repr(str(state))+');'
                'd=json.loads(p.read_text());d["status"]="finished";p.write_text(json.dumps(d))')
        elif kind=='worker_error':child_code='import time,sys;time.sleep(.4);sys.exit(7)'
        worker=subprocess.Popen([sys.executable,'-c',child_code],start_new_session=True,
                                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        value['worker_pid']=worker.pid;state.write_text(json.dumps(value))
        env=os.environ.copy();env['TASK1_SAFETY_SELFTEST']='1'
        if kind=='event_save_failure':event=folder/'missing_directory'/'event.json'
        supervisor=subprocess.Popen([sys.executable,str(guardian),str(state),str(event),str(released)],
             env=env,start_new_session=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        try:
            if kind=='corrupt_state':time.sleep(.4);state.write_text('broken json')
            until=time.monotonic()+12
            while supervisor.poll() is None and time.monotonic()<until:
                worker.poll()  # Reap exited child even on systems without /proc.
                time.sleep(.05)
            if supervisor.poll() is None:raise AssertionError('Supervisor failed to end: '+kind)
            output=supervisor.communicate()[0]
            assert supervisor.returncode==0,(kind,output)
            assert released.read_text()=='SIMULATED_RELEASE_ONLY\n',kind
            worker.wait(timeout=3)
            if kind!='event_save_failure':
                reason=json.loads(event.read_text())['reason']
                expected={'completed':'finished','worker_error':'worker_exited','deadline':'absolute_deadline',
                          'stalled':'no_work_progress','corrupt_state':'absolute_deadline'}[kind]
                assert reason==expected,(kind,reason)
            outcomes[kind]=True
        finally:
            if worker.poll() is None:worker.kill();worker.wait()
            if supervisor.poll() is None:supervisor.kill();supervisor.wait()
    return outcomes


if __name__=='__main__':print(json.dumps(run_tests(sys.argv[1],sys.argv[2]),indent=2))
