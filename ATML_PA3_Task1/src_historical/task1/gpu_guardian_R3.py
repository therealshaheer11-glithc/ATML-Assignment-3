"""R3 guardian: same bounded release policy plus nonblocking durable evidence."""
# Runtime unassignment endpoint: googlecolab/colabtools, google/colab/runtime.py (Apache-2.0).
import json
import os
import signal
import sys
import time
import urllib.request
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from watchdog_evidence import Evidence


def alive(pid):
    try:
        status=Path(f'/proc/{pid}/stat')
        if status.exists() and status.read_text().rsplit(')',1)[1].split()[0]=='Z':return False
        os.kill(pid,0);return True
    except (ProcessLookupError,FileNotFoundError):return False


def terminate(pid):
    if not pid or not alive(pid):return
    try:os.killpg(pid,signal.SIGTERM)
    except ProcessLookupError:return
    end=time.monotonic()+2
    while alive(pid) and time.monotonic()<end:time.sleep(.1)
    if alive(pid):
        try:os.killpg(pid,signal.SIGKILL)
        except ProcessLookupError:pass


def release(test_file=None):
    if test_file:
        if os.environ.get('TASK1_SAFETY_SELFTEST')!='1':raise RuntimeError('Test release requires self-test environment')
        Path(test_file).write_text('SIMULATED_RELEASE_ONLY\n');return 'simulated'
    address=os.environ.get('TBE_RUNTIME_ADDR')
    if not address:raise RuntimeError('Colab runtime-management address is unavailable')
    request=urllib.request.Request('http://'+address+'/unassign',data=b'',method='POST')
    errors=[]
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request,timeout=10) as response:
                if response.status!=200:raise RuntimeError('Runtime release request was rejected')
            return 'requested_not_independently_confirmed'
        except Exception as error:
            errors.append(type(error).__name__+': '+str(error))
            if attempt<2:time.sleep(1)
    raise RuntimeError('Three bounded runtime-release attempts failed: '+'; '.join(errors))



def supervise(state_path,event_path,test_file=None,poll=.5):
    state_path,event_path=Path(state_path),Path(event_path)
    # The initial record supplies an independent absolute deadline, even if later updates break.
    initial=json.loads(state_path.read_text());deadline=initial['deadline_wall'];last=initial
    evidence=Evidence(initial.get('evidence_directory'),initial.get('run_id'))
    while True:
        try:
            candidate=json.loads(state_path.read_text());last=candidate
            deadline=min(deadline,candidate['deadline_wall'])
        except Exception:pass
        now=time.time();pid=last.get('worker_pid');reason=None
        evidence.progress(last)
        # A worker's completion flag is not the parent's durable completion acknowledgement.
        # Never terminate a successful worker merely because it has published its result.
        status=last.get('status')
        if now>=deadline:reason='absolute_deadline'
        elif status in ('finished','failed'):reason=status
        elif initial.get('parent_pid') and not alive(initial['parent_pid']):reason='parent_exited'
        elif status=='worker_complete':
            grace=min(30.,max(.1,float(last.get('completion_grace_seconds',30.))))
            completed_at=last.get('worker_completed_wall',last.get('progress_wall',initial['progress_wall']))
            if now-completed_at>=grace:reason='completion_ack_timeout'
        elif now-last.get('progress_wall',initial['progress_wall'])>last.get('progress_timeout',900):reason='no_work_progress'
        elif pid and not alive(pid):reason='worker_exited'
        if reason:
            event={'reason':reason,'timestamp_wall':now,'worker_pid':pid,'release_confirmed':False,'last_phase':last.get('phase'),'last_progress_wall':last.get('progress_wall')}
            evidence.terminal('_release_intent.json',event)
            try:event_path.write_text(json.dumps(event,indent=2)+'\n')
            except Exception:pass
            try:terminate(pid)
            except OSError as error:
                # A failed signal must never prevent the VM-release request.
                event['worker_termination_error']=type(error).__name__+': '+str(error)
            try:event['release_result']=release(test_file)
            except Exception as error:event['release_error']=type(error).__name__+': '+str(error)
            try:event_path.write_text(json.dumps(event,indent=2)+'\n')
            except Exception:pass
            evidence.terminal('_release_response.json',event)
            return event
        time.sleep(poll)


if __name__=='__main__':
    supervise(sys.argv[1],sys.argv[2],sys.argv[3] if len(sys.argv)>3 else None)
