"""Parent ACK after saved results; guardian primary, bounded notebook fallback."""
import json
import threading
import time
import uuid
from pathlib import Path


def finalize(state_path,event_path,release_callback,succeeded,wait_seconds=35,fallback_seconds=8):
    state_path,event_path=Path(state_path),Path(event_path)
    state=json.loads(state_path.read_text())
    state.update(status='finished' if succeeded else 'failed',progress_wall=time.time())
    pending=state_path.with_name(state_path.name+'.pending_'+uuid.uuid4().hex)
    pending.write_text(json.dumps(state));pending.replace(state_path)
    until=time.monotonic()+wait_seconds
    event={}
    while time.monotonic()<until:
        try:event=json.loads(event_path.read_text())
        except Exception:pass
        if event.get('release_result'):
            return {'primary_request_accepted':True,'fallback_called':False,'release_confirmed':False}
        if event.get('release_error'):break
        time.sleep(min(.1,max(0,until-time.monotonic())))
    outcome={'primary_request_accepted':False,'fallback_called':True,'release_confirmed':False}
    def invoke():
        try:release_callback();outcome['fallback_request_returned']=True
        except Exception as error:outcome['fallback_error']=type(error).__name__+': '+str(error)
    thread=threading.Thread(target=invoke,daemon=True);thread.start();thread.join(fallback_seconds)
    outcome['fallback_still_running']=thread.is_alive()
    return dict(outcome)


def cpu_checks(directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    state=directory/'state.json';event=directory/'event.json';calls=[]
    def reset():
        state.write_text(json.dumps({'status':'worker_complete'}))
        if event.exists():event.unlink()
    reset();event.write_text(json.dumps({'release_result':'requested_not_independently_confirmed'}))
    result=finalize(state,event,lambda:calls.append(1),True,wait_seconds=.05)
    assert not calls and not result['fallback_called'] and json.loads(state.read_text())['status']=='finished'
    reset();event.write_text(json.dumps({'release_error':'primary unavailable'}))
    result=finalize(state,event,lambda:calls.append(1),True,wait_seconds=.05)
    assert calls==[1] and result['fallback_called']
    reset()
    def fails():raise RuntimeError('simulated release error')
    result=finalize(state,event,fails,True,wait_seconds=.05)
    assert 'fallback_error' in result and json.loads(state.read_text())['status']=='finished'
    reset();stop=threading.Event();began=time.monotonic()
    try:
        result=finalize(state,event,lambda:stop.wait(5),True,wait_seconds=.05,fallback_seconds=.05)
        assert result['fallback_still_running'] and time.monotonic()-began<1
    finally:stop.set()
    return {'primary_success_avoids_duplicate_release':True,'primary_failure_uses_fallback':True,
            'release_error_preserves_numerical_completion':True,'stuck_fallback_is_bounded':True}
