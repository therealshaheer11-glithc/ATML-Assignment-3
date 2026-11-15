# Task 1 - Block 11: L4 strict backward gate, fixed probes, original baseline.
# Standalone. Download the supplied ZIP and this cell BEFORE enabling L4.
import os, sys, io, json, time, math, uuid, hashlib, subprocess, threading, traceback, zipfile, importlib.util
from pathlib import Path

GUARDIAN_SOURCE = r'''"""Independent, standard-library runtime guardian; no torch/Colab imports needed."""
# Runtime unassignment endpoint: googlecolab/colabtools, google/colab/runtime.py (Apache-2.0).
import json
import os
import signal
import sys
import time
import urllib.request
from pathlib import Path


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
    while True:
        try:
            candidate=json.loads(state_path.read_text());last=candidate
            deadline=min(deadline,candidate['deadline_wall'])
        except Exception:pass
        now=time.time();pid=last.get('worker_pid');reason=None
        # A worker's completion flag is not the parent's durable completion acknowledgement.
        # Never terminate a successful worker merely because it has published its result.
        status=last.get('status')
        if now>=deadline:reason='absolute_deadline'
        elif status in ('finished','failed'):reason=status
        elif status=='worker_complete':
            grace=min(30.,max(.1,float(last.get('completion_grace_seconds',30.))))
            completed_at=last.get('worker_completed_wall',last.get('progress_wall',initial['progress_wall']))
            if now-completed_at>=grace:reason='completion_ack_timeout'
        elif now-last.get('progress_wall',initial['progress_wall'])>last.get('progress_timeout',900):reason='no_work_progress'
        elif pid and not alive(pid):reason='worker_exited'
        if reason:
            terminate(pid)
            event={'reason':reason,'timestamp_wall':now,'worker_pid':pid,'release_confirmed':False}
            try:event_path.write_text(json.dumps(event,indent=2)+'\n')
            except Exception:pass
            try:event['release_result']=release(test_file)
            except Exception as error:event['release_error']=type(error).__name__+': '+str(error)
            try:event_path.write_text(json.dumps(event,indent=2)+'\n')
            except Exception:pass
            return event
        time.sleep(poll)


if __name__=='__main__':
    supervise(sys.argv[1],sys.argv[2],sys.argv[3] if len(sys.argv)>3 else None)
'''

COORDINATOR_SOURCE = r'''"""Parent ACK after saved results; guardian primary, bounded notebook fallback."""
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
'''

GUARDIAN_SHA = '68f0baa5e7ae0eefbd3f5f4fc5e63505e008c64aefed14c471aa1cfe2c731257'
COORDINATOR_SHA = '614b5bab4cf1d5482b6b672fb79047709944167382414e4cdafc1f5cec20642b'
PACKAGE_SHA = 'bc5087eb5d028627726fd8673829ac9e84c98a67f360c6edd21d287e443b0a0a'
EXPECTED_NAMES = {'baseline_launch_checks.py', 'strict_backward_gate.py', 'release_coordination.py', 'prepare_baseline_cpu.py', 'launch_baseline.py', 'training_ops.py', 'archive_io.py', 'gpu_guardian_R2.py', 'training_core.py', 'baseline_hflm.py', 'evaluation_records.py', 'baseline_worker.py', 'evaluation_engine.py', 'baseline_runtime.py', 'baseline_asset_checks.py', 'stock_evaluation.py', 'restore_baseline_cpu.py'}

STARTED = time.time()
JOB = Path('/content') / ('task1_baseline_' + uuid.uuid4().hex)
JOB.mkdir()
STATE, EVENT = JOB / 'state.json', JOB / 'guardian_event.json'

def update_state(**changes):
    value = json.loads(STATE.read_text()) if STATE.exists() else {}
    value.update(changes)
    temporary = JOB / ('state_' + uuid.uuid4().hex + '.pending')
    temporary.write_text(json.dumps(value)); temporary.replace(STATE)

update_state(deadline_wall=STARTED + 110*60 - 45, progress_wall=STARTED,
             progress_timeout=120, status='setup', worker_pid=None)
guardian_file = JOB / 'guardian.py'
guardian_file.write_text(GUARDIAN_SOURCE)
assert hashlib.sha256(guardian_file.read_bytes()).hexdigest() == GUARDIAN_SHA
coord_file = JOB / 'release_coordination.py'
coord_file.write_text(COORDINATOR_SOURCE)
assert hashlib.sha256(coord_file.read_bytes()).hexdigest() == COORDINATOR_SHA
spec = importlib.util.spec_from_file_location('task1_release_coordinator', coord_file)
coordinator = importlib.util.module_from_spec(spec); spec.loader.exec_module(coordinator)
with (JOB / 'guardian_console.txt').open('x') as guard_log:
    guardian = subprocess.Popen([sys.executable, '-I', str(guardian_file), str(STATE), str(EVENT)],
        stdout=guard_log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)

SUCCESS = False
ERROR = None
try:
    probe = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                           capture_output=True, text=True, timeout=10, check=True)
    devices = probe.stdout.strip().splitlines()
    if len(devices) != 1 or 'L4' not in devices[0]:
        raise RuntimeError('This approved job requires one L4. Detected: ' + repr(devices))
    print('L4 detected. Repaired independent shutdown timer is running.', flush=True)
    print('Enter the CURRENT Colab Resources values. These are compute UNITS, not hours.', flush=True)
    balance = float(input('Available compute units: ').strip())
    rate = float(input('L4 compute units per hour: ').strip())
    if not math.isfinite(balance) or not math.isfinite(rate) or rate <= 0:
        raise ValueError('Invalid balance or usage rate')
    if balance > 78.26:
        raise RuntimeError('Balance exceeds the original budget reference; a CPU budget review is needed before spending.')
    room = balance - 23.26
    if room < 1.:
        raise RuntimeError('Less than one unit remains above the approved Task 1 reserve. Releasing for review.')
    allocation_seconds = min(110*60, room * 3600 / (rate * 1.10))
    deadline = STARTED + allocation_seconds - 45
    soft_wall = min(STARTED + 90*60, deadline - 180)
    if soft_wall < time.time() + 120:
        raise RuntimeError('Too little time remains to restore and run under the approved cap.')
    update_state(deadline_wall=deadline, progress_wall=time.time(), progress_timeout=120)
    print('Soft stop in %.1f minutes; absolute release deadline in %.1f minutes.' %
          ((soft_wall-time.time())/60, (deadline-time.time())/60), flush=True)
    from google.colab import files, drive
    print('Upload Task_1_Block_11_L4_Baseline.zip only.', flush=True)
    uploaded = files.upload()
    if len(uploaded) != 1:
        raise RuntimeError('Upload exactly the supplied ZIP.')
    payload = next(iter(uploaded.values()))
    if hashlib.sha256(payload).hexdigest() != PACKAGE_SHA:
        raise RuntimeError('Launch package fingerprint differs.')
    BUNDLE = JOB / 'code'; BUNDLE.mkdir()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if len(archive.namelist()) != len(EXPECTED_NAMES) or set(archive.namelist()) != EXPECTED_NAMES:
            raise RuntimeError('Unexpected package contents')
        for name in EXPECTED_NAMES:
            (BUNDLE / name).write_bytes(archive.read(name))
    update_state(progress_wall=time.time(), progress_timeout=120)
    print('Mount Drive when prompted; the independent timer remains active.', flush=True)
    drive.mount('/content/drive')
    ROOT = Path('/content/drive/MyDrive/ATML-PA-3/Task1')
    update_state(progress_wall=time.time(), progress_timeout=900)
    for name in EXPECTED_NAMES:
        sys.modules.pop(name[:-3], None)
    sys.path.insert(0, str(BUNDLE))
    spec = importlib.util.spec_from_file_location('task1_baseline_launcher', BUNDLE / 'launch_baseline.py')
    launcher = importlib.util.module_from_spec(spec); spec.loader.exec_module(launcher)
    saved = launcher.run({'root':str(ROOT), 'state':str(STATE), 'job':str(JOB),
        'started_wall':STARTED, 'balance':balance, 'rate':rate,
        'soft_wall':soft_wall, 'deadline_wall':deadline}, BUNDLE)
    SUCCESS = True
    print('TASK1_BLOCK11_SAVED — ' + saved['status'], flush=True)
except BaseException:
    ERROR = traceback.format_exc()
    print(ERROR, flush=True)
    (JOB / 'bootstrap_error.txt').write_text(ERROR)
    def save_error():
        try:
            target = Path('/content/drive/MyDrive/ATML-PA-3/Task1/logs') / (JOB.name + '_bootstrap_error.txt')
            target.write_text(ERROR)
        except Exception: pass
    thread = threading.Thread(target=save_error, daemon=True); thread.start(); thread.join(timeout=5)
finally:
    print('Requesting runtime release after saved results or an error.', flush=True)
    def release_from_notebook():
        from google.colab import runtime
        runtime.unassign()
    try:
        release_info = coordinator.finalize(STATE, EVENT, release_from_notebook, SUCCESS,
                                            wait_seconds=35, fallback_seconds=8)
        print('Release request status:', release_info, flush=True)
    except BaseException as error:
        print('Release coordination error:', type(error).__name__, flush=True)
        def fallback():
            try: release_from_notebook()
            except Exception as exc: print('Release fallback:', type(exc).__name__, flush=True)
        thread = threading.Thread(target=fallback, daemon=True); thread.start(); thread.join(timeout=8)
    print('If still connected after release attempts, use Runtime > Disconnect and delete runtime.', flush=True)
    print('Return the saved status/summary or error. Keep GPU off after this job.', flush=True)
