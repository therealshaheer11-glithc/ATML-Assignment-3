# Task 1 - Block 21: OFFICIAL Full FT training, update 0/resume to midpoint 625.
# Download the ZIP and copy this entire cell BEFORE enabling L4.
# No training. Normal 90-minute soft stop / 110-minute release limit, budget bounded.
import os, sys, io, json, time, math, uuid, hashlib, subprocess, threading, traceback, zipfile, importlib.util, shutil
from pathlib import Path

GUARDIAN_SOURCE = r'''"""R3 guardian: same bounded release policy plus nonblocking durable evidence."""
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
'''

EVIDENCE_SOURCE = r'''"""Best-effort Drive evidence must never block the independent release timer."""
import json
import threading
import time
import uuid
from pathlib import Path


class Evidence:
    def __init__(self, directory=None, run_id=None):
        self.directory = Path(directory) if directory else None
        self.run_id = run_id
        if run_id and ('/' in run_id or not run_id.startswith('task1_')):
            raise ValueError('Unsafe watchdog evidence name')
        self.pending = None
        self.last_attempt = 0.

    def _write(self, suffix, value):
        try:
            # Do not create directories before Drive is mounted.
            if not self.directory or not self.directory.is_dir(): return
            path = self.directory / (self.run_id + suffix)
            temporary = path.with_name(path.name + '.pending_' + uuid.uuid4().hex)
            temporary.write_text(json.dumps(value, indent=2) + '\n')
            temporary.replace(path)
        except Exception:
            pass

    def progress(self, state):
        now = time.monotonic()
        if not self.directory or now - self.last_attempt < 30:
            return
        if self.pending is not None and self.pending.is_alive(): return
        self.last_attempt = now
        snapshot = {'watchdog_observed_wall': time.time(), 'worker_state': dict(state),
                    'physical_release_verified': False}
        self.pending = threading.Thread(target=self._write, args=('_watchdog_progress.json', snapshot), daemon=True)
        self.pending.start()

    def terminal(self, suffix, event, wait_seconds=.25):
        if not self.directory: return
        thread = threading.Thread(target=self._write, args=(suffix, dict(event)), daemon=True)
        thread.start(); thread.join(wait_seconds)


def cpu_checks(directory):
    directory = Path(directory); directory.mkdir(parents=True)
    writer = Evidence(directory, 'task1_fixture')
    writer.progress({'progress_wall': 123., 'phase': 'fixture'})
    writer.pending.join(2)
    saved = json.loads((directory / 'task1_fixture_watchdog_progress.json').read_text())
    assert saved['worker_state']['progress_wall'] == 123.
    writer.terminal('_release_intent.json', {'reason': 'fixture'})
    assert json.loads((directory / 'task1_fixture_release_intent.json').read_text())['reason'] == 'fixture'
    gate = threading.Event()
    try:
        writer._write = lambda *args: gate.wait(5)
        start = time.monotonic()
        writer.terminal('_blocked.json', {}, wait_seconds=.03)
        assert time.monotonic() - start < .5
    finally: gate.set()
    missing = directory / 'not_mounted'
    Evidence(missing, 'task1_unmounted').terminal('_x.json', {})
    assert not missing.exists()
    return {'snapshot_preserves_actual_worker_progress_time': True,
            'release_intent_written_before_release': True,
            'blocked_evidence_write_is_bounded': True,
            'unmounted_Drive_path_is_not_created': True}
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

PACKAGE_SHA = '1def51a0f22e6e8467de8495e50fd6e2013321be321fbc79c0207fcf1e3806ee'
EXPECTED_NAMES = {'core/baseline_runtime.py', 'core/restore_training_cpu.py', 'core/training_runtime_checks.py', 'launch/full_training_worker.py', 'core/APPROVED_PLAN.json', 'launch/launch_training_full.py', 'core/training_ops.py', 'core/gpu_guardian_R3.py', 'core/training_core.py', 'core/training_checkpoints.py', 'core/archive_io.py', 'CORE_FILES.json', 'core/training_loop.py', 'core/official_training.py', 'launch/training_parent_checks.py', 'launch/training_launch_support.py', 'core/training_loop_checks.py', 'core/watchdog_evidence.py', 'core/release_coordination.py', 'core/restore_baseline_cpu.py', 'core/evaluation_records.py', 'launch/launch_baseline.py', 'core/prepare_training_cpu.py'}
EMBEDDED_SHAS = {'GUARDIAN_SOURCE': '12c7dfa4637bea83e7aacd821743ebdcb626c291ff8b14d1374befa37fd2a06d', 'EVIDENCE_SOURCE': '30688a268a2f2441c5dbe0adeabd78afad2a5d76af5c5e0747e31ed30de3618e', 'COORDINATOR_SOURCE': '614b5bab4cf1d5482b6b672fb79047709944167382414e4cdafc1f5cec20642b'}

STARTED = time.time()
ROOT = Path('/content/drive/MyDrive/ATML-PA-3/Task1')
JOB = Path('/content') / ('task1_full_ft_' + uuid.uuid4().hex)
JOB.mkdir()
STATE, EVENT = JOB / 'state.json', JOB / 'guardian_event.json'

def update_state(**changes):
    value = json.loads(STATE.read_text()) if STATE.exists() else {}
    value.update(changes)
    pending = JOB / ('state_' + uuid.uuid4().hex + '.pending')
    pending.write_text(json.dumps(value)); pending.replace(STATE)

update_state(deadline_wall=STARTED+110*60-45, progress_wall=STARTED,
    progress_timeout=120, status='setup', worker_pid=None, parent_pid=os.getpid(),
    evidence_directory=str(ROOT/'logs'), run_id=JOB.name)
for source_name, filename, source in (
    ('GUARDIAN_SOURCE','guardian.py',GUARDIAN_SOURCE),
    ('EVIDENCE_SOURCE','watchdog_evidence.py',EVIDENCE_SOURCE),
    ('COORDINATOR_SOURCE','release_coordination.py',COORDINATOR_SOURCE)):
    path=JOB/filename; path.write_text(source)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==EMBEDDED_SHAS[source_name]
spec=importlib.util.spec_from_file_location('task1_v2_release_coordinator',JOB/'release_coordination.py')
coordinator=importlib.util.module_from_spec(spec);spec.loader.exec_module(coordinator)
with (JOB/'guardian_console.txt').open('x') as guard_log:
    guardian=subprocess.Popen([sys.executable,'-I',str(JOB/'guardian.py'),str(STATE),str(EVENT)],
        stdin=subprocess.DEVNULL,stdout=guard_log,stderr=subprocess.STDOUT,start_new_session=True)

SUCCESS=False
ERROR=None
try:
    if shutil.which('nvidia-smi') is None:
        raise RuntimeError('No NVIDIA runtime is available. Select L4 before running this GPU cell.')
    probe=subprocess.run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],
        capture_output=True,text=True,timeout=10,check=True)
    devices=probe.stdout.strip().splitlines()
    if devices != ['NVIDIA L4']:
        raise RuntimeError('This approved training job requires one L4. Detected: '+repr(devices))
    print('L4 detected. Independent 110-minute training release timer is running.',flush=True)
    print('OFFICIAL Full FT training: midpoint update 625 / 20,000 examples, or a saved time-limit pause.',flush=True)
    print('Drive storage recorded: 400 GB total - 67.15 GB used = 332.85 GB free (user report).',flush=True)
    print('Enter CURRENT Resources values in compute UNITS, not hours.',flush=True)
    balance=float(input('Available compute units: ').strip())
    rate=float(input('L4 compute units per hour: ').strip())
    if not math.isfinite(balance) or not math.isfinite(rate) or rate<=0:
        raise ValueError('Invalid balance or rate')
    if balance>78.26:
        raise RuntimeError('Wallet increased above the initial reference; review the remaining approved allowance on CPU.')
    if balance-23.26<1.:
        raise RuntimeError('Less than one unit remains in the approved allowance. Releasing for budget review.')
    allocation=min(110*60,(balance-23.26)*3600/(rate*1.1))
    deadline=STARTED+allocation-45
    soft_wall=min(STARTED+90*60,deadline-20*60)
    if soft_wall<time.time()+120:
        raise RuntimeError('Too little bounded session time remains after setup prompts.')
    update_state(deadline_wall=deadline,progress_wall=time.time(),progress_timeout=120)
    print('Training work cutoff in %.1f minutes; release timer in %.1f minutes.'%
        ((soft_wall-time.time())/60,(deadline-time.time())/60),flush=True)
    from google.colab import files,drive
    print('Upload Task_1_Block_21_L4_Full_FT.zip only.',flush=True)
    uploaded=files.upload()
    if len(uploaded)!=1:raise RuntimeError('Upload exactly the supplied Full FT ZIP.')
    payload=next(iter(uploaded.values()))
    if hashlib.sha256(payload).hexdigest()!=PACKAGE_SHA:raise RuntimeError('Full FT ZIP fingerprint differs.')
    BUNDLE=JOB/'code';BUNDLE.mkdir()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if len(archive.namelist())!=len(EXPECTED_NAMES) or set(archive.namelist())!=EXPECTED_NAMES:
            raise RuntimeError('Unexpected Full FT ZIP contents')
        for name in EXPECTED_NAMES:
            path=BUNDLE/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(archive.read(name))
    update_state(progress_wall=time.time(),progress_timeout=120)
    drive.mount('/content/drive')
    update_state(progress_wall=time.time(),progress_timeout=900)
    for name in EXPECTED_NAMES:
        if name.endswith('.py'):sys.modules.pop(Path(name).stem,None)
    sys.path.insert(0,str(BUNDLE/'core'));sys.path.insert(0,str(BUNDLE/'launch'))
    spec=importlib.util.spec_from_file_location('task1_full_ft_launcher',BUNDLE/'launch/launch_training_full.py')
    launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)
    saved=launcher.run({'root':str(ROOT),'job':str(JOB),'state':str(STATE),
        'started_wall':STARTED,'balance':balance,'rate':rate,'soft_wall':soft_wall,
        'deadline_wall':deadline,'Drive_free_GB':332.85},BUNDLE)
    SUCCESS=True
    print('TASK1_BLOCK21_SAVED — '+saved['status'],flush=True)
    print('A PARTIAL_SAVED result is a resumable training pause. Update 625 is the Full FT midpoint.',flush=True)
except BaseException:
    ERROR=traceback.format_exc();print(ERROR,flush=True)
    (JOB/'bootstrap_error.txt').write_text(ERROR)
    def preserve_error():
        try:
            directory=ROOT/'logs'
            if directory.is_dir():(directory/(JOB.name+'_bootstrap_error.txt')).write_text(ERROR)
        except Exception:pass
    thread=threading.Thread(target=preserve_error,daemon=True);thread.start();thread.join(3)
finally:
    print('Requesting GPU runtime release after this bounded Full FT training session.',flush=True)
    def notebook_release():
        from google.colab import runtime
        runtime.unassign()
    try:
        release_info=coordinator.finalize(STATE,EVENT,notebook_release,SUCCESS,wait_seconds=35,fallback_seconds=8)
        print('Release request status:',release_info,flush=True)
    except BaseException as error:
        print('Release coordination:',type(error).__name__,flush=True)
        def fallback():
            try:notebook_release()
            except Exception as exc:print('Release fallback:',type(exc).__name__,flush=True)
        thread=threading.Thread(target=fallback,daemon=True);thread.start();thread.join(8)
    print('If still connected after release attempts, use Runtime > Disconnect and delete runtime.',flush=True)
    print('Keep GPU off and return the complete Full FT result JSON/status or error.',flush=True)
