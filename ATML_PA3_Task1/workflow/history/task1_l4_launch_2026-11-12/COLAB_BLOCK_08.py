# Task 1 - Block 08: L4 scratch check; standalone after changing runtime.
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
        if last.get('status') in ('finished','failed','worker_complete'):reason=last['status']
        elif now>=deadline:reason='absolute_deadline'
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

STARTED = time.time()
JOB = Path('/content') / ('task1_l4_' + uuid.uuid4().hex)
JOB.mkdir()
STATE = JOB / 'state.json'

def update_state(**changes):
    value = json.loads(STATE.read_text()) if STATE.exists() else {}
    value.update(changes)
    pending = JOB / ('state_' + uuid.uuid4().hex + '.pending')
    pending.write_text(json.dumps(value))
    pending.replace(STATE)

update_state(deadline_wall=STARTED + 1200 - 45, progress_wall=STARTED,
             progress_timeout=120, status='setup', worker_pid=None)
guardian_file = JOB / 'guardian.py'
guardian_file.write_text(GUARDIAN_SOURCE)
assert hashlib.sha256(guardian_file.read_bytes()).hexdigest() == '0de74d1150c7ffedf43edfc0994cd00330e4beaba29b17b4b4ac4d57a02039be'
with (JOB / 'guardian_console.txt').open('x') as guard_log:
    guardian = subprocess.Popen(
        [sys.executable, '-I', str(guardian_file), str(STATE), str(JOB / 'guardian_event.json')],
        stdout=guard_log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        start_new_session=True,
    )

SUCCESS = False
ERROR = None
try:
    probe = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                           capture_output=True, text=True, timeout=10, check=True)
    devices = probe.stdout.strip().splitlines()
    if len(devices) != 1 or 'L4' not in devices[0]:
        raise RuntimeError('This approved job requires one L4. Detected: ' + repr(devices))
    print('L4 detected. Independent shutdown timer is running.', flush=True)
    print('Open Colab Resources and enter its CURRENT balance and L4 usage rate. Numbers only.', flush=True)
    balance = float(input('Available compute units: ').strip())
    rate = float(input('L4 usage rate in compute units per hour: ').strip())
    if not math.isfinite(balance) or not math.isfinite(rate) or rate <= 0:
        raise ValueError('Invalid balance or usage rate')
    if balance < 23.26 + 1:
        raise RuntimeError('Insufficient allowance above the approved reserve. Releasing for review.')
    deadline = STARTED + min(1200, 3600 / (rate * 1.10)) - 45
    if deadline <= time.time() + 60:
        raise RuntimeError('Too little preflight time remains under the one-unit limit.')
    update_state(deadline_wall=deadline, progress_wall=time.time(), progress_timeout=120)
    print('Maximum time remaining before cleanup: %.1f minutes' % ((deadline-time.time())/60), flush=True)
    from google.colab import files
    print('Upload Task_1_Block_08_L4_Launch.zip now.', flush=True)
    uploaded = files.upload()
    if len(uploaded) != 1:
        raise RuntimeError('Upload exactly the supplied ZIP.')
    payload = next(iter(uploaded.values()))
    if hashlib.sha256(payload).hexdigest() != '646b584da81dd4d357a0b9e7f576ac93f77a7f90818bc57c4599c6cdcd79f019':
        raise RuntimeError('Launch package fingerprint differs.')
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if archive.namelist() != ['launch_preflight.py']:
            raise RuntimeError('Unexpected launch package contents')
        launch_file = JOB / 'launch_preflight.py'
        launch_file.write_bytes(archive.read('launch_preflight.py'))
    update_state(progress_wall=time.time(), progress_timeout=120)
    spec = importlib.util.spec_from_file_location('task1_l4_launcher', launch_file)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    launcher.run({'state':str(STATE), 'job':str(JOB), 'started_wall':STARTED,
                  'balance':balance, 'rate':rate})
    SUCCESS = True
except BaseException:
    ERROR = traceback.format_exc()
    print(ERROR, flush=True)
    (JOB / 'bootstrap_error.txt').write_text(ERROR)
    def save_error():
        try:
            path = Path('/content/drive/MyDrive/ATML-PA-3/Task1/logs') / (JOB.name + '_bootstrap_error.txt')
            path.write_text(ERROR)
        except Exception:
            pass
    thread = threading.Thread(target=save_error, daemon=True)
    thread.start()
    thread.join(timeout=5)
finally:
    update_state(status='finished' if SUCCESS else 'failed', progress_wall=time.time())
    print('Requesting GPU runtime release. Scratch results are on Drive when saving succeeded.', flush=True)
    def release_from_notebook():
        try:
            from google.colab import runtime
            runtime.unassign()
        except Exception as exc:
            print('Notebook release request failed:', type(exc).__name__, flush=True)
    thread = threading.Thread(target=release_from_notebook, daemon=True)
    thread.start()
    thread.join(timeout=10)
    if thread.is_alive():
        print('The independent guardian is also attempting release.', flush=True)
    print('If still connected after release attempts finish, use Runtime > Disconnect and delete runtime.', flush=True)
