"""Notebook controller. Starts independent watchdog before the GPU worker; no torch."""
import importlib.util,json,os,signal,subprocess,sys,time
from datetime import datetime,timezone,timedelta
from pathlib import Path
_spec=importlib.util.spec_from_file_location('task3_gpu_watchdog',Path(__file__).with_name('watchdog.py'));watchdog=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(watchdog)

def run(bundle,root,directory,started,session_id):
    directory=Path(directory);worker=None;monitor=None
    config={'session_id':session_id,'directory':str(directory),'started':started,'started_utc':(datetime.now(timezone.utc)-timedelta(seconds=time.monotonic()-started)).isoformat(),'drive_status_directory':str(root/'live/session_status'),'limits':watchdog.LIMITS,'parent':watchdog.process_info(os.getpid())}
    config_path=directory/'config.json';watchdog.write_json(config_path,config)
    log=(directory/'monitor_console.txt').open('w')
    try:
        monitor=subprocess.Popen([sys.executable,'-I',str(bundle/'watchdog.py'),'--config',str(config_path)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        deadline=time.monotonic()+10
        while not (directory/'ready.json').exists():
            if monitor.poll() is not None or time.monotonic()>=deadline:raise RuntimeError('Independent watchdog failed to start')
            time.sleep(.05)
        worker=subprocess.Popen([sys.executable,'-I',str(bundle/'worker_entry.py'),'--root',str(root),'--bundle',str(bundle),'--config',str(config_path)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True)
        for line in worker.stdout:print(line,end='',flush=True)
        worker.wait()
        # Watchdog normally kills the worker after its saved terminal state.
        deadline=time.monotonic()+15
        while not (directory/'release_result.json').exists() and time.monotonic()<deadline:
            if monitor.poll() is not None:break
            time.sleep(.05)
        if (directory/'release_result.json').exists():
            result=json.loads((directory/'release_result.json').read_text());print(json.dumps({'watchdog_reason':result['reason'],'release_attempts':result['release_attempts'],'confirmed_physical_deallocation':False},indent=2),flush=True)
            if not any(x.get('backend_accepted') for x in result['release_attempts']):raise RuntimeError('Release request not accepted. Disconnect and delete runtime manually.')
            return result
        raise RuntimeError('Worker stopped; independent release status unavailable. Disconnect and delete runtime if connected.')
    except BaseException:
        if worker is not None and worker.poll() is None:
            try:os.killpg(worker.pid,signal.SIGKILL)
            except ProcessLookupError:pass
        # Give the tested primary watchdog its bounded status-copy window first.
        watchdog.write_json(directory/'abort.json',{'release_after_monotonic':time.monotonic()+8})
        raise
    finally:log.close()
