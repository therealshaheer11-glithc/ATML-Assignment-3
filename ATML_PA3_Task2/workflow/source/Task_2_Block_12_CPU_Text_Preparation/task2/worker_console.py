"""Forward child stdout/stderr into notebook output and a local diagnostic log."""
from pathlib import Path
import os
import queue
import signal
import subprocess
import threading
import time


def run_logged_worker(command, environment, guard, log_path, snapshot=None, snapshot_interval=5):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True,exist_ok=True)
    messages = queue.Queue()
    process = subprocess.Popen(command,env=environment,start_new_session=True,
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',bufsize=1)
    guard.attach(process)
    def read_output():
        try:
            for line in process.stdout:
                messages.put(('line',line))
        except BaseException as error:
            messages.put(('reader_error',repr(error)))
        finally:
            messages.put(('end',None))
    reader = threading.Thread(target=read_output,daemon=True)
    reader.start()
    last_snapshot = time.monotonic()
    exited_at = None
    try:
        with log_path.open('w',encoding='utf-8') as log:
            while True:
                if time.monotonic()-guard.started >= guard.hard_seconds:
                    raise TimeoutError('GPU worker reached the controller deadline.')
                try:
                    kind,value = messages.get(timeout=0.2)
                except queue.Empty:
                    kind,value = None,None
                if kind == 'line':
                    log.write(value)
                    log.flush()
                    print(value,end='',flush=True)
                elif kind == 'reader_error':
                    raise IOError('Worker console capture failed: '+value)
                elif kind == 'end':
                    break
                if snapshot is not None and time.monotonic()-last_snapshot >= snapshot_interval:
                    log.flush()
                    snapshot(log_path)
                    last_snapshot = time.monotonic()
                if process.poll() is not None:
                    exited_at = time.monotonic() if exited_at is None else exited_at
                    if time.monotonic()-exited_at > 5:
                        # A descendant can inherit the pipe after its parent exits.
                        try:
                            os.killpg(process.pid,signal.SIGKILL)
                        except ProcessLookupError:
                            pass
            code = process.wait(timeout=5)
        reader.join(timeout=1)
        if snapshot is not None:
            snapshot(log_path)
        return code
    finally:
        if process.poll() is None:
            guard.stop_worker()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        process.stdout.close()
