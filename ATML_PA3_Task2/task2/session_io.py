"""Diagnostic persistence that does not depend on torch, numpy or Pillow imports."""
from contextlib import contextmanager
from pathlib import Path
import json
import signal
import time
import uuid


@contextmanager
def bounded_io(seconds,label):
    handler=signal.getsignal(signal.SIGALRM)
    timer=signal.getitimer(signal.ITIMER_REAL)
    started=time.monotonic()
    def expired(signum,frame):
        raise TimeoutError(label+' exceeded its time limit.')
    signal.signal(signal.SIGALRM,expired)
    signal.setitimer(signal.ITIMER_REAL,seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,handler)
        if timer[0]>0:
            signal.setitimer(signal.ITIMER_REAL,max(0.001,timer[0]-(time.monotonic()-started)),timer[1])


def save_bytes(path,content):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=content:
            raise IOError('Different diagnostic artifact retained: '+str(path))
        return
    temporary=path.with_name(path.name+'.part_'+uuid.uuid4().hex)
    try:
        temporary.write_bytes(content)
        assert temporary.read_bytes()==content
        temporary.replace(path)
        assert path.read_bytes()==content
    finally:
        temporary.unlink(missing_ok=True)


def save_json(path,value):
    save_bytes(path,(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode())
