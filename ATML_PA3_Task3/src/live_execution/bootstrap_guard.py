"""Pre-upload absolute guard, independent of notebook and packaged watchdog.
Production performs only Colab's native runtime-service POST. Test mode is loopback only.
"""
import json,os,signal,sys,time,urllib.request
from pathlib import Path

def identity(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return None if fields[0]=='Z' else 'proc:'+fields[19]
    except (OSError,IndexError):return None

def main(directory,started,session_id,test=False):
    directory=Path(directory)
    if test:
        if os.environ.get('TASK3_BOOTSTRAP_SELFTEST')!='1' or os.environ.get('TBE_RUNTIME_ADDR'):raise ValueError('Unsafe bootstrap test')
        address=os.environ['TASK3_TEST_RELEASE_ADDR'];limit=float(os.environ['TASK3_TEST_LIMIT']);setup=limit;poll=.02
        if not address.startswith('127.0.0.1:'):raise ValueError('Nonlocal bootstrap test')
    else:
        address=os.environ.get('TBE_RUNTIME_ADDR');limit=2685.;setup=600.;poll=.2
        if not address:raise RuntimeError('Runtime management address absent')
    why='guard_error'
    try:
        (directory/'bootstrap_ready.json').write_text(json.dumps({'session_id':session_id,'pid':os.getpid()}))
        why='absolute_deadline'
        while time.monotonic()-started<limit:
            if (directory/'abort.json').exists():
                try:when=json.loads((directory/'abort.json').read_text()).get('release_after_monotonic',0)
                except Exception:when=0
                if time.monotonic()>=when:why='explicit_abort';break
            if not (directory/'ready.json').exists() and time.monotonic()-started>=setup:why='preparation_timeout';break
            time.sleep(poll)
    except BaseException:why='guard_error'
    # The frozen primary watchdog owns registration; keep a separate backup kill path.
    try:
        ack=json.loads((directory/'worker_ack.json').read_text());info=ack['worker']
        if ack['session_id']==session_id and info['identity']==identity(info['pid']) and os.getpgid(info['pid'])==info['pid']:os.killpg(info['pid'],signal.SIGKILL)
    except Exception:pass
    result={'session_id':session_id,'reason':why,'confirmed_physical_deallocation':False,'release_attempts':[]}
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for _ in range(2):
        try:
            request=urllib.request.Request('http://'+address+'/unassign',data=b'',method='POST')
            with opener.open(request,timeout=3) as response:status=response.status
            result['release_attempts'].append({'status':status,'backend_accepted':status==200})
            if status==200:break
        except Exception as e:result['release_attempts'].append({'error_type':type(e).__name__})
    try:(directory/'bootstrap_release.json').write_text(json.dumps(result,indent=2))
    except Exception:pass

if __name__=='__main__':main(sys.argv[1],float(sys.argv[2]),sys.argv[3],len(sys.argv)>4 and sys.argv[4]=='--self-test')
