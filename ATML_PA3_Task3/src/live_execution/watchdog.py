"""Independent process watchdog. Production release mirrors Colab's backend POST.
Attribution: https://github.com/googlecolab/colabtools/blob/main/google/colab/runtime.py
No torch, model, CUDA, notebook frontend, or Drive access in the monitoring loop.
"""
import argparse,json,math,os,signal,subprocess,sys,time,urllib.request,uuid
from pathlib import Path

LIMITS={'new_calls':1800.,'call':600.,'setup':600.,'saving':60.,'absolute':2700.,'release_reserve':15.,'salvage':5.,'release_timeout':3.,'poll':.2}
PHASES={'setup','loading','generating','saving','idle','finished','error'}

def write_json(path,obj):
    path=Path(path);pending=path.with_name(path.name+'.tmp_'+uuid.uuid4().hex)
    pending.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');pending.replace(path)
def identity(pid):
    pid=int(pid);p=Path(f'/proc/{pid}/stat')
    if Path('/proc').is_dir():
        try:
            fields=p.read_text().rsplit(')',1)[1].split()
            if fields[0]=='Z':return None
            return 'proc:'+fields[19]
        except (OSError,IndexError):return None
    # Portable local CPU-test fallback; Colab uses /proc.
    try:
        r=subprocess.run(['ps','-p',str(pid),'-o','stat=','-o','lstart='],capture_output=True,text=True,timeout=1)
        parts=r.stdout.strip().split(maxsplit=1)
        return 'ps:'+parts[1] if r.returncode==0 and len(parts)==2 and not parts[0].startswith('Z') else None
    except (OSError,subprocess.TimeoutExpired):return None

def alive(info):
    try:return bool(isinstance(info,dict) and info.get('identity') and identity(info['pid'])==info['identity'])
    except (ValueError,TypeError,KeyError):return False
def process_info(pid):return {'pid':int(pid),'identity':identity(pid)}
def can_start_call(start,now,limits=LIMITS):return now-start<limits['new_calls']
def reason(config,state,now,parent_alive=True,worker_alive=True):
    l=config['limits'];start=config['started']
    if now-start>=l['absolute']-l['release_reserve']:return 'absolute_deadline_reserve'
    if not parent_alive:return 'notebook_parent_lost'
    if state is None:return 'setup_timeout' if now-start>=l['setup'] else None
    if state.get('session_id')!=config['session_id'] or state.get('phase') not in PHASES:return 'invalid_state'
    began=state.get('phase_started')
    if not isinstance(began,(int,float)) or not math.isfinite(began) or began<start-1 or began>now+1:return 'invalid_state'
    phase=state['phase'];elapsed=now-began
    if phase in ['finished','error']:return phase
    if state.get('worker') and not worker_alive:return 'worker_exited_without_terminal_state'
    if phase=='generating' and elapsed>=l['call']:return 'call_timeout'
    if phase=='saving' and elapsed>=l['saving']:return 'save_timeout'
    if phase in ['setup','loading','idle'] and elapsed>=l['setup']:return phase+'_timeout'
    return None

def kill_worker(info):
    result={'requested':False,'identity_matched':False}
    if not alive(info):return result
    result['identity_matched']=True
    try:
        # Worker must own its new process group. Never signal the notebook's group.
        if os.getpgid(info['pid'])!=info['pid']:raise RuntimeError('Worker is not process-group leader')
        os.killpg(info['pid'],signal.SIGKILL);result['requested']=True
    except ProcessLookupError:pass
    except Exception as e:result['error_type']=type(e).__name__
    return result

def release(address,timeout):
    # TBE_RUNTIME_ADDR is a host:port authority supplied by Colab, not a user URL.
    import urllib.parse
    p=urllib.parse.urlsplit('http://'+address)
    if not p.hostname or p.username or p.password or p.path or p.query or p.fragment:raise ValueError('Invalid runtime service authority')
    request=urllib.request.Request('http://'+address+'/unassign',data=b'',method='POST')
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request,timeout=timeout) as response:
        return {'requested':True,'http_status':response.status,'backend_accepted':response.status==200,'confirmed_physical_deallocation':False}

def salvage(config):
    if config.get('self_test') and config.get('self_test_salvage_hang'):time.sleep(30)
    dest=Path(config['drive_status_directory']);dest.mkdir(parents=True,exist_ok=True)
    raw=(Path(config['directory'])/'shutdown.json').read_bytes();p=dest/(config['session_id']+'.json')
    if p.exists():
        if p.read_bytes()!=raw:raise RuntimeError('Existing session status differs')
    else:
        with p.open('xb') as f:f.write(raw);f.flush()
    if p.read_bytes()!=raw:raise IOError('Status readback differs')

def bounded_salvage(config,config_path):
    proc=None
    try:
        proc=subprocess.Popen([sys.executable,'-I',str(Path(__file__).resolve()),'--config',str(config_path),'--salvage'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
        code=proc.wait(timeout=config['limits']['salvage']);return {'completed':code==0,'exit_code':code}
    except subprocess.TimeoutExpired:
        try:os.killpg(proc.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        try:proc.wait(timeout=.1)
        except subprocess.TimeoutExpired:pass
        return {'completed':False,'timeout':True}
    except Exception as e:return {'completed':False,'error_type':type(e).__name__}

def validate_config(config):
    if config.get('self_test'):
        if os.environ.get('TASK3_SUPERVISION_SELFTEST')!='1' or os.environ.get('TBE_RUNTIME_ADDR'):raise ValueError('Self-test cannot access native runtime release')
        address=config['test_release_address']
        if not address.startswith('127.0.0.1:'):raise ValueError('Self-test release must be loopback')
    else:
        if config['limits']!=LIMITS:raise ValueError('Production limits changed')
        if not Path(config['directory']).resolve().is_relative_to(Path('/content/atml_pa3_task3').resolve()):raise ValueError('Watchdog control files must be on local /content storage')
        address=os.environ.get('TBE_RUNTIME_ADDR')
        if not address:raise RuntimeError('Colab runtime management service absent')
    return address

def monitor(config_path):
    config=json.loads(Path(config_path).read_text());directory=Path(config['directory']);address=validate_config(config)
    state=None;registered_worker=None;why='watchdog_error';details={}
    try:
        write_json(directory/'ready.json',{'session_id':config['session_id'],'pid':os.getpid(),'independent_process':True})
        while True:
            try:
                state=json.loads((directory/'state.json').read_text())
                if not isinstance(state,dict):raise ValueError('State must be an object')
            except FileNotFoundError:state=None
            except Exception:why='invalid_state';break
            if state and state.get('worker'):
                info=state['worker']
                if registered_worker is None:
                    if not alive(info) or info['pid']==config['parent']['pid'] or os.getpgid(info['pid'])!=info['pid']:
                        why='invalid_worker_registration';break
                    registered_worker=info
                    write_json(directory/'worker_ack.json',{'session_id':config['session_id'],'worker':info,'watchdog_pid':os.getpid()})
                elif info!=registered_worker:
                    why='worker_identity_changed';break
            why=reason(config,state,time.monotonic(),alive(config['parent']),alive(registered_worker) if registered_worker else True)
            if why:break
            time.sleep(config['limits']['poll'])
    except BaseException as error:details={'watchdog_error_type':type(error).__name__}
    finally:
        # Drive copying is isolated and bounded; a failed save cannot block release.
        try:stopped=kill_worker(registered_worker) if registered_worker else {'requested':False}
        except Exception as error:stopped={'requested':False,'error_type':type(error).__name__}
        details.update(reason=why,last_state=state,session_id=config['session_id'],watchdog_pid=os.getpid(),worker_stop=stopped,confirmed_physical_deallocation=False)
        try:write_json(directory/'shutdown.json',details)
        except Exception:pass
        details['salvage']=bounded_salvage(config,config_path)
        attempts=[]
        for _ in range(2):
            try:
                response=release(address,config['limits']['release_timeout']);attempts.append(response)
                if response['backend_accepted']:break
            except Exception as e:attempts.append({'requested':True,'error_type':type(e).__name__,'confirmed_physical_deallocation':False})
        details['release_attempts']=attempts
        try:write_json(directory/'release_result.json',details)
        except Exception:pass
    return details

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--salvage',action='store_true');args=p.parse_args()
    c=json.loads(Path(args.config).read_text())
    if args.salvage:
        validate_config(c);salvage(c)
    else:monitor(args.config)
