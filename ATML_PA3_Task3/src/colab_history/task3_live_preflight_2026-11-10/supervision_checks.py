"""Real CPU subprocess/loopback-HTTP tests; never native Colab unassignment."""
import hashlib,http.server,json,os,signal,subprocess,sys,tempfile,threading,time,uuid
from pathlib import Path
import watchdog
import live_protocol

def run_checks():
    unit=[]
    def check(name,value):
        if not value:raise AssertionError(name)
        unit.append(name)
    start=1000.;cfg={'started':start,'limits':watchdog.LIMITS,'session_id':'unit'}
    state={'phase':'generating','phase_started':1100.,'session_id':'unit'}
    check('call_allowed_before30min',watchdog.can_start_call(start,2799.999))
    check('no_new_call_at30min',not watchdog.can_start_call(start,2800.))
    check('no_new_call_after30min',not watchdog.can_start_call(start,2800.001))
    check('active_call_continues_at_soft_limit',watchdog.reason(cfg,{**state,'phase_started':2750},2800.) is None)
    check('call_before10min',watchdog.reason(cfg,state,1699.999) is None)
    check('call_timeout_at10min',watchdog.reason(cfg,state,1700.)=='call_timeout')
    check('absolute_shutdown_reserves_release_time',watchdog.reason(cfg,state,3685.)=='absolute_deadline_reserve')
    check('saving_timeout',watchdog.reason(cfg,{**state,'phase':'saving'},1160.)=='save_timeout')
    check('setup_timeout',watchdog.reason(cfg,None,1600.)=='setup_timeout')
    check('parent_loss',watchdog.reason(cfg,state,1101.,False)=='notebook_parent_lost')
    check('nonfinite_state_time',watchdog.reason(cfg,{**state,'phase_started':float('nan')},1101.)=='invalid_state')
    check('unknown_state_phase',watchdog.reason(cfg,{**state,'phase':'unknown'},1101.)=='invalid_state')
    check('worker_exit',watchdog.reason(cfg,{**state,'worker':{'pid':1}},1101.,True,False)=='worker_exited_without_terminal_state')
    check('malformed_worker_identity_does_not_raise',not watchdog.alive({'bad':'value'}))
    cases=[('last_nested_box',r'A \boxed{1} then \boxed{\frac{2}{3}}',r'\frac{2}{3}'),('empty_last_box',r'\boxed{7} \boxed{ }',None),('broken_last_box',r'\boxed{7} \boxed{unfinished',None),('no_numeric_fallback','The answer is 42.',None),('trim_box',r'\boxed{ 2 }','2')]
    for name,text,answer in cases:check(name,live_protocol.extract_last_box(text)['answer']==answer)
    try:live_protocol.check_context([1]*32768,1)
    except ValueError:unit.append('context_overflow_rejected')
    else:raise AssertionError('context overflow accepted')
    requests=[]
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append({'path':self.path,'method':'POST'});self.send_response(200);self.end_headers();self.wfile.write(b'ok')
        def log_message(self,*args):pass
    server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();integrations=[]
    try:
        for name,phase,salvage_hang in [('completion','finished',False),('worker_error','error',False),('generation_hang','generating',False),('save_hang','saving',False),('absolute_deadline','idle',False),('save_cannot_block_release','finished',True),('malformed_state','malformed',False),('parent_lost','idle',False)]:
            with tempfile.TemporaryDirectory(prefix='task3_watchdog_test_') as raw:
                d=Path(raw);directory=d/'control';directory.mkdir();dest=d/'saved_status';dest.mkdir();previous=d/'verified_call.json';previous.write_bytes(b'previous verified output');before=hashlib.sha256(previous.read_bytes()).hexdigest()
                worker=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                watcher=None;began=time.monotonic();limits={**watchdog.LIMITS,'new_calls':1.,'call':.6,'setup':1.5,'saving':.5,'absolute':2.,'release_reserve':.5,'salvage':.25,'release_timeout':.5,'poll':.02}
                try:
                    info=watchdog.process_info(worker.pid)
                    config={'session_id':uuid.uuid4().hex,'directory':str(directory),'drive_status_directory':str(dest),'started':began,'limits':limits,'parent':watchdog.process_info(os.getpid()),'self_test':True,'test_release_address':'127.0.0.1:'+str(server.server_port),'self_test_salvage_hang':salvage_hang}
                    if name=='parent_lost':config['parent']={'pid':999999999,'identity':'missing'}
                    state={'session_id':config['session_id'],'phase':phase,'phase_started':began,'worker':info}
                    watchdog.write_json(directory/'state.json',state if phase!='malformed' else {**state,'phase':'setup'})
                    # Malformed-state scenario corrupts state only after worker registration.
                    path=directory/'config.json';watchdog.write_json(path,config);env=os.environ.copy();env.pop('TBE_RUNTIME_ADDR',None);env['TASK3_SUPERVISION_SELFTEST']='1'
                    count=len(requests);watcher=subprocess.Popen([sys.executable,'-I',str(Path(watchdog.__file__).resolve()),'--config',str(path)],env=env,start_new_session=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
                    if phase=='malformed':
                        ack_deadline=time.monotonic()+2
                        while not (directory/'worker_ack.json').exists() and time.monotonic()<ack_deadline:time.sleep(.01)
                        check('worker_registration_ack_before_work',(directory/'worker_ack.json').exists())
                        watchdog.write_json(directory/'state.json',['invalid state object'])
                    stdout,stderr=watcher.communicate(timeout=8)
                    if watcher.returncode:raise AssertionError(name+': '+stderr)
                    result=json.loads((directory/'release_result.json').read_text());expected={'completion':'finished','worker_error':'error','generation_hang':'call_timeout','save_hang':'save_timeout','absolute_deadline':'absolute_deadline_reserve','save_cannot_block_release':'finished','malformed_state':'invalid_state','parent_lost':'notebook_parent_lost'}[name]
                    check(name+'_reason',result['reason']==expected);check(name+'_backend_request',len(requests)==count+1 and requests[-1]=={'path':'/unassign','method':'POST'})
                    check(name+'_accepted_not_deallocation_claim',result['release_attempts'][-1]['backend_accepted'] and not result['confirmed_physical_deallocation'])
                    worker.wait(timeout=2);check(name+'_worker_stopped',worker.returncode is not None);check(name+'_prior_save_unchanged',hashlib.sha256(previous.read_bytes()).hexdigest()==before)
                    if salvage_hang:check('hung_status_copy_was_bounded',result['salvage'].get('timeout') is True)
                    integrations.append({'case':name,'reason':result['reason'],'elapsed_seconds':round(time.monotonic()-began,3),'worker_stopped':True,'loopback_release_POST_observed':True,'prior_saved_bytes_unchanged':True})
                finally:
                    for proc in [worker,watcher]:
                        if proc is not None and proc.poll() is None:
                            try:os.killpg(proc.pid,signal.SIGKILL)
                            except ProcessLookupError:pass
                            proc.wait(timeout=2)
    finally:server.shutdown();server.server_close()
    return {'passed_assertions':len(unit),'integration_scenarios':integrations,'unit_and_integration_assertions':unit,'native_runtime_release_executed':False,'GPU_used':False,'scope':'Real CPU child processes killed; release POST goes only to loopback test server. Actual GPU/OOM/CUDA-kernel hang and physical VM deallocation not established.'}

if __name__=='__main__':print(json.dumps(run_checks(),indent=2))
