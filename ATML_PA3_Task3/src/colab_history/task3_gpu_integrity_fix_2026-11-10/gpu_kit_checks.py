"""CPU-only tests using real call files, fake model execution and loopback release."""
import contextlib,copy,hashlib,http.server,importlib.util,io,json,os,shutil,signal,subprocess,sys,tempfile,threading,time,types,unittest
from pathlib import Path
from unittest import mock
import live_protocol,live_store as store,gpu_worker,watchdog,gpu_engine

BASE=Path(__file__).resolve().parent
class Tokenizer:
    def __init__(self,path):
        from tokenizers import Tokenizer as Raw
        self.raw=Raw.from_file(str(path))
    def apply_chat_template(self,messages,tokenize=True,**kwargs):
        text=live_protocol.expected_native_text(messages)
        return self.raw.encode(text,add_special_tokens=False).ids if tokenize else text
    def decode(self,ids,skip_special_tokens=False,**kwargs):return self.raw.decode(ids,skip_special_tokens=skip_special_tokens)
class Engine:
    def __init__(self,tok,protocol):self.tokenizer=tok;self.protocol=protocol;self.calls=[];self.fail=False;self.soft=False
    def infer(self,messages,cap,seed,phase_started):
        if self.soft:raise store.SoftLimitReached('test')
        if self.fail:raise RuntimeError('simulated worker error/OOM')
        self.calls.append((messages,cap,seed));encoded=live_protocol.encode_messages(self.tokenizer,messages);ids=self.tokenizer.raw.encode(r'\boxed{2}',add_special_tokens=False).ids+[151645];text=self.tokenizer.decode(ids,skip_special_tokens=True)
        return {**encoded,'input_tokens':len(encoded['input_ids']),'output_ids':ids,'raw_decoded_text':self.tokenizer.decode(ids),'generated_text':text,'termination':store.termination(ids,cap),'extraction':live_protocol.extract_last_box(text),'inference_seconds':1.0,'call_wall_seconds':1.1,'peak_allocated_bytes':100,'peak_reserved_bytes':120,'started_before_soft_limit':True,'completed_before_call_deadline':True,'generation_options':live_protocol.generation_options(self.protocol,cap)}

class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures=Path(os.environ.get('TASK3_TEST_FIXTURES',str(BASE)));cls.fixture=json.loads((fixtures/'TEST_LIVE_INPUTS.json').read_text());cls.rows=cls.fixture['questions'][:2];cls.protocol=json.loads((fixtures/'TEST_PROTOCOL.json').read_text());cls.tok=Tokenizer(fixtures/'TEST_tokenizer.json')
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.work=Path(self.tmp.name)/'work';self.work.mkdir();self.control_dir=Path(self.tmp.name)/'control';self.control_dir.mkdir()
        for n in ['calls','sessions','attempts']:(self.work/n).mkdir()
        self.config={'session_id':'test_session','directory':str(self.control_dir),'started':time.monotonic()};self.control=gpu_worker.Control(self.config);self.session_sha=store.save(self.work/'sessions/test_session.json',store.encode({'session_id':'test_session'}));self.engine=Engine(self.tok,self.protocol)
    def tearDown(self):self.tmp.cleanup()
    def run_schedule(self,previous=None,clock=time.monotonic):return gpu_worker.schedule(self.work,self.rows,self.protocol,'recipe','env',self.session_sha,self.control,self.engine,previous if previous is not None else {},clock)
    def scan(self):return store.scan(self.work,self.rows,'recipe','env',self.protocol,self.tok)
    def test_resume_and_shared_initial(self):
        clock=lambda:self.config['started']+600*len(self.engine.calls)
        r=self.run_schedule(clock=clock);self.assertEqual(r['reason'],'soft_limit');self.assertEqual(r['verified_calls'],3);self.assertEqual(len(self.engine.calls),3)
        before={p.name:p.read_bytes() for p in (self.work/'calls').iterdir()};previous=self.scan();self.engine.calls=[];r=self.run_schedule(previous);self.assertEqual(r['verified_calls'],10);self.assertEqual(len(self.engine.calls),7)
        for name,raw in before.items():self.assertEqual((self.work/'calls'/name).read_bytes(),raw)
        previous=self.scan();revision=previous[store.call_key(self.rows[0]['sample_id'],'revision')]['record'];initial=previous[store.call_key(self.rows[0]['sample_id'],'parallel_1')]
        self.assertEqual(revision['messages'][2]['content'],initial['record']['generated_text']);self.assertEqual(revision['shared_initial']['envelope_sha256'],initial['envelope_sha256']);self.assertEqual(len(self.scan()),10)
        self.engine.calls=[];self.run_schedule(previous);self.assertEqual(self.engine.calls,[])
    def test_corruption_pending_recovery_and_conflict(self):
        self.run_schedule();key=store.call_key(self.rows[0]['sample_id'],'long');p=self.work/'calls'/(key+'.json');raw=p.read_bytes();p.rename(p.with_name(p.name+'.pending_test'));self.assertEqual(len(self.scan()),10);self.assertEqual(p.read_bytes(),raw)
        bad=p.with_name(p.name+'.pending_bad');bad.write_bytes(b'incomplete')
        with self.assertRaises((ValueError,json.JSONDecodeError)):self.scan()
        self.assertEqual(bad.read_bytes(),b'incomplete');bad.unlink();p.with_name(p.name+'.pending_test').unlink()
        v=json.loads(raw);v['record']['seed']+=1;p.write_bytes(store.encode(store.envelope(v['record'])))
        with self.assertRaises(ValueError):self.scan()
    def test_save_failure_prevents_next_call_and_keeps_spool(self):
        original=store.save
        def fail(path,raw):
            if Path(path).parent.name=='calls':raise IOError('simulated Drive failure')
            return original(path,raw)
        with mock.patch.object(store,'save',side_effect=fail):
            with self.assertRaises(IOError):self.run_schedule()
        self.assertEqual(len(self.engine.calls),1);self.assertIsNotNone(self.control.pending);state=json.loads((self.control_dir/'state.json').read_text());self.assertEqual(state['phase'],'saving');self.assertIn('record',state['completed_call_envelope']);self.assertEqual(len(list(self.control_dir.glob('*_completed.json'))),1)
    def test_model_failure_and_soft_limit_never_create_completed_call(self):
        self.engine.fail=True
        with self.assertRaises(RuntimeError):self.run_schedule()
        self.assertEqual(list((self.work/'calls').iterdir()),[])
        self.config['session_id']='test_soft';self.control=gpu_worker.Control(self.config);self.engine.fail=False;self.engine.soft=True;r=self.run_schedule();self.assertEqual(r['reason'],'soft_limit');self.assertEqual(r['verified_calls'],0)
    def test_termination_edges(self):
        self.assertFalse(store.termination([151645],1)['truncated_by_cap']);self.assertTrue(store.termination([10],1)['truncated_by_cap']);self.assertEqual(store.termination([151645],1)['retained_nontermination_tokens'],0)
        for ids,cap in [([],1),([10],2),([151645,10],2),([200000],1)]:
            with self.assertRaises(ValueError):store.termination(ids,cap)
    def test_staging_checksum_and_reuse(self):
        source=Path(self.tmp.name)/'source';source.mkdir();dest=Path(self.tmp.name)/'dest';raw=b'asset bytes';(source/'test').write_bytes(raw);manifest={'files':[{'name':'test','bytes':len(raw),'sha256':store.digest(raw)}]};store.stage_assets(source,dest,manifest,notify=lambda *a,**k:None);store.stage_assets(source,dest,manifest,notify=lambda *a,**k:None);self.assertEqual((dest/'test').read_bytes(),raw);(dest/'test').write_bytes(b'changed')
        with self.assertRaises(ValueError):store.stage_assets(source,dest,manifest,notify=lambda *a,**k:None)

    def test_engine_measurement_path_and_timeout(self):
        ids_out=self.tok.raw.encode(r'\boxed{2}',add_special_tokens=False).ids+[151645]
        class Tensor:
            def __init__(self,ids):self.ids=ids;self.shape=(1,len(ids))
            def __getitem__(self,item):return Tensor(self.ids[item[1]])
            def detach(self):return self
            def cpu(self):return self
            def tolist(self):return self.ids
        cuda=types.SimpleNamespace(synchronize=lambda:None,manual_seed_all=lambda s:None,reset_peak_memory_stats=lambda:None,max_memory_allocated=lambda:100,max_memory_reserved=lambda:120)
        fake_torch=types.SimpleNamespace(cuda=cuda,long='long',manual_seed=lambda s:None,tensor=lambda x,**k:Tensor(x[0]),ones_like=lambda x:x,inference_mode=lambda:mock.MagicMock())
        class Model:
            def generate(_self,input_ids,**kwargs):return Tensor(input_ids.ids+ids_out)
        engine=object.__new__(gpu_engine.Engine);engine.torch=fake_torch;engine.tokenizer=self.tok;engine.protocol=self.protocol;engine.started=time.monotonic();engine.model=Model()
        with mock.patch.dict(sys.modules,{'transformers':types.SimpleNamespace(GenerationConfig=lambda **k:k)}):
            answer=engine.infer(self.rows[0]['initial_messages'],1024,123,time.monotonic());self.assertEqual(answer['output_ids'],ids_out);self.assertEqual(answer['extraction']['answer'],'2');self.assertGreater(answer['inference_seconds'],0)
            with self.assertRaises(ValueError):engine.infer(self.rows[0]['initial_messages'],1024,123,time.monotonic()-601)
            engine.started=time.monotonic()-1800
            with self.assertRaises(store.SoftLimitReached):engine.infer(self.rows[0]['initial_messages'],1024,123,time.monotonic())

    def test_GPU_environment_constraints_with_simulated_hardware(self):
        cpu={'torch':'2.14.1+cpu','transformers':'5.22.1'};fp={'version':'0.23.3','source_sha256':'same'};preflight={'cpu_versions':cpu,'cpu_source_fingerprints':{'tokenizers':fp}}
        props=types.SimpleNamespace(name='Tesla T4',total_memory=16*2**30)
        cuda=types.SimpleNamespace(is_available=lambda:True,device_count=lambda:1,get_device_properties=lambda i:props,get_device_capability=lambda i:(7,5),get_arch_list=lambda:['sm_75'],mem_get_info=lambda:(15*2**30,16*2**30))
        torch=types.SimpleNamespace(cuda=cuda,set_num_threads=lambda n:None,backends=types.SimpleNamespace(cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=True)),cudnn=types.SimpleNamespace(allow_tf32=True,version=lambda:9000)),version=types.SimpleNamespace(cuda='13.0'),_C=types.SimpleNamespace(__file__='fake.so'))
        with mock.patch.dict(sys.modules,{'torch':torch,'gpu_integrity':types.SimpleNamespace(fingerprint=lambda names,policy:{'tokenizers':fp},load_policy=lambda path:{'approval_id':'test'},cuda_dependency_inventory=lambda:{'simulated':True})}),mock.patch.object(gpu_engine.metadata,'version',side_effect=lambda n:'2.14.1+cu130' if n=='torch' else cpu[n]),mock.patch.object(gpu_engine.metadata,'distribution',return_value=types.SimpleNamespace(read_text=lambda n:'fake RECORD')),mock.patch.object(gpu_engine,'file_sha',return_value='fake-hash'),mock.patch.object(gpu_engine.platform,'platform',return_value='mock-platform'),mock.patch.object(gpu_engine.subprocess,'run',return_value=types.SimpleNamespace(stdout='GPU-test, 590.0')):
            signature,physical=gpu_engine.environment(preflight);self.assertEqual(signature['GPU_family'],'T4');self.assertFalse(signature['quantization'])
            props.name='A100'
            with self.assertRaises(ValueError):gpu_engine.environment(preflight)
            props.name='Tesla T4';cuda.mem_get_info=lambda:(0,16*2**30)
            with self.assertRaises(ValueError):gpu_engine.environment(preflight)

    def test_controller_real_CPU_worker_and_bounded_salvage(self):
        import gpu_driver
        requests=[]
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):requests.append(self.path);self.send_response(200);self.end_headers()
            def log_message(self,*args):pass
        server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            for phase in ['finished','error']:
                bundle=Path(self.tmp.name)/('bundle_'+phase);bundle.mkdir();shutil.copyfile(BASE/'watchdog.py',bundle/'watchdog.py')
                script="import sys,time,json\nfrom pathlib import Path\nsys.path.insert(0,"+repr(str(BASE))+ ")\nfrom gpu_worker import Control\np=Path(sys.argv[sys.argv.index('--config')+1]);c=json.loads(p.read_text());ctl=Control(c);ctl.handshake();ctl.pending={'cpu_test_payload':True};ctl.phase("+repr(phase)+")\nwhile True:time.sleep(.1)\n"
                (bundle/'worker_entry.py').write_text(script);directory=Path(self.tmp.name)/('controller_'+phase);directory.mkdir();original=gpu_driver.watchdog.write_json
                def write(path,value):
                    if Path(path).name=='config.json':value={**value,'self_test':True,'test_release_address':'127.0.0.1:'+str(server.server_port),'limits':{**watchdog.LIMITS,'poll':.02,'salvage':.5,'release_timeout':.5,'absolute':4.,'release_reserve':.5}}
                    return original(path,value)
                with mock.patch.dict(os.environ,{'TASK3_SUPERVISION_SELFTEST':'1'}),mock.patch.object(gpu_driver.watchdog,'write_json',side_effect=write):
                    os.environ.pop('TBE_RUNTIME_ADDR',None);result=gpu_driver.run(bundle,self.work,directory,time.monotonic(),'controller_'+phase)
                self.assertEqual(result['reason'],phase);saved=json.loads((self.work/'live/session_status'/('controller_'+phase+'.json')).read_text());self.assertTrue(saved['last_state']['completed_call_envelope']['cpu_test_payload'])
            self.assertEqual(requests,['/unassign','/unassign'])
        finally:server.shutdown();server.server_close()

def release_checks():
    requests=[]
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):requests.append(self.path);self.send_response(200);self.end_headers()
        def log_message(self,*args):pass
    server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();results=[]
    try:
        for abort in [False,True]:
            with tempfile.TemporaryDirectory() as raw:
                d=Path(raw);env=os.environ.copy();env.pop('TBE_RUNTIME_ADDR',None);env.update(TASK3_BOOTSTRAP_SELFTEST='1',TASK3_TEST_RELEASE_ADDR='127.0.0.1:'+str(server.server_port),TASK3_TEST_LIMIT='.4');before=len(requests)
                if abort:(d/'abort.json').write_text('{}')
                proc=subprocess.Popen([sys.executable,'-I',str(BASE/'bootstrap_guard.py'),str(d),str(time.monotonic()),'test','--self-test'],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
                out,err=proc.communicate(timeout=5)
                if proc.returncode:raise AssertionError(err.decode())
                result=json.loads((d/'bootstrap_release.json').read_text());assert len(requests)==before+1 and requests[-1]=='/unassign' and not result['confirmed_physical_deallocation'];results.append(result['reason'])
    finally:server.shutdown();server.server_close()
    return results

def run():
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests);stream=io.StringIO();
    with contextlib.redirect_stdout(io.StringIO()):result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    print(stream.getvalue(),flush=True)
    if not result.wasSuccessful():raise AssertionError('CPU execution-kit checks failed')
    return {'tests_passed':result.testsRun,'bootstrap_real_process_release_scenarios':release_checks(),'GPU_used':False,'real_model_loaded':False,'native_runtime_release_executed':False,'scope':'Real files and subprocesses/loopback service; fake model/CUDA timing API. Actual GPU hardware/FP16 engine/model output untested.'}
if __name__=='__main__':print(json.dumps(run(),indent=2))
