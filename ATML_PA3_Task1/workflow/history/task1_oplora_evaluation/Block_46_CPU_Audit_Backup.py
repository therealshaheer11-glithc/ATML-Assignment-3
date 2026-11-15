# Block 44 — OPLoRA final adapter export. Run this entire cell on CPU.
import os, shutil, subprocess, threading, json, hashlib, io, tarfile, zipfile, tempfile, uuid
from pathlib import Path
def cpu_guard():
    evidence = []
    if os.environ.get('COLAB_GPU', '0').lower().strip() not in ('', '0', 'false', 'none'):
        evidence.append('GPU allocation flag')
    if list(Path('/dev').glob('nvidia[0-9]*')):
        evidence.append('GPU device')
    if any(os.environ.get(k) for k in ('COLAB_TPU_ADDR', 'TPU_NAME', 'TPU_WORKER_ID')) or os.environ.get('PJRT_DEVICE', '').upper() == 'TPU':
        evidence.append('TPU allocation')
    if shutil.which('nvidia-smi'):
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                                    capture_output=True, text=True, timeout=10)
            if result.returncode or result.stdout.strip(): evidence.append('GPU present or status uncertain')
        except Exception:
            evidence.append('GPU status uncertain')
    if evidence:
        print('CPU required; requesting release:', evidence, flush=True)
        def release():
            try:
                from google.colab import runtime
                runtime.unassign()
            except Exception as error:
                print('Release request:', type(error).__name__, flush=True)
        thread = threading.Thread(target=release, daemon=True); thread.start(); thread.join(10)
        raise RuntimeError('Select CPU; end this notebook\'s remaining L4 session through Manage sessions if active.')
cpu_guard()
from google.colab import drive
drive.mount('/content/drive')

EXPECTED={'lora16_final': {'status': 'TASK1_LORA16_FINAL_COMPLETE_SAVED', 'run_id': 'task1_lora16_final_eval_e1bed1f92f954e50baef4a94f5e244e5', 'official_training_updates': 0, 'benchmark_completed': {'gsm8k_cot': 1319, 'hellaswag': 10042, 'boolq': 3270, 'arc_easy': 2376}, 'qualitative_completed': 24, 'FP32_probe_files_reused': 64, 'baseline_summary': {'status': 'TASK1_LORA16_FINAL_EVALUATION_COMPLETE', 'audit': {'condition': 'lora16_final', 'model_identity': '8f356e9f79687fdd71e1f8b2180f07f6399a5a4b534d155da1ce37985b42afc6', 'protocol_identity': 'b0cdaa6bab77b1221a581c5f14f02acdd754b2208035d42b9fd5abda8825acf3', 'gpu_implementation_identity': '761d133850f217ac5314260af10867e42698cb12ed5cb4abd8936239ca819f4d'}, 'benchmark_counts': {'gsm8k_cot': 1319, 'hellaswag': 10042, 'boolq': 3270, 'arc_easy': 2376}, 'metrics': {'hellaswag': {'acc,none': 0.3791077474606652, 'acc_norm,none': 0.4736108344951205}, 'boolq': {'acc,none': 0.7262996941896025}, 'arc_easy': {'acc,none': 0.6637205387205387, 'acc_norm,none': 0.6384680134680135}, 'gsm8k_cot': {'exact_match,flexible-extract': 0.6186504927975739, 'exact_match,strict-match': 0.0}}, 'original_probe_manifest': 'results/original/ACTIVATION_PROBES_COMPLETE.json', 'qualitative_responses': 24, 'FP32_activation_probes': 64, 'human_qualitative_annotation_pending': True, 'official_training_updates': 0, 'preliminary_batch_one_GSM_outputs_used_in_main_metrics': False}, 'elapsed_worker_seconds': 9487.515474161999, 'GPU_memory': {'peak_allocated_bytes': 2809755648, 'peak_reserved_bytes': 5421137920, 'total_gpu_bytes': 23659151360}}, 'oplora_final': {'status': 'TASK1_OPLORA_FINAL_COMPLETE_SAVED', 'run_id': 'task1_oplora_final_eval_e921fcbc369148b7936d80bcfd548bc0', 'official_training_updates': 0, 'benchmark_completed': {'gsm8k_cot': 1319, 'hellaswag': 10042, 'boolq': 3270, 'arc_easy': 2376}, 'qualitative_completed': 24, 'FP32_probe_files_reused': 64, 'baseline_summary': {'status': 'TASK1_OPLORA_FINAL_EVALUATION_COMPLETE', 'audit': {'condition': 'oplora_final', 'model_identity': 'b39053eb70727fa6ad0b9b50a824f186e33249d5f0c281b4d46bcaeac23d31b9', 'protocol_identity': 'b0cdaa6bab77b1221a581c5f14f02acdd754b2208035d42b9fd5abda8825acf3', 'gpu_implementation_identity': '71eb451000f2e66f149b146f0c4ffb2398481e42fb6051b4f1273207d2ac73d2', 'OPLoRA_assets_sha256': 'bce4edc6a4f68d3307506661ff41e8d78b326d6e8b6087dabf4627aabb5120b4', 'trained_checkpoint_update': 1250}, 'benchmark_counts': {'gsm8k_cot': 1319, 'hellaswag': 10042, 'boolq': 3270, 'arc_easy': 2376}, 'metrics': {'hellaswag': {'acc,none': 0.37801234813782114, 'acc_norm,none': 0.47410874327823144}, 'boolq': {'acc,none': 0.7299694189602447}, 'arc_easy': {'acc,none': 0.67003367003367, 'acc_norm,none': 0.6338383838383839}, 'gsm8k_cot': {'exact_match,flexible-extract': 0.621683093252464, 'exact_match,strict-match': 0.0}}, 'original_probe_manifest': 'results/original/ACTIVATION_PROBES_COMPLETE.json', 'qualitative_responses': 24, 'FP32_activation_probes': 64, 'human_qualitative_annotation_pending': True, 'official_training_updates': 0, 'preliminary_batch_one_GSM_outputs_used_in_main_metrics': False}, 'elapsed_worker_seconds': 15361.376136617, 'GPU_memory': {'peak_allocated_bytes': 3158407168, 'peak_reserved_bytes': 5758779392, 'total_gpu_bytes': 23659151360}}}

import gzip, math, time
root=Path('/content/drive/MyDrive/ATML-PA-3/Task1')
COUNTS={'gsm8k_cot':1319,'hellaswag':10042,'boolq':3270,'arc_easy':2376}
def encode(v):
    return (json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode()
def digest(b):return hashlib.sha256(b).hexdigest()
def identity(v):return digest(encode(v))
def validate_records(records,task,contract,rows):
    assert len(records)==COUNTS[task]
    for i,(record,row) in enumerate(zip(records,rows)):
        assert record['status']=='complete' and record['task']==task
        assert record['source_index']==row['source_index']==i and record['contract']==contract
        assert record['request_identity']==identity({'source_index':i,'requests':row['requests']})
        assert all(isinstance(v,(int,float)) and math.isfinite(v) for v in record['scores'].values())
    keys=set(records[0]['scores']);assert keys and all(set(r['scores'])==keys for r in records)
    return {k:sum(r['scores'][k] for r in records)/len(records) for k in keys}
started=time.monotonic()
with tempfile.TemporaryDirectory(dir='/content') as temp:
    temp=Path(temp); assert shutil.disk_usage(temp).free>2*2**30
    archive=temp/'evaluation_backup.zip';members={};reports={}
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        def add(relative):
            relative=str(relative);p=root/relative
            assert not Path(relative).is_absolute() and '..' not in Path(relative).parts
            assert p.is_file() and not p.is_symlink()
            raw=p.read_bytes()
            if relative in members:
                assert digest(raw)==members[relative]['sha256'];return raw
            z.writestr(relative,raw);members[relative]={'sha256':digest(raw),'bytes':len(raw)}
            return raw
        ready=json.loads(add('configs/BASELINE_CPU_READY_V1.json'))
        protocol=json.loads(add('configs/EVALUATION_PROTOCOL_V2_BATCH8.json'));add('configs/HARNESS_SOURCE_HASHES.json')
        prepared={}
        for task,entry in ready['benchmarks'].items():
            raw=add(entry['path']);assert digest(raw)==entry['sha256']
            prepared[task]=[json.loads(line) for line in gzip.decompress(raw).decode().splitlines()]
            assert len(prepared[task])==COUNTS[task]
        for condition,observed in EXPECTED.items():
            print('Checking',condition,'on CPU. Individual Drive reads may take several minutes.',flush=True)
            base=Path('results_batch8_V2')/condition
            summary=json.loads(add(base/'EVALUATION_COMPLETE.json'))
            assert summary==observed['baseline_summary'] and summary['benchmark_counts']==COUNTS
            assert identity(protocol)==summary['audit']['protocol_identity']
            contract=json.loads(add(base/'CONTRACT.json'))
            assert contract=={k:summary['audit'][k] for k in ('condition','model_identity','protocol_identity')}
            metrics={}
            for task,count in COUNTS.items():
                records=[]
                for i in range(count):
                    records.append(json.loads(add(base/task/f'{i:05}.json')))
                    if (i+1)%2000==0:print(condition,task,i+1,'records checked',flush=True)
                metrics[task]=validate_records(records,task,contract,prepared[task])
                assert metrics[task]==summary['metrics'][task]
                step=25 if task=='gsm8k_cot' else 100
                for i in range(0,count,step):
                    chunk=json.loads(add(base/'scored_chunks'/task/(str(i)+'.json')))
                    assert chunk['audit']==summary['audit']
                    assert identity(chunk['records'])==chunk['checksum'] and chunk['records']==records[i:i+step]
                print(condition,task,'complete; saved scores reaggregated.',flush=True)
            qual=json.loads(add(base/'QUALITATIVE_COMPLETE.json'))
            assert qual['count']==len(qual['entries'])==24 and qual['audit']==summary['audit']
            assert len({e['id'] for e in qual['entries']})==24
            for entry in qual['entries']:
                assert digest(add(entry['path']))==entry['sha256']
            raw_count=0
            for folder in ('raw_generation_batches','raw_likelihood_chunks'):
                paths=sorted((root/base/folder).glob('*.json'));assert paths
                for p in paths:
                    value=json.loads(add(p.relative_to(root)));checksum=value.pop('content_sha256')
                    assert identity(value)==checksum and value['contract']['audit']==summary['audit']
                    raw_count+=1
            for p in sorted((root/base).rglob('*')):
                if p.is_file():add(p.relative_to(root))
            run=observed['run_id']
            result=json.loads(add('logs/'+run+'_result.json'));assert result==observed
            receipt=json.loads(add('logs/'+run+'_session_final.json'))
            assert receipt['worker_result_validated'] is True and receipt['worker_exit_code']==0
            assert receipt['result_sha256']==digest(encode(result))
            add('logs/'+run+'_session_started.json');add('logs/'+run+'_worker_console.txt')
            implementation=summary['audit']['gpu_implementation_identity']
            code=Path('execution')/('baseline_batch8_V2_'+implementation)
            sources=json.loads(add(code/'FILES.json'));assert identity(sources)==implementation
            for name,h in sources.items():assert digest(add(code/name))==h
            reports[condition]={'counts':COUNTS,'metrics':metrics,'qualitative_responses':24,'raw_journals_checked':raw_count}
        manifest={'status':'TASK1_TWO_ADAPTER_EVALUATIONS_STRUCTURAL_BACKUP','conditions':reports,'files':members,
                  'fresh_stock_scorer_replay':False,'human_annotations_verified':False,'GPU_work_run':False,
                  'scope':'LoRA16 and OPLoRA evaluation records, prepared benchmark requests, code and completion receipts; excludes model weights and original probe tensor files.'}
        z.writestr('EXPORT_MANIFEST.json',encode(manifest))
    with zipfile.ZipFile(archive) as z:
        assert len(z.namelist())==len(set(z.namelist()))==len(members)+1
        assert z.testzip() is None
        for name,entry in members.items():
            raw=z.read(name);assert len(raw)==entry['bytes'] and digest(raw)==entry['sha256']
    def file_sha(p):
        h=hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda:f.read(8*2**20),b''):h.update(chunk)
        return h.hexdigest()
    h=file_sha(archive);dest=root/'exports'/('Task1_LoRA16_OPLoRA_Evaluations_'+h[:16]+'.zip')
    dest.parent.mkdir(exist_ok=True)
    if not dest.exists():
        pending=dest.with_name(dest.name+'.pending_'+uuid.uuid4().hex)
        shutil.copyfile(archive,pending);assert file_sha(pending)==h;pending.rename(dest)
    assert file_sha(dest)==h
    report={'status':'TASK1_BLOCK46_STRUCTURAL_AUDIT_BACKUP_OK','conditions':reports,'path':str(dest.relative_to(root)),
            'sha256':h,'bytes':dest.stat().st_size,'archived_files':len(members)+1,'GPU_work_run':False,
            'fresh_stock_scorer_replay':False,'elapsed_seconds':time.monotonic()-started}
    receipt_path=root/'logs'/('block46_audit_'+uuid.uuid4().hex+'.json')
    receipt_path.write_bytes(encode(report));assert json.loads(receipt_path.read_bytes())==report
    print(json.dumps(report,indent=2))
    print('Keep GPU off. Return this JSON. Fresh stock-scoring replay and remaining diagnostics are separate.')
