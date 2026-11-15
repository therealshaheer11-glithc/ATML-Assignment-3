"""CPU validation, official scoring, pair bootstrap, figures and results export."""
import os,sys,io,json,hashlib,zipfile,traceback,uuid,re,time,warnings,importlib.util,importlib.metadata
from pathlib import Path
from datetime import datetime,timezone
from google.colab import drive,files,runtime

if os.environ.get('COLAB_GPU','0') not in ('','0'):
    print('STOP - CPU REQUIRED. Select CPU, reconnect, then run this same cell.',flush=True)
    runtime.unassign()
    raise RuntimeError('CPU only; no training or generation is required.')
if not Path('/content/drive/MyDrive').exists():drive.mount('/content/drive')
RUN_ROOT=Path('/content/drive/MyDrive/ATML_PA3/task2/run_20261031_150949_1cff148a')
LOCAL_ROOT=Path('/content/atml_pa3_task2/run_20261031_150949_1cff148a')
os.environ.setdefault('MPLCONFIGDIR',str(LOCAL_ROOT/'matplotlib_cache'))
raw_package=globals().get('task2_results_package_bytes')
if not isinstance(raw_package,bytes):raise RuntimeError('Use the supplied Block16 ZIP loader.')
package_sha=hashlib.sha256(raw_package).hexdigest()
WORK=LOCAL_ROOT/'results_code'/package_sha[:16]
with zipfile.ZipFile(io.BytesIO(raw_package)) as archive:
    assert archive.testzip() is None
    manifest=json.loads(archive.read('package_manifest.json'))
    assert set(archive.namelist())=={r['path'] for r in manifest['files']}|{'package_manifest.json'}
    for record in manifest['files']:
        relative=Path(record['path']);assert not relative.is_absolute() and '..' not in relative.parts
        raw=archive.read(str(relative))
        assert len(raw)==record['bytes'] and hashlib.sha256(raw).hexdigest()==record['sha256']
        path=WORK/relative;path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():assert path.read_bytes()==raw
        else:path.write_bytes(raw)
sys.path.insert(0,str(WORK))
for name in list(sys.modules):
    if name=='task2' or name.startswith('task2.'):del sys.modules[name]
from task2.session_io import bounded_io,save_bytes,save_json
from task2.dataset import load_dataset,validate_dataset,file_sha
from task2.final_evaluation import build_plan,encoded,batches
from task2.evaluation_artifacts import read_receipt,verify_manifest
from task2.result_analysis import analyze_predictions,analyze_training,csv_bytes
from task2.result_plots import make_plots,gallery_html

attempt=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_')+uuid.uuid4().hex[:8]
started=time.monotonic();phase='startup'
reference=json.loads((WORK/'evaluation_reference.json').read_bytes())
expected=reference['result'];source=expected['source_reference']
out=LOCAL_ROOT/'results_exports'/attempt
out.mkdir(parents=True,exist_ok=False)
source_files=[]
def report(name):
    global phase
    phase=name;print('RESULTS PHASE:',name,flush=True)
def put(relative,raw):
    raw=raw if isinstance(raw,bytes) else encoded(raw)
    path=out/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    return path
def receipt_read(receipt,export_name):
    with bounded_io(120,'Source artifact read and checksum'):
        value=read_receipt(receipt,RUN_ROOT)
        raw=Path(receipt['path']).read_bytes()
        assert len(raw)==receipt['bytes'] and hashlib.sha256(raw).hexdigest()==receipt['sha256']
    put(export_name,raw);source_files.append({'export':export_name,**receipt})
    return value
def safe_path(path):
    path=Path(path)
    if not path.resolve().is_relative_to(RUN_ROOT.resolve()):raise ValueError('Source path is outside the task run.')
    return path

try:
    report('verifying completed final evaluation and frozen source identities')
    eval_pointer_path=RUN_ROOT/'manifests/active_task2_evaluation.json'
    train_pointer_path=RUN_ROOT/'manifests/active_task2_training.json'
    eval_pointer_raw=eval_pointer_path.read_bytes();eval_pointer=json.loads(eval_pointer_raw)
    train_pointer_raw=train_pointer_path.read_bytes();train_pointer=json.loads(train_pointer_raw)
    assert eval_pointer['complete'] and eval_pointer['verified_answers']==1280
    assert eval_pointer['recipe_sha256']==expected['recipe']['sha256']
    for key in ('recipe','plan','manifest'):assert eval_pointer[key]==expected[key]
    assert train_pointer['global_updates']==4500 and train_pointer['progress']==source['progress']
    for key in ('dataset_identity','execution_recipe_sha256','initial_manifest_sha256'):
        assert train_pointer[key]==source[key]
    put('source/active_evaluation.json',eval_pointer_raw);put('source/active_training.json',train_pointer_raw)
    recipe=receipt_read(expected['recipe'],'source/evaluation_recipe.json')
    plan=receipt_read(expected['plan'],'source/evaluation_plan.json')
    evaluation=receipt_read(expected['manifest'],'source/evaluation_manifest.json')
    progress=receipt_read(source['progress'],'source/training_progress.json')
    assert recipe['source_reference']==source and progress['checkpoint']==source['checkpoint']
    assert progress['stage_updates']=={'1':2250,'2':1500,'3':750}
    dataset_dir=RUN_ROOT/'datasets'/('task2_A012_'+source['dataset_identity'][:16])
    with bounded_io(180,'Frozen dataset and completed evaluation verification'):
        data=load_dataset(dataset_dir);assert data['identity']==source['dataset_identity']
        assert file_sha(dataset_dir/'freeze_receipt.json')==reference['freeze_receipt_sha256']
        put('source/freeze_receipt.json',(dataset_dir/'freeze_receipt.json').read_bytes())
        assert build_plan(data)==plan
        assert verify_manifest(evaluation,plan,expected['recipe']['sha256'],RUN_ROOT)==1280
    saved=json.loads((RUN_ROOT/'logs'/('block_15_'+reference['attempt']+'.json')).read_bytes())
    assert saved==expected
    saved_status=json.loads((RUN_ROOT/'logs'/('block_15_status_'+reference['attempt']+'.json')).read_bytes())
    assert saved_status['code']=='COMPLETE' and saved_status['verified_answers']==1280
    put('source/evaluation_result.json',saved);put('source/evaluation_status.json',saved_status)

    report('loading the checksum-locked official VQA normalizer')
    normalizer_path=RUN_ROOT/'code/vqa_normalization.py'
    assert file_sha(normalizer_path)==reference['expected_normalizer_sha256']==data['manifest']['normalizer_sha256']
    spec=importlib.util.spec_from_file_location('task2_locked_vqa_normalizer',normalizer_path)
    module=importlib.util.module_from_spec(spec)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',SyntaxWarning)  # Legacy regex literals; checksum-locked source stays unchanged.
        spec.loader.exec_module(module)
    normalize=module.normalize_answer
    for text,value in {'The two cats.':'2 cats','1,000':'1000','3.14':'3.14','dont':"don't",' YES!\n':'yes','':''}.items():
        assert normalize(text)==value
    validate_dataset(data,normalize)
    put('source/vqa_normalization.py',normalizer_path.read_bytes())
    for name in ('vqa_normalizer.json','vqa_source_snapshot.json','external/vqaEval_upstream.py'):
        path=RUN_ROOT/'provenance'/name
        if path.exists():put('source/normalizer_provenance/'+name,path.read_bytes())

    report('reading all 1,280 predictions and original-reference losses')
    rows=[];generation_seconds=0.;nll_seconds=0.
    for key,condition,batch in batches(plan):
        chunk=receipt_read(evaluation['batches'][key],'source/evaluation_batches/'+key+'.json')
        rows.extend(chunk['rows']);generation_seconds+=chunk['generation_seconds'];nll_seconds+=chunk['NLL_seconds']
    assert len(rows)==1280
    report('scoring predictions and computing 4,000 complete-pair bootstrap intervals')
    analysis=analyze_predictions(rows,normalize)
    put('results.json',analysis)
    for key,name in [('scored_rows','per_question'),('pair_rows','per_pair'),('aggregates','condition_and_type_metrics'),('advantages','correct_image_advantages'),('partner_following','partner_following')]:
        put('tables/'+name+'.csv',csv_bytes(analysis[key]))

    report('verifying all 4,500 update records and 12 development checkpoints')
    training=[]
    for index,receipt in enumerate(progress['training_log_receipts']):
        training.extend(receipt_read(receipt,'source/training_logs/'+Path(receipt['path']).name))
    training.sort(key=lambda r:r['global_update'])
    dev={str(step):receipt_read(receipt,'source/development/'+Path(receipt['path']).name) for step,receipt in progress['dev_receipts'].items()}
    stages=analyze_training(training,dev)
    put('tables/training_updates.csv',csv_bytes([{k:v for k,v in row.items() if k not in ('examples','pairs')} for row in training]))
    put('tables/training_stage_summary.csv',csv_bytes(stages))
    dev_table=[]
    for step in sorted(map(int,dev)):
        metrics=dev[str(step)];dev_table.append({'global_update':step,'suite':'caption','condition':'correct','metric':'caption_CE','value':metrics['caption']['correct_image_CE']})
        for suite,value in metrics['suites'].items():
            dev_table.append({'global_update':step,'suite':suite,'condition':'complementary','metric':'swap_penalty_NLL','value':value['swap_penalty_NLL']})
            for condition,condition_value in value['conditions'].items():
                dev_table.append({'global_update':step,'suite':suite,'condition':condition,'metric':'mean_reference_NLL','value':condition_value['mean_reference_NLL']})
    put('tables/development_curves.csv',csv_bytes(dev_table))

    report('collecting measured session costs without mixing them with update compute')
    sessions=[];cache_sessions=[]
    for path in sorted((RUN_ROOT/'logs').glob('block_14_*.json')):
        if not re.fullmatch(r'block_14_\d{8}_\d{6}_[0-9a-f]{8}\.json',path.name):continue
        value=json.loads(path.read_bytes())
        if value.get('status') not in ('passed','checkpointed') or value.get('execution_recipe_sha256')!=source['execution_recipe_sha256']:continue
        put('source/training_sessions/'+path.name,path.read_bytes())
        sessions.append({'attempt':path.stem.removeprefix('block_14_'),
            **{k:value.get(k) for k in ('start_global_update','end_global_update','successful_updates_this_segment',
                'training_update_wall_seconds','elapsed_seconds','qwen_source_staging_and_loading_seconds',
                'feature_staging_and_verification_seconds','peak_gpu_allocated_bytes','peak_gpu_reserved_bytes')},
            'source_sha256':file_sha(path)})
    for path in sorted((RUN_ROOT/'logs').glob('block_11_*.json')):
        if not re.fullmatch(r'block_11_\d{8}_\d{6}_[0-9a-f]{8}\.json',path.name):continue
        value=json.loads(path.read_bytes())
        if value.get('dataset_identity')!=source['dataset_identity'] or value.get('status') not in ('passed','checkpointed'):continue
        put('source/cache_sessions/'+path.name,path.read_bytes())
        cache_sessions.append({'attempt':path.stem.removeprefix('block_11_'),
            **{k:value.get(k) for k in ('computed_images_this_run','elapsed_seconds','compute_seconds_cuda_events',
                'preprocessing_and_transfer_seconds','input_archive_staging_seconds','save_and_verification_seconds',
                'peak_gpu_allocated_bytes','peak_gpu_reserved_bytes')},'source_sha256':file_sha(path)})
    positive=sorted([r for r in sessions if r['successful_updates_this_segment']>0],key=lambda r:(r['start_global_update'],r['end_global_update']))
    cursor=0;coverage=True
    for row in positive:
        if row['start_global_update']!=cursor:coverage=False
        cursor=row['end_global_update']
    coverage=coverage and cursor==4500
    resources={'training_stage_summary':stages,'successful_training_sessions':sessions,
        'successful_session_history_covers_0_to_4500_once':coverage,
        'training_updates_wall_seconds':sum(r['wall_seconds'] for r in training),
        'successful_session_wall_seconds_sum':sum(r['elapsed_seconds'] for r in positive),
        'successful_session_peak_gpu_allocated_bytes_max':max((r['peak_gpu_allocated_bytes'] for r in positive),default=None),
        'successful_session_peak_gpu_reserved_bytes_max':max((r['peak_gpu_reserved_bytes'] for r in positive),default=None),
        'cache_sessions':cache_sessions,'final_evaluation':saved,
        'all_saved_generation_seconds':generation_seconds,'all_saved_NLL_seconds':nll_seconds,
        'generation_seconds_per_answer':generation_seconds/1280,
        'parameter_counts':{'frozen_base':596049920,'connector':1837056,'lora':4587520,
            'training_stage1_trainable':1837056,'training_stage2_stage3_trainable':6424576,'evaluation_trainable':0},
        'notes':['Training update wall time excludes model loading, feature staging, development and checkpoint IO.',
            'Successful session wall time includes those overheads. Failed attempts/preflight and unrecorded lifecycle overhead are not included in that sum.',
            'GPU memory is PyTorch peak allocated/reserved, not total device usage. Evaluation peaks were reset after staging.',
            'Generation is synchronized batched wall time/answer, not single-question latency; no warmup excluded.',
            'No allocation-release success is inferred from a release request.']}
    put('resources.json',resources);put('tables/training_session_resources.csv',csv_bytes(sessions));put('tables/cache_resources.csv',csv_bytes(cache_sessions))

    report('rendering 10 result figures as PNG and SVG')
    plots=make_plots(analysis,training,dev,stages,out/'figures')
    from PIL import Image
    for path in plots:
        if path.suffix=='.png':
            with Image.open(path) as image:image.verify()
    report('building a gallery of every final pair with verified original images')
    image_pointer=json.loads((RUN_ROOT/'manifests/active_task2_images.json').read_bytes())
    assert image_pointer['dataset_identity']==source['dataset_identity']
    assert image_pointer['image_index_sha256']==reference['original_image_index_sha256']
    assert image_pointer['completion_sha256']==reference['original_images_completion_sha256']
    assert file_sha(safe_path(image_pointer['completion_path']))==reference['original_images_completion_sha256']
    index_path=safe_path(image_pointer['image_index_path'])
    assert file_sha(index_path)==image_pointer['image_index_sha256']
    original_index=[json.loads(line) for line in index_path.read_text().splitlines()]
    lookup={(r['source_split'],r['image_id']):r for r in original_index}
    needed=sorted({(row['source_split'],row['image_id']) for row in plan['rows']})
    assert len(needed)==320
    image_files={};image_receipts=[]
    for index,key in enumerate(needed):
        image=lookup[key];path=safe_path(RUN_ROOT/image['durable_relative_path'])
        with bounded_io(30,'Gallery original image read'):
            raw=path.read_bytes()
        assert len(raw)==image['bytes'] and hashlib.sha256(raw).hexdigest()==image['sha256']
        name='images/'+image['file_name'];put(name,raw);image_files[key]=name;image_receipts.append(image)
        if (index+1)%40==0:print('GALLERY ORIGINALS VERIFIED:',index+1,'/320',flush=True)
    put('source/gallery_image_receipts.json',image_receipts)
    put('all_final_pairs.html',gallery_html(analysis['scored_rows'],image_files))
    put('source/dataset_manifest.json',(dataset_dir/'dataset_manifest.json').read_bytes())
    for name in ('configs/exact_protocol_A005.json','configs/model_revisions.json','configs/training_lr_endpoints.json',
                 'manifests/selected/grounding_test.jsonl','manifests/selected/natural_test_formatted.jsonl'):
        put('source/dataset/'+name,(dataset_dir/name).read_bytes())
    put('source/artifact_receipts.json',source_files)
    environment={name:importlib.metadata.version(name) for name in ('numpy','matplotlib','Pillow')}
    environment['python']=sys.version
    put('source/analysis_environment.json',environment)
    put('source/results_code_package_receipt.json',{'package_sha256':package_sha,'attempt':attempt,'source_reference':reference})
    put('HOW_TO_VIEW.txt',b'Open figures/*.png or *.svg for plots; tables/*.csv for numerical results; all_final_pairs.html for all160 original pairs. results.json has all scored predictions and statistics; resources.json separates measured costs. source/ retains verified raw evaluation/training/dev records and identities. No report is included. Final scores are subset results. Do not tune or rerun based on final-test outcomes.\n')
    assert eval_pointer_path.read_bytes()==eval_pointer_raw and train_pointer_path.read_bytes()==train_pointer_raw
    report('checking export integrity and saving a durable results ZIP')
    inventory={'schema':'atml-pa3-task2-results-export-v1','attempt':attempt,
        'evaluation_recipe_sha256':expected['recipe']['sha256'],'normalizer_sha256':reference['expected_normalizer_sha256'],
        'bootstrap':analysis['bootstrap'],'answers':1280,'training_updates':4500,'figures':10,
        'files':[{'path':str(p.relative_to(out)),'bytes':p.stat().st_size,'sha256':file_sha(p)} for p in sorted(out.rglob('*')) if p.is_file()]}
    put('export_manifest.json',inventory)
    local_zip=out.parent/('Task_2_Results_'+attempt+'.zip')
    with zipfile.ZipFile(local_zip,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.rglob('*')):
            if path.is_file():archive.write(path,str(path.relative_to(out)))
    with zipfile.ZipFile(local_zip) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist())=={r['path'] for r in inventory['files']}|{'export_manifest.json'}
        for entry in inventory['files']:
            raw=archive.read(entry['path'])
            assert len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256']
    destination=dataset_dir/'results'/attempt/local_zip.name
    raw=local_zip.read_bytes();digest=hashlib.sha256(raw).hexdigest()
    with bounded_io(180,'Durable results ZIP save and readback'):
        save_bytes(destination,raw);assert file_sha(destination)==digest
    summary=[r for r in analysis['aggregates'] if r['question_type']=='all' and r['metric'] in ('accuracy','pair_accuracy','reference_NLL','empty_rate')]
    result={'block':'16','status':'passed','attempt':attempt,'analysis_package_sha256':package_sha,
        'answers_scored':1280,'training_updates_verified':4500,'development_checkpoints_verified':12,
        'figures_created':10,'figure_formats':['png','svg'],'gallery_pairs':160,'original_images_verified':320,
        'report_written':False,'training_performed':False,'generation_performed':False,'gpu_used':False,
        'evaluation_recipe_sha256':expected['recipe']['sha256'],'summary':summary,
        'results_zip':{'path':str(destination),'sha256':digest,'bytes':len(raw),'readback_verified':True},
        'elapsed_seconds':round(time.monotonic()-started,3),'analysis_environment':environment,
        'visual_layout_review':'Pending user download/upload for assistant inspection; files rendered and PNG decoding checked here.',
        'next_step':'Download the results ZIP and attach it here for result interpretation and visual checking. No report writing.'}
    save_json(RUN_ROOT/'logs'/('block_16_'+attempt+'.json'),result)
    print('BLOCK 16 PASSED\n'+json.dumps(result,indent=2),flush=True)
    print('\nRESULTS AND PLOTS SAVED - NO ERROR DETECTED\nNo further GPU work is required. Download the results ZIP and attach it here for checking.',flush=True)
    try:files.download(str(local_zip))
    except Exception as error:print('Browser download did not start. The verified ZIP remains on Drive:',str(destination),'Download error:',repr(error),flush=True)
except BaseException:
    diagnostic={'block':'16','status':'failed','phase':phase,'traceback':traceback.format_exc(),
        'training_performed':False,'generation_performed':False,'gpu_used':False}
    print('STOP - SEND THIS OUTPUT TO ME\n'+json.dumps(diagnostic,indent=2),flush=True)
    try:save_json(RUN_ROOT/'logs'/('block_16_failed_'+attempt+'.json'),diagnostic)
    except Exception as error:print('Diagnostic save failed:',repr(error),flush=True)
    raise
