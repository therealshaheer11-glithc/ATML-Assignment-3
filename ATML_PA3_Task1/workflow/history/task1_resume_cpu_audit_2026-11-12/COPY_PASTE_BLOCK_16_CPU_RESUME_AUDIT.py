# Task 1 - Block 16: CPU audit and packed resume snapshot. NO GPU, inference or training.
# Copy this ENTIRE file into one new Colab cell with CPU selected. No ZIP upload.
import datetime, hashlib, json, os, shutil, subprocess, sys, threading, time, uuid, traceback
from pathlib import Path
EXPECTED_FILES = {'batch8_records.py': '6738fcd2146a4057e54c7728129411700524e5f78a6efdfeccfd120f2589c4aa', 'batch8_hflm.py': '1b8b3810028123d6f6623398e028628cc3d9fc72070aad359d92fdc52023969e', 'batch8_pilot.py': '76171f667464327df7c3235cf7f9209dd18efaca8ff56b715c9493af8e156ec9', 'batch8_cpu_checks.py': 'ff5bfe4acc12b2e8ef0809f98c0d0b2d624f3e821567c32b06042043d1cdb905', 'prepare_batch8_cpu.py': '69a00eb778cdcb31a50e3908c87ce619aa34b3ca5e8407260e1b55cc5aeabcf9', 'restore_batch8_cpu.py': '32ebce8ea449913d28c8a6681bed8773b222c88babb07cbc09189d30207d8f53', 'archive_io.py': 'd5d059306f34a09eee8b4469d3f47597b07ec182f797d40e4a4958d2a1bf8f57', 'restore_baseline_cpu.py': 'f0e2d61a261c5ee8bd8b3ebc10250dd56defe9906ff139d21ef9ccb2745c7106', 'baseline_runtime.py': 'a2c420ed090689c1849d65d7de4717707a713937b1060bb9c65b6adf44a8723a', 'evaluation_records.py': '841fe3750ba33fb63d1a7e56a3ebd5248c013f3eada114e3690c4b5a8250c607', 'launch_baseline.py': 'e69a0ebbb48752a343490bfa0b38e94fec10ffe9d33b3205bde4dccf00aa0f95', 'release_coordination.py': '614b5bab4cf1d5482b6b672fb79047709944167382414e4cdafc1f5cec20642b', 'gpu_guardian_R3.py': '12c7dfa4637bea83e7aacd821743ebdcb626c291ff8b14d1374befa37fd2a06d', 'watchdog_evidence.py': '30688a268a2f2441c5dbe0adeabd78afad2a5d76af5c5e0747e31ed30de3618e', 'launch_batch8_pilot.py': 'c952aab3d630d37dcbaefe97bb394edbe2d412f816d7f3251f63f43dabb9ac06', 'batch8_pilot_entry.py': 'fead5e0d664b3ddbc02c63b2458d02ec7c818fe624f29da87f6a9c61d96f069f', 'pilot_launch_checks.py': 'd16b080d6658bd5ad96740e513f646e747d1b9001215691b143407991b6fb2db', 'guardian_selftest.py': '861987baeecb0a16ab60799f4b143f40d8680c967eaadb73951c9ad11dc07907', 'guardian_lifecycle_tests.py': '87509ab95e9947d875a4c7451d1482433d905c1858aceb7b74fbfaf55d4219c7', 'stock_evaluation.py': '9d6229054c031fb82887842cb5704f2cc9d4be17ba76e1d28e9a491bd00fc830', 'evaluation_v2_engine.py': '3a6307302cee3e8b97f39f33c1571ad79f8d23b577199846505241cc9a16ec19', 'evaluation_v2_hflm.py': '31cb7a1c82064df84f7672043d35d2404b4cd0a76aaa575963065c2a3695a441', 'baseline_v2_protocol.py': '3ac78bc667f4ef0ab1a4724ad04fbc9f5e1964c76c813d92acab4de094683413', 'baseline_v2_worker.py': '44d8ca63264df69f909be126ca965f2204ad944369c1696b7d095511e25f9392', 'launch_baseline_v2.py': 'd38dab7e09c47fb36f635e7a1585fea1b7156bcbe1feb7e36b4a58f1e298e494', 'baseline_v2_engine_checks.py': '5c0514b307b752228d13bd0d3fb449508038c911a276622448bbcb8452e651e5', 'baseline_v2_control_checks.py': '86d2c7718f162a169ac586fcbcd7f66365d61571c688feb4fd70e32d50f53e26'}
SNAPSHOT_SOURCE = '"""CPU-only audit and packed transport of existing V2 answers. No inference."""\nimport concurrent.futures\nimport gzip\nimport hashlib\nimport json\nimport math\nimport time\nimport zipfile\nfrom pathlib import Path\n\nfrom archive_io import digest, safe_relative\nfrom evaluation_records import COUNTS, aggregate, encoded, identity\nfrom batch8_records import load_batch\n\n\ndef read_parallel(paths, action, label):\n    paths = list(paths)\n    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:\n        # Bounded outstanding work; no thousands of decoded responses in a queue.\n        for start in range(0, len(paths), 128):\n            for result in pool.map(lambda p: (p, p.read_bytes()), paths[start:start + 128]):\n                action(*result)\n            if start == 0 or (start + 128) // 1024 != start // 1024 or start + 128 >= len(paths):\n                print(f\'{label}: {min(start + 128, len(paths))}/{len(paths)} files checked\', flush=True)\n\n\ndef validate_answer(record, task, source_index, row, contract):\n    assert record[\'contract\'] == contract and record[\'status\'] == \'complete\'\n    assert record[\'task\'] == task and record[\'source_index\'] == source_index\n    assert record[\'request_identity\'] == identity({\'source_index\': source_index, \'requests\': row[\'requests\']})\n    assert record[\'scores\'] and all(type(v) in (int, float) and math.isfinite(v) for v in record[\'scores\'].values())\n    assert len(record[\'response\']) == len(row[\'requests\'])\n\n\ndef pack_records(root, local, contract, implementation, protocol, minimum_counts):\n    root, local = Path(root), Path(local)\n    local.mkdir(parents=True, exist_ok=True)\n    namespace = \'results_batch8_V2/original\'\n    base = root / namespace\n    audit = {**contract, \'gpu_implementation_identity\': implementation}\n    ready = json.loads((root / \'configs/BASELINE_CPU_READY_V1.json\').read_text())\n    archive_path = local / \'verified_resume.zip\'\n    index = {\'schema_version\': 1, \'contract\': contract, \'numerical_implementation_identity\': implementation,\n             \'created_utc_unix\': time.time(), \'files\': {}, \'counts\': {}, \'complete_metrics\': {},\n             \'raw_generation_batches\': 0, \'raw_likelihood_chunks\': 0,\n             \'scope\': \'Verified existing bytes only; no inference, training or per-answer score recomputation\'}\n    rows_by_task = {}\n    with zipfile.ZipFile(archive_path, \'x\', zipfile.ZIP_DEFLATED) as archive:\n        def add(rel, data):\n            safe_relative(rel)\n            assert rel not in index[\'files\']\n            archive.writestr(rel, data)\n            index[\'files\'][rel] = {\'sha256\': hashlib.sha256(data).hexdigest(), \'bytes\': len(data)}\n\n        assert json.loads((base / \'CONTRACT.json\').read_text()) == contract\n        add(namespace + \'/CONTRACT.json\', (base / \'CONTRACT.json\').read_bytes())\n        for task, count in COUNTS.items():\n            entry = ready[\'benchmarks\'][task]\n            path = root / str(safe_relative(entry[\'path\']))\n            assert digest(path) == entry[\'sha256\']\n            with gzip.open(path, \'rt\') as stream: rows = [json.loads(line) for line in stream]\n            assert [r[\'source_index\'] for r in rows] == list(range(count))\n            rows_by_task[task] = rows\n            records = {}\n\n            def accept(path, data):\n                i = int(path.stem)\n                assert path.name == f\'{i:05}.json\' and 0 <= i < count and i not in records\n                record = json.loads(data)\n                validate_answer(record, task, i, rows[i], contract)\n                expected_metrics = {protocol[\'primary_metrics\'][task]}\n                if task in protocol[\'secondary_metrics\']: expected_metrics.add(protocol[\'secondary_metrics\'][task])\n                assert expected_metrics.issubset(record[\'scores\'])\n                records[i] = record\n                add(str(path.relative_to(root)), data)\n\n            started = time.monotonic()\n            read_parallel(sorted((base / task).glob(\'*.json\')), accept, task)\n            index.setdefault(\'record_read_and_check_seconds\', {})[task] = time.monotonic() - started\n            index[\'counts\'][task] = len(records)\n            assert len(records) >= minimum_counts[task], f\'{task}: fewer saved records than previously reported\'\n            if len(records) == count:\n                metrics = aggregate([records[i] for i in range(count)], task)\n                completion = base / f\'{task}_COMPLETE.json\'\n                saved = json.loads(completion.read_text())\n                assert saved == {\'audit\': audit, \'count\': count, \'metrics\': metrics}\n                index[\'complete_metrics\'][task] = metrics\n                add(str(completion.relative_to(root)), completion.read_bytes())\n            else:\n                assert not (base / f\'{task}_COMPLETE.json\').exists(), \'Incomplete split has a completion receipt\'\n                index.setdefault(\'partial_score_totals\', {})[task] = {\n                    k: sum(r[\'scores\'][k] for r in records.values()) for k in next(iter(records.values()))[\'scores\']\n                } if records else {}\n\n        # The raw cache is preserved too, including work from any unfinished chunk.\n        prompt_ids = {tuple(r[\'prompt_input_ids\']) for r in rows_by_task[\'gsm8k_cot\']}\n        def raw_generation(path, data):\n            value = json.loads(data); c = value[\'contract\']\n            assert path.stem == identity(c) and c[\'audit\'] == audit\n            # Production reader verifies checksum, tokens, row coverage and prefixes.\n            checked = load_batch(path, c)\n            prompts = c[\'padded_input_ids\']; width = len(prompts[0])\n            assert 1 <= len(prompts) <= 8 and all(len(p) == width for p in prompts)\n            assert c[\'max_length\'] - width == 1024 and c[\'generation_kwargs\'][\'do_sample\'] is False\n            for sequence, meta in zip(checked[\'sequence_ids\'], checked[\'row_metadata\']):\n                assert tuple(meta[\'input_ids\']) in prompt_ids and meta[\'physical_batch_identity\'] == path.stem\n                assert meta[\'physical_batch_size\'] == len(prompts) and meta[\'max_new_tokens\'] == 1024\n                new = sequence[width:]; semantic = meta[\'semantic_new_ids\']\n                assert semantic == new[:len(semantic)] and meta[\'raw_generated_tokens\'] == len(new)\n                assert meta[\'semantic_generated_tokens\'] == len(semantic)\n                assert meta[\'at_token_cap\'] == (len(semantic) == 1024)\n            index[\'raw_generation_batches\'] += 1\n            add(str(path.relative_to(root)), data)\n        read_parallel(sorted((base / \'raw_generation_batches\').glob(\'*.json\')), raw_generation, \'raw generation batches\')\n\n        def raw_likelihood(path, data):\n            value = json.loads(data); checksum = value.pop(\'content_sha256\')\n            assert identity(value) == checksum\n            c = value[\'contract\']; assert path.stem == identity(c) and c[\'audit\'] == audit\n            name = c[\'task\']; assert name in (\'hellaswag\', \'boolq\', \'arc_easy\')\n            ids = c[\'source_ids\']; assert ids and len(ids) <= 100 and ids[0] % 100 == 0\n            assert ids == list(range(ids[0], min(ids[0] + 100, COUNTS[name])))\n            expected = [r for i in ids for r in rows_by_task[name][i][\'requests\']]\n            assert c[\'requests\'] == expected\n            assert len(value[\'responses\']) == len(value[\'metadata\']) == len(expected)\n            assert all(len(r) == 2 and type(r[1]) is bool and math.isfinite(r[0]) for r in value[\'responses\'])\n            index[\'raw_likelihood_chunks\'] += 1\n            add(str(path.relative_to(root)), data)\n        read_parallel(sorted((base / \'raw_likelihood_chunks\').glob(\'*.json\')), raw_likelihood, \'raw likelihood chunks\')\n        index_bytes = encoded(index)\n        archive.writestr(\'SNAPSHOT_INDEX.json\', index_bytes)\n    # Independent ZIP readback checks every member against its captured byte hash.\n    with zipfile.ZipFile(archive_path) as archive:\n        assert len(archive.namelist()) == len(index[\'files\']) + 1\n        assert archive.read(\'SNAPSHOT_INDEX.json\') == index_bytes\n        for rel, entry in index[\'files\'].items():\n            data = archive.read(rel)\n            assert len(data) == entry[\'bytes\'] and hashlib.sha256(data).hexdigest() == entry[\'sha256\']\n    return archive_path, index\n'

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
ROOT=Path('/content/drive/MyDrive/ATML-PA-3/Task1')
RUN='task1_cpu_resume_'+uuid.uuid4().hex
LOCAL=Path('/content')/RUN
LOCAL.mkdir();BUNDLE=LOCAL/'code';BUNDLE.mkdir()
START=time.monotonic()
try:
    # Fingerprint matches the EXACT existing numerical implementation, not a new experiment.
    def encoded(value):
        return (json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode()
    implementation=hashlib.sha256(encoded(EXPECTED_FILES)).hexdigest()
    backup=ROOT/'execution'/('baseline_batch8_V2_'+implementation)
    assert json.loads((backup/'FILES.json').read_text())==EXPECTED_FILES
    print('1/4 Verifying unchanged Block 15 source and both terminal receipts on CPU...',flush=True)
    import concurrent.futures
    def copy_source(item):
        name,expected=item;data=(backup/name).read_bytes()
        assert hashlib.sha256(data).hexdigest()==expected,name
        (BUNDLE/name).write_bytes(data)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(copy_source,EXPECTED_FILES.items()))
    (BUNDLE/'resume_snapshot_cpu.py').write_text(SNAPSHOT_SOURCE)
    for name in list(EXPECTED_FILES)+['resume_snapshot_cpu.py']:sys.modules.pop(name[:-3],None)
    sys.path.insert(0,str(BUNDLE))
    from evaluation_records import identity,publish,COUNTS
    from baseline_runtime import publish_bytes
    from archive_io import digest
    from baseline_v2_protocol import prepare
    from launch_baseline_v2 import valid_result
    from resume_snapshot_cpu import pack_records
    adopted=prepare(ROOT)
    contract={'condition':'original','model_identity':adopted['model_identity'],
              'protocol_identity':adopted['protocol_identity']}
    sessions={}
    run_ids=['task1_baseline_batch8_V2_f6b8daf1b5a245bfb0eee562fecca690',
             'task1_baseline_batch8_V2_fcfbcfdb65b04d5f8b4ddaf780197501']
    for run_id in run_ids:
        started=json.loads((ROOT/'logs'/(run_id+'_session_started.json')).read_text())
        final=json.loads((ROOT/'logs'/(run_id+'_session_final.json')).read_text())
        result_path=ROOT/'logs'/(run_id+'_result.json');result=json.loads(result_path.read_text())
        assert started['implementation_identity']==implementation
        assert started['model_identity']==contract['model_identity']
        assert started['protocol_identity']==contract['protocol_identity']
        assert final['worker_exit_code']==0 and final['worker_result_validated'] is True
        assert final['result_sha256']==digest(result_path)
        assert valid_result(result,0,started)
        assert result['status']=='TASK1_BASELINE_V2_PARTIAL_SAVED' and result['official_training_updates']==0
        sessions[run_id]={'result':result,'final_receipt_sha256':digest(ROOT/'logs'/(run_id+'_session_final.json'))}
    print('2/4 Reading and validating actual saved answers with eight CPU read threads.',flush=True)
    print('Progress below counts actual files, independently of the last GPU session counters.',flush=True)
    archive,index=pack_records(ROOT,LOCAL,contract,implementation,adopted['protocol'],
        {'gsm8k_cot':850,'hellaswag':10042,'boolq':3270,'arc_easy':2376})
    print('3/4 Saving a hash-checked single-file resume snapshot to Drive...',flush=True)
    snapshot_id=identity(index)
    archive_rel='runtime/resume_snapshots/'+snapshot_id+'.zip'
    archive_sha=publish_bytes(ROOT/archive_rel,archive.read_bytes())
    ready={'status':'TASK1_BASELINE_V2_RESUME_SNAPSHOT_READY','snapshot_identity':snapshot_id,
        'archive_path':archive_rel,'archive_sha256':archive_sha,'archive_bytes':archive.stat().st_size,
        'index_sha256':hashlib.sha256(encoded(index)).hexdigest(),'counts':index['counts'],
        'contract':contract,'numerical_implementation_identity':implementation,
        'original_numerical_files':EXPECTED_FILES,'source_run_ids':run_ids,
        'snapshot_is_read_transport_only':True,'old_results_modified':False,
        'new_model_inference_run':False,'training_updates':0}
    readiness_rel='configs/BASELINE_V2_RESUME_READY_'+snapshot_id+'.json'
    publish(ROOT/readiness_rel,ready)
    report={'status':'TASK1_BLOCK16_CPU_RESUME_AUDIT_OK','snapshot_identity':snapshot_id,
        'readiness_path':readiness_rel,'readiness_sha256':digest(ROOT/readiness_rel),
        'actual_scored_counts':index['counts'],'required_counts':COUNTS,
        'remaining_GSM_scored_answers':COUNTS['gsm8k_cot']-index['counts']['gsm8k_cot'],
        'complete_split_metrics_from_saved_scores':index['complete_metrics'],
        'raw_generation_batches_validated':index['raw_generation_batches'],
        'raw_likelihood_chunks_validated':index['raw_likelihood_chunks'],
        'record_read_and_check_seconds':index['record_read_and_check_seconds'],
        'qualitative_responses_reverified':24,'FP32_probe_file_hashes_reverified':64,
        'both_worker_results_and_final_receipts_verified':True,'packed_archive_bytes':ready['archive_bytes'],
        'old_results_modified':False,'new_model_inference_run':False,'GPU_work_run':False,
        'score_recomputation':'No stock per-answer rescore; complete means independently reaggregated from saved scalar scores',
        'prior_slow_resume_cause':'Code traverses old answers with per-file Drive exists/read calls; current packed transport avoids that in the next GPU run. Exact prior wall-time attribution is unmeasured.',
        'physical_GPU_deallocation_independently_verified':False,
        'elapsed_seconds':time.monotonic()-START,
        'next':'Return this JSON with GPU off; use the forthcoming snapshot-aware resume cell, not old Block 15.'}
    publish(ROOT/'logs'/(RUN+'_audit.json'),report)
    print('4/4 Snapshot and audit saved.',flush=True)
    print(json.dumps(report,indent=2),flush=True)
    print('TASK1_BLOCK16_OK — keep GPU off and send the complete JSON.',flush=True)
except BaseException:
    error=traceback.format_exc();print(error,flush=True)
    try:
        directory=ROOT/'logs'
        if directory.is_dir():(directory/(RUN+'_audit_error.txt')).write_text(error)
    except Exception:pass
    print('CPU audit failed; do not enable GPU. Return the full error.',flush=True)
    raise
