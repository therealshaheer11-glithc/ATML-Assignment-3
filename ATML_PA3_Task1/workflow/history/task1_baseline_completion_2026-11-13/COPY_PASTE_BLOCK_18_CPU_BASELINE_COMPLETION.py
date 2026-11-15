# Task 1 - Block 18: CPU full stock-score replay and verified baseline export.
# Select CPU; paste this ENTIRE file into one cell. No ZIP upload or GPU use.
import concurrent.futures, hashlib, json, os, shutil, subprocess, sys, threading, time, uuid, traceback, zipfile
from pathlib import Path
EXPECTED_FILES = {'batch8_records.py': '6738fcd2146a4057e54c7728129411700524e5f78a6efdfeccfd120f2589c4aa', 'batch8_hflm.py': '1b8b3810028123d6f6623398e028628cc3d9fc72070aad359d92fdc52023969e', 'batch8_pilot.py': '76171f667464327df7c3235cf7f9209dd18efaca8ff56b715c9493af8e156ec9', 'batch8_cpu_checks.py': 'ff5bfe4acc12b2e8ef0809f98c0d0b2d624f3e821567c32b06042043d1cdb905', 'prepare_batch8_cpu.py': '69a00eb778cdcb31a50e3908c87ce619aa34b3ca5e8407260e1b55cc5aeabcf9', 'restore_batch8_cpu.py': '32ebce8ea449913d28c8a6681bed8773b222c88babb07cbc09189d30207d8f53', 'archive_io.py': 'd5d059306f34a09eee8b4469d3f47597b07ec182f797d40e4a4958d2a1bf8f57', 'restore_baseline_cpu.py': 'f0e2d61a261c5ee8bd8b3ebc10250dd56defe9906ff139d21ef9ccb2745c7106', 'baseline_runtime.py': 'a2c420ed090689c1849d65d7de4717707a713937b1060bb9c65b6adf44a8723a', 'evaluation_records.py': '841fe3750ba33fb63d1a7e56a3ebd5248c013f3eada114e3690c4b5a8250c607', 'launch_baseline.py': 'e69a0ebbb48752a343490bfa0b38e94fec10ffe9d33b3205bde4dccf00aa0f95', 'release_coordination.py': '614b5bab4cf1d5482b6b672fb79047709944167382414e4cdafc1f5cec20642b', 'gpu_guardian_R3.py': '12c7dfa4637bea83e7aacd821743ebdcb626c291ff8b14d1374befa37fd2a06d', 'watchdog_evidence.py': '30688a268a2f2441c5dbe0adeabd78afad2a5d76af5c5e0747e31ed30de3618e', 'launch_batch8_pilot.py': 'c952aab3d630d37dcbaefe97bb394edbe2d412f816d7f3251f63f43dabb9ac06', 'batch8_pilot_entry.py': 'fead5e0d664b3ddbc02c63b2458d02ec7c818fe624f29da87f6a9c61d96f069f', 'pilot_launch_checks.py': 'd16b080d6658bd5ad96740e513f646e747d1b9001215691b143407991b6fb2db', 'guardian_selftest.py': '861987baeecb0a16ab60799f4b143f40d8680c967eaadb73951c9ad11dc07907', 'guardian_lifecycle_tests.py': '87509ab95e9947d875a4c7451d1482433d905c1858aceb7b74fbfaf55d4219c7', 'stock_evaluation.py': '9d6229054c031fb82887842cb5704f2cc9d4be17ba76e1d28e9a491bd00fc830', 'evaluation_v2_engine.py': '3a6307302cee3e8b97f39f33c1571ad79f8d23b577199846505241cc9a16ec19', 'evaluation_v2_hflm.py': '31cb7a1c82064df84f7672043d35d2404b4cd0a76aaa575963065c2a3695a441', 'baseline_v2_protocol.py': '3ac78bc667f4ef0ab1a4724ad04fbc9f5e1964c76c813d92acab4de094683413', 'baseline_v2_worker.py': '44d8ca63264df69f909be126ca965f2204ad944369c1696b7d095511e25f9392', 'launch_baseline_v2.py': 'd38dab7e09c47fb36f635e7a1585fea1b7156bcbe1feb7e36b4a58f1e298e494', 'baseline_v2_engine_checks.py': '5c0514b307b752228d13bd0d3fb449508038c911a276622448bbcb8452e651e5', 'baseline_v2_control_checks.py': '86d2c7718f162a169ac586fcbcd7f66365d61571c688feb4fd70e32d50f53e26'}
EXPECTED_TRANSPORT = {'snapshot_transport.py': 'c41e56c632f1d53365f2207ada41f00a761b3a8ace0e185149f47d0254852472', 'snapshot_worker_entry.py': '49a14478ec5d7d19e7eae4ca1d46391000349141a15a6fb5101d5d739abf010f', 'launch_snapshot_resume.py': '867948b60f979f0ee6397b25175d180cadac08502a840bd3fb1b1c617b041f1c'}
SNAPSHOT_SOURCE = '"""CPU-only audit and packed transport of existing V2 answers. No inference."""\nimport concurrent.futures\nimport gzip\nimport hashlib\nimport json\nimport math\nimport time\nimport zipfile\nfrom pathlib import Path\n\nfrom archive_io import digest, safe_relative\nfrom evaluation_records import COUNTS, aggregate, encoded, identity\nfrom batch8_records import load_batch\n\n\ndef read_parallel(paths, action, label):\n    paths = list(paths)\n    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:\n        # Bounded outstanding work; no thousands of decoded responses in a queue.\n        for start in range(0, len(paths), 128):\n            for result in pool.map(lambda p: (p, p.read_bytes()), paths[start:start + 128]):\n                action(*result)\n            if start == 0 or (start + 128) // 1024 != start // 1024 or start + 128 >= len(paths):\n                print(f\'{label}: {min(start + 128, len(paths))}/{len(paths)} files checked\', flush=True)\n\n\ndef validate_answer(record, task, source_index, row, contract):\n    assert record[\'contract\'] == contract and record[\'status\'] == \'complete\'\n    assert record[\'task\'] == task and record[\'source_index\'] == source_index\n    assert record[\'request_identity\'] == identity({\'source_index\': source_index, \'requests\': row[\'requests\']})\n    assert record[\'scores\'] and all(type(v) in (int, float) and math.isfinite(v) for v in record[\'scores\'].values())\n    assert len(record[\'response\']) == len(row[\'requests\'])\n\n\ndef pack_records(root, local, contract, implementation, protocol, minimum_counts):\n    root, local = Path(root), Path(local)\n    local.mkdir(parents=True, exist_ok=True)\n    namespace = \'results_batch8_V2/original\'\n    base = root / namespace\n    audit = {**contract, \'gpu_implementation_identity\': implementation}\n    ready = json.loads((root / \'configs/BASELINE_CPU_READY_V1.json\').read_text())\n    archive_path = local / \'verified_resume.zip\'\n    index = {\'schema_version\': 1, \'contract\': contract, \'numerical_implementation_identity\': implementation,\n             \'created_utc_unix\': time.time(), \'files\': {}, \'counts\': {}, \'complete_metrics\': {},\n             \'raw_generation_batches\': 0, \'raw_likelihood_chunks\': 0,\n             \'scope\': \'Verified existing bytes only; no inference, training or per-answer score recomputation\'}\n    rows_by_task = {}\n    with zipfile.ZipFile(archive_path, \'x\', zipfile.ZIP_DEFLATED) as archive:\n        def add(rel, data):\n            safe_relative(rel)\n            assert rel not in index[\'files\']\n            archive.writestr(rel, data)\n            index[\'files\'][rel] = {\'sha256\': hashlib.sha256(data).hexdigest(), \'bytes\': len(data)}\n\n        assert json.loads((base / \'CONTRACT.json\').read_text()) == contract\n        add(namespace + \'/CONTRACT.json\', (base / \'CONTRACT.json\').read_bytes())\n        for task, count in COUNTS.items():\n            entry = ready[\'benchmarks\'][task]\n            path = root / str(safe_relative(entry[\'path\']))\n            assert digest(path) == entry[\'sha256\']\n            with gzip.open(path, \'rt\') as stream: rows = [json.loads(line) for line in stream]\n            assert [r[\'source_index\'] for r in rows] == list(range(count))\n            rows_by_task[task] = rows\n            records = {}\n\n            def accept(path, data):\n                i = int(path.stem)\n                assert path.name == f\'{i:05}.json\' and 0 <= i < count and i not in records\n                record = json.loads(data)\n                validate_answer(record, task, i, rows[i], contract)\n                expected_metrics = {protocol[\'primary_metrics\'][task]}\n                if task in protocol[\'secondary_metrics\']: expected_metrics.add(protocol[\'secondary_metrics\'][task])\n                assert expected_metrics.issubset(record[\'scores\'])\n                records[i] = record\n                add(str(path.relative_to(root)), data)\n\n            started = time.monotonic()\n            read_parallel(sorted((base / task).glob(\'*.json\')), accept, task)\n            index.setdefault(\'record_read_and_check_seconds\', {})[task] = time.monotonic() - started\n            index[\'counts\'][task] = len(records)\n            assert len(records) >= minimum_counts[task], f\'{task}: fewer saved records than previously reported\'\n            if len(records) == count:\n                metrics = aggregate([records[i] for i in range(count)], task)\n                completion = base / f\'{task}_COMPLETE.json\'\n                saved = json.loads(completion.read_text())\n                assert saved == {\'audit\': audit, \'count\': count, \'metrics\': metrics}\n                index[\'complete_metrics\'][task] = metrics\n                add(str(completion.relative_to(root)), completion.read_bytes())\n            else:\n                assert not (base / f\'{task}_COMPLETE.json\').exists(), \'Incomplete split has a completion receipt\'\n                index.setdefault(\'partial_score_totals\', {})[task] = {\n                    k: sum(r[\'scores\'][k] for r in records.values()) for k in next(iter(records.values()))[\'scores\']\n                } if records else {}\n\n        # The raw cache is preserved too, including work from any unfinished chunk.\n        prompt_ids = {tuple(r[\'prompt_input_ids\']) for r in rows_by_task[\'gsm8k_cot\']}\n        def raw_generation(path, data):\n            value = json.loads(data); c = value[\'contract\']\n            assert path.stem == identity(c) and c[\'audit\'] == audit\n            # Production reader verifies checksum, tokens, row coverage and prefixes.\n            checked = load_batch(path, c)\n            prompts = c[\'padded_input_ids\']; width = len(prompts[0])\n            assert 1 <= len(prompts) <= 8 and all(len(p) == width for p in prompts)\n            assert c[\'max_length\'] - width == 1024 and c[\'generation_kwargs\'][\'do_sample\'] is False\n            for sequence, meta in zip(checked[\'sequence_ids\'], checked[\'row_metadata\']):\n                assert tuple(meta[\'input_ids\']) in prompt_ids and meta[\'physical_batch_identity\'] == path.stem\n                assert meta[\'physical_batch_size\'] == len(prompts) and meta[\'max_new_tokens\'] == 1024\n                new = sequence[width:]; semantic = meta[\'semantic_new_ids\']\n                assert semantic == new[:len(semantic)] and meta[\'raw_generated_tokens\'] == len(new)\n                assert meta[\'semantic_generated_tokens\'] == len(semantic)\n                assert meta[\'at_token_cap\'] == (len(semantic) == 1024)\n            index[\'raw_generation_batches\'] += 1\n            add(str(path.relative_to(root)), data)\n        read_parallel(sorted((base / \'raw_generation_batches\').glob(\'*.json\')), raw_generation, \'raw generation batches\')\n\n        def raw_likelihood(path, data):\n            value = json.loads(data); checksum = value.pop(\'content_sha256\')\n            assert identity(value) == checksum\n            c = value[\'contract\']; assert path.stem == identity(c) and c[\'audit\'] == audit\n            name = c[\'task\']; assert name in (\'hellaswag\', \'boolq\', \'arc_easy\')\n            ids = c[\'source_ids\']; assert ids and len(ids) <= 100 and ids[0] % 100 == 0\n            assert ids == list(range(ids[0], min(ids[0] + 100, COUNTS[name])))\n            expected = [r for i in ids for r in rows_by_task[name][i][\'requests\']]\n            assert c[\'requests\'] == expected\n            assert len(value[\'responses\']) == len(value[\'metadata\']) == len(expected)\n            assert all(len(r) == 2 and type(r[1]) is bool and math.isfinite(r[0]) for r in value[\'responses\'])\n            index[\'raw_likelihood_chunks\'] += 1\n            add(str(path.relative_to(root)), data)\n        read_parallel(sorted((base / \'raw_likelihood_chunks\').glob(\'*.json\')), raw_likelihood, \'raw likelihood chunks\')\n        index_bytes = encoded(index)\n        archive.writestr(\'SNAPSHOT_INDEX.json\', index_bytes)\n    # Independent ZIP readback checks every member against its captured byte hash.\n    with zipfile.ZipFile(archive_path) as archive:\n        assert len(archive.namelist()) == len(index[\'files\']) + 1\n        assert archive.read(\'SNAPSHOT_INDEX.json\') == index_bytes\n        for rel, entry in index[\'files\'].items():\n            data = archive.read(rel)\n            assert len(data) == entry[\'bytes\'] and hashlib.sha256(data).hexdigest() == entry[\'sha256\']\n    return archive_path, index\n'
COMPLETION_SOURCE = '"""Replay frozen stock scoring and export a complete baseline without inference."""\nimport gzip\nimport hashlib\nimport json\nimport shutil\nimport sys\nimport time\nimport zipfile\nfrom pathlib import Path\n\nsys.path.insert(0, str(Path(__file__).resolve().parent))\nfrom archive_io import digest, safe_relative\nfrom evaluation_records import COUNTS, encoded, identity, publish\n\n\ndef replay_rows(task, name, rows, read_record, factory, scorer):\n    assert [r[\'source_index\'] for r in rows] == list(range(COUNTS[name]))\n    totals = {}\n    for row in rows:\n        i = row[\'source_index\']; record = read_record(i)\n        requests = [factory(request_type=r[\'request_type\'], doc=row[\'document\'],\n            arguments=tuple(r[\'arguments\']), idx=r[\'idx\'], metadata=(name, i, 1)) for r in row[\'requests\']]\n        metrics, filtered = scorer(task, requests, record[\'response\'])\n        assert metrics == record[\'scores\'], f\'Stock score mismatch: {name}:{i}\'\n        assert identity(filtered) == identity(record[\'metadata\'][\'filtered\']), f\'Stock filter mismatch: {name}:{i}\'\n        if totals: assert set(totals) == set(metrics)\n        for metric, value in metrics.items(): totals[metric] = totals.get(metric, 0.) + value\n        if (i + 1) % 1000 == 0: print(f\'Stock CPU replay {name}: {i+1}/{len(rows)} match\', flush=True)\n    return {metric: value / len(rows) for metric, value in totals.items()}\n\n\ndef native_main(inputs, archive_path, output):\n    import importlib.metadata\n    import lm_eval\n    from lm_eval.api.instance import Instance\n    from stock_evaluation import load_task, score\n    started = time.monotonic(); inputs = Path(inputs)\n    assert importlib.metadata.version(\'lm_eval\') == \'0.4.9.2\'\n    hashes = json.loads((inputs / \'configs/HARNESS_SOURCE_HASHES.json\').read_text())\n    for rel, expected in hashes.items(): assert digest(Path(lm_eval.__file__).parent / rel) == expected\n    ready = json.loads((inputs / \'configs/BASELINE_CPU_READY_V1.json\').read_text())\n    report = {\'status\': \'TASK1_ORIGINAL_BASELINE_STOCK_REPLAY_OK\', \'counts\': COUNTS,\n        \'metrics\': {}, \'per_answer_scores_and_filtered_responses_all_match\': True,\n        \'new_model_inference_run\': False, \'GPU_work_run\': False, \'training_updates\': 0}\n    with zipfile.ZipFile(archive_path) as archive:\n        index = json.loads(archive.read(\'SNAPSHOT_INDEX.json\')); assert index[\'counts\'] == COUNTS\n        for name, entry in ready[\'benchmarks\'].items():\n            path = inputs / str(safe_relative(entry[\'path\'])); assert digest(path) == entry[\'sha256\']\n            with gzip.open(path, \'rt\') as stream: rows = [json.loads(line) for line in stream]\n            task, _, yaml_path = load_task(inputs, name)\n            assert digest(yaml_path) == entry[\'stock_yaml_sha256\'] and task.config.num_fewshot == 0\n            def reader(i):\n                rel = f\'results_batch8_V2/original/{name}/{i:05}.json\'\n                data = archive.read(rel); assert hashlib.sha256(data).hexdigest() == index[\'files\'][rel][\'sha256\']\n                return json.loads(data)\n            report[\'metrics\'][name] = replay_rows(task, name, rows, reader, Instance, score)\n            assert report[\'metrics\'][name] == index[\'complete_metrics\'][name]\n            print(f\'Stock CPU replay {name}: all {len(rows)} saved answers match.\', flush=True)\n    report[\'elapsed_seconds\'] = time.monotonic() - started\n    publish(output, report)\n    print(json.dumps(report, indent=2), flush=True)\n\n\ndef make_export(base_archive, target, extras, summary, replay):\n    """Add checked sources/references/receipts to already verified answer bytes."""\n    assert replay[\'status\'] == \'TASK1_ORIGINAL_BASELINE_STOCK_REPLAY_OK\'\n    assert replay[\'metrics\'] == summary[\'metrics\'] and replay[\'counts\'] == summary[\'benchmark_counts\'] == COUNTS\n    target = Path(target); assert not target.exists(); shutil.copyfile(base_archive, target)\n    members = {}\n    with zipfile.ZipFile(target, \'a\', zipfile.ZIP_DEFLATED) as archive:\n        assert len(archive.namelist()) == len(set(archive.namelist()))\n        for rel in archive.namelist():\n            data = archive.read(rel); members[rel] = {\'sha256\': hashlib.sha256(data).hexdigest(), \'bytes\': len(data)}\n        for rel, data in extras.items():\n            safe_relative(rel); assert rel not in members\n            archive.writestr(rel, data)\n            members[rel] = {\'sha256\': hashlib.sha256(data).hexdigest(), \'bytes\': len(data)}\n        manifest = {\'status\': \'TASK1_ORIGINAL_BASELINE_VERIFIED_EXPORT\', \'benchmark_counts\': COUNTS,\n            \'metrics\': summary[\'metrics\'], \'audit\': summary[\'audit\'], \'files\': members,\n            \'stock_scores_and_filters_replayed_for_all_answers\': True,\n            \'human_qualitative_annotation_pending\': True, \'official_training_updates\': 0,\n            \'no_original_model_weights_included\': True}\n        archive.writestr(\'EXPORT_MANIFEST.json\', encoded(manifest))\n    with zipfile.ZipFile(target) as archive:\n        assert set(archive.namelist()) == set(members) | {\'EXPORT_MANIFEST.json\'}\n        assert json.loads(archive.read(\'EXPORT_MANIFEST.json\')) == manifest\n        for rel, entry in members.items():\n            data = archive.read(rel)\n            assert len(data) == entry[\'bytes\'] and hashlib.sha256(data).hexdigest() == entry[\'sha256\']\n    return manifest\n\n\nif __name__ == \'__main__\': native_main(*sys.argv[1:])\n'

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
RUN='task1_baseline_completion_cpu_'+uuid.uuid4().hex
JOB=Path('/content')/RUN;JOB.mkdir();BUNDLE=JOB/'code';BUNDLE.mkdir()
START=time.monotonic()
try:
    print('1/4 Verifying frozen sources, completed baseline and final receipts...',flush=True)
    def encoded(value):
        return (json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode()
    implementation=hashlib.sha256(encoded(EXPECTED_FILES)).hexdigest()
    source=ROOT/'execution'/('baseline_batch8_V2_'+implementation)
    assert json.loads((source/'FILES.json').read_text())==EXPECTED_FILES
    def source_file(item):
        name,expected=item;data=(source/name).read_bytes()
        assert hashlib.sha256(data).hexdigest()==expected
        (BUNDLE/name).write_bytes(data)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(source_file,EXPECTED_FILES.items()))
    for name,text in [('resume_snapshot_cpu.py',SNAPSHOT_SOURCE),('baseline_completion_cpu.py',COMPLETION_SOURCE)]:
        (BUNDLE/name).write_text(text)
    for name in list(EXPECTED_FILES)+['resume_snapshot_cpu.py','baseline_completion_cpu.py']:sys.modules.pop(name[:-3],None)
    sys.path.insert(0,str(BUNDLE))
    from evaluation_records import COUNTS,identity,publish
    from baseline_runtime import publish_bytes
    from archive_io import digest,safe_relative
    from baseline_v2_protocol import prepare
    from launch_baseline_v2 import valid_result
    from resume_snapshot_cpu import pack_records
    from baseline_completion_cpu import make_export
    adopted=prepare(ROOT)
    run_id='task1_snapshot_baseline_adbf22d941054189984a7f17503275b2'
    result_path=ROOT/'logs'/(run_id+'_result.json')
    assert digest(result_path)=='a76bf1f88499fad7cb813a5f7dc4266732f5e200db5e5201fec9bd6d0207dc6d'
    value=json.loads(result_path.read_text())
    session=json.loads((ROOT/'logs'/(run_id+'_session_started.json')).read_text())
    final=json.loads((ROOT/'logs'/(run_id+'_session_final.json')).read_text())
    assert valid_result(value,0,session) and value['status']=='TASK1_BASELINE_V2_COMPLETE_SAVED'
    assert final['worker_exit_code']==0 and final['worker_result_validated'] is True
    assert final['result_sha256']==digest(result_path) and session['implementation_identity']==implementation
    summary=json.loads((ROOT/'results_batch8_V2/original/BASELINE_COMPLETE.json').read_text())
    assert summary==value['baseline_summary'] and summary['benchmark_counts']==COUNTS
    assert summary['audit']['model_identity']==adopted['model_identity']
    assert summary['audit']['protocol_identity']==adopted['protocol_identity']
    assert summary['audit']['gpu_implementation_identity']==implementation
    transport=value['packed_resume_transport']
    assert transport['record_IO']=={'snapshot_record_reads':16538,'remote_record_checks_for_unindexed_IDs':938,
        'new_records_saved_with_original_Drive_readback':469}
    transport_id=identity(EXPECTED_TRANSPORT)
    assert transport['transport_implementation_identity']==transport_id
    transport_source=ROOT/'execution'/('baseline_snapshot_transport_'+transport_id)
    assert json.loads((transport_source/'FILES.json').read_text())==EXPECTED_TRANSPORT
    transport_bytes={}
    for name,expected in EXPECTED_TRANSPORT.items():
        data=(transport_source/name).read_bytes();assert hashlib.sha256(data).hexdigest()==expected
        transport_bytes[name]=data
    print('2/4 Validating all 17,007 Drive answers, raw journals and complete split means on CPU...',flush=True)
    contract={k:summary['audit'][k] for k in ('condition','model_identity','protocol_identity')}
    archive,index=pack_records(ROOT,JOB,contract,implementation,adopted['protocol'],COUNTS)
    assert index['counts']==COUNTS and index['complete_metrics']==summary['metrics']
    # Copy only frozen text inputs needed by the stock scorer, not model weights.
    inputs=JOB/'inputs';inputs.mkdir()
    paths=['configs/BASELINE_CPU_READY_V1.json','configs/HARNESS_SOURCE_HASHES.json','manifests/BENCHMARK_FILES.json']
    ready=adopted['ready']
    for entry in ready['benchmarks'].values():paths.append(entry['path'])
    raw=json.loads((ROOT/'manifests/BENCHMARK_FILES.json').read_text())
    for entry in raw.values():paths.append(entry['path'])
    for rel in paths:
        safe_relative(rel);publish_bytes(inputs/rel,(ROOT/rel).read_bytes())
    assert digest(inputs/'configs/HARNESS_SOURCE_HASHES.json')==adopted['protocol']['harness_source_manifest_sha256']
    print('3/4 Restoring frozen CPU software, then replaying the stock scorer for every saved response...',flush=True)
    from restore_baseline_cpu import ensure_archive
    runtime=json.loads((ROOT/'configs/GPU_PREFLIGHT_KIT_READY.json').read_text())['portable_runtime']
    inventory_path=ROOT/'runtime'/('inventory_'+runtime['inventory_sha256']+'.json')
    assert digest(inventory_path)==runtime['inventory_sha256']
    inventory=json.loads(inventory_path.read_text())
    local=Path('/content/atml_pa3_task1');local.mkdir(exist_ok=True)
    assert not local.is_symlink() and inventory['base_directory']==str(local) and inventory['python_version']=='3.12.12'
    ensure_archive(ROOT,local,JOB,runtime,inventory['files'],local)
    env=os.environ.copy()
    for key in ('PYTHONPATH','PYTHONHOME','VIRTUAL_ENV','UV_SYSTEM_PYTHON'):env.pop(key,None)
    env.update(PYTHONNOUSERSITE='1',CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',
        HF_HUB_DISABLE_TELEMETRY='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    replay_path=JOB/'NATIVE_SCORE_REPLAY.json';process=None
    try:
        with (ROOT/'logs'/(RUN+'_stock_replay_console.txt')).open('x') as log:
            process=subprocess.Popen([str(local/'venv312/bin/python'),'-I','-u',str(BUNDLE/'baseline_completion_cpu.py'),
                str(inputs),str(archive),str(replay_path)],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                text=True,bufsize=1,start_new_session=True)
            for line in process.stdout:print(line,end='',flush=True);log.write(line);log.flush()
            assert process.wait()==0,'Stock scorer replay failed; return the preserved CPU console.'
    finally:
        if process is not None and process.poll() is None:
            import signal
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
    replay=json.loads(replay_path.read_text())
    assert replay['metrics']==summary['metrics'] and replay['counts']==COUNTS
    print('4/4 Preserving responses, probes, prompts, source and receipts in one verified baseline export...',flush=True)
    extras={}
    def include(rel):
        safe_relative(rel)
        if rel not in extras:extras[rel]=(ROOT/rel).read_bytes()
    for rel in paths:include(rel)
    for rel in ['results_batch8_V2/original/BASELINE_COMPLETE.json','results_batch8_V2/original/REUSED_REFERENCE_MANIFEST.json',
        'configs/EVALUATION_PROTOCOL_V1.json','configs/EVALUATION_PROTOCOL_V2_BATCH8.json',
        'configs/SOURCE_REVISIONS.json','configs/ENVIRONMENT_READY.json','configs/DEPENDENCY_LOCK.json','configs/requirements.lock.txt',
        'manifests/ORIGINAL_MODEL_FILES.json','manifests/TOKENIZER_FILES.json','sources/first_plot_questions.yaml',
        ready['qualitative_path'],ready['probe_path']]:include(rel)
    assert hashlib.sha256(extras['configs/requirements.lock.txt']).hexdigest()=='54ad919ba1a099e6327014d51b7f441ceee5df7c93330278e4f85ad82e43cc06'
    for rel in [summary['references']['qualitative_manifest'],summary['references']['probe_manifest']]:
        include(rel);manifest=json.loads(extras[rel])
        for entry in manifest['entries']:
            include(entry['path']);assert hashlib.sha256(extras[entry['path']]).hexdigest()==entry['sha256']
            if 'task' in entry:include(str(Path(entry['path']).with_suffix('.json')))
            else:include(json.loads(extras[entry['path']])['metadata']['raw_record'])
    for observed_run in ['task1_baseline_batch8_V2_f6b8daf1b5a245bfb0eee562fecca690',
        'task1_baseline_batch8_V2_fcfbcfdb65b04d5f8b4ddaf780197501',run_id]:
        for suffix in ('_session_started.json','_session_final.json','_result.json','_worker_console.txt'):
            include('logs/'+observed_run+suffix)
    for suffix in ('_transport_started.json','_transport_result.json'):include('logs/'+run_id+suffix)
    assert json.loads(extras['logs/'+run_id+'_transport_result.json'])==transport
    for name in EXPECTED_FILES:extras['reproducibility/core/'+name]=(BUNDLE/name).read_bytes()
    for name,data in transport_bytes.items():extras['reproducibility/transport/'+name]=data
    extras['reproducibility/resume_snapshot_cpu.py']=SNAPSHOT_SOURCE.encode()
    extras['reproducibility/baseline_completion_cpu.py']=COMPLETION_SOURCE.encode()
    extras['NATIVE_SCORE_REPLAY.json']=replay_path.read_bytes()
    target=JOB/'Task_1_Original_Baseline_V2.zip'
    export_manifest=make_export(archive,target,extras,summary,replay)
    export_id=identity(export_manifest)
    rel='exports/Task_1_Original_Baseline_V2_'+export_id[:16]+'.zip'
    archive_sha=publish_bytes(ROOT/rel,target.read_bytes())
    result={'status':'TASK1_BLOCK18_BASELINE_AUDIT_EXPORT_OK','export_identity':export_id,
        'export_path':rel,'export_sha256':archive_sha,'export_bytes':target.stat().st_size,
        'export_members':len(export_manifest['files'])+1,'benchmark_counts':COUNTS,'metrics':summary['metrics'],
        'all_17007_stock_scores_and_filtered_responses_replayed':True,
        'raw_generation_batches_validated':index['raw_generation_batches'],'raw_likelihood_chunks_validated':index['raw_likelihood_chunks'],
        'qualitative_responses_verified':24,'FP32_probe_file_hashes_verified':64,
        'human_qualitative_annotation_pending':True,'official_training_updates':0,
        'GPU_work_run':False,'new_model_inference_run':False,'prior_results_modified':False,
        'elapsed_seconds':time.monotonic()-START,'physical_GPU_deallocation_independently_verified':False,
        'next':'Return this complete JSON with GPU off. Next work is approved training-loop and recovery implementation; do not rerun baseline.'}
    publish(ROOT/'configs'/('ORIGINAL_BASELINE_V2_AUDITED_'+export_id+'.json'),result)
    publish(ROOT/'logs'/(RUN+'_audit.json'),result)
    print(json.dumps(result,indent=2),flush=True)
    print('TASK1_BLOCK18_OK — keep GPU off and send the complete JSON.',flush=True)
except BaseException:
    error=traceback.format_exc();print(error,flush=True)
    try:
        if (ROOT/'logs').is_dir():(ROOT/'logs'/(RUN+'_error.txt')).write_text(error)
    except Exception:pass
    print('CPU completion audit failed; keep GPU off and return the full error.',flush=True)
    raise
