"""Read original saved GPU evidence, validate the supervisor fix on CPU, record gates."""
import gzip
import hashlib
import importlib.util
import json
import statistics
import sys
import time
import uuid
from pathlib import Path


RUN_ID='task1_l4_d12659f4ab3c4149954744514eae36d7'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def dump(path,value):
    data=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=data:raise RuntimeError('Existing audit differs; preserving it: '+str(path))
        return
    pending=path.with_name(path.name+'.pending_'+uuid.uuid4().hex);pending.write_bytes(data)
    assert pending.read_bytes()==data
    if path.exists():raise RuntimeError('Concurrent review')
    pending.rename(path);assert path.read_bytes()==data


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj


def main(root,bundle):
    root,bundle=Path(root),Path(bundle);started=time.monotonic()
    print('1/5 Reading and validating the ORIGINAL saved L4 scratch result...',flush=True)
    result_path=root/'logs'/(RUN_ID+'_result.json')
    result=json.loads(result_path.read_text())
    assert result['status']=='TASK1_L4_PREFLIGHT_OK' and result['official_training_updates']==0
    assert result['scratch_weights_discarded'] and result['gpu_name']=='NVIDIA L4'
    modes=result['results'];assert [r['variant'] for r in modes]==['full','l2sp','oplora']
    summary={}
    for r in modes:
        assert r['scratch_successful_updates']==2 and len(r['steps'])==2
        assert (r['microbatch'],r['accumulation'],r['sequence_length'])==(2,16,2048)
        assert r['optimizer_states_fp32'] and not r['overflow_attempts']
        assert r['headroom_fraction_using_peak_reserved']>=.1
        assert r['trainable_parameters']==(10092544 if r['variant']=='oplora' else 596049920)
        summary[r['variant']]={'peak_reserved_GiB':round(r['peak_reserved_bytes']/2**30,3),
            'headroom_percent':round(100*r['headroom_fraction_using_peak_reserved'],2),
            'mean_scratch_update_seconds':statistics.mean(v['seconds'] for v in r['steps'])}
    session_path=root/'logs'/(RUN_ID+'_session.json')
    session=json.loads(session_path.read_text())
    assert session['displayed_units_per_hour']==2.02 and session['balance_at_start']==78.09
    print(json.dumps(summary,indent=2),flush=True)
    print('2/5 Testing repaired completion/parent acknowledgement and prior failure scenarios on CPU...',flush=True)
    test_root=Path('/content')/('task1_supervisor_R2_tests_'+uuid.uuid4().hex)
    original_tests=module(bundle/'guardian_selftest.py','original_safety_tests')
    prior=original_tests.run_tests(bundle/'gpu_guardian_R2.py',test_root/'prior')
    lifecycle=module(bundle/'guardian_lifecycle_tests.py','completion_safety_tests')
    new=lifecycle.run_tests(bundle/'gpu_guardian_R2.py',test_root/'lifecycle')
    assert len(prior)==6 and all(prior.values()) and len(new)==4 and all(new.values())
    coordination=module(bundle/'release_coordination.py','release_coordination')
    release_checks=coordination.cpu_checks(test_root/'release_coordination')
    print(json.dumps({'original_scenarios':prior,'completion_handshake_scenarios':new,
        'release_coordination':release_checks},indent=2),flush=True)
    print('3/5 Recording exact fixed-order training lengths for later cost planning...',flush=True)
    data=json.loads((root/'manifests/TRAINING_DATA_READY.json').read_text())
    source_ids=json.loads((root/'configs/TRAIN_SOURCE_IDS.json').read_text())
    seen=[];lengths=[];supervised=0
    for shard in data['shards']:
        path=root/shard['path'];assert digest(path)==shard['sha256']
        with gzip.open(path,'rt') as f:
            for line in f:
                row=json.loads(line);seen.append(row['source_index']);lengths.append(len(row['input_ids']))
                supervised+=row['n_supervised_tokens']
    assert seen==source_ids and len(lengths)==40000 and supervised==7100056
    micro_tokens=[2*max(lengths[i:i+2]) for i in range(0,40000,2)]
    update_tokens=[sum(micro_tokens[i:i+16]) for i in range(0,len(micro_tokens),16)]
    assert len(update_tokens)==1250
    lengths_record={'examples':40000,'updates':1250,'microbatch':2,'accumulation':16,
        'no_reordering_or_bucketing':True,'total_unpadded_input_tokens':sum(lengths),
        'total_dynamic_microbatch_padded_input_tokens':sum(micro_tokens),
        'total_supervised_tokens':supervised,'median_padded_tokens_per_update':statistics.median(update_tokens),
        'min_padded_tokens_per_update':min(update_tokens),'max_padded_tokens_per_update':max(update_tokens),
        'maximum_length_scratch_budget_tokens':1250*32*2048,
        'caution':'Token count is a cost proxy, not a linear wall-time forecast; evaluation and saves add cost.'}
    dump(root/'configs/FIXED_ORDER_LENGTH_COSTS.json',lengths_record)
    print('4/5 Acquiring and validating the approved unchanged public qualitative prompt file...',flush=True)
    qualitative=module(bundle/'qualitative_source.py','approved_public_questions')
    questions=qualitative.main(root,test_root)
    print(json.dumps({'prompts':questions['prompt_count'],'commit':questions['revision']['commit'],
        'sha256':questions['sha256'],'annotation_responses':questions['annotation_responses']},indent=2),flush=True)
    print('5/5 Preserving the repaired safety implementation and an honest result/release review...',flush=True)
    files=('gpu_guardian_R2.py','guardian_lifecycle_tests.py','guardian_selftest.py',
           'preflight_worker_R2.py','review_preflight_R2.py','qualitative_source.py','release_coordination.py')
    fingerprints={name:digest(bundle/name) for name in files}
    identity=hashlib.sha256(json.dumps(fingerprints,sort_keys=True).encode()).hexdigest()
    destination=root/'execution'/('safety_R2_'+identity)
    destination.mkdir(parents=True,exist_ok=True)
    for name in files:
        path=destination/name;payload=(bundle/name).read_bytes()
        if path.exists():assert digest(path)==fingerprints[name]
        else:path.write_bytes(payload);assert digest(path)==fingerprints[name]
    readiness={'schema_version':2,'implementation_path':str(destination.relative_to(root)),
        'files':fingerprints,'prior_safety_tests':prior,'completion_tests':new,'release_coordination_tests':release_checks,
        'legacy_block08_must_not_be_reused':True,'new_controller_integration_pending':True,
        'completion_protocol':'Worker publishes complete; parent verifies exit/result and records session; parent sends finished ACK; guardian releases. Unacknowledged completion gets at most 30 seconds, never beyond the absolute deadline.',
        'release_protocol':'Guardian is primary. Parent release is fallback after bounded observation, not an immediate competing normal request.',
        'physical_runtime_release_verified':False}
    dump(root/'configs/SUPERVISOR_R2_READY.json',readiness)
    review={'source_run':RUN_ID,'original_result_sha256':digest(result_path),
        'numerical_capacity':'passed under the recorded warn-only kernel settings',
        'scratch_summary':summary,'wrapper':'failed after result publication; completion race reproduced and repaired on CPU',
        'release':'not independently confirmed; notebook RuntimeManagementError is not proof of either success or failure',
        'original_session_record_status':session.get('status'),'original_evidence_preserved':True,
        'supervisor_R2_CPU_tests_passed':True,'strict_deterministic_gpu_check_pending':True,
        'future_determinism_setting':'torch.use_deterministic_algorithms(True, warn_only=False)',
        'determinism_reference':'PyTorch 2.8 attention_backward.cu selects deterministic Flash backward only when warn_only is false',
        'GPU_rerun_performed_for_wrapper_fix':False,'official_training_updates':0,
        'total_billed_units':'not determined from worker timing; includes restoration and release time',
        'qualitative_source_validated':True,'qualitative_source_commit':questions['revision']['commit']}
    dump(root/'logs'/(RUN_ID+'_review_R2.json'),review)
    receipt={'status':'TASK1_BLOCK09_CPU_REVIEW_OK','saved_GPU_result_validated':True,
        'supervisor_tests_passed':len(prior)+len(new),'release_coordination_checks':release_checks,'scratch_summary':summary,
        'fixed_order_length_statistics':lengths_record,'supervisor_R2_path':str(destination),
        'strict_deterministic_gpu_check_pending':True,'physical_release_confirmation':'not observable from this CPU audit',
        'qualitative_prompts_frozen':questions['prompt_count'],'qualitative_source_sha256':questions['sha256'],
        'gpu_work_run_in_this_stage':False,'official_training_started':False,
        'elapsed_seconds':round(time.monotonic()-started,1)}
    dump(root/'logs'/('block09_'+uuid.uuid4().hex+'.json'),receipt)
    print(json.dumps(receipt,indent=2),flush=True)
    print('TASK1_BLOCK09_CPU_REVIEW_OK — keep CPU selected and send the summary.',flush=True)


if __name__=='__main__':main(*sys.argv[1:])
