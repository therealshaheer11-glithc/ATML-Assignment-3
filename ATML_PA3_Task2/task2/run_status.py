"""Conservative, stdlib-only continuation verdicts after worker exit."""
import hashlib
import json
import math
from pathlib import Path
from .protocol import DEV_UPDATES,STAGES


def _json_receipt(receipt,root):
    path=Path(receipt['path'])
    if not path.resolve().is_relative_to(root.resolve()):raise ValueError('Artifact is outside the task run.')
    raw=path.read_bytes()
    if len(raw)!=receipt['bytes'] or hashlib.sha256(raw).hexdigest()!=receipt['sha256']:
        raise ValueError('Artifact checksum or byte count differs: '+str(path))
    return json.loads(raw)


def _checkpoint(receipt,root):
    path=Path(receipt['path'])
    if not path.resolve().is_relative_to(root.resolve()):raise ValueError('Checkpoint is outside the task run.')
    if receipt.get('readback_verified') is not True or path.stat().st_size!=receipt['bytes']:
        raise ValueError('Checkpoint lacks verified readback or differs in size.')
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    if digest.hexdigest()!=receipt['sha256']:raise ValueError('Checkpoint checksum differs.')


def assess_run(root,attempt,package_sha,worker_code,reference,controller_error=None,console_error=None,diagnostic_error=None):
    verdict={'schema':'atml-pa3-task2-continuation-verdict-v1','code':'REFER','safe_to_continue':False,
        'reason':None,'global_updates':None,'total_required_updates':4500}
    if worker_code!=0 or controller_error or console_error or diagnostic_error:
        verdict['reason']='Worker/controller failed or required diagnostic persistence failed.'
        return verdict
    try:
        root=Path(root)
        result=json.loads((root/'logs'/('block_14_'+attempt+'.json')).read_bytes())
        if result['status'] not in ('passed','checkpointed'):raise ValueError('No successful or safely checkpointed segment result.')
        if result['training_package_sha256']!=package_sha:raise ValueError('Result belongs to another package.')
        for key in ('dataset_identity','initial_manifest_sha256','execution_recipe_sha256'):
            if result[key]!=reference[key]:raise ValueError('Result identity differs: '+key)
        start,end=result['start_global_update'],result['end_global_update']
        if type(start) is not int or type(end) is not int or not reference['latest_observed_global_update']<=start<=end<=4500:
            raise ValueError('Update counters are invalid or older than the verified continuation source.')
        verdict['global_updates']=end
        if result['successful_updates_this_segment']!=end-start or not 0<=end-start<=250:
            raise ValueError('Successful-update counters disagree or exceed the segment limit.')
        if result['total_required_updates']!=4500 or result['segment_maximum_updates']!=250:
            raise ValueError('Training budget differs.')
        if not math.isfinite(result['AMP_scale']) or result['AMP_scale']<=0:raise ValueError('Invalid AMP scale.')
        pointer_path=root/'manifests/active_task2_training.json'
        pointer_raw=pointer_path.read_bytes();pointer=json.loads(pointer_raw)
        if pointer['global_updates']!=end:raise ValueError('Active pointer does not match the completed update.')
        for key in ('dataset_identity','initial_manifest_sha256','execution_recipe_sha256'):
            if pointer[key]!=result[key]:raise ValueError('Active pointer identity differs: '+key)
        progress=_json_receipt(pointer['progress'],root)
        for key in ('global_updates','stage','stage_updates','checkpoint','dev_receipts','training_log_receipts'):
            other={'global_updates':end,'stage':result['stage'],'stage_updates':result['stage_updates'],
                'checkpoint':result['checkpoint'],'dev_receipts':result['development_receipts'],
                'training_log_receipts':result['training_log_receipts']}[key]
            if progress[key]!=other:raise ValueError('Saved progress differs: '+key)
        counts={int(key):value for key,value in result['stage_updates'].items()}
        if set(counts)!=set(STAGES) or sum(counts.values())!=end or any(type(value) is not int or not 0<=value<=STAGES[stage]['updates'] for stage,value in counts.items()):
            raise ValueError('Stage counts disagree with successful updates.')
        if counts!={1:min(end,2250),2:min(max(end-2250,0),1500),3:max(end-3750,0)}:
            raise ValueError('Stage progression differs from the approved curriculum.')
        _checkpoint(result['checkpoint'],root)
        expected_update=1
        for receipt in result['training_log_receipts']:
            rows=_json_receipt(receipt,root)
            if not rows:raise ValueError('Empty training log shard.')
            for row in rows:
                if row['global_update']!=expected_update:raise ValueError('Training log has a missing or repeated update.')
                if len(row['examples'])!=16 or len(row['pair_kinds'])!=8:
                    raise ValueError('Training log does not contain complete effective16 batches.')
                if not math.isfinite(row['loss']) or len(row['micro_losses'])!=2 or not all(math.isfinite(value) for value in row['micro_losses']):
                    raise ValueError('Invalid training loss in the saved log.')
                expected_update+=1
        if expected_update!=end+1:raise ValueError('Saved training logs do not reach the final update.')
        for update in DEV_UPDATES:
            if update>end:continue
            metrics=_json_receipt(result['development_receipts'][str(update)],root)
            if metrics['global_update']!=update or metrics['initial_manifest_sha256']!=result['initial_manifest_sha256']:
                raise ValueError('Required development receipt differs.')
            if not metrics['training_state_and_RNG_restored_exactly'] or metrics['generation_performed']:
                raise ValueError('Development did not preserve the agreed monitoring contract.')
            losses=[metrics['caption']['correct_image_CE']]
            for suite in metrics['suites'].values():
                losses.append(suite['swap_penalty_NLL'])
                losses.extend(row['mean_reference_NLL'] for row in suite['conditions'].values())
            if not all(math.isfinite(value) for value in losses):raise ValueError('Nonfinite development metric.')
        if pointer_path.read_bytes()!=pointer_raw:raise ValueError('Active progress changed during the final check.')
        verdict['checkpoint_sha256']=result['checkpoint']['sha256']
        if end==4500:
            if counts!={1:2250,2:1500,3:750}:raise ValueError('Final curriculum is incomplete.')
            verdict.update(code='COMPLETE',reason='All 4500 training updates and required development checkpoints are verified.')
        elif end==start:
            verdict['reason']='No successful training updates completed in this session; request review before another allocation.'
        else:
            verdict.update(code='CONTINUE',safe_to_continue=True,
                reason='No worker/controller error detected; checkpoint, progress, logs and required development receipts verified.',
                next_segment_target=min(4500,end+250),time_limited_stop=result['status']=='checkpointed')
    except Exception as error:
        verdict['code']='REFER';verdict['safe_to_continue']=False
        verdict['reason']='Final verification failed: '+str(error)
    return verdict


def banner(verdict):
    code=verdict['code'];step=verdict.get('global_updates')
    if code=='CONTINUE' and verdict.get('safe_to_continue'):
        return ('\n'+'='*68+'\nSAFE TO CONTINUE - NO ERROR DETECTED\n'
            f'Verified progress: {step}/4500 updates. Checkpoint saved and checked.\n'
            'The runtime disconnect after this message is planned.\n'
            'NEXT: reconnect T4, rerun this same loader and upload this same ZIP.\n'
            'You can continue without sending me each successful run.\n'+'='*68)
    if code=='COMPLETE':
        return ('\n'+'='*68+'\nTRAINING COMPLETE - 4500/4500 UPDATES VERIFIED\n'
            'Do not run another training segment.\n'
            'Send me this output so we can start final evaluation and the report.\n'+'='*68)
    return ('\n'+'='*68+'\nSTOP - SEND THIS OUTPUT TO ME\n'
        +('Latest reported progress: '+str(step)+'/4500.\n' if step is not None else '')
        +'Reason: '+str(verdict.get('reason') or 'No verified completion result is available.')+'\n'
        +'Do not rerun yet. Send the complete output for review.\n'+'='*68)
