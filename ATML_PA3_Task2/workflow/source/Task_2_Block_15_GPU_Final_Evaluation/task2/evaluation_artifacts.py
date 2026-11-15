"""Readback-verified evaluation artifacts and independent final status checks."""
import hashlib
import json
from pathlib import Path
from .final_evaluation import batches, validate_chunk, encoded
from .session_io import bounded_io, save_bytes


def read_receipt(receipt, root):
    path = Path(receipt['path'])
    if not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('Evaluation artifact is outside the run.')
    raw = path.read_bytes()
    if len(raw) != receipt['bytes'] or hashlib.sha256(raw).hexdigest() != receipt['sha256']:
        raise ValueError('Evaluation artifact checksum differs: ' + str(path))
    return json.loads(raw)


def write_artifact(path, value):
    raw = encoded(value)
    with bounded_io(45, 'Evaluation artifact save and readback'):
        save_bytes(path, raw)
        if Path(path).read_bytes() != raw:
            raise IOError('Evaluation artifact readback differs.')
    return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw), 'readback_verified': True}


def verify_manifest(manifest, plan, recipe_sha, root):
    if manifest['recipe_sha256'] != recipe_sha or manifest['plan_sha256'] != hashlib.sha256(encoded(plan)).hexdigest():
        raise ValueError('Evaluation manifest recipe or plan differs.')
    expected = {key: (condition, rows) for key, condition, rows in batches(plan)}
    if not set(manifest['batches']) <= set(expected):
        raise ValueError('Evaluation manifest contains unknown or duplicated batches.')
    count = 0
    for key, receipt in manifest['batches'].items():
        condition, rows = expected[key]
        count += len(validate_chunk(read_receipt(receipt, root), recipe_sha, key, condition, rows))
    if count != manifest['verified_answers'] or manifest['complete'] != (len(manifest['batches']) == len(expected)):
        raise ValueError('Evaluation manifest completion counts disagree.')
    return count


def assess_evaluation(root, attempt, package_sha, worker_code, error=None):
    verdict = {'code': 'REFER', 'reason': 'No verified evaluation result.', 'verified_answers': None}
    if worker_code != 0 or error:
        verdict['reason'] = 'Worker/controller or diagnostic persistence failed: ' + str(error)
        return verdict
    try:
        root = Path(root)
        result = json.loads((root/'logs'/f'block_15_{attempt}.json').read_bytes())
        if result['package_sha256'] != package_sha or result['status'] not in ('passed', 'checkpointed'):
            raise ValueError('No matching successful evaluation result.')
        reference = result['source_reference']
        training_raw = (root/'manifests/active_task2_training.json').read_bytes()
        training = json.loads(training_raw)
        if training['global_updates'] != 4500 or training['progress'] != reference['progress']:
            raise ValueError('Final training source changed.')
        progress = read_receipt(training['progress'], root)
        if progress['checkpoint'] != reference['checkpoint'] or progress['stage_updates'] != {'1':2250,'2':1500,'3':750}:
            raise ValueError('Final training source is not the completed approved curriculum.')
        recipe = read_receipt(result['recipe'], root)
        if recipe['source_reference'] != reference:
            raise ValueError('Evaluation source differs.')
        plan = read_receipt(result['plan'], root)
        manifest = read_receipt(result['manifest'], root)
        count = verify_manifest(manifest, plan, result['recipe']['sha256'], root)
        evaluation = json.loads((root/'manifests/active_task2_evaluation.json').read_bytes())
        if evaluation['manifest'] != result['manifest'] or evaluation['recipe_sha256'] != result['recipe']['sha256']:
            raise ValueError('Active evaluation pointer differs.')
        if count != result['verified_answers'] or count != evaluation['verified_answers']:
            raise ValueError('Evaluation progress counts differ.')
        if result['training_state_modified'] or (root/'manifests/active_task2_training.json').read_bytes() != training_raw:
            raise ValueError('Training source was modified during evaluation verification.')
        verdict.update(verified_answers=count, total_answers=1280)
        if manifest['complete'] and count == 1280:
            verdict.update(code='COMPLETE', reason='All 1280 final generations and original-reference NLL records saved and verified. Switch to CPU for statistics and plots.')
        elif result['new_answers_this_run'] > 0:
            verdict.update(code='CONTINUE', reason='Verified partial evaluation; rerun the same package to finish remaining batches.')
        else:
            verdict['reason'] = 'No new answers completed; review before allocating another GPU.'
    except Exception as error:
        verdict['code'] = 'REFER'
        verdict['reason'] = 'Evaluation verification failed: ' + repr(error)
    return verdict


def banner(verdict):
    if verdict['code'] == 'COMPLETE':
        return '\nFINAL EVALUATION COMPLETE - 1280/1280 ANSWERS VERIFIED\nDo not rerun training or evaluation. Switch to CPU and send this output for scoring and plots.'
    if verdict['code'] == 'CONTINUE':
        return f'\nSAFE TO CONTINUE - NO ERROR DETECTED\nVerified evaluation: {verdict["verified_answers"]}/1280 answers.\nReconnect T4 and rerun this same loader/ZIP. Saved batches are reused; no training is performed.'
    return '\nSTOP - SEND THIS OUTPUT TO ME\n' + str(verdict.get('reason')) + '\nDo not rerun yet.'
