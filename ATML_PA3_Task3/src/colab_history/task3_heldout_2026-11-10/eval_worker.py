"""Bounded resumable CPU held-out worker; never evaluates held-out candidates."""
import argparse,hashlib,importlib.util,json,sys,time
from pathlib import Path

def digest(data):return hashlib.sha256(data).hexdigest()
def read_question(path,recipe_sha,q,guard):
    envelope=json.loads(path.read_text());record=envelope['record']
    if digest(guard.encode_json(record))!=envelope['record_sha256'] or record['recipe_sha256']!=recipe_sha or record['sample_id']!=q:raise ValueError('Corrupt or mismatched completed question: '+q)
    return record

def execute(root,bundle,soft_seconds=1500,expected_numpy='2.4.6',expected_threshold_sha='1ae6c94a10fba0d3ee2e4eab192e7f131244ca8c3bb1f7afc4e3c97427e7ce20'):
    from development import load_candidates,filename
    from evaluation import analyze_fixed_question
    from scoring import MathScorer,MathVistaNormalizer
    from scorer_worker import environment_fingerprint
    import numpy as np
    import pyarrow as pa
    if np.__version__!=expected_numpy or pa.__version__!='23.0.1':raise ValueError('CPU numerical environment differs')
    spec=importlib.util.spec_from_file_location('guard',bundle/'block_01.py');guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
    frozen=json.loads((root/'configs/task3_scorer_frozen.json').read_text())
    if environment_fingerprint()!=frozen['installed_source_fingerprints']:raise ValueError('Scorer dependency fingerprints differ')
    work=root/'results/task3_heldout_v1';recipe_bytes=(work/'recipe.json').read_bytes();recipe_sha=digest(recipe_bytes)
    started=time.monotonic();scorer=MathScorer();completed=0
    policy_bytes=(root/'configs/task3_adaptive_selected.json').read_bytes()
    if digest(policy_bytes)!=expected_threshold_sha:raise ValueError('Frozen thresholds changed')
    selected=json.loads(policy_bytes)['selected']
    trajectory_plan=json.loads((root/'configs/task3_fixed_trajectories.json').read_text())
    for dataset,folder,label in [('MATH','qwen25-math','math'),('MathVista','qwen25vl-mathvista','mathvista')]:
        ids=json.loads((root/f'manifests/task3_{label}_splits.json').read_text())['evaluation_ids']
        orders=json.loads((root/f'manifests/task3_{label}_orders.json').read_text())
        refs=json.loads((root/f'metadata/task3_{label}_reference_answers.json').read_text())
        meta=json.loads((root/f'metadata/task3_{label}_questions.json').read_text())
        target=work/label;target.mkdir(exist_ok=True)
        remaining=[]
        for q in ids:
            path=target/filename(q)
            if path.exists():
                record=read_question(path,recipe_sha,q,guard)
                if record['orders']!=orders[q] or record['dataset']!=dataset:raise ValueError('Saved question ordering differs')
                completed+=1
            else:remaining.append(q)
        if not remaining:continue
        print(f'{dataset}: {len(ids)-len(remaining)}/800 verified; loading only {len(remaining)} remaining held-out pools.',flush=True)
        pools,raw=load_candidates(root,folder,remaining,meta,scorer)
        for q in remaining:
            if time.monotonic()-started>=soft_seconds:return {'status':'PARTIAL','completed':completed}
            norm=scorer if dataset=='MATH' else MathVistaNormalizer(meta[q])
            trajectory_ids=[x['rollout_id'] for x in trajectory_plan if x['dataset']==label and x['sample_id']==q]
            record=analyze_fixed_question(dataset,q,pools[q],orders[q],refs[q]['official_answer'],norm,recipe_sha,selected[dataset],trajectory_ids)
            record['generation_flags']={i:{'capped':x['capped']} for i,x in raw[q].items()}
            envelope={'record':record,'record_sha256':digest(guard.encode_json(record))}
            guard.save_immutable(target/filename(q),guard.encode_json(envelope))
            read_question(target/filename(q),recipe_sha,q,guard);completed+=1
            print(f'{dataset} {q}: saved and verified; total {completed}/1600; elapsed {time.monotonic()-started:.1f}s',flush=True)
    return {'status':'COMPLETE','completed':completed}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--env',required=True);p.add_argument('--root',required=True);p.add_argument('--bundle',required=True);args=p.parse_args()
    sys.path.insert(0,args.bundle);sys.path.insert(0,args.env)
    print(json.dumps(execute(Path(args.root),Path(args.bundle))),flush=True)
