"""CPU-only final subset scores, complete-pair bootstrap and training summaries."""
from collections import Counter,defaultdict
import csv,io,json,math
import numpy as np
from .scoring import exact_match,vqa_soft_score
from .final_evaluation import CONDITIONS
from .protocol import DEV_UPDATES,STAGES,learning_rate

METRICS=('accuracy','canonical_accuracy','pair_accuracy','identical_rate','empty_rate',
    'truncation_rate','reference_NLL','generation_seconds_per_answer')


def rate_values(row):
    rates=row['learning_rates']
    if isinstance(rates,list) and len(rates)==2:return dict(zip(('connector','lora'),rates))
    if isinstance(rates,dict):return rates
    raise ValueError('Unknown recorded optimizer learning-rate format.')


def interval(values,seed=6304,samples=4000):
    values=np.asarray(values,dtype=np.float64)
    if values.ndim!=1 or len(values)==0 or not np.isfinite(values).all():
        raise ValueError('Bootstrap requires finite complete-pair values.')
    indices=np.random.default_rng(seed).integers(len(values),size=(samples,len(values)))
    draws=values[indices].mean(axis=1)
    lo,hi=np.percentile(draws,[2.5,97.5])
    return {'value':float(values.mean()),'ci95_low':float(lo),'ci95_high':float(hi),
        'pairs':len(values),'bootstrap_samples':samples,'bootstrap_seed':seed}


def analyze_predictions(rows,normalize):
    expected={('grounding',c):128 for c in CONDITIONS}|{('natural',c):192 for c in CONDITIONS}
    counts=Counter((r['suite'],r['condition']) for r in rows)
    if counts!=expected or len(rows)!=1280:
        raise ValueError('Final predictions must cover all 1280 original suite/condition examples.')
    identities=[(r['suite'],r['condition'],r['pair_id'],r['side']) for r in rows]
    if len(identities)!=len(set(identities)):
        raise ValueError('Repeated final prediction identity.')
    scored=[];groups=defaultdict(list)
    for row in rows:
        text=normalize(row['prediction']);canonical=normalize(row['canonical'])
        if canonical!=row['canonical']:
            raise ValueError('Frozen canonical answer changed under the locked normalizer.')
        em=exact_match(row['prediction'],row['canonical'],normalize)
        natural=row['suite']=='natural'
        original=vqa_soft_score(row['prediction'],row['human_answers'],normalize) if natural else em
        partner_em=exact_match(row['prediction'],row['partner_canonical'],normalize)
        partner_score=vqa_soft_score(row['prediction'],row['partner_human_answers'],normalize) if natural else partner_em
        record={**row,'normalized_prediction':text,'original_reference_score':original,
            'canonical_correct':em,'empty_answer':not text,'partner_canonical_match':partner_em,
            'partner_reference_score':partner_score,
            'neither_reference_supported':original==0 and partner_score==0}
        scored.append(record);groups[(row['suite'],row['condition'],row['pair_id'])].append(record)
    pairs=[]
    for (suite,condition,pid),members in sorted(groups.items()):
        members=sorted(members,key=lambda r:r['side'])
        if [r['side'] for r in members]!=[0,1] or members[0]['question_type']!=members[1]['question_type']:
            raise ValueError('Pair scoring requires both original sides of the same type.')
        pair={'suite':suite,'condition':condition,'pair_id':pid,'question_type':members[0]['question_type'],
            'accuracy':sum(r['original_reference_score'] for r in members)/2,
            'canonical_accuracy':sum(r['canonical_correct'] for r in members)/2,
            'pair_accuracy':float(all(r['canonical_correct'] for r in members)),
            'identical_rate':float(members[0]['normalized_prediction']==members[1]['normalized_prediction']),
            'empty_rate':sum(r['empty_answer'] for r in members)/2,
            'truncation_rate':sum(r['truncated'] for r in members)/2,
            'reference_NLL':sum(r['reference_NLL'] for r in members)/2,
            'generation_seconds_per_answer':sum(r['generation_seconds_per_answer'] for r in members)/2,
            'partner_canonical_rate':sum(r['partner_canonical_match'] for r in members)/2,
            'neither_canonical_rate':sum(not r['canonical_correct'] and not r['partner_canonical_match'] for r in members)/2,
            'neither_reference_supported_rate':sum(r['neither_reference_supported'] for r in members)/2,
            'partner_reference_score':sum(r['partner_reference_score'] for r in members)/2}
        pairs.append(pair)
    aggregates=[];advantages=[];following=[]
    for suite in ('grounding','natural'):
        suite_pairs=[p for p in pairs if p['suite']==suite]
        types=['all']+sorted({p['question_type'] for p in suite_pairs})
        for kind in types:
            selected=[p for p in suite_pairs if kind=='all' or p['question_type']==kind]
            by_condition={c:{p['pair_id']:p for p in selected if p['condition']==c} for c in CONDITIONS}
            ids=sorted(by_condition['correct'])
            if not ids or any(sorted(g)!=ids for g in by_condition.values()):
                raise ValueError('Paired conditions have different original pair IDs.')
            for condition in CONDITIONS:
                records=[by_condition[condition][pid] for pid in ids]
                for metric in METRICS:
                    aggregates.append({'suite':suite,'question_type':kind,'condition':condition,'metric':metric,
                        'questions':2*len(ids),**interval([r[metric] for r in records])})
            for other in CONDITIONS[1:]:
                for metric in ('accuracy','canonical_accuracy','pair_accuracy','reference_NLL'):
                    # Positive NLL advantage means the control has greater (worse) NLL.
                    sign=-1 if metric=='reference_NLL' else 1
                    diff=[sign*(by_condition['correct'][pid][metric]-by_condition[other][pid][metric]) for pid in ids]
                    advantages.append({'suite':suite,'question_type':kind,'comparison':'correct_vs_'+other,'metric':metric,
                        'definition':other+' minus correct' if sign==-1 else 'correct minus '+other,
                        'questions':2*len(ids),**interval(diff)})
            for metric in ('canonical_accuracy','partner_canonical_rate','neither_canonical_rate',
                           'partner_reference_score','neither_reference_supported_rate'):
                following.append({'suite':suite,'question_type':kind,'condition':'complementary','metric':metric,
                    'questions':2*len(ids),**interval([by_condition['complementary'][pid][metric] for pid in ids])})
    return {'schema':'atml-pa3-task2-final-results-v1','questions_per_condition':320,'answers':1280,
        'scored_rows':scored,'pair_rows':pairs,'aggregates':aggregates,'advantages':advantages,
        'partner_following':following,'bootstrap':{'unit':'complete original pair','samples':4000,'seed':6304,
            'interval':'pointwise percentile2.5/97.5; same sampled pair IDs within each subgroup across conditions',
            'multiplicity_adjustment':False},
        'metric_notes':{'grounding_accuracy':'Normalized exact match against original targets',
            'natural_accuracy':'Local VQAv2 ten leave-one-out soft consensus against all original human references; subset, not full-benchmark score',
            'pair_accuracy':'Both normalized predictions match the two original canonical answers',
            'empty_answers':'Remain in every accuracy denominator; identical empty pair is also an identical-answer pair',
            'reference_NLL':'Original content-only token mean within each example, equal example mean; EOT excluded',
            'timing':'Synchronized batched generation seconds divided by eight; throughput per answer, not isolated latency; intervals do not model shared-batch timing dependence',
            'partner_following':'Separately labeled complementary-condition diagnostic; never replaces primary original-reference scoring',
            'neither_reference_supported':'No support in either saved answer-reference set; does not establish visual unsupportedness',
            'confidence_intervals':'Pair-resampling pointwise intervals; dataset construction and shared categories can induce dependence beyond pairs'}}


def csv_bytes(rows):
    if not rows:return b''
    columns=list(dict.fromkeys(key for row in rows for key in row))
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
    for row in rows:
        writer.writerow({key:json.dumps(value,ensure_ascii=False) if isinstance(value,(dict,list,tuple)) else value for key,value in row.items()})
    return stream.getvalue().encode('utf-8')


def analyze_training(rows,dev):
    if len(rows)!=4500 or [r['global_update'] for r in rows]!=list(range(1,4501)):
        raise ValueError('Training history must contain each successful update1..4500 once.')
    if sorted(map(int,dev))!=list(DEV_UPDATES):
        raise ValueError('Required development history is incomplete.')
    stages=[]
    for stage in (1,2,3):
        selected=[r for r in rows if r['stage']==stage]
        if len(selected)!=STAGES[stage]['updates'] or [r['stage_update'] for r in selected]!=list(range(1,len(selected)+1)):
            raise ValueError('Training stage history differs from approved budgets.')
        mixture=Counter()
        for r in selected:
            if len(r['examples'])!=16 or len(r['pair_kinds'])!=8 or not math.isfinite(r['loss']):
                raise ValueError('Training history contains an invalid successful update.')
            if not math.isfinite(r['wall_seconds']) or r['wall_seconds']<=0:
                raise ValueError('Invalid measured training update time.')
            if len(r['micro_losses'])!=2 or not all(math.isfinite(v) for v in r['micro_losses']):
                raise ValueError('Invalid two-microbatch training losses.')
            if not math.isclose(r['loss'],sum(r['micro_losses'])/2,rel_tol=1e-6):
                raise ValueError('Training equal-example loss differs from two effective microbatches.')
            if not set(r['pair_kinds'])<={kind for kind,weight in STAGES[stage]['mixture']}:
                raise ValueError('Training mixture includes an unapproved component.')
            for kind in r['pair_kinds']:mixture[kind]+=2
            expected={'connector':learning_rate(STAGES[stage]['connector_lr'],stage,r['stage_update'])}
            if stage>=2:expected['lora']=learning_rate(STAGES[stage]['lora_lr'],stage,r['stage_update'])
            for name,value in expected.items():
                if not math.isclose(rate_values(r)[name],value,rel_tol=1e-9,abs_tol=1e-15):
                    raise ValueError('Recorded learning rate differs from the approved schedule.')
        stages.append({'stage':stage,'updates':len(selected),'examples':16*len(selected),
            'training_update_wall_seconds':sum(r['wall_seconds'] for r in selected),
            'mean_loss':sum(r['loss'] for r in selected)/len(selected),'realized_example_counts':dict(mixture)})
    for step,metrics in dev.items():
        if metrics['global_update']!=int(step) or metrics['generation_performed'] or not metrics['training_state_and_RNG_restored_exactly']:
            raise ValueError('Development history violates the monitoring contract.')
        if not math.isfinite(metrics['caption']['correct_image_CE']):raise ValueError('Invalid caption development loss.')
        for suite in metrics['suites'].values():
            if not math.isfinite(suite['swap_penalty_NLL']):raise ValueError('Invalid development swap penalty.')
            for condition in suite['conditions'].values():
                values=condition['per_example']
                if not values or not all(math.isfinite(r['reference_NLL']) for r in values):raise ValueError('Invalid development reference NLL.')
                if not math.isclose(condition['mean_reference_NLL'],sum(r['reference_NLL'] for r in values)/len(values),rel_tol=1e-6):
                    raise ValueError('Development aggregate differs from original per-example losses.')
    return stages
