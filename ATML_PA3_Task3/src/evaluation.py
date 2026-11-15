"""Held-out evaluation and outcome-independent signal diagnostics; no policy search."""
import numpy as np
from development import require
from selection import BUDGETS,evaluate_prefix,stop_from_states
from scoring import group_revealed
from dataclasses import asdict

def correlation(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    if len(a)<2 or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Invalid correlation inputs')
    a=a-a.mean();b=b-b.mean();den=np.linalg.norm(a)*np.linalg.norm(b)
    return None if den==0 else float(np.dot(a,b)/den)

def ranks(values):
    a=np.asarray(values);out=np.empty(len(a),dtype=np.float64);order=np.argsort(a,kind='stable');i=0
    while i<len(a):
        j=i+1
        while j<len(a) and a[order[j]]==a[order[i]]:j+=1
        out[order[i:j]]=(i+j-1)/2.;i=j
    return out

def quartile_accuracy(values,correct,top):
    a=np.asarray(values);y=np.asarray(correct,dtype=np.float64);order=np.argsort(-a if top else a,kind='stable');budget=len(a)/4.;total=0.;i=0
    while budget>0 and i<len(a):
        j=i+1
        while j<len(a) and a[order[j]]==a[order[i]]:j+=1
        take=min(budget,j-i);total+=take*float(y[order[i:j]].mean());budget-=take;i=j
    return total/(len(a)/4.)

def analyze_fixed_question(dataset,q,candidates,orders,gold,normalizer,recipe_sha,selected,trajectory_ids=()):
    require(len(candidates)==32 and set(candidates)==set(range(32)) and len(orders)==10 and all(sorted(o)==list(range(32)) for o in orders),'Invalid held-out pool/orders')
    require(set(selected)=={'agreement','quality'},'Exactly two previously selected policies required')
    correct={i:bool(normalizer.grade(candidates[i].answer.raw,gold)) for i in range(32)}
    rows=[];prefixes=[];states_saved=[]
    totals={f:{'family':f,'tau':s['tau'],'gamma':s['gamma'],'observations':0,'correct_total':0,'token_total':0,'rollouts_total':0,'stopping_counts':{str(n):0 for n in [4,8,16,32]},'incorrect_early_stops':0,'fixed32_recoverable_early_errors':0} for f,s in selected.items()}
    signal=None;trajectories={}
    for order_index,order in enumerate(orders):
        pool=[candidates[i] for i in order];results={};states=[]
        for n in BUDGETS:
            r=evaluate_prefix(pool[:n],normalizer.equivalent,group_revealed);results[n]=r
            oracle=any(correct[i] for i in order[:n]);prefixes.append({'order':order_index,'N':n,'pool_oracle':oracle,'distinct_groups':len(r['groups']),'pool_token_cost':int(r['lengths'].sum())})
            for method,index in r['selections'].items():
                i=order[index];good=correct[i];consumed=correct[order[0]] if method=='FirstAns' else oracle
                rows.append({'order':order_index,'N':n,'method':method,'selected_rollout_id':i,'selected_answer':candidates[i].answer.raw,'correct':good,'token_cost':r['token_costs'][method],'consumed_oracle':consumed,'sampling_failure':not consumed,'selection_failure':bool(consumed and not good)})
            if n in [4,8,16,32]:states.append(r['state'])
        states_saved.append([asdict(s) for s in states]);fixed_good=correct[order[results[32]['selections']['SC']]]
        for family,spec in selected.items():
            s=stop_from_states(iter(states),spec['tau'],spec['gamma']);good=correct[order[s.self_consistency_index]];r=totals[family]
            r['observations']+=1;r['correct_total']+=good;r['token_total']+=int(results[s.n]['lengths'].sum());r['rollouts_total']+=s.n;r['stopping_counts'][str(s.n)]+=1
            r['incorrect_early_stops']+=bool(s.n<32 and not good);r['fixed32_recoverable_early_errors']+=bool(s.n<32 and not good and fixed_good)
        if order_index==0:
            r=results[32];ys=[correct[i] for i in order];lengths=r['lengths'];score_values={**r['scores'],'length':lengths}
            signal={'rollout_ids':order,'correct':ys,'lengths':lengths.tolist(),'scores':{name:v.tolist() for name,v in r['scores'].items()},'associations':{},'confidence_vision':[]}
            for name,values in score_values.items():
                top=quartile_accuracy(values,ys,True);bottom=quartile_accuracy(values,ys,False)
                signal['associations'][name]={'spearman_correctness':correlation(ranks(values),ranks(ys)),'spearman_length':correlation(ranks(values),ranks(lengths)),'top_quartile_accuracy':top,'bottom_quartile_accuracy':bottom,'quartile_gap':top-bottom}
            if 'Z' in r['scores']:
                d=r['diagnostics']
                for index,i in enumerate(order):
                    c=d['token_confidence'][index];v=d['token_vision'][index];k=np.asarray(candidates[i].kl,dtype=np.float64)
                    cov=float(np.mean((c-c.mean())*(v-v.mean())))
                    require(np.isclose(r['scores']['Ztoken'][index]-r['scores']['Z'][index],cov,atol=1e-12,rtol=1e-10),'Fusion covariance identity failed')
                    signal['confidence_vision'].append({'rollout_id':i,'calibrated_pearson':correlation(c,v) if len(c)>1 else None,'raw_pearson':correlation(k,v) if len(k)>1 else None,'population_covariance':cov})
                    if i in trajectory_ids:trajectories[str(i)]={'kl':k.tolist(),'calibrated_confidence':c.tolist(),'vision':v.tolist(),'q05':d['q05'],'q95':d['q95']}
            # Confidence-versus-vision across rollouts is supplementary to token correlations.
            if 'V' in r['scores']:signal['rollout_confidence_vision']={name:correlation(r['scores'][name],r['scores']['V']) for name in ['K','C']}
    return {'schema_version':1,'recipe_sha256':recipe_sha,'dataset':dataset,'sample_id':q,'orders':orders,'candidate_correctness':correct,'candidate_answers':{i:{'raw':c.answer.raw,'kind':c.answer.kind,'reason':c.answer.reason,'retained_tokens':len(c.kl)} for i,c in candidates.items()},'prefixes':prefixes,'selections':rows,'observable_stopping_states':states_saved,'adaptive_totals':list(totals.values()),'signals_N32':signal,'fixed_trajectories':trajectories}

def aggregate_fixed(records,expected_questions=800):
    require(len(records)==expected_questions,'Incomplete held-out records');fixed={};adaptive={};oracle={}
    for record in records:
        require(len(record['prefixes'])==60 and len(record['adaptive_totals'])==2,'Incomplete held-out question')
        for row in record['selections']:
            k=(row['N'],row['method']);r=fixed.setdefault(k,{'N':row['N'],'method':row['method'],'observations':0,'correct_total':0,'token_total':0,'consumed_oracle_total':0})
            r['observations']+=1;r['correct_total']+=row['correct'];r['token_total']+=row['token_cost'];r['consumed_oracle_total']+=row['consumed_oracle']
        for row in record['prefixes']:
            r=oracle.setdefault(row['N'],{'N':row['N'],'observations':0,'oracle_total':0,'distinct_groups_total':0});r['observations']+=1;r['oracle_total']+=row['pool_oracle'];r['distinct_groups_total']+=row['distinct_groups']
        for row in record['adaptive_totals']:
            f=row['family']
            if f not in adaptive:adaptive[f]={**row,'stopping_counts':dict(row['stopping_counts'])}
            else:
                r=adaptive[f];require((r['tau'],r['gamma'])==(row['tau'],row['gamma']),'Held-out policies changed across questions')
                for name in ['observations','correct_total','token_total','rollouts_total','incorrect_early_stops','fixed32_recoverable_early_errors']:r[name]+=row[name]
                for n,c in row['stopping_counts'].items():r['stopping_counts'][n]+=c
    n=expected_questions*10
    for r in list(fixed.values())+list(adaptive.values()):require(r['observations']==n,'Wrong held-out denominator');r['accuracy']=r['correct_total']/n;r['mean_token_cost']=r['token_total']/n
    for r in adaptive.values():r['mean_rollouts']=r['rollouts_total']/n
    for r in oracle.values():require(r['observations']==n,'Wrong oracle denominator');r['oracle_coverage']=r['oracle_total']/n;r['mean_distinct_groups']=r['distinct_groups_total']/n
    return {'questions':expected_questions,'observations_per_method':n,'fixed_budget_results':list(fixed.values()),'adaptive_results':adaptive,'oracle_and_diversity':list(oracle.values()),'threshold_selection_performed':False}
