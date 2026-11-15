"""Development-only question analysis and count-exact policy aggregation."""
from dataclasses import asdict
from pathlib import Path
import hashlib,json
import numpy as np
from scoring import MathScorer,MathVistaNormalizer,group_revealed
from selection import Candidate,BUDGETS,TAUS,GAMMAS,evaluate_prefix,stop_from_states,choose_development_policy

def require(condition,message):
    if not condition:raise ValueError(message)
def policies():
    return [('agreement',t,None) for t in TAUS]+[('agreement',None,None)]+[('quality',t,g) for t in TAUS for g in GAMMAS]+[('quality',None,None)]
def policy_key(f,t,g):return json.dumps([f,t,g],separators=(',',':'))
def filename(q):return hashlib.sha256(q.encode()).hexdigest()+'.json'

def analyze_question(dataset,q,candidates,orders,gold,normalizer,recipe_sha):
    require(len(candidates)==32 and set(candidates)==set(range(32)),'Incomplete question pool')
    require(len(orders)==10 and all(sorted(o)==list(range(32)) for o in orders),'Invalid saved orders')
    # Reference grading is separate; none of these labels enter any selection call.
    correct={i:bool(normalizer.grade(candidates[i].answer.raw,gold)) for i in range(32)}
    rows=[];states_saved=[];prefix_summary=[]
    totals={policy_key(f,t,g):{'family':f,'tau':t,'gamma':g,'observations':0,'correct_total':0,'token_total':0,'rollouts_total':0,'stopping_counts':{str(n):0 for n in [4,8,16,32]},'incorrect_early_stops':0,'fixed32_recoverable_early_errors':0} for f,t,g in policies()}
    for order_index,order in enumerate(orders):
        pool=[candidates[i] for i in order];prefixes={};states=[]
        for n in BUDGETS:
            selected=evaluate_prefix(pool[:n],normalizer.equivalent,group_revealed);prefixes[n]=selected
            oracle=any(correct[i] for i in order[:n])
            prefix_summary.append({'order':order_index,'N':n,'pool_oracle':oracle,'distinct_groups':len(selected['groups']),'pool_token_cost':int(selected['lengths'].sum())})
            for method,index in selected['selections'].items():
                i=order[index];good=correct[i];consumed=correct[order[0]] if method=='FirstAns' else oracle
                rows.append({'order':order_index,'N':n,'method':method,'selected_rollout_id':i,'selected_answer':candidates[i].answer.raw,'correct':good,'token_cost':selected['token_costs'][method],'consumed_oracle':consumed,'sampling_failure':not consumed,'selection_failure':bool(consumed and not good)})
            if n in [4,8,16,32]:states.append(selected['state'])
        states_saved.append([asdict(s) for s in states])
        fixed_good=correct[order[prefixes[32]['selections']['SC']]]
        for family,tau,gamma in policies():
            stop=stop_from_states(iter(states),tau,gamma)
            good=correct[order[stop.self_consistency_index]]
            record=totals[policy_key(family,tau,gamma)]
            record['observations']+=1;record['correct_total']+=good
            record['token_total']+=int(prefixes[stop.n]['lengths'].sum());record['rollouts_total']+=stop.n
            record['stopping_counts'][str(stop.n)]+=1
            record['incorrect_early_stops']+=bool(stop.n<32 and not good)
            record['fixed32_recoverable_early_errors']+=bool(stop.n<32 and not good and fixed_good)
    return {'schema_version':1,'recipe_sha256':recipe_sha,'dataset':dataset,'sample_id':q,'orders':orders,
            'candidate_correctness':correct,'candidate_answers':{i:{'raw':c.answer.raw,'kind':c.answer.kind,'reason':c.answer.reason,'retained_tokens':len(c.kl)} for i,c in candidates.items()},
            'prefixes':prefix_summary,'selections':rows,'observable_stopping_states':states_saved,'adaptive_totals':list(totals.values())}

def aggregate(records,expected_questions=200):
    require(len(records)==expected_questions,'Incomplete development question records')
    fixed={};adaptive={}
    for record in records:
        require(len(record['prefixes'])==60 and len(record['adaptive_totals'])==27,'Incomplete question analysis')
        for row in record['selections']:
            k=(row['N'],row['method']);r=fixed.setdefault(k,{'N':row['N'],'method':row['method'],'observations':0,'correct_total':0,'token_total':0,'consumed_oracle_total':0})
            r['observations']+=1;r['correct_total']+=row['correct'];r['token_total']+=row['token_cost'];r['consumed_oracle_total']+=row['consumed_oracle']
        for row in record['adaptive_totals']:
            k=policy_key(row['family'],row['tau'],row['gamma'])
            if k not in adaptive:adaptive[k]={**row,'stopping_counts':dict(row['stopping_counts'])}
            else:
                r=adaptive[k]
                for field in ['observations','correct_total','token_total','rollouts_total','incorrect_early_stops','fixed32_recoverable_early_errors']:r[field]+=row[field]
                for n,count in row['stopping_counts'].items():r['stopping_counts'][n]+=count
    n=expected_questions*10
    require(all(r['observations']==n for r in list(fixed.values())+list(adaptive.values())),'Wrong aggregate denominators')
    chosen={}
    for family in ['agreement','quality']:
        family_rows=[r for r in adaptive.values() if r['family']==family]
        fallback=next(r for r in family_rows if r['tau'] is None)
        require(fallback['correct_total']==fixed[(32,'SC')]['correct_total'] and fallback['token_total']==fixed[(32,'SC')]['token_total'],'Fallback differs from fixed32 SC')
        chosen[family]=choose_development_policy(family_rows)
    for r in list(fixed.values())+list(adaptive.values()):
        r['accuracy']=r['correct_total']/n;r['mean_token_cost']=r['token_total']/n
    return {'questions':expected_questions,'observations_per_method':n,'fixed_budget_results':list(fixed.values()),'adaptive_grid':list(adaptive.values()),'selected_policies':chosen}

def load_candidates(root,folder,ids,metadata,scorer):
    import pyarrow.dataset as ds
    pools={q:{} for q in ids};raw={q:{} for q in ids}
    def rows(name,columns):
        source=ds.dataset(root/'sources/supplied'/folder/(name+'.parquet'),format='parquet')
        scanner=source.scanner(columns=columns,filter=ds.field('sample_id').isin(ids),batch_size=128)
        for batch in scanner.to_batches():yield from batch.to_pylist()
    for row in rows('generations',['sample_id','rollout_id','extracted_answer','n_generated_text_tokens','hit_max_new_tokens']):
        q=row['sample_id'];i=row['rollout_id'];require(q in raw and i not in raw[q],'Duplicate or off-split generation')
        raw[q][i]={'answer':(scorer if folder=='qwen25-math' else MathVistaNormalizer(metadata[q])).prepare(row['extracted_answer']),'length':row['n_generated_text_tokens'],'capped':row['hit_max_new_tokens']}
    for row in rows('probability_metrics',['sample_id','rollout_id','logp','kl_u_p']):
        q=row['sample_id'];i=row['rollout_id'];require(q in raw and i in raw[q] and i not in pools[q],'Bad probability join')
        item=raw[q][i];require(len(row['logp'])==len(row['kl_u_p'])==item['length'],'Probability alignment differs')
        pools[q][i]=Candidate(i,item['answer'],np.asarray(row['logp'],dtype=np.float64),np.asarray(row['kl_u_p'],dtype=np.float64))
    if folder=='qwen25vl-mathvista':
        seen=set()
        for row in rows('attention_metrics',['sample_id','rollout_id','img_attn_total','prompt_text_attn_total','generated_text_attn_total']):
            q=row['sample_id'];i=row['rollout_id'];require((q,i) not in seen and q in pools and i in pools[q],'Bad attention join');seen.add((q,i))
            c=pools[q][i];pools[q][i]=Candidate(i,c.answer,c.logp,c.kl,np.asarray([row['img_attn_total'],row['prompt_text_attn_total'],row['generated_text_attn_total']],dtype=np.float64))
        require(len(seen)==len(ids)*32,'Incomplete attention subset')
    require(all(set(p)==set(range(32)) for p in pools.values()),'Incomplete development pools')
    return pools,raw
