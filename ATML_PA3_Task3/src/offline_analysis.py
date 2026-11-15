"""Saved-result analysis; no model calls, rescoring, grouping changes or policy selection."""
import csv,hashlib,json
from pathlib import Path
import numpy as np

def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def jwrite(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def interval(values,weights):
    values=np.asarray(values,dtype=np.float64);finite=np.isfinite(values)
    if not finite.any():return {'mean':None,'low':None,'high':None,'valid_questions':0,'undefined_questions':len(values)}
    numerator=weights[:,finite]@values[finite];den=weights[:,finite].sum(axis=1);draws=numerator[den>0]/den[den>0]
    low,high=np.percentile(draws,[2.5,97.5],method='linear')
    return {'mean':float(values[finite].mean()),'low':float(low),'high':float(high),'valid_questions':int(finite.sum()),'undefined_questions':int((~finite).sum())}
def bootstrap_weights(n,index):
    rng=np.random.default_rng(np.random.SeedSequence([6304,3,index]));ids=rng.integers(0,n,size=(1000,n),dtype=np.int64)
    counts=np.zeros((1000,n),dtype=np.float64);np.add.at(counts,(np.arange(1000)[:,None],ids),1)
    return ids.astype(np.int32),counts

def analyze(data,out):
    assert np.__version__=='2.4.6';out.mkdir(parents=True,exist_ok=True)
    original=json.loads((data/'results/task3_heldout_v1/heldout_summary.json').read_text());all_metrics=[];differences=[];associations=[];vision_rows=[];by_dataset={};trajectory_rows=[]
    for dindex,(dataset,label) in enumerate([('MATH','math'),('MathVista','mathvista')]):
        records=[json.loads(p.read_text())['record'] for p in sorted((data/f'results/task3_heldout_v1/{label}').glob('*.json'))]
        records.sort(key=lambda r:r['sample_id']);assert len(records)==800 and len({r['sample_id'] for r in records})==800
        keys=sorted({(r['N'],r['method']) for r in records[0]['selections']})+[(0,'adaptive_agreement'),(0,'adaptive_quality')]
        columns={k:i for i,k in enumerate(keys)};accuracy=np.zeros((800,len(keys)));cost=np.zeros_like(accuracy);oracle=np.zeros_like(accuracy);pool=np.zeros((800,6));diversity=np.zeros_like(pool)
        for i,record in enumerate(records):
            counts=np.zeros(len(keys),dtype=int)
            for row in record['selections']:
                k=columns[(row['N'],row['method'])];accuracy[i,k]+=row['correct']/10;cost[i,k]+=row['token_cost']/10;oracle[i,k]+=row['consumed_oracle']/10;counts[k]+=1
            for row in record['adaptive_totals']:
                k=columns[(0,'adaptive_'+row['family'])];assert row['observations']==10
                accuracy[i,k]=row['correct_total']/10;cost[i,k]=row['token_total']/10;counts[k]=10
            assert (counts==10).all()
            for row in record['prefixes']:
                k=[1,2,4,8,16,32].index(row['N']);pool[i,k]+=row['pool_oracle']/10;diversity[i,k]+=row['distinct_groups']/10
            for raw_id,trajectory in record['fixed_trajectories'].items():trajectory_rows.append({'dataset':dataset,'sample_id':record['sample_id'],'rollout_id':int(raw_id),**trajectory})
        ids,weights=bootstrap_weights(800,dindex);np.save(out/(label+'_bootstrap_question_indices.npy'),ids)
        np.savez_compressed(out/(label+'_question_metrics.npz'),accuracy=accuracy,cost=cost,consumed_oracle=oracle,pool_oracle=pool,distinct_groups=diversity,question_ids=np.array([r['sample_id'] for r in records]),keys_json=np.array(json.dumps(keys)))
        # Sanity checks target the paired question bootstrap, independently of model results.
        assert (weights.sum(axis=1)==800).all()
        assert interval(np.zeros(800),weights)['high']==0
        assert np.isclose(interval(np.full(800,.25),weights)['low'],.25)
        metrics=[]
        for k,col in columns.items():
            n,method=k;stats=interval(accuracy[:,col],weights)
            cost_stats=interval(cost[:,col],weights)
            row={'dataset':dataset,'N':n,'method':method,'accuracy_percent':stats['mean']*100,'accuracy_low_percent':stats['low']*100,'accuracy_high_percent':stats['high']*100,'mean_token_cost':float(cost[:,col].mean()),'token_cost_low':cost_stats['low'],'token_cost_high':cost_stats['high'],'questions':800,'orders_per_question':10,'consumed_oracle_percent':None if n==0 else float(oracle[:,col].mean()*100)}
            metrics.append(row);all_metrics.append(row)
            reference=original[dataset]['adaptive_results'][method.removeprefix('adaptive_')] if n==0 else next(r for r in original[dataset]['fixed_budget_results'] if (r['N'],r['method'])==k)
            assert np.isclose(stats['mean'],reference['accuracy'],atol=1e-12) and np.isclose(row['mean_token_cost'],reference['mean_token_cost'],atol=1e-8)
            if n:assert (accuracy[:,col]<=oracle[:,col]+1e-12).all()
        # N1 selectors and FirstAns at every displayed budget must share the same outcome/cost.
        first=columns[(1,'FirstAns')]
        for k,col in columns.items():
            if k[0]==1 or k[1]=='FirstAns':assert np.allclose(accuracy[:,col],accuracy[:,first]) and np.allclose(cost[:,col],cost[:,first])
        pairs=[]
        for n in [1,2,4,8,16,32]:
            for k in keys:
                if k[0]==n and k[1]!='SC':pairs.append((k,(n,'SC')))
            for s in ['L','K']+(['V','Z','Ztoken'] if dataset=='MathVista' else []):pairs.append(((n,s+'_RWC'),(n,s+'_max')))
            if dataset=='MathVista':
                for a in ['max','RWC']:
                    for s in ['V','K','Ztoken']:pairs.append(((n,'Z_'+a),(n,s+'_'+a)))
        pairs += [((0,'adaptive_'+f),(32,'SC')) for f in ['agreement','quality']]+[((0,'adaptive_quality'),(0,'adaptive_agreement'))]
        # Compare each adaptive policy to both observed SC budgets bracketing its cost.
        fixed_SC=[(n,float(cost[:,columns[(n,'SC')]].mean())) for n in [1,2,4,8,16,32]]
        for family in ['agreement','quality']:
            adaptive_key=(0,'adaptive_'+family);mean_cost=float(cost[:,columns[adaptive_key]].mean())
            lower=max((x for x in fixed_SC if x[1]<=mean_cost),key=lambda x:x[0],default=None)
            upper=min((x for x in fixed_SC if x[1]>=mean_cost),key=lambda x:x[0],default=None)
            pairs.extend((adaptive_key,(x[0],'SC')) for x in [lower,upper] if x is not None)
        for a,b in dict.fromkeys(pairs):
            stats=interval(accuracy[:,columns[a]]-accuracy[:,columns[b]],weights)
            differences.append({'dataset':dataset,'A_N':a[0],'A_method':a[1],'B_N':b[0],'B_method':b[1],'difference_pp':stats['mean']*100,'low_pp':stats['low']*100,'high_pp':stats['high']*100,'questions':800,'bootstrap_draws':1000,'interval':'pointwise paired question percentile95%'})
        names=records[0]['signals_N32']['associations']
        signal_summaries=[]
        for name in names:
            for field in ['spearman_correctness','spearman_length','top_quartile_accuracy','bottom_quartile_accuracy','quartile_gap']:
                values=[r['signals_N32']['associations'][name][field] for r in records];stats=interval([np.nan if x is None else x for x in values],weights)
                row={'dataset':dataset,'score':name,'statistic':field,**stats};associations.append(row);signal_summaries.append(row)
        vision=[]
        if dataset=='MathVista':
            for field in ['calibrated_pearson','raw_pearson']:
                means=[];valid_rollouts=0
                for r in records:
                    vals=[x[field] for x in r['signals_N32']['confidence_vision'] if x[field] is not None];valid_rollouts+=len(vals);means.append(float(np.mean(vals)) if vals else np.nan)
                row={'dataset':dataset,'statistic':'mean_within_rollout_'+field,**interval(means,weights),'valid_rollouts':valid_rollouts,'total_rollouts':25600};vision.append(row);vision_rows.append(row)
            for field in ['K','C']:
                values=[r['signals_N32']['rollout_confidence_vision'][field] for r in records]
                row={'dataset':dataset,'statistic':'within_question_'+field+'_versus_V',**interval([np.nan if x is None else x for x in values],weights),'valid_rollouts':None,'total_rollouts':None};vision.append(row);vision_rows.append(row)
        adaptive_comparisons=[]
        for family in ['agreement','quality']:
            col=columns[(0,'adaptive_'+family)];ref=columns[(32,'SC')]
            draws=100*(1-(weights@cost[:,col])/(weights@cost[:,ref]))
            ci=np.percentile(draws,[2.5,97.5],method='linear')
            mean_cost=float(cost[:,col].mean())
            fixed=[r for r in metrics if r['method']=='SC']
            lower=max((r for r in fixed if r['mean_token_cost']<=mean_cost),key=lambda r:r['N'],default=None)
            upper=min((r for r in fixed if r['mean_token_cost']>=mean_cost),key=lambda r:r['N'],default=None)
            adaptive_comparisons.append({'family':family,'token_reduction_percent':float(100*(1-mean_cost/cost[:,ref].mean())),'token_reduction_low':float(ci[0]),'token_reduction_high':float(ci[1]),'lower_observed_fixed_SC':lower,'upper_observed_fixed_SC':upper,'interpolation_used':False})
        by_dataset[dataset]={'metrics':metrics,'oracle_percent':(pool.mean(axis=0)*100).tolist(),'mean_distinct_groups':diversity.mean(axis=0).tolist(),'signal_summaries':signal_summaries,'vision_summaries':vision,'adaptive':original[dataset]['adaptive_results'],'adaptive_cost_comparisons':adaptive_comparisons}
    write_csv(out/'method_metrics.csv',all_metrics);write_csv(out/'paired_accuracy_differences.csv',differences);write_csv(out/'signal_associations.csv',associations);write_csv(out/'confidence_vision.csv',vision_rows)
    assert len(trajectory_rows)==8;jwrite(out/'fixed_trajectories.json',trajectory_rows)
    jwrite(out/'analysis.json',{'datasets':by_dataset,'paired_differences':differences,'bootstrap':{'draws':1000,'unit':'question','orders_averaged_first':10,'seed':6304,'stream':'SeedSequence([6304,3,dataset_index])','indices_saved':True,'intervals':'pointwise percentile2.5/97.5 using linear interpolation; paired indices reused across methods'},'source_export_sha256':'de5798c0a62fc10682b5ce87f10880dd58c473bc94ca76413c6d21ecc0b3ee86','scoring_rerun':False,'policies_changed':False,'live_experiment_pending':True})
    return by_dataset

if __name__=='__main__':
    import sys
    analyze(Path(sys.argv[1]),Path(sys.argv[2]))
