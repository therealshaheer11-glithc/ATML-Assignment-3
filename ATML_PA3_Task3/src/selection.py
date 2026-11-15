"""Task 3 prefix-only quality, selection and adaptive stopping. No gold inputs."""
from dataclasses import dataclass
import numpy as np

BUDGETS=(1,2,4,8,16,32)
TAUS=(0.5,0.625,0.75,0.875,1.0)
GAMMAS=(0.,0.1,0.25,0.5)

@dataclass(frozen=True)
class Candidate:
    rollout_id:int
    answer:object
    logp:object
    kl:object
    attention:object=None


def vector(value):
    array=np.asarray(value,dtype=np.float64)
    if array.ndim!=1 or len(array)==0 or not np.isfinite(array).all():raise ValueError('Invalid token vector')
    return array


def quality(revealed):
    """Caller must pass only the currently revealed prefix; no full-pool calibration."""
    if not revealed:raise ValueError('Empty candidate prefix')
    logs=[vector(c.logp) for c in revealed];kls=[vector(c.kl) for c in revealed]
    if any(len(a)!=len(b) for a,b in zip(logs,kls)):raise ValueError('Probability list alignment differs')
    scores={'L':np.array([x.mean() for x in logs]),'K':np.array([x.mean() for x in kls])}
    lengths=np.array([len(x) for x in logs],dtype=np.int64)
    attention=[c.attention is not None for c in revealed]
    if any(attention) and not all(attention):raise ValueError('Mixed vision/nonvision candidates')
    diagnostics={}
    if all(attention):
        lo,hi=np.percentile(np.concatenate(kls),[5.,95.],method='linear')
        calibrated=[np.clip((k-lo)/max(hi-lo,1e-8),0.,1.) for k in kls]
        vision=[]
        for c,k in zip(revealed,kls):
            a=np.asarray(c.attention,dtype=np.float64)
            if a.shape!=(3,len(k)) or not np.isfinite(a).all() or (a<0).any():raise ValueError('Invalid attention groups')
            den=a.sum(axis=0)
            if (den<=0).any():raise ValueError('Zero attention denominator')
            vision.append(a[0]/den)
        scores['V']=np.array([v.mean() for v in vision])
        scores['C']=np.array([c.mean() for c in calibrated])
        scores['Z']=scores['V']*scores['C']
        scores['Ztoken']=np.array([np.mean(v*c) for v,c in zip(vision,calibrated)])
        diagnostics={'q05':float(lo),'q95':float(hi),'token_confidence':calibrated,'token_vision':vision}
    return scores,lengths,diagnostics


def rank_weights(scores):
    values=np.asarray(scores,dtype=np.float64)
    if values.ndim!=1 or len(values)==0 or not np.isfinite(values).all():raise ValueError('Invalid quality scores')
    n=len(values)
    if n==1:return np.ones(1,dtype=np.float64)
    order=np.argsort(-values,kind='stable');ranks=np.empty(n,dtype=np.float64)
    start=0
    while start<n:
        end=start+1
        while end<n and values[order[end]]==values[order[start]]:end+=1
        ranks[order[start:end]]=(start+1+end)/2.
        start=end
    return (n-ranks)/(n-1)


def vote(assignments,weights):
    weights=np.asarray(weights,dtype=np.float64)
    if len(assignments)!=len(weights) or not len(weights):raise ValueError('Vote alignment differs')
    if not np.isfinite(weights).all() or (weights<0).any():raise ValueError('Invalid vote weights')
    support={};first={}
    for i,(group,w) in enumerate(zip(assignments,weights)):
        support[group]=support.get(group,0.)+float(w);first.setdefault(group,i)
    winner=max(support,key=lambda g:(support[g],-first[g]))
    ordered=sorted(support.values(),reverse=True)
    total=float(weights.sum());runner=ordered[1] if len(ordered)>1 else 0.
    margin=(ordered[0]-runner)/total if total>0 else 0.
    return winner,first[winner],support,margin


@dataclass(frozen=True)
class PrefixState:
    n:int
    self_consistency_group:int
    unique_nonmissing_leader:bool
    agreement:float
    rwc_group:int
    rwc_margin:float
    self_consistency_index:int


def evaluate_prefix(revealed,equivalent,group_revealed):
    """Returns selectors and observable state, never correctness labels."""
    scores,lengths,diagnostics=quality(revealed)
    groups,assignments=group_revealed([c.answer for c in revealed],equivalent)
    n=len(revealed)
    sc_group,sc_index,counts,_=vote(assignments,np.ones(n))
    selections={'FirstAns':0,'SC':sc_index,'shortest':int(np.argmin(lengths))}
    selected_cost={'FirstAns':int(lengths[0]),'SC':int(lengths.sum()),'shortest':int(lengths.sum())}
    rwcs={}
    for name in ['L','K']+(['V','Z','Ztoken'] if 'Z' in scores else []):
        selections[name+'_max']=int(np.argmax(scores[name]))
        # Common integer rescaling is exactly equivalent to the required weights;
        # it prevents floating summation from changing boundary-margin decisions.
        units=np.rint(rank_weights(scores[name])*(2*(n-1) if n>1 else 1)).astype(np.int64)
        g,index,_,margin=vote(assignments,units)
        selections[name+'_RWC']=index;rwcs[name]=(g,margin)
        selected_cost[name+'_max']=selected_cost[name+'_RWC']=int(lengths.sum())
    leader_count=counts[sc_group]
    unique=sum(value==leader_count for value in counts.values())==1
    quality_name='Z' if 'Z' in scores else 'K'
    state=PrefixState(n,sc_group,bool(unique and not revealed[sc_index].answer.missing),leader_count/n,rwcs[quality_name][0],rwcs[quality_name][1],sc_index)
    return {'selections':selections,'token_costs':selected_cost,'groups':groups,'assignments':assignments,'state':state,'scores':scores,'lengths':lengths,'diagnostics':diagnostics}


def should_stop(state,tau,gamma=None):
    if state.n==32:return True
    if state.n not in (4,8,16):return False
    if not state.unique_nonmissing_leader or state.agreement<tau:return False
    return gamma is None or (state.self_consistency_group==state.rwc_group and state.rwc_margin>=gamma)


def stop_from_states(states,tau=None,gamma=None):
    """States are yielded lazily; after an early stop future states are not accessed."""
    expected=iter((4,8,16,32))
    for state in states:
        if state.n!=next(expected,None):raise ValueError('Invalid stopping sequence')
        if state.n==32 or (tau is not None and should_stop(state,tau,gamma)):return state
    raise ValueError('Stopping stream ended before a stop/fallback')


def choose_development_policy(rows):
    """Development-only exact count accounting; no held-out labels are accepted here.

    Each row contains correct_total, token_total, observations, tau and gamma.
    All development questions have the same ten orders, so these integer totals
    give exactly the prescribed question/order means without rounding a 1pp edge.
    """
    fallback=[r for r in rows if r['tau'] is None]
    if len(fallback)!=1:raise ValueError('Exactly one always32 fallback required')
    baseline=fallback[0];n=baseline['observations']
    if not isinstance(n,int) or n<=0:raise ValueError('Invalid observation count')
    for r in rows:
        if r['observations']!=n or not isinstance(r['correct_total'],int) or not 0<=r['correct_total']<=n or not isinstance(r['token_total'],int) or r['token_total']<0:raise ValueError('Invalid policy totals')
    feasible=[r for r in rows if (baseline['correct_total']-r['correct_total'])*100<=n]
    return min(feasible,key=lambda r:(r['token_total'],-r['correct_total'],float('inf') if r['tau'] is None else r['tau'],float('inf') if r.get('gamma') is None else r['gamma']))
