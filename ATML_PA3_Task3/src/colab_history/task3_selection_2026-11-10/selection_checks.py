"""Analytic and adversarial checks, with no supplied candidate records."""
from dataclasses import fields
import numpy as np
from scoring import Answer,group_revealed
from selection import Candidate,PrefixState,quality,rank_weights,vote,evaluate_prefix,should_stop,stop_from_states,choose_development_policy

def run_checks():
    names=[]
    def check(name,condition):
        if not condition:raise AssertionError(name)
        names.append(name)
    def raises(fn):
        try:fn()
        except ValueError:return True
        return False
    eq=lambda a,b:(a.missing and b.missing) or (not a.missing and not b.missing and a.value==b.value)
    def candidate(i,answer,kl,vision=None,logp=None):
        return Candidate(i,Answer(answer,'missing' if answer is None else 'mathvista',answer),[-1.]*len(kl) if logp is None else logp,kl,vision)
    c=[candidate(0,'A',[0.,10.],[[1.,0.],[0.,1.],[0.,0.]]),candidate(1,'B',[20.,30.],[[1.,1.],[1.,1.],[0.,0.]])]
    scores,lengths,d=quality(c)
    check('Token pooled linear percentiles',np.isclose(d['q05'],1.5) and np.isclose(d['q95'],28.5))
    check('Raw mean confidence and sampled logp',np.array_equal(scores['K'],[5.,25.]) and np.array_equal(scores['L'],[-1.,-1.]))
    check('Visual attention is normalized by eligible mass',np.array_equal(scores['V'],[.5,.5]))
    expected=[np.clip((np.array(x)-1.5)/27.,0,1) for x in [[0.,10.],[20.,30.]]]
    check('Calibrated means and trajectory product',np.allclose(scores['C'],[x.mean() for x in expected]) and np.allclose(scores['Z'],scores['C']*.5))
    for i in range(2):
        cov=np.mean((d['token_confidence'][i]-scores['C'][i])*(d['token_vision'][i]-scores['V'][i]))
        check('Token fusion population covariance '+str(i),np.isclose(scores['Ztoken'][i]-scores['Z'][i],cov))
    before=quality(c[:1])[0]
    c[1]=candidate(1,'B',[1e8,1e9],[[1.,1.],[0.,0.],[0.,0.]])
    check('Future candidates cannot change prefix calibration',all(np.array_equal(before[k],quality(c[:1])[0][k]) for k in before))
    constant=quality([candidate(0,'A',[2.,2.],[[1.,1.],[0.,0.],[0.,0.]])])[0]
    check('Zero percentile spread uses epsilon and clipping',constant['C'][0]==0 and constant['Z'][0]==0)
    check('Rank average ties',np.allclose(rank_weights([9,9,3,0]),[5/6,5/6,1/3,0]))
    check('All score ties yield equal half votes',np.array_equal(rank_weights([1,1,1,1]),[.5]*4))
    check('N1 score weight special case',rank_weights([1]).tolist()==[1])
    check('Vote ties use earliest support',vote([7,8,8,7],[1,1,1,1])[:2]==(7,0))
    check('One group margin has zero runner-up',vote([0,0,0],[1,1,1])[3]==1)
    missing=[candidate(i,None,[1.]) for i in range(4)]
    r=evaluate_prefix(missing,eq,group_revealed)
    check('Missing group stays in voting and cannot stop early',r['selections']['SC']==0 and not should_stop(r['state'],.5))
    for n in [1,2,4,8,16,32]:
        pool=[candidate(i,'A' if i%2==0 else 'B',list(range(i+1))) for i in range(n)]
        out=evaluate_prefix(pool,eq,group_revealed)
        check('FirstAns cost only first candidate '+str(n),out['token_costs']['FirstAns']==1)
        check('All competing selectors consume full prefix '+str(n),all(v==n*(n+1)//2 for k,v in out['token_costs'].items() if k!='FirstAns'))
        if n==1:check('Every N1 method returns sole candidate',set(out['selections'].values())=={0})
        check('Shortest control selects first shortest '+str(n),out['selections']['shortest']==0)
    tied=[candidate(i,'A' if i<3 else 'B',[1.]) for i in range(4)]
    integration=evaluate_prefix(tied,eq,group_revealed)['state']
    check('Integrated quality-aware margin and agreement',integration.agreement==.75 and integration.rwc_margin==.5 and should_stop(integration,.75,.5))
    mixed=[candidate(i,'A' if i<3 else 'B',[float(i)]) for i in range(4)]
    boundary=evaluate_prefix(mixed,eq,group_revealed)['state']
    check('Integrated zero-margin tie permits gamma0 only',boundary.rwc_margin==0 and should_stop(boundary,.75,0) and not should_stop(boundary,.75,.1))
    state=PrefixState(4,0,True,.75,0,.25,0)
    check('Agreement exact threshold',should_stop(state,.75) and not should_stop(state,.875))
    check('Quality margin exact threshold',should_stop(state,.75,.25) and not should_stop(state,.75,.5))
    check('Quality rule requires same answer',not should_stop(PrefixState(4,0,True,.75,1,1.,0),.5,0))
    check('Tied leader cannot stop',not should_stop(PrefixState(4,0,False,.5,0,1.,0),.5))
    check('Prefix state has no correctness or future fields',set(x.name for x in fields(PrefixState))=={'n','self_consistency_group','unique_nonmissing_leader','agreement','rwc_group','rwc_margin','self_consistency_index'})
    def stream():
        yield state
        raise AssertionError('Future state accessed after stop')
    check('Lazy early stop never reads future states',stop_from_states(stream(),.75).n==4)
    states=[PrefixState(n,0,False,.5,1,0.,0) for n in [4,8,16,32]]
    check('Always32 fallback returns current SC at32',stop_from_states(iter(states)).n==32)
    check('Nonqualifying policy reaches32',stop_from_states(iter(states),1.,.5).n==32)
    check('Missing stopping checkpoint rejected',raises(lambda:stop_from_states(iter(states[1:]),1.)))
    baseline={'tau':None,'gamma':None,'correct_total':1600,'token_total':200000,'observations':2000}
    edge={'tau':.75,'gamma':.1,'correct_total':1580,'token_total':100000,'observations':2000}
    bad={'tau':.5,'gamma':0.,'correct_total':1579,'token_total':1,'observations':2000}
    check('One percentage point boundary exact in counts',choose_development_policy([baseline,edge,bad])==edge)
    better=dict(edge,correct_total=1581,tau=.875)
    check('Equal cost breaks by accuracy',choose_development_policy([baseline,edge,better])==better)
    smaller=dict(edge,tau=.625)
    check('Equal cost/accuracy breaks by smaller tau',choose_development_policy([baseline,edge,smaller])==smaller)
    gamma=dict(edge,gamma=0.)
    check('Equal tau breaks by smaller gamma',choose_development_policy([baseline,edge,gamma])==gamma)
    check('Missing fallback rejected',raises(lambda:choose_development_policy([edge])))
    check('Nonfinite probability rejected',raises(lambda:quality([candidate(0,'A',[float('nan')])])))
    check('Zero attention denominator rejected',raises(lambda:quality([candidate(0,'A',[1.],[[0.],[0.],[0.]]) ])))
    check('Misaligned token arrays rejected',raises(lambda:quality([candidate(0,'A',[1.,2.],logp=[-1.])])))
    return {'checks_passed':len(names),'check_names':names,'supplied_candidate_records_read':False,'gold_answers_read':False,'development_thresholds_selected':False}


if __name__=='__main__':
    import sys,json
    from pathlib import Path
    result=run_checks();result['numpy_version']=np.__version__
    Path(sys.argv[1]).write_text(json.dumps(result,indent=2)+'\n')
    print('PREFIX SELECTION AND STOPPING CHECKS PASSED',flush=True)
