"""Retrospective summaries of frozen stopping states; no grading or tuning."""
import json
import sys
from collections import defaultdict
from dataclasses import fields
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    from .selection import PrefixState, stop_from_states
    from .offline_analysis import write_csv, jwrite
except ImportError:
    from selection import PrefixState, stop_from_states
    from offline_analysis import write_csv, jwrite

def main(data, out):
    out.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((data/'configs/task3_adaptive_selected.json').read_text())['selected']
    observations=[]
    summaries=[]
    diversity=[]
    fig, axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    fig2, ax2=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    for di,(dataset,label) in enumerate([('MATH','math'),('MathVista','mathvista')]):
        records=[json.loads(p.read_text())['record'] for p in sorted((data/f'results/task3_heldout_v1/{label}').glob('*.json'))]
        assert len(records)==800
        dataset_rows=[]
        for r in records:
            fraction=sum(r['candidate_correctness'].values())/32
            selected={(s['order'],s['N']):s for s in r['selections'] if s['method']=='SC'}
            for family,spec in frozen[dataset].items():
                totals=defaultdict(int)
                for order,states in enumerate(r['observable_stopping_states']):
                    state=stop_from_states((PrefixState(**s) for s in states),spec['tau'],spec['gamma'])
                    saved=selected[order,state.n]
                    row={'dataset':dataset,'sample_id':r['sample_id'],'order':order,'family':family,'stop_n':state.n,'agreement_at4':states[0]['agreement'],'unique_nonmissing_at4':states[0]['unique_nonmissing_leader'],'full_pool_correct_fraction_retrospective_only':fraction,'stop_correct':saved['correct'],'incorrect_early_stop':state.n<32 and not saved['correct'],'fixed32_recovered_early_error':state.n<32 and not saved['correct'] and selected[order,32]['correct'],'token_cost':saved['token_cost']}
                    dataset_rows.append(row)
                    totals['correct_total']+=row['stop_correct'];totals['token_total']+=row['token_cost'];totals['rollouts_total']+=state.n;totals['incorrect_early_stops']+=row['incorrect_early_stop'];totals['fixed32_recoverable_early_errors']+=row['fixed32_recovered_early_error']
                expected=next(t for t in r['adaptive_totals'] if t['family']==family)
                assert all(totals[k]==expected[k] for k in totals), 'Retrospective replay changed frozen outcome'
        observations.extend(dataset_rows)
        for family,color in [('agreement','#3679a2'),('quality','#b66b37')]:
            rr=[r for r in dataset_rows if r['family']==family]
            summaries.append({'dataset':dataset,'family':family,'question_order_observations':len(rr),'incorrect_early_stops':sum(r['incorrect_early_stop'] for r in rr),'fixed32_recoverable_early_errors':sum(r['fixed32_recovered_early_error'] for r in rr)})
            for col,key in enumerate(['agreement_at4','full_pool_correct_fraction_retrospective_only']):
                groups=defaultdict(list)
                for r in rr:groups[r[key]].append(r['stop_n'])
                xx=sorted(groups)
                axes[di,col].plot(xx,[np.mean(groups[v]) for v in xx],'.-',color=color,label=family)
                axes[di,col].set_ylim(3,33);axes[di,col].set_xlim(0,1.025);axes[di,col].set_ylabel(dataset+' · mean rollouts');axes[di,col].grid(alpha=.15)
        for n in [1,2,4,8,16,32]:
            pp=[p for r in records for p in r['prefixes'] if p['N']==n]
            value=float(np.mean([p['distinct_groups'] for p in pp]))
            diversity.append({'dataset':dataset,'N':n,'mean_distinct_answer_groups':value,'mean_repeated_group_candidates':n-value})
        dd=[d for d in diversity if d['dataset']==dataset]
        ax2[di].plot([d['N'] for d in dd],[d['mean_distinct_answer_groups'] for d in dd],'o-',label='Distinct answer groups',color='#3679a2')
        ax2[di].plot([d['N'] for d in dd],[d['N'] for d in dd],':',label='Every candidate distinct',color='#777777')
        ax2[di].set_xscale('log',base=2);ax2[di].set_xticks([1,2,4,8,16,32],[1,2,4,8,16,32]);ax2[di].set_xlabel('Revealed candidates N');ax2[di].set_ylabel('Mean distinct answer groups');ax2[di].set_title(dataset);ax2[di].legend();ax2[di].grid(alpha=.15)
    for ax in axes[-1,:]:ax.set_xlabel('Agreement among first 4 candidates' if ax==axes[-1,0] else 'Correct fraction in all 32 candidates')
    axes[0,0].set_title('Observable early agreement');axes[0,1].set_title('Retrospective pool correctness')
    axes[0,0].legend();axes[0,1].legend()
    fig.suptitle('Frozen stopping policies: observed evidence and retrospective difficulty',weight='bold')
    captions={
        '09_stopping_associations':'800 held-out questions per dataset, ten saved orders each. For each fixed family, points group question-order observations by the exact first-four leader agreement (left) or exact full-pool correct fraction (right); points are descriptive means, without intervals or minimum-count filtering. Full-pool correctness is retrospective only and never enters stopping. All saved per-question correct, cost, rollout, and early-error totals are exactly reproduced from saved observable states and the already-frozen policy, without symbolic regrading or tuning. Sparse groups may contain few questions; raw observations are saved. The earlier MATH test:4928 timeout limitation remains.',
        '10_answer_diversity':'Mean distinct revealed answer groups over 800 held-out questions and ten nested saved orders per dataset; groups use the original frozen equivalence and prefix processing. Dotted line is the maximum N groups. The gap N minus observed groups counts repeated answer-group support, not duplicate reasoning text or useless computation. Missing answers share one group and remain included. This descriptive curve does not identify causal effects of vision or prove answer correctness.',
    }
    for name,figure in [('09_stopping_associations',fig),('10_answer_diversity',fig2)]:
        for ext in ['png','svg']:figure.savefig(out/(name+'.'+ext),dpi=170,bbox_inches='tight',facecolor='white')
        (out/(name+'.txt')).write_text(captions[name]+'\n');plt.close(figure)
    write_csv(out/'stopping_observations.csv',observations)
    write_csv(out/'answer_diversity.csv',diversity)
    jwrite(out/'retrospective_summary.json',{'outcomes_changed':False,'policies_changed':False,'scoring_executed':False,'summaries':summaries,'observations':len(observations)})
    print(json.dumps(summaries,indent=2))

if __name__=='__main__':main(Path(sys.argv[1]),Path(sys.argv[2]))
