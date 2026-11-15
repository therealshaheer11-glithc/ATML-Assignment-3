"""Read-only result export and predeclared qualitative-case selection; no rescoring."""
import hashlib,json,re
from selection import PrefixState,stop_from_states

def sha_bytes(data):return hashlib.sha256(data).hexdigest()
def file_hash(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda:stream.read(4*1024*1024),b''):h.update(part)
    return h.hexdigest()
def require(c,message):
    if not c:raise ValueError(message)

def timeout_audit(text):
    pending=0;cases=[];total=0
    for line in text.splitlines():
        if 'Timeout during comparison' in line:pending+=1;total+=1
        match=re.match(r'^(MATH|MathVista) (.+): saved and verified; total ',line)
        if match and pending:
            cases.append({'dataset':match.group(1),'sample_id':match.group(2),'comparison_timeout_messages':pending});pending=0
    return {'comparison_timeout_messages':total,'question_intervals':cases,'unattributed_messages':pending,'interpretation':'Intervals identify the question being processed, not individual comparison pairs. The frozen library returned unverified comparisons under its five-second limit. Primary saved scores are unchanged.'}

def select_examples(records,selected,dataset):
    method='K_RWC' if dataset=='MATH' else 'Z_RWC'
    categories={name:[] for name in ['SC_wrong_quality_correct','SC_correct_quality_wrong','no_correct_candidate','correct_candidate_SC_missed']+[f'{f}_{x}' for f in ['agreement','quality'] for x in ['early_success','fixed32_recoverable_early_error']]}
    def add(name,q,order,ids,extra=None):
        group=categories[name]
        if len(group)<2 and not any(x['sample_id']==q for x in group):group.append({'sample_id':q,'order':order,'selected_rollout_ids':ids,**(extra or {})})
    for record in sorted(records,key=lambda r:r['sample_id']):
        q=record['sample_id'];truth={int(i):v for i,v in record['candidate_correctness'].items()}
        rows={(r['order'],r['method']):r for r in record['selections'] if r['N']==32}
        for order in range(10):
            sc=rows[(order,'SC')];quality=rows[(order,method)]
            if not sc['correct'] and quality['correct']:add('SC_wrong_quality_correct',q,order,[sc['selected_rollout_id'],quality['selected_rollout_id']])
            if sc['correct'] and not quality['correct']:add('SC_correct_quality_wrong',q,order,[sc['selected_rollout_id'],quality['selected_rollout_id']])
            if not any(truth.values()):add('no_correct_candidate',q,order,[sc['selected_rollout_id']])
            if any(truth.values()) and not sc['correct']:add('correct_candidate_SC_missed',q,order,[sc['selected_rollout_id']])
            states=[PrefixState(**s) for s in record['observable_stopping_states'][order]]
            for family,setting in selected.items():
                stop=stop_from_states(iter(states),setting['tau'],setting['gamma']);i=record['orders'][order][stop.self_consistency_index]
                if stop.n<32 and truth[i]:add(family+'_early_success',q,order,[i,sc['selected_rollout_id']],{'stopped_at':stop.n})
                if stop.n<32 and not truth[i] and sc['correct']:add(family+'_fixed32_recoverable_early_error',q,order,[i,sc['selected_rollout_id']],{'stopped_at':stop.n})
    return {'comparison':'SC versus '+method,'rule':'First two distinct lexicographic question IDs per category; earliest qualifying saved order. Both success/failure directions; fewer if absent. Additional timeout cases are explicitly separate diagnostics.','categories':categories}
