"""CPU-only integrity and saved-result reproduction; no generation or regrading."""
import csv, hashlib, json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent

def read(p):return json.loads(p.read_text())
def close(a,b):
    if not np.allclose(a,b,rtol=1e-10,atol=1e-9):raise AssertionError((a,b))
def table(p):
    with p.open() as f:return list(csv.DictReader(f))
def ci(values,ids):return np.percentile(np.asarray(values,dtype=float)[ids].mean(axis=1),[2.5,97.5],method='linear')
def main():
    manifest=read(ROOT/'RELEASE_MANIFEST.json')
    for e in manifest['files']:
        p=ROOT/e['path'];raw=p.read_bytes()
        assert len(raw)==e['bytes'] and hashlib.sha256(raw).hexdigest()==e['sha256'],e['path']
    offline=ROOT/'results/offline';checked=0;paired=0
    metrics=table(offline/'method_metrics.csv');differences=table(offline/'paired_accuracy_differences.csv')
    for dataset,label in [('MATH','math'),('MathVista','mathvista')]:
        with np.load(offline/(label+'_question_metrics.npz'),allow_pickle=False) as z:
            acc=z['accuracy'];cost=z['cost'];keys=[tuple(k) for k in json.loads(str(z['keys_json']))];qids=z['question_ids'];assert len(qids)==len(set(qids))==800
        ids=np.load(offline/(label+'_bootstrap_question_indices.npy'),allow_pickle=False);assert ids.shape==(1000,800) and ids.min()>=0 and ids.max()<800
        for row in metrics:
            if row['dataset']!=dataset:continue
            j=keys.index((int(row['N']),row['method']))
            close(acc[:,j].mean()*100,float(row['accuracy_percent']));close(cost[:,j].mean(),float(row['mean_token_cost']))
            close(ci(acc[:,j],ids)*100,[float(row['accuracy_low_percent']),float(row['accuracy_high_percent'])]);checked+=1
        for row in differences:
            if row['dataset']!=dataset:continue
            a=keys.index((int(row['A_N']),row['A_method']));b=keys.index((int(row['B_N']),row['B_method']))
            delta=100*(acc[:,a]-acc[:,b]);close(delta.mean(),float(row['difference_pp']));close(ci(delta,ids),[float(row['low_pp']),float(row['high_pp'])]);paired+=1
    live=ROOT/'results/live';a=read(live/'analysis.json');qs=read(live/'question_results.json');assert len(qs)==50
    ids=np.load(live/'bootstrap_question_indices.npy',allow_pickle=False);assert ids.shape==(1000,50) and ids.min()>=0 and ids.max()<50
    assert [q['sample_id'] for q in qs]==read(live/'bootstrap_question_ids.json')
    for m in a['metrics']:
        vv=[q['strategies'][m['method']] for q in qs];assert sum(v['correct'] for v in vv)==m['correct']
        for field in ['accuracy_percent','input_tokens','output_tokens','retained_nontermination_tokens','inference_seconds','call_wall_seconds','peak_allocated_bytes','peak_reserved_bytes']:
            vals=np.array([100*v['correct'] if field=='accuracy_percent' else v[field] for v in vv]);close(vals.mean(),m[field]['mean']);close(ci(vals,ids),[m[field]['low'],m[field]['high']])
    for d in a['paired_differences']:
        field='correct' if d['measure']=='accuracy_pp' else d['measure'];scale=100 if field=='correct' else 1
        vals=[scale*(float(q['strategies'][d['A']][field])-float(q['strategies'][d['B']][field])) for q in qs]
        close(np.mean(vals),d['mean']);close(ci(vals,ids),[d['low'],d['high']])
    # Cost identities use unique calls plus the shared initial, without executing a model.
    for field in ['input_tokens','output_tokens','inference_seconds']:
        unique=sum(c[field] for q in qs for c in q['calls'].values());shared=sum(q['calls']['parallel_1'][field] for q in qs)
        close(sum(q['strategies'][m][field] for q in qs for m in ['long','parallel','refinement']),unique+shared)
    outputs=read(live/'generated_responses.json');assert len(outputs)==250 and len({(r['sample_id'],r['call_name']) for r in outputs})==250
    from src.live_execution.live_protocol import extract_last_box
    for r in outputs:assert extract_last_box(r['generated_text'])==r['extraction']
    assert sum(r['termination']['truncated_by_cap'] for r in outputs)==24
    assert sum(r['extraction']['answer'] is None for r in outputs)==23
    assert sum(len(r['output_ids']) for r in outputs)==152187
    for q in qs:
        assert q['strategies']['refinement']['selected_call']=='revision'
    print(json.dumps({'status':'PASS','manifest_files':len(manifest['files']),'offline_method_rows_reproduced':checked,'offline_paired_intervals_reproduced':paired,'live_questions':50,'live_calls':250,'live_contrasts_reproduced':len(a['paired_differences']),'shared_cost_identities':'passed','raw_output_extractions_replayed':250,'numpy_used':np.__version__,'GPU_used':False,'fresh_symbolic_regrading':False},indent=2))
if __name__=='__main__':main()
