"""Standalone local figure/example browser; escapes all source-provided text."""
import hashlib,html,json,sys
from collections import defaultdict
from pathlib import Path

def escape(x):return html.escape(str(x),quote=True)
def main(data,out):
 a=json.loads((out/'analysis.json').read_text());selection=json.loads((data/'audit/example_selection.json').read_text());source=json.loads((data/'audit/source_backed_example_rollouts.json').read_text())
 css='body{font:16px/1.55 system-ui,sans-serif;color:#202830;max-width:1250px;margin:auto;padding:28px}h1,h2,h3{line-height:1.2}a{color:#14688b}figure{margin:32px 0;padding-bottom:20px;border-bottom:1px solid #ddd}img{max-width:100%;height:auto}figcaption{font-size:14px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:7px;border-bottom:1px solid #ddd}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f5;padding:14px}summary{cursor:pointer;padding:8px}details{border:1px solid #ddd;margin:8px 0;border-radius:5px}.note{background:#fff5df;padding:12px}.good{color:#226642}.bad{color:#9a3838}'
 def page(title,body):return '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+escape(title)+'</title><style>'+css+'</style><body>'+body+'</body></html>'
 body='<h1>Task 3 · saved offline results</h1><p>800 held-out questions per dataset; ten saved candidate orders averaged within each question. No model calls or rescoring in this analysis. Live generation remains pending.</p><p class="note">The frozen MATH evaluator logged 78 comparison timeouts while processing test:4928. Individual comparison pairs were not logged; their precise effect cannot be inferred. Scores and policies are preserved. Intervals below are pointwise 95% paired question bootstrap (1,000 draws), without a multiple-comparison correction.</p><p><a href="examples.html">Browse deterministic qualitative examples and original responses</a> · <a href="method_metrics.csv">All method metrics</a> · <a href="paired_accuracy_differences.csv">Paired comparisons</a> · <a href="signal_associations.csv">Signal correlations</a> · <a href="analysis.json">Full analysis definitions and adaptive budget brackets</a></p>'
 body+='<table><tr><th>Dataset</th><th>Method</th><th>Accuracy</th><th>95% interval</th><th>Mean tokens</th></tr>'
 for d,v in a['datasets'].items():
  for r in v['metrics']:
   if (r['N']==32 and r['method'] in ['SC','K_RWC','Z_RWC']) or r['N']==0:
    body+=f"<tr><td>{escape(d)}</td><td>{escape(r['method'])}</td><td>{r['accuracy_percent']:.2f}%</td><td>{r['accuracy_low_percent']:.2f}–{r['accuracy_high_percent']:.2f}%</td><td>{r['mean_token_cost']:,.0f}</td></tr>"
 body+='</table>'
 for p in sorted((out/'figures').glob('[0-9][0-9]_*.png')):
  caption=p.with_suffix('.txt').read_text();body+='<figure><h2>'+escape(p.stem.replace('_',' '))+'</h2><a href="figures/'+escape(p.name)+'"><img loading="lazy" src="figures/'+escape(p.name)+'" alt="'+escape(p.stem)+'"></a><figcaption>'+escape(caption)+'</figcaption><p><a href="figures/'+escape(p.with_suffix('.svg').name)+'">Scalable SVG</a></p></figure>'
 (out/'index.html').write_text(page('Task 3 offline results',body))
 body='<h1>Task 3 · original qualitative evidence</h1><p><a href="index.html">Back to figures and numerical results</a></p><p>First two distinct lexicographic question IDs per declared category; earliest qualifying saved order. All 32 source responses are preserved for each displayed question. Categories may overlap. The timeout diagnostic and eight preselected trajectory rollouts are supplementary. These examples illustrate saved outcomes; they do not estimate average performance. MathVista images remain in the supplied source package.</p>'
 counts={}
 for d,rows in source.items():
  grouped=defaultdict(list)
  for r in rows:grouped[r['sample_id']].append(r)
  reasons=defaultdict(list)
  for category,examples in selection[d]['categories'].items():
   for e in examples:reasons[e['sample_id']].append((category,e))
  label='math' if d=='MATH' else 'mathvista';counts[d]=len(grouped);body+='<h2>'+escape(d)+'</h2><p>'+escape(selection[d]['comparison'])+'</p>'
  for q,rollouts in sorted(grouped.items()):
   assert len(rollouts)==32 and {r['rollout_id'] for r in rollouts}==set(range(32))
   record=json.loads((data/'results/task3_heldout_v1'/label/(hashlib.sha256(q.encode()).hexdigest()+'.json')).read_text())['record']
   body+='<details id="'+escape(d+'_'+q)+'"><summary><strong>'+escape(q)+'</strong> · '+escape(', '.join(c for c,_ in reasons[q]) or 'Supplementary diagnostic / fixed trajectories')+'</summary><h3>Question</h3><pre>'+escape(rollouts[0]['question_text'])+'</pre><p>Reference: <code>'+escape(rollouts[0]['gold_answer'])+'</code></p>'
   for c,e in reasons[q]:body+='<pre>'+escape(json.dumps({'category':c,**e},indent=2))+'</pre>'
   if q=='test:4928':body+='<p class="note">Timeout diagnostic: 78 comparison messages occurred in this question interval. The stored grades are shown below, without new adjudication.</p>'
   body+='<table><tr><th>Rollout</th><th>Extracted answer</th><th>Frozen correct</th><th>Retained tokens</th><th>Capped</th></tr>'
   for r in sorted(rollouts,key=lambda r:r['rollout_id']):
    rid=r['rollout_id'];correct=record['candidate_correctness'][str(rid)];body+='<tr><td><a href="#'+escape(d+'_'+q+'_'+str(rid))+'">'+str(rid)+'</a></td><td>'+escape(r['extracted_answer'])+'</td><td>'+str(correct)+'</td><td>'+str(r['n_generated_text_tokens'])+'</td><td>'+str(r['hit_max_new_tokens'])+'</td></tr>'
   body+='</table>'
   for r in sorted(rollouts,key=lambda r:r['rollout_id']):body+='<details id="'+escape(d+'_'+q+'_'+str(r['rollout_id']))+'"><summary>Rollout '+str(r['rollout_id'])+' · original generated text</summary><pre>'+escape(r['generated_text'])+'</pre></details>'
   body+='</details>'
 (out/'examples.html').write_text(page('Task 3 qualitative evidence',body));print(json.dumps({'example_questions':counts,'source_responses':sum(len(v) for v in source.values()),'source_text_html_escaped':True}))
if __name__=='__main__':main(Path(sys.argv[1]),Path(sys.argv[2]))
