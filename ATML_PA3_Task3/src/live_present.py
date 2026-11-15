"""Scientific figures and an escaped, standalone browser for saved live results."""
import html
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS = {'long': 'Single longer', 'parallel': 'Parallel SC', 'refinement': 'Refinement'}
COLORS = ['#346d95', '#d28635', '#5b8a6a']

def main(snapshot, out):
    a = json.loads((out / 'analysis.json').read_text())
    questions = json.loads((out / 'question_results.json').read_text())
    figures = out / 'figures'
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none', 'savefig.dpi': 170})
    metrics = a['metrics']
    x = np.arange(3)
    labels = [LABELS[m['method']] for m in metrics]
    def save(fig, name, caption):
        for extension in ['png', 'svg']:
            fig.savefig(figures / (name + '.' + extension), bbox_inches='tight', facecolor='white')
        (figures / (name + '.txt')).write_text(caption + '\n')
        plt.close(fig)
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.1), layout='constrained')
    for ax, field, title, unit in zip(axes, ['accuracy_percent', 'output_tokens', 'input_tokens', 'inference_seconds'], ['Accuracy', 'Actual output', 'Actual input', 'Inference time'], ['Correct answers (%)', 'Generated tokens / question', 'Input tokens / question', 'Seconds / question']):
        values = np.array([m[field]['mean'] for m in metrics])
        low = np.array([m[field]['low'] for m in metrics])
        high = np.array([m[field]['high'] for m in metrics])
        ax.bar(x, values, color=COLORS, width=.63)
        ax.errorbar(x, values, yerr=[values-low, high-values], fmt='none', color='#333333', capsize=4)
        ax.set_xticks(x, labels, rotation=25, ha='right')
        ax.set_title(title, weight='bold')
        ax.set_ylabel(unit)
        ax.set_axisbelow(True)
        ax.grid(axis='y', alpha=.18)
        if field == 'accuracy_percent':
            ax.set_ylim(0, 100)
            for i, m in enumerate(metrics):
                ax.text(i, 3, f"{m['correct']}/50", ha='center', color='white', weight='bold')
    fig.suptitle('Live MATH: equal 3,072-token allowances, different actual costs', weight='bold')
    save(fig, '01_live_accuracy_cost', 'Same fixed 50 MATH evaluation questions and original Qwen2.5-3B-Instruct revision; FP16 T4, batch 1. Bars are question means, whiskers pointwise 95% question-bootstrap intervals (1,000 paired draws). Allowances are 3,072 output tokens per strategy/question; early EOS leaves unused allowance. Parallel SC uses three independent calls generated serially: runtime is summed inference, not concurrent latency. Shared initial input/output tokens and time are charged to both parallel and refinement. Input counts include the complete revision prompt. Accuracy-difference intervals, rather than overlap of individual intervals, are in paired_differences.csv; all three paired accuracy intervals include zero.')
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.1), layout='constrained')
    definitions = [
        ('Missing selected answer', [100*m['final_extraction_missing_count']/50 for m in metrics], 'Questions (%)'),
        ('Missing component extraction', [100*m['component_missing_calls']/m['component_call_count'] for m in metrics], 'Participating calls (%)'),
        ('Any component capped', [100*m['any_component_capped_questions']/50 for m in metrics], 'Questions (%)'),
    ]
    for ax, (title, values, ylabel) in zip(axes, definitions):
        ax.bar(x, values, color=COLORS, width=.63)
        ax.set_xticks(x, labels, rotation=20, ha='right')
        ax.set_title(title, weight='bold')
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(32, max(values)+6))
        for i, value in enumerate(values):
            ax.text(i, value+.7, f'{value:.1f}%', ha='center')
    fig.suptitle('Extraction and truncation remain in every denominator', weight='bold')
    save(fig, '02_live_failures', 'Missing means absent, empty, or unclosed last literal boxed extraction; no numeric fallback. Selected-answer rate uses 50 questions per strategy. Component extraction rate uses 50 long, 150 parallel, or 100 refinement-participating calls; the initial call belongs to both parallel and refinement. Any-component-capped rate uses 50 questions and counts questions with at least one token-cap termination (EOS at the cap is not truncation). All 24 uniquely capped calls are parallel candidates; six are shared initials. None of the 50 longer or 50 revision calls are capped. No missing/capped response is excluded or regenerated.')
    t = a['revision_transitions']
    matrix = np.array([[t['wrong_to_wrong'], t['wrong_to_correct']], [t['correct_to_wrong'], t['correct_to_correct']]])
    fig, ax = plt.subplots(figsize=(6.3, 4.6), layout='constrained')
    ax.imshow(matrix, cmap='Blues', vmin=0, vmax=50)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{matrix[i,j]} / 50', ha='center', va='center', fontsize=17, color='white' if matrix[i,j] > 25 else '#1b2d3d')
    ax.set_xticks([0,1], ['Wrong', 'Correct'])
    ax.set_yticks([0,1], ['Wrong', 'Correct'])
    ax.set_xlabel('Revised answer')
    ax.set_ylabel('Shared initial answer')
    ax.set_title('One beneficial revision; no harmful revisions observed', weight='bold', pad=16)
    save(fig, '03_revision_transitions', 'All 50 questions, comparing the first parallel response (shared initial) with the unconditional final revision. Wrong-to-correct: 1/20 initially wrong; correct-to-wrong: 0/30 initially correct. Missing initial extractions count as wrong. The sole beneficial case, test:2688, began with a capped initial response with no extracted answer; revision produced -1, also obtained by the longer response. This is recovery from an incomplete answer, not evidence of correcting an explicitly wrong extracted mathematical answer. Initial accuracy 30/50; revised accuracy 31/50. No harmful category examples exist in this sample; that does not establish safety of revision in general.')

    esc = lambda value: html.escape(str(value), quote=True)
    css = 'body{max-width:1320px;margin:auto;padding:28px;font:16px/1.5 system-ui;color:#24313a}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;border-bottom:1px solid #ddd;padding:8px}img{width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f5;padding:14px}details{border:1px solid #ddd;padding:10px;margin:12px 0}summary{cursor:pointer}.note{padding:12px;background:#fff4dc}figure{margin:32px 0}figcaption{font-size:14px}'
    def page(title, body):
        return '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+esc(title)+'</title><style>'+css+'</style><body>'+body+'</body></html>'
    body = '<h1>Task 3 · live inference</h1><p>50 fixed MATH evaluation questions · 250 unique verified calls · five completed T4 sessions · no extra generations.</p><p class="note">All paired accuracy-difference intervals include zero. These results do not establish a clear accuracy winner. Parallel candidates were generated serially at batch size 1; timing below is summed measured inference.</p><p><a href="examples.html">Responses and deterministic examples</a> · <a href="analysis.json">Full definitions and results</a> · <a href="method_metrics.csv">Metrics CSV</a> · <a href="paired_differences.csv">Paired intervals</a></p><table><tr><th>Strategy</th><th>Correct</th><th>Mean input tokens</th><th>Mean output tokens</th><th>Mean inference seconds</th><th>Max allocated GiB</th></tr>'
    for m in metrics:
        body += f"<tr><td>{esc(LABELS[m['method']])}</td><td>{m['correct']}/50 ({m['accuracy_percent']['mean']:.0f}%)</td><td>{m['input_tokens']['mean']:.2f}</td><td>{m['output_tokens']['mean']:.2f}</td><td>{m['inference_seconds']['mean']:.2f}</td><td>{m['peak_allocated_bytes']['maximum']/1024**3:.3f}</td></tr>"
    body += '</table>'
    for path in sorted(figures.glob('*.png')):
        body += '<figure><a href="figures/'+path.name+'"><img src="figures/'+path.name+'" alt="'+esc(path.stem)+'"></a><figcaption>'+esc(path.with_suffix('.txt').read_text())+'</figcaption><p><a href="figures/'+path.with_suffix('.svg').name+'">SVG</a></p></figure>'
    body += '<h2>Interpretation and accounting limits</h2><ul>'+''.join('<li>'+esc(line)+'</li>' for line in a['limits'])+'</ul>'
    (out/'index.html').write_text(page('Task 3 live results', body))
    selected = {q for values in a['examples'].values() for q in values}
    body = '<h1>Task 3 · live responses</h1><p><a href="index.html">Back to results</a></p><p>Declared example rule: first two distinct lexicographic question IDs per beneficial/harmful revision category; fewer if absent. Beneficial revision: '+esc(a['examples']['beneficial_revision'])+'. Harmful revision: none observed. All 50 questions and all 250 responses are available below.</p>'
    raw_calls = {}
    for path in (snapshot/'results/task3_live_generation_A002/calls').glob('*.json'):
        record = json.loads(path.read_text())['record']
        raw_calls[record['sample_id'],record['call_name']] = record
    for q in sorted(questions, key=lambda r: (r['sample_id'] not in selected, r['sample_id'])):
        qid=q['sample_id']
        body += '<details'+(' open' if qid in selected else '')+'><summary>'+esc(qid)+' · '+esc(q['revision_transition'])+(' · selected example' if qid in selected else '')+'</summary><h2>Question</h2><pre>'+esc(q['question'])+'</pre><p>Reference: <code>'+esc(q['reference'])+'</code></p><pre>'+esc(json.dumps(q['strategies'],indent=2))+'</pre>'
        for name in ['long','parallel_1','parallel_2','parallel_3','revision']:
            record=raw_calls[qid,name]
            body += '<details><summary>'+esc(name)+' · answer '+esc(record['extraction']['answer'])+' · correct '+esc(q['calls'][name]['correct'])+'</summary><h3>Native messages</h3><pre>'+esc(json.dumps(record['messages'],indent=2,ensure_ascii=False))+'</pre><h3>Full response</h3><pre>'+esc(record['generated_text'])+'</pre><h3>Accounting</h3><pre>'+esc(json.dumps(q['calls'][name],indent=2))+'</pre></details>'
        body += '</details>'
    (out/'examples.html').write_text(page('Task 3 live responses',body))
    print(json.dumps({'figures':3,'example_questions':a['examples'],'all_questions':len(questions),'all_responses':len(raw_calls),'source_text_html_escaped':True,'matplotlib':matplotlib.__version__,'numpy':np.__version__}))

if __name__ == '__main__':
    main(Path(sys.argv[1]),Path(sys.argv[2]))
