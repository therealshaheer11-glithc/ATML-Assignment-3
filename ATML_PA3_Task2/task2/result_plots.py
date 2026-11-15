"""Static scientific figures and a complete, unselected paired-results gallery."""
from pathlib import Path
import html,json
import numpy as np
from .result_analysis import rate_values

COLORS={'correct':'#2166ac','complementary':'#d95f02','mismatched':'#7570b3','no_image':'#666666'}
LABELS={'correct':'Correct','complementary':'Partner swap','mismatched':'Unrelated','no_image':'No image'}


def make_plots(result,training,dev,stages,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'figure.dpi':120,'savefig.dpi':170,'axes.titleweight':'bold'})
    out=Path(out);out.mkdir(parents=True,exist_ok=True);files=[]
    conditions=list(COLORS)
    def save(fig,name):
        fig.tight_layout()
        for extension in ('png','svg'):
            path=out/(name+'.'+extension);fig.savefig(path,bbox_inches='tight',facecolor='white');files.append(path)
        plt.close(fig)
    def boundaries(ax):
        for step in (2250,3750):ax.axvline(step,color='#888888',linestyle='--',linewidth=.9)
        ax.set_xlim(0,4500);ax.set_xlabel('Successful training update');ax.grid(alpha=.17)
    def metric(suite,condition,name,kind='all'):
        return next(r for r in result['aggregates'] if (r['suite'],r['condition'],r['metric'],r['question_type'])==(suite,condition,name,kind))
    def error(rows,factor=1):
        return np.array([[max(0,r['value']-r['ci95_low']) for r in rows],
            [max(0,r['ci95_high']-r['value']) for r in rows]])*factor
    def bars(ax,suite,name,factor=1):
        rows=[metric(suite,c,name) for c in conditions];x=np.arange(4)
        ax.bar(x,[r['value']*factor for r in rows],color=[COLORS[c] for c in conditions],
            yerr=error(rows,factor),capsize=3,width=.65)
        ax.set_xticks(x,[LABELS[c] for c in conditions],rotation=17,ha='right');ax.grid(axis='y',alpha=.15)
    steps=sorted(map(int,dev))
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    axes[0,0].plot([r['global_update'] for r in training],[r['loss'] for r in training],color='#2166ac',alpha=.18,linewidth=.6,label='Each update')
    for stage in (1,2,3):
        rows=[r for r in training if r['stage']==stage];loss=np.asarray([r['loss'] for r in rows]);n=min(50,len(rows))
        smoothed=np.convolve(loss,np.ones(n)/n,mode='valid')
        axes[0,0].plot([r['global_update'] for r in rows[n-1:]],smoothed,color='#133f68',linewidth=1.5,label='50-update mean' if stage==1 else None)
    axes[0,0].set_title('Training target + EOT loss');axes[0,0].set_ylabel('Mean token CE per example');axes[0,0].legend(fontsize=8)
    axes[0,1].plot(steps,[dev[str(u)]['caption']['correct_image_CE'] for u in steps],marker='o',color=COLORS['correct'])
    axes[0,1].set_title('Caption development loss');axes[0,1].set_ylabel('Mean token CE, including EOT')
    for ax,suite in zip(axes[1],('grounding','natural')):
        for condition in ('correct','complementary','no_image'):
            ax.plot(steps,[dev[str(u)]['suites'][suite]['conditions'][condition]['mean_reference_NLL'] for u in steps],marker='o',markersize=3,color=COLORS[condition],label=LABELS[condition])
        ax.set_title(suite.title()+' development NLL');ax.set_ylabel('Reference content-only NLL');ax.legend(fontsize=8)
    for ax in axes.flat:boundaries(ax)
    save(fig,'01_learning_curves')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,suite in zip(axes,('grounding','natural')):
        ax.plot(steps,[dev[str(u)]['suites'][suite]['swap_penalty_NLL'] for u in steps],marker='o',color='#2166ac')
        ax.axhline(0,color='black',linewidth=.8);ax.set_title(suite.title()+' development swap penalty')
        ax.set_ylabel('Partner-swap NLL − correct NLL');boundaries(ax)
    save(fig,'02_development_swap_penalties')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,suite in zip(axes,('grounding','natural')):
        bars(ax,suite,'accuracy',100);ax.set_ylim(0,105);ax.set_ylabel('Accuracy (%)')
        ax.set_title(suite.title()+(' · normalized exact match' if suite=='grounding' else ' · VQAv2 soft score'))
    fig.suptitle('Final original-reference accuracy · 95% complete-pair bootstrap intervals',fontsize=12)
    save(fig,'03_final_accuracy')
    fig,axes=plt.subplots(1,2,figsize=(12,4.7))
    for ax,suite in zip(axes,('grounding','natural')):
        types=sorted({r['question_type'] for r in result['pair_rows'] if r['suite']==suite});x=np.arange(len(types));width=.18
        for i,condition in enumerate(conditions):
            rows=[metric(suite,condition,'accuracy',kind) for kind in types]
            ax.bar(x+(i-1.5)*width,[r['value']*100 for r in rows],width=width,color=COLORS[condition],yerr=error(rows,100),capsize=2,label=LABELS[condition])
        ax.set_xticks(x,types);ax.set_ylim(0,105);ax.set_ylabel('Accuracy (%)');ax.set_title(suite.title());ax.grid(axis='y',alpha=.15)
    axes[1].legend(fontsize=8,loc='upper left',bbox_to_anchor=(1.01,1));fig.suptitle('Final accuracy by question type · original references',fontsize=12)
    save(fig,'04_accuracy_by_type')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,suite in zip(axes,('grounding','natural')):
        bars(ax,suite,'reference_NLL');ax.set_ylabel('Original answer content-only NLL');ax.set_title(suite.title())
    fig.suptitle('Final reference NLL · lower is better · 95% complete-pair intervals',fontsize=12)
    save(fig,'05_final_reference_NLL')
    fig,axes=plt.subplots(2,4,figsize=(16,7))
    for row,suite in enumerate(('grounding','natural')):
        for col,(name,label) in enumerate((('pair_accuracy','Both canonical answers correct'),('identical_rate','Identical answers within pair'),('empty_rate','Empty answers'),('truncation_rate','Reached 16-token cap'))):
            ax=axes[row,col];bars(ax,suite,name,100);ax.set_ylim(0,105);ax.set_title(label,fontsize=10)
            if col==0:ax.set_ylabel(suite.title()+' (%)')
    fig.suptitle('Pair and output diagnostics · empty outputs remain in score denominators',fontsize=12)
    save(fig,'06_pair_and_output_diagnostics')
    fig,axes=plt.subplots(2,2,figsize=(11,7.5))
    for col,suite in enumerate(('grounding','natural')):
        for row,name in enumerate(('accuracy','reference_NLL')):
            records=[next(r for r in result['advantages'] if r['suite']==suite and r['question_type']=='all' and r['metric']==name and r['comparison']=='correct_vs_'+c) for c in conditions[1:]]
            factor=100 if name=='accuracy' else 1;ax=axes[row,col]
            ax.bar(np.arange(3),[r['value']*factor for r in records],color=[COLORS[c] for c in conditions[1:]],yerr=error(records,factor),capsize=3)
            ax.axhline(0,color='black',linewidth=.8);ax.set_xticks(range(3),[LABELS[c] for c in conditions[1:]],rotation=12)
            ax.set_title(suite.title());ax.set_ylabel('Correct − control accuracy (pp)' if name=='accuracy' else 'Control − correct NLL');ax.grid(axis='y',alpha=.15)
    fig.suptitle('Correct-image advantages · positive favors the correct image',fontsize=12)
    save(fig,'07_correct_image_advantages')
    fig,ax=plt.subplots(figsize=(9,4))
    bottom=np.zeros(2)
    for name,label,color in [('canonical_accuracy','Original canonical answer','#2166ac'),('partner_canonical_rate','Partner canonical answer','#d95f02'),('neither_canonical_rate','Neither canonical answer','#aaaaaa')]:
        values=[next(r['value']*100 for r in result['partner_following'] if r['suite']==suite and r['question_type']=='all' and r['metric']==name) for suite in ('grounding','natural')]
        ax.bar([0,1],values,bottom=bottom,label=label,color=color,width=.6);bottom+=values
    ax.set_xticks([0,1],['Grounding','Natural VQA']);ax.set_ylim(0,100);ax.set_ylabel('Complementary-swap answers (%)');ax.legend(fontsize=8,bbox_to_anchor=(1.01,1),loc='upper left');ax.set_title('Partner-following diagnostic · separate from original-reference scores')
    save(fig,'08_partner_following')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for name,label,color in [('connector','Connector','#2166ac'),('lora','LoRA','#d95f02')]:
        points=[(r['global_update'],rate_values(r).get(name,0)) for r in training if rate_values(r).get(name,0)>0]
        axes[0].plot([p[0] for p in points],[p[1] for p in points],label=label,color=color)
    axes[0].set_yscale('log');axes[0].set_ylabel('Recorded learning rate');axes[0].set_title('Approved stage schedules');axes[0].legend();boundaries(axes[0])
    bottom=np.zeros(3)
    for kind,color in [('caption','#2166ac'),('presence','#4daf4a'),('counting','#984ea3'),('natural','#ff7f00')]:
        values=[100*s['realized_example_counts'].get(kind,0)/s['examples'] for s in stages]
        axes[1].bar([1,2,3],values,bottom=bottom,label=kind.title(),color=color);bottom+=values
    axes[1].set_xticks([1,2,3],['Stage 1','Stage 2','Stage 3']);axes[1].set_ylim(0,100);axes[1].set_ylabel('Realized training examples (%)');axes[1].set_title('Observed curriculum mixture');axes[1].legend(fontsize=8,loc='upper left',bbox_to_anchor=(1.01,1))
    save(fig,'09_learning_rates_and_mixture')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    axes[0].bar(['Stage 1','Stage 2','Stage 3'],[r['training_update_wall_seconds']/60 for r in stages],color=['#2166ac','#4daf4a','#ff7f00'])
    axes[0].set_ylabel('Measured update wall time (minutes)');axes[0].set_title('Training compute · loading/IO/evaluation excluded')
    x=np.arange(4);width=.34
    for i,suite in enumerate(('grounding','natural')):
        values=[metric(suite,c,'generation_seconds_per_answer')['value']*1000 for c in conditions]
        axes[1].bar(x+(i-.5)*width,values,width=width,label=suite.title())
    axes[1].set_xticks(x,[LABELS[c] for c in conditions],rotation=17,ha='right');axes[1].set_ylabel('Batched generation milliseconds / answer');axes[1].set_title('Generation throughput time · batch8');axes[1].legend(fontsize=8,loc='upper left',bbox_to_anchor=(1.01,1))
    save(fig,'10_measured_resources')
    return files


def gallery_html(scored_rows,image_files):
    """Show every final pair, in ID order; no prediction-based subset selection."""
    esc=lambda value:html.escape(str(value),quote=True)
    groups={}
    for row in scored_rows:groups.setdefault((row['suite'],row['pair_id']),{})[(row['condition'],row['side'])]=row
    cards=[]
    for (suite,pid),rows in sorted(groups.items()):
        a,b=rows[('correct',0)],rows[('correct',1)]
        cells=[]
        for side,row in enumerate((a,b)):
            image=image_files[(row['source_split'],row['image_id'])]
            cells.append('<div><img loading="lazy" src="'+esc(image)+'"><p>Image '+str(side+1)+' · '+esc(row['image_id'])+'</p><p>Canonical: <b>'+esc(row['canonical'])+'</b><br>Original target: '+esc(row['target'])+'</p>'+(('<details><summary>All ten human answers</summary>'+esc(json.dumps(row['human_answers'],ensure_ascii=False))+'</details>') if 'human_answers' in row else '')+'</div>')
        table=[]
        for condition in COLORS:
            cells2=[]
            for side in (0,1):
                row=rows[(condition,side)];prediction=esc(row['prediction']) if row['prediction'] else '<i>[empty]</i>'
                cells2.append('<td>'+prediction+'<br><small>Original-reference score '+f'{row["original_reference_score"]:.3f}'+' · NLL '+f'{row["reference_NLL"]:.3f}'+' · '+esc(row['finish_reason'])+'<br>Input image: '+esc(row['input_image_id'])+'</small></td>')
            table.append('<tr><th>'+LABELS[condition]+'</th>'+''.join(cells2)+'</tr>')
        cards.append('<article id="'+esc(suite+'_'+str(pid))+'"><h2>'+esc(suite.title()+' · '+str(pid)+' · '+a['question_type'])+'</h2><p class="question">'+esc(a['question'])+'</p><div class="pair">'+''.join(cells)+'</div><table><thead><tr><th>Condition</th><th>Original example 1</th><th>Original example 2</th></tr></thead><tbody>'+''.join(table)+'</tbody></table></article>')
    return ('<!doctype html><html><head><meta charset="utf-8"><title>Task 2 · all final pairs</title><style>body{font:16px/1.5 system-ui,sans-serif;background:#f5f5f5;color:#20242a;max-width:1100px;margin:35px auto;padding:0 18px}article{background:white;padding:24px;margin:30px 0;border:1px solid #ddd;border-radius:10px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}img{width:100%;height:300px;object-fit:contain;background:#eee}table{border-collapse:collapse;width:100%;margin-top:20px}td,th{text-align:left;padding:12px;border:1px solid #ddd;vertical-align:top}small{color:#555}h2{font-size:20px}.question{font-weight:650}details{font-size:13px}</style></head><body><h1>Task 2 · complete final paired results</h1><p>All160 original final pairs in suite/ID order. No success/failure-based selection. References always belong to the original examples, including swapped and unrelated-image conditions. These are subset scores. Use your browser’s Find to locate an ID or question.</p>'+''.join(cards)+'</body></html>').encode('utf-8')
