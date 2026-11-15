"""Plot frozen saved-result summaries. No generations or grading."""
import json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

LABELS={'FirstAns':'First answer','SC':'Self-consistency','L_max':'L max','L_RWC':'L RWC','K_max':'K max','K_RWC':'K RWC','shortest':'Shortest','V_max':'V max','V_RWC':'V RWC','Z_max':'VisConf max','Z_RWC':'VisConf RWC','Ztoken_max':'Token fusion max','Ztoken_RWC':'Token fusion RWC'}
COLORS={'FirstAns':'#777777','SC':'#222222','L_max':'#d99036','L_RWC':'#bd6c12','K_max':'#5597c5','K_RWC':'#176fa4','shortest':'#ad69a3','V_max':'#78a66d','V_RWC':'#3c852c','Z_max':'#d87878','Z_RWC':'#b33e43','Ztoken_max':'#af7dc4','Ztoken_RWC':'#80479b'}
BUDGETS=[1,2,4,8,16,32]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'figure.dpi':120,'savefig.dpi':180,'svg.fonttype':'none'})

def main(root):
 a=json.loads((root/'analysis.json').read_text());figs=root/'figures';figs.mkdir(exist_ok=True)
 def save(fig,name,caption):
  fig.savefig(figs/(name+'.png'),bbox_inches='tight');fig.savefig(figs/(name+'.svg'),bbox_inches='tight');plt.close(fig)
  (figs/(name+'.txt')).write_text(caption+'\n')
 def rows(dataset,method):return sorted([r for r in a['datasets'][dataset]['metrics'] if r['method']==method],key=lambda x:x['N'])
 def curve(ax,dataset,method,x='N',ci=False):
  r=rows(dataset,method);xx=[v[x] for v in r];yy=[v['accuracy_percent'] for v in r]
  ax.plot(xx,yy,marker='o',ms=4,color=COLORS[method],label=LABELS[method],linestyle='--' if method.endswith('_max') else '-')
  if ci:ax.fill_between(xx,[v['accuracy_low_percent'] for v in r],[v['accuracy_high_percent'] for v in r],color=COLORS[method],alpha=.07)
 def setup(ax,d):ax.set_title(d,weight='bold');ax.set_ylabel('Recorded accuracy (%)')
 fig,axs=plt.subplots(1,2,figsize=(13,5),layout='constrained')
 for ax,d in zip(axs,a['datasets']):
  methods=['FirstAns','SC','L_max','L_RWC','K_max','K_RWC','shortest'] if d=='MATH' else ['FirstAns','SC','K_RWC','V_RWC','Z_RWC','Ztoken_RWC']
  for m in methods:curve(ax,d,m,ci=m=='SC')
  ax.plot(BUDGETS,a['datasets'][d]['oracle_percent'],'k:',label='Pool oracle (diagnostic)');ax.set_xscale('log',base=2);ax.set_xticks(BUDGETS,BUDGETS);ax.set_xlabel('Revealed rollout budget N');setup(ax,d);ax.legend(fontsize=8,ncol=3,loc='upper center',bbox_to_anchor=(.5,-.17))
 fig.suptitle('More candidates raise the oracle; selection determines realized accuracy',fontsize=13)
 save(fig,'01_accuracy_budget','800 held-out questions per dataset; 10 nested saved orders averaged within each question. Oracle: at least one recorded correct answer in N candidates, not an implementable method. First answer consumes one candidate at every displayed N. Shading: pointwise 95% question bootstrap for SC only; all method intervals in CSV. Recorded bounded-scorer labels; MATH comparison timeouts on test:4928.')
 fig,axs=plt.subplots(1,2,figsize=(13,5),layout='constrained')
 for ax,d in zip(axs,a['datasets']):
  methods=['SC','K_RWC','L_RWC'] if d=='MATH' else ['SC','K_RWC','V_RWC','Z_RWC']
  for m in methods:curve(ax,d,m,x='mean_token_cost')
  r=rows(d,'FirstAns')[0];ax.scatter(r['mean_token_cost'],r['accuracy_percent'],c=COLORS['FirstAns'],s=45,label='First answer',zorder=4)
  for fam,col,marker in [('agreement','#ef9632','D'),('quality','#ad3f91','s')]:
   r=rows(d,'adaptive_'+fam)[0];ax.errorbar(r['mean_token_cost'],r['accuracy_percent'],xerr=[[r['mean_token_cost']-r['token_cost_low']],[r['token_cost_high']-r['mean_token_cost']]],yerr=[[r['accuracy_percent']-r['accuracy_low_percent']],[r['accuracy_high_percent']-r['accuracy_percent']]],fmt=marker,color=col,ms=7,capsize=2,label='Adaptive '+fam,zorder=6)
  setup(ax,d);ax.set_xlabel('Mean retained generated tokens / question');ax.legend(fontsize=8,ncol=2,loc='lower right')
 fig.suptitle('Accuracy versus observed token cost',fontsize=13)
 save(fig,'02_accuracy_cost','Fixed curves connect observed N=1,2,4,8,16,32 points as a visual guide, not interpolated evidence. Adaptive thresholds frozen using development data. Bars: separate pointwise 95% question-bootstrap accuracy and mean-cost intervals, not joint regions. Offline retained-token cost excludes prompt processing, scoring and attention collection; it is a compute proxy, not wall time. See adaptive_cost_comparisons in analysis.json for bracketing observed SC budgets.')
 fig,axs=plt.subplots(1,2,figsize=(13,6.3),layout='constrained')
 for ax,d in zip(axs,a['datasets']):
  rr=[r for r in a['datasets'][d]['metrics'] if r['N']==32];y=np.arange(len(rr));correct=np.array([r['accuracy_percent'] for r in rr]);oracle=np.array([r['consumed_oracle_percent'] for r in rr]);miss=oracle-correct
  ax.barh(y,correct,color='#4b8f76',label='Selected correct');ax.barh(y,miss,left=correct,color='#e2ab54',label='Correct available; selection missed');ax.barh(y,100-oracle,left=oracle,color='#bbbbbb',label='No correct consumed candidate');ax.set_yticks(y,[LABELS[r['method']] for r in rr]);ax.invert_yaxis();ax.set_xlim(0,100);ax.set_xlabel('Question × saved-order observations (%)');ax.set_title(d,weight='bold');ax.grid(axis='y',visible=False)
 fig.legend(handles=[plt.Rectangle((0,0),1,1,color=c) for c in ['#4b8f76','#e2ab54','#bbbbbb']],labels=['Selected correct','Correct available; selection missed','No correct consumed candidate'],loc='outside lower center',ncol=3,fontsize=9)
 fig.suptitle('Where errors arise at budget 32',fontsize=13)
 save(fig,'03_failure_decomposition','All percentages average ten orders within each of 800 questions. Categories partition consumed-candidate outcomes. First answer consumes only one, so its available-correct fraction equals its accuracy. Other methods consume all32. Labels use the frozen bounded scorer, including the reported MATH timeout case.')
 fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
 for ax,d in zip(axs,a['datasets']):
  rr=[r for r in a['datasets'][d]['signal_summaries'] if r['statistic']=='quartile_gap'];means=np.array([r['mean']*100 for r in rr]);lo=np.array([r['low']*100 for r in rr]);hi=np.array([r['high']*100 for r in rr]);y=np.arange(len(rr))
  ax.errorbar(means,y,xerr=[means-lo,hi-means],fmt='o',capsize=3,color='#176fa4');ax.axvline(0,color='#666',lw=.8);ax.set_yticks(y,[{'Z':'VisConf','Ztoken':'Token fusion','length':'Length (higher = longer)'}.get(r['score'],r['score']) for r in rr]);ax.invert_yaxis();ax.set_title(d,weight='bold');ax.set_xlabel('Top-quartile minus bottom-quartile accuracy (pp)');ax.grid(axis='y',visible=False)
 fig.suptitle('Does a higher signal identify more accurate rollouts within a question?',fontsize=13)
 save(fig,'04_signal_quartiles','N32 signals computed within each question, using equal fractional weights at tied quartile boundaries. Plot macro-averages800 question gaps. Error bars: pointwise 95% question bootstrap. Higher length means longer response, not the shortest-response selector. Associations do not establish causation; Spearman correctness/length associations and undefined counts are in signal_associations.csv.')
 fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
 for ax,aggregation in zip(axs,['max','RWC']):
  for s in ['K','V','Z','Ztoken']:curve(ax,'MathVista',s+'_'+aggregation)
  curve(ax,'MathVista','SC');ax.set_title('MathVista · '+aggregation,weight='bold');ax.set_xscale('log',base=2);ax.set_xticks(BUDGETS,BUDGETS);ax.set_xlabel('Revealed rollout budget N');ax.set_ylabel('Recorded accuracy (%)');ax.legend(fontsize=8,ncol=3,loc='upper center',bbox_to_anchor=(.5,-.17))
 fig.suptitle('Rollout-level fusion versus tokenwise fusion',fontsize=13)
 save(fig,'05_fusion','VisConf = mean(c) × mean(v); token fusion = mean(c × v); their difference is population covariance(c,v). Confidence c uses Q05/Q95 from the current revealed candidate prefix only and is recalibrated for every N/order. See paired_accuracy_differences.csv for paired fusion contrasts with pointwise95% intervals. All results800 held-out MathVista questions, ten orders each.')
 fig,axs=plt.subplots(1,2,figsize=(12,4.3),layout='constrained')
 for ax,d in zip(axs,a['datasets']):
  names=['Agreement','Quality-aware','Always 32'];bottom=np.zeros(3)
  for n,c in zip([4,8,16,32],['#87b4a0','#4e8f91','#4b779b','#364661']):
   values=np.array([a['datasets'][d]['adaptive'][fam]['stopping_counts'][str(n)]/80 for fam in ['agreement','quality']]+[100. if n==32 else 0.]);ax.bar(names,values,bottom=bottom,color=c,label=str(n));bottom+=values
  assert np.allclose(bottom,100);ax.set_ylim(0,105);ax.set_title(d,weight='bold');ax.set_ylabel('Question × saved-order observations (%)');ax.grid(axis='x',visible=False)
 fig.legend(*axs[0].get_legend_handles_labels(),title='Rollouts consumed',loc='outside lower center',ncol=4);fig.suptitle('Frozen controllers often stop at the first check',fontsize=13)
 save(fig,'06_stopping','Stopping counts over8000 question-order observations per dataset. Policies may check after4,8,16; otherwise32. MATH agreement tau=.625; quality tau=.5,gamma=0. MathVista agreement tau=.625; quality tau=.625,gamma=0. Chosen using development data only. Both output current self-consistency. Orders within a question are dependent; uncertainty elsewhere resamples questions.')
 trajectories=json.loads((root/'fixed_trajectories.json').read_text());mapping={(r['sample_id'],r['rollout_id']):r for r in trajectories};ordered_ids=['99','425','601','77'];assert {r['sample_id'] for r in trajectories}==set(ordered_ids)
 fig,axs=plt.subplots(4,2,figsize=(13,10),layout='constrained',sharey=True)
 for i,q in enumerate(ordered_ids):
  for j,rid in enumerate([0,1]):
   r=mapping[(q,rid)];ax=axs[i,j];x=np.arange(1,len(r['kl'])+1);ax.plot(x,r['calibrated_confidence'],color='#176fa4',lw=.8,alpha=.85);ax.plot(x,r['vision'],color='#b33e43',lw=.9,alpha=.9);ax.set_title(f'MathVista {q} · rollout {rid} · {len(x)} retained tokens',fontsize=10);ax.set_ylim(-.03,1.03)
   if j==0:ax.set_ylabel('Normalized signal')
   if i==3:ax.set_xlabel('Retained generated-token index')
 fig.legend(handles=[Line2D([0],[0],color=c) for c in ['#176fa4','#b33e43']],labels=['Calibrated confidence c','Vision allocation v'],loc='outside lower center',ncol=2);fig.suptitle('Eight trajectories fixed before inspecting outcomes',fontsize=13)
 save(fig,'07_trajectories','Rollouts0/1 from the first four saved MathVista evaluation IDs, selected before performance analysis. Full32 candidate-prefix calibration in the first saved order; no token smoothing/downsampling. v = image/(image+prompt_text+generated_text) attention. c and v share a0–1 scale but measure different quantities. Source tokens/raw KL and full arrays in fixed_trajectories.json/source export.')
 fig,ax=plt.subplots(figsize=(10,3.8),layout='constrained');rr=a['datasets']['MathVista']['vision_summaries'];y=np.arange(4);means=np.array([r['mean'] for r in rr]);lo=np.array([r['low'] for r in rr]);hi=np.array([r['high'] for r in rr]);names=['Within rollout: calibrated c vs v','Within rollout: raw KL vs v','Within question: K vs V','Within question: C vs V'];ax.errorbar(means,y,xerr=[means-lo,hi-means],fmt='o',capsize=3,color='#80479b');ax.axvline(0,color='#666',lw=.8);ax.set_yticks(y,[name+f"  (n={r['valid_questions']} questions)" for name,r in zip(names,rr)]);ax.invert_yaxis();ax.set_xlim(-.5,.05);ax.set_xlabel('Macro-mean Pearson correlation');ax.grid(axis='y',visible=False);ax.set_title('Confidence and vision allocation covary negatively on average',weight='bold')
 save(fig,'08_confidence_vision','Within-rollout Pearson correlations first averaged over32 rollouts per question, then questions. Raw KL and prefix-calibrated confidence shown separately;25600 valid rollout correlations. Within-question K/C vsV correlations use32 rollout-level means;795 valid questions,5 undefined. Undefined correlations excluded, not set to0. Bars pointwise95% question bootstrap. Negative association is descriptive, not evidence that vision attention harms correctness.')
 print(json.dumps({'figures':8,'formats':['png','svg'],'matplotlib':matplotlib.__version__,'numpy':np.__version__}))
if __name__=='__main__':main(Path(sys.argv[1]))
