from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
with (ROOT/"results/primary_metrics.csv").open() as f: rows=list(csv.DictReader(f))
conditions=list(dict.fromkeys(r["condition"] for r in rows))
labels=[next(r["label"] for r in rows if r["condition"]==c) for c in conditions]
benchmarks=["GSM8K","HellaSwag","BoolQ","ARC-Easy"]
colors=["#657383","#4176a4","#24557e","#b16a32","#7d6b9f","#4f8f76","#a44654"]
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
fig,axs=plt.subplots(1,4,figsize=(13.5,4.5),sharey=True)
for ax,benchmark in zip(axs,benchmarks):
    vals=[float(next(r["delta_from_original_pp"] for r in rows if r["condition"]==c and r["benchmark"]==benchmark)) for c in conditions]
    ax.barh(range(7),vals,color=colors,height=.65)
    ax.axvline(0,color="#777777",lw=.8)
    ax.set_title(benchmark,fontweight="bold")
    ax.set_xlim(min(-8,min(vals)-5),max(vals)+5)
    ax.set_yticks(range(7),labels)
    ax.set_xlabel("Change from original (pp)")
    ax.grid(axis="x",alpha=.18);ax.set_axisbelow(True)
    for j,v in enumerate(vals):ax.text(v+(.35 if v>=0 else -.35),j,f"{v:+.2f}",ha="left" if v>=0 else "right",va="center",fontsize=9)
axs[0].invert_yaxis()
fig.suptitle("Task 1: recorded benchmark changes",x=.03,ha="left",fontsize=16,fontweight="bold")
fig.text(.03,.04,"Source: saved completion receipts. Single runs; final per-example audit unfinished.\nGSM8K: flexible EM. HellaSwag / ARC-Easy: normalized accuracy. BoolQ: accuracy.",fontsize=9,color="#444444")
fig.subplots_adjust(left=.16,right=.98,top=.82,bottom=.24,wspace=.23)
out=ROOT/"figures";out.mkdir(exist_ok=True)
for suffix in ["png","svg"]:fig.savefig(out/("benchmark_changes."+suffix),dpi=180)
plt.close(fig)
