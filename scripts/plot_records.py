"""Regenerate the historical training figure without loading a VLA or using a GPU."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from project_paths import REPO, ROOT

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir',type=Path,default=ROOT/'figures')
args=parser.parse_args()
args.output_dir.mkdir(parents=True,exist_ok=True)

def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines()]

pilot=REPO/'records/step500'
continued=REPO/'records/step2000'
metrics=rows(pilot/'training-metrics.jsonl')+rows(continued/'training-metrics.jsonl')
old=rows(pilot/'fixed-losses.jsonl')
new=rows(continued/'fixed-losses.jsonl')
steps=np.array([r['step'] for r in metrics])
losses=np.array([r['loss'] for r in metrics])
fig,axes=plt.subplots(1,3,figsize=(13.5,3.9),layout='constrained')
axes[0].plot(steps,losses,color='#a2bcd0',alpha=.5,lw=.6,label='Per update')
window=50
axes[0].plot(steps[window-1:],np.convolve(losses,np.ones(window)/window,mode='valid'),color='#176497',lw=2,label='50-update mean')
axes[0].axvline(500,color='#777777',ls='--',lw=1,label='Resume at 500')
axes[0].set(xlabel='Cumulative optimizer updates',ylabel='Flow matching loss',title='Training loss (batch size 1)')
axes[0].legend(frameon=False,fontsize=8)
axes[1].plot(steps,[r['learning_rate'] for r in metrics],color='#267f5d',lw=1.8)
axes[1].axvline(500,color='#777777',ls='--',lw=1)
axes[1].set(xlabel='Cumulative optimizer updates',ylabel='Learning rate',title='Original schedule, no warm restart')
phases=[(0,old,'before'),(500,old,'after'),(2000,new,'step2000')]
samples={step:{r['task']:r['loss'] for r in source if r['phase']==phase and r['split']=='validation'} for step,source,phase in phases}
for task in samples[0]:
    axes[2].plot([0,500,2000],[samples[s][task] for s in [0,500,2000]],color='#b8bec5',marker='o',ms=3,lw=.8,alpha=.7)
means=[np.mean(list(samples[s].values())) for s in [0,500,2000]]
axes[2].plot([0,500,2000],means,color='#9f3c46',marker='o',lw=2.5,label='Mean of 10 frames')
axes[2].set(xlabel='Cumulative optimizer updates',ylabel='Fixed-noise flow loss',title='Held-out trajectory midpoints',xticks=[0,500,2000])
axes[2].legend(frameon=False,fontsize=8)
for ax in axes:
    ax.grid(alpha=.18)
    ax.spines[['top','right']].set_visible(False)
target=args.output_dir/'training-2000.png'
fig.savefig(target,dpi=180)
plt.close(fig)
print('FIGURE_SAVED',target)
