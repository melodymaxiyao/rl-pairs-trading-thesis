from common import *
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def save(fig,name):
 import io
 fig.tight_layout();b=io.BytesIO();fig.savefig(b,format='png',dpi=300);plt.close(fig);publish(PLOTS/name,b.getvalue());count('figures_written')
def plots(ladder,policies,newdaily,prior):
 cells=['R_TR_aln','B3_TR_aln','TD8','TD8_tuned','TD16_align','TD16_strength','TD8_group']
 f=ladder[ladder.cost.eq(10)&ladder.era.eq('pooled')];fig,axes=plt.subplots(1,3,figsize=(13,4))
 for ax,pool in zip(axes,POOLS):
  g=f[f.pool.eq(pool)].set_index('cell').reindex(cells);ax.bar(range(len(g)),g.estimate,color=['#0072B2','#56B4E9','#E69F00','#009E73','#332288','#882255','#44AA99']);ax.errorbar(range(len(g)),g.estimate,yerr=np.array([g.estimate-g.ci95_low,g.ci95_high-g.estimate]),fmt='none',color='#444444',capsize=2);ax.set_xticks(range(len(g)),cells,rotation=65,ha='right');ax.set_xlabel(pool);ax.axhline(0,color='#777777',lw=.7)
 axes[0].set_ylabel('Mean net (bps/pair-year, 10 bps)');save(fig,'ladder_with_tuned.png')
 fig,axes=plt.subplots(1,3,figsize=(12,4))
 from sources import frozen
 for ax,pool in zip(axes,POOLS):
  old=[]
  for y in YEARS:
   q=frozen(f'outputs/models/{pool}_{y}_TD8')['ensemble'];act=q[:,1]>q[:,0];old.append(np.stack([act[::2],~act[1::2]]))
  a=policies[policies.pool.eq(pool)].groupby('bin')[['enter','hold']].mean().reindex(['<0.5','0.5–1.5','1.5–2.5','>=2.5']).to_numpy().T
  arr=np.vstack([np.mean(old,axis=0),a]);ax.imshow(arr,vmin=0,vmax=1,cmap='cividis',aspect='auto');ax.set_yticks(range(4),['TD8 enter','TD8 hold','Tuned enter','Tuned hold']);ax.set_xticks(range(4),['<0.5','0.5–1.5','1.5–2.5','>=2.5'],rotation=35);ax.set_xlabel(pool+' |d| bin')
  for i in range(4):
   for j in range(4):ax.text(j,i,f'{arr[i,j]:.2f}',ha='center',va='center',color='white' if arr[i,j]<.45 else 'black')
 save(fig,'policy_td8_vs_tuned.png')
 fig,ax=plt.subplots(figsize=(9,4));colors=['#0072B2','#56B4E9','#E69F00','#009E73']
 for cell,color in zip(['R_TR_aln','B3_TR_aln','TD8','TD8_tuned'],colors):
  data=newdaily if cell=='TD8_tuned' else prior;g=data[data.pool.eq('P_RAND')&data.cell.eq(cell)&data.cost.eq(10)&data.variant.eq('primary')&data.world.eq('real')].sort_values('date');ax.plot(pd.to_datetime(g.date),g.net_bps.cumsum(),label=cell,color=color)
 ax.set_xlabel('Test date');ax.set_ylabel('Cumulative net (bps, additive)');ax.legend();save(fig,'cumulative_portfolio.png')
