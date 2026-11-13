"""Scientific output figures, 300 dpi, with one y-axis per panel."""
from common import *
from jobs import CELLS
import numpy as np,pandas as pd
def render(ladder,diag,traj,deploy,daily,old_daily,differences,new,refs):
 import matplotlib.pyplot as plt
 palette={'R_TR_aln':'#555555','B3_TR_aln':'#88CCEE','TD8':'#AA4499','TD8_C01g1':'#EE7733','TD8_tuned':'#0077BB','SPI_all':'#009988','SPI_TD':'#CC3311'}
 def save(fig,name):
  fig.tight_layout();buffer=io.BytesIO();fig.savefig(buffer,format='png',dpi=300,metadata={'Software':HERE.name});publish(PLOTS/(name+'.png'),buffer.getvalue());plt.close(fig);count('figures_created');checkpoint_guard()
 order=['R_TR_aln','B3_TR_aln','TD8','TD8_C01g1','TD8_tuned','SPI_all','SPI_TD']
 fig,axes=plt.subplots(1,3,figsize=(14,5),sharey=True)
 for ax,pool in zip(axes,POOLS):
  f=ladder.loc[ladder.pool.eq(pool)&ladder.cost.eq(10)&ladder.era.eq('pooled')].set_index('cell').reindex(order)
  for i,(cell,row) in enumerate(f.iterrows()):
   ax.plot([row.ci95_low,row.ci95_high],[i,i],color=palette[cell]);ax.scatter(row.estimate,i,color=palette[cell],s=25)
  ax.axvline(0,color='#888888',lw=.6);ax.set_yticks(range(len(order)),order);ax.set_title(pool);ax.set_xlabel('Net bps per pair-year; 95% year-block CI')
 save(fig,'01_ladder')
 counts=[]
 for pool in POOLS:
  for cell in ['TD8','TD8_tuned',*CELLS]:
   f=new.loc[new.pool.eq(pool)&new.cost.eq(10)&new.cell.eq(cell)] if cell in CELLS else refs.loc[refs.pool.eq(pool)&refs.cost.eq(10)&refs.world.eq('real')&refs.cell.eq(cell)]
   counts.append(dict(pool=pool,cell=cell,never=int(f.groupby('test_year').n_trades.sum().eq(0).sum())))
 f=pd.DataFrame(counts);fig,axes=plt.subplots(1,3,figsize=(14,4),sharey=True)
 for ax,pool in zip(axes,POOLS):
  g=f[f.pool.eq(pool)];ax.bar(g.cell,g.never,color=[palette[c] for c in g.cell]);ax.set_title(pool);ax.set_ylim(0,22);ax.tick_params(axis='x',rotation=40);ax.set_ylabel('Never-entering folds / 22')
 save(fig,'02_never_entering')
 fig,axes=plt.subplots(2,2,figsize=(13,7));v=diag.loc[diag.pool.eq('P_RAND')&diag.seed.eq('ensemble')]
 for i,cell in enumerate(['TD8_C01g1','TD8_tuned']):
  for j,position in enumerate(['flat','in-trade']):
   f=v.loc[v.cell.eq(cell)&v.position.eq(position)].drop_duplicates(['test_year','bin']).pivot(index='bin',columns='test_year',values='delta_Q').reindex(index=range(4),columns=YEARS)
   limit=max(float(np.nanmax(np.abs(f.to_numpy()))),1e-12);im=axes[i,j].imshow(f,aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit)
   axes[i,j].set_title(cell+' / '+('enter − stay' if position=='flat' else 'hold − exit'));axes[i,j].set_xlabel('Test year');axes[i,j].set_ylabel('Absolute-z bin');axes[i,j].set_xticks(range(0,22,3),YEARS[::3],rotation=45);fig.colorbar(im,ax=axes[i,j],label='Q difference (bps)')
 save(fig,'03_Q_heatmaps')
 fig,axes=plt.subplots(3,2,figsize=(13,10))
 for row,pool in enumerate(POOLS):
  for cell in ['TD8_C01g1','TD8_tuned']:
   f=traj.loc[traj.pool.eq(pool)&traj.cell.eq(cell)];wide=f.pivot_table(index='epoch',columns=['seed','state','action'],values='Q')
   # Carry stopped seeds' final values for display only, never for training.
   wide=wide.ffill()
   for col,position in enumerate(['flat','in-trade']):
    states=range(0,8,2) if position=='flat' else range(1,8,2);values=[]
    for state in states:
     a=wide.xs((state,0),level=['state','action'],axis=1);b=wide.xs((state,1),level=['state','action'],axis=1)
     values.append((b-a if position=='flat' else a-b).mean(axis=1))
    series=pd.concat(values,axis=1).mean(axis=1);axes[row,col].plot(series.index,series,color=palette[cell],label=cell)
   axes[row,0].set_ylabel(pool+' mean enter − stay Q');axes[row,1].set_ylabel(pool+' mean hold − exit Q')
  for ax in axes[row]:ax.set_xlabel('Epoch; stopped seed final values carried for display');ax.legend()
 fig.suptitle('Fixed 2004 fold: tuned trace is an exact acceptance reproduction',y=1.01);save(fig,'04_Q_trajectories')
 fig,axes=plt.subplots(1,2,figsize=(14,5))
 g=deploy.loc[deploy.pool.eq('P_RAND')&deploy.cost.eq(10)&deploy.primary_lag&deploy.learner.eq('SPI_all')&deploy.world.eq('real')].sort_values('test_year')
 values=np.array([[(int(r.deployed_bits)>>b)&1 for b in range(10)] for r in g.itertuples()]).T
 axes[0].imshow(values,aspect='auto',vmin=0,vmax=1,cmap='Greens');axes[0].set_yticks(range(10),['E0','E1','E2','E3','K2','K3','X0','X1','X2','X3']);axes[0].set_xticks(range(0,len(g),3),g.test_year.iloc[::3],rotation=45);axes[0].set_title('P_RAND real deployed switches')
 rates=[]
 for pool in POOLS:
  f=deploy.loc[deploy.pool.eq(pool)&deploy.cost.eq(10)&deploy.primary_lag&deploy.learner.eq('SPI_all')]
  rates.append([f.loc[f.world.eq(w),'deployed_bits'].ne(0).mean() for w in ['real','N2']])
 x=np.arange(3);a=np.array(rates);axes[1].bar(x-.18,a[:,0],.36,label='real');axes[1].bar(x+.18,a[:,1],.36,label='N2 (both replicates)');axes[1].axhline(.1,color='#555555',ls='--',lw=.8);axes[1].set_xticks(x,POOLS);axes[1].set_ylabel('Share deploying ≥1 switch');axes[1].legend();save(fig,'05_SPI_acceptance')
 fig,ax=plt.subplots(figsize=(12,5))
 for cell in ['R_TR_aln','TD8_tuned','SPI_all','SPI_TD']:
  f=daily.loc[daily.pool.eq('P_RAND')&daily.cost.eq(10)&daily.cell.eq(cell)&daily.world.eq('real')] if cell in CELLS else old_daily.loc[old_daily.pool.eq('P_RAND')&old_daily.cost.eq(10)&old_daily.cell.eq(cell)&old_daily.variant.eq('primary')]
  f=f.sort_values('date');ax.plot(pd.to_datetime(f.date),f.net_bps.cumsum(),label=cell,color=palette[cell])
 ax.set_ylabel('Cumulative portfolio net (bps)');ax.legend();save(fig,'06_cumulative')
 fig,ax=plt.subplots(figsize=(12,5));f=differences.loc[differences.pool.eq('P_RAND')&differences.cost.eq(10)].pivot(index='test_year',columns='cell',values='delta_vs_L0');x=np.arange(len(f))
 ax.bar(x-.2,f.SPI_all,.4,label='SPI_all − L0',color=palette['SPI_all']);ax.bar(x+.2,f.TD8_tuned,.4,label='TD8_tuned − L0',color=palette['TD8_tuned']);ax.axhline(0,color='#555555',lw=.7);ax.set_xticks(x,f.index,rotation=45);ax.set_ylabel('Annual IPW pair-year difference (bps)');ax.legend();save(fig,'07_annual_variance')
