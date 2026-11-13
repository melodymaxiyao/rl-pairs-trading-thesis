from common import *
import numpy as np,pandas as pd

def make(wide,pi,persist,dose,power,hindsight):
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
 def save(fig,name):
  fig.tight_layout();p=PLOTS/name
  if not p.exists():
   import io
   b=io.BytesIO();fig.savefig(b,format='png',dpi=300,metadata={'Software':'residual_learnability_v1'});publish(p,b.getvalue())
  plt.close(fig)
 def errors(ax,g,x='decile',label=None):
  g=g.sort_values(x);a=g.estimate.to_numpy(float);lo=g.ci95_low.to_numpy(float);hi=g.ci95_high.to_numpy(float);good=np.isfinite(a)&np.isfinite(lo)&np.isfinite(hi)
  ax.plot(g[x],a,'o-',label=label);ax.fill_between(g[x].to_numpy(float)[good],lo[good],hi[good],alpha=.15);ax.axhline(0,color='grey',lw=.6)
 f=wide[wide.pool.eq('P_RAND')&wide.window.eq('T')];eras=sorted(f.era.unique());fig,axes=plt.subplots(2,len(eras),figsize=(15,6),squeeze=False)
 for i,name in enumerate(['ADF_c','ADF_ct']):
  for j,e in enumerate(eras):
   for world in ['real','N2']:
    g=f.loc[f.era.eq(e)&f.world.eq(world)];x=g[name+'__statistic'].to_numpy(float);weights=g.weight.to_numpy();ok=np.isfinite(x);axes[i,j].hist(x[ok],weights=weights[ok],bins=60,density=True,alpha=.45,label=world)
   axes[i,j].set_title(name+' / '+e);axes[i,j].legend()
 save(fig,'01_adf_distributions.png')
 fig,ax=plt.subplots(figsize=(10,4));p=pi[pi.pool.eq('P_RAND')&pi.group.astype(str).isin(map(str,YEARS))]
 for name in ['ADF_c','ADF_ct','theta_21']:
  g=p[p.statistic.eq(name)].copy();g['year']=g.group.astype(int);g=g.sort_values('year');ax.plot(g.year,g.one_minus_pi0,'o-',label=name)
 ax.set(xlabel='Fold test year',ylabel='1 - Storey pi0',title='Per-fold points; year-block CIs reported for eras and pooled estimates');ax.legend();save(fig,'02_pi0_by_fold.png')
 fig,axes=plt.subplots(1,2,figsize=(9,4));g=persist[persist.kind.eq('B_pass_given_A')&persist.pool.eq('P_RAND')]
 for ax,screen in zip(axes,['RC','RP']):
  q=g[g.screen.eq(screen)];labels=[];values=[];low=[];high=[]
  for world in ['real','N2']:
   for passing in [True,False]:
    r=q[q.world.eq(world)&q.A_pass.eq(passing)].iloc[0];labels.append(world+' / A '+('pass' if passing else 'fail'));values.append(r.estimate);low.append(r.ci95_low);high.append(r.ci95_high)
  ax.bar(range(4),values);ax.vlines(range(4),low,high,color='black');ax.set_xticks(range(4),labels,rotation=25,ha='right');ax.set(title=screen,ylabel='B pass probability')
 save(fig,'03_persistence.png')
 fig,axes=plt.subplots(2,2,figsize=(11,7));d=dose[dose.pool.eq('P_RAND')]
 for ax,(name,g) in zip(axes.flat,d.groupby('statistic',sort=True)):
  for cell,q in g.groupby('cell'):errors(ax,q,label=cell)
  ax.set(title=name,xlabel='Train statistic decile',ylabel='real - N2 net (bps/pair-year)');ax.legend(fontsize=7)
 save(fig,'04_dose_response.png')
 fig,axes=plt.subplots(1,2,figsize=(12,4))
 for ax,kind in zip(axes,['nominal_5','N2_calibrated_structure_5']):
  for name,g in power[power.calibration.eq(kind)].groupby('statistic'):
   ax.plot(g.w,g.rejection_rate,'o-',label=name)
  ax.set(title=kind,xlabel='w (k=4 capped if registered)',ylabel='Rejection rate');ax.legend(fontsize=7)
 save(fig,'05_power.png')
 fig,ax=plt.subplots(figsize=(8,4));h=hindsight[hindsight.kind.eq('theta21_decile_vs_real_net')]
 for cell,g in h.groupby('cell'):errors(ax,g,label=cell)
 ax.set(xlabel='Test-year theta21 decile',ylabel='Real net (bps/pair-year)',title='Hindsight, not a screen');ax.legend();save(fig,'06_hindsight.png')
