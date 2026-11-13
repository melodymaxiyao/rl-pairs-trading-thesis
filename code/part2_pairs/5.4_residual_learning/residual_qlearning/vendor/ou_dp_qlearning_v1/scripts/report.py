"""Descriptive frozen predictions, tables and publication export PNGs."""
from common import *
import pandas as pd
from scipy.special import erfi
from scipy.optimize import brentq
def csv_publish(path,frame):publish(path,frame.to_csv(index=False,float_format='%.17g',lineterminator='\n'))
def predictions(cells,tier):
 c=cells.query('tier == 1');means=c.groupby(['h','P','learner']).mean(numeric_only=True);rows=[]
 pow=c.query("learner == 'Q_POW'");o1details=[]
 for h in [5,10,20,40]:
  a=pow.query('h == @h and P == 100');b=pow.query('h == @h and P == 600');o1details.append(dict(h=h,regret_P100=a.regret_oracle.mean(),regret_P600=b.regret_oracle.mean(),equality_P600=b.raw_equals_oracle.mean(),physical_equality_P600=b.physical_equals_oracle.mean()))
 o1=all(d['regret_P600']<d['regret_P100'] and (d['h']>20 or d['equality_P600']>=.5) for d in o1details)
 rows.append(dict(id='O1',status='supported' if o1 else 'not supported',details=json.dumps(clean(o1details))))
 differences=[]
 for h in [5,10,20,40]:
  for P in [100,600]:differences.append(dict(h=h,P=P,plugin_minus_pow=float(means.loc[(h,P,'PlugIn-8'),'true_value']-means.loc[(h,P,'Q_POW'),'true_value'])))
 o2=all(d['plugin_minus_pow']>0 for d in differences) and all(next(d['plugin_minus_pow'] for d in differences if d['h']==h and d['P']==100)>=next(d['plugin_minus_pow'] for d in differences if d['h']==h and d['P']==600) for h in [5,10,20,40])
 rows.append(dict(id='O2',status='supported' if o2 else 'not supported',details=json.dumps(differences)))
 base=next(d['plugin_minus_pow'] for d in differences if d['h']==20 and d['P']==600);gaps=[]
 if tier>=2:
  for arm,w in [('W',.3),('K',0.)]:
   a=cells.query('tier == 2 and arm == @arm and w == @w').groupby('learner').true_value.mean();gaps.append(dict(arm=arm,w=w,gap=float(a['PlugIn-8']-a['Q_POW']),correct_gap=base))
 rows.append(dict(id='O3',status=('supported' if any(d['gap']<base or d['gap']<0 for d in gaps) else 'not supported') if tier>=2 else 'not evaluated: Tier2 pending',details=json.dumps(gaps)))
 constant=c.query("h == 20 and P == 600 and learner == 'Q_C01'");fraction=float(constant.no_entry_policy.mean());rows.append(dict(id='O4',status='supported' if fraction>=.8 else 'not supported',details=json.dumps(dict(no_entry_world_fraction=fraction,threshold=.8))))
 ratios=[]
 for h in [5,10,20]:
  for P in [100,600]:
   opt=float(means.loc[(h,P,'OPT-grid'),'true_value']);ora=float(means.loc[(h,P,'Oracle-8'),'true_value']);ratios.append(dict(h=h,P=P,gap_fraction=(opt-ora)/opt if opt>0 else None,OPT=opt,Oracle8=ora))
 supported=all(d['gap_fraction'] is not None and d['gap_fraction']<=.15 for d in ratios);rows.append(dict(id='O5',status='supported' if supported else 'not supported or undefined nonpositive OPT',details=json.dumps(ratios)))
 rules=c.query("learner in ['AL rule','L0-type rule']").groupby(['h','P','learner']).regret_oracle.mean().reset_index().to_dict('records');rows.append(dict(id='O6',status='descriptive; no direction predicted',details=json.dumps(rules)))
 return pd.DataFrame(rows)
def bertram(h,sd,cost):
 # Symmetric continuous-time no-stop reward/time: (2*b-c)/erfi(b/(sqrt(2)*sd)).
 lo=cost/(2*sd)+1e-8
 f=lambda z:2*sd*erfi(z/np.sqrt(2))-(2*sd*z-cost)*np.sqrt(2/np.pi)*np.exp(z*z/2)
 z=brentq(f,lo,max(6.,lo+3.));return float(z*sd)
def make_plots(cells,validation,worlds,out):
 import matplotlib.pyplot as plt,io
 def finish(fig,name):
  buffer=io.BytesIO();fig.savefig(buffer,format='png',dpi=300,bbox_inches='tight',metadata={'Software':'ou_dp_qlearning_v1'});plt.close(fig);publish(out/name,buffer.getvalue())
 c=cells.query('tier == 1');fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True)
 for ax,P in zip(axes,[100,600]):
  for learner in spec()['learners']:
   d=c.query('P == @P and learner == @learner').groupby('h').regret_oracle.mean();ax.plot(d.index,d.values,marker='o',label=learner)
  ax.set(xlabel='Half-life (sessions)',ylabel='Oracle-8 regret (bps / pair-year)',title=f'P = {P}')
 axes[1].legend(fontsize=7);finish(fig,'regret_by_half_life.png')
 d=cells.query("h == 20 and P == 600 and cost == 10 and learner in ['PlugIn-8','Q_POW']").groupby(['arm','w','learner']).true_value.mean().unstack('learner');gaps=d['PlugIn-8']-d['Q_POW'];fig,ax=plt.subplots(figsize=(7,4));ax.bar([f'{a} w={w:g}' if a=='W' else a for a,w in gaps.index],gaps.values);ax.axhline(0,color='black',lw=.7);ax.set(ylabel='PlugIn-8 minus Q_POW true value (bps / pair-year)');finish(fig,'plugin_minus_Q_POW.png')
 w=next(w for w in worlds if w['job']['tier']==1 and w['job']['h']==20 and w['job']['P']==600);p=w['policies'];opt=p['OPT-grid'];x=opt['x']/opt['scale'];fig,axes=plt.subplots(1,2,figsize=(10,4));axes[0].imshow(opt['actions'][:,1,:].T,origin='lower',aspect='auto',extent=[252,1,x[0],x[-1]],vmin=-1,vmax=1,cmap='coolwarm');axes[0].set(xlabel='Days to go',ylabel='True standardized spread',title='OPT-grid: target from flat');from dp import policy_array
 actions=policy_array(opt['x'],p['Oracle-8']['scale'],opt['scale'],bits=p['Oracle-8']['bits']);axes[1].plot(x,actions[1]);axes[1].set(xlabel='True standardized spread',ylabel='Target from flat',title='Oracle-8 bin policy');finish(fig,'policy_map.png')
 q=validation.query("type == 'Q_projection' and h == 20 and P == 600 and tier == 1");fig,ax=plt.subplots(figsize=(5,5));ax.scatter(q.exact_Q,q.learned_Q,s=9,alpha=.4);v=np.r_[q.exact_Q,q.learned_Q];v=v[np.isfinite(v)];lims=[v.min(),v.max()] if len(v) else [-1,1];ax.plot(lims,lims,color='black',lw=.8);ax.set(xlabel='Occupancy-projected finite-horizon Q',ylabel='Q_POW learned Q (bps)');finish(fig,'Q_values.png')
def report(worlds,tier,base=None,plots=True):
 base=RUN if base is None else inside(base);tables=base/'outputs/tables';plots_path=base/'outputs/plots';tables.mkdir(parents=True,exist_ok=True);plots_path.mkdir(parents=True,exist_ok=True)
 cells=pd.DataFrame([r for w in worlds for r in w['rows']]);parameters=pd.DataFrame([w['parameters'] for w in worlds]);policies={__import__('jobs').identifier(w['job']):w['policies'] for w in worlds};validation=pd.DataFrame([r for w in worlds for r in w['validation']])
 # Thresholds are the first-day flat entry boundary on the positive side.
 thresholds=[]
 for w in worlds:
  if w['job']['tier']!=1:continue
  p=w['policies']['OPT-grid'];x=p['x'];active=(x>0)&(p['actions'][0,1]!=0);entry=float(x[active].min()) if active.any() else np.nan
  # For a long entered below zero, the upper hold boundary is its reversion exit.
  hold=(abs(x)<4*p['scale'])&(p['actions'][0,2]==1);upper=float(x[hold].max()) if hold.any() else np.nan;after=np.flatnonzero((x>upper)&(abs(x)<4*p['scale'])&(p['actions'][0,2]==0));exit_long=float(x[after[0]]) if len(after) else np.nan;analytic=bertram(w['job']['h'],p['scale'],w['job']['cost'])
  thresholds.append(dict(w['job'],type='Bertram_threshold',OPT_entry_bps=entry,OPT_long_exit_bps=exit_long,Bertram_entry_bps=analytic,Bertram_long_exit_bps=analytic,entry_difference_bps=entry-analytic,exit_difference_bps=exit_long-analytic,objective_note='finite-year stop-constrained total reward vs symmetric continuous no-stop long-run reward/time; flat entry at -b for long, symmetric exit at +b'))
 validation=pd.concat([validation,pd.DataFrame(thresholds)],ignore_index=True);pred=predictions(cells,tier)
 for name,frame in [('cells',cells),('parameters',parameters),('validation',validation),('predictions',pred)]:csv_publish(tables/(name+'.csv'),frame)
 from report_artifacts import stream_policy_json
 stream_policy_json(tables/'policies.json',policies)
 if plots:make_plots(cells,validation,worlds,plots_path)
 lines=[f"{r.id}: {r.status}; {r.details}" for r in pred.itertuples()]
 text='Synthetic OU results\n\n'+f'Completed through Tier {tier}: {len(worlds)} worlds, {len(cells)} learner rows. No real-data trials.\n\n'
 text+='Q1 (empirical policy convergence): '+lines[0]+'\n\nQ2 (parametric efficiency): '+lines[1]+'\n\nQ3 (misspecification): '+lines[2]+'\n\nQ4 (constant-step abstention): '+lines[3]+'\n\n'+lines[4]+'\n\n'+lines[5]+'\n\n'
 text+='All predictions are descriptive. Fine-grid values use a discretized model; continuous-path nets are separate. Coarse state aliasing and omitted time preclude a Watkins–Dayan convergence claim. Estimated scales change physical policy classes. W is nonstationary: Oracle-8 uses independent selection/evaluation samples, the nominal OU reference is not an upper bound, and W regret is Oracle-8 only. Gaussian working-model parameter SEs under misspecification do not imply robust coverage. Bertram thresholds optimize a different, long-run objective. Stationarity, coarse bins and one-year episodes limit interpretation.\n'
 publish(base/'RESULTS.md',text);return dict(worlds=len(worlds),rows=len(cells),summary=[lines[0],lines[1],lines[2],lines[3],lines[4]+'; '+lines[5]])
