"""Descriptive synthetic predictions; calibration fit stays visible."""
from common import *
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp,gc
from benchmarks import selection,truth_table,observations,model,statistics,policy_name
from observed import execute,policy,al
from worlds import world_job

def write_csv(path,frame):publish(path,frame.to_csv(index=False,float_format='%.17g',lineterminator='\n'))
def point_results(rank,point,workers=1):
 stem=f'runtime/point_results/{rank}';old=cached(stem)
 if old is not None:return old
 selected=selection(rank,point);truth=truth_table(rank,point,selected);args=[(rank,point,j,selected) for j in range(spec()['worlds'])];pool=ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) if workers>1 else None
 try:
  iterator=pool.map(world_job,args,chunksize=1) if pool else map(world_job,args);worlds=[]
  for result in iterator:worlds.append(result);print('World committed',rank,result['world']+1,'/',spec()['worlds'],flush=True);checkpoint()
 finally:
  if pool:pool.shutdown(wait=True,cancel_futures=True)
 data=observations(model(point),spec()['oracle']['truth_years'],['truth']);refs={str(c):execute(data,policy(selected['bits'][str(c)]),c)[:,0] for c in spec()['costs']};base=execute(data,al(),0);baselines={str(c):base[:,1]-c*base[:,2] for c in spec()['costs']};rows=[];policies=[];gate_tests=[];seen={}
 for w in worlds:
  for source in w['rows']:
   c=source['cost'];pol=source['policy'];name=policy_name(pol)
   if source['learner']=='AL rule':name=policy_name(dict(kind=2,enter=1.25,exit=.5))
   if source['learner']=='SPI_TD':
    key=(name,c)
    if key not in seen:seen[key]=statistics(execute(data,pol,0),pol,c,refs[str(c)],baselines[str(c)])
    actual=seen[key]
   else:actual=truth['rows'][name][str(c)]
   row=dict(rank=rank,h0=point['h0'],w=point['w'],world=w['world'],cost=c,learner=source['learner'],true_value=actual['net'],true_value_MC_SE=actual['net_SE'],test_net=source['test']['net'],test_net_SE=source['test']['net_SE'],turnover=actual['turnover'],holding_period=actual['holding_period'],gross_exposure=actual['gross_exposure'],regret=actual['regret_oracle'],regret_MC_SE=actual['regret_oracle_SE'],breakeven_cost=actual['breakeven'],gross_value=actual['gross'],gain_over_AL=actual['gain_over_AL'],gain_over_AL_MC_SE=actual['gain_over_AL_SE'],SPI_harmful=bool(source['learner']=='SPI_TD' and actual['gain_over_AL']< -1),harm_MC_uncertain=bool(source['learner']=='SPI_TD' and abs(actual['gain_over_AL']+1)<=1.96*actual['gain_over_AL_SE']),value_method='independent50000-series-year common MC; shared within point',plugin_valid=source['plugin_valid']);rows.append(row);policies.append(dict(rank=rank,world=w['world'],cost=c,learner=source['learner'],policy=pol,training_hash=w['training_hash'],test_hash=w['test_hash']))
  for gate in w['gates']:
   for row in gate['tests']:gate_tests.append(dict(rank=rank,world=w['world'],cost=gate['cost'],**row))
 del data;gc.collect();result=dict(rank=rank,point=point,rows=rows,policies=policies,gate_tests=gate_tests,oracle=selected,truth_series_years=spec()['oracle']['truth_years']);save(stem,result);return result

def predictions(frame):
 rows=[]
 for rank,g in frame.groupby('rank',sort=False):
  primary=g[g.cost==5];mean=primary.groupby('learner').mean(numeric_only=True);O=mean.loc['Oracle-8'];A=mean.loc['AL rule']
  rows.append(dict(rank=rank,id='R1',cost=5,estimate=O.true_value,criterion='Oracle net>0 and breakeven>AL',supported=bool(O.true_value>0 and O.breakeven_cost>A.breakeven_cost),oracle_breakeven=O.breakeven_cost,AL_breakeven=A.breakeven_cost))
  rows.append(dict(rank=rank,id='R2',cost=5,estimate=min(mean.loc['Q_POW'].gain_over_AL,mean.loc['Band-train'].gain_over_AL),criterion='both mean true gains>0',supported=bool(mean.loc['Q_POW'].gain_over_AL>0 and mean.loc['Band-train'].gain_over_AL>0)))
  for learner in ['Q_POW','PlugIn-8']:rows.append(dict(rank=rank,id='R3',cost=5,learner=learner,estimate=mean.loc[learner].regret,criterion='descriptive regret; no direction',supported=None))
  for c,z in g[g.learner=='SPI_TD'].groupby('cost'):
   rate=float(z.SPI_harmful.mean());rows.append(dict(rank=rank,id='R4',cost=int(c),estimate=rate,criterion='harmful true gain< -1bps/year rate<=.10',supported=bool(rate<=.10),worlds=len(z),MC_uncertain_worlds=int(z.harm_MC_uncertain.sum())))
  rows.append(dict(rank=rank,id='R5',cost=5,estimate=O.turnover-A.turnover,criterion='primary-cost Oracle turnover<AL; other costs descriptive',supported=bool(O.turnover<A.turnover)))
 return pd.DataFrame(rows)

def _render(calibration,results,tier):
 rows=[r for p in results for r in p['rows']];frame=pd.DataFrame(rows);pred=predictions(frame);policies=[p for r in results for p in r['policies']];gate=[p for r in results for p in r['gate_tests']];write_csv(RUN/'cells.csv',frame);write_csv(RUN/'predictions.csv',pred);write(RUN/'policies.json',dict(policies=policies));write_csv(RUN/'outputs/tables/cells.csv',frame);write_csv(RUN/'outputs/tables/predictions.csv',pred);write_csv(RUN/'outputs/tables/SPI_tests.csv',pd.DataFrame(gate));plots(calibration,frame)
 best=frame[(frame['rank']=='best')&(frame.cost==5)].groupby('learner').mean(numeric_only=True);O=best.loc['Oracle-8'];A=best.loc['AL rule'];spi_rows=frame[(frame['rank']=='best')&(frame.learner=='SPI_TD')];rates={int(c):float(g.SPI_harmful.mean()) for c,g in spi_rows.groupby('cost')};chosen=calibration['chosen'];acceptable=calibration['adequate_25pct_diagnostic'];profits=[name for name in best.index if best.loc[name].true_value>0]
 implication=('No real-strength claim yet: the fixed OU+RW family does not match every excess target within25%; inspect the reported errors before designing a real run.' if not acceptable else ('Use the profitable synthetic policies as candidates; preregister a separate real RF/RP experiment with train-only selection and explicit K2 accounting at2/5/10bps. Synthetic residual-position costs are K1-like and do not establish real portfolio profitability.' if profits else 'Do not launch a large real RL search on this evidence: no tested learner is positive at5bps; retain low-cost2bps as a descriptive sensitivity and include AL/Band/PlugIn/Q/SPI in any separately preregistered real test.'))
 lines=[f"Calibration h0={chosen['h0']}, w={chosen['w']:g}, relative SSE={chosen['objective']:.6g}; per-target relative errors={chosen['relative_errors']}; second h0={calibration['second']['h0']},w={calibration['second']['w']:g}.",f"c=5: Oracle net={O.true_value:.6f}, breakeven={O.breakeven_cost:.6f}; AL net={A.true_value:.6f}, breakeven={A.breakeven_cost:.6f} (bps/year; cost bps/side).",'c=5 true values: '+', '.join(f'{name}={best.loc[name].true_value:.6f}' for name in ['Q_POW','PlugIn-8','Band-train'])+'.','SPI harmful rates (best point): '+', '.join(f'c{c}={rate:.1%}' for c,rate in rates.items())+'.',implication]
 body='# Real-strength OU synthetic study\n\n'+'\n\n'.join(lines)+'\n\n'+('Full requested two-point design completed.' if tier==3 else 'Tier2 only: second-point sensitivity is still pending; this is not the full requested completion.')+'\n\nProfitable policies at5bps (point MC estimates): '+(', '.join(profits) if profits else 'none')+'. The MC-selected Oracle is a within8-state reference with finite selection/evaluation error; it is not a numerically proven upper bound. Band/SPI are not restricted to that8-bit class.\n\nAll policies use the same causal60-session rolling OU score and the same0<half-life<30, finite-estimate filter. Invalid estimates force charged closure and prevent entry. The old static normalization and its4-sigma boundary are not used for trading. The imported8-state map, reverting targets, reward and POW update body are unchanged; the observation facade passes infinity on invalid observations and twice the per-side fee to the old half-cost trainer. Terminal liquidation is charged.\n\nPlugIn fits a pooled stationary AR(1), then selects among256 policies using50000 independent fitted-model MC years with these rolling observations. Rolling state is history-dependent, so a one-dimensional Bellman shortcut is not used. Invalid pooled estimates deterministically abstain. Oracle selection and truth each use separate50000-year streams; world tests use fresh5000 years. Data and learning streams are common across costs/learners and across the two calibration points.\n\nCalibration has500 series times40 independent years at each of36 fixed points. Each episode has60 history sessions then252 traded sessions; the OU component starts stationary and RW starts0 at the beginning of history. Temporal medians are per series over all40 years; N2 pools per-series medians over both replicas, matching sealed transfer weighting. Exact unrounded targets are used. Sigma scaling hits the sealed innovation median; per-target fit errors and the second-best point are retained. The25% flag is a descriptive adequacy diagnostic, not a grid-expansion or stopping rule.\n\nSPI uses train-only daily mean incremental P&L for one-state Q overrides versus AL, Newey-West lag21 and Holm. Annual terminal fees are assigned to the final actual training session for HAC; the trainer retains its zero-return terminal observation. Harm is true MC value below AL by more than1bps per series-year; MC ambiguity near that boundary is reported.\n\nThese costs are on a unit residual position (K1-like); real RF/RP hedge netting, capital scaling, borrow and financing are outside this synthetic experiment. Real-strategy trial count remains552. No real strategy was fitted or tried.\n\nTables: cells.csv, predictions.csv, policies.json; per-switch tests and plots under outputs/. Calibration.json shows every fixed grid point and exact target-relative errors. Raw calibration arrays are temporary scratch, hash-recorded before removal after each point; all selected summaries and model/seed checkpoints remain sealed and are deterministically regenerable from code and stream keys.\n'
 publish(RUN/'README.md',body);write(RUN/'summary.json',dict(lines=lines,tier=tier,real_trial_count=552,new_real_trials=0,profitable_at5=profits,calibration_adequacy_diagnostic=acceptable,SPI_harmful_rates=rates));return lines

def plots(calibration,frame):
 import matplotlib.pyplot as plt
 targets=calibration['targets'];chosen=[calibration['chosen'],calibration['second']];keys=['IC_excess_h5','IC_excess_h1','increment_R2_excess','sigma_epsilon_median'];fig,axes=plt.subplots(1,2,figsize=(12,4));x=np.arange(4)
 for j,p in enumerate(chosen):axes[0].bar(x+(j-.5)*.3,[p['fitted'][k]/targets[k] for k in keys],width=.3,label='best' if j==0 else 'second')
 axes[0].axhline(1,color='black',linewidth=.8);axes[0].set_xticks(x,['IC5','IC1','Increment R²','σepsilon']);axes[0].set_ylabel('Fitted / exact target');axes[0].legend();points=calibration['points'];axes[1].scatter([p['h0'] for p in points],[p['w'] for p in points],c=[np.log10(max(p['objective'],1e-12)) for p in points],cmap='viridis');axes[1].set(xlabel='Latent half-life',ylabel='RW innovation variance share');fig.tight_layout();fig.savefig(RUN/'outputs/plots/calibration_fit.png',dpi=250);plt.close(fig)
 ranks=frame['rank'].unique();fig,axes=plt.subplots(len(ranks),1,figsize=(10,4*len(ranks)),squeeze=False)
 for ax,rank in zip(axes[:,0],ranks):
  for name,g in frame[frame['rank']==rank].groupby('learner'):
   z=g.groupby('cost').true_value.mean();ax.plot(z.index,z.values,marker='*' if name=='Oracle-8' else 'o',label=name)
  ax.axhline(0,color='grey',linewidth=.8);ax.set(title=rank,xlabel='Cost (bps per side)',ylabel='True MC net (bps/year)');ax.legend(ncol=3)
 fig.tight_layout();fig.savefig(RUN/'outputs/plots/net_against_cost.png',dpi=250);plt.close(fig)
 fig,ax=plt.subplots(figsize=(9,6))
 for name,g in frame[frame.cost==5].groupby('learner'):ax.scatter(g.turnover,g.true_value,label=name,marker='*' if name=='Oracle-8' else 'o',alpha=.65)
 ax.axhline(0,color='grey',linewidth=.8);ax.set(xlabel='Turnover per series-year',ylabel='True net at5bps (bps/year)');ax.legend();fig.tight_layout();fig.savefig(RUN/'outputs/plots/turnover_against_net.png',dpi=250);plt.close(fig)

def render(calibration,results,tier):
 global RUN
 if tier==3:return _render(calibration,results,tier)
 original=RUN;out=RUN/('tier_'+str(tier));(out/'outputs/tables').mkdir(parents=True,exist_ok=True);(out/'outputs/plots').mkdir(parents=True,exist_ok=True)
 RUN=out
 try:return _render(calibration,results,tier)
 finally:RUN=original
