"""Descriptive reporting and the four frozen primary tests; no strategy selection."""
from common import *
import pandas as pd
from accounting import performance
from learning import holm,nw
from task_stats import bootstrap,dsr_pbo,market
from experiment import wanted,cell_root

def emit(root,name,frame):
 p=root/name
 if name.endswith('.gz'):publish(p,csv_bytes(frame))
 else:publish(p,frame.to_csv(index=False,float_format='%.17g'))

def paired(series,kind,left,right):
 f=series[(series.definition==kind)&(series.world=='real')&(series.cost==5)];a=f[(f.learner==left[0])&(f.gate==left[1])];b=f[(f.learner==right[0])&(f.gate==right[1])];keys=['series','test_year'];v=a[keys+['net_bps','weight']].merge(b[keys+['net_bps']],on=keys,validate='one_to_one',suffixes=('_a','_b'));assert len(v)==len(a)==len(b);v['value']=v.net_bps_a-v.net_bps_b;return v

def report(tier):
 root=RUN/f'tier_{tier}';root.mkdir(exist_ok=True);paths=[]
 for stage in range(1,tier+1):
  for kind in ['RP','RF']:
   for year in range(2004,2026):
    for cost,gate,learners in wanted(stage):
     for learner in learners:paths.append(cell_root(kind,year,gate,cost,learner))
 series=pd.concat([pd.read_csv(p/'per_series_year.csv.gz',float_precision='round_trip') for p in paths],ignore_index=True);daily=pd.concat([pd.read_csv(p/'portfolio_daily.csv.gz',float_precision='round_trip') for p in paths],ignore_index=True);bins=pd.concat([pd.read_csv(p/'entries_exits.csv.gz',float_precision='round_trip') for p in paths],ignore_index=True)
 from missing_prices import LOG_COLUMNS
 missing_frames=[pd.read_csv(p/'missing_price_log.csv',float_precision='round_trip') for p in paths]
 baseline=pd.read_csv(RUN/'audit/missing_price_revision_v1/stock_days.csv');emit(root,'missing_price_log.csv',pd.concat([baseline,*missing_frames],ignore_index=True).reindex(columns=LOG_COLUMNS))
 emit(root,'per_series_year.csv.gz',series);emit(root,'portfolio_daily.csv.gz',daily);emit(root,'entries_exits.csv',bins)
 hypothesis=[]
 for id,kind,left,right in [('H1','RP',('Q_best','G0'),('AL','G0')),('H2','RF',('Q_best','G0'),('AL','G0')),('H3','RP',('Q_best','G1'),('Q_best','G0')),('H4','RF',('Q_best','G1'),('Q_best','G0'))]:hypothesis.append(dict(id=id,definition=kind,left=str(left),right=str(right),**bootstrap(paired(series,kind,left,right))))
 hp=holm([r['p_one_sided'] for r in hypothesis])
 for r,p in zip(hypothesis,hp):r['holm_p']=float(p);r['supported']=bool(r['estimate']>0 and p<=.05)
 emit(root,'hypotheses.csv',pd.DataFrame(hypothesis))
 summary=[];absolute=[];contrasts=[];market_rows=[];null_rows=[];gate_rows=[];exits=[];spi=[];policies={}
 keys=['definition','gate','cost','learner','world','replicate'];d=__import__('data').panel();spy={str(k):float(v) for k,v in zip(d['dates'],d['benchmark'])}
 for key,g in daily.groupby(keys,sort=True):
  meta=dict(zip(keys,key));g=g.sort_values('date')
  for era,gg in [('pooled',g),*[(f'{a}-{b}',g[(g.test_year>=a)&(g.test_year<=b)]) for a,b in [(2004,2008),(2009,2012),(2013,2016),(2017,2020),(2021,2025)]]]:
   if not len(gg):continue
   for convention in ['K2','PP_FULL']+(['K1'] if key[2]==5 and tier==3 else []):summary.append(dict(**meta,era=era,accounting=convention,annual_K2_turnover=float(252*gg.K2.mean()),**performance(gg.gross_bps,gg[convention],key[2],gg.gross_exposure)))
  if key[4]=='real':
   b=np.array([spy[t] for t in g.date]);market_rows.append(dict(**meta,series='gross',**market(g.gross_bps,b)));market_rows.append(dict(**meta,series='K2_net',**market(g.net_bps,b)))
   if key[2]==5:
    a=series[(series.definition==key[0])&(series.gate==key[1])&(series.cost==5)&(series.learner==key[3])&(series.world=='real')].copy();a['value']=a.net_bps;v=bootstrap(a);yr=a.groupby('test_year').apply(lambda x:np.average(x.net_bps,weights=x.weight),include_groups=False);absolute.append(dict(**meta,positive_fold_share=float((yr>0).mean()),**v))
 for kind in ['RP','RF']:
  for gate in ['G0','G1']:
   available=set(series[(series.definition==kind)&(series.gate==gate)].learner)
   for a,b in [('Q_best','AL'),('Q_best','Band'),('Band','AL'),('PlugIn','Q_best'),('Q_best','Q_POW')]:
    if {a,b}<=available:
     v=paired(series,kind,(a,gate),(b,gate));contrasts.append(dict(definition=kind,gate=gate,comparison=a+' minus '+b,**bootstrap(v)))
   for learner in sorted(available):
    if learner in set(series[(series.definition==kind)&(series.gate=='G0')].learner) and gate=='G1':gate_rows.append(dict(definition=kind,learner=learner,**bootstrap(paired(series,kind,(learner,'G1'),(learner,'G0')))))
    f=daily[(daily.definition==kind)&(daily.gate==gate)&(daily.cost==5)&(daily.learner==learner)];real=f[f.world=='real'].set_index('date');null=f[f.world=='N2'].groupby('date').net_bps.mean();a=real.net_bps-null;null_rows.append(dict(definition=kind,gate=gate,learner=learner,**nw(a.to_numpy(),21),null_replicates=2,interpretation='matched N2 mean; K2 null exposure bookkeeping, not joint-market prices'))
 for p in paths:
  c=read(p/'complete.json');policy=c['policy'];parts=p.name.split('_');kind=parts[0];year=int(parts[1]);gate=parts[2];cost=int(parts[3][1:]);learner='_'.join(parts[4:]);policies[p.name]=policy;exits+=read(p/'exit_comparison.json')
  if learner=='SPI':spi += [dict(definition=kind,test_year=year,gate=gate,cost=cost,**r) for r in policy['tests']]
 spi_diagnostics=[]
 for (kind,gate,cost,world,rep),g in daily[daily.learner=='SPI'].groupby(['definition','gate','cost','world','replicate']):
  baseline=daily[(daily.definition==kind)&(daily.gate==gate)&(daily.cost==cost)&(daily.world==world)&(daily.replicate==rep)&(daily.learner=='AL')].set_index('date');pair=g.set_index('date').net_bps-baseline.net_bps;year=pair.groupby(pair.index.str[:4]).sum();frame=pd.DataFrame(dict(test_year=year.index.astype(int),value=year.to_numpy()+10.,weight=np.ones(len(year))));ni=bootstrap(frame);spi_diagnostics.append(dict(definition=kind,gate=gate,cost=cost,world=world,replicate=rep,deviation_rate=float(g.SPI_override_decisions.sum()/g.valid_decisions.sum()) if g.valid_decisions.sum() else np.nan,harmful_fold_share_beyond10bps=float((year< -10).mean()),mean_SPI_minus_AL_bps=float(year.mean()),noninferiority_margin_bps=10,NI_ci_low=ni['ci95_low'],NI_p=ni['p_one_sided'],NI_supported=bool(ni['ci95_low']>0)))
 emit(root,'SPI_diagnostics.csv',pd.DataFrame(spi_diagnostics))
 emit(root,'absolute_net.csv',pd.DataFrame(absolute));emit(root,'contrasts.csv',pd.DataFrame(contrasts));emit(root,'gate_effects.csv',pd.DataFrame(gate_rows));emit(root,'market.csv',pd.DataFrame(market_rows));emit(root,'real_minus_N2.csv',pd.DataFrame(null_rows));emit(root,'performance.csv',pd.DataFrame(summary));emit(root,'AL_exit_comparison.csv',pd.DataFrame(exits));emit(root,'spi_tests.csv',pd.DataFrame(spi));write(root/'policies.json',policies)
 frozen=pd.read_csv(RUN/'predictions.csv');frozen['observed']=np.nan;frozen['matched_sign']=False
 for i,r in frozen.iterrows():
  lookup={'Q_best - AL':'Q_best minus AL','Q_best - Band':'Q_best minus Band','Band - AL':'Band minus AL'};matches=[x for x in contrasts if x['definition']==r.get('residual') and x['gate']=='G0' and x['comparison']==lookup.get(r.get('contrast'))]
  if matches:frozen.loc[i,'observed']=matches[0]['estimate'];frozen.loc[i,'matched_sign']=(matches[0]['estimate']<0 if '<' in r['frozen_expected_direction'] else matches[0]['estimate']>0)
 emit(root,'predictions.csv',frozen)
 accounting=dsr_pbo(daily,tier);emit(root,'dsr_pbo.csv',pd.DataFrame(accounting));make_plots(root,pd.DataFrame(summary),daily,bins,pd.DataFrame(absolute))
 # NW companions for every primary pair of portfolio series.
 companions=[]
 for r in hypothesis:
  import ast
  a,g1=ast.literal_eval(r['left']);b,g2=ast.literal_eval(r['right']);f=daily[(daily.definition==r['definition'])&(daily.cost==5)&(daily.world=='real')];aa=f[(f.learner==a)&(f.gate==g1)].set_index('date').net_bps;bb=f[(f.learner==b)&(f.gate==g2)].set_index('date').net_bps;companions.append(dict(id=r['id'],**nw((aa-bb).to_numpy(),21)))
 emit(root,'NW_companions.csv',pd.DataFrame(companions))
 lines=["H1–H4: "+'; '.join(f"{r['id']}={r['estimate']:.3f} [{r['ci95_low']:.3f},{r['ci95_high']:.3f}], Holm p={r['holm_p']:.4g}" for r in hypothesis)+'. Frozen synthetic signs: Q_best<AL, Q_best<Band, Band>AL.',"Absolute net5/G0: "+'; '.join(f"{r['definition']} {r['learner']}={r['estimate']:.3f}" for r in absolute if r['gate']=='G0')+' bps/series-year.',"Gate effects: "+('; '.join(f"{r['definition']} {r['learner']}={r['estimate']:.3f}" for r in gate_rows) if tier>=2 else 'Tier1 AL/Q_best only; remaining learners await Tier2.'),"Turnover5/G0: "+'; '.join(f"{r['definition']} {r['learner']}={r['annual_K2_turnover']:.3f} annual K2" for r in summary if r['gate']=='G0' and r['cost']==5 and r['world']=='real' and r['era']=='pooled' and r['accounting']=='K2'),"DSR/PBO: "+'; '.join(str({k:v for k,v in r.items() if k in ['kind','scope','DSR','PBO','status','N']}) for r in accounting)]
 write(root/'summary.json',dict(lines=lines,tier=tier,real_trials=572,planned_new_strategy_cells=20,descriptive_comparisons_select_no_new_policy=True));prefix=(RUN/'README.preregistered.md').read_text();body=prefix+'\n\n## Sealed Tier '+str(tier)+' results\n\n'+'\n\n'.join(lines)+'\n\nAll performance tables use fixed committed capital. Employed-capital returns use a constant average-exposure divisor; their Sharpe ratios therefore equal the committed-capital Sharpe.\n';publish(root/'README.md',body);return lines

def make_plots(root,summary,daily,bins,absolute):
 import matplotlib.pyplot as plt
 directory=root/'plots';directory.mkdir(exist_ok=True)
 def finish(fig,name):fig.tight_layout();fig.savefig(directory/name,dpi=140);plt.close(fig)
 f=absolute[(absolute.gate=='G0')&(absolute.world=='real')];fig,axes=plt.subplots(1,2,figsize=(12,4))
 for ax,kind in zip(axes,['RP','RF']):
  g=f[f.definition==kind];ax.errorbar(np.arange(len(g)),g.estimate,yerr=np.maximum(0,np.vstack([g.estimate-g.ci95_low,g.ci95_high-g.estimate])),fmt='o');ax.set_xticks(np.arange(len(g)),g.learner,rotation=35);ax.axhline(0,color='gray');ax.set_title(kind);ax.set_ylabel('Net bps per series-year, cost5')
 finish(fig,'01_net5.png');q=summary[(summary.gate=='G0')&(summary.world=='real')&(summary.cost==5)&(summary.era=='pooled')&(summary.accounting=='K2')];fig,ax=plt.subplots(figsize=(8,5))
 for r in q.itertuples():
  gg=daily[(daily.definition==r.definition)&(daily.gate=='G0')&(daily.world=='real')&(daily.cost==5)&(daily.learner==r.learner)];x=252*gg.K2.mean();ax.scatter(x,r.net_annual_bps);ax.annotate(r.definition+' '+r.learner,(x,r.net_annual_bps),fontsize=7)
 ax.axhline(0,color='gray');ax.set_xlabel('Annual K2 turnover');ax.set_ylabel('Annual net bps on committed capital');finish(fig,'02_net_turnover.png')
 fig,axes=plt.subplots(1,2,figsize=(12,4))
 for ax,kind in zip(axes,['RP','RF']):
  f=summary[(summary.definition==kind)&(summary.gate=='G0')&(summary.world=='real')&(summary.era=='pooled')&(summary.accounting=='K2')]
  for learner,g in f.groupby('learner'):g=g.sort_values('cost');ax.plot(g.cost,g.net_annual_bps,'o-',label=learner)
  ax.axhline(0,color='gray');ax.set_title(kind);ax.set_xlabel('Cost bps per side');ax.set_ylabel('Annual net bps');ax.legend(fontsize=7)
 finish(fig,'03_net_cost.png');fig,axes=plt.subplots(3,1,figsize=(12,9),sharex=True)
 for ax,kind in zip(axes[:2],['RP','RF']):
  f=daily[(daily.definition==kind)&(daily.gate=='G0')&(daily.cost==5)&(daily.world=='real')]
  for learner,g in f.groupby('learner'):g=g.sort_values('date');ax.plot(pd.to_datetime(g.date),g.net_bps.cumsum(),label=learner)
  ax.set_ylabel(kind+' cumulative net bps');ax.legend(fontsize=7,ncol=3)
 d=__import__('data').panel();ok=d['dates']>=np.datetime64('2004-01-01');axes[2].plot(d['dates'][ok],np.nancumsum(d['benchmark'][ok]));axes[2].set_ylabel('SPY cumulative bps');finish(fig,'04_cumulative.png')
 fig,axes=plt.subplots(2,2,figsize=(12,8))
 for row,kind in enumerate(['RP','RF']):
  for col,side in enumerate(['entry','exit']):
   f=bins[(bins.definition==kind)&(bins.gate=='G0')&(bins.cost==5)&(bins.world=='real')&(bins.side==side)]
   for learner,g in f.groupby('learner'):
    a=g.groupby('bin')[['transitions','opportunities']].sum();axes[row,col].plot(a.index,a.transitions/a.opportunities,'o-',label=learner)
   axes[row,col].set_title(kind+' '+side);axes[row,col].set_xlabel('state bin');axes[row,col].set_ylabel('Realised transition rate');axes[row,col].legend(fontsize=7)
 finish(fig,'05_policy_bins.png')
