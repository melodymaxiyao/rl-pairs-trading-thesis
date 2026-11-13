"""Descriptive cost diagnostics only; the same gross series under each convention."""
from common import *
import numpy as np,pandas as pd
from accounting import performance,exact_arrays
from sealed_functions import ols_hac

def csv_output(name,frame):publish(RUN/'outputs/tables'/name,frame.to_csv(index=False,float_format='%.17g',lineterminator='\n'))
def era_year(y):return next(f'{a}-{b}' for a,b in spec()['eras'] if a<=y<=b)
def spell_stats(g):
 closed=int(g.closed_trades_today.sum());duration=int(g.closed_duration_today.sum())
 return dict(mean_holding_sessions=duration/closed if closed else np.nan,closed_spells=closed,entries=int(g.entries.sum()),mean_positions=float(g.positions.mean()),max_positions=int(g.positions.max()),mean_unnetted_gross_exposure=float(g.unnetted_gross_exposure.mean()),mean_netted_gross_exposure=float(g.portfolio_netted_gross_exposure.mean()),mean_stock_gross_exposure=float(g.stock_leg_gross_exposure.mean()),mean_hedge_gross_exposure=float(g.hedge_gross_exposure.mean()))
def render(rows,data):
 daily=pd.DataFrame(rows).sort_values(['definition','date']);daily['year']=daily.date.str[:4].astype(int);daily['era']=[era_year(y) for y in daily.year];assert not daily.duplicated(['definition','date']).any()
 # Independent17-significant-digit CSV reproduction, not just aggregate-envelope values.
 sealed=pd.read_csv(source().verify('outputs/tables/rule_daily.csv.gz'),float_precision='round_trip');sealed=sealed[(sealed.world=='real')&(sealed.replicate==0)]
 for kind,g in daily.groupby('definition',sort=True):
  for c in [5,10,20]:
   ref=sealed[(sealed.definition==kind)&(sealed.cost_per_side_bps==c)].sort_values('date');exact_arrays(g.gross_bps,ref.gross_bps,kind+' CSV gross');exact_arrays(g.K0,ref.turnover,kind+' CSV turnover');exact_arrays(g.gross_bps-c*g.K0,ref.net_bps,kind+f' CSV net{c}')
 tables=[];turn=[];market=[];long=[]
 bench=pd.Series(data['benchmark'],index=data['dates'].astype(str))
 for kind,g in daily.groupby('definition',sort=True):
  conventions=['K0','K1','K2']+(['K3'] if kind=='RF' else [])
  groups=[('pooled',g)]+[(era,z) for era,z in g.groupby('era',sort=True)]
  for era,z in groups:
   gross=ols_hac(z.gross_bps,bench.reindex(z.date),5,252);market.append(dict(definition=kind,era=era,series='gross',cost_bps_per_side=0,**gross))
   for convention in conventions:
    for c in [5,10,20]:
     metrics=performance(z.gross_bps,z[convention],c);tables.append(dict(definition=kind,era=era,convention=convention,cost_bps_per_side=c,**metrics,**spell_stats(z)))
     if convention=='K2':market.append(dict(definition=kind,era=era,series='K2_net',cost_bps_per_side=c,**ols_hac(z.gross_bps-c*z.K2,bench.reindex(z.date),5,252)))
  for year,z in [('pooled',g)]+[(int(y),v) for y,v in g.groupby('year')]:
   total={column:float(z[column].sum()) for column in ['K0','K1','K2','stock_leg_gross_turnover','stock_leg_netted_turnover','hedge_gross_turnover','hedge_netted_turnover','factor_basket_turnover']+(['K3'] if kind=='RF' else [])}
   total.update({column+'_annualised':value/len(z)*252 for column,value in list(total.items())})
   valid=z[np.isfinite(z.gross_bps)];k0=float(valid.K0.sum());h=float(valid.hedge_gross_turnover.sum());total.update(definition=kind,year=year,days=len(z),valid_PnL_days=len(valid),era='pooled' if year=='pooled' else era_year(year),hedge_share_K0_all=float(z.hedge_gross_turnover.sum()/z.K0.sum()),hedge_share_K0_valid=h/k0 if k0 else np.nan,total_cost_reduction_K2=1-float(z.K2.sum()/z.K0.sum()),**spell_stats(z));turn.append(total)
  for convention in conventions:
   for c in [5,10,20]:
    x=g[['definition','date','gross_bps','positions','era']].copy();x['convention']=convention;x['cost_bps_per_side']=c;x['turnover']=g[convention];x['cost_bps']=c*g[convention];x['net_bps']=g.gross_bps-x.cost_bps;long.append(x)
 accounting=pd.DataFrame(tables);turnover=pd.DataFrame(turn);exposure=pd.DataFrame(market);ledger=pd.concat(long,ignore_index=True)
 csv_output('cost_accounting.csv',accounting);csv_output('turnover_decomposition.csv',turnover);csv_output('market_exposure.csv',exposure);csv_output('daily_accounting.csv',daily);csv_output('daily_net_returns.csv',ledger)
 # Convenient requested filenames at run root are exact immutable copies of the tables.
 publish(RUN/'cost_accounting.csv',(RUN/'outputs/tables/cost_accounting.csv').read_bytes());publish(RUN/'turnover_decomposition.csv',(RUN/'outputs/tables/turnover_decomposition.csv').read_bytes())
 plots(daily,ledger,accounting,bench)
 pooled=accounting[accounting.era=='pooled'];shares=turnover[turnover.year.astype(str)=='pooled'];summary=[]
 summary.append('K0: each position own-stock leg plus its stock-level combined factor/pair hedge changes; no cross-position netting; daily basis/beta/resizing included.')
 summary.append('K0 hedge turnover share (valid P&L days): '+ '; '.join(f'{r.definition} {r.hedge_share_K0_valid:.2%}' for r in shares.itertuples()))
 summary.append('10bps net Sharpe: '+'; '.join(kind+' '+', '.join(f'{k}={pooled[(pooled.definition==kind)&(pooled.convention==k)&(pooled.cost_bps_per_side==10)].net_Sharpe.iloc[0]:.6f}' for k in ['K1','K2']) for kind in ['RF','RP']))
 summary.append('Breakeven bps/side: '+'; '.join(kind+' '+', '.join(f'{r.convention}={r.breakeven_bps_per_side:.6f}' for r in pooled[(pooled.definition==kind)&(pooled.cost_bps_per_side==10)].itertuples()) for kind in ['RF','RP']))
 profit=[]
 for kind in ['RF','RP']:
  for c in [5,10]:
   row=pooled[(pooled.definition==kind)&(pooled.convention=='K2')&(pooled.cost_bps_per_side==c)].iloc[0];profit.append(f'{kind} K2 at {c}bps: annual mean net {row.net_annual_bps:.6f}bps; Sharpe {row.net_Sharpe:.6f}; '+('positive' if row.net_annual_bps>0 else 'nonpositive')+'.')
 body='''# RF/RP cost accounting of unchanged sealed strategies\n\nTrial count remains552. No RL, model/PCA/OU fit, new signals, thresholds,\nportfolio normalisation or strategy trial was introduced. Saved signals, beta\nand eigenportfolio weights were replayed through the exact sealed position rule.\nAll daily K0 gross/turnover/net5,10,20 and position/holding counts reproduce\nunrounded sealed checkpoints exactly; an independent round-trip CSV check also\npasses.\n\nK0 combines15 factors **inside each RF position** before taking stock-level\nabsolute trade changes, then sums across positions. It charges the own-stock leg\nand each position's hedge separately. It is not15 independently charged factors.\nIt includes daily changes in PCA basis, betas, active-count allocation and gross\nnormalisers, even for continuing signals. RP K0 similarly charges pair legs\nseparately. K1 charges only those source own-stock legs, with free hedges. RP K1\nkeeps pair-level own legs; K2 also nets repeated stocks across pairs. K2 nets\nall own/hedge stock weights together across the portfolio. K3 nets the15 factor\nexposures across RF positions, charges each changing factor stock basket plus\nown-stock legs; it includes PCA basis changes and does not net across factors.\nNo convention rescales leverage when netted exposure falls. K1 is a free-hedge\nbenchmark, not a universal lower bound to K2, which may offset stock and hedge.\n\n'''
 body+='\n'.join(summary)+'\n\n'+'\n'.join(profit)+'\n\n'
 body+='''Annual return is252 times mean daily bps, matching the sealed report. Sharpe\nuses sample SD and sqrt252. CAGR and compounded-NAV maximum drawdown are supplied\nas companions, with NAV starting1; additive-bps drawdown also accompanies the\nsource-style cumulative P&L plot. Missing gross stays missing under every cost\nconvention; breakeven uses turnover on the same valid daily observations.\nTurnover yearly rows are sums in notional units; valid-day hedge shares reflect\nreported costs, with all-date shares shown separately. Holding-period summaries\nuse source completed spells, assigned to the closing date/era. Era slicing does\nnot reset the portfolio. RP still closes at each year end; RF carries as sealed.\n\nSPY alpha/beta uses the exact T7_5 OLS-HAC implementation with lag5 and252-session\nannualisation; raw daily bps, intercept, and paired missing-date deletion. Its\nsealed realised-close benchmark has a missing first session each year; no values\nare filled. These regressions are descriptive accounting diagnostics, not new\nstrategy fits/trials.\n\nK2 is the user-defined portfolio-level convention. A-L discusses netting long\nand short factor exposures; this task does not replicate its different leverage,\nfinancing or stock-selection choices. Reference: https://math.nyu.edu/inmemoriam/avellaneda/AvellanedaLeeStatArb20090616.pdf\n\nInput limitations remain those of the sealed study: development/reused years,\nreconstructed membership/current GICS, survivorship, close execution, missing\nprices and no borrow/financing charges. This audit changes transaction accounting\nonly and does not establish independent out-of-sample profitability.\n\nTables: cost_accounting.csv, turnover_decomposition.csv, market_exposure.csv,\ndaily_accounting.csv and daily_net_returns.csv under outputs/tables. RF monthly\nand RP yearly weight ledgers/checkpoints retain the replay evidence; dense RF\nposition-by-stock matrices are reconstructed in memory and not duplicated daily.\n'''
 publish(RUN/'README.md',body);write(RUN/'summary.json',dict(lines=summary,K2_profitability=profit,trials=552,new_trials=0,RL_fits=0,source_strategy_model_fits=0,descriptive_market_regressions=len(exposure),K0_exact=True,source_input_records=read(RUN/'audit/input_registration.json')['files']))
 return summary

def plots(daily,ledger,accounting,bench):
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(3,1,figsize=(12,10),sharex=True)
 for ax,kind in zip(axes[:2],['RF','RP']):
  for convention,g in ledger[(ledger.definition==kind)&(ledger.cost_bps_per_side==10)].groupby('convention',sort=True):
   ax.plot(pd.to_datetime(g.date),g.net_bps.cumsum(),label=convention,linewidth=1)
  ax.set(title=kind+' — unchanged positions,10bps per side',ylabel='Cumulative P&L (bps)');ax.legend();ax.axhline(0,color='grey',linewidth=.5)
 dates=pd.to_datetime(bench.index);ok=(dates.year>=2004)&(dates.year<=2025);axes[2].plot(dates[ok],bench.iloc[np.flatnonzero(ok)].cumsum(),label='SPY');axes[2].set(title='SPY — separate panel',ylabel='Cumulative return (bps)',xlabel='Date');axes[2].legend();fig.tight_layout();fig.savefig(RUN/'outputs/plots/cumulative_net10.png',dpi=300);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
 for ax,kind in zip(axes,['RF','RP']):
  g=accounting[(accounting.definition==kind)&(accounting.era=='pooled')&(accounting.cost_bps_per_side==10)]
  ax.bar(g.convention,g.breakeven_bps_per_side);ax.set(title=kind,xlabel='Cost convention',ylabel='Breakeven cost (bps/side)');ax.axhline(0,color='grey',linewidth=.5)
 fig.tight_layout();fig.savefig(RUN/'outputs/plots/breakeven_cost.png',dpi=300);plt.close(fig)
