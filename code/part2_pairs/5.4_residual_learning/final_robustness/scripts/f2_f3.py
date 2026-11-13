import common as C
C.guard()
import sources as S
import numpy as np,pandas as pd
from scipy.stats import spearmanr
import gzip,io,json,hashlib
from pathlib import Path
YEARS=list(range(2004,2026));ANNUAL=None;CELLS={}
def dfbytes(b,compressed=False):return pd.read_csv(io.BytesIO(b),compression='gzip' if compressed else None,keep_default_na=False,float_precision='round_trip')
def load_f2():
 global ANNUAL
 part=Path('/Users/maxiyao/FYP_runs/closing_checks_v1');seal=C.js(part/'part1/complete.json');assert seal['status']=='complete'
 # Original source SHA ledger is authenticated by the sealed Part1 manifest.
 manifest=C.js(part/'part1/manifest.json');print('Part1 manifest verified in source comparison',flush=True)
 annual=pd.DataFrame(C.rows(part/'part1_year_data.csv'));records=[];proof=[]
 for kind in ['RP','RF']:
  for cost in [5,10]:
   for learner in ['Q_best','Band','AL']:
    for year in YEARS:
     C.resources();rel=f'outputs/cells/{kind}_{year}_G0_c{cost}_{learner}/per_series_year.csv.gz';b=S.core(rel);f=dfbytes(b,True);real=f[f.world=='real'].copy();assert len(real) and abs(real.weight.sum()-1)<1e-12
     value=float(np.sum(real.net_bps.to_numpy(float)*real.weight.to_numpy(float)));old=annual[(annual.family==kind)&(annual.learner==learner)&(annual.year==str(year))&(annual.cost==str(cost))];assert len(old)==1
     np.testing.assert_allclose(value,float(old.iloc[0].N),atol=1e-10,rtol=0)
     records.append({'family':kind,'cost':cost,'learner':learner,'year':year,'net_bps':value,'source_file':str(S.CORE/rel),'source_sha256':hashlib.sha256(b).hexdigest(),'robustness_check':True})
     CELLS[kind,cost,learner,year]=real
    proof.append({'family':kind,'cost':cost,'learner':learner,'Part1_all22annualvalues_exact':True})
 ANNUAL=pd.DataFrame(records)
 hyp=dfbytes(S.core('tier_3/hypotheses.csv.gz'),True)
 follow=C.js(S.CORE/'post_tier3_review_v1/followup_runs/1793580484598758000/results.json')
 comparisons=[]
 for kind,H in [('RP','H1'),('RF','H2')]:
  q=ANNUAL[(ANNUAL.family==kind)&(ANNUAL.cost==5)&(ANNUAL.learner=='Q_best')].net_bps.to_numpy();a=ANNUAL[(ANNUAL.family==kind)&(ANNUAL.cost==5)&(ANNUAL.learner=='AL')].net_bps.to_numpy();old=float(hyp[hyp.id==H].iloc[0].estimate);np.testing.assert_allclose(np.mean(q-a),old,rtol=0,atol=1e-9);comparisons.append({'family':kind,'cost':5,'comparison':H,'estimate':float(np.mean(q-a)),'sealed':old,'match':True})
  q=ANNUAL[(ANNUAL.family==kind)&(ANNUAL.cost==10)&(ANNUAL.learner=='Q_best')].net_bps.to_numpy();a=ANNUAL[(ANNUAL.family==kind)&(ANNUAL.cost==10)&(ANNUAL.learner=='AL')].net_bps.to_numpy();old=next(z['estimate'] for z in follow['contrasts'] if z['definition']==kind and z['cost']==10 and z['comparison']=='Q_best − AL');np.testing.assert_allclose(np.mean(q-a),old,rtol=0,atol=1e-9);comparisons.append({'family':kind,'cost':10,'estimate':float(np.mean(q-a)),'sealed':old,'match':True})
 C.dump(C.P/'audit/F2_validation.json',{'passes':True,'annual_checks':264,'source':'same exact per_series_year files as Part1','contrasts':comparisons,'table54_current_document_location':'not supplied; original T7_2/follow-up yearly-K2 contrasts reproduced; T7_3 day-weighted performance is not substituted'})
 return ANNUAL

def f2_results():
 global ANNUAL
 assert ANNUAL is not None;assert C.js(C.P/'audit/F2_validation.json')['passes']
 idx=np.random.default_rng(20261026).integers(22,size=(5000,22));out=[]
 for (kind,cost,learner),group in ANNUAL.groupby(['family','cost','learner'],sort=False):
  vals=group.sort_values('year').net_bps.to_numpy();assert len(vals)==22
  boot=vals[idx].mean(axis=1);lo,hi=np.percentile(boot,[2.5,97.5]);out.append({'family':kind,'cost_bps':int(cost),'learner':learner,'comparison':learner+' − Always Flat','mean_net_bps':float(vals.mean()),'ci95_low':float(lo),'ci95_high':float(hi),'p_one_sided':float(np.mean(boot<=0)),'years':22,'bootstrap_draws':5000,'bootstrap_seed':20261026,'unit':'bps per committed-capital year; equal22years','Holm':'not applied','robustness_check':True})
 C.csvout(C.P/'F2_vs_flat.csv',out);ANNUAL.to_csv(C.P/'F2_annual_source_values.csv.gz',index=False,compression='gzip',float_format='%.17g');return out

def load_f3():
 import csv
 p=S.panel();meta=p['meta'];dates=pd.DatetimeIndex(p['dates']);Y=dates.year.to_numpy();symbols=meta['symbols'];lookup={s:i for i,s in enumerate(symbols)}
 base='experiments/15_pair_trading/fresh_holdout_data_v1/'
 frames={};source_paths={}
 for name in ['membership_monthly.csv','failed_tickers.csv','selected_changes.csv']:
  path,b=S.input_bytes(base+name);assert b is not None,(name,'unavailable');frames[name]=pd.read_csv(io.BytesIO(b),keep_default_na=False,dtype=str);source_paths[name]=str(path)
 membership=frames['membership_monthly.csv'];membership['month_end']=pd.to_datetime(membership.month_end)
 assert membership.month_end.nunique()==312 and membership.yahoo_symbol.nunique()==834,('membership summary mismatch',membership.month_end.nunique(),membership.yahoo_symbol.nunique())
 # Download receipts are provenance only; actual coverage uses original quote bytes and the exact experiment calendar.
 ap=C.js(S.APP/'complete.json');assert C.sha(S.APP/'manifest.json')==ap['manifest_sha256'];am=C.js(S.APP/'manifest.json')['files'];assert C.sha(S.APP/'price_download_receipts.csv')==am['price_download_receipts.csv'];receipts=pd.DataFrame(C.rows(S.APP/'price_download_receipts.csv'))
 quote=np.zeros((26,len(symbols)),bool);uncertain=np.zeros_like(quote);prefix=base+'data/raw/prices/';calendar_set=set(dates)
 prices={Path(k).name[:-10].upper():k for k in S.REG if k.startswith(prefix)}
 missing=[]
 for j,symbol in enumerate(symbols):
  C.resources()
  if symbol not in prices:
   assert meta['price_coverage'].get(symbol) is None,(symbol,'metadata says prices exist but source absent');continue
  path,b=S.input_bytes(prices[symbol])
  if b is None:
   for k,year in enumerate(range(2000,2026)):
    if np.isfinite(p['returns'][Y==year,j]).any():quote[k,j]=True
    else:uncertain[k,j]=True
   missing.append(prices[symbol]);continue
  f=pd.read_csv(io.BytesIO(b),usecols=['Date','Adj Close'],keep_default_na=False);dt=pd.to_datetime(f.Date,utc=True).dt.tz_localize(None).dt.normalize();price=pd.to_numeric(f['Adj Close'],errors='coerce');valid=np.isfinite(price)&(price>0)&dt.isin(calendar_set)
  for year in dt[valid].dt.year.unique():
   if 2000<=year<=2025:quote[int(year)-2000,j]=True
  if j%100==0:print('F3 original price coverage',j,'/',len(symbols),flush=True)
 aliases={str(t):set(g.yahoo_symbol) for t,g in membership.groupby('ticker')};changes=frames['selected_changes.csv'];changes['year']=pd.to_datetime(changes.effective_date).dt.year
 failed=set(frames['failed_tickers.csv'].yahoo_symbol)
 rows=[]
 for year in range(2000,2026):
  k=year-2000;mm=membership[(membership.month_end.dt.year==year)&(membership.month_end.dt.month==12)];assert mm.month_end.nunique()==1
  members=sorted(set(mm.yahoo_symbol));assert all(n in lookup for n in members);idx=np.array([lookup[n] for n in members]);unknown=int(uncertain[k,idx].sum());count=int(quote[k,idx].sum());den=len(members)
  removed=sorted(set(changes.loc[(changes.year==year)&(changes.removed!=''),'removed']));nr_missing=0;nr_unknown=0
  for ticker in removed:
   candidates=aliases.get(ticker,set());symbol=next(iter(candidates)) if len(candidates)==1 else ticker if ticker in lookup else None
   if symbol is None or symbol not in lookup:nr_unknown+=1
   elif uncertain[k,lookup[symbol]]:nr_unknown+=1
   elif not quote[k,lookup[symbol]]:nr_missing+=1
  raw={'year':year,'year_end_members':den,'members_with_price':count if not unknown else '', 'coverage_fraction':count/den if not unknown else '', 'missing_fraction':1-count/den if not unknown else '', 'unknown_coverage_members':unknown,'members_with_finite_panel_return':int(np.isfinite(p['returns'][Y==year][:,idx]).any(axis=0).sum()),'removed_observed_in_saved_changes':len(removed),'removed_observed_without_price':nr_missing,'removed_observed_price_status_unknown':nr_unknown,'removal_history_complete':False,'coverage_definition':'at least1 finite positive adjusted close on actual experiment-calendar dates in year','membership_source':source_paths['membership_monthly.csv'],'removals_source':source_paths['selected_changes.csv'],'robustness_check':True}
  if str(year) in meta['pairs']:
   raw['RP_formation_pool_pairs']=len(meta['pairs'][str(year)])
   raw['RP_formation_pool_distinct_legs']=len(set(r['leg_A'] for r in meta['pairs'][str(year)])|set(r['leg_B'] for r in meta['pairs'][str(year)]))
   for learner in ['Q_best','Band','AL']:
    for kind in ['RP','RF']:
     f=CELLS[kind,5,learner,year];active=f[f.entries>0];raw[f'{kind}_{learner}_c5_entered_series']=len(active)
     if kind=='RP':
      pairs={r['pair']:r for r in meta['pairs'][str(year)]};legs=set()
      for n in active.series:
       # Per-series key may be year-prefixed; only exact original mapping accepted.
       key=str(n)
       if key not in pairs and key.startswith(str(year)+':'):key=key[len(str(year))+1:]
       assert key in pairs,('unmatched RP series',year,n)
       legs.update([pairs[key]['leg_A'],pairs[key]['leg_B']])
      raw[f'RP_{learner}_c5_entered_distinct_legs']=len(legs)
  rows.append(raw)
 C.dump(C.P/'audit/F3_validation.json',{'passes':True,'membership_months':312,'unique_members':834,'panel_symbols_including_benchmarks':len(symbols),'panel_shape':list(p['returns'].shape),'missing_price_sources':missing,'receipts_rows':len(receipts),'removal_history_complete':False,'removed_counts_role':'known saved dated events only; not asserted complete','coverage_rule':'original adjusted-close bytes restricted to actual experiment calendar; separate strict finite-return count retained'})
 return rows

def f3_results(rows):
 C.csvout(C.P/'F3_coverage_by_year.csv',rows)
 f=pd.DataFrame(rows);top=f.sort_values(['missing_fraction','year'],ascending=[False,True]).head(5);C.csvout(C.P/'F3_top5_missing.csv',top.astype(object).where(top.notna(),'').to_dict('records'))
 cor=[]
 for (kind,cost,learner),g in ANNUAL[ANNUAL.learner.isin(['Q_best','AL'])].groupby(['family','cost','learner'],sort=False):
  z=g.merge(f[['year','missing_fraction']],on='year').dropna();result=spearmanr(z.missing_fraction,z.net_bps);cor.append({'family':kind,'cost_bps':int(cost),'learner':learner,'spearman_rho':float(result.statistic) if np.isfinite(result.statistic) else '', 'n_years':len(z),'years':'2004–2025','missingness':'year-end members without any price during year','descriptive_only':True,'robustness_check':True})
 C.csvout(C.P/'F3_missingness_correlations.csv',cor)
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(8,4));line,=ax.plot(f.year,f.coverage_fraction*100,color='#2166ac',marker='o',ms=3,lw=1.3);ax.set_xlabel('Year');ax.set_ylabel('Year-end members with price (%)');ax.grid(axis='y',alpha=.2);fig.tight_layout()
 np.testing.assert_array_equal(line.get_ydata(),f.coverage_fraction.to_numpy()*100);assert ax.get_title()==''
 fig.savefig(C.P/'F3_coverage_robustness.png',dpi=300);plt.close(fig);C.dump(C.P/'audit/F3_plot_test.json',{'passes':True,'plotted_values_equal_table':True,'dpi':300,'internal_title':False,'source_table_sha256':C.sha(C.P/'F3_coverage_by_year.csv')})
 return cor

if __name__=='__main__':
 load_f2();f2=f2_results();print('F2 COMPLETE',len(f2),flush=True)
 rows=load_f3();cor=f3_results(rows);print('F3 COMPLETE',len(rows),flush=True)
 C.dump(C.P/'audit/F2_F3_inputs.json',C.READS);C.dump(C.P/'audit/F2_F3_unavailable.json',C.MISSING)
