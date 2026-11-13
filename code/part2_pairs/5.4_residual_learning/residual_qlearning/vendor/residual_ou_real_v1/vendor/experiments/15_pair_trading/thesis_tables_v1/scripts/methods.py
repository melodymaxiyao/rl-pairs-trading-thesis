"""Only the two registered post-hoc analyses and decimal precision checking."""
import numpy as np,pandas as pd
from decimal import Decimal,ROUND_HALF_UP
from common import COUNTS
def rounded_match(value,literal):
 text=str(literal).replace('−','-').replace('+','').replace(',','').strip()
 if not np.isfinite(float(value)):return False
 d=Decimal(text);return Decimal(str(float(value))).quantize(Decimal(1).scaleb(d.as_tuple().exponent),rounding=ROUND_HALF_UP)==d
def weighted_median(x,w):
 x=np.asarray(x,float);w=np.asarray(w,float);ok=~np.isnan(x)&np.isfinite(w)&(w>0);x=x[ok];w=w[ok]
 COUNTS['rows_aggregated']+=len(x)
 if not len(x):return np.nan
 order=np.argsort(x,kind='stable');x=x[order];w=w[order]
 return float(x[np.searchsorted(w.cumsum(),w.sum()/2-1e-9*w.sum(),side='left')])
def quality_rows(frame):
 f=frame[frame.in_random.eq(True)].copy()
 cols=['leg_return_corr_A','spread_sd_bps_A','quarterly_beta_cv_A']
 missing=f[cols].isna().any(axis=1);dropped=int(missing.sum());f=f.loc[~missing].copy()
 ranks=f.groupby('test_year',sort=True)[cols].rank(method='average',pct=True)
 f['quality_score']=(ranks[cols[0]]+1-ranks[cols[1]]+1-ranks[cols[2]])/3
 f['decile']=f.groupby('test_year',sort=True).quality_score.transform(lambda s:pd.qcut(s.rank(method='first'),10,labels=False)+1).astype(int)
 f['P_KM_le63']=(f.KM_med_2_0_B<=63).astype(float)
 f['reverted_share']=f.reverted_share_2_0_B.fillna(0)
 f['zero_crossings']=f.zero_crossings_per_year_B
 f['cross_sector_share']=f.sector_relation.eq('cross_sector').astype(float)
 COUNTS['rows_aggregated']+=len(f)
 return f,dropped
def quality_summary(frame,draws=2000):
 f,dropped=quality_rows(frame);rows=[];metrics=['P_KM_le63','reverted_share','zero_crossings','cross_sector_share']
 for era,g in [('pooled',f)]+list(f.groupby('era',sort=True)):
  for decile,d in g.groupby('decile',sort=True):
   row=dict(era=era,decile=decile,rows=len(d),weight=d.ipw.sum())
   for col in metrics:row[col]=np.average(d[col],weights=d.ipw)
   rows.append(row)
  years=sorted(g.test_year.unique());rng=np.random.default_rng(20261023)
  ix=rng.integers(len(years),size=(draws,len(years)));counts=np.array([np.bincount(x,minlength=len(years)) for x in ix])
  for col in metrics:
   vals=[]
   for dec in [10,1]:
    d=g[g.decile==dec];num=[];den=[]
    for y in years:
     e=d[d.test_year==y];num.append(np.sum(e.ipw*e[col]));den.append(e.ipw.sum())
    boot=(counts@num)/(counts@den);vals.append((np.sum(num)/np.sum(den),boot))
   b=vals[0][1]-vals[1][1]
   rows.append(dict(era=era,decile='top-bottom',metric=col,estimate=vals[0][0]-vals[1][0],ci95_low=np.quantile(b,.025),ci95_high=np.quantile(b,.975),bootstrap_draws=draws,seed=20261023))
 return pd.DataFrame(rows),dict(input_random_rows=int(frame.in_random.eq(True).sum()),missing_score_rows=dropped,KM_B_undefined=int(f.KM_med_2_0_B.isna().sum()),reverted_share_no_event_filled_zero=int(f.reverted_share_2_0_B.isna().sum()))
def ols_hac(y,x,lag,annualisation):
 y=np.asarray(y,float);x=np.asarray(x,float);ok=np.isfinite(x)&np.isfinite(y);x=x[ok];y=y[ok];COUNTS['rows_aggregated']+=len(y)
 X=np.column_stack([np.ones(len(x)),x])
 if len(y)<=2 or np.linalg.matrix_rank(X)<2:return dict(status='rank deficient',n=len(y))
 b=np.linalg.lstsq(X,y,rcond=None)[0];u=y-X@b;A=np.linalg.inv(X.T@X);z=X*u[:,None];meat=z.T@z
 for k in range(1,min(lag,len(y)-1)+1):
  C=z[k:].T@z[:-k];meat+=(1-k/(lag+1))*(C+C.T)
 cov=A@meat@A;se=np.sqrt(max(0.,cov[0,0]));mean=y.mean()*annualisation
 return dict(status='ok',n=len(y),beta=float(b[1]),alpha_annual_bps=float(b[0]*annualisation),alpha_annual_se_bps=float(se*annualisation),t_alpha=float(b[0]/se) if se>0 else np.nan,correlation=float(np.corrcoef(x,y)[0,1]) if y.std()>0 else np.nan,mean_annual_bps=float(mean),market_share=float(1-b[0]*annualisation/mean) if mean!=0 else np.nan,appraisal_ratio=np.nan,lag=lag,annualisation=annualisation)
def market_regressions(daily):
 rows=[];companions=[]
 spy=daily[(daily.pool=='P_RAND')&(daily.cell=='SPY')&(daily.variant=='primary')][['date','net_bps']].copy()
 assert not spy.date.duplicated().any();spy=spy.rename(columns={'net_bps':'market_SPY'})
 for (pool,variant),g in daily.groupby(['pool','variant'],sort=True):
  wide=g.pivot(index='date',columns='cell',values='net_bps').join(spy.set_index('date')).sort_index()
  # Avoid column name collision: the source's SPY cell is a strategy, market is fixed reference.
  for freq,periods,lag in [('daily',252,5),('monthly',12,3),('annual',1,1)]:
   w=wide.copy();w.index=pd.to_datetime(w.index)
   if freq!='daily':w=w.groupby(w.index.to_period('M' if freq=='monthly' else 'Y')).sum(min_count=1)
   for cell in [x for x in w.columns if x!='market_SPY']:
    r=ols_hac(w[cell],w['market_SPY'],lag,periods);COUNTS['regressions_run']+=1
    rows.append(dict(pool=pool,variant=variant,cell=cell,frequency=freq,post_hoc=True,**r))
   if {'R_TR_aln','H1s'}.issubset(w.columns):
    z=w[['R_TR_aln','H1s','market_SPY']].dropna();x=np.column_stack([np.ones(len(z)),z.market_SPY]);a=z.R_TR_aln.to_numpy();b=z.H1s.to_numpy();ra=a-x@np.linalg.lstsq(x,a,rcond=None)[0];rb=b-x@np.linalg.lstsq(x,b,rcond=None)[0];COUNTS['regressions_run']+=2
    COUNTS['rows_aggregated']+=2*len(z);mix=(a+b)/2;sd=mix.std(ddof=1)
    companions.append(dict(pool=pool,variant=variant,frequency=freq,raw_correlation=np.corrcoef(a,b)[0,1],SPY_removed_correlation=np.corrcoef(ra,rb)[0,1],mix_50_50_Sharpe=np.sqrt(periods)*mix.mean()/sd if sd>0 else np.nan,post_hoc=True,n=len(z)))
 return pd.DataFrame(rows),pd.DataFrame(companions)
