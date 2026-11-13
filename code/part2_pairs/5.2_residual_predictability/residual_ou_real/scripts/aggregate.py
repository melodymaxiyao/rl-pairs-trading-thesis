"""Bounded-memory daily inference and actual stock/hedge turnover accounting."""
from common import *
import pandas as pd
from inputs import panel,index
from models import update_positions
from stats import spearman,weighted_ranks,pooled_regression
from jobs import WORLD,identifier

def ic_records(kind,date,score,half,previous,future,weight):
 rows=[]
 for wi,(world,rep) in enumerate(WORLD):
  top=np.zeros(len(weight),bool);ok=np.isfinite(previous[wi])&np.isfinite(weight)&(weight>0)
  if ok.any():top[ok]=weighted_ranks(previous[wi,ok],weight[ok])>2/3
  for hi,h in enumerate(spec()['horizons']):
   x=-score[wi];y=future[wi,:,hi];valid=np.isfinite(x)&np.isfinite(y)&(weight>0);w=weight[valid];xx=x[valid];yy=y[valid]
   rows.append(dict(definition=kind,date=str(date),world=world,replicate=rep,h=h,ic=spearman(x,y,weight),ic_previous_fast_tercile=spearman(x[top],y[top],weight[top]),n=int(valid.sum()),sum_w=float(w.sum()),sum_x=float(w@xx),sum_y=float(w@yy),sum_xx=float(w@(xx*xx)),sum_xy=float(w@(xx*yy)),sum_yy=float(w@(yy*yy))))
 return rows

def holdings_update(old,new,durations,closed):
 durations[old!=0]+=1
 for i in np.flatnonzero((old!=0)&(new==0)):closed.append(int(durations[i]));durations[i]=0
 durations[(old==0)&(new!=0)]=0

def rp_aggregate(year,jobs):
 data=panel();dates=data['dates'];features=[];structure=[];persist=[];calibration=[];unavailable=[]
 for j in jobs:
  r=cached('runtime/jobs/'+identifier(j));features+=r['series'];structure+=r['structure'];persist+=r['persistence'];calibration+=r['calibration'];unavailable+=r['unavailable']
 features.sort(key=lambda x:x['series']);assert features,('No available P_RAND series',year)
 ts=features[0]['indices'];assert all(np.array_equal(f['indices'],ts) for f in features)
 w=np.array([f['weight'] for f in features]);score=np.stack([f['score'] for f in features],axis=-1);half=np.stack([f['half'] for f in features],axis=-1);previous=np.stack([f['previous_kappa'] for f in features],axis=-1);future=np.stack([f['future'] for f in features],axis=2);alpha=np.stack([f['alpha'] for f in features],axis=-1);beta=np.stack([f['beta'] for f in features],axis=-1);count_series=len(features)
 positions=np.zeros((3,count_series),np.int8);prior_stock=np.zeros((3,count_series));prior_hedge=prior_stock.copy();durations=np.zeros_like(positions,int);closed=[[] for _ in range(3)];daily=[];ic=[]
 for ti,t in enumerate(ts):
  ic+=ic_records('RP',dates[t],score[:,ti],half[:,ti],previous[:,ti],future[:,ti],w)
  for wi,(world,rep) in enumerate(WORLD):
   new=update_positions(positions[wi],score[wi,ti],half[wi,ti],np.isfinite(beta[ti]))
   if ti==len(ts)-1:new[:]=0
   prior_closes=len(closed[wi]);holdings_update(positions[wi],new,durations[wi],closed[wi]);new_closes=closed[wi][prior_closes:];active=new!=0;n=int(active.sum());stock=np.zeros(count_series);hedge=stock.copy()
   if n:
    # The economic rule assigns equal gross to each active observed position.
    # IPW applies to the statistical cross-section, not hedge allocation.
    allocation=np.full(n,1/n);stock[active]=new[active]/(1+abs(beta[ti,active]))*allocation;hedge[active]=-stock[active]*beta[ti,active]
   gross=0.
   if ti:
    held=prior_stock[wi]!=0;one=alpha[ti-1]+future[wi,ti-1,:,0]
    gross=float(prior_stock[wi,held]@one[held]) if np.isfinite(one[held]).all() else np.nan
   turnover=float(abs(stock-prior_stock[wi]).sum()+abs(hedge-prior_hedge[wi]).sum());prior_stock[wi]=stock;prior_hedge[wi]=hedge;positions[wi]=new
   daily.append(dict(definition='RP',world=world,replicate=rep,date=str(dates[t]),gross_bps=gross,turnover=turnover,positions=n,observed_series=count_series,mean_closed_holding=float(np.mean(closed[wi])) if closed[wi] else np.nan,closed_trades=len(closed[wi]),closed_trades_today=len(new_closes),closed_duration_today=sum(new_closes)))
  checkpoint_guard()
 return dict(ic=ic,daily=daily,structure=structure,persistence=persist,calibration=calibration,unavailable=unavailable)

def factor_aggregate(kind,jobs):
 data=panel();dates=data['dates'];R=data['returns'];N=R.shape[1];symbols=data['meta']['symbols'];positions=np.zeros((3,N),np.int8);stock=np.zeros((3,N));hedge=np.zeros((3,N,N));durations=np.zeros((3,N),int);closed=[[] for _ in range(3)];previous_alpha=np.zeros(N);previous_next=np.full((3,N),np.nan);kappa_history={};day_count=len(np.flatnonzero((dates>=np.datetime64('2004-01-01'))&(dates<=np.datetime64('2025-12-31'))));cal_values=np.full((3,4,N,day_count),np.nan,np.float64);cal_names=['half_life','sigma_epsilon','sigma_eq','increment_predictive_r2'];ic=[];daily=[];structure=[];persist=[];coverage=[];test_index=0;cal_eligible=np.zeros(N,int)
 for job in jobs:
  value=cached('runtime/jobs/'+identifier(job))
  for day in value['days']:
   t=day['index'];idx=day['indices'];previous=kappa_history.get(t-60,np.full((3,N),np.nan))[:,idx];now=np.full((3,N),np.nan);now[:,idx]=day['kappa'];kappa_history[t]=now
   for old in list(kappa_history):
    if old<t-60:kappa_history.pop(old)
   if job['year']<2004:continue
   cal_eligible[day['members']]+=1
   coverage.append(dict(definition=kind,date=str(dates[t]),status=day['status'],admissible_series=len(idx),reason=day.get('reason','')))
   ic+=ic_records(kind,dates[t],day['score'],day['half'],previous,day['future'],np.ones(len(idx)))
   for ci,key in enumerate(cal_names):
    values=day['calibration'][key].copy();values[~np.isfinite(day['half'])]=np.nan;cal_values[:,ci,idx,test_index]=values
   test_index+=1
   if t%60==59:
    for wi,(world,rep) in enumerate(WORLD):persist.append(dict(definition=kind,world=world,replicate=rep,date=str(dates[t]),previous_date=str(dates[t-60]),fold=job['year'],window='test',rank_correlation=spearman(now[wi,idx],previous[wi]),n=int((np.isfinite(now[wi,idx])&np.isfinite(previous[wi])).sum())))
   coverage_end=None
   if kind=='RE' and data['meta'].get('price_coverage'):
    coverage_end=min(data['meta']['price_coverage'][symbol]['last_date'] for symbol in spec()['available_sector_etfs'] if data['meta']['price_coverage'].get(symbol))
   economic_in_scope=coverage_end is None or str(dates[t])<=coverage_end
   for wi,(world,rep) in enumerate(WORLD):
    f=day['structure'][wi];structure.append(dict(definition=kind,world=world,replicate=rep,test_year=job['year'],series='member-days',weight=1.,days=int(f[5]),valid=int(f[0]),fast=int(f[1]),significant=int(f[2]),no_reversion=int(f[3]),nonpositive_b=int(f[4]),half_histogram=day['half_histogram'][wi]))
    full_s=np.full(N,np.nan);full_h=np.full(N,np.nan);full_s[idx]=day['score'][wi];full_h[idx]=day['half'][wi];eligible=np.zeros(N,bool);eligible[idx]=day['status']=='ok';new=update_positions(positions[wi],full_s,full_h,eligible)
    if t==len(dates)-1 or (coverage_end is not None and str(dates[t])==coverage_end):new[:]=0
    prior_closes=len(closed[wi]);holdings_update(positions[wi],new,durations[wi],closed[wi]);new_closes=closed[wi][prior_closes:];n=int((new!=0).sum());newstock=np.zeros(N);newhedge=np.zeros((N,N))
    if n:
     local=new[idx];weight=local/day['gross']/n;newstock[idx]=weight;newhedge[idx]=-(weight[:,None]*day['beta'])@day['weights'].T
    held=stock[wi]!=0
    if world=='real':
     used=held|np.any(hedge[wi]!=0,axis=0);gross=float(stock[wi,used]@R[t,used]+hedge[wi][:,used].sum(axis=0)@R[t,used]) if np.isfinite(R[t,used]).all() else np.nan
    else:
     one=previous_alpha+previous_next[wi];gross=float(stock[wi,held]@one[held]) if np.isfinite(one[held]).all() else np.nan
    if not economic_in_scope:gross=np.nan
    turnover=float(abs(newstock-stock[wi]).sum()+abs(newhedge-hedge[wi]).sum());stock[wi]=newstock;hedge[wi]=newhedge;positions[wi]=new
    daily.append(dict(definition=kind,world=world,replicate=rep,date=str(dates[t]),gross_bps=gross,turnover=turnover,positions=n,observed_series=len(idx),mean_closed_holding=float(np.mean(closed[wi])) if closed[wi] else np.nan,closed_trades=len(closed[wi]),closed_trades_today=len(new_closes),closed_duration_today=sum(new_closes)))
   previous_alpha=np.zeros(N);previous_next=np.full((3,N),np.nan)
   if day['status']=='ok':previous_alpha[idx]=day['alpha'];previous_next[:,idx]=day['next_return']
   checkpoint_guard()
  del value
 # Each stock is one calibration series, across its valid reverting test days.
 calibration=[]
 import warnings
 with warnings.catch_warnings():
  warnings.simplefilter('ignore',RuntimeWarning);medians=np.nanmedian(cal_values,axis=3)
 for wi,(world,rep) in enumerate(WORLD):
  for j,symbol in enumerate(symbols):
   valid=int(np.isfinite(cal_values[wi,0,j]).sum())
   if cal_eligible[j]>0:calibration.append(dict(definition=kind,world=world,replicate=rep,series=symbol,weight=1.,valid_reverting_days=valid,total_days=int(cal_eligible[j]),**{key:float(medians[wi,k,j]) for k,key in enumerate(cal_names)}))
 return dict(ic=ic,daily=daily,structure=structure,persistence=persist,calibration=calibration,coverage=coverage)

def aggregate_all(jobs):
 manifests=[]
 for year in spec()['years']:
  stem=f'runtime/aggregates/RP_{year}';r=cached(stem)
  if r is None:r=rp_aggregate(year,[j for j in jobs if j['definition']=='RP' and j['year']==year]);save(stem,r);checkpoint_guard()
  manifests.append(stem)
 for kind in ['RF']+(['RE'] if spec()['available_sector_etfs'] else []):
  stem='runtime/aggregates/'+kind;r=cached(stem)
  if r is None:r=factor_aggregate(kind,[j for j in jobs if j['definition']==kind]);save(stem,r);count('new_rule_evaluations');checkpoint_guard()
  manifests.append(stem)
 return manifests
