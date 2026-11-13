import support as S
S.guard()
import sys,io,gzip,json,csv,math
import numpy as np
import pandas as pd
from pathlib import Path
R=Path('/Users/maxiyao/single_stock_runs/inputs_restored_v1');G=Path('/Users/maxiyao/single_stock_runs/gateb_v1')
sys.path.insert(0,str(R))
import fyp_rl.data as FD
def blocked(*a,**k):raise RuntimeError('Downloads blocked: sealed restored cache only')
FD.yf.download=blocked
from fyp_rl.execution import simulate_positions
from fyp_rl.metrics import sharpe_ratio,cagr,annualized_volatility,max_drawdown
TICKERS=['SPY','XLU','KO','IWM','NVDA','TSLA','GOOGL'];HREF=[5,10,20];KEYS={5:7,10:14,20:27}
PH='8802a6f29e89c383eb50876fcdd64206faa9fe9eb1c12363967f57dab024d447';AH='9185cd4dc828c03aea03d2f3ffa7c12341ad0d535b80f88c3e340d53073738a6'
def frame(path):
 b=S.raw(path)
 if str(path).endswith('.gz'):b=gzip.decompress(b)
 return pd.read_csv(io.BytesIO(b),keep_default_na=False,float_precision='round_trip')
def load():
 cal=frame(R/'experiments/turning_point_detection/new_stage1/tables/fold_calendar.csv')
 ref=frame(R/'experiments/turning_point_detection/new_stage1/tables/reference_turns.csv');ref['date']=pd.to_datetime(ref.date)
 cfg=S.js(G/'gateb_config.json');assert cfg['primary']=={'h_ref':10,'cost':.001,'init':'state'}
 oldhash=S.js(G/'input_hashes_before.json')
 # Verify restored bytes against both restoration manifests and gateb's recorded hashes where available.
 manifests=[S.js(R/'MANIFEST.json'),S.js(R/'MANIFEST_ADDENDUM_001.json')]
 rows=[]
 for m in manifests:rows.extend(m.get('files',[]))
 expected={z['relative_path']:z['sha256'] for z in rows}
 px={}
 for t in TICKERS:
  p=R/f'data/raw/{t.lower()}_daily_adjusted_ohlcv.csv';assert S.sha(p)==expected[str(p.relative_to(R))]
  fs=cal[cal.ticker==t];assert len(fs)==(11 if t=='TSLA' else 16)
  start=min(fs.train_start);end=max(fs.test_end)
  px[t]=FD.load_ohlcv(t,start,end,raw_dir=R/'data/raw')
 for p in [R/'experiments/turning_point_detection/new_stage1/tables/reference_turns.csv',R/'experiments/turning_point_detection/new_stage1/tables/fold_calendar.csv']:
  assert S.sha(p)==expected[str(p.relative_to(R))]
 # Imported execution/metrics/data modules are also exact manifest members.
 for name,m in list(sys.modules.items()):
  p=getattr(m,'__file__',None)
  if p and Path(p).is_relative_to(R) and Path(p).suffix=='.py':
   rel=str(Path(p).relative_to(R));b=S.raw(p)
   if rel in expected:assert __import__('hashlib').sha256(b).hexdigest()==expected[rel]
 return px,cal,ref
def sim_index(px,folds):
 start=pd.Timestamp(min(folds.test_start));end=pd.Timestamp(max(folds.test_end));i=px.index.searchsorted(start)-1;assert i>=0
 return px.index[i:px.index.searchsorted(end,side='right')]
def targets(px,folds,ref,ticker,h,L):
 ix=sim_index(px,folds);target=pd.Series(np.nan,index=ix);equality=[];audit=[]
 for f in folds.sort_values('test_start').to_dict('records'):
  events=ref[(ref.ticker==ticker)&(ref.h_ref==h)&(ref.fold_id==f['fold_id'])].sort_values('date',kind='stable').copy()
  assert len(events) and not events.date.duplicated().any()
  loc=px.index.get_indexer(events.date);assert (loc>=0).all()
  shifted=loc+L;valid=shifted<len(px);ev=events[valid].copy();ev['date']=px.index[shifted[valid]]
  if L==0:assert events[['date','type']].reset_index(drop=True).equals(ev[['date','type']].reset_index(drop=True))
  dates=ix[(ix>=pd.Timestamp(f['test_start']))&(ix<=pd.Timestamp(f['test_end']))];prev=ix[ix.get_loc(dates[0])-1]
  prior=ev[ev.date<=prev];w0=0 if prior.empty else int(prior.iloc[-1]['type']=='trough')
  inyear=ev[(ev.date>=dates[0])&(ev.date<=dates[-1])]
  sig=pd.Series(np.nan,index=dates)
  for dt,typ in zip(inyear.date,inyear.type):sig.loc[dt]=float(typ=='trough')
  target.loc[dates]=sig.ffill().fillna(w0).to_numpy();target.loc[prev]=w0
  audit.append({'ticker':ticker,'h_ref':h,'L':L,'fold_id':f['fold_id'],'n_source_events':len(events),'n_shifted_test_events':len(inyear),'initial_position':w0,'n_outside_loaded_calendar':int((~valid).sum()),'INFEASIBLE':True})
 assert target.notna().all()
 return target,audit
def bh_results(px,cal):
 out={}
 for t in TICKERS:
  ix=sim_index(px[t],cal[cal.ticker==t]);x=px[t].loc[ix,['Open','Close']]
  out[t]=simulate_positions(x,pd.Series(1.,index=ix),cost=.001,initial_position=0.)
 return out
def verify_bh(out):
 summary=frame(G/'tables/gateb_summary.csv.gz');years=frame(G/'tables/gateb_by_year.csv.gz');checks=[]
 for t,res in out.items():
  row=summary[(summary.ticker==t)&(summary.setting=='primary')];assert len(row)==1;row=row.iloc[0]
  for k,value in {'ann_return':cagr(res.equity_curve),'ann_vol':annualized_volatility(res.daily_returns),'sharpe':sharpe_ratio(res.daily_returns),'max_drawdown':max_drawdown(res.equity_curve)}.items():
   old=float(row['bh_'+k]);err=abs(value-old);assert err<=1e-9,(t,k,value,old,err)
   checks.append({'ticker':t,'period':'full','metric':k,'recomputed':value,'sealed':old,'abs_error':err})
  for y,r in res.daily_returns.groupby(res.daily_returns.index.year):
   row=years[(years.ticker==t)&(years.year==y)];assert len(row)==1;row=row.iloc[0]
   eq=np.r_[1.,np.cumprod(1+r.to_numpy())]
   for k,v in [('annual_return',eq[-1]-1),('max_drawdown_in_year',np.min(eq/np.maximum.accumulate(eq)-1))]:
    old=float(row['bh_'+k]);err=abs(v-old);assert err<=1e-9,(t,y,k,v,old,err)
    checks.append({'ticker':t,'period':str(y),'metric':k,'recomputed':float(v),'sealed':old,'abs_error':float(err)})
 assert len(checks)==28+214
 return checks
def manual_first_five(px,cal):
 t='SPY';ix=sim_index(px[t],cal[cal.ticker==t])[:6];x=px[t].loc[ix,['Open','Close']]
 result=simulate_positions(x,pd.Series(1.,index=ix),.001,initial_position=0)
 V=1.;hand=[];costs=[]
 for i in range(1,6):
  wo=0. if i==1 else 1.;gap=1+wo*(float(x.Open.iloc[i])/float(x.Close.iloc[i-1])-1)
  fee=V*gap*.001 if i==1 else 0.;nxt=(V*gap-fee)*(float(x.Close.iloc[i])/float(x.Open.iloc[i]))
  hand.append(nxt/V-1);costs.append(fee);V=nxt
 np.testing.assert_allclose(result.daily_returns,hand,atol=1e-14,rtol=0);np.testing.assert_allclose(result.daily.cost_paid,costs,atol=1e-14,rtol=0)
 return pd.DataFrame({'date':result.daily_returns.index,'hand_daily_return':hand,'engine_daily_return':result.daily_returns.to_numpy(),'hand_cost_paid':costs,'engine_cost_paid':result.daily.cost_paid.to_numpy(),'INFEASIBLE_diagnostic_context':True})
def starts_for(n,reps=5000,seed=12345,L=20):
 return np.random.default_rng(seed).integers(0,n-L+1,size=(reps,int(np.ceil(n/L))))
def moment_boot(series,bh,mask,starts,L=20,chunk=250):
 # series: n_dates x n_settings. Count each sampled date, same as explicit gathered returns.
 a=np.nan_to_num(series,nan=0.);b=np.nan_to_num(bh,nan=0.);mask=mask.astype(float);n=len(b)
 ans=np.empty((len(starts),a.shape[1]))
 for i in range(0,len(starts),chunk):
  st=starts[i:i+chunk];idx=(st[:,:,None]+np.arange(L)).reshape(len(st),-1)[:,:n]
  C=np.zeros((len(st),n))
  np.add.at(C,(np.arange(len(st))[:,None],idx),1)
  den=C@mask;mu=C@a/den[:,None];var=np.maximum(0,C@(a*a)/den[:,None]-mu*mu)
  bm=C@b/den;bv=np.maximum(0,C@(b*b)/den-bm*bm)
  with np.errstate(divide='ignore',invalid='ignore'):
   sh=np.where(var>0,np.sqrt(252)*mu/np.sqrt(var),0);bs=np.where(bv>0,np.sqrt(252)*bm/np.sqrt(bv),0)
  ans[i:i+len(st)]=sh-bs[:,None]
 return ans
def literal_boot(series,bh,mask,starts,L=20):
 out=[]
 for st in starts:
  ix=(st[:,None]+np.arange(L)).reshape(-1)[:len(bh)];ix=ix[mask[ix]]
  a=series[ix];b=bh[ix];sd=a.std(0,ddof=0);bs=b.std(ddof=0)
  out.append(np.divide(np.sqrt(252)*a.mean(0),sd,out=np.zeros_like(sd),where=sd>0)-(np.sqrt(252)*b.mean()/bs if bs>0 else 0))
 return np.array(out)
def cross(point):
 q=np.flatnonzero(np.asarray(point)<=0);return int(q[0]) if len(q) else float('inf')
def csvout(name,f):f.to_csv(S.O/name,index=False,float_format='%.17g',compression='gzip' if name.endswith('.gz') else None)
def record_imports():
 for name,m in list(sys.modules.items()):
  p=getattr(m,'__file__',None)
  if p and Path(p).is_relative_to(R) and Path(p).suffix=='.py':S.raw(p)
def main():
 assert S.sha(S.O/'preregistration.md')==PH and S.sha(S.O/'preregistration_addendum.md')==AH
 px,cal,ref=load();bh=bh_results(px,cal)
 try:checks=verify_bh(bh);hand=manual_first_five(px,cal)
 except Exception as e:
  S.dump(S.O/'BH_validation.json',{'passes':False,'error':str(e),'oracle_results_computed':False});raise
 S.dump(S.O/'BH_validation.json',{'passes':True,'tolerance':1e-9,'n_checks':len(checks),'max_abs_error':max(r['abs_error'] for r in checks),'checks':checks})
 csvout('BH_manual_first5.csv',hand)
 for t,res in bh.items():
  f=res.daily.copy();f['ticker']=t;f['diagnostic_context']='INFEASIBLE';csvout(f'BH_daily_{t}.csv.gz',f.reset_index())
 record_imports();before=dict(S.INPUTS);S.dump(S.O/'audit/input_hashes_before.json',before)
 point=np.empty((3,41,7));returns={};events=[]
 for ti,t in enumerate(TICKERS):
  arr=[]
  for hi,h in enumerate(HREF):
   for L in range(41):
    S.resources();tar,audit=targets(px[t],cal[cal.ticker==t],ref,t,h,L)
    res=simulate_positions(px[t].loc[tar.index,['Open','Close']],tar,.001,initial_position=0)
    assert res.daily_returns.index.equals(bh[t].daily_returns.index)
    point[hi,L,ti]=sharpe_ratio(res.daily_returns)-sharpe_ratio(bh[t].daily_returns)
    arr.append(res.daily_returns.to_numpy());events+=audit
  returns[t]=np.array(arr).T;print('INFEASIBLE curves',t,'123 settings',flush=True)
 union=pd.DatetimeIndex(sorted(set().union(*(set(v.daily_returns.index) for v in bh.values()))))
 starts=starts_for(len(union));ewdraw=np.zeros((5000,123));keydraw={}
 for t in TICKERS:
  ix=union.get_indexer(bh[t].daily_returns.index);a=np.full((len(union),123),np.nan);b=np.full(len(union),np.nan);a[ix]=returns[t];b[ix]=bh[t].daily_returns
  mask=np.isfinite(b);draw=moment_boot(a,b,mask,starts)
  # Real-data check against literal gateb-definition resampling before keeping bootstrap results.
  np.testing.assert_allclose(draw[:8,[7,41+14,82+27]],literal_boot(a[:,[7,41+14,82+27]],b,mask,starts[:8]),rtol=0,atol=1e-11)
  ewdraw+=draw/7;print('INFEASIBLE joint bootstrap',t,flush=True)
 ew=point.mean(axis=2);draws=ewdraw.reshape(5000,3,41);keys=[];breakeven=[]
 for hi,h in enumerate(HREF):
  L=KEYS[h];v=draws[:,hi,L];lo,up=np.percentile(v,[2.5,97.5]);p=min(1.,2*min(np.mean(v<=0),np.mean(v>=0)))
  keys.append({'h_ref':h,'L':L,'delta_sharpe':ew[hi,L],'ci_lo':lo,'ci_hi':up,'p':p,'primary':h==10,'INFEASIBLE':True})
  roots=np.array([cross(z) for z in draws[:,hi,:]]);order=np.sort(roots);q=order[[int(np.ceil(.025*len(roots)))-1,int(np.ceil(.975*len(roots)))-1]]
  def label(x):return '>40' if not np.isfinite(x) else str(int(x))
  breakeven.append({'h_ref':h,'L_star':label(cross(ew[hi])),'ci_lo':label(q[0]),'ci_hi':label(q[1]),'bootstrap_censored_share':float(np.mean(~np.isfinite(roots))),'definition':'first lag<=40 with EW delta<=0; otherwise right-censored >40','INFEASIBLE':True})
 csvout('lagged_oracle_key.csv',pd.DataFrame(keys));csvout('breakeven_lag.csv',pd.DataFrame(breakeven));csvout('event_shift_audit.csv.gz',pd.DataFrame(events))
 curve=[]
 for hi,h in enumerate(HREF):
  for L in range(41):
   curve.append({'h_ref':h,'L':L,'ticker':'EW','delta_sharpe':ew[hi,L],'INFEASIBLE':True})
   curve.extend({'h_ref':h,'L':L,'ticker':t,'delta_sharpe':point[hi,L,ti],'INFEASIBLE':True} for ti,t in enumerate(TICKERS))
 csvout('lagged_oracle_curve.csv',pd.DataFrame(curve));np.savez_compressed(S.O/'bootstrap_EW.npz',draws=draws,starts=starts)
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(8,4.5))
 for hi,h in enumerate(HREF):
  line,=ax.plot(range(41),ew[hi],label=f'h_ref={h}');ax.axvline(KEYS[h],ls='--',lw=.9,color=line.get_color(),alpha=.65)
 ax.axhline(0,lw=.8,color='gray');ax.set_xlabel('Lag L (trading days)');ax.set_ylabel('Equal-weight ΔSharpe');ax.legend(frameon=False);fig.tight_layout();fig.savefig(S.O/'lagged_oracle_INFEASIBLE.png',dpi=300);plt.close(fig)
 record_imports()
 for p,r in before.items():assert S.sha(p)==r['sha256']
 # New import-only dependencies are read and checked before/after at this final stage.
 for p,r in list(S.INPUTS.items()):assert S.sha(p)==r['sha256']
 S.dump(S.O/'audit/input_hashes_after.json',S.INPUTS)
 row=next(r for r in keys if r['primary']);interpret='oracle retains positive value despite this delay' if row['ci_lo']>0 else 'delay alone sufficient to erase value' if row['ci_hi']<0 else 'inconclusive; interval includes zero'
 text='# lagged_oracle_v1 — INFEASIBLE (uses future information)\n\nThis is an oracle diagnostic, not a tradable strategy. No new real trading trials, cumulative580.\n\nBH full/annual sealed validation:PASS;242 checks, max absolute error '+str(max(r['abs_error'] for r in checks))+'. First5-day hand formula:PASS. Original daily equality unavailable; approved replacement before outcomes is in preregistration_addendum.md. Reconstructed BH daily files retained.\n\n'+pd.DataFrame(keys).to_csv(index=False)+'\n\n'+pd.DataFrame(breakeven).to_csv(index=False)+'\n\nPrimary interpretation: '+interpret+'. Not in gateb Holm family. L* is a discrete first crossing; >40 means right-censored. Bootstrap uses identical block starts and full/TSLA masks, with moment implementation tested against literal resampling. Sources and hashes in audit/input_hashes_before/after.json. Saved reference events only, no event recomputation or fit.\n'
 (S.O/'README.md').write_text(text);print(text,flush=True)
 S.dump(S.O/'scientific_complete.json',{'status':'numerics_complete_pending_environment_seal','INFEASIBLE':True,'real_trials':580,'new_real_trials':0,'keys':keys,'breakeven':breakeven,'interpretation':interpret})
if __name__=='__main__':main()
