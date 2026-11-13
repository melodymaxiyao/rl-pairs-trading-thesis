"""Read-only sealed panel and causal contexts kept in RAM only."""
from common import *
M=load_module('_coint_sealed_models','residual_ou_real_v1','scripts/models.py')
_PANEL=None

def panel():
 global _PANEL
 if _PANEL is not None:return _PANEL
 s=source();meta=s.cache('runtime/data/calendar_v2/panel');prefix='runtime/data/calendar_v2/'
 for rel,h in meta['files'].items():assert s.expected[rel]==h;s.verify(rel)
 _PANEL=dict(meta=meta,**{k:np.load(s.verify(prefix+k+'.npy'),mmap_mode='r',allow_pickle=False) for k in ['dates','returns','membership','benchmark']})
 return _PANEL

def index(symbol,d):
 symbol=d['meta']['aliases'].get(str(symbol).upper(),str(symbol).upper());return d['meta']['symbols'].index(symbol) if symbol in d['meta']['symbols'] else -1

def pool(year):
 rel=f'experiments/15_pair_trading/trend_anchor_trading_v1/runtime/pools/{year}.pkl.gz';registration=read(source().verify('audit/input_registration.json'));r=registration['selected'][rel];p=INPUT/rel
 assert sha(p)==r['sha256'] and p.stat().st_size==int(r['size'])
 with gzip.open(p,'rb') as f:v=pickle.load(f)
 assert len(v['training_subset'])==600 and len(set(v['training_subset']))==600;return v

def adf_batch(e):
 """ADF constant/lag1 exactly the statsmodels OLS design; vectorised 60 windows."""
 e=np.asarray(e,float);shape=e.shape[:-1];flat=e.reshape(-1,60);out=np.zeros(len(flat),bool);ok=np.isfinite(flat).all(axis=1)
 if not ok.any():return out.reshape(shape)
 x=np.cumsum(flat[ok],axis=1);dx=np.diff(x,axis=1);y=dx[:,1:];lev=x[:,1:-1];lag=dx[:,:-1];X=np.stack([lev,lag,np.ones_like(lev)],axis=-1)
 # SVD pseudoinverse, matching statsmodels' pinv OLS (including rank-deficient windows).
 pinv=np.linalg.pinv(X,rcond=1e-15);coef=np.einsum('nij,nj->ni',pinv,y);err=y-np.einsum('nij,nj->ni',X,coef);rank=np.linalg.matrix_rank(X);dof=len(y[0])-rank;var=np.sum(err*err,axis=1)/dof;cov00=np.sum(pinv[:,0]**2,axis=1)
 with np.errstate(invalid='ignore',divide='ignore'):statistic=coef[:,0]/np.sqrt(var*cov00)
 from statsmodels.tsa.adfvalues import mackinnoncrit
 critical=mackinnoncrit(N=1,regression='c',nobs=58)[1];out[ok]=np.isfinite(statistic)&(statistic<critical);return out.reshape(shape)

def sufficient(e,gross):
 e=np.asarray(e,float);g=np.asarray(gross,float);good=np.isfinite(e).all(axis=1)&np.isfinite(g)&(g>0)
 if not good.any():return np.zeros(6)
 x=np.column_stack([np.zeros(good.sum()),np.cumsum(e[good]/g[good,None],axis=1)]);u=x[:,:-1];v=x[:,1:]
 return np.array([u.size,u.sum(),v.sum(),np.sum(u*u),np.sum(u*v),np.sum(v*v)],float)

def fitted_ar1(stats):
 n,su,sv,suu,suv,svv=np.sum(stats,axis=0);den=suu-su*su/n;b=(suv-su*sv/n)/den;a=(sv-b*su)/n
 if not np.isfinite(b) or not 0<b<1:raise RuntimeError(f'Pooled training AR1 is not stationary: b={b}; no clipping or model change')
 sse=svv-2*a*sv-2*b*suv+n*a*a+2*a*b*su+b*b*suu;sigma=np.sqrt(max(0,sse)/(n-2));assert sigma>0;return dict(b=float(b),a=float(a),mean=float(a/(1-b)),sigma_epsilon=float(sigma),n_transitions=int(n),half_life=float(np.log(.5)/np.log(b)))

def _empty(kind,year,ids,rows=None):
 d=panel();years=d['dates'].astype('datetime64[Y]').astype(int)+1970;ts=np.flatnonzero(years==year);N=len(ids);T=len(ts);k=1 if kind=='RP' else 15
 return dict(kind=kind,year=year,ids=ids,rows=rows,ts=ts,dates=np.asarray(d['dates'][ts]),score=np.full((3,N,T),np.nan),half=np.full((3,N,T),np.nan),adf=np.zeros((3,N,T),bool),alpha=np.full((N,T),np.nan),beta=np.full((N,T,k),np.nan),gross=np.full((N,T),np.nan),W=None,stats=np.zeros(6),N=N,T=T)

def rp_context(year,rows,verify_stored=False):
 d=panel();R=d['returns'];ctx=_empty('RP',year,[r['pair'] for r in rows],rows);ts=ctx['ts'];start=max(0,ts[0]-59);end=ts[-1]+1;sign_dates=d['dates'][start:end];ctx['ia']=np.array([index(r['leg_A'],d) for r in rows]);ctx['ib']=np.array([index(r['leg_B'],d) for r in rows]);stats=[]
 refs={}
 if verify_stored:
  for part in range(12):refs.update({f['pair']:f for f in source().cache(f'runtime/jobs/RP_{year}_{part:02d}')['series']})
 for i,row in enumerate(rows):
  ia,ib=ctx['ia'][i],ctx['ib'][i]
  if min(ia,ib)<0:continue
  e,a,b=M.pair_rolling(R[start:end,ia],R[start:end,ib]);assert len(e)==len(ts);ctx['alpha'][i]=a;ctx['beta'][i,:,0]=b;ctx['gross'][i]=1+abs(b);stats.append(sufficient(e,1+abs(b)))
  for wi in range(3):
   signs=np.ones(len(sign_dates)) if wi==0 else M.sign_series(source().prereg['null_seed'],f'RP|{year}:{row["pair"]}',wi-1,sign_dates)
   ee=e*np.lib.stride_tricks.sliding_window_view(signs,60);fit=M.ou_fit(ee);ctx['score'][wi,i]=fit['s'];ctx['half'][wi,i]=fit['half_life'];ctx['adf'][wi,i]=adf_batch(ee)
  if refs:
   ref=refs[row['pair']]
   for key,refkey in [('score','score'),('half','half')]:assert np.array_equal(ctx[key][:,i],ref[refkey],equal_nan=True),('RP sealed fixture mismatch',year,row['pair'],key)
  if i%100==0:checkpoint()
 ctx['stats']=np.sum(stats,axis=0) if stats else np.zeros(6);return ctx

def rf_context(year,verify_stored=True):
 d=panel();R=d['returns'];N=R.shape[1];ctx=_empty('RF',year,list(d['meta']['symbols']));T=ctx['T'];ctx['W']=np.zeros((T,N,15));cached_days={}
 for month in range(1,13):
  stem=f'runtime/jobs/RF_{year}_{month:02d}'
  if stem+'.pkl.gz' in source().expected:
   for day in source().cache(stem)['days']:cached_days[day['index']]=day
 for ti,t in enumerate(ctx['ts']):
  checkpoint();history=R[max(0,t-251):t+1];members=np.flatnonzero(d['membership'][t]);valid=members[np.isfinite(history[:,members]).all(axis=0)] if len(history)==252 else np.array([],int);valid=valid[np.std(history[:,valid],axis=0)>0] if len(valid) else valid;old=cached_days.get(int(t))
  if old is not None:
   if old['status']!='ok':continue
   assert np.array_equal(old['indices'],valid);W,b,a,gross,e=from_saved_RF(old,history)
  elif len(valid)>=16:
   try:model=M.factor_window(history[:,valid])
   except (ValueError,np.linalg.LinAlgError):continue
   W=np.zeros((N,15));W[valid]=model['weights'];b=model['beta'];a=model['alpha'];gross=model['gross'];e=model['residual'].T
  else:continue
  ctx['W'][ti]=W;ctx['alpha'][valid,ti]=a;ctx['beta'][valid,ti]=b;ctx['gross'][valid,ti]=gross;ctx['stats']+=sufficient(e,gross)
  for wi in range(3):
   signs=np.ones_like(e) if wi==0 else np.stack([M.sign_series(source().prereg['null_seed'],'RF|'+d['meta']['symbols'][j],wi-1,d['dates'][t-59:t+1]) for j in valid]);ee=rf_signed(e,signs);fit=M.ou_fit(ee);ctx['score'][wi,valid,ti]=fit['s'];ctx['half'][wi,valid,ti]=fit['half_life'];ctx['adf'][wi,valid,ti]=adf_batch(ee)
   if old is not None and verify_stored:
    assert np.array_equal(fit['s'],old['score'][wi],equal_nan=True),('RF score reconstruction differs',year,int(t),wi)
    assert np.array_equal(fit['half_life'],old['half'][wi],equal_nan=True),('RF half-life reconstruction differs',year,int(t),wi)
    # Execution uses saved sealed scores after bit-exact reconstruction.
    ctx['score'][wi,valid,ti]=old['score'][wi];ctx['half'][wi,valid,ti]=old['half'][wi]
 return ctx

def scores(ctx,gate,wi=0):
 z=ctx['score'][wi];h=ctx['half'][wi];ok=np.isfinite(z)&np.isfinite(h)&(h>0)&(h<30)
 if gate=='G1':ok &= ctx['adf'][wi]
 return np.where(ok,z,np.inf)

def entry_vector(ctx,i,t):
 d=panel();N=d['returns'].shape[1];v=np.zeros(N);g=ctx['gross'][i,t];assert np.isfinite(g) and g>0
 if ctx['kind']=='RP':v[ctx['ia'][i]]+=1/g;v[ctx['ib'][i]]-=ctx['beta'][i,t,0]/g
 else:v[i]+=1/g;v-=ctx['W'][t]@ctx['beta'][i,t]/g
 return v

def reward_cube(contexts):
 """Gross-normalised entry-frozen forward return; RAM only, padded invalid days."""
 from missing_prices import marked_prices
 d=panel();mark=marked_prices(d);assert mark.get('training_quote_gaps',0)==0;R=mark['returns'];T=max(c['T'] for c in contexts);count=sum(c['N'] for c in contexts);cube=np.full((count,T,T),np.nan);offset=0
 for c in contexts:
  n,L=c['N'],c['T'];ts=c['ts'];future=R[ts]
  for t in range(L):
   if c['kind']=='RP':
    ia=np.maximum(c['ia'],0);ib=np.maximum(c['ib'],0);values=(future[:,ia].T-c['beta'][:,t,0,None]*future[:,ib].T)/c['gross'][:,t,None]
   else:
    W=c['W'][t];needed=np.any(W!=0,axis=1);factor=future[:,needed]@W[needed];values=(future.T-c['beta'][:,t]@factor.T)/c['gross'][:,t,None]
   pad=T-L;cube[offset:offset+n,pad+t,pad:pad+L-1]=values[:,1:];cube[offset:offset+n,pad+t,T-1]=0
  offset+=n
 return cube

def training_scores(contexts,gate):
 T=max(c['T'] for c in contexts);return np.concatenate([np.pad(scores(c,gate),((0,0),(T-c['T'],0)),constant_values=np.inf) for c in contexts])

def from_saved_RF(day,history):
 valid=day['indices'];W=day['weights'];b=day['beta'];a=day['alpha'];gross=day['gross'];f=history[-60:,valid]@W[valid];X=np.column_stack([np.ones(60),f]);e=(history[-60:,valid]-X@np.vstack([a,b.T])).T;return W,b,a,gross,e

def rf_signed(e,signs):
 # Preserve the source60-by-member C layout before transposing for ou_fit.
 return (np.ascontiguousarray(e.T)*np.ascontiguousarray(signs.T)).T
