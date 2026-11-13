import common as C
C.guard()
import sources as S
import numpy as np,pandas as pd
import ast,hashlib,gzip,pickle,io,json,time
from scipy.stats import t as student_t
P=S.PART;NS={'np':np,'student_t':student_t,'sliding_window_view':np.lib.stride_tricks.sliding_window_view}
def extract(path,names):
 tree=ast.parse(C.text(path));nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names];assert len(nodes)==len(names)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),NS)
extract(S.RES/'scripts/models.py',['ou_fit','sign_series','pair_rolling'])
extract(S.RES/'scripts/stats.py',['weighted_ranks','spearman'])
extract(P/'scripts/residuals.py',['pair_out_of_sample'])
original_ou_fit=NS['ou_fit']
def score_only(residual):
 e=np.asarray(residual,float);scalar=e.ndim==1
 if scalar:e=e[None,:]
 n=e.shape[-1];x=np.concatenate([np.zeros((*e.shape[:-1],1)),np.cumsum(e,axis=-1)],axis=-1);u=x[...,:-1];v=x[...,1:]
 uc=u-u.mean(axis=-1,keepdims=True);vc=v-v.mean(axis=-1,keepdims=True);den=np.sum(uc*uc,axis=-1);valid=np.isfinite(e).all(axis=-1)&(den>0)&(n>2)
 with np.errstate(divide='ignore',invalid='ignore'):
  b=np.sum(uc*vc,axis=-1)/den;a=v.mean(axis=-1)-b*u.mean(axis=-1);error=v-a[...,None]-b[...,None]*u;variance=np.sum(error**2,axis=-1)/(n-2)
  reverting=valid&(b>0)&(b<1)&(variance>0);logb=np.log(np.where(reverting,b,np.nan));half=np.log(2)/(-logb);m=a/(1-b);sigma_eq=np.sqrt(variance/(1-b*b));score=(x[...,-1]-m)/sigma_eq
 out={'s':np.where(valid,np.where(reverting,score,np.nan),np.nan),'half_life':np.where(valid,half,np.nan)}
 return {k:v[0] if scalar else v for k,v in out.items()}
# Only unused diagnostics (CDF, AR coefficient SE, R2) are omitted; arithmetic for score/half-life is unchanged.
rng=np.random.default_rng(20261026)
for _example in [rng.normal(size=(100,60)),np.zeros((3,60)),np.full((2,60),np.nan)]:
 _a=original_ou_fit(_example);_b=score_only(_example)
 for _k in ['s','half_life']:np.testing.assert_array_equal(_a[_k],_b[_k])
NS['ou_fit']=score_only
original_ranks=NS['weighted_ranks']
def ranks(x,w):
 x=np.asarray(x,float);w=np.asarray(w,float);order=np.argsort(x,kind='stable');a=x[order];v=w[order];start=np.r_[0,np.flatnonzero(a[1:]!=a[:-1])+1];stop=np.r_[start[1:],len(a)];group=np.add.reduceat(v,start);mid=group.cumsum()-group/2;out=np.empty(len(x));out[order]=np.repeat(mid/w.sum(),stop-start);return out
NS['weighted_ranks']=ranks
SEED=json.loads(S.resraw('preregistration.json'))['null_seed']
meta_panel=None

def panel():
 global meta_panel
 if meta_panel is None:meta_panel=S.panel()
 return meta_panel

def auth_group(root,name):
 receipt=C.js(root/'complete.json')
 # Group receipts contain exact paths relative to partial run.
 files=receipt.get('files',{})
 assert files,('empty group receipt',root)
 target=root/name;rel=str(target.relative_to(P));expected=files.get(rel,files.get(name))
 if isinstance(expected,dict):expected=expected['sha256']
 assert expected,('unsealed group member',target,list(files)[:4]);assert C.sha(target)==expected
 return target

SIGN_CACHE={}
def signs(kind,year,names,dates,rep):
 key=(kind,year)
 if SIGN_CACHE.get('key')!=key:
  p=panel();ix=np.flatnonzero(p['dates'].astype('datetime64[Y]').astype(int)+1970==year);first=max(0,int(ix[0])-60);last=min(len(p['dates']),int(ix[-1])+6)
  fullnames=[r['pair'] for r in p['meta']['pairs'][str(year)]] if kind=='RP' else list(p['meta']['symbols'])
  SIGN_CACHE.clear();SIGN_CACHE.update(key=key,names=fullnames,lookup={n:i for i,n in enumerate(fullnames)},dates=p['dates'][first:last],reps={})
 if rep not in SIGN_CACHE['reps']:
  SIGN_CACHE['reps'][rep]=np.stack([NS['sign_series'](SEED,f'RP|{year}:{n}' if kind=='RP' else 'RF|'+n,rep,SIGN_CACHE['dates']) for n in SIGN_CACHE['names']])
 a=int(np.searchsorted(SIGN_CACHE['dates'],dates[0]));b=a+len(dates);np.testing.assert_array_equal(SIGN_CACHE['dates'][a:b],dates)
 full=SIGN_CACHE['reps'][rep]
 return full[:,a:b] if names==SIGN_CACHE['names'] else full[np.array([SIGN_CACHE['lookup'][n] for n in names]),a:b]

def rp_refs(year):
 r={}
 for part in range(12):r.update({v['pair']:v for v in S.cache(f'runtime/jobs/RP_{year}_{part:02d}')['series']})
 return r

def checked(a,b,label,rtol=1e-10,atol=1e-8):
 np.testing.assert_array_equal(np.isfinite(a),np.isfinite(b),err_msg=label+' finite mask')
 np.testing.assert_allclose(a,b,rtol=rtol,atol=atol,equal_nan=True,err_msg=label)

def rp_batches(year,validation=False):
 p=panel();R=p['returns'];dates=p['dates'];Y=dates.astype('datetime64[Y]').astype(int)+1970;ts=np.flatnonzero(Y==year);rows=p['meta']['pairs'][str(year)];names=[r['pair'] for r in rows];lookup={n:i for i,n in enumerate(p['meta']['symbols'])};refs=rp_refs(year);idxA=np.array([lookup.get(p['meta']['aliases'].get(r['leg_A'],r['leg_A']),-1) for r in rows]);idxB=np.array([lookup.get(p['meta']['aliases'].get(r['leg_B'],r['leg_B']),-1) for r in rows]);weights=np.array([r['ipw'] for r in rows]);weights/=weights.sum()
 alpha=np.stack([refs[n]['alpha'] for n in names]);beta=np.stack([refs[n]['beta'] for n in names]);stored=np.stack([refs[n]['score'] for n in names],axis=1);future=np.stack([refs[n]['future'][:,:,:2] for n in names],axis=1)
 for j,t in enumerate(ts):
  h=R[t-59:t+1,np.maximum(idxA,0)].T-beta[:,j,None]*R[t-59:t+1,np.maximum(idxB,0)].T
  old=R[t-59:t+1,np.maximum(idxA,0)].T-alpha[:,j,None]-beta[:,j,None]*R[t-59:t+1,np.maximum(idxB,0)].T
  invalid=(idxA<0)|(idxB<0)|~np.isfinite(beta[:,j]);h[invalid]=np.nan;old[invalid]=np.nan
  forward=[];oldfuture=[]
  for hi,H in enumerate([1,5]):
   v=np.full((len(rows),H),np.nan)
   if j+H<len(ts):v=R[t+1:t+H+1,np.maximum(idxA,0)].T-beta[:,j,None]*R[t+1:t+H+1,np.maximum(idxB,0)].T;v[invalid]=np.nan
   forward.append(v);oldfuture.append(future[:,:,j,hi])
  yield dict(t=int(t),j=j,names=names,h=h,old=old,forward=forward,alpha=alpha[:,j],weights=weights,stored_score=stored[:,:,j],stored_future=oldfuture,dates=dates,ts=ts)

def rf_batches(year):
 p=panel();R=p['returns'];dates=p['dates'];Y=dates.astype('datetime64[Y]').astype(int)+1970;ts=np.flatnonzero(Y==year);lookup={int(t):j for j,t in enumerate(ts)};names=p['meta']['symbols']
 for month in range(1,13):
  for v in S.cache(f'runtime/jobs/RF_{year}_{month:02d}')['days']:
   if v['status']!='ok':continue
   t=int(v['index']);j=lookup[t];idx=v['indices'];W=v['weights'];b=v['beta'];a=v['alpha'];needed=np.flatnonzero(np.any(W!=0,axis=1));h=(R[t-59:t+1,idx]-(R[t-59:t+1][:,needed]@W[needed])@b.T).T
   # Match original subtraction order y-alpha-factor, not h-alpha regrouping.
   old=(R[t-59:t+1,idx]-a-(R[t-59:t+1][:,needed]@W[needed])@b.T).T
   forward=[]
   for H in [1,5]:
    z=np.full((len(idx),H),np.nan)
    if j+H<len(ts):z=(R[t+1:t+H+1,idx]-(R[t+1:t+H+1][:,needed]@W[needed])@b.T).T
    forward.append(z)
   yield dict(t=t,j=j,names=[names[i] for i in idx],h=h,old=old,forward=forward,alpha=a,weights=np.ones(len(idx)),stored_score=v['score'],stored_future=[v['future'][:,:,hi] for hi in range(2)],dates=dates,ts=ts,idx=idx,W=W,beta=b)

def iter_batches(kind,year):return rp_batches(year) if kind=='RP' else rf_batches(year)

def validate_core():
 counts={'core_signal_values':0,'core_target_values':0,'N2_signal_values':0};checks=[]
 for kind in ['RP','RF']:
  for year in range(2004,2026):
   C.resources();n=0
   for z in iter_batches(kind,year):
    t=z['t'];new=z['h']-z['h'].mean(axis=-1,keepdims=True)
    score=NS['ou_fit'](new)['s'];checked(score,z['stored_score'][0],f'xi1 score {kind}/{year}/{t}');counts['core_signal_values']+=len(score)
    for hi,H in enumerate([1,5]):
     y=z['forward'][hi].sum(axis=-1);old=z['stored_future'][hi][0];tr=old+H*z['alpha'];checked(y,tr,f'xi1 target {kind}/{year}/{t}/{H}');counts['core_target_values']+=len(y)
    # Date-keyed N2 streams and exact original after-intercept construction.
    for rep in range(2):
     s=signs(kind,year,z['names'],z['dates'][t-59:t+1],rep);fit=NS['ou_fit'](z['old']*s)['s'];checked(fit,z['stored_score'][rep+1],f'N2 score {kind}/{year}/{t}/{rep}');counts['N2_signal_values']+=len(fit)
     for hi,H in enumerate([1,5]):
      if not np.isfinite(z['forward'][hi]).any():y=np.full(len(z['h']),np.nan)
      else:
       fs=signs(kind,year,z['names'],z['dates'][t+1:t+H+1],rep);original=(z['forward'][hi]-z['alpha'][:,None])*fs;y=original.sum(axis=-1)
      checked(y,z['stored_future'][hi][rep+1],f'N2 original target {kind}/{year}/{t}/{rep}/{H}')
    n+=1
   checks.append({'kind':kind,'year':year,'days_checked':n});print('F1 xi=1 / N2 validation',kind,year,n,flush=True)
 C.dump(C.P/'audit/F1_core_validation.json',{'passes':True,'results_computed':False,'counts':counts,'groups':checks,'tolerance':{'rtol':1e-10,'atol':1e-8}})

def validate_original_N3():
 p=panel();R=p['returns'];dates=p['dates'];Y=dates.astype('datetime64[Y]').astype(int)+1970;names=p['meta']['symbols'];lookup={n:i for i,n in enumerate(names)};report=[]
 # Validate both original N3 replicas on every saved RP formation series, without any pooled fit.
 for year in range(2004,2026):
  root=P/f'outputs/N3_intercept_null/RP_formation/{year}';file=auth_group(root,'formation.npz')
  with np.load(file,allow_pickle=False) as a:old=a['N3'];ix=a['indices'];ids=a['ids']
  rows=p['meta']['pairs'][str(year)];assert list(ids)==[r['pair'] for r in rows];end=int(ix[-1]);err=0.
  for i,row in enumerate(rows):
   aa=R[:end+1,lookup[p['meta']['aliases'].get(row['leg_A'],row['leg_A'])]];bb=R[:end+1,lookup[p['meta']['aliases'].get(row['leg_B'],row['leg_B'])]]
   real,alpha,beta=NS['pair_out_of_sample'](aa,bb);valid=np.isfinite(real);prime=aa[60:]-beta*bb[60:]
   for rep in range(2):
    s=NS['sign_series'](SEED,f'RP|{year}:{row["pair"]}',rep,dates[:end+1]);sa=np.lib.stride_tricks.sliding_window_view((s*aa)[:-1],60).mean(axis=-1);sb=np.lib.stride_tricks.sliding_window_view((s*bb)[:-1],60).mean(axis=-1)
    e=s[60:]*prime-(sa-beta*sb);e[~valid]=np.nan;x=np.cumsum(np.where(np.isfinite(e),e,0));x[~np.isfinite(e)]=np.nan
    target=x[ix-60];checked(target,old[rep,i],f'original N3 RP {year}/{i}/{rep}');mask=np.isfinite(target)
    if mask.any():err=max(err,float(np.max(np.abs(target[mask]-old[rep,i,mask]))))
   if i%500==0:C.resources();print('F1 original N3 validation RP',year,i,'/',len(rows),flush=True)
  report.append({'kind':'RP','year':year,'replicas':[0,1],'series':len(rows),'max_abs_error':err})
  C.dump(C.P/'audit/F1_original_N3_validation_progress.json',report)
 # RF original N3 increments in all test years; use previous-day SEALED hedge, never refit PCA.
 for year in range(2004,2026):
  root=P/f'outputs/N3_intercept_null/increments_RF/{year}';file=auth_group(root,'increments.npz')
  with np.load(file,allow_pickle=False) as a:old=a['e'];ix=a['indices']
  months={};err=0.
  for j,t in enumerate(ix):
   prior=int(t)-1;month=str(dates[prior].astype('datetime64[M]'));yy,mm=map(int,month.split('-'))
   if month not in months:
    months[month]={int(v['index']):v for v in S.cache(f'runtime/jobs/RF_{yy}_{mm:02d}')['days']}
    if len(months)>2:months.pop(next(iter(months)))
   assert prior in months[month],('sealed RF previous hedge missing',year,j,prior)
   v=months[month][prior];new=np.full((2,len(names)),np.nan)
   if v['status']=='ok':
    ids=v['indices'];W=v['weights'];b=v['beta'];needed=np.flatnonzero(np.any(W!=0,axis=1));hist=(R[t-60:t,ids]-(R[t-60:t][:,needed]@W[needed])@b.T).T;prime=R[t,ids]-(R[t,needed]@W[needed])@b.T
    for rep in range(2):
     s=signs('RF',year,[names[i] for i in ids],dates[t-60:t+1],rep);new[rep,ids]=s[:,-1]*prime-(s[:,:60]*hist).mean(axis=-1)
    new[:,~np.isfinite(old[0,:,j])]=np.nan
   checked(new,old[:,:,j],f'original N3 RF {year}/{j}');mask=np.isfinite(new)
   if mask.any():err=max(err,float(np.max(abs(new[mask]-old[:,:,j][mask]))))
  report.append({'kind':'RF','year':year,'replicas':[0,1],'sessions':len(ix),'max_abs_error':err});C.dump(C.P/'audit/F1_original_N3_validation_progress.json',report);print('F1 original N3 validation RF',year,flush=True)
 C.dump(C.P/'audit/F1_original_N3_validation.json',{'passes':True,'groups':report,'no_new_MLE_PCI':True,'tolerance':{'rtol':1e-10,'atol':1e-8}})

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--gate',choices=['core','N3'],required=True);a=p.parse_args()
 try:
  validate_core() if a.gate=='core' else validate_original_N3()
 except Exception as e:
  C.dump(C.P/'audit/F1_gate_failure.json',{'gate':a.gate,'status':'STOPPED_BEFORE_IC_RESULTS','error':str(e),'new_IC_results_computed':False,'no_tuning':True});raise
 finally:C.dump(C.P/'audit/F1_gate_inputs.json',C.READS);C.dump(C.P/'audit/F1_unavailable.json',C.MISSING)
