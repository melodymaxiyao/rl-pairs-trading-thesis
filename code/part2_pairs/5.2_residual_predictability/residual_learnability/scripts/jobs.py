"""Train statistics first; a separate hindsight phase; immutable small chunks."""
from common import *
import numpy as np,pandas as pd
from sources import members,context,legs,setup,MEM
from statistics import window_statistics,all_stats,adf_aic,secondary,adf,ecm,predict
from inference import calibrate

def sample_plan():
 old=cached('runtime/sample_plan')
 if old is not None:return old
 rows=[dict(test_year=y,pair=r['pair'],relation=r['sector_relation']) for y in YEARS for r in members(y)['P_RAND']];f=pd.DataFrame(rows);assert len(f)==66000
 rng=np.random.default_rng(stable_seed('aic_sample'));aic=set(f.iloc[np.sort(rng.choice(len(f),5000,replace=False))].apply(lambda r:(r.test_year,r.pair),axis=1))
 strata=list(f.groupby(['test_year','relation'],sort=True));raw=np.array([len(v)*5000/len(f) for _,v in strata]);quota=np.floor(raw).astype(int)
 for j in np.argsort(-(raw-quota),kind='stable')[:5000-quota.sum()]:quota[j]+=1
 eg=set()
 for ((y,rel),g),n in zip(strata,quota):
  random=np.random.default_rng(stable_seed('eg_sample',int(y),rel));ix=random.choice(len(g),int(n),replace=False)
  eg.update((int(y),p) for p in g.iloc[ix].pair)
 assert len(aic)==len(eg)==5000
 return save('runtime/sample_plan',dict(aic=sorted(aic),eg=sorted(eg),strata=[dict(year=y,relation=r,n=int(n)) for ((y,r),_),n in zip(strata,quota)]))

def base_meta(row,y,world,rep):
 return dict(pair=row['pair'],test_year=y,era=era(y),world=world,replicate=rep,leg_A=row['leg_A'],leg_B=row['leg_B'],relation=row['sector_relation'])
def windows(cal,y):
 return dict(T=(cal.year>=y-3)&(cal.year<y),A=(cal.year>=y-3)&(cal.year<=y-2),B=cal.year==y-1)
def synthetic_legs(row,y,world,rep,c):
 a,b=legs(row,y);m=setup().market
 if world=='N2':
  signs=m.rng(row['pair'],y,world,rep).choice(np.array([-1.,1.]),size=len(a)-1)
  return [np.r_[v[0],v[0]+np.cumsum(np.diff(v)*signs)] for v in [a,b]]
 if world=='N1':
  cal=c['cal'];train=(cal.year>=y-3)&(cal.year<y);rr=np.diff(np.array([a[train],b[train]]),axis=1);sa,sb=rr.std(axis=1,ddof=1);rho=float(np.corrcoef(rr)[0,1]);random=m.rng(row['pair'],y,world,rep);random.bit_generator.state=c['N1_state_before_draw']
  return setup().originals.gaussian(len(cal),sa,sb,rho,random)
 return a,b

def train_chunk(y,offset,rows,plan):
 stem=f'runtime/train_chunks/{y}/{offset:05d}';old=cached(stem)
 if old is not None:return old
 aic=set(map(tuple,plan['aic']));eg=set(map(tuple,plan['eg']));out=[];unavailable=[]
 for row in rows:
  for world,rep in WORLDS:
   c=context(row,y,world,rep);meta=base_meta(row,y,world,rep)
   if c.get('unavailable'):unavailable.append(dict(meta,reason=c['reason']));continue
   cal=c['cal'];S=c['level'];d=c['anchors']['A_TR']['d'];masks=windows(cal,y)
   assert not (cal.year>=y).any(),'Screen context read test year'
   for window,mask in masks.items():
    values=window_statistics(S,d,cal,y,window)
    if window=='T' and (y,row['pair']) in aic:values.update(adf_aic(S[mask]))
    if window=='T' and (y,row['pair']) in eg:
     a,b=synthetic_legs(row,y,world,rep,c);values.update(secondary(a[mask],b[mask]))
    for name,value in values.items():
     out.append(dict(meta,window=window,statistic_name=name,beta=c['beta'],start_date=str(cal[mask][0].date()),end_date=str(cal[mask][-1].date()),sessions=int(mask.sum()),anchor_invalid_sessions=int((~np.isfinite(d[mask])).sum()),**value))
  MEM.clear()
 return save(stem,dict(frame=pd.DataFrame(out),unavailable=unavailable))

def train_fold(y):
 start=time.monotonic();plan=sample_plan();ps=members(y);union={}
 for pool in POOLS:
  for r in ps[pool]:
   if r['pair'] in union:
    assert [union[r['pair']][k] for k in ['leg_A','leg_B']]==[r[k] for k in ['leg_A','leg_B']]
   else:union[r['pair']]=r
 ordered=[union[k] for k in sorted(union)];pieces=[];missing=[]
 for offset in range(0,len(ordered),50):
  value=train_chunk(y,offset,ordered[offset:offset+50],plan);pieces.append(value['frame']);missing.extend(value['unavailable'])
  if offset%250==0:print('Train statistics',y,min(offset+50,len(ordered)),'/',len(ordered),flush=True)
 raw=pd.concat(pieces,ignore_index=True);count('pool_fold_statistical_jobs')
 for pool in POOLS:
  stem=f'runtime/pool_folds/{pool}_{y}';old=cached(stem)
  if old is not None:continue
  weights={r['pair']:r['weight'] for r in ps[pool]};f=raw[raw.pair.isin(weights)].copy();f['weight']=f.pair.map(weights);f['pool']=pool
  f=calibrate(f).sort_values(['pair','world','replicate','window','statistic_name'],kind='stable').reset_index(drop=True)
  save(stem,dict(frame=f,unavailable=[dict(v,pool=pool,weight=weights[v['pair']]) for v in missing if v['pair'] in weights],members=len(weights)))
  print('Completed',pool,y,'statistics; pair-folds',len(weights),flush=True)
  if y==YEARS[0] and pool==POOLS[0]:
   workers=int(os.environ.get('RESIDUAL_WORKERS','1'));elapsed=time.monotonic()-start;print(f'Measured first pool-fold (shared union contexts included): {elapsed:.1f}s; remaining train folds roughly {elapsed*(len(YEARS)-1)/workers/3600:.2f}h at workers={workers}; power/hindsight/report/hashes additional. Continues.',flush=True)
 return dict(year=y,seconds=time.monotonic()-start,union_pairs=len(ordered))

def hindsight_chunk(y,offset,rows):
 stem=f'runtime/hindsight_chunks/{y}/{offset:05d}';old=cached(stem)
 if old is not None:return old
 out=[];missing=[]
 for row in rows:
  for world,rep in [('real',0),('N2',0),('N2',1)]:
   c=context(row,y,world,rep,True);meta=dict(base_meta(row,y,world,rep),pool='P_RAND',weight=row['weight'],window='E',label='hindsight, not a screen')
   if c.get('unavailable'):missing.append(dict(meta,reason=c['reason']));continue
   mask=c['cal'].year==y;S=c['level'][mask];d=c['anchors']['A_TR']['d'][mask]
   for name,value in dict(ADF_ct=adf(S,'ct'),ECM=ecm(S),theta_21=predict(S,d,21)).items():out.append(dict(meta,statistic_name=name,**value))
  MEM.clear()
 return save(stem,dict(frame=pd.DataFrame(out),unavailable=missing))
def hindsight_fold(y):
 ps=members(y)['P_RAND'];out=[];missing=[]
 for offset in range(0,len(ps),50):
  v=hindsight_chunk(y,offset,ps[offset:offset+50]);out.append(v['frame']);missing.extend(v['unavailable'])
 value=save(f'runtime/hindsight/{y}',dict(frame=pd.concat(out,ignore_index=True),unavailable=missing));print('Completed hindsight',y,flush=True);return dict(year=y)
