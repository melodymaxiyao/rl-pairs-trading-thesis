import f1_core as A
C=A.C;S=A.S;np=A.np;pd=A.pd;P=A.P
import gzip,io,json

def ic(x,y,w):return A.NS['spearman'](x,y,w)

def year(kind,year):
 root=C.P/'F1_daily';root.mkdir(exist_ok=True);out=root/f'{kind}_{year}.csv.gz'
 if out.exists():raise RuntimeError(('refuse to overwrite F1 year',out))
 records=[]
 for z in A.iter_batches(kind,year):
  C.resources();t=z['t'];date=str(z['dates'][t]);w=z['weights'];old=z['old'];h=z['h']
  worlds=[('real',-1,None)]+[('N2',r,A.signs(kind,year,z['names'],z['dates'][t-59:t+1],r)) for r in range(2)]+[('N3_core',r,A.signs(kind,year,z['names'],z['dates'][t-59:t+1],r)) for r in range(20)]
  for world,rep,sg in worlds:
   if world=='real':e=old;score=z['stored_score'][0]
   elif world=='N2':e=old*sg;score=z['stored_score'][rep+1]
   else:
    e=h*sg;e=e-e.mean(axis=-1,keepdims=True);score=A.NS['ou_fit'](e)['s']
   for hi,H in enumerate([1,5]):
    if not np.isfinite(z['forward'][hi]).any():continue
    if world=='real':y=z['forward'][hi].sum(axis=-1)
    else:y=(z['forward'][hi]*A.signs(kind,year,z['names'],z['dates'][t+1:t+H+1],rep)).sum(axis=-1)
    for signal,x in [('s_roll',-score),('neg_e',-e[:,-1])]:
     value=ic(x,y,w);n=int(np.sum(np.isfinite(x)&np.isfinite(y)&(w>0)))
     records.append({'family':kind,'year':year,'date':date,'signal':signal,'h':H,'world':world,'replica':rep,'IC':value if np.isfinite(value) else '', 'n':n,'scope':'core_headline_time_t_window','robustness_check':True})
  if z['j']%60==0:print('F1 IC',kind,year,z['j'],'/',len(z['ts']),flush=True)
 pd.DataFrame(records).to_csv(out,index=False,compression='gzip',float_format='%.17g')
 print('F1 IC complete',kind,year,len(records),flush=True)

def bootstrap_summary(real,null,dates):
 years=np.asarray(dates,dtype='datetime64[D]').astype('datetime64[Y]').astype(int)+1970
 uq=sorted(set(years));assert len(uq)==22
 d=real-null.mean(axis=1);s=np.array([d[years==y].sum() for y in uq]);n=np.array([np.sum(years==y) for y in uq]);ix=np.random.default_rng(20261026).integers(22,size=(5000,22));boot=s[ix].sum(axis=1)/n[ix].sum(axis=1);lo,hi=np.percentile(boot,[2.5,97.5])
 return dict(real=float(real.mean()),null_mean=float(null.mean()),excess=float(d.mean()),ci95_low=float(lo),ci95_high=float(hi),n_common_days=len(d),years=len(uq),MC_standard_error=float(null.mean(axis=0).std(ddof=1)/np.sqrt(null.shape[1])))

def summarize():
 frames=[pd.read_csv(p,compression='gzip',keep_default_na=False,float_precision='round_trip') for p in sorted((C.P/'F1_daily').glob('*.csv.gz'))];assert len(frames)==44
 df=pd.concat(frames,ignore_index=True);df['IC']=pd.to_numeric(df.IC,errors='coerce')
 source=P/'outputs/A1_sanity_v2/base_report/summary.csv.gz';old=pd.read_csv(io.BytesIO(C.raw(source)),compression='gzip',keep_default_na=False,float_precision='round_trip')
 rows=[];baseline=[]
 for kind in ['RP','RF']:
  for sig in ['s_roll','neg_e']:
   for h in [1,5]:
    f=df[(df.family==kind)&(df.signal==sig)&(df.h==h)];cols=[('real',-1)]+[('N2',r) for r in range(2)]+[('N3_core',r) for r in range(20)]
    v=f.pivot(index='date',columns=['world','replica'],values='IC');a=v[[('real',-1),('N2',0),('N2',1)]].dropna();b=v[[('real',-1)]+[('N3_core',r) for r in range(20)]].dropna()
    r2=bootstrap_summary(a[('real',-1)].to_numpy(),a[[('N2',0),('N2',1)]].to_numpy(),a.index)
    r3=bootstrap_summary(b[('real',-1)].to_numpy(),b[[('N3_core',r) for r in range(20)]].to_numpy(),b.index)
    if sig=='s_roll':
     found=old[(old.family==kind)&(old.signal=='core_headline')&(old.scope=='earlier_chapter_original_scope')&(old.target=='tradable')&(old.h==h)];assert len(found)==1;ref=found.iloc[0]
     for k,other in [('real','real'),('null_mean','N2_mean'),('excess','real_minus_N2'),('ci95_low','ci95_low'),('ci95_high','ci95_high')]:np.testing.assert_allclose(r2[k],float(ref[other]),rtol=0,atol=1e-12,err_msg=f'N2 headline summary {kind}/{h}/{k}')
     assert r2['n_common_days']==int(ref.common_days);baseline.append({'family':kind,'h':h,'exact_headline_N2_summary_reproduced':True,'common_days':int(ref.common_days)})
    diff=abs(r3['excess']-r2['excess'])/abs(r2['excess']) if r2['excess']!=0 else None
    verdict='descriptive only'
    if sig=='s_roll' and h==5:
     if r3['ci95_low']>0:verdict='对零基准构造稳健' if diff is not None and diff<=.5 else '仍显著，但幅度对零基准敏感'
     elif r3['ci95_low']<=0<=r3['ci95_high']:verdict='不稳健'
     else:verdict='不稳健；CI全负'
    rows.append({'signal':sig,'family':kind,'h':h,'N2_excess_IC':r2['excess'],'N2_ci95_low':r2['ci95_low'],'N2_ci95_high':r2['ci95_high'],'N2_replicas':2,'N3_version':'N3_core','N3_excess_IC':r3['excess'],'N3_ci95_low':r3['ci95_low'],'N3_ci95_high':r3['ci95_high'],'N3_replicas':20,'N3_null_mean_MCSE':r3['MC_standard_error'],'relative_change_from_N2':diff if diff is not None else '', 'decision':verdict,'N2_common_days':r2['n_common_days'],'N3_common_days':r3['n_common_days'],'N2_source':'reproduced sealed core_headline summary' if sig=='s_roll' else 'new same-definition core last-increment comparator; historical causal neg_e not substituted','robustness_check':True})
 C.csvout(C.P/'F1_N2_vs_N3.csv',rows);C.dump(C.P/'audit/F1_N2_headline_reproduction.json',baseline)
 return rows

if __name__=='__main__':
 assert C.js(C.P/'audit/F1_core_validation.json')['passes'];assert C.js(C.P/'audit/F1_original_N3_validation.json')['passes'];assert C.js(C.P/'audit/unit_tests.json')['passes']
 C.dump(C.P/'audit/F1_inputs_before.json',dict(C.READS))
 try:
  for kind in ['RP','RF']:
   for yr in range(2004,2026):year(kind,yr)
  result=summarize();print('F1 COMPLETE',result,flush=True)
 finally:C.dump(C.P/'audit/F1_compute_inputs.json',C.READS)
