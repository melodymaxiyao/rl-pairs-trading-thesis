"""Original block resampling, paired means and centred one-sided inference."""
from rw_io import *
from sources import original
import numpy as np
import pandas as pd

@lru_cache(None)
def bootstrap_source():
    unit=types.SimpleNamespace(unit_name=lambda pool,year,state:f'production_{pool}_{year}_{state}')
    with aliases(baseline_io=facade('baseline_io'),frozen_sources=original(),jobs=unit):
        return load(BASELINE/'scripts/analysis_core.py','_rw_original_resampling')

def estimate(frame,columns,scheme='year',draws=5000,seed=20261022):
    if frame.empty:return {k:dict(estimate=None,ci95_low=None,ci95_high=None,p_positive=None,p_negative=None,n=0) for k in columns}
    key='test_year' if scheme=='year' else 'cluster'
    names,w=bootstrap_source().sample_weights(frame[key],draws,seed)
    a=frame[columns].to_numpy(float);labels=frame[key].to_numpy()
    sums=np.array([np.nansum(a[labels==name],axis=0) for name in names])
    counts=np.array([np.isfinite(a[labels==name]).sum(axis=0) for name in names])
    den=w@counts;values=np.divide(w@sums,den,out=np.full((draws,len(columns)),np.nan),where=den>0)
    result={}
    for j,column in enumerate(columns):
        point=float(frame[column].mean());v=values[:,j];valid=v[np.isfinite(v)];defined=len(names)>=2 and len(valid)>0 and np.isfinite(point)
        lo,hi=np.quantile(valid,[.025,.975]) if defined else (None,None)
        result[column]=dict(estimate=point,ci95_low=lo,ci95_high=hi,
            p_positive=float((1+np.sum(v-point>=point)+np.sum(~np.isfinite(v)))/(draws+1)) if defined else None,
            p_negative=float((1+np.sum(v-point<=point)+np.sum(~np.isfinite(v)))/(draws+1)) if defined else None,
            n=int(np.isfinite(a[:,j]).sum()),blocks=len(names),undefined_draws=int((~np.isfinite(v)).sum()))
    record('bootstrap',bootstrap_draws=draws,bootstrap_statistics=len(columns),scheme=scheme)
    return clean(result)

def holm(ps):
    return original().holm_source()([float(p) if p is not None else 1. for p in ps])

def family(real,null):
    metrics=['gross_bps','net_bps','drift_bps','timing_bps']
    r=real[(real.pool=='EXP_G_cap')&(real.view=='NEW')&(real.cost==10)]
    n=null[(null.pool=='EXP_G_cap')&(null.variant=='primary')&(null.cost==10)]
    n=n.groupby(['pair','test_year','cell'],as_index=False)[metrics].mean()
    mapping=read(note(BASELINE/'runtime/clusters.json'))['EXP_G_cap']['pair_cluster']
    results=[]
    for scheme in ['year','cluster']:
        base=n[n.cell=='B2'].copy();base['cluster']=base.pair.map(mapping);assert base.cluster.notna().all()
        stats=estimate(base,['drift_bps','timing_bps','gross_bps'],scheme)
        supported=(stats['drift_bps']['ci95_high'] is not None and stats['drift_bps']['ci95_high']<0
                   and stats['timing_bps']['ci95_low']>0 and stats['gross_bps']['ci95_low']<=0<=stats['gross_bps']['ci95_high'])
        results.append(dict(id='M1',scheme=scheme,cell='B2',metric='gross_bps',**stats['gross_bps'],
            null_drift=stats['drift_bps'],null_timing=stats['timing_bps'],supported=supported,in_Holm=False))
        rows=[]
        for ident,cell,metric in [('M2','B2','timing_bps'),('M3','z9/TD_ens','timing_bps'),('M4','B2','gross_bps'),('M5','H1s','net_bps')]:
            a=r[r.cell==cell][['pair','test_year',metric]].merge(n[n.cell==cell][['pair','test_year',metric]],on=['pair','test_year'],suffixes=('_real','_null'),validate='one_to_one')
            a['delta']=a[metric+'_real']-a[metric+'_null'];a['cluster']=a.pair.map(mapping)
            stat=estimate(a,['delta'],scheme)['delta']
            rows.append(dict(id=ident,scheme=scheme,cell=cell,metric=metric,**stat,in_Holm=True))
        adjusted=holm([r['p_positive'] for r in rows])
        for row,p in zip(rows,adjusted):row.update(p_raw=row['p_positive'],p_Holm=float(p),supported=bool(row['estimate'] is not None and row['estimate']>0 and p<=.05))
        results.extend(rows)
    return results
