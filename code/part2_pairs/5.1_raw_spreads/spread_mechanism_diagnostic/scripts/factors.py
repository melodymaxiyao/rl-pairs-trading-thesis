"""Leave-both-legs-out daily PIT sector proxies and formation-only regression."""
import numpy as np
import pandas as pd
from mio import record,count

def sector_panel(cal,logs,classification,membership):
    sectors=sorted({r['sector'] for r in classification.values() if r.get('classification_namespace')=='GICS sub-industry' and r.get('industry') and r.get('sector')})
    sums={s:np.zeros(len(cal)) for s in sectors};counts={s:np.zeros(len(cal),int) for s in sectors}
    months=sorted(membership);snapshots=np.array(months,dtype='datetime64[D]')
    dates=cal.to_numpy(dtype='datetime64[D]');where=np.searchsorted(snapshots,dates,side='left')-1
    active={};returns={};known=where>=0
    for ticker,values in logs.items():
        r=classification.get(ticker,{})
        if r.get('classification_namespace')!='GICS sub-industry' or not r.get('industry') or not r.get('sector'):continue
        daily=np.r_[np.nan,np.diff(values)*10000];returns[ticker]=daily
        member=np.array([i>=0 and ticker in membership[months[i]] for i in where])
        eligible=member&np.isfinite(daily);active[ticker]=eligible
        sums[r['sector']]+=np.where(eligible,daily,0.);counts[r['sector']]+=eligible
    return dict(sums=sums,counts=counts,active=active,returns=returns,known=known,
                sector_of={t:classification[t]['sector'] for t in active})

def leave_out(panel,a,b,sector):
    total=panel['sums'][sector].copy();n=panel['counts'][sector].copy()
    for ticker in set([a,b]):
        if ticker not in panel['active']:continue
        eligible=panel['active'][ticker]
        # A ticker is subtracted only from its own sector.
        if panel['sector_of'][ticker]!=sector:continue
        total-=np.where(eligible,panel['returns'][ticker],0.);n-=eligible
    value=np.divide(total,n,out=np.full(len(n),np.nan),where=n>0)
    value[~panel['known']]=np.nan
    return value,n

def fit_residual(r,F,formation,score,remove_alpha=False):
    valid=np.isfinite(r)&np.isfinite(F).all(axis=1)
    used=formation&valid
    assert used.sum()>=20,'Too few formation factor observations'
    assert valid[score].all(),'Incomplete score-year factors'
    design=np.column_stack([np.ones(used.sum()),F[used]])
    coefficients,_,rank,_=np.linalg.lstsq(design,r[used],rcond=None);count('factor_regressions')
    residual=r-F@coefficients[1:]
    if remove_alpha:residual-=coefficients[0]
    return residual,dict(intercept=float(coefficients[0]),loadings=coefficients[1:].tolist(),rank=int(rank),
                         formation_sessions_used=int(used.sum()),formation_sessions_lost=int(formation.sum()-used.sum())),used

def cumulative_on_available(r):
    finite=np.flatnonzero(np.isfinite(r));assert len(finite)
    # No gaps may be silently filled once the common formation window begins.
    assert np.isfinite(r[finite[0]:]).all(),'Interior factor gap'
    result=np.full(len(r),np.nan);result[finite[0]:]=np.cumsum(r[finite[0]:])
    return result
