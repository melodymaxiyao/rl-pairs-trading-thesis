"""Synthetic partial-cointegration paths with per-pair deterministic streams."""
from sc_common import *
from sc_sources import anchor,market,eda
import numpy as np,pandas as pd
@lru_cache(None)
def calendar():return pd.DatetimeIndex(np.concatenate([pd.bdate_range(str(y)+'-01-01',periods=252).values for y in range(2000,2005)]))
def innovations(phi,w,sigma):return np.sqrt(w*sigma*sigma*(1+phi)/2),sigma*np.sqrt(1-w)
def path(h,w,sigma,key,i,student=False):
    rng=np.random.default_rng(stable_seed('path',key,int(i)))
    phi=float(np.exp(-np.log(2)/h));sm,sr=innovations(phi,w,sigma);n=1260
    draw=lambda size:rng.standard_t(4,size)/np.sqrt(2) if student else rng.standard_normal(size)
    if student:
        m=0.;burn=int(np.ceil(100*h))
        for x in draw(burn):m=phi*m+sm*x
        count('burn_in_steps',burn)
    else:m=float(rng.normal()*sm/np.sqrt(1-phi*phi))
    em,er=draw(n-1)*sm,draw(n-1)*sr;M=np.empty(n);M[0]=m
    for t,x in enumerate(em,1):M[t]=phi*M[t-1]+x
    S=M+np.r_[0.,np.cumsum(er)];r=np.r_[np.nan,np.diff(S)]
    count('synthetic_paths');return S,r

def episodes(h,w,sigma,key,i,student=False,heterogeneous=False,only_test=False):
    typ=i%2 if heterogeneous else 1
    wi=(0. if typ==0 else 2*w) if heterogeneous else w
    S,r=path(h,wi,sigma,key,i,student)
    m,s=anchor().rolling(S,252,True);d=(S-m)/s;drift,strong,t=market().drift_surface(r);cal=calendar()
    es=[]
    for y in [2001,2002,2003,2004]:
        ix=np.flatnonzero(cal.year==y)
        e=dict(pair=f'synthetic_{i:06d}',year=y,window='test' if y==2004 else 'train',fold=1,construction='F',dates=cal[ix],z=d[ix],forward_bps=np.r_[np.diff(S[ix]),0.],drift=drift[ix],strong=strong[ix],drift_t=t[ix],previous_z=float(d[ix[0]-1]),beta=np.ones(252),true_type=typ,level=S[ix])
        assert np.isfinite(e['z']).all();es.append(e)
    if heterogeneous:
        ev=event_values([e for e in es if e['window']=='train'])
        score=float(np.mean([r['net'] for r in ev])) if ev else np.nan
        for e in es:e['proxy_score']=score
    return [e for e in es if e['window']=='test'] if only_test else es

def training(h,w,sigma,key,n=600,student=False,heterogeneous=False):
    es=[]
    for i in range(n):es.extend(e for e in episodes(h,w,sigma,key,i,student,heterogeneous) if e['window']=='train')
    return es

def test_batches(h,w,sigma,key,n=5000,student=False,heterogeneous=False,size=100):
    for start in range(0,n,size):
        yield [episodes(h,w,sigma,key,i,student,heterogeneous,True)[0] for i in range(start,min(n,start+size))]

def risk_limits(es):
    events=eda().entry_events(es,1.5);r=eda().derive_limits(events)
    return dict(H=int(r['H']),stop=float(r['z_stop']),lock=True,incremental=False,nbins=9)

def event_values(es):
    """Same governing within-year event function, without driver overhead."""
    out=[]
    from sc_sources import positions
    for e in es:
        level=e['level']
        for i,j,rev in positions(e['z']):
            out.append(dict(net=float(-np.sign(e['z'][i])*(level[j]-level[i])-10),duration=j-i,reverted=rev))
    return out
