"""Identical fixed-beta diagnostics for real and matched Gaussian paths."""
from mio import *
from sources import helpers,universe
import numpy as np
import pandas as pd

Q=[2,5,10,20,60,120,250]
DURATION_BINS=367  # Valid for any within-calendar-year daily session index.

def fixed_spread(cal,a,b,year):
    formation=cal.year==year-4;score=(cal.year>=year-3)&(cal.year<year)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    _,beta,_=helpers().comp.ols(a[formation],b[formation]);count('formation_beta_fits')
    ra=np.r_[np.nan,np.diff(a)*10000];rb=np.r_[np.nan,np.diff(b)*10000]
    r=(ra-beta*rb)/(1+abs(beta));level=np.r_[0.,np.cumsum(r[1:])]
    return dict(beta=float(beta),r=r,level=level,ra=ra,rb=rb,formation=formation,score=score)

def variance(r,qs=Q):
    r=np.asarray(r,float);assert np.isfinite(r).all()
    value=helpers().comp.variance_ratios([np.r_[0.,np.cumsum(r)]],qs=tuple(qs));count('VR_calls')
    return {f'VR_{q}':value[f'VR_{q}'] for q in qs}

def drift(values):
    values=np.asarray(values,float);assert np.isfinite(values).all()
    mean=float(values.mean());sd=float(values.std(ddof=1))
    t=mean/(sd/np.sqrt(len(values))) if sd>0 else 0. if mean==0 else np.sign(mean)*np.inf
    return dict(mean=mean,t=float(t),sign=int(np.sign(mean)),n=len(values))

def drift_windows(cal,r,year):
    masks={'lag2':cal.year==year-3,'lag1':cal.year==year-2,
           'last6months':(cal.year==year-2)&(cal.month>=7),'A':(cal.year>=year-3)&(cal.year<=year-2)}
    outcome=drift(r[cal.year==year-1]);out={f'outcome_{k}':v for k,v in outcome.items()}
    for name,mask in masks.items():out.update({name+'_'+k:v for k,v in drift(r[mask]).items()})
    return out

def classify(position,r,ra,rb,beta,volumes=None):
    spike=False;volume_available=True
    if volumes is not None:
        checks=[]
        for volume in volumes:
            for t in [position,position-1]:
                before=volume[t-60:t]
                available=t>=60 and len(before)==60 and np.isfinite(before).all() and np.isfinite(volume[t])
                volume_available &= available
                if available:checks.append(volume[t]>=3*np.median(before))
        spike=any(checks)
    before=r[position-60:position]
    valid=position>=60 and len(before)==60 and np.isfinite(before).all()
    sd=float(np.std(before,ddof=1)) if valid else np.nan
    jump=bool(valid and abs(r[position])>=3*sd)
    raw='A' if abs(ra[position])>abs(rb[position]) else 'B' if abs(ra[position])<abs(rb[position]) else 'tie'
    weighted='A' if abs(ra[position])>abs(beta*rb[position]) else 'B' if abs(ra[position])<abs(beta*rb[position]) else 'tie'
    return dict(trigger='volume-spike' if spike else 'jump' if jump else 'gradual',
        larger_raw_leg=raw,larger_spread_contribution=weighted,volume_window_available=volume_available if volumes is not None else False,
        jump_window_available=bool(valid),entry_return_bps=float(r[position]),trailing_return_sd_bps=sd)

def annual_events(cal,a,b,year,pair,spread,volumes=None):
    h=helpers();es=universe().make_episodes(h,cal,a,b,year,pair,range(year-3,year))
    source=universe().screen_events(es,h.ho.gate().diag.dec.interval,2.)
    lookup={d:i for i,d in enumerate(cal)};events=[]
    episodes={e['year']:e for e in es}
    for event in source:
        start=lookup[event['entry_date']];end=lookup[event['end_date']]
        e=episodes[event['entry_date'].year];local=e['dates'].get_loc(event['entry_date'])
        gross=-np.sign(e['z'][local])*(spread['level'][end]-spread['level'][start])
        events.append(dict(entry_date=str(event['entry_date'].date()),end_date=str(event['end_date'].date()),duration=int(event['duration']),
            reverted=bool(event['reverted']),gross_bps=float(gross),governing_F_gross_bps=float(event['ideal_pnl_bps']),
            **classify(start,spread['r'],spread['ra'],spread['rb'],spread['beta'],volumes)))
    count('trigger_events',len(events));return events

def level_events(cal,level,formation,year):
    mu=float(np.mean(level[formation]));sd=float(np.std(level[formation],ddof=1))
    assert np.isfinite(mu) and sd>0
    z=(level-mu)/sd;events=[]
    for y in range(year-3,year):
        indices=np.flatnonzero(cal.year==y)
        for start,end,reverted in universe().positions(z[indices],2.):
            i,j=indices[start],indices[end]
            events.append(dict(duration=int(end-start),reverted=bool(reverted),gross_bps=float(-np.sign(z[i])*(level[j]-level[i]))))
    return events

def histogram(events):
    deaths=np.zeros(DURATION_BINS);censors=np.zeros(DURATION_BINS)
    for e in events:(deaths if e['reverted'] else censors)[e['duration']]+=1
    return deaths,censors
