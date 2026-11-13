"""Causal/hindsight deviations and governing non-overlapping event accounting."""
from aio import count
from originals import helpers,positions,original_episodes
import numpy as np
import pandas as pd

ANCHORS=['A_F','A_R252','A_R63','A_TR','H_Y','H_W','F_orig']
INFEASIBLE={'H_Y','H_W'}
ADAPTIVE={'A_R252','A_R63','A_TR'}
HORIZONS=[5,21,63,126]
BINS=1100

def rolling(level,n,trend=False):
    level=np.asarray(level,float);N=len(level);mu=np.full(N,np.nan);sd=mu.copy()
    if N<=n:return mu,sd
    windows=np.lib.stride_tricks.sliding_window_view(level,n)[:-1]
    means=windows.mean(axis=1);centered=windows-means[:,None]
    if trend:
        x=np.arange(n,dtype=float)-(n-1)/2;slopes=centered@x/(x@x)
        residual=centered-slopes[:,None]*x
        mu[n:]=means+slopes*(n-(n-1)/2);sd[n:]=residual.std(axis=1,ddof=1)
        count('trailing_trend_fits',len(windows))
    else:mu[n:]=means;sd[n:]=centered.std(axis=1,ddof=1)
    return mu,sd

def anchors(cal,level,year,episodes):
    N=len(cal);out={}
    for anchor in ANCHORS:
        mu=np.full(N,np.nan);sd=mu.copy();z=mu.copy()
        if anchor=='F_orig':
            for e in episodes:z[cal.year==e['year']]=e['z']
            out[anchor]=dict(m=mu,s=sd,d=z);continue
        if anchor in ['A_F','H_Y']:
            for y in range(year-3,year):
                source=cal.year==(y-1 if anchor=='A_F' else y);target=cal.year==y
                mu[target]=level[source].mean();sd[target]=level[source].std(ddof=1)
        elif anchor in ['A_R252','A_R63','A_TR']:
            mu,sd=rolling(level,63 if anchor=='A_R63' else 252,anchor=='A_TR')
        elif anchor=='H_W':
            mask=(cal.year>=year-3)&(cal.year<year);x=np.arange(mask.sum(),dtype=float)
            alpha,beta,residual=helpers().comp.ols(level[mask],x);count('hindsight_trend_fits')
            mu[mask]=alpha+beta*x;sd[mask]=residual.std(ddof=1)
        with np.errstate(invalid='ignore',divide='ignore'):z=(level-mu)/sd
        z[~np.isfinite(z)|(sd<=0)]=np.nan;out[anchor]=dict(m=mu,s=sd,d=z)
    return out

def drift_alignment(r,entry,direction):
    prior=np.asarray(r,float)[entry-252:entry]
    if entry<252 or len(prior)!=252 or not np.isfinite(prior).all():return np.nan,np.nan
    mean=float(prior.mean());return (1 if direction*np.sign(mean)>0 else -1),mean

def driver(S,anchor,entry,exit):
    d=anchor['d'];m=anchor['m'];s=anchor['s'];delta=d[exit]-d[entry];q=-np.sign(d[entry])
    numerator=S[exit]-S[entry];denominator=s[entry]*delta
    assert np.isfinite([delta,denominator,numerator]).all() and denominator!=0
    pieces=[numerator/s[entry],-(m[exit]-m[entry])/s[entry],(S[exit]-m[exit])*(1/s[exit]-1/s[entry])]
    assert np.isclose(sum(pieces),delta,rtol=1e-10,atol=1e-8)
    share=numerator/denominator
    return dict(price_share=share,anchor_and_scale_share=1-share,
        driver_numerator=q*numerator,driver_denominator=q*denominator,
        deviation_change=delta,price_deviation_part=pieces[0],anchor_deviation_part=pieces[1],scale_deviation_part=pieces[2],
        price_component_share=pieces[0]/delta,anchor_component_share=pieces[1]/delta,scale_component_share=pieces[2]/delta,
        driver_outside_unit_interval=bool(share<0 or share>1))

def events(cal,level,r,anchor,anchor_id,regime,year):
    assert not (anchor_id=='F_orig' and regime=='episode-free')
    score=(cal.year>=year-3)&(cal.year<year);all_indices=np.flatnonzero(score);last=all_indices[-1]
    blocks=[np.flatnonzero(cal.year==y) for y in range(year-3,year)] if regime=='within-year' else [all_indices]
    result=[];coverage=dict(scoring_sessions=int(score.sum()),unavailable_deviation_sessions=int((score&~np.isfinite(anchor['d'])).sum()))
    for indices in blocks:
        valid=np.isfinite(anchor['d'][indices]);where=np.flatnonzero(valid)
        if not len(where):continue
        assert valid[where[0]:].all(),'Interior deviation gap'
        indices=indices[where[0]:];z=anchor['d'][indices]
        for start,end,reverted in positions(z):
            i,j=int(indices[start]),int(indices[end]);direction=-int(np.sign(z[start]));gross=direction*(level[j]-level[i])
            alignment,prior=drift_alignment(r,i,direction)
            row=dict(anchor=anchor_id,feasibility='INFEASIBLE' if anchor_id in INFEASIBLE else 'CAUSAL',regime=regime,
                entry_index=i,exit_index=j,entry_date=str(cal[i].date()),exit_date=str(cal[j].date()),duration=int(j-i),
                direction=direction,d_entry=float(z[start]),d_exit=float(z[end]),reverted=bool(reverted),
                S_entry=float(level[i]),S_exit=float(level[j]),m_entry=float(anchor['m'][i]),m_exit=float(anchor['m'][j]),
                s_entry=float(anchor['s'][i]),s_exit=float(anchor['s'][j]),
                censored=not reverted,gross_bps=float(gross),net_bps=float(gross-10),hit=bool(gross>10),
                drift_alignment=alignment,prior_drift_mean_bps=prior,year_crossing=bool(cal[i].year!=cal[j].year))
            count('event_pnl_evaluations')
            for h in HORIZONS:
                end_bound=indices[-1] if regime=='within-year' else last
                row[f'forward_{h}']=float(direction*(level[i+h]-level[i])) if i+h<=end_bound else np.nan
                row[f'forward_{h}_excluded']=bool(i+h>end_bound)
                row[f'forward_{h}_exclusion_reason']='' if i+h<=end_bound else 'year_end' if regime=='within-year' else 'window_end'
            if reverted and anchor_id in ADAPTIVE:row.update(driver(level,anchor,i,j))
            result.append(row)
    return result,coverage

def calculate(cal,level,r,year,episodes):
    surfaces=anchors(cal,level,year,episodes);rows=[];coverage=[]
    for name in ANCHORS:
        for regime in ['within-year','episode-free']:
            if name=='F_orig' and regime=='episode-free':continue
            ev,cov=events(cal,level,r,surfaces[name],name,regime,year)
            rows.extend(ev);coverage.append(dict(anchor=name,feasibility='INFEASIBLE' if name in INFEASIBLE else 'CAUSAL',regime=regime,**cov))
    return rows,coverage
