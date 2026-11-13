"""Frozen two-stage calibration; no real-data computation or real price reads."""
from sc_common import *
from sc_simulation import episodes,event_values
from sc_sources import eda
import numpy as np,pandas as pd

def summarize(rows):
    if not rows:return dict(mean_net=np.nan,KM_median=np.nan,n_events=0)
    f=pd.DataFrame(rows);curve=eda().km(f);med=eda().km_quantile(curve,.5)
    return dict(mean_net=float(f.net.mean()),KM_median=float(med),n_events=len(f))

def point(job):
    stage,h,w=job;stem=f'runtime/calibration/{stage}_h{h}_w{w:.3f}'
    old=cached(stem)
    if old is not None:return old
    COUNTS.clear();started=time.monotonic();cfg=spec();N=cfg['calibration']['pair_folds_per_point'];B=cfg['calibration']['monte_carlo_batches'];sigma=cfg['calibration']['targets']['sigma_bps']
    allrows=[];means=[];medians=[]
    for b in range(B):
        block=[]
        for i in range(b*(N//B),(b+1)*(N//B)):
            es=episodes(h,w,sigma,['calibration',stage,h,round(w,3)],i)
            block.extend(event_values([e for e in es if e['window']=='train']))
        s=summarize(block);means.append(s['mean_net']);medians.append(s['KM_median']);allrows.extend(block)
    result=summarize(allrows)
    def mcse(x):return float(np.std(x,ddof=1)/np.sqrt(B)) if np.isfinite(x).all() else np.nan
    result.update(stage=stage,h=h,w=w,mean_net_mcse=mcse(means),KM_median_mcse=mcse(medians),KM_undefined_batches=int((~np.isfinite(medians)).sum()),mcse_method=f'{B} independent pair-fold batches; approximate median SE',entries_per_pair_year=len(allrows)/(N*3),pair_folds=N,elapsed_seconds=time.monotonic()-started,counts=dict(COUNTS))
    return save(stem,result)

def objective(row,baseline):
    t=spec()['calibration']['targets'];excess=row['mean_net']-baseline['mean_net']
    value=((excess-t['excess_net'])/t['excess_net'])**2+((row['KM_median']-t['KM_median'])/t['KM_median'])**2 if np.isfinite(row['KM_median']) else np.inf
    return dict(row,excess_mean_net=excess,excess_mean_net_mcse=0. if row is baseline else float(np.hypot(row['mean_net_mcse'],baseline['mean_net_mcse'])),objective=float(value),baseline_stage=baseline['stage'])

def run(workers,parallel):
    path=HERE/'outputs/tables/calibration.json'
    if path.exists():return read(path)
    hgrid=spec()['calibration']['half_lives'];coarse=parallel([(1,h,i/100) for h in hgrid for i in range(51)],point,workers)
    baseline={h:next(r for r in coarse if r['h']==h and r['w']==0) for h in hgrid}
    rows=[objective(r,baseline[r['h']]) for r in coarse];first=min(rows,key=lambda r:(r['objective'],r['h'],r['w']))
    assert np.isfinite(first['objective']),'No finite calibration objective'
    if first['w']==0:
        result=dict(status='stopped: coarse winner has no edge',stage1=rows,stage2=[],chosen=first,targets=spec()['calibration']['targets'])
    else:
        center=round(first['w']*1000);ws=[i/1000 for i in range(max(0,center-10),min(500,center+10)+1)]
        fine=parallel([(2,first['h'],w) for w in ws],point,workers);fine=[objective(r,baseline[first['h']]) for r in fine]
        best=min(fine,key=lambda r:(r['objective'],r['w']));assert np.isfinite(best['objective']),'No finite fine-grid objective';result=dict(status='calibrated',stage1=rows,stage2=fine,coarse_winner=first,chosen=best,targets=spec()['calibration']['targets'])
    write(path,result);return read(path)
