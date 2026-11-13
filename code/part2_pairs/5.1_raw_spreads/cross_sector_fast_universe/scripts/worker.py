from universe_io import *
from checkpoints import ready,save_rows,load_context
from numeric import evaluate_pair,make_episodes,episode_statistics
from events import screen_events,summarize
import numpy as np
import os

@lru_cache(maxsize=1)
def context(year):
    return load_context(year)

def simulate_logs(length,year,index,volatilities):
    rng=np.random.default_rng(np.random.SeedSequence([20261020,year,index]))
    choice=rng.integers(0,len(volatilities),size=2)
    sa,sb=volatilities[choice[0]],volatilities[choice[1]]
    la=np.r_[np.log(100.),np.log(100.)+np.cumsum(rng.normal(0,sa,length-1))]
    lb=np.r_[np.log(100.),np.log(100.)+np.cumsum(rng.normal(0,sb,length-1))]
    return la,lb,sa,sb

def real_chunk(year,number,rows):
    os.environ['FAST_SCOPE']='production_real'
    guard()
    marker=HERE/f'runtime/checkpoints/real_{year}_{number:05d}.json'
    if ready(marker):return str(marker)
    ctx=context(year)
    values=[evaluate_pair(ctx,r) for r in rows]
    record('real_chunk_completed',candidate_pair_folds=len(values),eligible_pair_folds=sum(r['eligible'] for r in values))
    save_rows(f'real_{year}_{number:05d}',values,test_year=year,chunk=number,kind='real')
    return str(marker)

def null_chunk(year,number,indices,volatilities):
    os.environ['FAST_SCOPE']='production_null'
    guard()
    marker=HERE/f'runtime/checkpoints/null_{year}_{number:05d}.json'
    if ready(marker):return str(marker)
    ctx=context(year); h=ctx.h; output=[]
    for index in indices:
        la,lb,sa,sb=simulate_logs(len(ctx.cal),year,index,volatilities)
        es=make_episodes(h,ctx.cal,la,lb,year,f'NULL_{index:04d}',range(year-3,year))
        final,_,_=episode_statistics(h,es)
        inner,_,_=episode_statistics(h,es[:2],True)
        validation=summarize(screen_events(es[-1:],h.ho.gate().diag.dec.interval))
        result=dict(test_year=year,era=era(year),simulation=index,eligible=True,inner_eligible=True,
                    sigma_A=sa,sigma_B=sb,validation_KM_med=validation['KM_med'],**final,
                    **{'inner_'+k:v for k,v in inner.items()})
        output.append(result)
        record('null_pair_simulated',null_pairs=1,test_rows_used=0)
    save_rows(f'null_{year}_{number:05d}',output,test_year=year,chunk=number,kind='null')
    return str(marker)
