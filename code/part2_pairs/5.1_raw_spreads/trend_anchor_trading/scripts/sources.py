"""Unchanged numerical sources with own-root IO routing; main-only imports."""
from aio import *
import numpy as np,pandas as pd

@lru_cache(None)
def baseline_sources():
    with aliases(baseline_io=facade('baseline_io')):
        return load(BASELINE/'scripts/frozen_sources.py','_ta_original_baseline')

def core():return baseline_sources().numerical().core

def eda():return baseline_sources().numerical().eda

@lru_cache(None)
def anchor_jobs():
    import originals,engine
    with aliases(aio=facade('aio',SEED=20261024),originals=originals,engine=engine):
        return load(ANCHOR/'scripts/jobs.py','_ta_original_anchor_jobs')

def anchor_sample(year):return anchor_jobs().sample(year)

@lru_cache(None)
def inventory():
    rows=read(note(SECTOR/'outputs/tables/pair_fold_inventory.json'))
    return [dict(r) for r in rows if r['EXP_G_cap'] and not r['existing']]

@lru_cache(maxsize=1)
def baseline_fold(year):
    from store import pickle_read
    return pickle_read(note(BASELINE/f'runtime/prepared/{year}.pkl'))

def legacy_episode(pair,year,score_year):
    b=baseline_fold(year)
    e=next(e for e in b['episodes'] if e['pair']==pair and e['year']==score_year)
    return dict(e)

def legacy_risk(year):return baseline_fold(year)['pools']['EXP_G_cap']['risks']['F']['final']

def original_inference():return baseline_sources().inference()

def decompose(p,r,c,sign=1):
    p=np.atleast_2d(np.asarray(p,float));g=p*r;fees=np.abs(np.diff(p,axis=1,prepend=0))*c/2
    out=baseline_sources().decomposition().decompose(p,r,g,fees,sign or 1)
    if not sign:
        out['train_drift_sign']=0
        for k in ['drift_aligned_share','aligned_exposure','test_direction_agrees','test_direction_zero']:out[k]=None
    return out

@lru_cache(None)
def original_kernel_fixture_source():
    # Match the unchanged producer at the same saved fit level, not just metrics.
    return baseline_sources().td_producer()
