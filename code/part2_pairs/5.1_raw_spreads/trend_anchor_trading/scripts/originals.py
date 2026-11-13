"""Import original numeric bodies unchanged with own-root IO facades."""
from aio import *

@lru_cache(None)
def modules():
    with aliases(mio=facade('mio')):
        sources=load(MECHANISM/'scripts/sources.py','_anchor_original_sources')
        km=load(MECHANISM/'scripts/statistics.py','_anchor_original_weighted_KM')
    with aliases(mio=facade('mio'),sources=sources):
        diag=load(MECHANISM/'scripts/diagnostics.py','_anchor_original_fixed_spread')
        reader=load(MECHANISM/'scripts/reader.py','_anchor_original_train_reader')
    return types.SimpleNamespace(sources=sources,diagnostics=diag,reader=reader,km=km)

def helpers():return modules().sources.helpers()
def calendar(year):
    h=modules().sources.set_year(year);return h.comp.CAL.copy()
def original_episodes(cal,a,b,year,pair):
    return modules().sources.universe().make_episodes(helpers(),cal,a,b,year,pair,range(year-3,year))
def positions(z):return modules().sources.universe().positions(z,2.)
def original_events(episodes):
    return modules().sources.universe().screen_events(episodes,helpers().ho.gate().diag.dec.interval,2.)
def fixed_spread(cal,a,b,year):return modules().diagnostics.fixed_spread(cal,a,b,year)
def read_prices(path,year):return modules().reader.read_train(note(path),year,('Adj Close',))
def read_sample(year):return modules().reader.read_sample(year)
def gaussian(length,sa,sb,rho,rng):return modules().sources.null_generator()(length,sa,sb,rho,rng)

def product_limit(d,c):
    import numpy as np
    p,median=modules().km.product_limit(d,c)
    median[np.atleast_2d(np.asarray(d)+np.asarray(c)).sum(axis=1)==0]=np.nan
    return p,median
