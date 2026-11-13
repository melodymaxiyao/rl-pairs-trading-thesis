"""Load sealed numerical bodies unchanged, with IO-only module facades."""
from sc_common import *
import importlib.util,types,contextlib
import numpy as np,pandas as pd
from functools import lru_cache

@contextlib.contextmanager
def aliases(**items):
    old={k:sys.modules.get(k) for k in items};sys.modules.update(items)
    try:yield
    finally:
        for k,v in old.items():
            if v is None:sys.modules.pop(k,None)
            else:sys.modules[k]=v

def load(path,name):
    registry=read(HERE/'audit/source_registration.json')
    rel=str(Path(path).relative_to(ROOT));assert sha(path)==registry[rel],('Source hash mismatch',rel)
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

def facade(name,**values):
    m=types.ModuleType(name);m.__dict__.update(dict(np=np,pd=pd,Path=Path,lru_cache=lru_cache,ctypes=__import__('ctypes'),subprocess=__import__('subprocess'),time=time,os=os,sys=sys,hashlib=hashlib,json=json,types=types,count=count,record=record,sha=sha,read=read,write=write));m.__dict__.update(values);return m

@lru_cache(None)
def native():
    with aliases(aio=facade('aio',HERE=SOURCE,V2=PT/'qlearning_zpos_v2')):
        return load(SOURCE/'scripts/native.py','_sc_original_native')

@lru_cache(None)
def anchor():
    # Only rolling/events/drift_alignment are invoked. The old reader and price
    # entry points are not imported into this synthetic executor.
    originals=facade('originals',helpers=lambda:None,positions=positions,original_episodes=lambda *a:None)
    with aliases(aio=facade('aio'),originals=originals):
        return load(SOURCE/'scripts/engine.py','_sc_original_anchor')

@lru_cache(None)
def market():
    with aliases(aio=facade('aio'),store=facade('store',cached=lambda *a:None,commit=lambda stem,value:value),originals=facade('originals',helpers=lambda:None,fixed_spread=None,original_episodes=None,gaussian=None),engine=anchor()):
        return load(SOURCE/'scripts/market.py','_sc_original_market')

def require_train(es):assert es and all(e['window']=='train' for e in es)
def h1_sign(es):
    require_train(es);s={}
    for e in es:s[e['pair']]=s.get(e['pair'],0)+float(e['forward_bps'].sum())
    return {p:int(np.sign(v)) for p,v in s.items()}
def data_facade():return facade('data',HERE=PT/'qlearning_zpos_v2',REG=read(PT/'qlearning_zpos_v2/preregistration.json'),require_train=require_train,gates=lambda:pd.DataFrame(columns=['pair','class']),rebinned=lambda e,coarse=False:e,h1_sign=h1_sign)
@lru_cache(None)
def core():
    with aliases(data=data_facade()):return load(PT/'qlearning_zpos_v2/scripts/core.py','_sc_original_core')
@lru_cache(None)
def eda():
    with aliases(data=data_facade()):return load(PT/'qlearning_zpos_v2/scripts/eda.py','_sc_original_eda')

@lru_cache(None)
def positions_function():
    with aliases(screen=facade('screen',km_quantiles=None)):
        return load(PT/'cross_sector_fast_universe_v1/scripts/events.py','_sc_original_events').positions
def positions(z):return positions_function()(z,2.)

def selection(train,risks,fit_func,gammas=(.90,.99)):
    """Original v3 select body; only pool membership/IO/fit are adapted."""
    from copy import copy
    original=core();proxy=types.SimpleNamespace(**original.__dict__);proxy.fit=fit_func
    reg=dict(read(PT/'qlearning_zpos_v2/preregistration.json'));reg['gamma_grid']=list(gammas)
    pairs=sorted({e['pair'] for e in train})
    def pooled(es,pool,fold,pairs):return [e for e in es if e['window']=='train' and e['pair'] in set(pairs)]
    prepare=facade('q3prepare',prepared=None,pools_saved=None,costs_lookup=lambda c:{},pool_train=pooled,fold_years=lambda es,f:sorted({e['year'] for e in es if e['window']=='train'}),eligible=lambda c,f:pairs,POOL_SPECS={'synthetic':{'construction':'F'}})
    source=facade('q3source',v2=lambda:types.SimpleNamespace(core=proxy,data=data_facade(),REG=reg),risk_settings=lambda pool,fold,scenario,phase:risks[phase],ledger_costs=None)
    common=facade('q3common',POOLS=HERE/'runtime',TABLES=HERE/'outputs/tables')
    with aliases(q3common=common,q3source=source,q3prepare=prepare):
        m=load(PT/'qlearning_zpos_v3/scripts/q3jobs.py','_sc_original_select')
    return m.select(train,'synthetic','primary',1,{'inner':(pairs,False),'final':(pairs,False)},[],tag='synthetic')
