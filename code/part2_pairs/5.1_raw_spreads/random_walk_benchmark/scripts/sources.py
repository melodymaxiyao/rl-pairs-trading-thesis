"""Unchanged source imports with own-root IO and forbidden fit/planning entrypoints."""
from rw_io import *
import numpy as np

@lru_cache(None)
def original():
    native=note(BASELINE/'runtime/q_kernel.so');destination=HERE/'runtime/q_kernel.so'
    if not destination.exists():
        with destination.open('xb') as stream:stream.write(native.read_bytes())
    assert sha(destination)==sha(native)
    with aliases(baseline_io=facade('baseline_io')):
        module=load(BASELINE/'scripts/frozen_sources.py','_rw_original_baseline_sources')
    for name in ['original_td','td_producer','model','plan','mb']:setattr(module,name,forbidden)
    n=module.numerical();n.core.fit=forbidden;n.core.kernel().train=forbidden
    return module

@lru_cache(None)
def helpers():
    with aliases(sector_io=facade('sector_io')):
        module=load(SECTOR/'scripts/frozen_helpers.py','_rw_original_F_G_helpers')
    return module.helpers()

@lru_cache(None)
def fast_modules():
    screen=load(FAST/'scripts/screen.py','_rw_original_fast_screen')
    with aliases(screen=screen):events=load(FAST/'scripts/events.py','_rw_original_fast_events')
    reader=load(FAST/'scripts/train_reader.py','_rw_original_date_first_reader')
    io=facade('universe_io',helpers=helpers)
    with aliases(universe_io=io,screen=screen,events=events,train_reader=reader):
        numeric=load(FAST/'scripts/numeric.py','_rw_original_fast_numeric')
    checkpoint=types.SimpleNamespace(ready=checkpoint_false,save_rows=forbidden,load_context=forbidden)
    with aliases(universe_io=io,checkpoints=checkpoint,numeric=numeric,events=events):
        worker=load(FAST/'scripts/worker.py','_rw_original_universe_null_generator')
    return types.SimpleNamespace(screen=screen,events=events,reader=reader,numeric=numeric,worker=worker)

def checkpoint_false(*args):return False

def bins(episode):
    return dict(episode,bins=np.searchsorted(original().numerical().REG['z_edges'],episode['z'],side='right').astype(np.int32))

def episodes(cal,la,lb,year,pair):
    h=helpers()
    es=fast_modules().numeric.make_episodes(h,cal,la,lb,year,pair,range(year-3,year+1))
    return [bins(dict(e,window='train' if e['year']<year else 'test')) for e in es]

def training_pass(es,kind):
    assert len(es)==3 and all(e['window']=='train' for e in es)
    if kind=='ALL':return True
    if kind=='G':
        ho=helpers().ho;events=ho.gate_events(es,3)
        if events.empty:return False
        return bool(ho.decide(events,[(es[0]['pair'],es[0]['fold'])]).iloc[0]['passes'])
    assert kind=='FAST63'
    stats,_,_=fast_modules().numeric.episode_statistics(helpers(),es)
    return bool(stats['FAST63'])

def limits(es):
    n=original().numerical();n.data.require_train(es)
    events=n.eda.entry_events(es,1.5);value=n.eda.derive_limits(events)
    record('null_limits',derive_limits_calls=1,train_events=len(events),real_test_numeric_rows=0)
    return dict(H=int(value['H']),stop=float(value['z_stop']),lock=True,incremental=False,nbins=9),value

def decomposition():return original().decomposition().decompose
