"""Original F/OLS/VR/AR and screen bodies; original Gaussian log generator."""
from mio import *

@lru_cache(None)
def helpers():
    with aliases(sector_io=facade('sector_io')):module=load(SECTOR/'scripts/frozen_helpers.py','_mech_original_helpers')
    return module.helpers()

def set_year(year):
    h=helpers();h.comp.CAL=h.full_cal[(h.full_cal.year>=year-4)&(h.full_cal.year<year)]
    return h

@lru_cache(None)
def universe():
    stub=types.ModuleType('train_reader')
    stub.read_window=stub.complete_test=stub.admit=forbidden
    screen=load(FAST/'scripts/screen.py','_mech_original_screen')
    with aliases(screen=screen):events=load(FAST/'scripts/events.py','_mech_original_events')
    with aliases(universe_io=facade('universe_io',set_year=set_year),train_reader=stub,screen=screen,events=events):
        numeric=load(FAST/'scripts/numeric.py','_mech_original_numeric')
    numeric.FoldData=numeric.evaluate_pair=forbidden
    return types.SimpleNamespace(make_episodes=numeric.make_episodes,screen_events=events.screen_events,
                                 positions=events.positions,km_quantiles=screen.km_quantiles)

@lru_cache(None)
def null_generator():
    stub=types.ModuleType('sources')
    stub.helpers=stub.fast_modules=stub.episodes=stub.training_pass=forbidden
    with aliases(rw_io=facade('rw_io'),sources=stub):module=load(RWB/'scripts/simulation.py','_mech_original_gaussian_draw')
    return module.draw_logs

@lru_cache(None)
def weighted_helpers():
    return load(PROPERTY/'scripts/pp_stats.py','_mech_original_weighted_stats')

def seed_stream(pair,year,seed=SEED):
    import numpy as np
    key=json.dumps([seed,'MECHANISM',int(year),pair,0,'primary'],separators=(',',':'))
    digest=hashlib.sha256(key.encode()).digest()
    return np.random.default_rng(np.random.SeedSequence(np.frombuffer(digest,dtype='<u4').tolist())),digest.hex()
