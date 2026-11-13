"""Frozen numerical bodies routed to task-local IO; no learner entry point."""
from common import *
from collections import OrderedDict
MEM=OrderedDict()
def frozen(stem):return source_pickle(stem)
def store_cached(stem):
 if stem.startswith('runtime/contexts/'):
  if stem in MEM:MEM.move_to_end(stem);return MEM[stem]
  return None
 p=SOURCE/(stem+'.pkl.gz');return source_pickle(stem) if p.exists() else None
def store_commit(stem,value,**kw):
 assert stem.startswith('runtime/contexts/')
 MEM[stem]=value;MEM.move_to_end(stem)
 while len(MEM)>12:MEM.popitem(last=False)
 return value
class Baseline:
 def __getitem__(self,k):return fingerprint(note(ROOT/k))
@lru_cache(None)
def setup():
 a=load(SOURCE/'scripts/aio.py','_residual_original_aio');a.HERE=HERE;a.TABLES=TABLES;a.MODELS=MODELS;a.PLOTS=PLOTS;a.configure()
 a.note=note;a.baseline=lambda:Baseline();a.count=count;a.record=record;a.write=a.write_json=a.json_write=write;a.receipt=lambda *a,**kw:None
 a.load=load;a.sha=sha;a.fp=fingerprint;a.inside=inside
 sys.modules['aio']=a
 st=types.ModuleType('store');st.cached=store_cached;st.commit=store_commit;st.pickle_read=lambda p:pickle.load(gzip.open(note(p),'rb'));sys.modules['store']=st
 o=load(SOURCE/'scripts/originals.py','originals');en=load(SOURCE/'scripts/engine.py','engine')
 warm=load(SOURCE/'audit/repairs/anchor_warmup_v3/warmup.py','warmup')
 m=load(SOURCE/'audit/repairs/anchor_warmup_v3/market_corrected.py','market')
 src=load(SOURCE/'scripts/sources.py','_residual_original_sources')
 inf=load(SOURCE/'scripts/inference.py','_residual_original_inference')
 m.prices=bounded_prices(m.prices)
 # Numpy source methods for statistics and a single rule fixture only.
 configure();return types.SimpleNamespace(io=a,market=m,anchor=en,originals=o,sources=src,inference=inf)
def bounded_prices(function):
 def checked(ticker,year,include_test):
  assert not include_test or os.environ.get('RESIDUAL_READ_PURPOSE') in ['hindsight','reproduction_fixture'],'Test prices prohibited for screen computations'
  return function(ticker,year,include_test)
 checked.cache_clear=function.cache_clear
 return checked
@lru_cache(maxsize=1)
def members(y):return frozen(f'runtime/pools/{y}')
def context(row,y,world='real',rep=0,hindsight=False):
 os.environ['RESIDUAL_READ_PURPOSE']='hindsight' if hindsight else 'screen'
 return setup().market.context(row,y,world,rep,'primary',hindsight)
def legs(row,y):
 import numpy as np
 c=context(row,y);cal=c['cal'];m=setup().market
 return [np.log(m.prices(row[k],y,False).reindex(cal).to_numpy(float)) for k in ['leg_A','leg_B']]
def synthetic_module():
 """The source path() body and its exact stable_seed, with IO-only facades."""
 import numpy as np
 def source_seed(*parts):
  words=np.frombuffer(hashlib.sha256(json.dumps([20261027,*parts],sort_keys=True).encode()).digest(),dtype='<u4')
  return int(np.random.SeedSequence(words.tolist()).generate_state(1,dtype=np.uint64)[0])
 fac=types.ModuleType('sc_common');fac.__dict__.update(globals());fac.stable_seed=source_seed
 src=types.ModuleType('sc_sources');src.anchor=lambda:setup().anchor;src.market=lambda:setup().market;src.eda=lambda:None
 with aliases(sc_common=fac,sc_sources=src):return load(SYN/'scripts/sc_simulation.py','_residual_synthetic')
