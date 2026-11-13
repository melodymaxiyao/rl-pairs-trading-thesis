"""Unchanged imported bodies, with task-local IO and sealed pool/limit readers."""
from common import *
MEM={}
def frozen(stem):return source_pickle(stem)
def store_cached(stem):
 if stem.startswith('runtime/contexts/'):
  if stem in MEM:return MEM[stem]
  p=SOURCE/(stem+'.pkl.gz')
  if p.exists():MEM[stem]=source_pickle(stem);return MEM[stem]
  return None
 p=SOURCE/(stem+'.pkl.gz')
 return source_pickle(stem) if p.exists() else None
def store_commit(stem,value,**kw):
 assert stem.startswith('runtime/contexts/'),'No source pool, limit or selection may be recomputed'
 MEM[stem]=value;return value
class Baseline:
 def __getitem__(self,k):note(ROOT/k);return PINNED[k]
@lru_cache(None)
def setup():
 import numpy as np,pandas as pd
 # aio's initial import changes environment only; all subsequent bindings point here.
 a=load(SOURCE/'scripts/aio.py','_tuned_original_aio');a.HERE=HERE;a.TABLES=TABLES;a.MODELS=MODELS;a.PLOTS=PLOTS
 a.configure();a.note=note;a.baseline=lambda:Baseline();a.count=count;a.record=record;a.write=a.write_json=a.json_write=write
 a.receipt=lambda *args,**kw:None
 a.load=load;a.sha=sha;a.fp=fingerprint;a.inside=inside
 sys.modules['aio']=a
 st=types.ModuleType('store');st.cached=store_cached;st.commit=store_commit;st.pickle_read=lambda p:pickle.load(gzip.open(note(p),'rb'))
 sys.modules['store']=st
 o=load(SOURCE/'scripts/originals.py','originals')
 en=load(SOURCE/'scripts/engine.py','engine')
 warm=load(SOURCE/'audit/repairs/anchor_warmup_v3/warmup.py','warmup')
 market=load(SOURCE/'audit/repairs/anchor_warmup_v3/market_corrected.py','market')
 native=load(SOURCE/'scripts/native.py','native')
 src=load(SOURCE/'scripts/sources.py','_tuned_original_sources')
 # Numerical core's native loader is read-only and requires this sealed binary.
 qlib=HERE/'runtime/q_kernel.so'
 if not qlib.exists():publish(qlib,note(SOURCE/'runtime/q_kernel.so').read_bytes())
 plans=types.ModuleType('plans');plans.plan=lambda pool,y,world='real',rep=0,variant='primary':frozen(f'runtime/plans/{pool}_{y}_{world}_{rep}_{variant}')
 plans.b3=plans.td=lambda *args,**kw:(_ for _ in ()).throw(RuntimeError('Unregistered training or selection'))
 plans.OFF=dict(H=0,stop=0.,lock=False,incremental=False,nbins=9)
 pools=types.ModuleType('pools');pools.prepare=lambda y:frozen(f'runtime/pools/{y}')
 with aliases(sources=src,plans=plans,pools=pools):exe=load(SOURCE/'scripts/execution.py','_tuned_original_execution')
 inf=load(SOURCE/'scripts/inference.py','_tuned_original_inference')
 facade=types.ModuleType('sc_common');facade.__dict__.update(a.__dict__);facade.__dict__.update(HERE=HERE,ROOT=ROOT,PT=PT,write=write,count=count,record=record)
 synsrc=types.ModuleType('sc_sources');synsrc.native=lambda:native
 with aliases(sc_common=facade,sc_sources=synsrc):tuned=load(SYN/'scripts/sc_kernel.py','_tuned_unchanged_kernel')
 configure()
 return types.SimpleNamespace(io=a,market=market,native=native,tuned=tuned,sources=src,execution=exe,inference=inf)
def members(pool,y):return frozen(f'runtime/pools/{y}')[pool]
def risk(pool,y,world='real',rep=0):return frozen(f'runtime/plans/{pool}_{y}_{world}_{rep}_primary')['risks']['A_TR']['final']
def training(pool,y):
 state=setup();saved=frozen(f'runtime/pools/{y}');wanted=set(saved['training_subset']) if pool=='P_RAND' else {r['pair'] for r in saved[pool]};es=[]
 for row in saved[pool]:
  if row['pair'] not in wanted:continue
  c=state.market.context(row,y,'real',0,'primary',False)
  assert not c.get('unavailable'),('Sealed real training member unavailable',pool,y,row['pair'])
  es.extend(c['episodes']['A_TR']);MEM.clear()
 assert es and all(e['window']=='train' and y-3<=e['year']<y for e in es)
 original=frozen(f'outputs/models/{pool}_{y}_TD8')
 assert sorted(wanted)==original['training_pairs'],'Sealed training subset changed'
 return es,sorted(wanted)
def test_episode(row,y,world,rep):
 c=setup().market.context(row,y,world,rep,'primary',True)
 if c.get('unavailable'):return None,c['reason']
 es=[e for e in c['episodes']['A_TR'] if e['year']==y];assert len(es)==1
 return es[0],None
