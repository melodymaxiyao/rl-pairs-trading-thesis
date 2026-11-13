"""Load unchanged numerical bodies; regenerate contexts, never source runtime caches."""
from common import *
import numpy as np,pandas as pd,ctypes
MEM=OrderedDict()
def reset_contexts():
 MEM.clear()
def context_cache(stem):return MEM.get(stem)
def context_commit(stem,value,**kw):
 assert stem.startswith('runtime/contexts/')
 MEM[stem]=value
 while len(MEM)>8:MEM.popitem(last=False)
 return value
@lru_cache(None)
def setup():
 a=load(SOURCE/'scripts/aio.py','_rms_original_aio');a.HERE=HERE;a.TABLES=TABLES;a.MODELS=MODELS;a.PLOTS=PLOTS
 a.configure=configure;a.note=note;a.load=load;a.count=count;a.record=record;a.sha=sha;a.fp=fingerprint
 a.write=a.write_json=a.json_write=write;a.inside=inside;a.baseline=lambda:registration();a.receipt=lambda *args,**kw:None
 sys.modules['aio']=a
 st=types.ModuleType('store');st.cached=context_cache;st.commit=context_commit;st.pickle_read=source_pickle_at
 sys.modules['store']=st
 original=load(SOURCE/'scripts/originals.py','originals');engine=load(SOURCE/'scripts/engine.py','engine')
 warm=load(SOURCE/'audit/repairs/anchor_warmup_v3/warmup.py','warmup')
 market=load(SOURCE/'audit/repairs/anchor_warmup_v3/market_corrected.py','market')
 market.cached=context_cache;market.commit=context_commit
 native=load(SOURCE/'scripts/native.py','native');native.build=lambda:note(SOURCE/'runtime/adapter_kernel.so')
 src=load(SOURCE/'scripts/sources.py','_rms_sealed_sources')
 plans=types.ModuleType('plans');plans.plan=lambda pool,y,world='real',rep=0,variant='primary':source_pickle(f'runtime/plans/{pool}_{y}_{world}_{rep}_{variant}')
 plans.b3=plans.td=lambda *args,**kw:(_ for _ in ()).throw(RuntimeError('Source fitting/selection prohibited'))
 plans.OFF=dict(H=0,stop=0.,lock=False,incremental=False,nbins=9)
 pools=types.ModuleType('pools');pools.prepare=lambda y:source_pickle(f'runtime/pools/{y}')
 with aliases(sources=src,plans=plans,pools=pools):execution=load(SOURCE/'scripts/execution.py','_rms_sealed_execution')
 inference=load(SOURCE/'scripts/inference.py','_rms_year_inference')
 ms_sources=load(MECH/'scripts/sources.py','_rms_ms_sources')
 with aliases(sources=ms_sources):ms_kernel=load(MECH/'scripts/kernel.py','_rms_ms_kernel')
 with aliases(sources=ms_sources,kernel=ms_kernel):
  ms_inf=load(MECH/'scripts/inference.py','_rms_ms_inference')
  with aliases(inference=ms_inf):ms_policies=load(MECH/'scripts/policies.py','_rms_ms_policies')
 # This interface constructs no synthetic worlds until an explicit fixture asks.
 ms_state=ms_sources.setup()
 with aliases(sc_common=sys.modules['common'],sc_sources=types.SimpleNamespace(native=lambda:native)):
  tuned_kernel=load(SYN/'scripts/sc_kernel.py','_rms_tuned_kernel')
 tuned_kernel.kernel=lambda:ms_state.old_kernel.kernel()
 args=list(native.kernel().ta_train.argtypes);args.insert(-5,ctypes.c_int)
 ms_state.old_kernel.kernel().sc_train.argtypes=args;ms_state.old_kernel.kernel().sc_train.restype=None
 with aliases(common=types.SimpleNamespace(COUNTS=COUNTS)):
  methods=load(THESIS/'scripts/methods.py','_rms_thesis_methods')
 configure()
 return types.SimpleNamespace(io=a,market=market,native=native,sources=src,execution=execution,inference=inference,
                              ms_sources=ms_sources,ms_state=ms_state,kernel=ms_kernel,policies=ms_policies,stats=ms_inf,tuned=tuned_kernel,methods=methods)
def members(pool,y):return source_pickle(f'runtime/pools/{y}')[pool]
def plan(pool,y,world='real',rep=0):return source_pickle(f'runtime/plans/{pool}_{y}_{world}_{rep}_primary')
def risk(pool,y,world='real',rep=0):return plan(pool,y,world,rep)['risks']['A_TR']['final']
def tuned_model(pool,y):return source_pickle_at(TUNED/f'outputs/models/{pool}_{y}_TD8_tuned.pkl.gz')
def training(pool,y,world='real',rep=0):
 s=setup();saved=source_pickle(f'runtime/pools/{y}');wanted=set(saved['training_subset']) if pool=='P_RAND' else {r['pair'] for r in saved[pool]}
 model=source_pickle(f'outputs/models/{pool}_{y}_TD8');assert sorted(wanted)==model['training_pairs']
 es=[];weights={};missing=[]
 for row in saved[pool]:
  if row['pair'] not in wanted:continue
  c=s.market.context(row,y,world,rep,'primary',False)
  if c.get('unavailable'):
   assert world=='N1';missing.append(dict(pair=row['pair'],reason=c['reason']));reset_contexts();continue
  weights[row['pair']]=float(row['weight']);es.extend(c['episodes']['A_TR']);reset_contexts()
  checkpoint_guard()
 assert es and all(e['window']=='train' and y-3<=e['year']<y and max(e['dates'].year)<y for e in es)
 return es,weights,missing
def test_episode(row,y,world,rep):
 c=setup().market.context(row,y,world,rep,'primary',True)
 if c.get('unavailable'):return None,c['reason']
 es=[e for e in c['episodes']['A_TR'] if e['year']==y];assert len(es)==1
 return es[0],None
