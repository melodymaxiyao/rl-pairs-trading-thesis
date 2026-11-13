"""Re-evaluate all frozen baseline worlds before any new production learner."""
from common import *
from simulation import simulate
from pipeline import features
from benchmarks import model,observations,statistics
from observed import execute,policy,al,summary
from qkernel import nominal_se,policy_bits
import gc
LEARNERS=['Oracle-8','AL rule','Band-train','Q_POW']
def point():
 p=read(synthetic().verify('calibration.json'))['chosen'];assert p['h0']==20 and p['w']==.6;return p
def source_world(world,c):return synthetic().cache(f'runtime/worlds/best_{world:02d}_c{c}')
def source_q(world,c):
 src=synthetic();stem=f'runtime/models/best_world{world:02d}_c{c}';m=src.cache(stem);fits=[src.cache(stem+f'_seed{j:02d}') for j in range(10)];q=np.mean([f['q'] for f in fits],axis=0)
 assert np.array_equal(q,m['q']) and policy_bits(q)==m['bits'];return dict(q=q,bits=m['bits'],se=nominal_se(fits),fits=fits)
def train_data(world):
 p=simulate(model(point()),1500,['ou_realstrength_v1','train',world],sessions=312)
 h=hashlib.sha256(np.ascontiguousarray(p).tobytes()).hexdigest();data=features(p);del p;return data,h
def equal(a,b,label):
 if isinstance(a,dict):
  assert set(a)==set(b),(label,set(a)^set(b))
  for k,v in a.items():equal(v,b[k],label+'.'+str(k))
 elif isinstance(a,(float,int,np.number)):
  assert a==b or np.isnan(a) and np.isnan(b),('Not bit-exact',label,a,b)
 else:assert a==b,(label,a,b)
def truth_reproduction():
 stem='runtime/checkpoints/baseline_truth';prior=cached(stem)
 if prior is not None:return prior
 src=synthetic();reference=src.cache('runtime/point_results/best');selection=src.cache('runtime/benchmarks/selection_best');data=observations(model(point()),50000,['truth']);rows={};wanted={}
 for w in range(20):
  for c in [5,10]:
   for name,pol in source_world(w,c)['policies'].items():
    if name in LEARNERS:wanted[(c,json.dumps(pol,sort_keys=True))]=pol
 refs={c:execute(data,policy(selection['bits'][str(c)]),c)[:,0] for c in [5,10]};bo=execute(data,al(),0);base={c:bo[:,1]-c*bo[:,2] for c in [5,10]}
 for (c,key),pol in wanted.items():
  stat=statistics(execute(data,pol,0),pol,c,refs[c],base[c]);rows[str(c)+'|'+key]=stat
 for row in reference['rows']:
  if row['cost'] not in [5,10] or row['learner'] not in LEARNERS:continue
  c=row['cost'];pol=source_world(row['world'],c)['policies'][row['learner']];actual=rows[str(c)+'|'+json.dumps(pol,sort_keys=True)]
  for field,sourcefield in [('net','true_value'),('net_SE','true_value_MC_SE'),('turnover','turnover'),('holding_period','holding_period'),('breakeven','breakeven_cost'),('gross','gross_value'),('regret_oracle','regret')]:equal(actual[field],row[sourcefield],f"truth.w{row['world']}.c{c}.{row['learner']}.{field}")
 del data;gc.collect();return save(stem,dict(status='PASS_BIT_EXACT',cells=160,statistics=rows))
def reproduce_world(world):
 stem=f'runtime/checkpoints/baseline_world{world:02d}';old=cached(stem)
 if old is not None:return old
 train,h=train_data(world);test=observations(model(point()),5000,['test',world]);testhash=hashlib.sha256(test['s'].tobytes()+test['returns'].tobytes()).hexdigest();checks=[]
 for c in [5,10]:
  old=source_world(world,c);assert h==old['training_hash'] and testhash==old['test_hash']
  q=source_q(world,c)
  for row in old['rows']:
   if row['learner'] not in LEARNERS:continue
   actual=summary(execute(test,row['policy'],c));equal(actual,row['test'],f"test.w{world}.c{c}.{row['learner']}");checks.append(dict(cost=c,learner=row['learner'],test=actual))
 del train,test;gc.collect();result=dict(status='PASS_BIT_EXACT',world=world,training_hash=h,test_hash=testhash,checks=checks);save(stem,result);print('Sealed baseline world reproduced',world+1,'/20',flush=True);checkpoint();return result
def reproduce_all(workers=1):
 path=RUN/'audit/baseline_reproduction.json'
 if path.exists():
  receipt=read(path);assert receipt['status']=='PASS_BIT_EXACT' and receipt['worlds']==20;return receipt
 truth=truth_reproduction()
 if workers>1:
  from concurrent.futures import ProcessPoolExecutor
  import multiprocessing as mp
  with ProcessPoolExecutor(workers,mp_context=mp.get_context('spawn')) as pool:worlds=list(pool.map(reproduce_world,range(20),chunksize=1))
 else:worlds=[reproduce_world(w) for w in range(20)]
 receipt=dict(status='PASS_BIT_EXACT',worlds=20,costs=[5,10],baseline_cells=160,learners=LEARNERS,training_and_test_stream_hashes_exact=True,truth_sample_exact=True,new_learners_started=False,source_seal=read(RUN/'audit/source_seal.json'),world_receipts=[dict(world=w['world'],training_hash=w['training_hash'],test_hash=w['test_hash']) for w in worlds],truth_checkpoint_sha256=sha(RUN/'runtime/checkpoints/baseline_truth.pkl.gz'))
 write(path,receipt);return receipt
