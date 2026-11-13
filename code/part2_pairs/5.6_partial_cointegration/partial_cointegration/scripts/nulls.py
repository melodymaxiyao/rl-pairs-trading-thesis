"""Exact-mask20,000 RW null calibration; numeric critical registered before use."""
import hashlib
import numpy as np
from io_utils import *
from groups import done,finish
import pci
def seed(*key):return int.from_bytes(hashlib.sha256(encode(['partial_coint_kalman_v1',20261103,*key])).digest()[:8],'little')&((1<<63)-1)
def chunk(mask,offset):
 mask=np.asarray(mask,bool);f=np.flatnonzero(mask);mask=mask[f[0]:f[-1]+1];key=hashlib.sha256(mask.tobytes()).hexdigest();part=ROOT/f'runtime/null_chunks/{key}/{offset:05d}'
 if done(part):return dict(root=str(part),cached=True)
 rows=[]
 for i in range(offset,offset+100):
  checkpoint();rng=np.random.default_rng(seed('RW_LR',key,i));x=np.cumsum(rng.normal(size=len(mask)));x[~mask]=np.nan;result=pci.fit(x,pooled=False,checkpoint=checkpoint)
  if not result['converged']:raise Pause(f'RW calibration did not converge:mask{key},rep{i}; no critical substituted')
  rows.append(dict(index=i,LR=result['LR'],boundary=result['boundary'],RW_boundary=result['RW_boundary']))
 return finish(part,{'LR.json':rows},phase='RW_null_calibration',mask=key)
def critical(mask):
 mask=np.asarray(mask,bool);finite=np.flatnonzero(mask);assert len(finite)>=3
 mask=mask[finite[0]:finite[-1]+1];key=hashlib.sha256(mask.tobytes()).hexdigest();root=ROOT/'preregistration/critical_values'/key
 if done(root):return J(root/'critical.json')
 stats=[];attempts=0
 for offset in range(0,20000,100):
  part=ROOT/f'runtime/null_chunks/{key}/{offset:05d}'
  if not done(part):chunk(mask,offset)
  stats += J(part/'LR.json')
  if offset%1000==0:print('PCI null',key[:8],offset+100,'/20000',flush=True)
 lr=np.array([v['LR'] for v in stats]);assert len(lr)==20000 and np.isfinite(lr).all();value=float(np.quantile(lr,.95,method='higher'))
 result=dict(mask_sha256=key,mask=mask.tolist(),formation_calendar_span=len(mask),observed_sessions=int(mask.sum()),replications=20000,critical_5percent=value,seed=20261103,quantile_method='higher',RW_boundary_share=float(np.mean([v['RW_boundary'] for v in stats])),generation='RW unit increments over every calendar session; exact observed formation mask; same secondary optimizer as real; numeric immutable registration before any use',LR_summary=dict(min=float(lr.min()),median=float(np.median(lr)),maximum=float(lr.max())))
 finish(root,{'critical.json':result},phase='preregistered_numeric_PCI_critical');return result
