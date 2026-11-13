"""Unchanged native switches, with the preregistered IPW/cost-aware gate."""
from common import *
from real_sources import setup
import numpy as np,ctypes
SWITCHES=['E0','E1','E2','E3','K2','K3','X0','X1','X2','X3']
def execute(e,bits,risk,world='real'):
 s=setup();n=s.native;k=s.kernel.lib();fn=k.ms_policy
 fn.argtypes=[ctypes.c_int,n.DP,n.DP,n.IP,ctypes.c_int,ctypes.c_int,ctypes.c_double,ctypes.c_int,n.IP,n.IP];fn.restype=None
 z=np.ascontiguousarray(e['z'],float);g=np.ascontiguousarray(e['drift'],float);strong=np.ascontiguousarray(e['strong'],np.int32)
 p=np.zeros(len(z),np.int32);details=np.zeros((len(z),6),np.int32)
 fn(len(z),n.dp(z),n.dp(g),n.ip(strong),int(bits),risk['H'],risk['stop'],int(risk['lock']),n.ip(p),n.ip(details))
 count('executor_calls')
 if world!='real':count('null_executions')
 return p,details
def net(e,p,cost):return np.asarray(p)*e['forward_bps']-cost*np.abs(np.diff(p,prepend=0))/2
def daily_differences(es,weights,risk,proposal,costs=COSTS,world='real'):
 assert es and all(e['window']=='train' for e in es)
 bits=[j for j in range(10) if proposal&(1<<j)];out={c:[] for c in costs}
 for year in sorted({e['year'] for e in es}):
  subset=[e for e in es if e['year']==year];dates=np.unique(np.concatenate([np.asarray(e['dates']) for e in subset]));den=np.zeros(len(dates));tot={c:np.zeros((len(bits),len(dates))) for c in costs}
  for e in subset:
   ix=np.searchsorted(dates,np.asarray(e['dates']));assert np.array_equal(dates[ix],np.asarray(e['dates']))
   weight=float(weights[e['pair']]);assert weight>0
   baseline=execute(e,0,risk,world)[0];alternatives=[execute(e,1<<b,risk,world)[0] for b in bits]
   for cost in costs:
    a=np.array([net(e,p,cost)-net(e,baseline,cost) for p in alternatives])
    if not bits:a=np.empty((0,len(ix)))
    tot[cost][:,ix]+=weight*a
   den[ix]+=weight
  assert (den>0).all()
  for cost in costs:out[cost].append(tot[cost]/den[None,:])
  checkpoint_guard()
 return {c:np.concatenate(v,axis=1) for c,v in out.items()},bits
def decisions(series,bits,lag):
 s=setup();rows=[]
 for i,b in enumerate(bits):
  rows.append(dict(switch=SWITCHES[b],bit=b,proposed=True,**s.stats.nw_mean(series[i],lag)));count('switch_tests')
 reject,padj=s.stats.holm([r['p'] for r in rows]);deployed=0
 for row,yes,p in zip(rows,reject,padj):row.update(accepted=bool(yes),p_Holm=float(p));deployed|=(1<<row['bit']) if yes else 0
 for b in [2,3]:
  k=b+2
  if deployed&(1<<b) and deployed&(1<<k):
   e=next(r for r in rows if r['bit']==b);v=next(r for r in rows if r['bit']==k)
   drop=k if e['z']>=v['z'] else b;deployed&=~(1<<drop)
 assert deployed in s.policies.INDEX
 return deployed,rows
def gate(es,weights,risk,proposal,world='real',costs=COSTS):
 values,bits=daily_differences(es,weights,risk,proposal,costs,world);tests=[];deployed=[]
 for cost in costs:
  for lag in [63,21,126]:
   accepted,rows=decisions(values[cost],bits,lag)
   tests.extend(dict(r,cost=cost,primary_lag=lag==63) for r in rows)
   deployed.append(dict(cost=cost,lag=lag,primary_lag=lag==63,proposed_bits=proposal,deployed_bits=accepted,n_switches=accepted.bit_count(),training_days=values[cost].shape[1],test_rows_used=0))
 return tests,deployed
