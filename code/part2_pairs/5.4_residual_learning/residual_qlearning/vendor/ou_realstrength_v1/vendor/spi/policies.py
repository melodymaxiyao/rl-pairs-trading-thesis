from common import *
from sources import setup
from kernel import lib,policy_id
from inference import nw_mean,holm
import itertools,numpy as np,ctypes
SWITCHES=['E0','E1','E2','E3','K2','K3','X0','X1','X2','X3']

def enumerate_space():
 out=[]
 for a,b,c,d in itertools.product(range(2),range(2),range(3),range(3)):
  for x in range(16):
   bits=a|(b<<1)|((c==1)<<2)|((d==1)<<3)|((c==2)<<4)|((d==2)<<5)|(x<<6);out.append(int(bits))
 assert len(out)==576 and len(set(out))==576 and out[0]==0;return out
SPACE=enumerate_space();INDEX={b:j for j,b in enumerate(SPACE)}
def propose(q):
 bits=0
 for b in range(4):
  if q[2*b,1]>q[2*b,0]:bits|=1<<b
  elif b>=2:bits|=1<<(4+b-2)
  if q[2*b+1,1]>q[2*b+1,0]:bits|=1<<(6+b)
 return bits

def descriptor(q=None,bits=None,entry=None,exit=0.,flat=False):
 return dict(q=np.zeros((8,2)) if q is None else np.asarray(q,float),kind=3 if flat else 2 if bits is not None else 1 if entry is not None else 0,bits=int(bits or 0),entry=float(entry or 0),exit=float(exit))

def batch(es,policies,risk):
 assert es and policies;n=len(es[0]['z']);assert all(len(e['z'])==n for e in es)
 z=np.ascontiguousarray([e['z'] for e in es],float);g=np.ascontiguousarray([e['drift'] for e in es],float);r=np.ascontiguousarray([e['forward_bps'] for e in es],float)
 qs=np.ascontiguousarray([p['q'] for p in policies],float);kind=np.array([p['kind'] for p in policies],np.int32);entry=np.array([p['entry'] for p in policies],float);exit=np.array([p['exit'] for p in policies],float);bits=np.array([p['bits'] for p in policies],np.int32)
 native=setup().native;k=lib();k.ms_batch.argtypes=[ctypes.c_int,ctypes.c_int,native.DP,native.DP,native.DP,ctypes.c_int,native.DP,native.IP,native.DP,native.DP,native.IP,ctypes.c_int,ctypes.c_double,ctypes.c_int,native.DP,native.DP];k.ms_batch.restype=None
 out=np.zeros((len(policies),len(es),4));daily=np.zeros((len(policies),n));k.ms_batch(len(es),n,native.dp(z),native.dp(g),native.dp(r),len(policies),native.dp(qs),native.ip(kind),native.dp(entry),native.dp(exit),native.ip(bits),risk['H'],risk['stop'],int(risk['lock']),native.dp(out),native.dp(daily));count('policy_evaluations',len(es)*len(policies));return out,daily

def train_daily(es,policies,risk):
 assert es and all(e['window']=='train' for e in es);years=sorted({e['year'] for e in es});out=[]
 for year in years:
  subset=[e for e in es if e['year']==year];tot=np.zeros((len(policies),len(subset[0]['z'])))
  for start in range(0,len(subset),100):
   _,daily=batch(subset[start:start+100],policies,risk);tot+=daily
  out.append(tot/len(subset))
 return np.concatenate(out,axis=1)

def gate(es,risk,proposal,lag=63):
 switches=[j for j in range(10) if proposal&(1<<j)];series=train_daily(es,[descriptor(bits=0)]+[descriptor(bits=1<<j) for j in switches],risk);rows=[]
 for i,j in enumerate(switches):rows.append(dict(switch=SWITCHES[j],bit=j,proposed=True,**nw_mean(series[i+1]-series[0],lag)))
 reject,padj=holm([r['p'] for r in rows]);bits=0
 for r,yes,p in zip(rows,reject,padj):r.update(accepted=bool(yes),p_Holm=p);bits|=(1<<r['bit']) if yes else 0
 for b in [2,3]:
  e=b;k=4+b-2
  if bits&(1<<e) and bits&(1<<k):
   a=next(r for r in rows if r['bit']==e);v=next(r for r in rows if r['bit']==k);drop=k if a['z']>=v['z'] else e;bits&=~(1<<drop)
 assert bits in INDEX;return bits,rows
