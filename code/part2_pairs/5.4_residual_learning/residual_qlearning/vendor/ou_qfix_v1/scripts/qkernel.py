"""Double learner and MC diagnostic ABI; source evaluation stays unchanged."""
from common import *
from native import DP,IP,LP,dp,ip,lp
import ctypes as C
LIB=None
def build():
 target=RUN/'runtime/qfix.so'
 if not target.exists():
  pending=target.with_name(target.name+'.pending_build');command=['/usr/bin/cc','-O3','-std=c11','-shared','-fPIC',str(RUN/'scripts/qfix.c'),'-o',str(pending)]
  p=subprocess.run(command,capture_output=True,text=True);write(event('build_qfix'),dict(command=command,returncode=p.returncode,stderr=p.stderr));assert p.returncode==0,p.stderr;pending.rename(target)
 return dict(source_sha256=sha(RUN/'scripts/qfix.c'),binary_sha256=sha(target))
def library():
 global LIB
 if LIB is None:
  LIB=C.CDLL(str(RUN/'runtime/qfix.so'))
  LIB.double_train.argtypes=[C.c_int,C.c_int,DP,DP,C.c_uint64,C.c_uint64,C.c_double,C.c_int,C.c_int,DP,LP,DP,DP,LP];LIB.double_train.restype=None
  LIB.double_target.argtypes=[DP,DP,C.c_int];LIB.double_target.restype=C.c_double
  LIB.truth_q.argtypes=[C.c_int,C.c_int,DP,DP,C.POINTER(C.c_ubyte),C.c_int,C.c_double,DP,LP];LIB.truth_q.restype=None
  LIB.entry_diagnostics.argtypes=[C.c_int,C.c_int,DP,C.POINTER(C.c_ubyte),C.c_int,C.c_int,C.c_double,C.c_double,LP];LIB.entry_diagnostics.restype=None
 return LIB
def double_fit(z,returns,seed,cost,epochs=300,early=True):
 z=np.ascontiguousarray(z,float);ret=np.ascontiguousarray(returns,float);assert z.shape==ret.shape
 q=np.zeros((2,8,2));vis=np.zeros((2,8,2),np.int64);diag=np.zeros((2,8,2,10));curves=np.zeros((epochs,3));result=np.zeros(6,np.int64)
 coin_seed=stable_seed('ou_qfix_v1','double_table_choice',int(seed))
 library().double_train(*z.shape,dp(z),dp(ret),int(seed),coin_seed,float(cost),epochs,int(early),dp(q),lp(vis),dp(diag),dp(curves),lp(result))
 count('new_synthetic_Double_seed_fits');count('new_learning_steps',int(result[4]))
 return dict(q=q,visits=vis,diagnostics=diag,curves=curves[:int(result[0])],epochs=int(result[0]),converged=bool(result[1]),convergence_epoch=int(result[2]),learning_steps=int(result[4]),seed=int(seed),table_choice_seed=coin_seed)
def nominal_se(fits,double=False):
 se=[]
 for f in fits:
  diag=np.asarray(f['diagnostics']);sd=diag[...,7];v=diag[...,1]
  one=sd*np.sqrt(v)
  if double:one=one.mean(axis=0)
  se.append(one)
 return np.mean(se,axis=0)
def policy_bits(q,se=None,z=1.645):
 q=np.asarray(q,float);assert q.shape==(8,2);a=q[:,1]>q[:,0]
 if se is not None and z!=0:
  se=np.asarray(se,float);margin=np.sqrt(se[:4,0]**2+se[:4,1]**2)
  a[:4]=np.isfinite(margin)&(q[:4,1]-q[:4,0]>z*margin)
 return sum(int(x)<<s for s,x in enumerate(a))
def mc_q(data,bits,cost):
 s=np.ascontiguousarray(data['s'],float);r=np.ascontiguousarray(data['returns'],float);v=np.ascontiguousarray(data['valid'],np.uint8);n,T=r.shape;out=np.empty((2,8,2));counts=np.empty((8,2),np.int64)
 library().truth_q(n,T,dp(s),dp(r),v.ctypes.data_as(C.POINTER(C.c_ubyte)),int(bits),float(cost),dp(out),lp(counts))
 return dict(q=out[0],MC_SE=out[1],observations=counts,series_years=n,reference='same full truth histories; all valid decision times; inventories0,+1,-1; two counterfactual actions; finite-year liquidation; cluster-ratio MC SE by independent year; not a Markov-state Bellman quantity')
def entries(data,pol):
 s=np.ascontiguousarray(data['s'],float);v=np.ascontiguousarray(data['valid'],np.uint8);n,T=data['returns'].shape;out=np.zeros(16,np.int64)
 library().entry_diagnostics(n,T,dp(s),v.ctypes.data_as(C.POINTER(C.c_ubyte)),pol['kind'],int(pol.get('bits',0)),float(pol.get('enter',1.25)),float(pol.get('exit',.5)),lp(out))
 return dict(opportunities=out[:4],entries=out[4:8],all_flat_days=int(out[8]),all_entries=int(out[9]),eligible_flat_days=int(out[10]),holding_days=int(out[11]),forced_closures=int(out[12]),all_days=int(out[13]))
