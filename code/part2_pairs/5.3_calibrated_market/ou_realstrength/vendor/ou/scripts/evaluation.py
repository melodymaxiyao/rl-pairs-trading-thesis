"""Continuous-path common-random-number evaluation; no model fitting."""
from common import *
from native import C,DP,IP,dp,ip
import subprocess
LIB=None
def build_rollout():
 source=RUN/'scripts/rollout.c';target=RUN/'runtime/rollout.so'
 if not target.exists():
  cmd=['/usr/bin/cc','-O3','-std=c11','-include','stdlib.h','-shared','-fPIC',str(source),'-o',str(target)+'.pending_build']
  p=subprocess.run(cmd,capture_output=True,text=True);assert p.returncode==0,p.stderr;Path(str(target)+'.pending_build').rename(target)
 return dict(source_sha256=sha(source),binary_sha256=sha(target))
def execute(paths,policy,cost,stop_sd):
 global LIB
 if LIB is None:
  LIB=C.CDLL(str(RUN/'runtime/rollout.so'));LIB.rollout.argtypes=[C.c_int,C.c_int,DP,C.c_int,C.c_int,C.c_double,C.c_double,C.c_double,C.c_int,C.c_double,C.c_double,IP,DP];LIB.rollout.restype=None
 paths=np.ascontiguousarray(paths,float);n,T=paths.shape[0],paths.shape[1]-1;out=np.empty((n,5));kind=policy['kind'];x=policy.get('x',np.array([0.,1.]));grid=np.ascontiguousarray(policy.get('actions',np.zeros(1)),np.int32)
 LIB.rollout(n,T,dp(paths),kind,int(policy.get('bits',0)),float(policy['scale']),4*stop_sd,cost,len(x),float(x[0]),float(x[1]-x[0]),ip(grid),dp(out));count('synthetic_policy_pair_year_evaluations',n)
 return out
def summary(values):return dict(net=float(values[:,0].mean()),net_SE=float(values[:,0].std(ddof=1)/np.sqrt(len(values))),gross=float(values[:,1].mean()),turnover=float(values[:,2].mean()),entries=float(values[:,3].mean()),abstention=float(values[:,4].mean()),never_enter_fraction=float(np.mean(values[:,3]==0)))
def policy(bits,scale):return dict(kind=0,bits=int(bits),scale=float(scale))
def grid_policy(result,x,scale):return dict(kind=3,actions=result['actions'],x=x,scale=scale)
def mc_oracle(paths,scale,stop_sd,cost):
 values=np.array([execute(paths,policy(b,scale),cost,stop_sd)[:,0].mean() for b in range(256)]);bits=int(np.argmax(values));count('W_oracle_selection_policy_values',256);return dict(bits=bits,value=float(values[bits]),values=values)
