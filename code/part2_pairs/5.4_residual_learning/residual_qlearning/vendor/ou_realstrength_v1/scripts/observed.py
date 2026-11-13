"""Shared causal observations, unchanged8 states, explicit per-side accounting."""
from common import *
import ctypes as C
from native import DP,IP,dp,ip
LIB=None

def build():
 src=RUN/'scripts/observed.c';dest=RUN/'runtime/observed.so'
 if not dest.exists():
  tmp=dest.with_name(dest.name+'.pending_'+uuid.uuid4().hex);cmd=['/usr/bin/cc','-O3','-std=c11','-shared','-fPIC',str(src),'-o',str(tmp)];p=subprocess.run(cmd,capture_output=True,text=True);write(event('build_observed'),dict(command=cmd,returncode=p.returncode,stderr=p.stderr));assert p.returncode==0,p.stderr;tmp.rename(dest)
 return dict(source_sha256=sha(src),binary_sha256=sha(dest))

def execute(data,policy,cost,daily=False):
 global LIB
 if LIB is None:
  LIB=C.CDLL(str(RUN/'runtime/observed.so'));LIB.observed.argtypes=[C.c_int,C.c_int,DP,DP,C.POINTER(C.c_ubyte),C.c_int,C.c_int,C.c_int,C.c_double,C.c_double,C.c_double,DP,DP,C.c_int];LIB.observed.restype=None
 s=np.ascontiguousarray(data['s'],float);ret=np.ascontiguousarray(data['returns'],float);valid=np.ascontiguousarray(data['valid'],np.uint8);n,T=ret.shape;assert s.shape==valid.shape==(n,T+1)
 out=np.empty((n,8));days=np.empty((n,T)) if daily else np.empty(1)
 LIB.observed(n,T,dp(s),dp(ret),valid.ctypes.data_as(C.POINTER(C.c_ubyte)),policy['kind'],int(policy.get('bits',0)),int(policy.get('mask',0)),float(policy.get('enter',1.25)),float(policy.get('exit',.5)),float(cost),dp(out),dp(days),int(daily));count('synthetic_policy_series_year_evaluations',n)
 return (out,days) if daily else out

def policy(bits):return dict(kind=0,bits=int(bits))
def al():return dict(kind=1)
def band(enter,exit):return dict(kind=2,enter=float(enter),exit=float(exit))
def spi(bits,mask):return dict(kind=3,bits=int(bits),mask=int(mask))
def summary(out):
 n=len(out);return dict(net=float(out[:,0].mean()),net_SE=float(out[:,0].std(ddof=1)/np.sqrt(n)),gross=float(out[:,1].mean()),turnover=float(out[:,2].mean()),entries=float(out[:,3].mean()),holding_period=float(out[:,5].sum()/out[:,6].sum()) if out[:,6].sum() else np.nan,closed_trades=int(out[:,6].sum()),gross_exposure=float(out[:,7].mean()),breakeven=float(out[:,1].mean()/out[:,2].mean()) if out[:,2].mean()>0 else np.nan)
