"""Local binary adapter; original signed-action mode is unchanged C source."""
from aio import *
import ctypes,subprocess
import numpy as np
IP=ctypes.POINTER(ctypes.c_int);DP=ctypes.POINTER(ctypes.c_double);LP=ctypes.POINTER(ctypes.c_int64)
def ip(a):return a.ctypes.data_as(IP)
def dp(a):return a.ctypes.data_as(DP)
def lp(a):return a.ctypes.data_as(LP)
def build():
    p=HERE/'runtime/adapter_kernel.so'
    if p.exists():return p
    assert sha(HERE/'scripts/original_q_kernel.c')==sha(V2/'scripts/q_kernel.c')
    target=HERE/('runtime/adapter_kernel.pending_'+str(time.time_ns())+'.so')
    cmd=['/usr/bin/cc','-O3','-std=c11','-shared','-fPIC',str(HERE/'scripts/adapter_kernel.c'),'-o',str(target)]
    result=subprocess.run(cmd,capture_output=True,text=True);write(HERE/f'audit/native_build_{time.time_ns()}.json',dict(command=cmd,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
    assert result.returncode==0,result.stderr
    assert not p.exists();os.link(target,p);target.unlink();return p
@lru_cache(None)
def kernel():
    k=ctypes.CDLL(str(build()))
    k.ta_policy.argtypes=[ctypes.c_int,DP,DP,IP,DP,ctypes.c_int,ctypes.c_int,ctypes.c_double,ctypes.c_double,ctypes.c_int,ctypes.c_int,ctypes.c_double,ctypes.c_int,IP,IP]
    k.ta_train.argtypes=[ctypes.c_int,IP,DP,DP,IP,DP,ctypes.c_int,IP,DP,DP,IP,DP,ctypes.c_uint64,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_double,ctypes.c_int,DP,DP,IP,LP,LP]
    k.ledger.argtypes=[ctypes.c_int,DP,IP,ctypes.c_double,DP]
    k.allowed_mask.argtypes=[ctypes.c_int]*4+[ctypes.c_double]+[ctypes.c_int]*2+[ctypes.c_double]+[ctypes.c_int]*2
    return k

def encode(d,p,drift,strong,feature=0):
    b=int(np.searchsorted([.5,1.5,2.5],abs(d),side='right'));s=b*2+int(p!=0)
    al=-np.sign(d)*drift>0
    return s*2+int(not (al if feature==1 else strong)) if feature else s

def policy(e,q=None,feature=0,entry=None,exit=0.,aligned=True,risk=None):
    risk=risk or dict(H=0,stop=0.,lock=False)
    d=np.ascontiguousarray(e['z'],float);drift=np.ascontiguousarray(e['drift'],float);strong=np.ascontiguousarray(e['strong'],np.int32)
    assert np.isfinite(d).all() and len(d)==len(drift)==len(strong)
    ns=16 if feature else 8;q=np.zeros((ns,2)) if q is None else np.ascontiguousarray(q,float)
    assert q.shape==(ns,2)
    p=np.zeros(len(d),np.int32);details=np.zeros((len(d),6),np.int32)
    kernel().ta_policy(len(d),dp(d),dp(drift),ip(strong),dp(q),feature,int(entry is not None),float(entry or 0),float(exit),int(aligned),risk['H'],risk['stop'],int(risk['lock']),ip(p),ip(details))
    count('rule_executions' if entry is not None else 'TD_executor_calls');return p,details

def pack(es):
    offsets=np.r_[0,np.cumsum([len(e['z']) for e in es])].astype(np.int32)
    def cat(k,dtype):return np.ascontiguousarray(np.concatenate([e[k] for e in es]) if es else np.zeros(0),dtype)
    return offsets,cat('z',float),cat('drift',float),cat('strong',np.int32),cat('forward_bps',float)

def fit(es,val,alpha,gamma,seed,risk,feature=0,aligned=True,maxepochs=300,early=True):
    assert es and all(e['window']=='train' for e in es+val)
    o,d,g,s,r=pack(es);vo,vd,vg,vs,vr=pack(val);ns=16 if feature else 8
    q=np.zeros((ns,2));curves=np.full((maxepochs,3),np.nan);snaps=np.zeros((maxepochs//10,ns),np.int32);visits=np.zeros_like(q,np.int64);out=np.zeros(6,np.int64)
    kernel().ta_train(len(es),ip(o),dp(d),dp(g),ip(s),dp(r),len(val),ip(vo),dp(vd),dp(vg),ip(vs),dp(vr),int(seed),alpha,gamma,10.,maxepochs,int(early),feature,int(aligned),risk['H'],risk['stop'],int(risk['lock']),dp(q),dp(curves),ip(snaps),lp(visits),lp(out))
    epochs,conv,when,passes,steps,evals=map(int,out)
    record('TD_fit',TD_fits=1,rl_fits=1,learning_steps=steps,training_episodes=passes,training_validation_evaluations=evals,feature=feature,seed=int(seed),epochs=epochs,test_rows_used=0)
    return dict(q=q,curves=curves[:epochs],snapshots=snaps[:epochs//10],visits=visits,meta=dict(epochs=epochs,converged=bool(conv),convergence_epoch=when,learning_steps=steps,alpha=alpha,gamma=gamma,seed=int(seed)))
