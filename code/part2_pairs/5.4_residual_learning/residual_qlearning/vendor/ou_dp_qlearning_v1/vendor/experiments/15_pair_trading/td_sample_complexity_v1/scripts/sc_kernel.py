"""Tuned update adapter; source protocol remains exactly callable."""
from sc_common import *
from sc_sources import native
import numpy as np,ctypes,subprocess
@lru_cache(None)
def kernel():
    path=HERE/'runtime/tuned_kernel.so'
    if not path.exists():
        temp=HERE/('runtime/tuned_kernel.pending_'+str(time.time_ns())+'.so')
        cmd=['/usr/bin/cc','-O3','-std=c11','-shared','-fPIC',str(HERE/'scripts/tuned_kernel.c'),'-o',str(temp)]
        p=subprocess.run(cmd,capture_output=True,text=True)
        write(HERE/f'audit/build_{time.time_ns()}.json',dict(command=cmd,returncode=p.returncode,stderr=p.stderr))
        assert p.returncode==0,p.stderr
        os.link(temp,path);temp.unlink()
    old=native();k=ctypes.CDLL(str(path))
    arguments=list(old.kernel().ta_train.argtypes);arguments.insert(-5,ctypes.c_int);k.sc_train.argtypes=arguments;k.sc_train.restype=None
    k.sc_batch.argtypes=[ctypes.c_int,ctypes.c_int,old.DP,old.DP,old.DP,ctypes.c_int,old.DP,old.IP,old.DP,old.DP,ctypes.c_int,ctypes.c_double,ctypes.c_int,old.DP,old.LP];k.sc_batch.restype=None
    k.sc_rate.argtypes=[ctypes.c_int64];k.sc_rate.restype=ctypes.c_double
    return k

def fit(es,val,alpha,gamma,seed,risk,tuned=False,maxepochs=300,early=True):
    assert es and all(e['window']=='train' for e in es+val)
    old=native();o,d,g,s,r=old.pack(es);vo,vd,vg,vs,vr=old.pack(val)
    q=np.zeros((8,2));curves=np.full((maxepochs,3),np.nan);snaps=np.zeros((maxepochs//10,8),np.int32);visits=np.zeros((8,2),np.int64);out=np.zeros(6,np.int64)
    kernel().sc_train(len(es),old.ip(o),old.dp(d),old.dp(g),old.ip(s),old.dp(r),len(val),old.ip(vo),old.dp(vd),old.dp(vg),old.ip(vs),old.dp(vr),int(seed),alpha,gamma,10.,maxepochs,int(early),0,1,risk['H'],risk['stop'],int(risk['lock']),int(tuned),old.dp(q),old.dp(curves),old.ip(snaps),old.lp(visits),old.lp(out))
    epochs,conv,when,passes,steps,evals=map(int,out)
    count('TD_fits');count('learning_steps',steps)
    return dict(q=q,curves=curves[:epochs],snapshots=snaps[:epochs//10],visits=visits,meta=dict(epochs=epochs,converged=bool(conv),convergence_epoch=when,learning_steps=steps,alpha='1/n^.7' if tuned else alpha,gamma=gamma,seed=int(seed)))

def descriptor(q=None,entry=None,exit=0.):
    return dict(q=np.zeros((8,2)) if q is None else np.asarray(q,float),kind=int(entry is not None),entry=float(entry or 0),exit=float(exit))
def policy_q(bits):
    q=np.zeros((8,2));q[np.arange(8),[(bits>>s)&1 for s in range(8)]]=1.;return q

def batch(es,policies,risk):
    old=native();assert es and policies
    n=len(es[0]['z']);assert all(len(e['z'])==n for e in es)
    z=np.ascontiguousarray([e['z'] for e in es],float);g=np.ascontiguousarray([e['drift'] for e in es],float);r=np.ascontiguousarray([e['forward_bps'] for e in es],float)
    qs=np.ascontiguousarray([p['q'] for p in policies],float);kinds=np.ascontiguousarray([p['kind'] for p in policies],np.int32)
    entries=np.ascontiguousarray([p['entry'] for p in policies],float);exits=np.ascontiguousarray([p['exit'] for p in policies],float)
    out=np.zeros((len(policies),len(es),3));vis=np.zeros((len(policies),8),np.int64)
    kernel().sc_batch(len(es),n,old.dp(z),old.dp(g),old.dp(r),len(policies),old.dp(qs),old.ip(kinds),old.dp(entries),old.dp(exits),risk['H'],risk['stop'],int(risk['lock']),old.dp(out),old.lp(vis))
    count('policy_evaluations',len(es)*len(policies));return out,vis
