"""Instrumented shared update; original environment/greedy/RNG bodies included unchanged."""
from common import *
from sources import setup
import ctypes,subprocess,numpy as np

@lru_cache(None)
def lib():
 target=HERE/('runtime/mechanism_'+sha(HERE/'scripts/mechanism_kernel.c')[:16]+'.so')
 if not target.exists():
  tmp=HERE/f'runtime/mechanism.pending_{time.time_ns()}.so';cmd=['/usr/bin/cc','-O3','-std=c11','-shared','-fPIC',str(HERE/'scripts/mechanism_kernel.c'),'-o',str(tmp)]
  p=subprocess.run(cmd,capture_output=True,text=True);write(HERE/f'audit/build_{time.time_ns()}.json',dict(command=cmd,returncode=p.returncode,stderr=p.stderr));assert p.returncode==0,p.stderr
  try:os.link(tmp,target)
  except FileExistsError:assert sha(tmp)==sha(target),'Concurrent deterministic build differs'
  tmp.unlink()
 n=setup().native;k=ctypes.CDLL(str(target));a=list(n.kernel().ta_train.argtypes);a=a[:-5]+[ctypes.c_int,ctypes.c_int,n.IP,ctypes.c_int]+a[-5:]+[n.DP,n.DP];k.ms_train.argtypes=a;k.ms_train.restype=None
 return k

def fit(es,val,step,gamma,seed,risk,maxepochs=300,early=True,order='original'):
 assert step in ['C01','C05','CAP','POW'] and es and all(e['window']=='train' for e in es+val)
 native=setup().native;train=list(reversed(es)) if order=='reverse' else list(es);o,d,g,s,r=native.pack(train);vo,vd,vg,vs,vr=native.pack(val)
 names=sorted({e['pair'] for e in train});mapping={p:j for j,p in enumerate(names)};pairs=np.array([mapping[e['pair']] for e in train],np.int32)
 q=np.zeros((8,2));curves=np.full((maxepochs,3),np.nan);snaps=np.zeros((maxepochs//10,8),np.int32);vis=np.zeros((8,2),np.int64);out=np.zeros(7,np.int64);diag=np.zeros((16,10));trajectory=np.zeros((maxepochs,16));alpha=.01 if step=='C01' else .05
 lib().ms_train(len(train),native.ip(o),native.dp(d),native.dp(g),native.ip(s),native.dp(r),len(val),native.ip(vo),native.dp(vd),native.dp(vg),native.ip(vs),native.dp(vr),int(seed),alpha,gamma,10.,maxepochs,int(early),0,1,risk['H'],risk['stop'],int(risk['lock']),0 if step.startswith('C') and step!='CAP' else 2 if step=='CAP' else 1,int(order=='fixed'),native.ip(pairs),len(names),native.dp(q),native.dp(curves),native.ip(snaps),native.lp(vis),native.lp(out),native.dp(diag),native.dp(trajectory))
 epochs,conv,when,passes,steps,evals=map(int,out[:6]);count('TD_fits');count('rl_fits');count('learning_steps',steps);count('training_episodes',passes);count('training_validation_evaluations',evals)
 return dict(q=q,curves=curves[:epochs],snapshots=snaps[:epochs//10],visits=vis,diagnostics=diag.reshape(8,2,10),trajectory=trajectory[:epochs].reshape(epochs,8,2),meta=dict(epochs=epochs,converged=bool(conv),convergence_epoch=when,learning_steps=steps,seed=int(seed),alpha=step,gamma=gamma,order=order,early=early,epsilon_stream_checksum=int(out[6])))

def rate(step,n):
 assert n>=1
 return .01 if step=='C01' else .05 if step=='C05' else min(.05,n**(-.7)) if step=='CAP' else n**(-.7)

def policy_id(q):return sum(int(q[s,1]>q[s,0])<<s for s in range(8))

def fit_cached(es,val,step,gamma,seed,risk,*,maxepochs=300,early=True,order='original',context=None):
 h=hashlib.sha256()
 for e in es+val:
  h.update(json.dumps([e['pair'],e['year'],e['window']]).encode())
  for key in ['z','drift','strong','forward_bps']:h.update(np.ascontiguousarray(e[key]).tobytes())
 sig=dict(arrays=h.hexdigest(),ntrain=len(es),nval=len(val),step=step,gamma=gamma,seed=int(seed),risk=risk,maxepochs=maxepochs,early=early,order=order,context=context);digest=hashlib.sha256(json.dumps(sig,sort_keys=True).encode()).hexdigest();stem='runtime/fits/'+digest;old=cached(stem)
 if old is not None:return old
 started=time.monotonic();result=fit(es,val,step,gamma,seed,risk,maxepochs,early,order);result['meta']['wall_seconds']=time.monotonic()-started;result['signature']=sig;return save(stem,result)
