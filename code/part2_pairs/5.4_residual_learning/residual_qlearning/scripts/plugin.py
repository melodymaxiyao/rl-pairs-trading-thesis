"""Independent fitted-model Monte Carlo enumeration, not exact DP."""
from common import *
from data import M,adf_batch
from kernel import mc_moments

def generate(model,n,rng):
 T=253;burn=60;b=model['b'];sigma=model['sigma_epsilon'];mu=model['mean'];x=np.empty((n,T+burn));x[:,0]=mu+sigma/np.sqrt(1-b*b)*rng.standard_normal(n)
 innovations=rng.standard_normal((n,T+burn-1))
 for t in range(1,T+burn):x[:,t]=model['a']+b*x[:,t-1]+sigma*innovations[:,t-1]
 e=np.diff(x,axis=1);win=np.lib.stride_tricks.sliding_window_view(e,60,axis=1);fit=M.ou_fit(win);z=fit['s'];h=fit['half_life'];valid=np.isfinite(z)&np.isfinite(h)&(h>0)&(h<30);out=np.where(valid,z,np.inf);returns=np.column_stack([e[:,60:],np.zeros(n)])
 assert out.shape==returns.shape==(n,T);adf=adf_batch(win);return out,np.where(adf,out,np.inf),returns

def moments(model,kind,year,stream,total=50000):
 seed=stable_seed('Plugin',kind,year,stream);rng=np.random.default_rng(seed);acc={'G0':np.zeros((5,256)),'G1':np.zeros((5,256))};begin=0
 prefix=f'Plugin_{kind}_{year}_{stream}_done'
 states=sorted((RUN/'runtime/checkpoints').glob(prefix+'*.pkl.gz'),key=lambda p:int(p.name[len(prefix):-7]))
 if states:
  stem=str(states[-1].relative_to(RUN))[:-7];old=cached(stem);assert old['model']==model and old['total']==total and old['seed']==seed;acc=old['moments'];begin=old['done'];rng.bit_generator.state=old['rng_state']
 for start in range(begin,total,128):
  checkpoint();batch=min(128,total-start);z,z1,r=generate(model,batch,rng)
  for gate,zz in [('G0',z),('G1',z1)]:acc[gate]+=mc_moments(zz,r)
  done=start+batch
  if done%5120==0 or done==total:print('Plugin MC',kind,year,stream,done,'/',total,flush=True);save('runtime/checkpoints/'+prefix+str(done),dict(moments=acc,done=done,total=total,seed=seed,model=model,rng_state=rng.bit_generator.state))

 return dict(moments=acc,n=total,seed=seed)

def choose(model,kind,year):
 stem=f'runtime/checkpoints/plugin_{kind}_{year}';v=cached(stem)
 if v is not None:return v
 sel=moments(model,kind,year,'selection');report=moments(model,kind,year,'reporting');out={}
 for gate in ['G0','G1']:
  out[gate]={}
  for c in [5,2,10]:
   m=sel['moments'][gate];net=(m[0]-c*m[1])/sel['n'];bits=int(np.argmax(net));r=report['moments'][gate];n=report['n'];mean,se=mean_se(r,c,bits,n)
   out[gate][str(c)]=dict(kind=0,bits=bits,model=model,selection_net_bps=float(net[bits]),fitted_model_value_bps=float(mean),MC_SE_bps=float(se),selection_seed=sel['seed'],reporting_seed=report['seed'],selection_years=n,reporting_years=n,label='fitted-model Monte Carlo policy evaluation (approximate)')
 count('Plugin_MC_years',100000);return save(stem,out)

def mean_se(moments,cost,bits,n):
 r=np.asarray(moments,float);mean=(r[0,bits]-cost*r[1,bits])/n;ss=r[2,bits]+cost*cost*r[3,bits]-2*cost*r[4,bits];se=np.sqrt(max(0,ss-n*mean*mean)/(n-1)/n);return float(mean),float(se)
