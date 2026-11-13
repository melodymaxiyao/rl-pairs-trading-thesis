"""Finite-grid Bellman recursion and exhaustive stationary8-state evaluation."""
from common import *
from simulation import Model
from scipy.special import ndtr
from scipy.stats import t as student_t
POSITIONS=np.array([-1,0,1])
def states(x,position,scale):
 s=np.asarray(x)/scale
 if position==0:return np.searchsorted([.5,1.5,2.5],np.abs(s),side='right')
 z=-np.sign(position)*s;return 4+np.where(z>=1.5,0,np.where(z>=.5,1,np.where(z>=-.5,2,3)))
def grid_model(model,points=401,width=6.,reference_sd=None):
 scale=model.sd if reference_sd is None else reference_sd;x=np.linspace(-width*scale,width*scale,points);edges=np.r_[-np.inf,(x[1:]+x[:-1])/2,np.inf];mu=model.mean+model.phi*(x-model.mean)
 cdf=student_t.cdf((edges[None,:]-mu[:,None])/(model.sigma_e/np.sqrt(2)),4) if model.family=='T4' else ndtr((edges[None,:]-mu[:,None])/model.sigma_e)
 kernel=np.maximum(np.diff(cdf,axis=1),0.);kernel/=kernel.sum(axis=1)[:,None]
 # Discretized stationary-law initial mass; T4 uses an independent MC stationary sample.
 if model.family=='T4':
  from simulation import simulate
  starts=simulate(model,20000,['t4_grid_initial'],sessions=1)[:,0];initial=np.histogram(starts,edges)[0].astype(float);initial/=initial.sum()
 else:initial=np.diff(ndtr((edges-model.mean)/model.sd));initial/=initial.sum()
 return x,kernel,initial
def destinations(x,position,stop_sd,scale=None,bits=None,rule=None):
 x=np.asarray(x);flat=position==0;target=np.full(x.shape,position,int)
 if bits is not None:
  action=(int(bits)>>states(x,position,scale))&1
 elif rule=='AL rule':action=(np.abs(x/scale)>1.25) if flat else (-np.sign(position)*x/scale<.5)
 elif rule=='L0-type rule':action=(np.abs(x/scale)>=2) if flat else (-np.sign(position)*x/scale<=0)
 else:raise ValueError('Missing policy')
 target=np.where(action, -np.sign(x).astype(int) if flat else 0,target)
 return np.where(np.abs(x)>=4*stop_sd,0,target)
def policy_array(x,scale,stop_sd,bits=None,rule=None):return np.asarray([destinations(x,p,stop_sd,scale,bits,rule) for p in POSITIONS])
def evaluate(x,kernel,initial,actions,cost=10.,sessions=252,details=False,stationary_batch=False):
 actions=np.asarray(actions,int);batched=stationary_batch
 # Batched stationary actions:3×N×K; single time-dependent: T×3×N.
 if actions.ndim==2:actions=actions[...,None];batched=True
 k=actions.shape[-1] if batched else 1;V=np.broadcast_to(-cost/2*np.abs(POSITIONS)[:,None,None],(3,len(x),k)).copy();dx=kernel@x-x;history=[None]*(sessions+1) if details else None
 if details:assert k==1;history[sessions]=V[...,0].copy()
 for day in range(sessions-1,-1,-1):
  future=np.stack([kernel@V[j] for j in range(3)]);chosen=actions if batched else actions[day,...,None];out=np.empty_like(V)
  for j,p in enumerate(POSITIONS):
   a=chosen[j];out[j]=a*dx[:,None]-cost/2*np.abs(a-p)+np.take_along_axis(future.transpose(1,2,0),(a+1)[...,None],axis=2)[...,0]
  V=out
  if details:history[day]=V[...,0].copy()
 values=initial@V[1]
 return (values,history) if details else values
def optimal(x,kernel,initial,stop_sd,cost=10.,sessions=252):
 V=np.broadcast_to(-cost/2*np.abs(POSITIONS)[:,None],(3,len(x))).copy();dx=kernel@x-x;policy=np.zeros((sessions,3,len(x)),np.int8)
 for day in range(sessions-1,-1,-1):
  future=V@kernel.T;out=np.empty_like(V)
  for j,p in enumerate(POSITIONS):
   other=-np.sign(x).astype(int) if p==0 else np.zeros(len(x),int);stay=p*dx+future[j];change=other*dx-cost/2*np.abs(other-p)+future[other+1,np.arange(len(x))];allow=(np.abs(x)<4*stop_sd)&((x!=0) if p==0 else True);chosen=np.where(allow&(change>stay),other,p);chosen=np.where(np.abs(x)>=4*stop_sd,0,chosen);policy[day,j]=chosen;out[j]=chosen*dx-cost/2*np.abs(chosen-p)+future[chosen+1,np.arange(len(x))]
  V=out
 return dict(value=float(initial@V[1]),actions=policy)
def oracle8(x,kernel,initial,scale,stop_sd,cost=10.,sessions=252):
 actions=np.stack([policy_array(x,scale,stop_sd,bits=b) for b in range(256)],axis=2);values=evaluate(x,kernel,initial,actions,cost,sessions,stationary_batch=True);best=int(np.argmax(values));count('oracle8_policy_values',256)
 return dict(bits=best,value=float(values[best]),values=values)
def physical_equal(x,bits_a,scale_a,bits_b,scale_b,stop_sd):return bool(np.array_equal(policy_array(x,scale_a,stop_sd,bits_a),policy_array(x,scale_b,stop_sd,bits_b)))
def exact_q_projection(x,kernel,initial,bits,scale,stop_sd,cost=10.,sessions=252):
 chosen=policy_array(x,scale,stop_sd,bits);_,history=evaluate(x,kernel,initial,chosen,cost,sessions,details=True);mass=np.zeros((3,len(x)));mass[1]=initial;num=np.zeros((8,2));den=np.zeros((8,2));dx=kernel@x-x
 for day in range(sessions+1):
  future=history[day+1]@kernel.T if day<sessions else np.zeros((3,len(x)))
  for j,p in enumerate(POSITIONS):
   labels=states(x,p,scale)
   for action in [0,1]:
    target=np.full(len(x),p) if action==0 else (-np.sign(x).astype(int) if p==0 else np.zeros(len(x),int));valid=np.ones(len(x),bool)
    if day==sessions:valid[:]=action==(0 if p==0 else 1);target[:]=0;value=-cost/2*abs(p)*np.ones(len(x))
    else:
     valid&=(np.abs(x)<4*stop_sd)| (action==(0 if p==0 else 1));valid&=~((p==0)&(action==1)&(x==0));value=target*dx-cost/2*np.abs(target-p)+future[target+1,np.arange(len(x))]
    for b in range(8):
     w=mass[j]*(labels==b)*valid;num[b,action]+=w@value;den[b,action]+=w.sum()
  if day<sessions:
   incoming=np.zeros_like(mass)
   for j in range(3):
    for target in POSITIONS:incoming[target+1]+=mass[j]*(chosen[j]==target)
   mass=incoming@kernel
 return np.divide(num,den,out=np.full_like(num,np.nan),where=den>0),den
