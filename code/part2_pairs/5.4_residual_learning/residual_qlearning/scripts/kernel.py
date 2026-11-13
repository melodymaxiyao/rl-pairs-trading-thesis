"""Unchanged POW/Double update bodies; entry-frozen hedge reward environment."""
from common import *
import ctypes as C,re
DP=C.POINTER(C.c_double);IP=C.POINTER(C.c_int);LP=C.POINTER(C.c_int64)
def dp(a):return a.ctypes.data_as(DP)
def ip(a):return a.ctypes.data_as(IP)
def lp(a):return a.ctypes.data_as(LP)
def region(text,needle):
 i=text.index(needle);j=text.index('{',i)+1;n=1
 while n:n+=(text[j]=='{')-(text[j]=='}');j+=1
 return text[i:j]
def build():
 ou=vendor('ou_realstrength_v1','vendor/ou/runtime/ou_kernel.c').read_text();base=vendor('ou_dp_qlearning_v1','vendor/experiments/15_pair_trading/trend_anchor_trading_v1/scripts/original_q_kernel.c')
 double=vendor('ou_qfix_v1','scripts/qfix.c').read_text();old=ou.split('\n',1)[0];assert old.startswith('#include ')
 callback='''
static const double*reward_cube=NULL;static int reward_T=0,reward_mode=0,reward_entry=0,reward_failed=0;
void set_environment(const double*x,int T,int mode){reward_cube=x;reward_T=T;reward_mode=mode;reward_entry=0;reward_failed=0;}
int environment_failed(void){return reward_failed;}
double reward_bps(int p,int prev,double forward,double c){
 if(!reward_mode)return original_scalar_reward(p,prev,forward,c);
 int idx=(int)forward,episode=idx/reward_T,t=idx%reward_T;
 if(!prev&&p)reward_entry=t;
 double r=p?reward_cube[((size_t)episode*reward_T+reward_entry)*reward_T+t]:0.;
 if(p&&!isfinite(r)){reward_failed=1;return NAN;}
 return p*r-c*abs(p-prev)/2.;
}
'''
 routed='#define reward_bps original_scalar_reward\n#include '+json.dumps(str(base))+'\n#undef reward_bps\n'+callback+ou[len(old):]
 # Original source update bodies are compared literally, including whitespace.
 assert region(routed,'void ms_train(')==region(ou,'void ms_train(')
 publish(RUN/'runtime/frozen_ou.c',routed)
 routed_double=double.replace(double.split('\n')[1],'#include "frozen_ou.c"',1)
 assert region(routed_double,'void double_train(')==region(double,'void double_train(')
 publish(RUN/'runtime/frozen_q.c',routed_double+'\n'+(RUN/'scripts/policy_kernel.c').read_text())
 target=RUN/'runtime/frozen_q.so'
 if not target.exists():
  p=subprocess.run(['/usr/bin/cc','-O3','-std=c11','-shared','-fPIC',str(RUN/'runtime/frozen_q.c'),'-o',str(target)+'.pending_build'],capture_output=True,text=True);write(event('compile'),dict(returncode=p.returncode,stderr=p.stderr));assert p.returncode==0,p.stderr;Path(str(target)+'.pending_build').rename(target)
 proof=dict(ms_train_sha256=hashlib.sha256(region(ou,'void ms_train(').encode()).hexdigest(),double_train_sha256=hashlib.sha256(region(double,'void double_train(').encode()).hexdigest(),trainer_bodies_unchanged=True,environment='entry-frozen gross-normalised reward cube; scalar mode exact old reward',binary_sha256=sha(target))
 write(RUN/'audit/kernel_import_proof.json',proof);return proof
_LIB=None
def library():
 global _LIB
 if _LIB is None:
  assert sha(RUN/'runtime/frozen_q.so')==read(RUN/'audit/kernel_import_proof.json')['binary_sha256'],'Native binary changed'
  _LIB=C.CDLL(str(RUN/'runtime/frozen_q.so'));_LIB.set_environment.argtypes=[DP,C.c_int,C.c_int];_LIB.environment_failed.restype=C.c_int
  _LIB.double_train.argtypes=[C.c_int,C.c_int,DP,DP,C.c_uint64,C.c_uint64,C.c_double,C.c_int,C.c_int,DP,LP,DP,DP,LP]
  common=[C.c_int,IP,DP,DP,IP,DP,C.c_int,IP,DP,DP,IP,DP,C.c_uint64,C.c_double,C.c_double,C.c_double,C.c_int,C.c_int,C.c_int,C.c_int,C.c_int,C.c_double,C.c_int]
  _LIB.ms_train.argtypes=common+[C.c_int,C.c_int,IP,C.c_int,DP,DP,IP,LP,LP,DP,DP]
  _LIB.mc_moments.argtypes=[C.c_int,C.c_int,DP,DP,DP]
  _LIB.position_paths.argtypes=[C.c_int,C.c_int,DP,C.c_int,C.c_int,C.c_double,C.c_double,IP]
  _LIB.enumerate_values.argtypes=[C.c_int,C.c_int,DP,DP,C.c_double,DP,DP]
 return _LIB

def fit(z,cube,seed,cost,double=False,epochs=300,early=True,scalar_returns=None):
 count('real_kernel_seed_fits' if os.environ.get('OU_SCOPE')=='production' else 'synthetic_fixture_kernel_seed_fits')
 z=np.ascontiguousarray(z,float);z=np.where(np.isfinite(z),z,np.inf);ne,T=z.shape
 scalar=scalar_returns is not None;r=np.ascontiguousarray(scalar_returns if scalar else np.arange(ne*T).reshape(ne,T),float);cube=np.ascontiguousarray(cube,float)
 k=library();k.set_environment(dp(cube),T,0 if scalar else 1)
 curves=np.zeros((epochs,3));result=np.zeros(7,np.int64)
 if double:
  q=np.zeros((2,8,2));vis=np.zeros((2,8,2),np.int64);diag=np.zeros((2,8,2,10));coin=stable_seed('ou_qfix_v1','double_table_choice',int(seed))
  k.double_train(ne,T,dp(z),dp(r),int(seed),coin,float(cost),epochs,int(early),dp(q),lp(vis),dp(diag),dp(curves),lp(result))
 else:
  q=np.zeros((8,2));vis=np.zeros((8,2),np.int64);diag=np.zeros((8,2,10));snaps=np.zeros((epochs//10,8),np.int32);trajectory=np.zeros((epochs,16));offsets=np.arange(ne+1,dtype=np.int32)*T;g=np.zeros(ne*T);strong=np.zeros(ne*T,np.int32);zero=np.array([0],np.int32);empty=np.empty(0);ei=np.empty(0,np.int32);pairs=np.arange(ne,dtype=np.int32)//3
  k.ms_train(ne,ip(offsets),dp(z),dp(g),ip(strong),dp(r),0,ip(zero),dp(empty),dp(empty),ip(ei),dp(empty),int(seed),.05,1.,2*float(cost),epochs,int(early),0,0,0,np.inf,0,1,0,ip(pairs),len(set(pairs)),dp(q),dp(curves),ip(snaps),lp(vis),lp(result),dp(diag),dp(trajectory))
 assert not k.environment_failed(),'Unknown held-leg return in training; no invented returns allowed'
 assert np.isfinite(q).all();return dict(q=q,visits=vis,diagnostics=diag,curves=curves[:result[0]],epochs=int(result[0]),converged=bool(result[1]),convergence_epoch=int(result[2]),learning_steps=int(result[4]),seed=int(seed))

def policy(fits,double=False,lcb=False):
 q=np.mean([np.mean(f['q'],axis=0) if double else f['q'] for f in fits],axis=0)
 ses=[]
 for f in fits:
  d=f['diagnostics'];se=d[...,7]*np.sqrt(d[...,1]);ses.append(se.mean(axis=0) if double else se)
 se=np.mean(ses,axis=0);action=q[:,1]>q[:,0]
 if lcb:margin=np.sqrt(se[:4,0]**2+se[:4,1]**2);action[:4]=np.isfinite(margin)&((q[:4,1]-q[:4,0])>1.645*margin)
 return dict(kind=0,bits=sum(int(v)<<i for i,v in enumerate(action)),q=q,se=se,SE_label='nominal TD-target-noise SE; no 95% coverage guarantee')

def positions(z,p):
 z=np.ascontiguousarray(np.where(np.isfinite(z),z,np.inf),float);out=np.empty(z.shape,np.int32);library().position_paths(*z.shape,dp(z),int(p.get('kind',0)),int(p.get('bits',0)),float(p.get('enter',1.25)),float(p.get('exit',.5)),ip(out));return out

def enumerate_mc(z,r,cost):
 z=np.ascontiguousarray(np.where(np.isfinite(z),z,np.inf),float);r=np.ascontiguousarray(r,float);s=np.zeros(256);ss=np.zeros(256);library().enumerate_values(*z.shape,dp(z),dp(r),float(cost),dp(s),dp(ss));return s,ss

def mc_moments(z,r):
 z=np.ascontiguousarray(np.where(np.isfinite(z),z,np.inf),float);r=np.ascontiguousarray(r,float);out=np.zeros((5,256));library().mc_moments(*z.shape,dp(z),dp(r),dp(out));return out

def cube_values(z,cube,pol,cost,realized=False):
 p=positions(z,pol);n,T=p.shape;entry=np.zeros(n,int);previous=np.zeros(n,int);g=np.zeros((n,T));fees=np.zeros((n,T));ii=np.arange(n)
 for t in range(T):
  new=p[:,t];entry[(previous==0)&(new!=0)]=t;f=cube[ii,entry,t];held=new!=0;assert np.isfinite(f[held]).all(),'Unknown held-leg training return'
  g[held,t]=new[held]*f[held];fees[:,t]=cost*abs(new-previous);previous=new
 if realized:g=np.column_stack([np.zeros(n),g[:,:-1]])
 return g-fees
