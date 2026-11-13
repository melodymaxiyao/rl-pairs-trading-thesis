"""Imported trainer body, plus an explicitly separate OU state/environment shim."""
from common import *
import ctypes as C,subprocess,re,ast,types
VENDOR=RUN/'vendor/experiments/15_pair_trading'
DP=C.POINTER(C.c_double);IP=C.POINTER(C.c_int);LP=C.POINTER(C.c_int64)
def dp(a):return np.ascontiguousarray(a).ctypes.data_as(DP)
def ip(a):return np.ascontiguousarray(a,dtype=np.int32).ctypes.data_as(IP)
def lp(a):return np.ascontiguousarray(a,dtype=np.int64).ctypes.data_as(LP)
def region(text,start):
 first=text.index(start);a=text.index('{',first);depth=1;i=a+1
 while depth:
  depth+=(text[i]=='{')-(text[i]=='}');i+=1
 return text[first:i]
def build():
 registration=json.loads((RUN/'audit/source_reconstruction.json').read_text())['found']
 for key,v in registration.items():assert sha(RUN/'vendor'/key)==v['sha256']
 adapter=VENDOR/'trend_anchor_trading_v1/scripts/adapter_kernel.c';original=VENDOR/'trend_anchor_trading_v1/scripts/original_q_kernel.c';mech=VENDOR/'td_mechanism_spi_v1/scripts/mechanism_kernel.c';tuned=VENDOR/'trend_anchor_td_tuned_v1/scripts/tuned_kernel.c'
 body=region(mech.read_text(),'void ms_train(');greedy=region(adapter.read_text(),'static int ta_greedy(');evaluate=region(adapter.read_text(),'static double ta_evaluate(')
 shim='''
static int ta_state(double d,int p,int align,int strong,int feature){
 if(!p){double z=fabs(d);return z<.5?0:z<1.5?1:z<2.5?2:3;}
 double z=-p*d;return 4+(z>=1.5?0:z>=.5?1:z>=-.5?2:3);
}
static int ta_targets(Env*e,double d,int terminal,int H,double stop,int lock,int align,int allowcounter,int*reason,int*bm,int targets[2]){
 targets[0]=e->prev;targets[1]=e->prev?0:(d>0?-1:d<0?1:0);*reason=0;
 if(terminal||fabs(d)>=stop){*reason=terminal?3:1;*bm=1<<(e->prev?1:0);}
 else *bm=(!e->prev&&d==0)?1:3;
 return *bm;
}
'''
 text='#include '+json.dumps(str(original))+'\n'+shim+'\n'+greedy+'\n'+evaluate+'\n'+body+'\n'
 publish(RUN/'runtime/ou_kernel.c',text)
 # Only the include pathname is routed; all legacy numerical bytes remain unchanged.
 legacy=tuned.read_text();legacy=legacy.replace('"/Users/maxiyao/Desktop/FYP/experiments/15_pair_trading/trend_anchor_trading_v1/scripts/adapter_kernel.c"',json.dumps(str(adapter)))
 publish(RUN/'runtime/legacy_tuned_kernel.c',legacy)
 legacy_ms=mech.read_text().replace('"/Users/maxiyao/Desktop/FYP/experiments/15_pair_trading/trend_anchor_trading_v1/scripts/adapter_kernel.c"',json.dumps(str(adapter)))
 publish(RUN/'runtime/legacy_mechanism_kernel.c',legacy_ms)
 builds={}
 for name in ['ou_kernel','legacy_tuned_kernel','legacy_mechanism_kernel']:
  target=RUN/'runtime'/(name+'.so')
  if not target.exists():
   command=['/usr/bin/cc','-O3','-std=c11','-shared','-fPIC',str(RUN/'runtime'/(name+'.c')),'-o',str(target)+'.pending_build']
   process=subprocess.run(command,capture_output=True,text=True);write(RUN/f'audit/build_{name}.json',dict(command=command,returncode=process.returncode,stderr=process.stderr));assert process.returncode==0,process.stderr
   Path(str(target)+'.pending_build').rename(target)
  builds[name]=sha(target)
 proof=dict(status='PASS',trainer_body_sha256=hashlib.sha256(body.encode()).hexdigest(),original_trainer_file_sha256=sha(mech),greedy_body_sha256=hashlib.sha256(greedy.encode()).hexdigest(),rng_reward_and_advance_file_sha256=sha(original),only_environment_and_state_adapter_changed=True,legacy_pathname_routing_only=True,compiled=builds)
 write(RUN/'audit/kernel_import_proof.json',proof);return proof
LIBRARIES={}
def library(name='ou_kernel'):
 if name in LIBRARIES:return LIBRARIES[name]
 k=C.CDLL(str(RUN/'runtime'/(name+'.so')))
 common=[C.c_int,IP,DP,DP,IP,DP,C.c_int,IP,DP,DP,IP,DP,C.c_uint64,C.c_double,C.c_double,C.c_double,C.c_int,C.c_int,C.c_int,C.c_int,C.c_int,C.c_double,C.c_int]
 if name=='legacy_tuned_kernel':k.sc_train.argtypes=common+[C.c_int,DP,DP,IP,LP,LP];k.sc_train.restype=None
 else:k.ms_train.argtypes=common+[C.c_int,C.c_int,IP,C.c_int,DP,DP,IP,LP,LP,DP,DP];k.ms_train.restype=None
 LIBRARIES[name]=k;return k
def fit_arrays(z,returns,seed,step='POW',cost=10.,stop=4.,epochs=300,early=True,legacy=False,pairids=None,drift=None,strong=None):
 z=np.ascontiguousarray(z,float);returns=np.ascontiguousarray(returns,float);ne,n=z.shape;assert returns.shape==z.shape
 offsets=np.arange(ne+1,dtype=np.int32)*n;d=z.ravel();r=returns.ravel();g=np.zeros_like(d) if drift is None else np.ascontiguousarray(drift,float).ravel();s=np.zeros(len(d),np.int32) if strong is None else np.ascontiguousarray(strong,np.int32).ravel();vo=np.array([0],np.int32);empty=np.empty(0);ei=np.empty(0,np.int32)
 q=np.zeros((8,2));curves=np.full((epochs,3),np.nan);snaps=np.zeros((epochs//10,8),np.int32);visits=np.zeros((8,2),np.int64);out=np.zeros(7,np.int64);diag=np.zeros((16,10));trajectory=np.zeros((epochs,16));pairs=np.arange(ne,dtype=np.int32) if pairids is None else np.ascontiguousarray(pairids,np.int32)
 args=[ne,ip(offsets),dp(d),dp(g),ip(s),dp(r),0,ip(vo),dp(empty),dp(empty),ip(ei),dp(empty),int(seed),.01 if step=='C01' else .05,1.,float(cost),epochs,int(early),0,int(legacy),12 if legacy else 0,float(stop),int(legacy)]
 name='legacy_mechanism_kernel' if legacy else 'ou_kernel'
 library(name).ms_train(*args,0 if step=='C01' else 1,0,ip(pairs),len(set(pairs)),dp(q),dp(curves),ip(snaps),lp(visits),lp(out),dp(diag),dp(trajectory))
 count('TD_seed_fits');count('learning_steps',int(out[4]));n_epochs=int(out[0])
 return dict(q=q,visits=visits,curves=curves[:n_epochs],snapshots=snaps[:n_epochs//10],trajectory=trajectory[:n_epochs].reshape(-1,8,2),diagnostics=diag.reshape(8,2,10),epochs=n_epochs,converged=bool(out[1]),convergence_epoch=int(out[2]),learning_steps=int(out[4]))
def train(paths,scale,stop_sd,step,cost=10.,epochs=300,checkpoint=None):
 z=paths/scale;returns=np.column_stack([np.diff(paths,axis=1),np.zeros(len(paths))]);pairs=np.arange(len(paths),dtype=np.int32)//3
 fits=[]
 for j in range(10):
  stem=f'{checkpoint}/{step}_seed{j}' if checkpoint else None;fit=cached(stem) if stem else None
  if fit is None:
   fit=fit_arrays(z,returns,td_seed(j),step,cost,4*stop_sd/scale,epochs,pairids=pairs)
   if stem:save(stem,fit)
  fits.append(fit)
  if stem:checkpoint_guard()
 q=np.mean([f['q'] for f in fits],axis=0);bits=sum(int(q[s,1]>q[s,0])<<s for s in range(8));return dict(q=q,bits=bits,scale=scale,fits=fits,seeds=[td_seed(j) for j in range(10)])
def sealed_fixture():
 source=INPUT/'experiments/15_pair_trading/trend_anchor_td_tuned_v1/runtime/acceptance/resume_fixture.pkl.gz';registry=json.loads((RUN/'audit/input_registration.json').read_text());assert sha(source)==registry['fixture']['sha256']
 with gzip.open(source,'rb') as f:value=pickle.load(f)
 assert hashlib.sha256(value['payload']).hexdigest()==value['sha256'];expected=pickle.loads(value['payload'])['values'];assert len(expected)==2
 # Execute the unchanged sealed tiny() body. Only its common IO namespace is supplied.
 p=VENDOR/'trend_anchor_td_tuned_v1/scripts/fixtures.py';text=p.read_text();node=next(v for v in ast.parse(text).body if isinstance(v,ast.FunctionDef) and v.name=='tiny');fragment=ast.get_source_segment(text,node)
 import pandas as pd
 namespace=dict(np=np,pd=pd);exec(compile(fragment,str(p),'exec'),namespace);episodes=namespace['tiny']();z=np.array([e['z'] for e in episodes]);returns=np.array([e['forward_bps'] for e in episodes]);drift=np.array([e['drift'] for e in episodes]);strong=np.array([e['strong'] for e in episodes]);checks=[]
 for seed,reference in zip([111,222],expected):
  fit=fit_arrays(z,returns,seed,step='POW',stop=5.,epochs=20,early=False,legacy=True,drift=drift,strong=strong);actual=hashlib.sha256(fit['q'].tobytes()+fit['visits'].tobytes()).hexdigest();assert actual==reference,('Sealed TD_tuned fixture mismatch',seed,actual,reference);checks.append(dict(seed=seed,sha256=actual))
 return dict(status='PASS',fixtures=checks,real_price_reads=0,fixture='sealed TD_tuned resume fixture and unchanged tiny() generator',kernel='unchanged instrumented POW body in original environment')
