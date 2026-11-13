"""Independent series-year batches; no raw paths or episode arrays are persisted."""
import ast,dataclasses,hashlib,json,pathlib,resource,sys,types
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from io_utils import *
import gate_tests as gt
from real_jobs import done,finish
def imported_functions():
 import bootstrap
 c,d,a=bootstrap.core_modules();source=c.seal('ou_realstrength_v1');module=types.ModuleType('_gate_frozen_synthetic');sys.modules[module.__name__]=module
 ns=module.__dict__;ns.update(np=np,dataclass=dataclasses.dataclass,count=count,hashlib=hashlib,json=json)
 p=source.verify('scripts/common.py');tree=ast.parse(p.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='stable_seed'];exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns)
 p=source.verify('scripts/simulation.py');tree=ast.parse(p.read_text());nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Model' or isinstance(n,ast.FunctionDef) and n.name=='simulate'];exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns)
 calibration=J(source.verify('calibration.json'));assert calibration['chosen']['h0']==20 and calibration['chosen']['w']==.6 and calibration['second']['h0']==8 and calibration['second']['w']==.8
 return ns['Model'],ns['simulate'],d.M.ou_fit,ns['stable_seed'],calibration
def summarize(values):
 v=np.asarray(values,float);good=np.isfinite(v);return dict(n=int(good.sum()),sum=float(v[good].sum()),sumsq=float((v[good]**2).sum()),undefined=int((~good).sum()))
def policy_net(scores,valid,returns,mask):
 # Same real causal target: gross on old position, then atomic decision/fee,
 # forced invalid exit, terminal liquidation on the final actual session.
 n,T=scores.shape;old=np.zeros(n,int);net=np.zeros(n);turn=np.zeros(n)
 for t in range(T):
  z=scores[:,t];ok=valid[:,t]&mask[:,t];new=old.copy();flat=old==0;exit_rule=(-old*z)<.5
  new[(~flat)&exit_rule]=0;enter=flat&ok&(abs(z)>1.25);new[enter]=-np.sign(z[enter]).astype(int);new[~ok]=0
  if t==T-1:new[:]=0
  cost=abs(new-old);net+=old*returns[:,t]-5*cost;turn+=cost;old=new
 return net,turn
def null_label(world,test):
 if test=='KPSS':true=world.startswith('OU')
 elif test.startswith('VR'):true=world=='RW'
 else:true=not world.startswith('OU')
 return 'H0 true' if true else 'H0 false'
def batch(world,index):
 root=ROOT/f'runtime/synthetic/{world}_{index:04d}'
 if done(root):return dict(cached=True,root=str(root))
 Model,simulate,ou_fit,stable_seed,calibration=imported_functions();spec=J(ROOT/'preregistration.json');n=spec['part_B']['batch_series'];start=index*n;assert start+n<=20000
 point=calibration['second'] if world=='second' else calibration['chosen'];scale=point['sigma_total']
 if world=='RW':model=Model(20,scale,family='W',w=1.)
 elif world.startswith('OU'):model=Model(float(world[2:]),scale,family='OU',w=0.)
 else:model=Model(point['h0'],scale,family='W',w=point['w'])
 paths=simulate(model,n,['gate_power_v1','B',20261102,world],sessions=502,start=start);e=np.diff(paths,axis=1)
 # Decision day0 observes250 history increments and starts flat, as in real replay.
 recent=sliding_window_view(e,60,axis=1)[:,190:442];fit=ou_fit(recent);scores=fit['s'];half=fit['half_life'];valid=np.isfinite(scores)&np.isfinite(half)&(half>0)&(half<30);returns=e[:,249:501]
 ungated,turn=policy_net(scores,valid,returns,np.ones_like(valid));rows=[]
 for W in [60,120,250]:
  checkpoint();observations=sliding_window_view(e,W,axis=1)[:,250-W:502-W];assert observations.shape==(n,252,W)
  # Limit SVD workspace independently of batch size and window length.
  test_values={name:np.empty((n,252)) for name in ['ADF','DFGLS','KPSS','VR5','VR10']}
  for offset in range(0,n,10):
   sample=observations[offset:offset+10];values=gt.statistics(sample)
   for name,v in values.items():test_values[name][offset:offset+len(v)]=v
  for name,values in test_values.items():
   finite=np.isfinite(values);reject=gt.rejection(name,values,W);mask=gt.gate(name,values,W);net,gturn=policy_net(scores,valid,returns,mask);den=finite.sum(axis=1);rates=np.divide(reject.sum(axis=1),den,out=np.full(n,np.nan),where=den>0)
   rows.append(dict(world=world,h0=model.half_life,w=model.w,window=W,test=name,null_status=null_label(world,name),rate_interpretation='size' if null_label(world,name)=='H0 true' else 'power',rejection=summarize(rates),reject_count=int(reject.sum()),observations=int(finite.sum()),net=summarize(net),turnover=summarize(gturn),ungated_net=summarize(ungated),ungated_turnover=summarize(turn),net_removed=summarize(ungated-net),final_window_rejections=int(reject[:,-1].sum()),final_window_n=int(finite[:,-1].sum()),c=W*(1-model.phi)))
 finish(root,{'sums.json':rows},phase='B',world=world,index=index,series_years=n);return dict(root=str(root),series_years=n)
def theory(c,index):
 root=ROOT/f'runtime/theory/c{c:g}_{index:03d}'
 if done(root):return dict(cached=True,root=str(root))
 Model,simulate,ou_fit,seed,calibration=imported_functions();n=100;steps=4096;phi=np.exp(-c/steps);series=[]
 for j in range(n):
  rng=np.random.default_rng(seed('gate_power_v1','local_diffusion',20261102,c,index*n+j));v=np.empty(steps+1);v[0]=rng.normal()/np.sqrt(2*c) if c>0 else 0.;noise=rng.normal(size=steps)/np.sqrt(steps)
  for t in range(steps):v[t+1]=phi*v[t]+noise[t]
  x=v[:-1];y=np.diff(v);xc=x-x.mean();yc=y-y.mean();b=(xc@yc)/(xc@xc);res=yc-b*xc;t=b/np.sqrt((res@res/(steps-2))/(xc@xc));series.append(t< -2.86154)
 finish(root,{'sums.json':dict(c=c,n=n,rejected=sum(series),steps=steps,initialization='stationary OU initial state for c>0; zero RW origin atc0',reference='local-to-unity diffusion Monte Carlo reference, not a closed-form exact power formula')},phase='theory');return dict(root=str(root))
def size_check():
 root=ROOT/'runtime/size_checks'
 if done(root):return J(root/'size.json')
 rows=[]
 for W in [250]:
  rw=[J(p/'sums.json') for p in sorted((ROOT/'runtime/synthetic').glob('RW_*')) if done(p)];assert len(rw)==200
  for name in ['ADF','DFGLS','VR5','VR10']:
   values=[v for pack in rw for v in pack if v['window']==W and v['test']==name];n=sum(v['final_window_n'] for v in values);k=sum(v['final_window_rejections'] for v in values);rate=k/n;tolerance=max(.015,4*np.sqrt(.05*.95/n));rows.append(dict(test=name,W=W,null='pure RW',n=n,rejections=k,rate=rate,tolerance=tolerance,pass_size=bool(abs(rate-.05)<=tolerance)))
  rng=np.random.default_rng(20261102);k=n=0
  for offset in range(0,20000,100):
   checkpoint();levels=rng.normal(size=(100,250));increments=np.column_stack([levels[:,0],np.diff(levels,axis=1)]);v=gt.kpss(increments);k+=int(gt.rejection('KPSS',v,250).sum());n+=len(v)
  rate=k/n;tolerance=max(.015,4*np.sqrt(.05*.95/n));rows.append(dict(test='KPSS',W=W,null='iid level noise',n=n,rejections=k,rate=rate,tolerance=tolerance,pass_size=bool(abs(rate-.05)<=tolerance)))
 finish(root,{'size.json':rows},phase='null_size_checks');return rows
