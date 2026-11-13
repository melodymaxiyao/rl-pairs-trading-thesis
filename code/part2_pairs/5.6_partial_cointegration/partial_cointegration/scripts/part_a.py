"""Independent100-series panel-years, frozen simulations and Oracle8 executor."""
import ast,ctypes as C,dataclasses,types,sys,gzip
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from io_utils import *
from groups import *
import pci,nulls,residuals,llt,time
from scipy.special import ndtr
def imported():
 import bootstrap
 c,d,a=bootstrap.core_modules();seal=c.seal('ou_realstrength_v1');namespace=dict(np=np,dataclass=dataclasses.dataclass,count=count,hashlib=hashlib,json=json);module=types.ModuleType('_partial_frozen_sim');sys.modules[module.__name__]=module;module.__dict__.update(namespace);ns=module.__dict__
 for rel,names in [('scripts/common.py',['stable_seed']),('scripts/simulation.py',['Model','simulate'])]:
  p=seal.verify(rel);tree=ast.parse(p.read_text());nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names];exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns)
 source=seal.verify('scripts/observed.py');node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='execute');DP=C.POINTER(C.c_double);IP=C.POINTER(C.c_int);ns.update(C=C,DP=DP,IP=IP,dp=lambda x:x.ctypes.data_as(DP),ip=lambda x:x.ctypes.data_as(IP),LIB=None,RUN=seal.root)
 # Pin the saved native executor rather than compile or write into its source run.
 binary=seal.verify('runtime/observed.so');lib=C.CDLL(str(binary));lib.observed.argtypes=[C.c_int,C.c_int,DP,DP,C.POINTER(C.c_ubyte),C.c_int,C.c_int,C.c_int,C.c_double,C.c_double,C.c_double,DP,DP,C.c_int];lib.observed.restype=None;ns['LIB']=lib;exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
 calibration=J(seal.verify('calibration.json'));return ns,calibration,d
def world_model(name,ns,cal):
 point=cal['second'] if name=='second' else cal['chosen'];h=point['h0'];w=point['w'];sigma=point['sigma_total']
 if name=='OU20':h=20;w=0
 if name=='RW':w=1
 return ns['Model'](h,sigma,family='W',w=w),h,w
def observations(paths,pooled,d,warm=None):
 fitN=756;fit=pci.fit(paths[:,:fitN],pooled=True,checkpoint=checkpoint,warm_start=warm) if pooled is None else pooled;n,T=paths.shape;score=np.full((n,T),np.nan)
 if fit['converged']:
  _,states=pci.filter_values(paths,fit['rho'],fit['q'],True);score=states[:,:,1]/np.sqrt(np.asarray(fit['scale'])[:,None]/(1-fit['rho']**2))
  if not fit['half_life']<30:score[:]=np.nan
 e=np.diff(paths,axis=1);f=d.M.ou_fit(sliding_window_view(e,60,axis=1));roll=np.full_like(paths,np.nan);roll[:,60:]=f['s'];valid_roll=np.zeros_like(paths,bool);valid_roll[:,60:]=np.isfinite(f['s'])&np.isfinite(f['half_life'])&(f['half_life']>0)&(f['half_life']<30)
 return fit,dict(s=np.ascontiguousarray(score[:,756:]),valid=np.isfinite(score[:,756:]),returns=np.ascontiguousarray(e[:,756:])),dict(s=np.ascontiguousarray(roll[:,756:]),valid=valid_roll[:,756:],returns=np.ascontiguousarray(e[:,756:]))
def policy_values(execute,data):
 values=np.empty((256,3))
 for bits in range(256):
  if bits%32==0:checkpoint()
  out=execute(data,dict(kind=0,bits=bits),5);net=out[:,0];values[bits]=[net.sum(),net@net,out[:,2].sum()]
 return values
def job(world,index):
 root=ROOT/f'runtime/part_A/{world}_{index:03d}'
 if done(root):return dict(root=str(root),cached=True)
 ns,cal,d=imported();model,h,w=world_model(world,ns,cal);n=100;start=index*n;execute=ns['execute'];stats=residuals.imported_stats();results={};oracle={};params=[];powers=[]
 previous=ROOT/f'runtime/part_A/{world}_{index-1:03d}';warm=J(previous/'primary.json') if index>0 and done(previous) else {}
 selection=ns['simulate'](model,n,['partial_coint_kalman_v1',20261103,'Oracle_selection',world],sessions=1008,start=start);sf,sK,sroll=observations(selection,None,d,warm.get('selection_fit'))
 oracle['selection_K']=policy_values(execute,sK);oracle['selection_roll_new']=policy_values(execute,sroll);del selection,sK,sroll
 paths=ns['simulate'](model,n,['partial_coint_kalman_v1',20261103,'truth',world],sessions=1008,start=start);fit,K,roll=observations(paths,None,d,warm.get('fit'));oracle['truth_K']=policy_values(execute,K);oracle['truth_roll_new']=policy_values(execute,roll)
 for name,obs in [('Kalman',K),('roll_new',roll)]:
  out=execute(obs,dict(kind=1),5);results[name]=dict(net_sum=float(out[:,0].sum()),net_sumsq=float(out[:,0]@out[:,0]),turnover_sum=float(out[:,2].sum()),series=n)
 ic=[]
 for t in range(252):
  for horizon in [1,5]:
   if t+horizon>252:continue
   target=paths[:,756+t+horizon]-paths[:,756+t];common=K['valid'][:,t]&roll['valid'][:,t]&np.isfinite(target)
   for name,obs in [('Kalman',K),('roll_new',roll)]:ic.append(dict(world=world,panel=index,t=t,h=horizon,signal=name,ic=stats.spearman(-obs['s'][common,t],target[common]),n=int(common.sum())))
 secondary=[];critical= nulls.critical(np.ones(756,bool));testincrements=np.diff(paths,axis=1);adf=d.adf_batch(sliding_window_view(testincrements,60,axis=1)[:,697:949]);assert adf.shape==(100,252)
 for j,x in enumerate(paths):
  checkpoint();v=pci.fit(x[:756],pooled=False,checkpoint=checkpoint);row=dict(world=world,panel=index,series_index=start+j,converged=v['converged'],ADF60_rejection=float(adf[j].mean()),rho=None,R2_MR=None,LR=None,PCI_reject=None,PCI_pass=None,boundary=None,RW_boundary=None)
  if v['converged']:row.update(rho=v['rho'],R2_MR=v['R2_MR'],LR=v['LR'],PCI_reject=bool(v['LR']>critical['critical_5percent']),PCI_pass=bool(v['LR']>critical['critical_5percent'] and v['R2_MR']>=.5),boundary=v['boundary'],RW_boundary=v['RW_boundary'])
  secondary.append(row)
 LLT_clock=time.monotonic();previousLLT=json.loads(gzip.decompress((previous/'LLT_comparison.json.gz').read_bytes())) if index>0 and done(previous) else None;comp=llt.compare(paths[:,:756],fit,warm=previousLLT,checkpoint=checkpoint);LLT_ic=[];slope_ic=[]
 full=comp['full'];LLT_scores=np.full((n,253),np.nan);slopeprob=LLT_scores.copy()
 if full['converged']:
  value=llt.profile(paths,full['rho'],full['q_eta'],full['q_zeta'],states=True)['states'];sc=np.asarray(full['scale']);LLT_scores=value[:,756:,2]/np.sqrt(sc[:,None]/(1-full['rho']**2));variance=value[:,756:,4]*sc[:,None];sd=np.sqrt(variance);ratio=np.divide(value[:,756:,1],sd,out=np.full_like(sd,np.nan),where=sd>0);slopeprob=ndtr(ratio);zero=(sd==0)&np.isfinite(value[:,756:,1]);slopeprob[zero]=(value[:,756:,1][zero]>0).astype(float)
  if not full['half_life']<30:LLT_scores[:]=np.nan
 for t in range(252):
  for h in [1,5]:
   if t+h>252:continue
   y=paths[:,756+t+h]-paths[:,756+t]
   for label,baseline in [('Kalman',K),('roll_new',roll)]:
    common=np.isfinite(LLT_scores[:,t])&baseline['valid'][:,t]&np.isfinite(y)
    LLT_ic.append(dict(world=world,panel=index,t=t,h=h,comparison='LLT_vs_'+label,LLT_IC=stats.spearman(-LLT_scores[common,t],y[common]),comparator_IC=stats.spearman(-baseline['s'][common,t],y[common]),n=int(common.sum())))
  for h in [5,20]:
   if t+h>252:continue
   y=paths[:,756+t+h]-paths[:,756+t];common=np.isfinite(slopeprob[:,t])&np.isfinite(y);slope_ic.append(dict(world=world,panel=index,t=t,h=h,IC=stats.spearman(slopeprob[common,t],y[common]),n=int(common.sum())))
 timing=ROOT/f'runtime/A_LLT_timings/{world}_{index:03d}'
 if done(timing):
  recorded=J(timing/'time.json');comp['test']['measured_LLT_total_seconds']=recorded['seconds'];comp['test']['measured_LLT_seconds']=recorded['fit_seconds']
 else:
  comp['test']['measured_LLT_total_seconds']=time.monotonic()-LLT_clock;finish(timing,{'time.json':dict(seconds=comp['test']['measured_LLT_total_seconds'],fit_seconds=comp['test']['measured_LLT_seconds'])},phase='descriptive_LLT_timing')
 print('Synthetic LLT measured',world,index,comp['test']['measured_LLT_total_seconds'],'seconds',flush=True)
 primary=dict(world=world,panel=index,series=n,truth_rho=None if w==1 else model.phi,truth_q=None if w==1 else w/(1-w),truth_R2_MR=0. if w==1 else pci.ratio(model.phi,w/(1-w)),fit={k:v for k,v in fit.items() if k not in ['scale','n','runs']},selection_fit={k:v for k,v in sf.items() if k not in ['scale','n','runs']},independent_streams=['Oracle_selection','truth'])
 return finish(root,{'primary.json':primary,'secondary.csv.gz':pd.DataFrame(secondary),'ic.csv.gz':pd.DataFrame(ic),'AL.json':results,'oracle.npz':array_bytes(**oracle),'LLT_comparison.json.gz':gzip.compress(encode(comp),compresslevel=6,mtime=0),'LLT_ic.csv.gz':pd.DataFrame(LLT_ic),'LLT_slope_IC.csv.gz':pd.DataFrame(slope_ic)},phase='A',world=world,index=index,series_years=n)
