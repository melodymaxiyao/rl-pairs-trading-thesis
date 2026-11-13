"""Read-only original numerical functions with task-local IO; no scientific source edits."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import common as U
from pathlib import Path
import ast,sys,types,io,json,gzip,pickle,hashlib,stat,shutil,time,ctypes,datetime,collections
import numpy as np,pandas as pd
P=U.P;CORE=P.parent/'coint_residual_qlearning_v1';PART=P.parent/'partial_coint_kalman_v1';CLEAN=PART/'outputs/N3_intercept_null/CLEAN'
U.guard()
LEDGER=P/'audit/trading_inputs.json';H=U.js(P/'input_hashes_before.json');AUTH={};CACHE={};MODULES={};started=time.monotonic()
def raw(p):
 p=Path(p);b=U.raw(p);v=U.INPUTS.get(str(p),dict(sha256=hashlib.sha256(b).hexdigest(),size=len(b)))
 if str(p)in H:assert H[str(p)]==v,('Changed input',str(p))
 H[str(p)]=v;return b
def sha(p):return hashlib.sha256(raw(p)).hexdigest()
def read(p):return json.loads(raw(p))
def checkpoint():
 assert shutil.disk_usage(P).free>=2*1024**3,'Free disk below2GiB; stop with retained checkpoints'
 assert sha(P/'preregistration.md')==U.js(P/'audit/registration.json')['preregistration_sha256']
def track_code(p):
 p=Path(p);b=raw(p);return b.decode()
def forbidden(*a,**k):raise AssertionError('New Q-learning, MLE or source write forbidden')
def module(name,p,inject=None,remove_imports=()):
 s=track_code(p);tree=ast.parse(s);tree.body=[n for n in tree.body if not isinstance(n,ast.ImportFrom)or n.module not in remove_imports]
 m=types.ModuleType(name);m.__file__=str(p);m.__dict__.update(inject or {});sys.modules[name]=m;exec(compile(tree,str(p),'exec'),m.__dict__);MODULES[name]=m;return m
def functions(p,names,ns):
 tree=ast.parse(track_code(p));nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name in names];assert set(n.name for n in nodes)==set(names);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns);return ns
# Authenticate registration hashes for every original source function loaded.
reg=read(CORE/'code_registration.json')['files'];seal=read(CORE/'tier_3/complete.json');mf=read(CORE/'tier_3/manifest.json')['files'];assert sha(CORE/'code_registration.json')==seal['code_registration_sha256'];assert sha(CORE/'tier_3/manifest.json')==seal['manifest_sha256']
def cfile(rel):
 p=CORE/rel;expected=reg.get(rel,mf.get(rel));assert expected and sha(p)==expected,('Original registration mismatch',rel);return p
base=dict(np=np,pd=pd,Path=Path,RUN=CORE,INPUT=Path('/Users/maxiyao/Desktop/FYP'),SOURCE=P.parent/'residual_ou_real_v1',sha=sha,read=read,hashlib=hashlib,json=json,gzip=gzip,pickle=pickle,os=os,sys=sys,stat=stat,time=time,checkpoint=checkpoint,Counter=collections.Counter)
base['meta']=lambda p:dict(size=Path(p).lstat().st_size,mode=stat.S_IMODE(Path(p).lstat().st_mode),mtime_ns=Path(p).lstat().st_mtime_ns,ino=Path(p).lstat().st_ino,flags=Path(p).lstat().st_flags)
assert sha(CORE/'preregistration.json')==seal['preregistration_sha256']
base['spec']=lambda:read(CORE/'preregistration.json')
base['_BANK']={};functions(cfile('scripts/common.py'),['Sealed','seal','source','vendor','load_module'],base)
# Original loader hashes vendored files; guard forbids old-run writes.
base.update(publish=forbidden,write=forbidden,replace=forbidden,count=forbidden,event=forbidden)
k=module('_pci_core_kernel',cfile('scripts/kernel.py'),base,('common',));k.fit=forbidden;k.build=forbidden
# Explicitly pin native execution binary and import proof.
proof=read(CORE/'audit/kernel_import_proof.json');assert sha(CORE/'runtime/frozen_q.so')==proof['binary_sha256']
d=module('_pci_core_data',cfile('scripts/data.py'),base,('common',))
missing=module('_pci_core_missing',cfile('scripts/missing_prices.py'),dict(base,source=d.source),('common',))
a=module('_pci_core_accounting',cfile('scripts/accounting.py'),dict(base,entry_vector=d.entry_vector,panel=d.panel,scores=d.scores,M=d.M,positions=k.positions,marked_prices=missing.marked_prices,causal_target=missing.causal_target,LOG_COLUMNS=missing.LOG_COLUMNS),('common','data','kernel','missing_prices'))
sys.modules['data']=d;sys.modules['kernel']=k;sys.modules['missing_prices']=missing
# Loader uses only original pure numerical modules and verified cached inputs.
res=types.SimpleNamespace();ns=dict(np=np,sliding_window_view=np.lib.stride_tricks.sliding_window_view);functions(PART/'scripts/residuals.py',['pair_out_of_sample','factor_out_of_sample'],ns);res.pair_out_of_sample=ns['pair_out_of_sample'];res.factor_out_of_sample=ns['factor_out_of_sample']
pci=module('_pci_filter',PART/'scripts/pci.py');pci.fit=forbidden;sha(PART/'runtime/kalman.so')
# Capture the exact original input construction/parameter replay functions; no module entry is invoked.
refns=dict(np=np,P=PART,OUT=PART/'outputs/A1_sanity_v2')
def done(root):
 cp=Path(root)/'complete.json'
 if not cp.exists():return False
 r=read(cp);assert r['status']=='complete'
 for rel,h in r['files'].items():assert sha(PART/rel)==h
 return True
refns['done']=done;functions(PART/'audit/A1_sanity_v2/audit.py',['rp_reference_subset','check_parameter'],refns)
bootstrap=types.SimpleNamespace(core_modules=lambda:(types.SimpleNamespace(source=d.source),d,a))
ns=dict(np=np,bootstrap=bootstrap,P=PART,u=types.SimpleNamespace(J=read,checkpoint=checkpoint),residuals=res,done=done)
functions(PART/'audit/N3_intercept_null/run.py',['levels','sign_rows'],ns);n3=types.SimpleNamespace(levels=ns['levels'],sign_rows=ns['sign_rows'],item3_module=lambda:types.SimpleNamespace(rp_reference_subset=refns['rp_reference_subset'],check_parameter=refns['check_parameter']))
ns['n3']=n3;functions(PART/'audit/N3_intercept_null/CLEAN_addendum/run.py',['clean_inputs'],ns);clean_inputs=ns['clean_inputs']
# All original functions are reused with unchanged bodies. Preserve the original Band grid selection body.
bandns=dict(np=np,cube_values=k.cube_values);functions(cfile('scripts/learning.py'),['band'],bandns);band=bandns['band']
def fit_record(family,year,wi):
 root=CLEAN/f'pooled/{family}_{year}_{wi}';assert done(root);return read(root/'fit.json'),np.asarray(read(root/'eligible.json'),int)
def filtered_levels(levels,fit,eligible):
 assert fit['converged'];x=np.asarray(levels);out=np.full_like(x,np.nan);_,states=pci.filter_values(x[eligible],fit['rho'],fit['q'],states=True);out[eligible]=states[:,:,1]/np.sqrt(np.asarray(fit['scale'])[:,None]/(1-fit['rho']**2));out[~np.isfinite(x)]=np.nan;return out

def signal_history(family,year):
 checkpoint();data=clean_inputs(family,year);result={};checks=[]
 for wi in [0,1]:
  fit,eligible=fit_record(family,year,wi);assert np.array_equal(data['eligible'],eligible);score=filtered_levels(data['levels'][wi],fit,eligible);root=CLEAN/f'signals/{family}_{year}_{wi}';assert done(root)
  with np.load(io.BytesIO(raw(root/'signals.npz')),allow_pickle=False)as z:saved=z['K_clean']
  regenerated=score[:,data['form']:];np.testing.assert_allclose(regenerated,saved,rtol=0,atol=1e-10,equal_nan=True,err_msg=f'CLEAN test signal reproduction {family}/{year}/world{wi}')
  checks.append(dict(family=family,year=year,world=wi,shape=list(saved.shape),maximum_absolute_error=float(np.nanmax(abs(regenerated-saved))),passes=True))
  result[wi]=score
 return data,result,checks

def original_contexts(family,year,training=True):
 if family=='RP':
  pool=d.pool(year);mapping={r['pair']:r for r in pool['P_RAND']};assert set(mapping)==set(r['pair']for r in d.panel()['meta']['pairs'][str(year)]);trainrows=[mapping[n]for n in pool['training_subset']];testrows=d.panel()['meta']['pairs'][str(year)];test=d.rp_context(year,testrows,verify_stored=True);contexts=[d.rp_context(y,trainrows)for y in range(year-3,year)]if training else[]
 else:test=d.rf_context(year);contexts=[d.rf_context(y)for y in range(year-3,year)]if training else[]
 return test,contexts

def original_policy(family,year,learner):
 p=cfile(f'outputs/cells/{family}_{year}_G0_c5_{learner}/complete.json');return read(p)['policy']

def clean_training(data,score,contexts):
 # No use of test-year inputs; contexts may contain different RF membership rows but panel IDs are fixed.
 lookup={str(name):i for i,name in enumerate(data['ids'])};idx={int(t):j for j,t in enumerate(data['indices'])};T=max(c['T']for c in contexts);zs=[];kept=[]
 for ctx in contexts:
  ii=np.array([lookup[n]for n in ctx['ids']]);tt=np.array([idx[int(t)]for t in ctx['ts']]);assert np.all(tt<data['form']);s=score[np.ix_(ii,tt)];zs.append(np.pad(np.where(np.isfinite(s),s,np.inf),((0,0),(T-ctx['T'],0)),constant_values=np.inf));kept.append(np.ones(ctx['N'],bool)if ctx['kind']=='RP'else np.isfinite(ctx['score'][0]).any(axis=1))
 return np.concatenate(zs)[np.concatenate(kept)],np.concatenate(kept)

def execute(ctx,policy,signal=None,wi=0):
 if signal is None:return a.execution(ctx,policy,'G0',wi,details=False,cost=5)
 original=a.scores
 def current(c,gate,w):
  if c is ctx and w==wi:return np.where(np.isfinite(signal),signal,np.inf)
  return original(c,gate,w)
 a.scores=current
 try:return a.execution(ctx,policy,'G0',wi,details=False,cost=5)
 finally:a.scores=original

def baseline_check(ctx,year,family):
 proofs=[];baseline={}
 for learner in ['AL','Band']:
  pol=original_policy(family,year,learner);v=execute(ctx,pol);rel=f'outputs/cells/{family}_{year}_G0_c5_{learner}/portfolio_daily.csv.gz';p=cfile(rel);f=pd.read_csv(io.BytesIO(raw(p)),compression='gzip',float_precision='round_trip',keep_default_na=False);f=f[(f.world=='real')&(f.replicate==0)].sort_values('date');assert len(f)==len(v['daily'])
  for col in ['gross_bps','K2','PP_FULL','K1','gross_exposure','positions','entries','exits']:
   observed=np.array([r[col]for r in v['daily']]);expected=f[col].to_numpy();np.testing.assert_array_equal(observed,expected,err_msg=f'Original engine exact reproduction {family}/{year}/{learner}/{col}')
  net=np.array([r['gross_bps']-5*r['K2']for r in v['daily']]);np.testing.assert_array_equal(net,f.net_bps.to_numpy(),err_msg=f'net arithmetic {family}/{year}/{learner}')
  baseline[learner]=dict(gross_bps=float(f.gross_bps.sum()),net_bps=float(f.net_bps.sum()),turnover=float(f.K2.sum()),source=str(p),source_key=f'world=real;replicate=0;test_year={year};learner={learner};cost=5;G0')
  proofs.append(dict(family=family,year=year,learner=learner,exact_daily_fields=True,source_sha256=sha(p)))
 return baseline,proofs

def persist_inputs():
 for p,v in U.INPUTS.items():
  if p in H:assert H[p]==v
  H[p]=v
 U.dump(LEDGER,H)

def summarize_run(ctx,v,unit,year,family,baseline,valid,original_valid,world='real'):
 # Preserve original sum-of-daily order and plain K2 arithmetic in saved daily table.
 gross=np.array([r['gross_bps']for r in v['daily']],float);turn=np.array([r['K2']for r in v['daily']],float);net=gross-5*turn
 return dict(family=family,year=year,unit=unit,world=world,cost_bps=5,gross_bps=float(gross.sum()),net_bps=float(net.sum()),annual_K2=float(turn.sum()),baseline_learner='Band'if unit=='Band_K_main'else'AL',baseline_net_bps=baseline['net_bps'],baseline_gross_bps=baseline['gross_bps'],baseline_annual_K2=baseline['turnover'],net_difference_bps=float(net.sum()-baseline['net_bps']),gross_difference_bps=float(gross.sum()-baseline['gross_bps']),valid_series_days=int(valid.sum()),original_AL_valid_series_days=int(original_valid.sum()),common_valid_series_days=int((valid&original_valid).sum()),original_only_valid_series_days=int((original_valid&~valid).sum()),baseline_source=baseline['source'],baseline_row_key=baseline['source_key'])

import atexit
atexit.register(persist_inputs)
