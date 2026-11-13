"""Authenticated unchanged TAV bootstrap/DSR/CSCV numerical functions."""
from common import *
import ast,types

def extracted(filename,names,namespace):
 p=RUN/'vendor/TAV'/filename;copies=read(RUN/'audit/TAV_code_copies.json');rec=next(r for r in copies.values() if r['copy']==str(p.relative_to(RUN)));assert sha(p)==rec['sha256'];tree=ast.parse(p.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names];assert {n.name for n in nodes}==set(names);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),namespace);return namespace

def bootstrap(frame):
 import pandas as pd
 ns=dict(np=np,pd=pd,count=count);f=extracted('inference.py',['blocks','estimate'],ns);return f['estimate'](frame,draws=5000)

def dsr_pbo(daily,tier):
 import pandas as pd
 from scipy.stats import norm,rankdata
 from itertools import combinations
 ns=extracted('s2inference.py',['dsr','pbo'],dict(np=np,norm=norm,rankdata=rankdata,combinations=combinations));f=daily[(daily.world=='real')];rows=[]
 for scope in ['GLOBAL','RP','RF']:
  a=f if scope=='GLOBAL' else f[f.definition==scope];keys=['definition','gate','cost','learner'];series=[];annual=[];labels=[]
  for key,g in a.groupby(keys,sort=True):
   label='|'.join(map(str,key));labels.append(label);series.append(g.set_index('date').net_bps.rename(label));annual.append(g.groupby('test_year').net_bps.sum().reindex(range(2004,2026)).to_numpy(float))
  matrix=pd.concat(series,axis=1).dropna();sr=matrix.mean()/matrix.std(ddof=1);valid=sr[np.isfinite(sr)]
  if len(valid)>=2:
   best=valid.idxmax();x=matrix[best].to_numpy();v=x-x.mean();m2=np.mean(v*v);out=ns['dsr'](float(sr[best]),len(x),float(np.mean(v**3)/m2**1.5),float(np.mean(v**4)/m2**2),float(valid.var(ddof=1)),572);rows.append(dict(kind='DSR',scope=scope,best_for_accounting_only=best,selects_no_strategy=True,**out))
  else:rows.append(dict(kind='DSR',scope=scope,status='fewer than two finite Sharpe cells',N=572))
  required=72 if scope=='GLOBAL' else 36
  if tier==3 and len(annual)==required and np.isfinite(annual).all():
   out,splits=ns['pbo'](np.asarray(annual));assert out['n_splits']==705432;rows.append(dict(kind='PBO',scope=scope,PBO=out['PBO'],n_splits=out['n_splits'],available_cells=len(annual),status='all planned cells'));del splits,out
  else:rows.append(dict(kind='PBO',scope=scope,status='NA until all planned cells are sealed; no imputation',requested_cells=required,available_cells=len(annual)))
 return rows

def market(y,x):
 p=vendor('rf_cost_accounting_v1','scripts/sealed_functions.py');ns=dict(np=np,COUNTS=COUNTS);tree=ast.parse(p.read_text());n=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='ols_hac');exec(compile(ast.Module(body=[n],type_ignores=[]),str(p),'exec'),ns);return ns['ols_hac'](y,x,5,252)
