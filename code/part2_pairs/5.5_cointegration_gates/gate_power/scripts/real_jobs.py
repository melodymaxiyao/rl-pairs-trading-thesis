"""Exact frozen AL replay plus observational trade instrumentation and alternative masks."""
import ast,gzip,json,os,pathlib,resource,time,types
import numpy as np,pandas as pd
from io_utils import *
import gate_tests as gt
GATES=['G_ADF120','G_DFGLS60','G_VR5_60','G_KPSS60'];POL=dict(kind=1,enter=1.25,exit=.5)
def context(kind,year):
 import bootstrap
 c,d,a=bootstrap.core_modules()
 if kind=='RP':ctx=d.rp_context(year,d.panel()['meta']['pairs'][str(year)],verify_stored=True)
 else:ctx=d.rf_context(year)
 return c,d,a,ctx
def sealed_frame(kind,year,gate,name):
 mf=J(CORE/'tier_3/manifest.json')['files'];rel=f'outputs/cells/{kind}_{year}_{gate}_c5_AL/{name}';assert H(CORE/rel)==mf[rel];return pd.read_csv(CORE/rel,float_precision='round_trip')
def finish(root,outputs,**metadata):
 files={}
 for name,value in outputs.items():
  if isinstance(value,pd.DataFrame):raw=gzip.compress(value.to_csv(index=False,float_format='%.17g').encode(),compresslevel=6,mtime=0)
  else:raw=encode(value)
  publish(root/name,raw);files[str((root/name).relative_to(ROOT))]=H(root/name)
 write(root/'complete.json',dict(status='complete',files=files,preregistration_sha256=H(ROOT/'preregistration.json'),**metadata));journal('group_committed',group=str(root.relative_to(ROOT)),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,df=df())
def done(root):
 if not (root/'complete.json').exists():return False
 seal=J(root/'complete.json');assert seal['preregistration_sha256']==H(ROOT/'preregistration.json');assert {str(p.relative_to(ROOT)) for p in root.rglob('*') if p.is_file()}==set(seal['files'])|{str((root/'complete.json').relative_to(ROOT))}
 for rel,digest in seal['files'].items():assert H(ROOT/rel)==digest
 return True
def reproduce(kind,year):
 root=ROOT/f'runtime/reproduction/{kind}_{year}'
 if done(root):return dict(cached=True,root=str(root))
 c,d,a,ctx=context(kind,year);results=[]
 for gate in ['G0','G1']:
  checkpoint();v=a.execution(ctx,POL,gate,0,details=False,cost=5);f=sealed_frame(kind,year,gate,'portfolio_daily.csv.gz');f=f[f.world=='real'].sort_values('date')
  for col in ['gross_bps','K2','PP_FULL','K1','gross_exposure','net_stock_gross','positions','entries','exits','SPI_override_decisions','valid_decisions']:
   np.testing.assert_array_equal(np.array([row[col] for row in v['daily']]),f[col].to_numpy(),err_msg=f'Exact reproduction {kind}/{year}/{gate}/{col}')
  observed=np.array([row['gross_bps']-5*row['K2'] for row in v['daily']]);np.testing.assert_array_equal(observed,f.net_bps.to_numpy())
  s=sealed_frame(kind,year,gate,'per_series_year.csv.gz');s=s[s.world=='real'].set_index('series');rows={row['series']:row for row in v['series']};assert set(rows)==set(s.index)
  for name,row in rows.items():
   for col in ['weight','gross_bps','K2','PP_FULL','K1','entries','exits']:np.testing.assert_array_equal(np.array(row[col]),np.array(s.loc[name,col]))
  previous=np.column_stack([np.zeros(ctx['N'],int),v['positions'][:,:-1]]);base=np.isfinite(d.scores(ctx,'G0',0));base[:,-1]=False;flat=(previous==0)&base;passing=flat if gate=='G0' else flat&ctx['adf'][0]
  results.append(dict(gate=gate,portfolio_net_sum=float(observed.sum()),series=len(rows),flat_opportunities=int(flat.sum()),passing_flat=int(passing.sum()),status='EXACT_ALL_DAILY_AND_SERIES_FIELDS_PASS'))
 finish(root,{'reproduction.json':results},phase='reproduction',kind=kind,year=year);return dict(root=str(root),status='EXACT_PASS')
def windows(ctx,wi,ti,indices,W,d):
 data=d.panel();t=ctx['ts'][ti];assert t>=W-1;ids=np.asarray(indices,int);history=data['returns'][t-W+1:t+1];dates=data['dates'][t-W+1:t+1]
 if ctx['kind']=='RP':
  ia=ctx['ia'][ids];ib=ctx['ib'][ids];ok=(ia>=0)&(ib>=0);e=history[:,np.maximum(ia,0)].T-ctx['alpha'][ids,ti,None]-ctx['beta'][ids,ti,0,None]*history[:,np.maximum(ib,0)].T;e[~ok]=np.nan
 else:
  valid=np.flatnonzero(np.isfinite(ctx['alpha'][:,ti]));e=np.full((len(ids),W),np.nan)
  if len(valid):
   factors=history[:,valid]@ctx['W'][ti,valid];residual=(history[:,valid]-np.column_stack([np.ones(W),factors])@np.vstack([ctx['alpha'][valid,ti],ctx['beta'][valid,ti].T])).T
   positions={int(v):i for i,v in enumerate(valid)}
   for j,i in enumerate(ids):
    if int(i) in positions:e[j]=residual[positions[int(i)]]
 if wi:
  signs=np.stack([d.M.sign_series(d.source().prereg['null_seed'],f'RP|{ctx["year"]}:{ctx["ids"][i]}' if ctx['kind']=='RP' else 'RF|'+ctx['ids'][i],wi-1,dates) for i in ids]);e*=signs
 return e
def instrument(a,ctx,d,wi):
 # Only observation is inserted; every original floating-point operation stays intact.
 original=(CORE/'scripts/accounting.py').read_text();reg=J(CORE/'code_registration.json');assert H(CORE/'scripts/accounting.py')==reg['files']['scripts/accounting.py']
 node=next(n for n in ast.parse(original).body if isinstance(n,ast.FunctionDef) and n.name=='execution');text=ast.get_source_segment(original,node);assert text.count('old=new.copy()')==1
 text=text.replace('old=new.copy()',"observer(ti,old,new,gross,changed,allocation,w,pending_forced);old=new.copy()")
 active={};trades=[]
 def observer(ti,old,new,gross,changed,allocation,w,pending_forced):
  for i,row in active.items():row['portfolio_net_bps']+=float(gross[i])
  entries=[int(i) for i in changed if old[i]==0 and new[i]!=0];features={}
  if entries:
   e=windows(ctx,wi,ti,entries,60,d);stat=gt.adf(e);p=gt.adf_pvalue(stat);fits=d.M.ou_fit(e)
   np.testing.assert_array_equal(gt.rejection('ADF',stat,60),ctx['adf'][wi,entries,ti])
   for j,i in enumerate(entries):features[i]=dict(adf_stat=float(stat[j]),adf_p=float(p[j]),gate_pass=bool(ctx['adf'][wi,i,ti]),abs_s=float(abs(ctx['score'][wi,i,ti])),half_life=float(ctx['half'][wi,i,ti]),sigma_eq=float(fits['sigma_eq'][j]))
  for i,fee in zip(changed,allocation):
   i=int(i)
   if old[i]:
    row=active.pop(i);row['portfolio_net_bps']-=5*float(fee);row.update(net_bps=row['portfolio_net_bps']/row['capital_weight'],holding=ti-row['entry_index'],exit_date=str(ctx['dates'][ti]),exit_type='terminal' if ti==ctx['T']-1 else 'forced_invalid' if pending_forced[i] or not np.isfinite(ctx['score'][wi,i,ti]) else 'rule');trades.append(row)
   if new[i]:active[i]=dict(definition=ctx['kind'],test_year=ctx['year'],world='real' if wi==0 else 'N2',replicate=0 if wi==0 else wi-1,series=ctx['ids'][i],entry_index=ti,entry_date=str(ctx['dates'][ti]),capital_weight=float(w[i]),portfolio_net_bps=-5*float(fee),**features[i])
 namespace=dict(a.__dict__);namespace['observer']=observer;exec(compile(text,'<frozen execution + observational hook>','exec'),namespace)
 value=namespace['execution'](ctx,POL,'G0',wi,details=False,cost=5);assert not active
 reference=sealed_frame(ctx['kind'],ctx['year'],'G0','portfolio_daily.csv.gz');reference=reference[(reference.world==('real' if wi==0 else 'N2'))&(reference.replicate==(0 if wi==0 else wi-1))].sort_values('date')
 np.testing.assert_array_equal(np.array([row['gross_bps']-5*row['K2'] for row in value['daily']]),reference.net_bps.to_numpy())
 np.testing.assert_allclose(sum(row['portfolio_net_bps'] for row in trades),reference.net_bps.sum(),rtol=1e-10,atol=1e-10)
 return trades
def anatomy(kind,year):
 root=ROOT/f'runtime/anatomy/{kind}_{year}'
 if done(root):return dict(cached=True,root=str(root))
 c,d,a,ctx=context(kind,year);rows=[]
 for wi in range(3):checkpoint();rows+=instrument(a,ctx,d,wi)
 columns=['definition','test_year','world','replicate','series','entry_index','entry_date','capital_weight','portfolio_net_bps','adf_stat','adf_p','gate_pass','abs_s','half_life','sigma_eq','net_bps','holding','exit_date','exit_type']
 finish(root,{'trades.csv.gz':pd.DataFrame(rows,columns=columns)},phase='A',kind=kind,year=year);return dict(root=str(root),trades=len(rows))
def masks(ctx,d):
 shapes=(3,ctx['N'],ctx['T']);result={g:np.zeros(shapes,bool) for g in GATES}
 for ti in range(ctx['T']):
  if ti%20==0:checkpoint();print('Gate features',ctx['kind'],ctx['year'],ti,'/',ctx['T'],flush=True)
  for wi in range(3):
   for start in range(0,ctx['N'],128):
    ids=np.arange(start,min(start+128,ctx['N']));e=windows(ctx,wi,ti,ids,60,d);e120=windows(ctx,wi,ti,ids,120,d)
    result['G_ADF120'][wi,ids,ti]=gt.gate('ADF',gt.adf(e120),120);result['G_DFGLS60'][wi,ids,ti]=gt.gate('DFGLS',gt.dfgls(e),60);result['G_KPSS60'][wi,ids,ti]=gt.gate('KPSS',gt.kpss(e),60);result['G_VR5_60'][wi,ids,ti]=gt.gate('VR5',gt.vr(e,5)[1],60)
 return result
def alternatives(kind,year):
 root=ROOT/f'runtime/alternatives/{kind}_{year}'
 if done(root):return dict(cached=True,root=str(root))
 c,d,a,ctx=context(kind,year);mask=masks(ctx,d);series=[];daily=[];coverage=[];missing=[]
 for gate in GATES:
  alternate=dict(ctx,adf=mask[gate])
  for wi in range(3):
   checkpoint();value=a.execution(alternate,POL,'G1',wi,details=False,cost=5);meta=dict(definition=kind,test_year=year,gate=gate,learner='AL',cost=5,world='real' if wi==0 else 'N2',replicate=0 if wi==0 else wi-1)
   series += [dict(row,**meta,net_bps=row['gross_bps']-5*row['K2']) for row in value['series']];daily += [dict(row,**meta,net_bps=row['gross_bps']-5*row['K2']) for row in value['daily']];missing += [dict(row,**meta) for row in value['missing_price_log']]
   # Flat opportunities are defined on the gate's own prior position and base-valid score.
   positions=value['positions'];previous=np.column_stack([np.zeros(ctx['N'],int),positions[:,:-1]]);base=np.isfinite(d.scores(ctx,'G0',wi));base[:,-1]=False;valid_flat=(previous==0)&base;coverage.append(dict(**meta,flat_opportunities=int(valid_flat.sum()),passing_flat=int((valid_flat&mask[gate][wi]).sum())))
 from missing_prices import LOG_COLUMNS
 finish(root,{'series.csv.gz':pd.DataFrame(series),'daily.csv.gz':pd.DataFrame(daily),'coverage.json':coverage,'missing_price_log.csv.gz':pd.DataFrame(missing).reindex(columns=LOG_COLUMNS)},phase='C',kind=kind,year=year,new_real_cells_family=8);return dict(root=str(root),rows=len(series))
