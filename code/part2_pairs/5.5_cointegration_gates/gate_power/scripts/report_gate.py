"""Streaming calendar-year inference and report; no policy/learner fitting."""
import ast,collections,gzip,io,json,math,pathlib
import numpy as np,pandas as pd
from scipy.stats import t as student_t
from io_utils import *
from real_jobs import GATES,done
YEARS=list(range(2004,2026));W=None
def bootstrap_weights():
 global W
 if W is None:
  p=CORE/'vendor/TAV/inference.py';reg=J(CORE/'code_registration.json');assert H(p)==reg['files']['vendor/TAV/inference.py'];tree=ast.parse(p.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='blocks'];ns=dict(np=np);exec(compile(ast.Module(body=nodes,type_ignores=[]),'<unchanged year blocks>','exec'),ns);names,W=ns['blocks'](YEARS,5000);assert names==YEARS
 return W
def boot(sums,counts):
 sums=np.asarray(sums,float);counts=np.asarray(counts,float);total=counts.sum();point=sums.sum()/total if total>0 else np.nan;n=bootstrap_weights()@sums;d=bootstrap_weights()@counts;draws=np.divide(n,d,out=np.full(len(n),np.nan),where=d>0);finite=np.isfinite(draws)
 return dict(estimate=point,ci95_low=float(np.quantile(draws[finite],.025)) if finite.any() else np.nan,ci95_high=float(np.quantile(draws[finite],.975)) if finite.any() else np.nan,p_one_sided=float((1+np.sum(~finite)+np.sum(draws[finite]-point>=point))/5001) if np.isfinite(point) else 1.,undefined_draws=int((~finite).sum()))
def emit(name,frame):
 if not isinstance(frame,pd.DataFrame):frame=pd.DataFrame(frame)
 publish(ROOT/'outputs/tables'/name,frame.to_csv(index=False,float_format='%.17g'))
def anatomy_tables():
 # Only a one-dimensional p-value/rank vector is kept across groups; trade frames
 # and regression cross-products are consumed one calendar year at a time.
 summaries=[];regressions=[];p1=None;p2=None;decile_plot=[]
 for kind in ['RP','RF']:
  paths=[ROOT/f'runtime/anatomy/{kind}_{y}' for y in YEARS];assert all(done(p) for p in paths)
  def load(path):
   f=pd.read_csv(path/'trades.csv.gz',float_precision='round_trip',dtype={'adf_p':float,'adf_stat':float,'abs_s':float,'half_life':float,'net_bps':float,'portfolio_net_bps':float});return f.sort_values(['series','entry_date','world','replicate'],kind='stable').reset_index(drop=True)
  pvalues=[];sizes=[]
  for path in paths:
   f=load(path);pvalues.append(f.adf_p.to_numpy());sizes.append(len(f))
  pvalues=np.concatenate(pvalues);finite=np.flatnonzero(np.isfinite(pvalues));order=finite[np.argsort(pvalues[finite],kind='stable')];deciles=np.zeros(len(pvalues),np.int8);deciles[order]=np.minimum(9,np.arange(len(order))*10//max(1,len(order)))+1
  aggregates=collections.defaultdict(lambda:np.zeros((22,6)));reg={world:(np.zeros((22,6,6)),np.zeros((22,6)),np.zeros(22),np.zeros(22,int)) for world in [('real',0),('N2',0),('N2',1)]};offset=0
  for yi,path in enumerate(paths):
   checkpoint();f=load(path);f['p_decile']=deciles[offset:offset+len(f)];offset+=len(f)
   for (world,rep),part in f.groupby(['world','replicate']):
    for category,column in [('gate','gate_pass'),('decile','p_decile')]:
     for value,g in part.groupby(column):aggregates[(world,int(rep),category,str(value))][yi]=[len(g),g.net_bps.sum(),(g.net_bps>0).sum(),g.holding.sum(),g.portfolio_net_bps.sum(),(g.net_bps*g.net_bps).sum()]
    use=part[np.isfinite(part[['adf_stat','abs_s','half_life','net_bps']]).all(axis=1)];bins=np.searchsorted([5,10,20],use.half_life.to_numpy(),side='right');X=np.column_stack([np.ones(len(use)),use.adf_stat,use.abs_s,*[(bins==k).astype(float) for k in [1,2,3]]]);y=use.net_bps.to_numpy();xx,xy,yy,nn=reg[(world,int(rep))];xx[yi]=X.T@X;xy[yi]=X.T@y;yy[yi]=y@y;nn[yi]=len(y)
   del f
  # Gate-fail total contribution uses all trades in its denominator, matched by year.
  for key,array in sorted(aggregates.items(),key=lambda x:str(x[0])):
   world,rep,category,value=key;total=aggregates.get((world,rep,'gate','False'),np.zeros((22,6)))+aggregates.get((world,rep,'gate','True'),np.zeros((22,6)))
   stat=boot(array[:,1],array[:,0]);hold=boot(array[:,3],array[:,0]);hit=boot(array[:,2],array[:,0]);contribution=boot(array[:,4],np.ones(22));den=bootstrap_weights()@total[:,4];num=bootstrap_weights()@array[:,4];ratio=np.divide(num,den,out=np.full(len(num),np.nan),where=abs(den)>1e-15);finite=np.isfinite(ratio)
   share=array[:,4].sum()/total[:,4].sum() if abs(total[:,4].sum())>1e-15 else np.nan
   row=dict(definition=kind,world=world,replicate=rep,category=category,bin=value,trade_count=int(array[:,0].sum()),**stat,hit_rate=hit['estimate'],mean_holding=hold['estimate'],total_net_contribution_bps_year=contribution['estimate'],share_AL_net=share,share_ci95_low=float(np.quantile(ratio[finite],.025)) if finite.any() else np.nan,share_ci95_high=float(np.quantile(ratio[finite],.975)) if finite.any() else np.nan);summaries.append(row)
   if category=='decile':decile_plot.append(row)
   if kind=='RP' and world=='real' and category=='gate' and value=='False':p1=dict(id='P1',alternative='positive',**stat)
  for (world,rep),(xx,xy,yy,nn) in reg.items():
   n=int(nn.sum());g=int((nn>0).sum());bread=np.linalg.pinv(xx.sum(axis=0));coef=bread@xy.sum(axis=0);scores=xy-np.einsum('gij,j->gi',xx,coef);cov=bread@(scores.T@scores)@bread*g/(g-1)*(n-1)/(n-6);se=np.sqrt(max(0,cov[1,1]));statistic=coef[1]/se if se>0 else np.nan;p=float(student_t.cdf(statistic,g-1)) if np.isfinite(statistic) else 1.;critical=student_t.ppf(.975,g-1);row=dict(definition=kind,world=world,replicate=rep,n=n,clusters=g,adf_slope=float(coef[1]),cluster_se=se,ci95_low=float(coef[1]-critical*se),ci95_high=float(coef[1]+critical*se),p_negative=p);regressions.append(row)
   if kind=='RP' and world=='real':p2=dict(id='P2',alternative='negative',estimate=float(coef[1]),ci95_low=row['ci95_low'],ci95_high=row['ci95_high'],p_one_sided=p)
 emit('trades_by_gate.csv',summaries);emit('trade_regressions.csv',regressions);assert p1 is not None and p2 is not None
 return p1,p2,summaries
def core_portfolios():
 mf=J(CORE/'tier_3/manifest.json')['files'];out={}
 for kind in ['RP','RF']:
  for gate in ['G0','G1']:
   for y in YEARS:
    rel=f'outputs/cells/{kind}_{y}_{gate}_c5_AL/per_series_year.csv.gz';assert H(CORE/rel)==mf[rel];f=pd.read_csv(CORE/rel,float_precision='round_trip')
    for (world,rep),g in f.groupby(['world','replicate']):out[(kind,gate,world,int(rep),y)]=float(np.sum(g.weight*g.net_bps))
 return out
def real_tables():
 base=core_portfolios();values={};coverages=collections.defaultdict(lambda:np.zeros(2));turnovers=collections.defaultdict(lambda:np.zeros(2))
 for kind in ['RP','RF']:
  for y in YEARS:
   path=ROOT/f'runtime/alternatives/{kind}_{y}';assert done(path);f=pd.read_csv(path/'series.csv.gz',float_precision='round_trip');daily=pd.read_csv(path/'daily.csv.gz',float_precision='round_trip')
   for (gate,world,rep),g in f.groupby(['gate','world','replicate']):values[(kind,gate,world,int(rep),y)]=float(np.sum(g.weight*g.net_bps))
   for row in J(path/'coverage.json'):coverages[(kind,row['gate'],row['world'],row['replicate'])]+=np.array([row['passing_flat'],row['flat_opportunities']])
   for (gate,world,rep),g in daily.groupby(['gate','world','replicate']):turnovers[(kind,gate,world,int(rep))]+=np.array([g.K2.sum(),len(g)])
   for row in J(ROOT/f'runtime/reproduction/{kind}_{y}/reproduction.json'):coverages[(kind,row['gate'],'real',0)]+=np.array([row['passing_flat'],row['flat_opportunities']])
   mf=J(CORE/'tier_3/manifest.json')['files']
   for gate in ['G0','G1']:
    rel=f'outputs/cells/{kind}_{y}_{gate}_c5_AL/portfolio_daily.csv.gz';assert H(CORE/rel)==mf[rel];benchmark=pd.read_csv(CORE/rel,float_precision='round_trip')
    for (world,rep),g in benchmark.groupby(['world','replicate']):turnovers[(kind,gate,world,int(rep))]+=np.array([g.K2.sum(),len(g)])
 all_values=dict(base,**{})
 all_values.update(values);rows=[];contrasts=[]
 for kind in ['RP','RF']:
  for gate in ['G0','G1',*GATES]:
   for world,rep in [('real',0),('N2',0),('N2',1)]:
    v=np.array([all_values[(kind,gate,world,rep,y)] for y in YEARS]);stat=boot(v,np.ones(22));coverage=coverages.get((kind,gate,world,rep));turn=turnovers.get((kind,gate,world,rep));rows.append(dict(definition=kind,gate=gate,world=world,replicate=rep,**stat,annual_K2_turnover=252*turn[0]/turn[1] if turn is not None else np.nan,coverage=coverage[0]/coverage[1] if coverage is not None and coverage[1] else np.nan))
   for comparator in (['G0','G1','N2mean'] if gate in GATES else ['G0','N2mean']):
    v=np.array([all_values[(kind,gate,'real',0,y)]-(.5*(all_values[(kind,gate,'N2',0,y)]+all_values[(kind,gate,'N2',1,y)]) if comparator=='N2mean' else base[(kind,comparator,'real',0,y)]) for y in YEARS]);result=boot(v,np.ones(22));name='P3' if kind=='RP' and gate=='G_VR5_60' and comparator=='G1' else 'P4' if kind=='RP' and gate=='G_DFGLS60' and comparator=='G1' else 'descriptive';contrasts.append(dict(id=name,definition=kind,gate=gate,comparator=comparator,alternative='positive',**result))
 emit('cells.csv',rows);return rows,contrasts
def coverage_decomposition():
 mf=J(CORE/'tier_3/manifest.json')['files'];rel='tier_3/performance.csv.gz';assert H(CORE/rel)==mf[rel];f=pd.read_csv(CORE/rel,float_precision='round_trip');f=f[(f.world=='real')&(f.era=='pooled')&(f.accounting=='K2')];a=f[f.gate=='G0'];b=f[f.gate=='G1'];keys=['definition','learner','cost'];m=a.merge(b,on=keys,validate='one_to_one',suffixes=('_0','_1'));assert len(m)==36;m['retention']=m.annual_K2_turnover_1/m.annual_K2_turnover_0;m['gate_effect']=m.net_annual_bps_1-m.net_annual_bps_0;valid=np.isfinite(m[['retention','gate_effect','net_annual_bps_0']]).all(axis=1);v=m[valid];y=v.gate_effect.to_numpy();results=[]
 for name,X in [('coverage_and_profitability',np.column_stack([np.ones(len(v)),v.net_annual_bps_0,v.retention])),('lost_coverage_times_profitability',np.column_stack([np.ones(len(v)),-(1-v.retention)*v.net_annual_bps_0]))]:
  coef=np.linalg.lstsq(X,y,rcond=None)[0];res=y-X@coef;results.append(dict(model=name,n=len(v),excluded=int((~valid).sum()),coefficients=coef.tolist(),R2=1-float(res@res)/float(np.sum((y-y.mean())**2)),interpretation='descriptive fit across36 sealed policy/residual/cost cells; not causal identification'))
 emit('coverage_decomposition.csv',m);write(ROOT/'outputs/tables/coverage_decomposition_fits.json',results);return results
def synthetic_tables():
 packs=collections.defaultdict(list)
 for p in sorted((ROOT/'runtime/synthetic').glob('*')):
  assert done(p)
  for row in J(p/'sums.json'):packs[(row['world'],row['window'],row['test'])].append(row)
 powers=[];values=[]
 def mean(rows,key):
  n=sum(r[key]['n'] for r in rows);s=sum(r[key]['sum'] for r in rows);s2=sum(r[key]['sumsq'] for r in rows);return s/n,np.sqrt(max(0,(s2-s*s/n)/(n-1))/n)
 for key,rows in sorted(packs.items()):
  assert len(rows)==200 and sum(r['rejection']['n']+r['rejection']['undefined'] for r in rows)==20000
  world,W0,test=key;rate,se=mean(rows,'rejection');net,nse=mean(rows,'net');base,bse=mean(rows,'ungated_net');removed,rse=mean(rows,'net_removed');turn,tse=mean(rows,'turnover');shared=dict(world=world,window=W0,test=test,null_status=rows[0]['null_status'],rate_interpretation=rows[0]['rate_interpretation'],series_years=20000,c=rows[0]['c']);powers.append(dict(**shared,rejection_rate=rate,MC_SE=se,actual_rejected_windows=sum(r['reject_count'] for r in rows),actual_testable_windows=sum(r['observations'] for r in rows)));values.append(dict(**shared,gated_AL_net=net,gated_MC_SE=nse,ungated_AL_net=base,ungated_MC_SE=bse,AL_net_removed=removed,removed_MC_SE=rse,share_net_removed=removed/base if abs(base)>1e-15 else np.nan,annual_turnover=turn))
 theory=collections.defaultdict(lambda:np.zeros(2))
 for p in sorted((ROOT/'runtime/theory').glob('*')):
  assert done(p);v=J(p/'sums.json');theory[v['c']]+=np.array([v['rejected'],v['n']])
 curve=[dict(c=c,rejection_rate=v[0]/v[1],series=int(v[1]),MC_SE=np.sqrt(v[0]/v[1]*(1-v[0]/v[1])/v[1]),reference='local-to-unity diffusion approximation; stationary initial OU') for c,v in sorted(theory.items())];assert all(v['series']==5000 for v in curve)
 comparison=[];cx=np.array([r['c'] for r in curve]);cy=np.array([r['rejection_rate'] for r in curve]);cs=np.array([r['MC_SE'] for r in curve])
 for row in powers:
  if row['world'].startswith('OU') and row['test']=='ADF':
   reference=float(np.interp(row['c'],cx,cy));se=float(np.hypot(row['MC_SE'],np.interp(row['c'],cx,cs)));comparison.append(dict(world=row['world'],window=row['window'],c=row['c'],observed=row['rejection_rate'],local_reference=reference,difference=row['rejection_rate']-reference,combined_MC_SE=se,within_4MCSE=abs(row['rejection_rate']-reference)<=4*se,limitation='Finite-window/asymptotic and curve-interpolation error is not MC sampling uncertainty'))
 emit('power_table.csv',powers);emit('gate_value_synthetic.csv',values);emit('local_diffusion_reference.csv',curve);emit('theory_comparison.csv',comparison);return powers,values,curve
def primary_tests(p1,p2,contrasts):
 # Gate traversal order is independent of preregistered hypothesis order.
 expected=['P1','P2','P3','P4'];rows=[p1,p2,*[x for x in contrasts if x['id'] in ['P3','P4']]]
 counts=collections.Counter(row['id'] for row in rows)
 assert counts==collections.Counter(expected),('Missing or duplicate primary tests',dict(counts))
 by_id={row['id']:row for row in rows}
 return [by_id[name] for name in expected]
def report():
 cached=ROOT/'runtime/report_complete.json'
 if cached.exists():
  seal=J(cached);assert seal['preregistration_sha256']==H(ROOT/'preregistration.json')
  for rel,digest in seal['files'].items():assert H(ROOT/rel)==digest
  return J(ROOT/'outputs/summary.json')['lines']
 root=ROOT/'outputs';p1,p2,anatomy=anatomy_tables();cells,contrasts=real_tables();decomp=coverage_decomposition();power,value,theory=synthetic_tables();primary=primary_tests(p1,p2,contrasts);p=np.array([x['p_one_sided'] for x in primary]);order=np.argsort(p,kind='stable');adjusted=np.empty(4);adjusted[order]=np.minimum(1,np.maximum.accumulate(p[order]*(4-np.arange(4))))
 for row,q in zip(primary,adjusted):row['holm_p']=float(q);row['supported']=bool(q<=.05 and (row['estimate']<0 if row['id']=='P2' else row['estimate']>0))
 emit('contrasts.csv',[*primary,*[x for x in contrasts if x['id']=='descriptive']]);a=next(x for x in power if x['world']=='calibrated' and x['window']==60 and x['test']=='ADF');v=next(x for x in value if x['world']=='calibrated' and x['window']==60 and x['test']=='ADF');real=[x for x in cells if x['definition']=='RP' and x['world']=='real'];best=max(real,key=lambda x:x['estimate']);comparators=[x for x in value if x['world']=='calibrated' and x['window']==60 and x['test'] in ['DFGLS','VR5','VR10']]
 comparison=pd.read_csv(ROOT/'outputs/tables/theory_comparison.csv');monotonic=all(np.all(np.diff(pd.DataFrame(power).query("world==@name and test=='ADF'").sort_values('c').rejection_rate)>=0) for name in ['OU5','OU10','OU20','OU40'])
 predictions=[dict(id='B1',prediction='ADF60 rate<.20 and share removed>.50',observed=bool(a['rejection_rate']<.20 and v['share_net_removed']>.50)),dict(id='B2',prediction='OU rate rises with c and matches local approximation within MC error',observed=bool(monotonic and comparison.within_4MCSE.all())),dict(id='B3',prediction='DFGLS and VR remove less net than ADF60',observed=all(x['share_net_removed']<v['share_net_removed'] for x in comparators)),dict(id='A1',prediction='RP gate-fail trade mean net>0',observed=p1['estimate']>0)];emit('predictions.csv',predictions)
 lines=[f"ADF60 calibrated rejection rate={a['rejection_rate']:.6f} (unit-root H0 true); AL net removed={v['share_net_removed']:.6f}.",f"RP gate-fail trade mean net={p1['estimate']:.6f} [{p1['ci95_low']:.6f},{p1['ci95_high']:.6f}]; P1 Holm={primary[0]['holm_p']:.6g}.",'; '.join(f"{x['id']} estimate={x['estimate']:.6f},Holm p={x['holm_p']:.6g}" for x in primary[1:]),f"Best gate by observed RP net (descriptive only)={best['gate']}; net={best['estimate']:.6f}; coverage={best['coverage']:.6f}.",'Cointegration pre-tests target their own nulls; failure to reject a unit root does not rule out a tradable transitory component, and this experiment does not identify a unique real-data permanent/transitory decomposition.']
 body='# gate_power_v1\n\n'+'\n\n'.join(lines)+'\n\n## Plain answers\n\nPower and value are reported separately: in mixed RW+OU, ADF/DFGLS H0 is true, VR H0 is false, and KPSS stationarity H0 is false. ADF rejection there is size behavior, not evidence that the entire residual is stationary. Gate harm on real data can combine lost profitable exposure and altered exit timing; the coverage regressions are descriptive.\n\nThe synthetic mixture is a conditional model experiment, not proof that real residuals follow this decomposition. The source calibration itself did not meet every25% fit diagnostic; its model mismatch is retained. A useful pre-test must be judged by cost-adjusted net and coverage as well as statistical rejection; the observed best real gate is a descriptive winner and is not retrained or deployed.\n\nFor pair selection, failure of a cointegration pre-test alone does not establish the absence of an exploitable transitory component; conversely, mean reversion in a synthetic component does not establish profitable real trading.\n\n## Inference and resources\n\nYear-block bootstrap seed20261026,5000draws; P2 uses year-clustered CR1 covariance. Holm adjustment covers P1–P4. No Q-learning fits or new MC policy selection. Eight preregistered real rule cells extend572 to580 trials. The core source and exports are unchanged.\n\n## Methods sources\n\n[Statsmodels KPSS null and auto-bandwidth](https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.kpss.html); [DF-GLS reference and ERS convention](https://bashtage.github.io/arch/unitroot/generated/arch.unitroot.DFGLS.html); [official reference implementation of DF-GLS and robust VR](https://bashtage.github.io/arch/_modules/arch/unitroot/unitroot.html). Production test code uses NumPy/stdlib; source rolling fits/accounting remain imported unchanged.\n'
 best_contrast=next(row for row in contrasts if row['definition']=='RP' and row['gate']==best['gate'] and row['comparator']=='G0')
 evidence='the descriptive interval is above zero' if best_contrast['ci95_low']>0 else 'the descriptive interval includes zero' if best_contrast['ci95_high']>=0 else 'the descriptive interval is below zero'
 body+=f"\n## Data-specific answers\n\nIn the calibrated world, the observed ADF60 rejection rate is{a['rejection_rate']:.6f}, and the signed fraction of ungated AL net removed is{v['share_net_removed']:.6f}. Because this mixed world has a unit root, the ADF rate is size behavior under its own true null. Low rejection and trading-value loss must therefore be distinguished from low power against a stationary alternative.\n\nReal RP gate-fail trades have mean net{p1['estimate']:.6f},95%CI[{p1['ci95_low']:.6f},{p1['ci95_high']:.6f}]. The lost-coverage×ungated-profitability diagnostic has R²={decomp[1]['R2']:.6f}; this measures descriptive fit and does not establish a causal decomposition.\n\nThe observed RP winner is{best['gate']}. Its net difference from G0 is{best_contrast['estimate']:.6f},95%CI[{best_contrast['ci95_low']:.6f},{best_contrast['ci95_high']:.6f}]; {evidence}. This is an outcome-selected descriptive comparison; confirmation remains limited to the preregistered P1–P4 family. Formal permanent/transitory identification and a new pair-selection policy are outside this experiment.\n"
 publish(ROOT/'README.md',body)
 if (root/'summary.json').exists():assert J(root/'summary.json')['lines']==lines
 else:write(root/'summary.json',dict(lines=lines,real_trials=580,new_real_trials=8,source_writes=0,df=df()))
 plots(power,theory,anatomy,cells)
 files={str(p.relative_to(ROOT)):H(p) for p in root.rglob('*') if p.is_file()};files['README.md']=H(ROOT/'README.md');write(cached,dict(status='complete',files=files,preregistration_sha256=H(ROOT/'preregistration.json')))
 return lines
def plots(power,theory,anatomy,cells):
 import matplotlib.pyplot as plt
 def save(fig,name):
  checkpoint();fig.tight_layout();stream=io.BytesIO();fig.savefig(stream,format='png',dpi=140);plt.close(fig);publish(ROOT/'outputs/plots'/name,stream.getvalue())
 fig,ax=plt.subplots(figsize=(7,5));curve=pd.DataFrame(theory);ax.plot(curve.c,curve.rejection_rate,label='local diffusion reference')
 for name,g in pd.DataFrame(power).query("test=='ADF'").groupby('world'):
  if name.startswith('OU'):ax.plot(g.c,g.rejection_rate,'o',label=name)
 ax.set(xlabel='c = W(1 − phi)',ylabel='Actual rejection rate');ax.legend(fontsize=8);save(fig,'01_ADF_power_vs_c.png')
 fig,axes=plt.subplots(1,2,figsize=(11,4))
 for kind,ax in zip(['RP','RF'],axes):
  f=pd.DataFrame(anatomy);f=f[(f.definition==kind)&(f.category=='decile')&(f.bin.astype(str)!='0')]
  for (world,rep),g in f.groupby(['world','replicate']):g=g.assign(decile=g.bin.astype(int)).sort_values('decile');ax.plot(g.decile,g.estimate,'o-',label=world+str(rep))
  ax.set(title=kind,xlabel='ADF p-value decile',ylabel='Mean trade net bps');ax.legend()
 save(fig,'02_trade_net_by_p_decile.png');fig,ax=plt.subplots(figsize=(7,5))
 for row in cells:
  if row['definition']=='RP' and row['world']=='real':ax.scatter(row['coverage'],row['estimate']);ax.annotate(row['gate'],(row['coverage'],row['estimate']))
 ax.set(xlabel='Share of valid flat decisions passing gate',ylabel='Real RP net bps/year');save(fig,'03_gate_coverage_net.png')
