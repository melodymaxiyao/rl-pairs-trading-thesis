"""Independent resumable residual-fitting jobs; no rule execution or real work in preparation."""
from common import *
from inputs import panel,index,SECTOR_ETF
from models import ou_fit,pair_rolling,factor_window,regress,sign_series
from streams import param_shard
from numpy.lib.stride_tricks import sliding_window_view
WORLD=[('real',0),('N2',0),('N2',1)]
HALF_BINS=np.r_[np.arange(0,122,2),np.inf]
FIELDS=['b','b_se','kappa','half_life','m','sigma_epsilon','sigma_eq','s','increment_predictive_r2','valid','reverting','fast','no_reversion','nonpositive_b','significantly_below_one']
def identifier(job):return f"{job['definition']}_{job['year']}_{job['part']:02d}"
def design():
 jobs=[dict(definition='RP',year=y,part=b) for y in spec()['years'] for b in range(12)]
 for kind in ['RF']+(['RE'] if spec()['available_sector_etfs'] else []):
  jobs += [dict(definition=kind,year=2003,part=m) for m in [10,11,12]]
  jobs += [dict(definition=kind,year=y,part=m) for y in spec()['years'] for m in range(1,13)]
 return jobs

def calibration(fits,which,series,weight,definition):
 result=[]
 for wi,(world,rep) in enumerate(WORLD):
  f=fits[wi];ok=f['reverting'][which]
  row=dict(series=series,weight=weight,definition=definition,world=world,replicate=rep,valid_reverting_days=int(ok.sum()),total_days=len(which))
  for key in ['half_life','sigma_epsilon','sigma_eq','increment_predictive_r2']:
   a=np.asarray(f[key])[which];row[key]=float(np.nanmedian(a[ok])) if ok.any() else np.nan
  result.append(row)
 return result

def parameter_rows(fits,which,ends,series,weight,definition,year,dates):
 for wi,(world,rep) in enumerate(WORLD):
  f=fits[wi]
  for loc in which:
   yield dict(definition=definition,world=world,replicate=rep,series=series,test_year=year,date=str(dates[ends[loc]]),weight=weight,**{k:f[k][loc] for k in FIELDS})

def rp_rows(job,result):
 data=panel();R=data['returns'];dates=data['dates'];years=dates.astype('datetime64[Y]').astype(int)+1970;y=job['year'];test=np.flatnonzero(years==y)
 if not len(test):raise ValueError(f'Calendar has no sessions for RP test year {y}; no fit attempted')
 first=np.flatnonzero(years>=y-3)[0];start=max(0,first-59);end=test[-1]+1;ends=np.arange(start+59,end);locs=np.searchsorted(ends,test);members=data['meta']['pairs'][str(y)][job['part']*250:(job['part']+1)*250]
 for row in members:
  ia=index(row['leg_A'],data);ib=index(row['leg_B'],data);name=f"{y}:{row['pair']}";weight=row['ipw']
  if ia is None or ib is None:
   result['unavailable'].append(dict(series=name,reason='sealed leg symbol unavailable'))
   result['series'].append(dict(series=name,pair=row['pair'],weight=weight,indices=test,score=np.full((3,len(test)),np.nan),half=np.full((3,len(test)),np.nan),previous_kappa=np.full((3,len(test)),np.nan),future=np.full((3,len(test),3),np.nan),alpha=np.full(len(test),np.nan),beta=np.full(len(test),np.nan)))
   for world,rep in WORLD:
    result['calibration'].append(dict(series=name,weight=weight,definition='RP',world=world,replicate=rep,valid_reverting_days=0,total_days=len(test),**{key:np.nan for key in ['half_life','sigma_epsilon','sigma_eq','increment_predictive_r2']}))
    result['structure'].append(dict(definition='RP',world=world,replicate=rep,test_year=y,series=name,weight=weight,days=len(test),valid=0,fast=0,significant=0,no_reversion=0,nonpositive_b=0,half_histogram=np.zeros(len(HALF_BINS)-1,int)))
    for t in test:yield dict(definition='RP',world=world,replicate=rep,series=name,test_year=y,date=str(dates[t]),weight=weight,**{key:False if key in ['valid','reverting','fast','no_reversion','nonpositive_b','significantly_below_one'] else np.nan for key in FIELDS})
   continue
  residual,alpha,beta=pair_rolling(R[start:end,ia],R[start:end,ib]);fits=[];futures=np.full((3,len(test),3),np.nan)
  for wi,(world,rep) in enumerate(WORLD):
   signs=np.ones(len(dates)) if world=='real' else sign_series(spec()['null_seed'],'RP|'+name,rep,dates)
   signs_window=sliding_window_view(signs[start:end],60);fits.append(ou_fit(residual*signs_window))
   for hi,h in enumerate(spec()['horizons']):
    for ti,t in enumerate(test):
     if t+h>=len(dates) or years[t+h]!=y:continue
     f=(R[t+1:t+h+1,ia]-alpha[locs[ti]]-beta[locs[ti]]*R[t+1:t+h+1,ib])*signs[t+1:t+h+1]
     if np.isfinite(f).all():futures[wi,ti,hi]=f.sum()
  previous=np.searchsorted(ends,test-60);previous_ok=(previous>=0)&(previous<len(ends))
  prev=np.full((3,len(test)),np.nan)
  for wi in range(3):prev[wi,previous_ok]=fits[wi]['kappa'][previous[previous_ok]]
  endpoints=np.flatnonzero((ends%60==59)&(ends-60>=ends[0]));previous_end=np.searchsorted(ends,ends[endpoints]-60)
  persistence=[dict(date=str(dates[ends[t]]),previous_date=str(dates[ends[t]-60]),fold=y,series=name,definition='RP',world=world,replicate=rep,weight=weight,window='train' if years[ends[t]]<y else 'test',current=float(fits[wi]['kappa'][t]),previous=float(fits[wi]['kappa'][previous_end[j]])) for wi,(world,rep) in enumerate(WORLD) for j,t in enumerate(endpoints)]
  result['persistence'].extend(persistence);result['calibration'].extend(calibration(fits,locs,name,weight,'RP'))
  result['series'].append(dict(series=name,pair=row['pair'],weight=weight,indices=test,score=np.array([f['s'][locs] for f in fits],np.float64),half=np.array([f['half_life'][locs] for f in fits],np.float64),previous_kappa=prev.astype(np.float64),future=futures.astype(np.float64),alpha=alpha[locs].astype(np.float64),beta=beta[locs].astype(np.float64)))
  for wi,(world,rep) in enumerate(WORLD):
   f=fits[wi];result['structure'].append(dict(definition='RP',world=world,replicate=rep,test_year=y,series=name,weight=weight,days=len(locs),valid=int(f['valid'][locs].sum()),fast=int(f['fast'][locs].sum()),significant=int(f['significantly_below_one'][locs].sum()),no_reversion=int(f['no_reversion'][locs].sum()),nonpositive_b=int(f['nonpositive_b'][locs].sum()),half_histogram=np.histogram(f['half_life'][locs][np.isfinite(f['half_life'][locs])],bins=HALF_BINS)[0]))
  yield from parameter_rows(fits,locs,ends,name,weight,'RP',y,dates)
  count('OU_window_fits',3*len(ends))


def factor_rows(job,result):
 data=panel();R=data['returns'];dates=data['dates'];membership=data['membership'];symbols=data['meta']['symbols'];months=dates.astype('datetime64[M]');chosen=np.flatnonzero(months==np.datetime64(f"{job['year']}-{job['part']:02d}"));kind=job['definition'];N=len(symbols);sigma_fields=['half_life','sigma_epsilon','sigma_eq','increment_predictive_r2']
 for t in chosen:
  history=R[max(0,t-251):t+1];members=np.flatnonzero(membership[t]);complete=members[np.isfinite(history[:,members]).all(axis=0)] if len(history)==252 else np.array([],int);valid=complete[np.std(history[:,complete],axis=0)>0] if len(complete) else complete
  if kind=='RE':
   members=np.array([j for j in members if SECTOR_ETF.get(data['meta']['sectors'].get(symbols[j])) in spec()['available_sector_etfs']],int);valid=members.copy()
   # Available sealed ETF variants are separate sector cohorts. Currently XLU only.
   if len(spec()['available_sector_etfs'])!=1:raise RuntimeError('Multiple ETF cohorts need an explicit joint RE aggregation adapter')
   etf=data['indices'][spec()['available_sector_etfs'][0]];window=R[max(0,t-59):t+1];valid=valid[np.isfinite(window[:,valid]).all(axis=0)] if len(window)==60 and np.isfinite(window[:,etf]).all() else np.array([],int)
  day=dict(index=int(t),definition=kind,year=job['year'],indices=valid,members=members,score=np.full((3,len(valid)),np.nan,np.float64),half=np.full((3,len(valid)),np.nan,np.float64),kappa=np.full((3,len(valid)),np.nan,np.float64),future=np.full((3,len(valid),3),np.nan,np.float64),next_return=np.full((3,len(valid)),np.nan,np.float64),calibration={key:np.full((3,len(valid)),np.nan,np.float64) for key in sigma_fields},structure=np.zeros((3,6),int),half_histogram=np.zeros((3,len(HALF_BINS)-1),int),status='unavailable')
  day['structure'][:,5]=len(members)
  required=16 if kind=='RF' else 1
  if len(valid)>=required:
   try:
    if kind=='RF':
     model=factor_window(history[:,valid]);W=np.zeros((N,15));W[valid]=model['weights'];beta=model['beta'];alpha=model['alpha'];residual=model['residual'];gross=model['gross']
    else:
     residual,alpha,beta=regress(window[:,valid],window[:,etf]);W=np.zeros((N,1));W[etf,0]=1.;gross=1+abs(beta[:,0])
    day.update(status='ok',weights=W.astype(np.float64),beta=beta.astype(np.float64),alpha=alpha.astype(np.float64),gross=gross.astype(np.float64))
    fits=[]
    for wi,(world,rep) in enumerate(WORLD):
     sign_dates=dates[t-59:min(len(dates),t+11)];signs=np.ones((len(sign_dates),len(valid))) if world=='real' else np.column_stack([sign_series(spec()['null_seed'],kind+'|'+symbols[j],rep,sign_dates) for j in valid]);fit=ou_fit((residual*signs[:60]).T);fits.append(fit);day['score'][wi]=fit['s'];day['half'][wi]=fit['half_life'];day['kappa'][wi]=fit['kappa'];day['half_histogram'][wi]=np.histogram(fit['half_life'][np.isfinite(fit['half_life'])],bins=HALF_BINS)[0]
     for key in sigma_fields:day['calibration'][key][wi]=fit[key]
     day['structure'][wi]=[fit['valid'].sum(),fit['fast'].sum(),fit['significantly_below_one'].sum(),fit['no_reversion'].sum(),fit['nonpositive_b'].sum(),len(members)]
     if t+1<len(dates):
      needed=np.flatnonzero(np.any(W!=0,axis=1));next_e=R[t+1,valid]-alpha-(R[t+1,needed]@W[needed])@beta.T;day['next_return'][wi]=next_e*signs[60]
     for hi,h in enumerate(spec()['horizons']):
      if t+h>=len(dates) or dates[t+h].astype('datetime64[Y]')!=dates[t].astype('datetime64[Y]'):continue
      rf=R[t+1:t+h+1];required_columns=np.flatnonzero(np.any(W!=0,axis=1));factors=rf[:,required_columns]@W[required_columns];forward=rf[:,valid]-alpha-factors@beta.T;forward*=signs[60:60+h];ok=np.isfinite(forward).all(axis=0);day['future'][wi,ok,hi]=forward[:,ok].sum(axis=0)
    if job['year']>=2004:
     for wi,(world,rep) in enumerate(WORLD):
      for j,stock in enumerate(valid):
       yield dict(definition=kind,world=world,replicate=rep,series=symbols[stock],test_year=job['year'],date=str(dates[t]),weight=1.,**{key:fits[wi][key][j] for key in FIELDS})
    count('OU_window_fits',3*len(valid));count('PCA_fits',int(kind=='RF'))
   except (ValueError,np.linalg.LinAlgError) as exc:day['status']='unavailable';day['reason']=str(exc)
  else:day['reason']='insufficient admissible history or PCA members'
  # Missing-history current members receive explicit unavailable OU rows.
  if job['year']>=2004:
   omitted=set(members)-set(valid) if day['status']=='ok' else set(members)
   for stock in sorted(omitted):
    for world,rep in WORLD:
     yield dict(definition=kind,world=world,replicate=rep,series=symbols[stock],test_year=job['year'],date=str(dates[t]),weight=1.,**{key:False if key in ['valid','reverting','fast','no_reversion','nonpositive_b','significantly_below_one'] else np.nan for key in FIELDS})
  result['days'].append(day)

def work(job):
 guard();os.environ['ROU_SCOPE']='production';name=identifier(job);old=cached('runtime/jobs/'+name)
 if old is not None:return dict(job=job,cached=True)
 result=dict(job=job,series=[],days=[],structure=[],persistence=[],calibration=[],unavailable=[])
 shard=param_shard(name,rp_rows(job,result) if job['definition']=='RP' else factor_rows(job,result));result['parameter_shard']=shard
 save('runtime/jobs/'+name,result);count('completed_jobs');checkpoint_guard();return dict(job=job,cached=False)
