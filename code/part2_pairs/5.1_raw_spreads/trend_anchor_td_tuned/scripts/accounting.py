from common import *
import numpy as np,pandas as pd
from sources import setup,frozen
def old_daily():
 path=note(SOURCE/'outputs/tables/portfolio_daily.csv.gz');wanted=set(frozen('runtime/accounting/CSCV_GLOBAL')['cells']);out=[]
 for f in pd.read_csv(path,usecols=['pool','cell','cost','variant','world','date','net_bps'],chunksize=200000):
  f=f[f.world.eq('real')].copy();f['name']=f['pool']+'|'+f.cell+'|'+f.cost.astype(str)+'|'+f.variant;out.append(f[f.name.isin(wanted)])
 return pd.concat(out,ignore_index=True)
def run(real,daily):
 source=setup().sources.original_inference();old=frozen('runtime/accounting/CSCV_GLOBAL');prior=old_daily();rows=[]
 new=daily[daily.world.eq('real')].copy();new['name']=new['pool']+'|'+new.cell+'|'+new.cost.astype(str)+'|'+new.variant
 for pool in ['GLOBAL',*POOLS]:
  names=[n for n in old['cells'] if pool=='GLOBAL' or n.startswith(pool+'|')];indices=[old['cells'].index(n) for n in names];matrix=[v for v in np.asarray(old['annual_matrix'])[indices]]
  both=pd.concat([prior,new],ignore_index=True);both=both[both.pool.eq(pool)] if pool!='GLOBAL' else both
  series=[];series_names=[]
  for name,g in both.groupby('name',sort=True):series.append(g.set_index('date').net_bps);series_names.append(name)
  combined=pd.concat(series,axis=1);combined.columns=series_names;common=combined.dropna();sr=common.mean()/common.std(ddof=1);valid=np.isfinite(sr);best=sr[valid].idxmax();x=common[best].to_numpy();v=x-x.mean();m2=np.mean(v*v)
  for N in [len(series_names),544]:
   out=source.dsr(float(sr[best]),len(x),float(np.mean(v**3)/m2**1.5),float(np.mean(v**4)/m2**2),float(sr[valid].var(ddof=1)),N)
   rows.append(dict(kind='DSR',pool=pool,best_cell_for_accounting_only=best,**out));count('DSR_evaluations')
  f=real if pool=='GLOBAL' else real[real.pool.eq(pool)]
  for key,g in f.groupby(['pool','cell','cost'],sort=True):
   annual=g.groupby('test_year').apply(lambda a:np.average(a.loc[a.available,'net_bps'],weights=a.loc[a.available,'weight']),include_groups=False).reindex(YEARS)
   assert np.isfinite(annual).all();matrix.append(annual.to_numpy());names.append('|'.join(map(str,[*key,'primary'])))
  assert len(matrix)==len(series_names)
  out,splits=source.pbo(np.asarray(matrix));count('CSCV_splits',out['n_splits'])
  save(f'runtime/accounting/CSCV_{pool}',dict(summary=out,cells=names,annual_matrix=np.asarray(matrix)))
  rows.append(dict(kind='PBO',pool=pool,PBO=out['PBO'],n_splits=out['n_splits'],n_cells=len(names),cumulative_trials=544));del splits
 csv_new(TABLES/'dsr_pbo.csv',rows);return pd.DataFrame(rows),prior
