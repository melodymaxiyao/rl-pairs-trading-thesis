"""Fixed36-point indirect inference with exact source-statistic definitions."""
from common import *
from simulation import Model,simulate
from pipeline import calibration_panel,pooled_panel_statistics
import itertools,gc
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

def panel_job(task):
 guard();os.environ['OU_SCOPE']='calibration';index,h,w,year=task
 p=simulate(Model(h,1.,family='W',w=w),spec()['calibration']['series'],['ou_realstrength_v1','calibration',year],sessions=60+252)
 return calibration_panel(p,year)

def cleanup(index):
 for year in range(spec()['calibration']['years']):
  stem=f'runtime/calibration/point{index:02d}/year{year:02d}'
  for suffix in ['.pkl.gz','.ready.json']:
   path=RUN/(stem+suffix)
   if path.exists():path.unlink()

def calibrate(workers=1):
 final=cached('runtime/calibration_result')
 if final is not None:return final
 targets=spec()['calibration']['targets'];grid=list(itertools.product(spec()['calibration']['half_lives'],spec()['calibration']['w']));points=[]
 pool=ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) if workers>1 else None
 try:
  for index,(h,w) in enumerate(grid):
   stem=f'runtime/calibration_points/p{index:02d}';done=cached(stem)
   if done is not None:points.append(done);cleanup(index);continue
   panels=[];missing=[]
   for year in range(spec()['calibration']['years']):
    old=cached(f'runtime/calibration/point{index:02d}/year{year:02d}')
    if old is None:missing.append((index,h,w,year))
   iterator=pool.map(panel_job,missing,chunksize=1) if pool else map(panel_job,missing)
   for task,panel in zip(missing,iterator):
    save(f'runtime/calibration/point{index:02d}/year{task[3]:02d}',panel);checkpoint()
   hashes={}
   for year in range(spec()['calibration']['years']):
    key=f'runtime/calibration/point{index:02d}/year{year:02d}';panels.append(cached(key));hashes[key]=sha(RUN/(key+'.pkl.gz'))
   statistics=pooled_panel_statistics(panels);unit_sigma=statistics['sigma']['real'];assert unit_sigma>0 and np.isfinite(unit_sigma);scale=targets['sigma_epsilon_median']/unit_sigma
   fitted=dict(statistics['excess'],sigma_epsilon_median=unit_sigma*scale);errors={key:(fitted[key]-value)/abs(value) for key,value in targets.items()};objective=sum(v*v for v in errors.values())
   done=dict(index=index,h0=h,w=w,sigma_total=scale,sigma_ou=scale*np.sqrt(1-w),sigma_rw=scale*np.sqrt(w),fitted=fitted,relative_errors=errors,objective=objective,unit_statistics=statistics,panel_hashes=hashes,series_years=spec()['calibration']['series_years_per_point'])
   save(stem,done);points.append(done);cleanup(index);del panels;gc.collect();print('Calibration',index+1,'/36',f'h={h},w={w:g},relative_SSE={objective:.8g}',flush=True);checkpoint()
 finally:
  if pool:pool.shutdown(wait=True,cancel_futures=True)
 ranked=sorted(points,key=lambda row:(row['objective'],row['index']));result=dict(points=points,chosen=ranked[0],second=ranked[1],targets=targets,criterion='equal-weight squared relative errors; sigma scaled exactly',adequate_25pct_diagnostic=all(abs(v)<=.25 for v in ranked[0]['relative_errors'].values()),diagnostic_is_not_a_selection_or_stopping_rule=True)
 save('runtime/calibration_result',result);write(RUN/'calibration.json',result);return result
