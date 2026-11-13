"""One resumable low-priority command, hard stop after the sealed Part B decision."""
import argparse,concurrent.futures,fcntl,multiprocessing,pathlib,resource,signal,statistics,subprocess,sys,traceback
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from io_utils import *
import bootstrap
from groups import done
PY=str(INPUT/'.venv/bin/python');WORLDS=['OU20','calibrated','second','RW']
def setup(science=False):
 bootstrap.validate_sources();install_guard()
 if not science:bootstrap.views()
 return bootstrap.core_modules() if science else None
def other_tasks():
 active=[]
 for line in subprocess.check_output(['ps','-ww','-axo','pid=,ppid=,command='],text=True).splitlines():
  p=line.split(None,2)
  if len(p)!=3:continue
  pid,ppid,cmd=p
  if int(pid) in [os.getpid(),os.getppid()]:continue
  if 'python' in cmd.lower() and any(s in cmd for s in [str(INPUT),str(ROOT.parent)]) and str(ROOT) not in cmd and '/python-env-tools/' not in cmd:active.append(line)
 assert not active,('Another FYP computation is running',active)
def task_root(task):
 p,*v=task
 if p=='R':return ROOT/f'runtime/reproduction/{v[0]}_{v[1]}'
 if p=='F':return ROOT/f'runtime/RF_increments/{v[0]}'
 if p=='A':return ROOT/f'runtime/part_A/{v[0]}_{v[1]:03d}'
 if p=='B':return ROOT/f'runtime/part_B/{v[0]}_{v[1]}_{v[2]}'
 if p=='B1':return ROOT/f'runtime/part_B1/{v[0]}_{v[1]}_{v[2]}'
 if p=='B2':return ROOT/f'runtime/B2_receipts/{v[0]}_{v[1]}_{v[2]}'
 if p=='NB':return ROOT/f'runtime/bucket_null_chunks/{v[0]}/{v[1]:05d}'
 if p=='N':
  import numpy as np
  key=hashlib.sha256(np.ones(756,bool).tobytes()).hexdigest();return ROOT/f'runtime/null_chunks/{key}/{v[0]:05d}'
 raise AssertionError(task)
def work(task):
 setup(science=True);signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop);checkpoint();phase,*args=task;start=time.monotonic()
 if phase=='R':
  import reproduce;value=reproduce.job(*args)
 elif phase=='F':
  import residuals;value=residuals.rf_year(*args)
 elif phase=='NB':
  import null_buckets;value=null_buckets.chunk(*args)
 elif phase=='N':
  import nulls,numpy as np;value=nulls.chunk(np.ones(756,bool),*args)
 elif phase=='A':
  import part_a;value=part_a.job(*args)
 elif phase=='B1':
  import part_b1;value=part_b1.job(*args)
 elif phase=='B2':
  import part_b
  from groups import finish
  value=part_b.job(*args);value=finish(task_root(task),{'source.json':value},phase='B2',kind=args[0],year=args[1],wi=args[2])
 elif phase=='B':
  import part_b;value=part_b.job(*args)
 else:raise AssertionError(task)
 row=dict(task=task,seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,result=value);journal('job_finished',**row);return row
def estimate():
 components=dict(retained_signals_params_IC_and_fit_receipts=1280*2**20,two_historical_snapshots_helpers=288*2**20,null_calibrations_and_LLT_outputs=160*2**20,reports_atomic_publication_and_control=128*2**20);peak=sum(components.values());return dict(additional_peak_bytes=peak,start_bytes=math.ceil(1.3*peak+1.5*GIB),components=components,task_budget_bytes=2*GIB,method='Frozen matrix dimensions, gzip worst-case retained signal maps, compact fit/IC records and measured analogous historical snapshot sizes; task budget and pause guards remain binding')
def choose_workers(cores,available,peak):
 reserve=2*GIB;return max(1,min(max(1,cores-2),int(.5*max(0,available-reserve)//max(peak,1))))
def worker_peak(phase,measurements):
 # Measured, equal-sized stage-A groups replace the pre-measurement allocation
 # bound. Do not borrow small A groups to size the unmeasured large B folds.
 observed=[r['peak_rss_bytes'] for r in measurements if r['task'][0]==phase]
 if phase in ['B1','B2','NB'] and not observed:return int(1.25*GIB) # mandatory fresh-process planned probe precedes scheduling
 return max(observed) if observed else max([int(1.25*GIB)]+[r['peak_rss_bytes'] for r in measurements])
def resource_plan(phase,remaining,cap):
 memory=memory_snapshot();cores=int(subprocess.check_output(['sysctl','-n','hw.logicalcpu'],text=True));total=int(subprocess.check_output(['sysctl','-n','hw.memsize'],text=True));rows=[json.loads(s) for s in (ROOT/'runtime/journal.jsonl').read_text().splitlines()] if (ROOT/'runtime/journal.jsonl').exists() else [];measurements=[r for r in rows if r.get('action')=='job_finished'];peak=worker_peak(phase,measurements);workers=choose_workers(cores,memory['available'],math.ceil(1.25*peak));workers=min(workers,cap) if cap else workers
 if phase=='A':workers=min(workers,4)
 if phase in ['B','B1','B2']:workers=min(workers,6)
 samples=[r['seconds'] for r in measurements if r['task'][0]==phase][-20:];projection=statistics.median(samples)*math.ceil(remaining/workers) if samples else None
 print(f'{phase}: RAM total={total/GIB:.2f}GiB,free={memory["free"]/GIB:.2f}GiB,available={memory["available"]/GIB:.2f}GiB; cores={cores}; observed/analytical peak={peak/GIB:.3f}GiB +25%margin; reserve>=2GiB; --workers {workers}; nice10.',flush=True)
 print(f'{phase}: remaining groups={remaining}; estimated compute hours={projection/3600:.2f}' if projection is not None else f'{phase}: runtime estimate will be printed from the first planned completed group; new exact-mask null calibrations add unknown work.',flush=True)
 journal('resource_plan',phase=phase,workers=workers,memory=memory,peak_rss_bytes=peak,remaining_groups=remaining,estimated_compute_seconds=projection);return workers
def stage(phase,tasks,cap=None,probe=True):
 tasks=[tuple(t) for t in tasks if not done(task_root(t))]
 if not tasks:return
 records=list(map(json.loads,(ROOT/'runtime/journal.jsonl').read_text().splitlines())) if (ROOT/'runtime/journal.jsonl').exists() else []
 resource_plan(phase,len(tasks),cap)
 if probe and not any(r.get('action')=='job_finished' and r['task'][0]==phase for r in records):
  # A planned group is committed once; this is not an extra scientific trial.
  probe_index=next((i for i,t in enumerate(tasks) if phase not in ['B1','B2'] or not done(ROOT/f'runtime/part_B/{t[1]}_{t[2]}_{t[3]}')),0)
  first=tasks.pop(probe_index);print('Measuring first planned group:',first,flush=True);p=subprocess.run([PY,'-I','-B',str(ROOT/'scripts/run_partial.py'),'--job',json.dumps(first)],check=False)
  if p.returncode:raise Pause(f'Planned group probe stopped with code{p.returncode}; resume same command')
 workers=resource_plan(phase,len(tasks),cap);replace(ROOT/'status.json',dict(status='running',phase=phase,workers=workers,part_C_started=False,planned_trials=594))
 offset=0
 while offset<len(tasks):
  checkpoint();workers=min(workers,resource_plan(phase,len(tasks)-offset,cap));wave=tasks[offset:offset+workers]
  # A fresh process for every group gives a per-group RSS peak instead of
  # the parent's lifetime high-water mark, and releases group arrays promptly.
  with concurrent.futures.ProcessPoolExecutor(workers,mp_context=multiprocessing.get_context('spawn'),max_tasks_per_child=1) as pool:
   futures=[pool.submit(work,t) for t in wave]
   try:
    for f in concurrent.futures.as_completed(futures):print('Finished',f.result(),flush=True);checkpoint()
   except BaseException:
    pathlib.Path(os.environ['GATE_POWER_STOP']).touch(exist_ok=True)
    for f in futures:f.cancel()
    raise
  if phase=='B1':print_split_estimates(cap)
  offset+=len(wave)
def print_split_estimates(cap):
 root=ROOT/'runtime/split_runtime_estimate_after5'
 from groups import finish
 if done(root):return
 completed=list((ROOT/'runtime/part_B1').glob('*/complete.json'))
 if len(completed)<5:return
 records=[json.loads(l) for l in (ROOT/'runtime/journal.jsonl').read_text().splitlines()]
 samples=[r['seconds'] for r in records if r.get('action')=='job_finished' and r['task'][0]=='B1']
 old=[r['seconds'] for r in records if r.get('action')=='job_finished' and r['task'][0]=='B']
 workers=resource_plan('B1',132-len(completed),cap);baseline=max(samples) if samples else None
 # Conservatively use the slower observed B1 group because early groups reuse
 # completed pooled fits; range is provisional until cold groups are measured.
 b1=None if baseline is None else baseline*(132-len(completed))/workers
 b2=None if not old else statistics.median(old)*sum(not done(ROOT/f'runtime/part_B/{kind}_{y}_{wi}') for kind in ['RP','RF'] for y in range(2004,2026) for wi in range(3))
 memory=memory_snapshot();cores=int(subprocess.check_output(['sysctl','-n','hw.logicalcpu'],text=True));provisional_workers=choose_workers(cores,memory['available'],math.ceil(1.25*1.25*GIB));provisional_workers=min(6,provisional_workers,cap or 6)
 last={};chunk_seconds=[]
 for r in records:
  if r.get('action')=='group_committed' and r.get('group','').startswith('runtime/null_chunks/'):
   pid=r['pid'];stamp=r['time_ns']/1e9
   if pid in last and 0<stamp-last[pid]<120:chunk_seconds.append(stamp-last[pid])
   last[pid]=stamp
 calibration=None if not chunk_seconds else statistics.median(chunk_seconds)*600/provisional_workers
 projected_B2=None if b2 is None or calibration is None else b2/provisional_workers+calibration
 value=dict(B1_completed=len(completed),B1_remaining=132-len(completed),chosen_B1_workers=workers,B1_measured_seconds=samples,B1_projection_hours=None if b1 is None else b1/3600,B2_serial_reference_hours=None if b2 is None else b2/3600,B2_provisional_workers=provisional_workers,B2_bucket_calibration_replications=60000,B2_calibration_projection_hours=None if calibration is None else calibration/3600,B2_total_projection_hours=None if projected_B2 is None else projected_B2/3600,B2_projection_range_hours=None if projected_B2 is None else [.5*projected_B2/3600,2*projected_B2/3600],limitations='Provisional: first groups may reuse pooled fits; B2 old timings include primary/old calibration and precede chunking. B2 worker assumption uses initial1.25GiB bound+25percent and will be replaced by the fresh planned B2 probe. Bucket timings proxy completed old100-rep chunks. Future years may cost more. Not a completion deadline.')
 finish(root,{'estimate.json':value},phase='B1_B2_runtime_estimate_after_first5')
 print('NEW B1/B2 RUNTIME ESTIMATES after first5 B1 groups:',json.dumps(value),flush=True)
def all_tasks():
 return [('R',kind,y) for y in range(2004,2026) for kind in ['RP','RF']]+[('F',y) for y in RF_years()]+[('A',world,i) for i in range(200) for world in WORLDS]+[('B',kind,y,wi) for y in range(2004,2026) for kind in ['RP','RF'] for wi in range(3)]
def RF_years():
 c,d,a=bootstrap.core_modules();first=int(d.panel()['dates'].astype('datetime64[Y]').astype(int).min()+1970);return range(first,2026)
def validate_A_size():
 import pandas as pd,numpy as np
 rows=[]
 for i in range(200):
  root=task_root(('A','RW',i));assert done(root);rows.append(pd.read_csv(root/'secondary.csv.gz',float_precision='round_trip'))
 f=pd.concat(rows,ignore_index=True);valid=f[f.converged==True];rate=float(valid.PCI_reject.mean()) if len(valid) else None;tol=max(.015,4*np.sqrt(.05*.95/max(1,len(valid))));result=dict(status='PASS' if rate is not None and abs(rate-.05)<=tol else 'FAIL',rate=rate,n=len(valid),invalid=20000-len(valid),tolerance=float(tol),sample='independent Part A pureRW truth cohort,not the calibration quantile sample')
 write(ROOT/'runtime/independent_RW_size_check.json',result);print('Independent PCI size check:',result,flush=True)
 if result['status']!='PASS':raise Pause('Independent RW size check failed; actual rate retained; no Part B or threshold retuning')
def verify():
 bootstrap.validate_sources();seal=J(ROOT/'complete.json');assert seal['status']=='complete_through_B' and seal['part_C_started']==False and seal['new_real_trading_cells_executed']==0;assert H(ROOT/'manifest.json')==seal['manifest_sha256']
 for rel,digest in J(ROOT/'manifest.json')['files'].items():assert H(ROOT/rel)==digest,rel
 assert all(done(task_root(t)) for t in all_tasks());assert all(done(task_root((phase,kind,y,wi))) for phase in ['B1','B2'] for y in range(2004,2026) for kind in ['RP','RF'] for wi in range(3));decision=J(ROOT/'go_nogo.json');assert not decision['part_C_started'];print('VERIFIED through Part B; Part C not started; no new real trading trials.',flush=True)
def runtime_manifest_files():
 files={}
 # Historical verifier vendors are authenticated evidence copies of OTHER
 # tasks. Preserve their bytes, not their original relative group namespaces.
 for p in (ROOT/'runtime').rglob('*'):
  if not p.is_file():continue
  relative=p.relative_to(ROOT/'runtime')
  if relative.parts[0].startswith('historical_'):files[str(p.relative_to(ROOT))]=H(p)
 for prefix in ['reproduction','RF_increments','null_chunks','part_A','part_B','A_LLT_timings','B_fits','part_B1','B1_report','B2_receipts','B2_bucket_overlays','bucket_null_chunks','bucket_pattern_sensitivity','split_runtime_estimate_after5']:
  for p in (ROOT/'runtime'/prefix).rglob('complete.json'):
   assert done(p.parent);files[str(p.relative_to(ROOT))]=H(p)
   for rel,digest in J(p)['files'].items():files[rel]=digest
 return files
def final_print():
 print('README:',ROOT/'README.md',flush=True)
 for line in J(ROOT/'summary.json')['lines']:print(line,flush=True)
 print('df Avail',df()['available_bytes']/GIB,'GiB. STOPPED AFTER PART B. Part C, relocation and slimming are not launched.',flush=True)
def integrity(label):
 print('Verifying sealed core/native upstream and complete historical scope:',label,flush=True);p=subprocess.run([PY,'-I','-B',str(CORE/'scripts/verify.py'),'--tier','3'],capture_output=True,text=True);publish(event('core_'+label).with_suffix('.log'),p.stdout+p.stderr);assert p.returncode==0,p.stderr[-4000:];bootstrap.native_verification(label);bootstrap.historical(label+'_'+str(time.time_ns()))
def main(args):
 setup();signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop);os.environ['GATE_POWER_STOP']=str(ROOT/f'runtime/STOP_{os.getpid()}');cap=None if args.workers=='auto' else int(args.workers);assert cap is None or cap>=1
 with (ROOT/'runtime/runner.lock').open('a+') as lock,(CORE/'runtime/runner.lock').open('r') as corelock,((ROOT.parent/'gate_power_v1')/'runtime/runner.lock').open('r') as gatelock:
  for l in [lock,corelock,gatelock]:fcntl.flock(l,fcntl.LOCK_EX|fcntl.LOCK_NB)
  other_tasks();e=estimate();disk=df();print(f'Additional peak estimate={e["additional_peak_bytes"]/GIB:.3f}GiB; start gate={e["start_bytes"]/GIB:.3f}GiB; Avail={disk["available_bytes"]/GIB:.3f}GiB',flush=True)
  if disk['available_bytes']<e['start_bytes']:raise Pause('Start disk gate not met; no computation launched')
  mem=memory_snapshot()
  if mem['available']<3*GIB:raise Pause('Not enough available RAM above the2GiB reserve for even one worker')
  os.environ['PARTIAL_RAM_GUARD']='1';os.environ['PARTIAL_INITIAL_SWAPOUTS']=str(mem['swapouts']);checkpoint();c,d,a=bootstrap.core_modules();env=c.environment();write(event('environment'),env);write(event('launch_estimate'),dict(**e,df=disk,memory=mem))
  if args.check_only:print('CHECK_ONLY_PASS: correct environment, approved prompts/location and source pins; no fits or replay.',flush=True);return
  if (ROOT/'complete.json').exists():verify();final_print();return
  integrity('launch');stage('R',[('R',kind,y) for y in range(2004,2026) for kind in ['RP','RF']],cap);assert all(done(task_root(t)) for t in all_tasks() if t[0]=='R');print('Exact original AL/Band/Q_best reproduction all44fold groups PASS. New science may start.',flush=True)
  stage('F',[('F',y) for y in RF_years()],cap);stage('N',[('N',i) for i in range(0,20000,100)],cap)
  import nulls,numpy as np
  nulls.critical(np.ones(756,bool))
  for index in range(200):stage('A',[('A',world,index) for world in WORLDS],cap)
  validate_A_size()
  # Warm starts require ordered years; six independent family/world chains.
  assert (ROOT/'audit/speedup_B1_B2_v1/chunk_verification.json').exists(), 'Completed-group chunk verification required before B1'
  for year in range(2004,2026):stage('B1',[('B1',kind,year,wi) for kind in ['RP','RF'] for wi in range(3)],cap)
  import part_b1,null_buckets
  primary=part_b1.report();print('B1 GO/NO-GO:',json.dumps(primary),'— primary decision available; B2 secondary follows; Part C not started.',flush=True)
  null_buckets.design()
  stage('NB',[('NB',lower,offset) for lower in [500,600,700] for offset in range(0,20000,100)],cap)
  for lower in [500,600,700]:null_buckets.critical(lower)
  null_buckets.sensitivity()
  for year in range(2004,2026):stage('B2',[('B2',kind,year,wi) for kind in ['RP','RF'] for wi in range(3)],cap)

  import report_partial
  report_partial.report();print('Part B decision:',J(ROOT/'go_nogo.json')['decision'],'— STOPPING BEFORE PART C.',flush=True);integrity('final');checkpoint();files={}
  for folder in ['scripts','tests','audit','outputs','preregistration']:
   for p in (ROOT/folder).rglob('*'):
    if p.is_file():files[str(p.relative_to(ROOT))]=H(p)
  for p in ROOT.iterdir():
   if p.is_file() and p.name not in ['manifest.json','complete.json','status.json']:files[p.name]=H(p)
  files.update(runtime_manifest_files())
  for p in [ROOT/'runtime/kalman.so',ROOT/'runtime/report_B_complete.json']:files[str(p.relative_to(ROOT))]=H(p)
  if (ROOT/'manifest.json').exists():
   for rel,digest in J(ROOT/'manifest.json')['files'].items():assert H(ROOT/rel)==digest
  else:write(ROOT/'manifest.json',dict(files=files))
  write(ROOT/'complete.json',dict(status='complete_through_B',manifest_sha256=H(ROOT/'manifest.json'),preregistration_sha256=H(ROOT/'preregistration.json'),go=J(ROOT/'go_nogo.json')['decision'],part_C_started=False,real_trials_before=580,new_real_trading_cells_executed=0,conditional_planned_trials=594));replace(ROOT/'status.json',dict(status='complete_through_B',go=J(ROOT/'go_nogo.json')['decision'],part_C_started=False));verify();journal('complete_stop_B',df=df());final_print()
if __name__=='__main__':
 try:
  if len(sys.argv)>1 and sys.argv[1]=='--dispatch':bootstrap.dispatch(pathlib.Path(sys.argv[2]),sys.argv[3:])
  elif len(sys.argv)>1 and sys.argv[1]=='--job':
   task=json.loads(sys.argv[2]);print('Planned probe finished',work(task),flush=True)
  else:
   p=argparse.ArgumentParser();p.add_argument('--workers',default='auto');p.add_argument('--check-only',action='store_true');p.add_argument('--verify',action='store_true');args=p.parse_args()
   if args.verify:setup();verify();final_print()
   else:main(args)
 except BaseException as error:
  if isinstance(error,SystemExit) and error.code==0:raise
  try:journal('stopped',reason=str(error),traceback=traceback.format_exc(),df=df())
  except BaseException:pass
  print('PAUSED/STOPPED; completed groups retained. Resume the same Terminal command.',str(error),flush=True);raise
