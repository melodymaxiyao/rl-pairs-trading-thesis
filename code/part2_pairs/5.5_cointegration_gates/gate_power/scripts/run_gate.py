"""Single resumable command: verification, exact reproduction, A/B/C, seal and stop."""
import argparse,concurrent.futures,fcntl,importlib.util,multiprocessing,os,pathlib,resource,signal,subprocess,sys,time,traceback
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from io_utils import *
import bootstrap
PY=str(INPUT/'.venv/bin/python')
def setup(science=False):
 bootstrap.validate_sources();install_guard()
 if not science:bootstrap.views()
 return bootstrap.core_modules() if science else None
def run_job(task):
 setup(science=True);signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop);checkpoint();phase,*args=task
 import real_jobs,synthetic_jobs
 start=time.monotonic()
 if phase=='R':value=real_jobs.reproduce(*args)
 elif phase=='A':value=real_jobs.anatomy(*args)
 elif phase=='C':value=real_jobs.alternatives(*args)
 elif phase=='B':value=synthetic_jobs.batch(*args)
 elif phase=='T':value=synthetic_jobs.theory(*args)
 else:raise AssertionError(task)
 result=dict(task=task,seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,result=value);journal('job_finished',**result);return result
def estimate():
 # Disk accounting includes compressed trade records, compact rule exports,
 # synthetic sufficient statistics, two full historical snapshots and temps.
 components=dict(trade_records_and_rule_exports=512*2**20,synthetic_and_theory_sufficient_statistics=64*2**20,two_historical_snapshots_and_helpers=288*2**20,reports_temp_and_control=128*2**20)
 additional=sum(components.values());return dict(additional_peak_bytes=additional,components=components,start_bytes=math.ceil(1.3*additional+1.5*GIB),task_budget_bytes=2*GIB,assumptions='streamed cell/trade outputs,100 synthetic series per batch, no retained raw paths, one-group reports; task2GiB and Avail1.5GiB guards binding. System swap is outside this file estimate; workers reserve30percent available RAM.')
def start_gate():
 e=estimate();sample=df();write(event('launch_estimate'),dict(**e,df=sample));print(f"Additional disk peak {e['additional_peak_bytes']/GIB:.3f}GiB; start gate {e['start_bytes']/GIB:.3f}GiB; Avail {sample['available_bytes']/GIB:.3f}GiB",flush=True)
 if sample['available_bytes']<e['start_bytes']:raise Pause(f"Start gate short by {(e['start_bytes']-sample['available_bytes'])/GIB:.3f}GiB")
 return e
def choose_workers(cores,free,peak):return max(1,min(max(1,cores-2),int(.7*free//peak)))
def memory():
 import re
 total=int(subprocess.check_output(['sysctl','-n','hw.memsize'],text=True));cores=int(subprocess.check_output(['sysctl','-n','hw.logicalcpu'],text=True));raw=subprocess.check_output(['vm_stat'],text=True);page=int(re.search(r'page size of (\d+) bytes',raw)[1]);v={k:int(x) for k,x in re.findall(r'^([^:\n]+):\s+(\d+)\.',raw,re.M)};return dict(total=total,cores=cores,available=page*(v['Pages free']+v.get('Pages speculative',0)+v.get('Pages inactive',0)),raw=raw,definition='free+speculative+inactive; active/wired/compressor excluded; no double-counting purgeable')
def probe(task,phase):
 p=ROOT/f'runtime/probe_{phase}.json'
 if p.exists():
  row=J(p);assert row['task'][0]==phase
  from real_jobs import done
  assert done(pathlib.Path(row['result']['root'])),'Measured planned group must remain sealed and verified'
  return row
 if not p.exists():
  intent=ROOT/f'runtime/probe_{phase}_intent.json'
  if intent.exists() and J(intent)['task']!=list(task):
   publish(ROOT/f'runtime/old_probe_intents/{phase}_{time.time_ns()}.json',intent.read_bytes());replace(intent,dict(task=task))
  else:write(intent,dict(task=task))
  print('Measuring planned group peak RAM:',task,flush=True);process=subprocess.run([PY,'-I','-B',str(ROOT/'scripts/run_gate.py'),'--probe',phase],check=False)
  if process.returncode:raise Pause(f'Planned RAM probe stopped with code{process.returncode}; repeat the command to resume')
 row=J(p);assert row['task']==list(task);return row
def workers_for(phase,tasks):
 if not tasks:return 1
 # Each phase measures one actual pending planned group once and reuses it.
 # The largest observed peak across completed probes is retained conservatively.
 selected=tasks[0]
 if phase in ['R','A','C']:
  c,d,a=bootstrap.core_modules();meta=d.panel()['meta'];dates=d.panel()['dates'];years=dates.astype('datetime64[Y]').astype(int)+1970
  selected=max(tasks,key=lambda task:((len(meta['symbols'])*44 if task[1]=='RF' else len(meta['pairs'][str(task[2])])*12)*int((years==task[2]).sum()),task[2]))
 measurement=probe(selected,phase)
 peaks=[J(p)['peak_rss_bytes'] for p in (ROOT/'runtime').glob('probe_*.json') if not p.name.endswith('_intent.json')];assert peaks;peak=max(peaks);ram=memory();workers=choose_workers(ram['cores'],ram['available'],peak)
 if os.environ.get('GATE_WORKER_CAP'):workers=min(workers,int(os.environ['GATE_WORKER_CAP']))
 samples=[J(p)['seconds'] for p in (ROOT/'runtime').glob(f'probe_{phase}.json')]
 seconds=(samples[0]*len(tasks)/workers) if samples else None
 print(f"{phase}: total RAM {ram['total']/GIB:.2f}GiB; available RAM {ram['available']/GIB:.2f}GiB; cores {ram['cores']}; observed peak/worker {peak/GIB:.3f}GiB; chosen --workers {workers}; nice10",flush=True)
 if seconds is not None:print(f"{phase}: remaining compute planning estimate {seconds/3600:.2f}h, based on one planned probe; context/fold variation and serial verification/reporting add uncertainty.",flush=True)
 journal('resource_plan',phase=phase,workers=workers,peak_rss_bytes=peak,memory=ram,planning_seconds=seconds);return workers
def phase_tasks(phase):
 import real_jobs
 if phase in ['R','A','C']:
  dirname={'R':'reproduction','A':'anatomy','C':'alternatives'}[phase]
  return [(phase,kind,y) for y in range(2004,2026) for kind in ['RP','RF'] if not real_jobs.done(ROOT/f'runtime/{dirname}/{kind}_{y}')]
 if phase=='B':return [('B',world,i) for world in ['OU5','OU10','OU20','OU40','calibrated','second','RW'] for i in range(200) if not real_jobs.done(ROOT/f'runtime/synthetic/{world}_{i:04d}')]
 if phase=='T':return [('T',c,i) for c in J(ROOT/'preregistration.json')['local_diffusion_reference']['c_grid'] for i in range(50) if not real_jobs.done(ROOT/f'runtime/theory/c{c:g}_{i:03d}')]
 raise AssertionError(phase)
def phase(phase):
 tasks=phase_tasks(phase)
 if not tasks:return
 workers=workers_for(phase,tasks);tasks=phase_tasks(phase);print(f'Phase{phase}: {len(tasks)} groups remaining,workers{workers}',flush=True);replace(ROOT/'status.json',dict(status='running',phase=phase,workers=workers,real_trials_registered=580))
 if workers==1:
  for task in tasks:print('Finished',run_job(task),flush=True)
 else:
  # New pool per bounded wave avoids process-replacement deadlocks and releases
  # each fold's caches/RSS without changing deterministic per-series streams.
  for offset in range(0,len(tasks),workers):
   checkpoint()
   with concurrent.futures.ProcessPoolExecutor(workers,mp_context=multiprocessing.get_context('spawn')) as pool:
    futures=[pool.submit(run_job,t) for t in tasks[offset:offset+workers]]
    try:
     for future in concurrent.futures.as_completed(futures):print('Finished',future.result(),flush=True);checkpoint()
    except BaseException:
     pathlib.Path(os.environ['GATE_POWER_STOP']).touch(exist_ok=True)
     for future in futures:future.cancel()
     raise
 assert not phase_tasks(phase);write(ROOT/f'runtime/phase_{phase}_complete.json',dict(status='complete',phase=phase,preregistration_sha256=H(ROOT/'preregistration.json')));journal('phase_complete',phase=phase,df=df())
def other_tasks():
 lines=subprocess.check_output(['ps','-ww','-axo','pid=,ppid=,command='],text=True).splitlines();active=[]
 for line in lines:
  p=line.split(None,2)
  if len(p)!=3:continue
  pid,ppid,cmd=p
  if int(pid) in [os.getpid(),os.getppid()]:continue
  if 'python' in cmd.lower() and any(s in cmd for s in [str(INPUT),str(ROOT.parent)]) and str(ROOT) not in cmd and '/python-env-tools/' not in cmd:active.append(line)
 assert not active,('Another FYP task is running',active)
def verify():
 bootstrap.validate_sources();p=ROOT/'complete.json';seal=J(p);assert seal['status']=='complete' and seal['real_trials']==580 and seal['new_real_trials']==8;assert H(ROOT/'manifest.json')==seal['manifest_sha256'];mf=J(ROOT/'manifest.json')['files']
 for rel,digest in mf.items():assert H(ROOT/rel)==digest,('Gate sealed output changed',rel)
 from real_jobs import done
 for phase in ['R','A','B','T','C']:assert not phase_tasks(phase)
 print('VERIFIED gate_power_v1; new real cells8; cumulative trials580; Q fits0',flush=True);return seal
def final_print():
 audit=J(CORE/'post_tier3_review_v1/disk_audit.json');print(f"Disk audit: df drop{audit['tier23_avail_drop_bytes']/GIB:.6f}GiB; own-run allocation growth{audit['current_minus_baseline_allocated_bytes']/GIB:.6f}GiB; system swap{audit['swap_allocated_bytes']/GIB:.6f}GiB; checkpoint bodies retired now0; df before/after{audit['df_before']['available_bytes']/GIB:.6f}/{audit['df_after']['available_bytes']/GIB:.6f}GiB",flush=True);print((CORE/'tier3_followup.md').read_text(),flush=True)
 summary=J(ROOT/'outputs/summary.json');print('README:',ROOT/'README.md',flush=True)
 for line in summary['lines']:print(line,flush=True)
 print('df now',df()['available_bytes']/GIB,'GiB; stopped. Relocation/slimming remain paused.',flush=True)
def main(args):
 setup();signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop);os.environ['GATE_POWER_STOP']=str(ROOT/f'runtime/STOP_{os.getpid()}')
 if args.workers!='auto':
  assert args.workers.isdecimal() and int(args.workers)>=1,'--workers must be auto or a positive upper bound';os.environ['GATE_WORKER_CAP']=args.workers
 else:os.environ.pop('GATE_WORKER_CAP',None)
 with (ROOT/'runtime/runner.lock').open('a+') as lock,(CORE/'runtime/runner.lock').open('r') as core_lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(core_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);other_tasks();start_gate();budget()
  c,d,a=bootstrap.core_modules();env=c.environment();write(event('environment'),env)
  if args.check_only:print('PREPARATION_CHECK_PASS_NO_REAL_REPLAY_OR_NEW_COMPUTATIONS',flush=True);return
  if (ROOT/'complete.json').exists():verify();final_print();return
  replace(ROOT/'status.json',dict(status='verifying_before_reproduction',real_trials_registered=580))
  # Every invocation, including a resume, gets fresh checks. An earlier pass
  # must never conceal subsequent drift anywhere in the historical scope.
  print('Verifying sealed core, native upstream runs and full historical scope before reproduction/resume.',flush=True)
  p=subprocess.run([PY,'-I','-B',str(CORE/'scripts/verify.py'),'--tier','3'],capture_output=True,text=True);publish(event('core_verify').with_suffix('.log'),p.stdout+p.stderr);assert p.returncode==0,p.stderr[-2000:]
  bootstrap.native_verification('launch');bootstrap.historical('gate_launch_'+str(time.time_ns()));write(event('launch_integrity_pass'),dict(status='PASS',preregistration_sha256=H(ROOT/'preregistration.json')))
  phase('R');assert not phase_tasks('R');print('Exact AL G0/G1 reproduction complete across44 fold groups; new scientific parts may now run.',flush=True)
  phase('A');phase('B')
  import synthetic_jobs
  sizes=synthetic_jobs.size_check();assert all(row['pass_size'] for row in sizes),('Null-size check failed; actual rates retained; no threshold retuning',sizes)
  phase('T');phase('C')
  import report_gate
  lines=report_gate.report();bootstrap.native_verification('final');bootstrap.historical('gate_final_'+str(time.time_ns()));checkpoint()
  files={}
  for folder in ['scripts','tests','vendor','audit','outputs']:
   for p in (ROOT/folder).rglob('*'):
    if p.is_file():files[str(p.relative_to(ROOT))]=H(p)
  for p in ROOT.glob('*.json'):
   if p.name not in ['manifest.json','complete.json','status.json']:files[p.name]=H(p)
  for p in (ROOT/'runtime').rglob('complete.json'):files[str(p.relative_to(ROOT))]=H(p)
  files['README.md']=H(ROOT/'README.md');files['RUN_GATE_POWER.zsh']=H(ROOT/'RUN_GATE_POWER.zsh');write(ROOT/'manifest.json',dict(files=files));write(ROOT/'complete.json',dict(status='complete',manifest_sha256=H(ROOT/'manifest.json'),preregistration_sha256=H(ROOT/'preregistration.json'),real_trials=580,new_real_trials=8,Q_learning_fits=0));replace(ROOT/'status.json',dict(status='complete',real_trials=580));verify();journal('complete_stop',df=df());final_print()
if __name__=='__main__':
 try:
  if len(sys.argv)>1 and sys.argv[1]=='--dispatch':bootstrap.dispatch(pathlib.Path(sys.argv[2]),sys.argv[3:])
  elif len(sys.argv)>1 and sys.argv[1]=='--probe':
   phase_name=sys.argv[2];task=J(ROOT/f'runtime/probe_{phase_name}_intent.json')['task'];row=run_job(task);write(ROOT/f'runtime/probe_{phase_name}.json',row)
  else:
   parser=argparse.ArgumentParser();parser.add_argument('--check-only',action='store_true');parser.add_argument('--verify',action='store_true');parser.add_argument('--workers',default='auto',help='auto by RAM/cores, or a lower positive worker cap');args=parser.parse_args()
   if args.verify:setup();verify()
   else:main(args)
 except BaseException as exc:
  try:journal('paused' if isinstance(exc,(Pause,KeyboardInterrupt)) else 'stopped',reason=str(exc),traceback=traceback.format_exc(),df=df());print('PAUSED/STOPPED; completed groups retained. Resume the same command.',str(exc),flush=True)
  except BaseException:pass
  raise
