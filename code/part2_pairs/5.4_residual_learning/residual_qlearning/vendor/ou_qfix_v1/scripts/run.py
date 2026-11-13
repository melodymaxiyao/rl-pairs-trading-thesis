"""One low-priority command: prove frozen baselines, run fixes, report, seal."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
import argparse,signal,traceback
from integrity import inputs,check_code
def historical(label):
 p=subprocess.run([str(INPUT/'.venv/bin/python'),'-I','-B',str(RUN/'scripts/historical_entry.py'),label])
 if p.returncode==130:raise Pause('Historical disk/stop guard; checkpoints retained')
 assert p.returncode==0,('Historical checker failed',p.returncode)
 assert (RUN/'audit'/('historical_'+label+'.json')).exists()
def discard_checkpoint_bodies():
 complete=read(RUN/'complete.json');assert sha(RUN/'manifest.json')==complete['manifest_sha256']
 manifest=read(RUN/'manifest.json')['files'];rel='receipts/checkpoint_discard.json';assert sha(RUN/rel)==manifest[rel];plan=read(RUN/rel)
 for name,row in plan['files'].items():
  p=RUN/name;assert p.is_relative_to(RUN/'runtime/checkpoints') and p.suffixes[-2:]==['.pkl','.gz']
  if p.exists():assert sha(p)==row['sha256'];p.unlink()
def seal(lines):
 files={}
 for p in sorted((RUN/'runtime/checkpoints').glob('*.pkl.gz')):
  rel=str(p.relative_to(RUN));ready=p.with_suffix('').with_suffix('.ready.json');assert ready.exists() and read(ready)['sha256']==sha(p);files[rel]=dict(sha256=sha(p),size=p.stat().st_size,ready=str(ready.relative_to(RUN)))
 write(RUN/'receipts/checkpoint_discard.json',dict(authorization='User task: checkpoint bodies deleted after sealing; receipts kept',preregistration_sha256=sha(RUN/'preregistration.json'),files=files,scientific_exports='outputs/models contains seed tables/uncertainty/stopping data; cells/q_bias/entries/policies preserve final science; no raw episodes ever saved'))
 production=sorted((RUN/'runtime/checkpoints').glob('new_world*_c*_seed*.ready.json'));assert len(production)==400
 write(RUN/'operation_ledger.json',dict(real_trials=552,new_real_trials=0,source_writes=0,bookkeeping_writes=1,downloads=0,package_changes=0,production_Double_seed_fits=len(production),new_Q_POW_fits=0,LCB_reuses_POW=True,DBL_LCB_reuses_DBL=True,episode_arrays_persisted=0))
 paths=[]
 for p in RUN.rglob('*'):
  if not p.is_file():continue
  rel=str(p.relative_to(RUN))
  if rel in ['complete.json','audit.json','manifest.json','pipeline.lock'] or '.pending_' in p.name or p.name.startswith('STOP_'):continue
  if any(part in ['tmp','cache','mpl'] for part in p.relative_to(RUN).parts):continue
  if rel in files:continue
  paths.append(p)
 write(RUN/'manifest.json',dict(files={str(p.relative_to(RUN)):sha(p) for p in sorted(paths)},discarded_checkpoint_manifest='receipts/checkpoint_discard.json'))
 write(RUN/'audit.json',dict(status='PASS_WITH_REGISTERED_CONDITIONALS',manifest_sha256=sha(RUN/'manifest.json'),code_registration_sha256=sha(RUN/'code_registration.json'),preregistration_sha256=sha(RUN/'preregistration.json'),real_trials=552,new_real_trials=0,standing_exact_paths=8908,disk=budget_guard(True)))
 write(RUN/'complete.json',dict(status='complete',audit_sha256=sha(RUN/'audit.json'),manifest_sha256=sha(RUN/'manifest.json'),real_trials=552,new_real_trials=0))
 from verify import verify
 verify();discard_checkpoint_bodies();verify();print('README:',RUN/'README.md',flush=True)
 for line in lines:print(line,flush=True)
def run(args):
 check_code();no_other_task();guard()
 with (RUN/'pipeline.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if (RUN/'complete.json').exists():
   from verify import verify
   verify();discard_checkpoint_bodies();verify();print('README:',RUN/'README.md')
   for line in read(RUN/'summary.json')['lines']:print(line)
   return
  space(start=True);os.environ['OU_QFIX_STOP']=str(RUN/'runtime'/('STOP_'+str(time.time_ns())))
  def stop(signum,frame):Path(os.environ['OU_QFIX_STOP']).touch(exist_ok=True);print('Pause requested; saving current seed/panel.',flush=True)
  signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
  token=str(time.time_ns());inputs('before_'+token);historical('before_'+token);checkpoint()
  if args.preflight_only:print('Preflight PASS; no production started.');return
  from reproduction import reproduce_all
  reproduce_all(args.workers);checkpoint()
  from learn import fit_world_cost
  argslist=[(w,c) for w in range(20) for c in [5,10]]
  if args.workers>1:
   import multiprocessing as mp
   from concurrent.futures import ProcessPoolExecutor
   with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn')) as pool:results=list(pool.map(fit_world_cost,argslist,chunksize=1))
  else:results=[fit_world_cost(a) for a in argslist]
  from evaluate import evaluate
  from report import render
  result=evaluate(results);lines=render(result);checkpoint();inputs('after_'+token);historical('after_'+token);checkpoint();seal(lines)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,choices=[1,2],default=1);parser.add_argument('--preflight-only',action='store_true');args=parser.parse_args()
 try:run(args)
 except Pause as exc:print('PAUSED:',str(exc),'Saved jobs retained; resume the same command.',flush=True);sys.exit(130)
 except Exception as exc:
  try:write(event('stopped'),dict(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),source_writes=0,new_real_trials=0))
  except Exception:pass
  print('STOPPED:',type(exc).__name__,str(exc),flush=True);raise
