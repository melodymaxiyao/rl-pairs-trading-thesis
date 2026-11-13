"""One low-priority resumable accounting command; no strategy trials."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
import fcntl,signal,traceback

def check_code():
 reg=read(RUN/'code_registration.json')
 for rel,digest in reg['files'].items():assert sha(RUN/rel)==digest,('Own frozen code changed',rel)
 assert sha(RUN/'preregistration.json')==reg['preregistration_sha256']

def run():
 check_code();space(start=True);no_other_task();guard();environment()
 with (RUN/'pipeline.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if (RUN/'complete.json').exists():
   from verify import verify
   verify();print('Already complete and verified');return
  os.environ['COST_ACCOUNTING_STOP']=str(RUN/'runtime'/('STOP_'+str(time.time_ns())))
  def stop(signum,frame):
   Path(os.environ['COST_ACCOUNTING_STOP']).touch(exist_ok=True);print('Pause requested: save current accounting segment.',flush=True)
  signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
  from integrity import verify_inputs,historical
  token=str(time.time_ns());verify_inputs('before_'+token,original=True);h=historical('before_'+token)
  write(event('preflight_complete'),dict(status='PASS',historical_checked=h['remaining_checked'],source_seal=spec()['sealed_completion_sha256']))
  from replay import panel,rf_replay,rp_replay
  data=panel();rf=rf_replay(data);rp=rp_replay(data);checkpoint()
  from report import render
  lines=render(rf+rp,data);checkpoint();verify_inputs('after_'+token,original=True)
  if not (RUN/'audit/historical_after.json').exists():historical('after')
  write(RUN/'operation_ledger.json',dict(new_strategy_trials=0,cumulative_trials=552,RL_fits=0,signal_refits=0,PCA_refits=0,OU_refits=0,position_rule_changes=0,descriptive_market_regressions=read(RUN/'summary.json')['descriptive_market_regressions'],downloads=0,source_writes=2,scientific_source_writes=0,authorized_policy_writes=2,bookkeeping_writes=1,bookkeeping_write='exact authorized RUN_LOCATIONS registration',K0_daily_exact_reproduction=True,worlds=['real']))
  paths=[]
  for folder in ['scripts','tests','vendor','outputs','runtime/checkpoints','audit']:
   paths += [p for p in (RUN/folder).rglob('*') if p.is_file() and '.pending_' not in p.name and p.name not in {'audit.json','manifest.json'}]
  paths += [p for folder in (RUN/'runtime').glob('historical_*') for p in folder.rglob('*') if p.is_file() and '.pending_' not in p.name]
  paths += [RUN/name for name in ['preregistration.json','code_registration.json','README.md','request.md','summary.json','cost_accounting.csv','turnover_decomposition.csv','operation_ledger.json','RUN_ALL.zsh','VERIFY.zsh']]
  manifest={str(p.relative_to(RUN)):sha(p) for p in sorted(set(paths))};write(RUN/'manifest.json',dict(files=manifest))
  audit=dict(status='PASS',source_completion_sha256=spec()['sealed_completion_sha256'],manifest_sha256=sha(RUN/'manifest.json'),source_inputs=source().checked,K0_exact=True,new_strategy_trials=0,cumulative_trials=552,RL_fits=0,source_writes=2,scientific_source_writes=0,authorized_policy_writes=2,free_bytes=shutil.disk_usage(RUN).free)
  write(RUN/'audit.json',audit);write(RUN/'complete.json',dict(status='complete',audit_sha256=sha(RUN/'audit.json'),manifest_sha256=sha(RUN/'manifest.json'),preregistration_sha256=sha(RUN/'preregistration.json'),new_strategy_trials=0,cumulative_trials=552))
  from verify import verify
  verify();print('README:',RUN/'README.md',flush=True)
  for line in lines:print(line,flush=True)
if __name__=='__main__':
 try:run()
 except Pause as exc:print('PAUSED:',str(exc),'Resume the same command after space recovers.',flush=True);sys.exit(130)
 except Exception as exc:
  try:write(event('stopped'),dict(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),new_strategy_trials=0,source_writes=0))
  except Exception:pass
  print('STOPPED:',type(exc).__name__,str(exc),flush=True);raise
