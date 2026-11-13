"""Single-process continuation, checkpoint reuse, tests, time/disk gates and final reports."""
import os,sys,fcntl,time,runpy,hashlib,json,datetime,shutil
from pathlib import Path
P=Path(__file__).resolve().parents[1]
os.environ.update(PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',TMPDIR=str(P/'runtime'),MPLCONFIGDIR=str(P/'runtime/mpl'),XDG_CACHE_HOME=str(P/'runtime/cache'))
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
(P/'runtime').mkdir(exist_ok=True)
with (P/'runtime/trading.lock').open('a+')as lock:
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise SystemExit('This task is already running; no duplicate execution started.')
 registration=json.loads((P/'audit/code_registration_trading.json').read_text())
 for rel,digest in registration['files'].items():
  p=P/rel;s=p.lstat();assert not s.st_flags&0x40000000;assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,('Task code changed',rel)
 assert shutil.disk_usage(P).free>=2*1024**3,'Free disk below2GiB'
 # No simultaneous other FYP heavy task. Ignore this launcher's ancestry and terminal followers.
 import subprocess
 other=[]
 for line in subprocess.check_output(['ps','-axo','pid=,command='],text=True).splitlines():
  q=line.strip().split(None,1)
  if len(q)!=2 or int(q[0])==os.getpid():continue
  command=q[1]
  if 'python' in command.lower()and '/Users/maxiyao/FYP_runs/'in command and str(P)not in command and 'exec_command'not in command:other.append(line)
 assert not other,('Another FYP Python task is running',other)
 print('START SINGLE PROCESS',datetime.datetime.now(datetime.timezone.utc).isoformat(),'cached pilot folds will be reused',flush=True)
 import run_trading as rt
 def budget():
  done=[json.loads(p.read_text())for p in (P/'folds').glob('*.json')]
  expected=0
  for family in ['RP','RF']:
   values=[r['seconds']for r in done if r['family']==family]
   if values:expected+=22*sum(values[-3:])/len(values[-3:])
  assert expected<=4*3600,('Estimated full runtime exceeds4hours; stop for user decision',expected)
  return len(done),expected
 for year in range(2004,2026):
  for family in ['RP','RF']:
   n,expected=budget();print('PROGRESS',n,'/44','estimated_total_minutes',round(expected/60,1),flush=True);rt.one(family,year)
 print('ALL44FOLDS_DONE; aggregating registered year blocks',flush=True)
 runpy.run_path(str(P/'scripts/aggregate.py'),run_name='__main__')
 runpy.run_path(str(P/'scripts/final_report.py'),run_name='__main__')
 print('COMPLETE',P/'README.md',flush=True)
