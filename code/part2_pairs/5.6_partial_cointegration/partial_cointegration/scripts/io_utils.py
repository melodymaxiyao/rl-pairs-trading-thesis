"""Gate task-only writes, strict no-hydration, shared storage lock and resumable guards."""
import contextlib,fcntl,gzip,hashlib,json,math,os,pathlib,signal,stat,subprocess,sys,threading,time,uuid
ROOT=pathlib.Path('/Users/maxiyao/FYP_runs/partial_coint_kalman_v1');CORE=ROOT.parent/'coint_residual_qlearning_v1';INPUT=pathlib.Path('/Users/maxiyao/Desktop/FYP');GIB=2**30
_PHYSICAL_OPEN=open;_PHYSICAL_LSTAT=os.lstat;_RLOCK=threading.RLock();_STATE=threading.local()
_GUARD_INSTALLED=False
class Pause(RuntimeError):pass
def H(path):
 p=pathlib.Path(path);s=p.lstat();assert stat.S_ISREG(s.st_mode) and not s.st_flags&0x40000000,('Never read unavailable/nonregular file',str(p))
 with p.open('rb') as f:value=hashlib.file_digest(f,'sha256').hexdigest()
 t=p.lstat();assert (s.st_ino,s.st_size,s.st_mtime_ns)==(t.st_ino,t.st_size,t.st_mtime_ns)
 return value
def J(path):H(path);return json.loads(pathlib.Path(path).read_text())
def syncdir(path):
 fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
def df():
 raw=subprocess.check_output(['df','-k',str(ROOT)],text=True);return dict(time_ns=time.time_ns(),available_bytes=int(raw.splitlines()[1].split()[3])*1024,raw=raw)
@contextlib.contextmanager
def storage():
 with _RLOCK:
  depth=getattr(_STATE,'depth',0)
  if depth:
   _STATE.depth=depth+1
   try:yield
   finally:_STATE.depth=depth
   return
  with (ROOT/'runtime/storage.lock').open('a+') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX);_STATE.depth=1
   try:yield
   finally:_STATE.depth=0;fcntl.flock(lock,fcntl.LOCK_UN)
def budget(force=False):
 with storage():
  logical=allocated=0
  for parent,dirs,files in os.walk(ROOT,followlinks=False):
   for name in files:s=(pathlib.Path(parent)/name).lstat();logical+=s.st_size;allocated+=s.st_blocks*512
  if max(logical,allocated)>2*GIB:raise Pause('Gate task2GiB budget reached; journal retained')
  return dict(logical=logical,allocated=allocated,budget=2*GIB)
_LAST_CHECK=0.
def memory_snapshot():
 import re
 raw=subprocess.check_output(['vm_stat'],text=True);page=int(re.search(r'page size of (\d+) bytes',raw)[1]);v={k:int(x) for k,x in re.findall(r'^([^:\n]+):\s+(\d+)\.',raw,re.M)}
 return dict(free=page*v['Pages free'],available=page*(v['Pages free']+v.get('Pages speculative',0)+v.get('Pages inactive',0)),swapouts=v.get('Swapouts',0),raw=raw)
def checkpoint():
 global _LAST_CHECK
 if pathlib.Path(os.environ.get('GATE_POWER_STOP',str(ROOT/'STOP'))).exists():raise Pause('Stop requested; completed groups retained')
 now=time.monotonic()
 if now-_LAST_CHECK<2:return
 _LAST_CHECK=now
 if df()['available_bytes']<int(1.5*GIB):raise Pause('df Avail below1.5GiB; resumable')
 budget()
 if os.environ.get('PARTIAL_RAM_GUARD')=='1':
  m=memory_snapshot()
  if m['available']<2*GIB:raise Pause('Available RAM below2GiB reserve; saved groups retained')
  if m['swapouts']>int(os.environ['PARTIAL_INITIAL_SWAPOUTS']):raise Pause('New system swap-outs observed; pause rather than increase memory pressure')
def clean(value):
 import numpy as np
 if isinstance(value,np.ndarray):return clean(value.tolist())
 if isinstance(value,np.generic):return clean(value.item())
 if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
 if isinstance(value,(tuple,list)):return [clean(v) for v in value]
 if isinstance(value,float) and not math.isfinite(value):return None
 if isinstance(value,pathlib.Path):return str(value)
 return value
def encode(value):return (json.dumps(clean(value),sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def publish(path,raw):
 p=pathlib.Path(path);assert p.resolve().is_relative_to(ROOT);raw=raw.encode() if isinstance(raw,str) else raw
 with storage():
  if p.exists():assert p.read_bytes()==raw,('Immutable task artifact differs',str(p));return
  b=budget();minimum=int(1.5*GIB)
  if len(raw)>2**20 and df()['available_bytes']-len(raw)<minimum:raise Pause('Publication would cross1.5GiB floor')
  if max(b['logical'],b['allocated'])+len(raw)>2*GIB:raise Pause('Publication would cross2GiB task budget')
  p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_name(p.name+'.pending_'+uuid.uuid4().hex)
  with temp.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
  try:os.link(temp,p)
  except FileExistsError:assert p.read_bytes()==raw
  finally:temp.unlink()
  syncdir(p.parent)
def write(path,value):publish(path,encode(value))
def replace(path,value):
 p=pathlib.Path(path);raw=encode(value)
 with storage():
  b=budget();assert p.resolve().is_relative_to(ROOT)
  if max(b['logical'],b['allocated'])+len(raw)>2*GIB:raise Pause('Atomic control write exceeds2GiB')
  p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_name(p.name+'.pending_'+uuid.uuid4().hex)
  with temp.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
  os.replace(temp,p);syncdir(p.parent)
def event(prefix):return ROOT/'audit'/f'{prefix}_{time.time_ns()}_{uuid.uuid4().hex}.json'
def journal(action,**kw):
 with storage():
  with (ROOT/'runtime/journal.jsonl').open('a') as f:f.write(json.dumps(clean(dict(time_ns=time.time_ns(),pid=os.getpid(),action=action,**kw)),sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
  syncdir(ROOT/'runtime')
def count(name,n=1):
 with (ROOT/'runtime'/f'counts_{os.getpid()}.jsonl').open('a') as f:f.write(json.dumps([name,n])+'\n')
def install_guard():
 global _GUARD_INSTALLED
 if _GUARD_INSTALLED:return
 _GUARD_INSTALLED=True
 ignored=INPUT/'experiments/15_pair_trading/trend_anchor_td_tuned_v1/audit/repairs/protected_diff_v3/finder_snapshot_1792731898004837000 2.json'
 def allowed(path):
  p=pathlib.Path(os.fsdecode(path)).absolute().resolve();return p==pathlib.Path(os.devnull) or p.is_relative_to(ROOT) or p.is_relative_to(CORE/'prompt_controls/state')
 def hook(event,args):
  if event.startswith('socket.') or event=='os.system':raise PermissionError('No network/downloads or shell commands')
  if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
   p=pathlib.Path(os.fsdecode(args[0])).absolute();assert p!=ignored,'Never read ignored conflict copy'
   try:s=p.lstat()
   except FileNotFoundError:s=None
   assert not s or not s.st_flags&0x40000000,('Never hydrate dataless content',str(p))
   mode,flags=args[1:3];writing=isinstance(mode,str) and any(c in mode for c in 'wax+') or isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)
   if writing:assert allowed(p),('External mutation denied',str(p))
  if event in ['os.scandir','os.listdir'] and isinstance(args[0],(str,bytes,os.PathLike)):assert not pathlib.Path(args[0]).lstat().st_flags&0x40000000
  if event in ['os.remove','os.rmdir','os.mkdir','os.chmod','os.chflags','os.utime'] and isinstance(args[0],(str,bytes,os.PathLike)):assert allowed(args[0]),('External mutation denied',event,args[0])
  if event in ['os.rename','os.link']:
   for p in args[:2]:
    if isinstance(p,(str,bytes,os.PathLike)):assert allowed(p)
  if event=='os.symlink':raise PermissionError('No new symlinks')
 sys.addaudithook(hook)
def stop(signum,frame):
 pathlib.Path(os.environ['GATE_POWER_STOP']).touch(exist_ok=True);print('Pause requested; finishing current saved group.',flush=True)
