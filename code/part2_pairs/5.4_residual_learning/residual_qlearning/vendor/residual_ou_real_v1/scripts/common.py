"""External-run IO, immutable checkpoints, explicit integrity and operation logs."""
import os,sys,time,json,hashlib,gzip,pickle,tempfile,stat,shutil,fcntl,uuid
from pathlib import Path
from contextlib import contextmanager
import numpy as np
RUN=Path(__file__).resolve().parents[1];INPUT=Path('/Users/maxiyao/Desktop/FYP')
TABLES=RUN/'outputs/tables';PLOTS=RUN/'outputs/plots';DATALESS=0x40000000
sys.dont_write_bytecode=True
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
os.environ.update(PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',MPLBACKEND='Agg',MPLCONFIGDIR=str(RUN/'runtime/mpl'),XDG_CACHE_HOME=str(RUN/'runtime/cache'),TMPDIR=str(RUN/'runtime/tmp'))
tempfile.tempdir=str(RUN/'runtime/tmp')
class Pause(RuntimeError):pass
_IGNORED_CONFLICT_READS=set()
def deny_conflict_read(path):_IGNORED_CONFLICT_READS.add(os.path.abspath(os.fspath(path)))
def spec():return json.loads((RUN/'preregistration.json').read_text())
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple,np.ndarray)):return [clean(v) for v in x]
 if isinstance(x,(np.bool_,bool)):return bool(x)
 if isinstance(x,(np.integer,int)):return int(x)
 if isinstance(x,(np.floating,float)):return float(x) if np.isfinite(x) else None
 if isinstance(x,Path):return str(x)
 return x
def sha(p):
 p=Path(p);s=p.lstat();assert not s.st_flags&DATALESS,('Dataless read prohibited',str(p))
 if p.is_symlink():assert not p.stat().st_flags&DATALESS
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def inside(p):
 p=Path(p).absolute();assert p.resolve().is_relative_to(RUN),('Write outside own run',str(p));return p
def event_path(prefix):
 # Clock readings are labels, not unique IDs: different workers can share one.
 return RUN/f'audit/{prefix}_{time.time_ns()}_{os.getpid()}_{uuid.uuid4().hex}.json'
def publish(p,data):
 p=inside(p);p.parent.mkdir(parents=True,exist_ok=True);data=data.encode() if isinstance(data,str) else data
 if p.exists():assert p.read_bytes()==data,('Existing artifact differs; not overwritten',str(p));return
 pending=p.with_name(p.name+f'.pending_{os.getpid()}_{uuid.uuid4().hex}')
 with pending.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 try:
  try:os.link(pending,p)
  except FileExistsError:assert p.read_bytes()==data,('Existing artifact differs; not overwritten',str(p))
 finally:pending.unlink() # Only this call's unpublished temporary file.
def write(p,v):publish(p,json.dumps(clean(v),sort_keys=True,indent=2,allow_nan=False)+'\n')
def count(operation,n=1):
 assert isinstance(n,int) and n>=0
 with (RUN/f'audit/counts_{os.environ.get("ROU_SCOPE","preparation")}_{os.getpid()}.jsonl').open('a') as f:f.write(json.dumps([operation,n])+'\n')
def ledger():
 result={};scopes={}
 for p in sorted((RUN/'audit').glob('counts_*.jsonl')):
  for line in p.read_text().splitlines():
   key,n=json.loads(line);result[key]=result.get(key,0)+n;scope=p.stem[len('counts_'):].rsplit('_',1)[0];group=scopes.setdefault(scope,{});group[key]=group.get(key,0)+n
 return dict(operations=result,scopes=scopes,downloads=0,package_installs=0,new_real_strategy_trials=0,real_data_trials=0)
def scientific_hash(value):
 # Scientific identity must not depend on pickle's object-alias memo or array ownership.
 def canonical(v):
  if isinstance(v,np.ndarray):return dict(array_dtype=v.dtype.str,shape=list(v.shape),data_sha256=hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest())
  if isinstance(v,dict):return {str(k):canonical(x) for k,x in v.items()}
  if isinstance(v,(tuple,list)):return [canonical(x) for x in v]
  return clean(v)
 return hashlib.sha256(json.dumps(canonical(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def save(stem,value):
 raw=pickle.dumps(value,protocol=5);envelope=dict(payload=raw,sha256=hashlib.sha256(raw).hexdigest(),preregistration_sha256=sha(RUN/'preregistration.json'))
 p=RUN/(stem+'.pkl.gz');publish(p,gzip.compress(pickle.dumps(envelope,protocol=5),mtime=0));write(RUN/(stem+'.ready.json'),dict(file=str(p.relative_to(RUN)),sha256=sha(p)));return value
def cached(stem):
 p=RUN/(stem+'.pkl.gz');r=RUN/(stem+'.ready.json')
 if not p.exists():assert not r.exists();return None
 if r.exists():assert sha(p)==json.loads(r.read_text())['sha256']
 with gzip.open(p,'rb') as f:v=pickle.load(f)
 assert v['preregistration_sha256']==sha(RUN/'preregistration.json') and hashlib.sha256(v['payload']).hexdigest()==v['sha256']
 if not r.exists():write(r,dict(file=str(p.relative_to(RUN)),sha256=sha(p)))
 return pickle.loads(v['payload'])
def checkpoint_guard():
 if shutil.disk_usage(RUN).free<5*2**30:raise Pause('Free space below5GiB; checkpoint retained')
 if os.environ.get('ROU_STOP_FILE') and Path(os.environ['ROU_STOP_FILE']).exists():raise Pause('Stop requested at checkpoint')
def stable_seed(*parts):
 words=np.frombuffer(hashlib.sha256(json.dumps([20261027,*parts],separators=(',',':')).encode()).digest(),dtype='<u4')
 return int(np.random.SeedSequence(words).generate_state(1,dtype=np.uint64)[0])
def td_seed(j):return int(np.random.SeedSequence([20261020,1,j,1,0]).generate_state(1,dtype=np.uint64)[0])
@contextmanager
def lock():
 with (RUN/'pipeline.lock').open('a+') as f:
  fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);yield
_GUARD_INSTALLED=False
_READONLY_GUARD=False
def guard(readonly=False):
 global _GUARD_INSTALLED,_READONLY_GUARD
 _READONLY_GUARD=_READONLY_GUARD or readonly
 if _GUARD_INSTALLED:return
 _GUARD_INSTALLED=True
 def at_path(p,fd=None):
  q=Path(os.fsdecode(p))
  if not q.is_absolute() and isinstance(fd,int) and fd>=0:
   base=fcntl.fcntl(fd,50,b'\0'*1024).split(b'\0',1)[0] # Darwin F_GETPATH
   q=Path(os.fsdecode(base))/q
  return q
 def hook(event,args):
  if event in ['socket.connect','socket.getaddrinfo']:raise PermissionError('No network/downloads')
  writes=[]
  if event in ['os.scandir','os.listdir'] and isinstance(args[0],(str,bytes,os.PathLike)):
   q=Path(os.fsdecode(args[0]));assert not q.lstat().st_flags&DATALESS,('No dataless directory expansion',str(q))
  if event=='open':
   p,mode,flags=args
   if isinstance(p,(str,bytes,os.PathLike)):
    q=Path(os.fsdecode(p)).absolute()
    if os.path.abspath(os.fspath(q)) in _IGNORED_CONFLICT_READS:raise PermissionError('Ignored iCloud conflict copy must never be opened: '+str(q))
    try:s=q.lstat()
    except OSError:s=None
    if s is not None and s.st_flags&DATALESS:raise PermissionError('Dataless content prohibited: '+str(q))
    if s is not None and stat.S_ISLNK(s.st_mode) and q.stat().st_flags&DATALESS:raise PermissionError('Dataless symlink target prohibited: '+str(q))
   if isinstance(mode,str) and any(c in mode for c in 'wax+') or isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):writes=[(p,None)]
  elif event in ['os.mkdir','os.remove','os.rmdir','os.chmod','os.chflags','os.utime']:
   index={'os.mkdir':2,'os.remove':1,'os.rmdir':1,'os.chmod':2,'os.utime':3}.get(event);writes=[(args[0],args[index] if index is not None and len(args)>index else None)]
  elif event in ['os.rename','os.link']:writes=[(args[0],args[2]),(args[1],args[3])]
  elif event=='os.symlink':writes=[(args[1],args[2])]
  for p,fd in writes:
   if isinstance(p,(str,bytes,os.PathLike)) and os.fsdecode(p)!=os.devnull:
    assert not _READONLY_GUARD,('Read-only verifier',str(p));inside(at_path(p,fd))
 sys.addaudithook(hook)
