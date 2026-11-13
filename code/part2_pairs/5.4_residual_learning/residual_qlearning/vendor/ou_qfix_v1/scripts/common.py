"""Own-run IO and authenticated sealed inputs; no writes into source runs."""
import os,sys,time,json,hashlib,gzip,pickle,stat,shutil,tempfile,subprocess,csv,uuid,fcntl
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
RUN=Path(__file__).resolve().parents[1];INPUT=Path('/Users/maxiyao/Desktop/FYP');SOURCE=Path('/Users/maxiyao/FYP_runs/residual_ou_real_v1');DATALESS=0x40000000
sys.dont_write_bytecode=True
os.environ.update(PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',MPLBACKEND='Agg',TMPDIR=str(RUN/'runtime/tmp'),XDG_CACHE_HOME=str(RUN/'runtime/cache'),MPLCONFIGDIR=str(RUN/'runtime/mpl'));tempfile.tempdir=str(RUN/'runtime/tmp')
def sha(p):
 p=Path(p);s=p.lstat();assert stat.S_ISREG(s.st_mode) and not s.st_flags&DATALESS,('Unavailable or non-regular input',str(p))
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def meta(p):
 s=Path(p).lstat();return dict(size=s.st_size,mode=stat.S_IMODE(s.st_mode),mtime_ns=s.st_mtime_ns,ino=s.st_ino,flags=s.st_flags)
def read(p):
 p=Path(p);assert not p.lstat().st_flags&DATALESS;return json.loads(p.read_text())
def spec():return read(RUN/('preregistration.json' if (RUN/'preregistration.json').exists() else 'preregistration.DRAFT.json'))
class Pause(RuntimeError):pass
def space(start=False):
 minimum=(8 if start else 5)*2**30
 if shutil.disk_usage(RUN).free<minimum:raise Pause(f'Free space below{minimum//2**30}GiB; saved synthetic checkpoints retained')
 return shutil.disk_usage(RUN).free
def no_other_task():
 active=[]
 for line in subprocess.check_output(['ps','-ww','-axo','pid=,ppid=,command='],text=True).splitlines():
  fields=line.split(None,2)
  if len(fields)!=3:continue
  pid,ppid,cmd=fields
  if int(pid) in {os.getpid(),os.getppid()}:continue
  if 'python' in cmd.lower() and any(x in cmd for x in [str(INPUT),str(RUN.parent),'experiments/15_pair_trading/']):active.append(line)
 assert not active,('Other FYP process',active)
def checkpoint():
 space()
 budget_guard()
 if Path(os.environ.get('OU_QFIX_STOP',str(RUN/'STOP'))).exists():raise Pause('Stop requested; saved synthetic checkpoints retained')
def clean(v):
 import numpy as np
 if isinstance(v,np.ndarray):return clean(v.tolist())
 if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [clean(x) for x in v]
 if isinstance(v,np.generic):return clean(v.item())
 if isinstance(v,float):return v if np.isfinite(v) else None
 if isinstance(v,Path):return str(v)
 return v
def publish(path,raw):
 path=Path(path);assert path.resolve().is_relative_to(RUN);raw=raw.encode() if isinstance(raw,str) else raw
 if path.exists():assert path.read_bytes()==raw,('Immutable artifact differs',str(path));return
 path.parent.mkdir(exist_ok=True,parents=True);pending=path.with_name(path.name+'.pending_'+uuid.uuid4().hex)
 with pending.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 try:os.link(pending,path)
 except FileExistsError:assert path.read_bytes()==raw
 finally:pending.unlink()
def write(path,v):publish(path,json.dumps(clean(v),sort_keys=True,indent=2,allow_nan=False)+'\n')
def event(prefix):return RUN/'audit'/f'{prefix}_{time.time_ns()}_{uuid.uuid4().hex}.json'
def save(stem,obj):
 raw=pickle.dumps(obj,protocol=5);envelope=dict(payload=raw,sha256=hashlib.sha256(raw).hexdigest(),preregistration_sha256=sha(RUN/'preregistration.json'))
 path=RUN/(stem+'.pkl.gz');publish(path,gzip.compress(pickle.dumps(envelope,protocol=5),mtime=0));write(RUN/(stem+'.ready.json'),dict(sha256=sha(path)));return obj
def cached(stem):
 path=RUN/(stem+'.pkl.gz')
 if not path.exists():return None
 ready=RUN/(stem+'.ready.json')
 if ready.exists():assert sha(path)==read(ready)['sha256']
 with gzip.open(path,'rb') as f:v=pickle.load(f)
 assert v['preregistration_sha256']==sha(RUN/'preregistration.json') and hashlib.sha256(v['payload']).hexdigest()==v['sha256']
 if not ready.exists():write(ready,dict(sha256=sha(path)))
 return pickle.loads(v['payload'])
_SOURCE=None
class Sealed:
 def __init__(self):
  p=read(RUN/'audit/input_seals.json')['residual'];self.complete=read(SOURCE/'complete.json');assert sha(SOURCE/'complete.json')==p['sealed_completion_sha256'] and self.complete['cumulative_trials']==552
  assert sha(SOURCE/'audit.json')==self.complete['audit_sha256']==p['sealed_audit_sha256'];self.audit=read(SOURCE/'audit.json');self.registration=read(SOURCE/'code_registration.json');assert sha(SOURCE/'code_registration.json')==self.audit['code_registration_sha256']==p['sealed_code_registration_sha256']
  self.expected=dict(self.audit['artifacts'])|self.registration['files'];self.checked={};assert sha(SOURCE/'preregistration.json')==self.registration['preregistration_sha256'];self.prereg=read(SOURCE/'preregistration.json')
 def verify(self,rel):
  assert not Path(rel).is_absolute() and '..' not in Path(rel).parts and rel in self.expected,('Unsealed input',rel)
  p=SOURCE/rel;m=meta(p);old=self.checked.get(rel)
  if old and old['metadata']==m:return p
  assert sha(p)==self.expected[rel],('Sealed input hash mismatch',rel)
  if rel in self.registration['metadata']:assert m==self.registration['metadata'][rel],('Sealed code metadata drift',rel)
  self.checked[rel]=dict(sha256=self.expected[rel],metadata=m);return p
 def cache(self,stem):
  path=self.verify(stem+'.pkl.gz');ready=self.verify(stem+'.ready.json');assert sha(path)==read(ready)['sha256']
  with gzip.open(path,'rb') as f:v=pickle.load(f)
  assert v['preregistration_sha256']==self.registration['preregistration_sha256'] and hashlib.sha256(v['payload']).hexdigest()==v['sha256'];return pickle.loads(v['payload'])
def source():
 global _SOURCE
 if _SOURCE is None:_SOURCE=Sealed()
 return _SOURCE
def environment():
 import importlib.metadata as md,platform
 src=source();p=src.verify('audit/pip_freeze_all.txt');expected=dict(line.split('==',1) for line in p.read_text().splitlines() if line)
 canonical=lambda d:{k.lower().replace('_','-'):v for k,v in d.items()}
 actual={d.metadata['Name']:d.version for d in md.distributions() if d.metadata['Name']}
 assert canonical(actual)==canonical(expected),'Environment mismatch; no package changes allowed'
 assert sys.executable==str(INPUT/'.venv/bin/python') and platform.python_version()=='3.13.2'
 return dict(executable=sys.executable,python=platform.python_version(),packages=len(actual),freeze_sha256=sha(p))
_IGNORED={str(INPUT/'experiments/15_pair_trading/trend_anchor_td_tuned_v1/audit/repairs/protected_diff_v3/finder_snapshot_1792731898004837000 2.json')}
_GUARD_INSTALLED=False
def guard():
 global _GUARD_INSTALLED
 if _GUARD_INSTALLED:return
 _GUARD_INSTALLED=True
 def at_path(p,fd=None):
  q=Path(os.fsdecode(p))
  if not q.is_absolute() and isinstance(fd,int) and fd>=0:
   parent=fcntl.fcntl(fd,50,b'\0'*1024).split(b'\0',1)[0];q=Path(os.fsdecode(parent))/q
  return q.absolute()
 def hook(event,args):
  if event in ['socket.connect','socket.getaddrinfo']:raise PermissionError('No network/downloads')
  paths=[]
  if event=='open':
   p,mode,flags=args
   if isinstance(p,(str,bytes,os.PathLike)):
    q=Path(os.fsdecode(p)).absolute();assert str(q) not in _IGNORED,'Never open ignored conflict copy'
    try:s=q.lstat()
    except FileNotFoundError:s=None
    assert not s or not s.st_flags&DATALESS,('No dataless read',str(q))
   if isinstance(mode,str) and any(c in mode for c in 'wax+') or isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):paths=[(p,None)]
  elif event in ['os.scandir','os.listdir'] and isinstance(args[0],(str,bytes,os.PathLike)):assert not Path(args[0]).lstat().st_flags&DATALESS
  elif event in ['os.rename','os.link']:paths=[(args[0],args[2]),(args[1],args[3])]
  elif event=='os.symlink':paths=[(args[1],args[2])]
  elif event in ['os.mkdir','os.remove','os.rmdir','os.utime','os.chmod','os.chflags']:
   k={'os.mkdir':2,'os.remove':1,'os.rmdir':1,'os.utime':3,'os.chmod':2}.get(event);paths=[(args[0],args[k] if k is not None and len(args)>k else None)]
  for p,fd in paths:
   if isinstance(p,(str,bytes,os.PathLike)) and os.fsdecode(p)!=os.devnull:assert at_path(p,fd).resolve().is_relative_to(RUN),('External mutation forbidden',str(p))
 sys.addaudithook(hook)

import numpy as np
from collections import Counter
COUNTS=Counter()
def count(operation,n=1):
 assert isinstance(n,int) and n>=0
 scope=os.environ.get('OU_SCOPE','preparation')
 with (RUN/'audit'/('counts_'+scope+'_'+str(os.getpid())+'.jsonl')).open('a') as f:f.write(json.dumps([operation,n])+'\n')
def checkpoint_guard():checkpoint()

def stable_seed(*parts):
 words=np.frombuffer(hashlib.sha256(json.dumps([20261027,*parts],separators=(',',':')).encode()).digest(),dtype='<u4')
 return int(np.random.SeedSequence(words).generate_state(1,dtype=np.uint64)[0])

def td_seed(j):return int(np.random.SeedSequence([20261020,1,j,1,0]).generate_state(1,dtype=np.uint64)[0])

_BUDGET_TIME=0
def budget_guard(force=False):
 global _BUDGET_TIME
 if not force and time.monotonic()-_BUDGET_TIME<10:return
 _BUDGET_TIME=time.monotonic();logical=allocated=0
 for parent,dirs,files in os.walk(RUN,followlinks=False):
  for name in files:
   p=Path(parent)/name;s=p.lstat();logical+=s.st_size;allocated+=s.st_blocks*512
 if max(logical,allocated)>int(1.5*2**30):raise Pause('Own-run1.5GiB disk budget reached; checkpoints retained')
 return dict(logical=logical,allocated=allocated,budget=int(1.5*2**30))

class SyntheticSeal:
 def __init__(self):
  self.anchor=read(RUN/'audit/source_seal.json');self.root=Path(self.anchor['root'])
  for name,key in [('complete.json','complete_sha256'),('audit.json','audit_sha256'),('manifest.json','manifest_sha256'),('code_registration.json','code_registration_sha256'),('preregistration.json','preregistration_sha256')]:assert sha(self.root/name)==self.anchor[key],('Source anchor changed',name)
  self.expected=read(self.root/'manifest.json')['files']|read(self.root/'code_registration.json')['files']
  self.checked={}
 def verify(self,rel):
  assert not Path(rel).is_absolute() and '..' not in Path(rel).parts and rel in self.expected,('Unsealed synthetic input',rel)
  p=self.root/rel;assert sha(p)==self.expected[rel],('Synthetic input changed',rel)
  self.checked[rel]=dict(sha256=self.expected[rel],metadata=meta(p));return p
 def cache(self,stem):
  p=self.verify(stem+'.pkl.gz');ready=self.verify(stem+'.ready.json');assert sha(p)==read(ready)['sha256']
  with gzip.open(p,'rb') as f:e=pickle.load(f)
  assert e['preregistration_sha256']==self.anchor['preregistration_sha256'] and hashlib.sha256(e['payload']).hexdigest()==e['sha256'];return pickle.loads(e['payload'])
_SYNTHETIC=None
def synthetic():
 global _SYNTHETIC
 if _SYNTHETIC is None:_SYNTHETIC=SyntheticSeal()
 return _SYNTHETIC
