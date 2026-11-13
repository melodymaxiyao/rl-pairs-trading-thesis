"""Task-local immutable checkpoints, read-only inputs, and measured accounting."""
import os,sys,json,time,hashlib,gzip,pickle,fcntl,tempfile,types,importlib.util,io,stat,shutil
from pathlib import Path
from contextlib import contextmanager
from functools import lru_cache
from collections import Counter,OrderedDict
sys.dont_write_bytecode=True;sys.modules['readline']=None
HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parents[2];PT=HERE.parent
SOURCE=BASE=PT/'trend_anchor_trading_v1';TUNED=PT/'trend_anchor_td_tuned_v1';MECH=PT/'td_mechanism_spi_v1'
SYN=PT/'td_sample_complexity_v1';THESIS=PT/'thesis_tables_v1';COMPACT=PT/'storage_compaction_v1';PRUNE=PT/'tav1_runtime_prune_v1'
TABLES=HERE/'outputs/tables';MODELS=HERE/'outputs/models';PLOTS=FIGURES=HERE/'outputs/plots'
YEARS=list(range(2004,2026));POOLS=['P_RAND','P_EXP','P_QUAL'];COSTS=[5,10,20]
WORLDS=[('real',0),('N1',0),('N1',1),('N2',0),('N2',1)]
PINNED={};COUNTS=Counter();HANDLES={};HASHING=False;INTEGRITY=os.environ.get('RMS_INTEGRITY')=='1'
FORBIDDEN=[SOURCE/'runtime/contexts',SOURCE/'runtime/cells']
DATALESS=0x40000000
def configure():
 for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
 os.environ.update(PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',MPLBACKEND='Agg',TMPDIR=str(HERE/'runtime/tmp'),SQLITE_TMPDIR=str(HERE/'runtime/tmp'),MPLCONFIGDIR=str(HERE/'runtime/mpl'),XDG_CACHE_HOME=str(HERE/'runtime/cache'))
 tempfile.tempdir=str(HERE/'runtime/tmp')
configure()
def read(p):return json.loads(Path(p).read_text())
def spec():return read(HERE/'preregistration.json')
def environment_status():
 import importlib.metadata
 frozen=read(SOURCE/'audit/code_registration.json');expected=frozen['package_versions']
 current={name:importlib.metadata.version(name) for name in expected}
 differences={name:dict(sealed=version,current=current[name]) for name,version in expected.items() if current[name]!=version}
 if sys.version.split()[0]!=frozen['python_version']:differences['python']=dict(sealed=frozen['python_version'],current=sys.version.split()[0])
 if Path(sys.executable).absolute()!=ROOT/'.venv/bin/python' or Path(sys.prefix)!=ROOT/'.venv':differences['interpreter']=dict(required=str(ROOT/'.venv/bin/python'),current=sys.executable,prefix=sys.prefix)
 return dict(status='PASS' if not differences else 'BLOCKED_ENVIRONMENT_DRIFT',expected=expected,current=current,differences=differences)
def check_environment():
 result=environment_status();assert not result['differences'],('Sealed environment changed; no packages modified',result['differences']);return result
def inside(p):
 p=Path(p).absolute();assert p.resolve().is_relative_to(HERE),('External write prohibited',p);return p
def sha(p):
 global HASHING
 p=Path(p);assert not p.lstat().st_flags&DATALESS,('Dataless content read prohibited',p)
 before=HASHING;HASHING=True
 try:
  with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
 finally:HASHING=before
def clean(v):
 import numpy as np
 if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
 if isinstance(v,(list,tuple,np.ndarray)):return [clean(x) for x in v]
 if isinstance(v,(np.bool_,bool)):return bool(v)
 if isinstance(v,(np.integer,int)):return int(v)
 if isinstance(v,(np.floating,float)):return float(v) if np.isfinite(v) else None
 if isinstance(v,Path):return str(v)
 return v
def publish(p,data):
 p=inside(p);p.parent.mkdir(parents=True,exist_ok=True)
 if isinstance(data,str):data=data.encode()
 if p.exists():assert p.read_bytes()==data,('Existing output differs; preserved',p);return
 tmp=p.with_name(p.name+'.pending_'+str(time.time_ns()))
 with tmp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 try:os.link(tmp,p)
 except FileExistsError:assert p.read_bytes()==data
 tmp.unlink()
 fd=os.open(p.parent,os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
def write(p,v):publish(p,json.dumps(clean(v),sort_keys=True,indent=2,allow_nan=False)+'\n')
def save(stem,value):
 old=cached(stem)
 if old is not None:return old
 raw=pickle.dumps(value,protocol=5);envelope=dict(payload=raw,sha256=hashlib.sha256(raw).hexdigest())
 p=HERE/(stem+'.pkl.gz');publish(p,gzip.compress(pickle.dumps(envelope,protocol=5),mtime=0))
 write(HERE/(stem+'.ready.json'),dict(file=str(p.relative_to(HERE)),sha256=sha(p)));return value
def cached(stem):
 p=HERE/(stem+'.pkl.gz');r=HERE/(stem+'.ready.json')
 if not p.exists():assert not r.exists();return None
 if r.exists():assert sha(p)==read(r)['sha256'],('Checkpoint changed',p)
 with gzip.open(p,'rb') as f:v=pickle.load(f)
 assert hashlib.sha256(v['payload']).hexdigest()==v['sha256']
 if not r.exists():write(r,dict(file=str(p.relative_to(HERE)),sha256=sha(p)))
 return pickle.loads(v['payload'])
def csv_new(p,frame):
 import pandas as pd
 if not isinstance(frame,pd.DataFrame):frame=pd.DataFrame(frame)
 data=frame.to_csv(index=False,float_format='%.17g',lineterminator='\n').encode()
 publish(p,gzip.compress(data,mtime=0) if str(p).endswith('.gz') else data)
def validation_fingerprint():
 paths=[p for directory in ['scripts','tests'] for p in (HERE/directory).rglob('*') if p.is_file() and p.suffix in {'.py','.c'}]
 paths += [HERE/'verify_real_mechanism_spi.py',HERE/'RUN_ALL.zsh',HERE/'pytest.ini',HERE/'preregistration.json',HERE/'audit/source_registration.json']
 paths += list(HERE.glob('addendum_*.json'))+list((HERE/'runtime').glob('*.so'))
 entries={str(p.relative_to(HERE)):sha(p) for p in sorted(paths)}
 return hashlib.sha256(json.dumps(dict(files=entries,packages=environment_status()['current'],python=sys.version.split()[0]),sort_keys=True).encode()).hexdigest()
def passing_tests(label,required=True):
 path=HERE/f'audit/tests_{label}_{validation_fingerprint()}.ready.json'
 if not path.exists():
  assert not required,('No passing tests for current code',label,path)
  return None
 result=read(path);assert result['status']=='PASS' and sha(HERE/result['log'])==result['log_sha256']
 return dict(result,receipt=str(path.relative_to(HERE)))
def artifact_paths(manifest):
 excluded=[HERE/'runtime/tmp',HERE/'runtime/mpl',HERE/'runtime/cache']
 for p in sorted(HERE.rglob('*')):
  if not p.is_file() or p in {Path(manifest),HERE/'complete.json',HERE/'pipeline.lock'} or p.name=='.DS_Store' or '.pending_' in p.name or any(p.is_relative_to(q) for q in excluded):continue
  yield p
def count(op,n=1):
 assert isinstance(n,int) and n>=0
 scope=os.environ.get('RMS_SCOPE','preparation');key=(scope,os.getpid())
 if key not in HANDLES:HANDLES[key]=(HERE/f'audit/counts_{scope}_{key[1]}.jsonl').open('ab',buffering=0)
 HANDLES[key].write((json.dumps([op,n])+'\n').encode());COUNTS[op]+=n
def record(operation,**kw):
 count('events_'+operation)
 for k,v in kw.items():
  measured=k in {'TD_fits','rl_fits','learning_steps','training_episodes','training_validation_evaluations','trading_evaluations','unavailable_pair_worlds','N1_failed_worlds'} or k.endswith(('_calls','_evaluations','_rows','_sessions'))
  if measured and isinstance(v,int) and not isinstance(v,bool) and v>=0:count(k,v)
def ledger():
 scopes={}
 for p in sorted((HERE/'audit').glob('counts_*.jsonl')):
  scope=p.stem[7:].rsplit('_',1)[0];c=scopes.setdefault(scope,Counter())
  for line in p.read_text().splitlines():
   k,n=json.loads(line);c[k]+=n
 restoration=HERE/'audit/pandas_restore/restoration_result.json'
 if restoration.exists():
  result=read(restoration);outcome=read(HERE/'addendum_007_pandas_restore_outcome.json')
  assert result['status']=='PASS' and sha(restoration)==outcome['result_sha256']
  scopes['authorized_environment_restore']=Counter(downloads=result['downloads'],package_installs=result['package_installs'],scientific_data_downloads=0)
 total=Counter(TD_fits=0,learning_steps=0,executor_calls=0,null_executions=0,switch_tests=0,bootstrap_draws=0,downloads=0,package_installs=0)
 for c in scopes.values():total.update(c)
 return dict(total=dict(total),scopes={k:dict(v) for k,v in scopes.items()},downloads=total['downloads'],package_installs=total['package_installs'])
def fingerprint(p):
 p=Path(p);s=p.lstat()
 if p.is_symlink():return dict(target=os.readlink(p),size=s.st_size,mtime_ns=s.st_mtime_ns,mode=stat.S_IMODE(s.st_mode),flags=s.st_flags,ino=s.st_ino,dev=s.st_dev,nlink=s.st_nlink)
 return dict(sha256=sha(p),size=s.st_size,mtime_ns=s.st_mtime_ns,mode=stat.S_IMODE(s.st_mode),flags=s.st_flags)
@lru_cache(None)
def registration():return read(HERE/'audit/source_registration.json')
def note(p):
 p=Path(p)
 if p.is_relative_to(HERE):return p
 assert not any(p.is_relative_to(q) for q in FORBIDDEN),('Source runtime scientific read prohibited',p)
 key=str(p.relative_to(ROOT));s=p.lstat()
 assert key not in {'.gitignore','experiments/15_pair_trading/economic_peer_residual_screen_rolling/.gitignore'},('Non-scientific, non-live input prohibited',key)
 assert not s.st_flags&DATALESS,('Dataless input prohibited',key)
 if key in PINNED:
  old=PINNED[key];assert (s.st_size,s.st_mtime_ns)==(old['size'],old['mtime_ns']),('Input metadata changed',key)
 else:
  value=fingerprint(p)
  if key in registration():assert value['sha256']==registration()[key],('Registered source changed',key)
  PINNED[key]=value
  pth=HERE/f'audit/opened_inputs_{os.environ.get("RMS_SCOPE","preparation")}_{os.getpid()}.jsonl'
  with pth.open('a') as f:f.write(json.dumps(dict(path=key,fingerprint=value))+'\n');f.flush()
 return p
def source_pickle_at(p):
 p=note(p)
 with (gzip.open(p,'rb') if str(p).endswith('.gz') else p.open('rb')) as f:v=pickle.load(f)
 if isinstance(v,dict) and 'payload' in v:
  assert hashlib.sha256(v['payload']).hexdigest()==v.get('sha256',v.get('payload_sha256'));v=pickle.loads(v['payload'])
 count('sealed_checkpoints_read');return v
def source_pickle(stem):return source_pickle_at(SOURCE/(stem+'.pkl.gz'))
def csv_read(p,**kw):
 import pandas as pd
 kw.setdefault('float_precision','round_trip');return pd.read_csv(note(p),**kw)
@contextmanager
def aliases(**items):
 old={k:sys.modules.get(k) for k in items};sys.modules.update(items)
 try:yield
 finally:
  for k,v in old.items():
   if v is None:sys.modules.pop(k,None)
   else:sys.modules[k]=v
def load(p,name):
 p=note(p);key=str(p.relative_to(ROOT));assert key in registration(),('Unregistered numerical source',key)
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);configure();return m
def source_fold(y):return y-(2004 if y<=2008 else 2009 if y<=2012 else 2013 if y<=2017 else 2018 if y<=2020 else 2021)+1
def era(y):return 'E1' if y<=2008 else 'E2' if y<=2012 else 'E3' if y<=2017 else 'E4' if y<=2020 else 'E5'
def seed(y,j):
 import numpy as np
 return int(np.random.SeedSequence([20261020,source_fold(y),j,1,0]).generate_state(1,dtype=np.uint64)[0])
def stable_seed(*parts):
 import numpy as np
 words=np.frombuffer(hashlib.sha256(json.dumps([20261027,*parts],sort_keys=True).encode()).digest(),dtype='<u4')
 return int(np.random.SeedSequence(words.tolist()).generate_state(1,dtype=np.uint64)[0])
class CleanPause(RuntimeError):pass
def checkpoint_guard():
 if shutil.disk_usage(ROOT).free<5*2**30:raise CleanPause('Free space below 5 GiB; checkpoint saved. Resume the same command after space recovers.')
 flag=os.environ.get('RMS_STOP_FILE')
 if flag and Path(flag).exists():raise CleanPause('Interrupt requested; checkpoint saved.')
def dirpath(p,fd):
 p=Path(os.fsdecode(p))
 if p.is_absolute() or fd is None or fd==-1:return p
 return Path(fcntl.fcntl(fd,50,b'\0'*1024).split(b'\0',1)[0].decode())/p
def guard(readonly=False):
 def hook(event,args):
  if event in ['socket.connect','socket.getaddrinfo']:raise PermissionError('No network/downloads')
  paths=[]
  if event=='open':
   p,mode,flags=args
   if isinstance(p,(str,bytes,os.PathLike)):
    q=Path(os.fsdecode(p)).absolute()
    if any(q.is_relative_to(v) for v in FORBIDDEN) and not HASHING and not INTEGRITY:raise PermissionError('Forbidden source runtime read: '+str(q))
    if q.exists() and q.lstat().st_flags&DATALESS:raise PermissionError('No dataless content access: '+str(q))
   if isinstance(mode,str) and any(c in mode for c in 'wax+') or isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):paths=[p]
  elif event in ['os.remove','os.rmdir']:paths=[dirpath(args[0],args[1])]
  elif event=='os.mkdir':paths=[dirpath(args[0],args[2])]
  elif event in ['os.rename','os.link']:paths=[dirpath(args[0],args[2]),dirpath(args[1],args[3])]
  elif event=='os.symlink':paths=[dirpath(args[1],args[2])]
  elif event in ['os.chmod','os.chflags','os.utime']:paths=[args[0]]
  for p in paths:
   if isinstance(p,(str,bytes,os.PathLike)) and os.fsdecode(p)!=os.devnull:
    if readonly:raise PermissionError('Read-only verification')
    inside(p)
 sys.addaudithook(hook)
