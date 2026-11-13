"""Own-root immutable IO and append-only measured operations."""
import os,sys,json,time,hashlib,gzip,pickle,fcntl,tempfile,importlib.util,types
from pathlib import Path
from functools import lru_cache
from contextlib import contextmanager
from collections import Counter
sys.dont_write_bytecode=True;sys.modules['readline']=None
HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parents[2];PT=HERE.parent
SOURCE=PT/'trend_anchor_trading_v1';SYN=PT/'td_sample_complexity_v1'
TABLES=HERE/'outputs/tables';MODELS=HERE/'outputs/models';PLOTS=HERE/'outputs/plots'
YEARS=list(range(2004,2026));POOLS=['P_RAND','P_EXP','P_QUAL'];COSTS=[5,10,20]
WORLDS=[('real',0),('N1',0),('N1',1),('N2',0),('N2',1)]
CELLS=['TD8_tuned','TD8_tuned_z'];PINNED={};COUNT_HANDLES={}
def configure():
 for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
 os.environ.update(PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',MPLBACKEND='Agg')
 for k,d in [('TMPDIR','tmp'),('MPLCONFIGDIR','mpl_cache'),('XDG_CACHE_HOME','cache')]:os.environ[k]=str(HERE/'runtime'/d)
 tempfile.tempdir=str(HERE/'runtime/tmp')
configure()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def spec():return read(HERE/'preregistration.json')
def utc():
 import datetime
 return datetime.datetime.now(datetime.timezone.utc).isoformat()
def inside(p):
 p=Path(p).absolute();assert p.is_relative_to(HERE),('External write forbidden',p);return p
def clean(v):
 import numpy as np
 if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
 if isinstance(v,(tuple,list,np.ndarray)):return [clean(x) for x in v]
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
 with tmp.open('xb') as f:f.write(data)
 os.link(tmp,p);tmp.unlink()
def write(p,v):publish(p,json.dumps(clean(v),sort_keys=True,indent=2,allow_nan=False)+'\n')
def save(stem,value):
 p=HERE/(stem+'.pkl.gz');r=HERE/(stem+'.ready.json');old=cached(stem)
 if old is not None:return old
 payload=pickle.dumps(value,protocol=5);envelope={'payload':payload,'sha256':hashlib.sha256(payload).hexdigest()}
 import io
 b=io.BytesIO()
 with gzip.GzipFile(fileobj=b,mode='wb',mtime=0,filename='') as f:pickle.dump(envelope,f,protocol=5)
 publish(p,b.getvalue());write(r,dict(file=str(p.relative_to(HERE)),sha256=sha(p)))
 return value
def cached(stem):
 p=HERE/(stem+'.pkl.gz');r=HERE/(stem+'.ready.json')
 if not p.exists():assert not r.exists();return None
 if r.exists():assert sha(p)==read(r)['sha256'],('Changed checkpoint',p)
 with gzip.open(p,'rb') as f:v=pickle.load(f)
 assert hashlib.sha256(v['payload']).hexdigest()==v['sha256']
 if not r.exists():write(r,dict(file=str(p.relative_to(HERE)),sha256=sha(p)))
 return pickle.loads(v['payload'])
def csv_new(p,f):
 import pandas as pd
 if not isinstance(f,pd.DataFrame):f=pd.DataFrame(f)
 data=f.to_csv(index=False,float_format='%.17g').encode()
 if str(p).endswith('.gz'):data=gzip.compress(data,mtime=0)
 publish(p,data)
def count(op,n=1):
 assert isinstance(n,int) and n>=0
 key=(os.environ.get('TUNED_SCOPE','production'),os.getpid())
 if key not in COUNT_HANDLES:COUNT_HANDLES[key]=(HERE/f'audit/counts_{key[0]}_{key[1]}.jsonl').open('ab',buffering=0)
 COUNT_HANDLES[key].write((json.dumps([op,n])+'\n').encode())
def record(operation,**kw):
 if operation=='policy_ledger':count('trading_evaluations',int(kw['trading_evaluations']));return
 for k,v in kw.items():
  if isinstance(v,int) and not isinstance(v,bool) and k in ['TD_fits','rl_fits','learning_steps','training_episodes','training_validation_evaluations']:count(k,v)
 if operation=='TD_fit':return
 with (HERE/f'audit/events_{os.environ.get("TUNED_SCOPE","production")}_{os.getpid()}.jsonl').open('a') as f:f.write(json.dumps(clean(dict(operation=operation,**kw)),allow_nan=False)+'\n')
def ledger():
 scopes={}
 for p in sorted((HERE/'audit').glob('counts_*.jsonl')):
  scope=p.stem[len('counts_'):].rsplit('_',1)[0];c=scopes.setdefault(scope,Counter())
  for line in p.read_text().splitlines():
   k,v=json.loads(line);c[k]+=v
 total=Counter()
 for c in scopes.values():total.update(c)
 return dict(total=dict(total),scopes={k:dict(v) for k,v in scopes.items()},downloads=0,package_installs=0)
def fingerprint(p):
 p=Path(p)
 if p.is_symlink():return dict(symlink=os.readlink(p))
 s=p.stat();return dict(sha256=sha(p),size=s.st_size,mtime_ns=s.st_mtime_ns)
def note(p):
 p=Path(p)
 if p.is_relative_to(HERE):return p
 k=str(p.relative_to(ROOT))
 if k in PINNED:
  s=p.stat();v=PINNED[k];assert s.st_size==v['size'] and s.st_mtime_ns==v['mtime_ns'],('Input metadata changed',p)
 else:
  v=fingerprint(p);PINNED[k]=v
  registry=read(HERE/'audit/source_registration.json')
  if k in registry:assert v['sha256']==registry[k],('Registered source changed',k)
  record('input_pin',path=k,fingerprint=v)
 return p
def source_pickle(stem):
 p=note(SOURCE/(stem+'.pkl.gz'))
 assert p.exists(),('Required sealed checkpoint missing',p)
 with gzip.open(p,'rb') as f:return pickle.load(f)
def source_fold(y):
 start=2004 if y<=2008 else 2009 if y<=2012 else 2013 if y<=2017 else 2018 if y<=2020 else 2021
 return y-start+1
def era(y):return 'E1' if y<=2008 else 'E2' if y<=2012 else 'E3' if y<=2017 else 'E4' if y<=2020 else 'E5'
def seed(y,s):
 import numpy as np
 return int(np.random.SeedSequence([20261020,source_fold(y),s,1,0]).generate_state(1,dtype=np.uint64)[0])
def guard(readonly=False):
 def hook(event,args):
  if event in ['socket.connect','socket.getaddrinfo']:raise PermissionError('No network or downloads')
  paths=[]
  if event=='open':
   p,mode,flags=args
   if isinstance(mode,str) and any(c in mode for c in 'wax+') or isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):paths=[p]
  elif event in ['os.mkdir','os.remove','os.rmdir','os.chmod','os.utime']:paths=[args[0]]
  elif event in ['os.rename','os.link']:paths=args[:2]
  elif event=='os.symlink':paths=[args[1]]
  for p in paths:
   if isinstance(p,(str,bytes,os.PathLike)) and os.fsdecode(p)!=os.devnull:
    if readonly:raise PermissionError('Read-only verifier')
    inside(os.fsdecode(p))
 sys.addaudithook(hook)
@contextmanager
def aliases(**items):
 old={k:sys.modules.get(k) for k in items};sys.modules.update(items)
 try:yield
 finally:
  for k,v in old.items():
   if v is None:sys.modules.pop(k,None)
   else:sys.modules[k]=v
def load(p,name):
 p=note(p);s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
