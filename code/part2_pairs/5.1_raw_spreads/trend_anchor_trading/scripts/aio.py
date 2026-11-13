"""Own-root IO, input seals, deterministic checkpoints and measured diagnostic operations."""
import sys
sys.dont_write_bytecode=True
sys.modules['readline']=None
import os,json,hashlib,datetime,tempfile,fcntl,types,importlib.util,pickle,time
from pathlib import Path
from functools import lru_cache
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parents[2];PT=ROOT/'experiments/15_pair_trading'
SECTOR=PT/'universe_sector_expansion_v1';STATE=PT/'state_ablation_v1';NOISE=PT/'learner_noise_robustness_v1';TREND=PT/'trend_exposure_diagnostic_v1'
FRESH=PT/'fresh_holdout_data_v1';V2=Q=PT/'qlearning_zpos_v2';V3=PT/'qlearning_zpos_v3';QI=PT/'qlearning_improvement_v1';FOLD=PT/'qlearning_folded_v1'
A=PT/'turning_point_pair_oracle_v1/stage_a';B=A.parent/'stage_b';UNI=PT/'universe_expansion_v4';HO=PT/'qlearning_holdout_pre2018_v1';PERSONAL=PT/'holdout_expanded_personal_gate_v1'
GATE=PT/'expanded_convergence_gate_v1';DIAG=PT/'zreversion_diagnostic_v2';STAGE1=PT/'refresh_spectrum_stage1_v1';STAGE2=PT/'stage2_selector_v1'
TABLES=HERE/'outputs/tables';MODELS=HERE/'outputs/models'
POOLS=['EXP_G_cap','EXP_G','EXP_ALL'];STATES=['z9','zs','zv','zvs','f','z9_R'];CASES=['0','5','10','20','pair']
NB={'z9':9,'zs':18,'zv':18,'zvs':36,'f':3,'z9_R':9}
CELLS=['z9|TD_z','z9|TD_ens','z9|MB','zs|MB','zv|MB','zvs|MB','f|MB','z9_R|TD_ens','z9_R|MB']
BENCHMARKS=[c+'|'+m for c in ['F','R'] for m in ['B0','B2','B3','H1s','SPY']]
ERAS={'E1':(2004,2008),'E2':(2009,2012),'E3':(2013,2017),'E4':(2018,2020),'E5':(2021,2025)}
def era(y):return next(k for k,(a,b) in ERAS.items() if a<=y<=b)
def source_fold(y):return y-ERAS[era(y)][0]+1
def configure():
    os.environ.update(PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',CGATE_READ_ONLY='1',ZDIAG_READ_ONLY='1',MPLBACKEND='Agg')
    for k,d in [('TMPDIR','tmp'),('XDG_CACHE_HOME','cache'),('MPLCONFIGDIR','mpl_cache')]:os.environ[k]=str(HERE/'runtime'/d)
    for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
    tempfile.tempdir=str(HERE/'runtime/tmp')
configure()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def fp(p):
    p=Path(p)
    if p.is_symlink():return dict(symlink=str(p.readlink()))
    s=p.stat();return dict(sha256=sha(p),size=s.st_size,mtime_ns=s.st_mtime_ns)
def read(p):return json.loads(Path(p).read_text())
def clean(v):
    import numpy as np,pandas as pd
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple,np.ndarray)):return [clean(x) for x in v]
    if isinstance(v,(bool,np.bool_)):return bool(v)
    if isinstance(v,(int,np.integer)):return int(v)
    if isinstance(v,(float,np.floating)):
        return float(v) if np.isfinite(v) else 'inf' if v>0 else '-inf' if v<0 else None
    if isinstance(v,(Path,pd.Timestamp,datetime.date)):return str(v)
    return v
def inside(p):
    p=Path(p).resolve();assert p==HERE or HERE in p.parents,('External write forbidden',p);return p
def equivalent(a,b):
    import numpy as np,pandas as pd,math
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equivalent(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return isinstance(b,(list,tuple)) and len(a)==len(b) and all(equivalent(x,y) for x,y in zip(a,b))
    if isinstance(a,np.ndarray):
        if not isinstance(b,np.ndarray) or a.shape!=b.shape or a.dtype!=b.dtype:return False
        return bool(np.allclose(a,b,rtol=0,atol=1e-9,equal_nan=True)) if np.issubdtype(a.dtype,np.number) else bool(np.array_equal(a,b))
    if isinstance(a,(pd.Index,pd.Series,pd.DataFrame)):return isinstance(b,type(a)) and a.equals(b)
    if isinstance(a,(float,np.floating)):
        return isinstance(b,(int,float,np.integer,np.floating)) and ((np.isnan(a) and np.isnan(b)) or math.isclose(float(a),float(b),rel_tol=0,abs_tol=1e-9))
    return a==b
def write(p,value):
    p=inside(p);value=clean(value);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():assert equivalent(read(p),value),('Existing JSON differs beyond 1e-9; preserve and stop',p);return
    tmp=p.with_name(p.name+'.partial.'+str(time.time_ns()))
    with tmp.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
    assert not p.exists();os.rename(tmp,p)
read_json=read;write_json=json_write=write
def npz(p,**arrays):
    import numpy as np
    p=inside(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():
        with np.load(p,allow_pickle=False) as saved:
            assert set(saved.files)==set(arrays)
            for k,v in arrays.items():
                v=np.asarray(v);a=saved[k];assert a.shape==v.shape and a.dtype==v.dtype
                if np.issubdtype(v.dtype,np.number):np.testing.assert_allclose(a,v,rtol=0,atol=1e-9,equal_nan=True)
                else:np.testing.assert_array_equal(a,v)
        return
    tmp=p.with_name(p.name+'.partial.'+str(time.time_ns()))
    with tmp.open('xb') as f:np.savez_compressed(f,**arrays)
    assert not p.exists();os.rename(tmp,p)
def dump(p,value):
    p=inside(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():
        assert equivalent(pickle.loads(p.read_bytes()),value),('Existing pickle differs beyond 1e-9',p);return
    tmp=p.with_name(p.name+'.partial.'+str(time.time_ns()))
    with tmp.open('xb') as f:pickle.dump(value,f,protocol=5)
    assert not p.exists();os.rename(tmp,p)
def receipt(p,files,**details):write(p,dict(files={str(q.relative_to(HERE)):fp(q) for q in files},**details))
def checkpoint(p):
    if not p.exists():return False
    r=read(p)
    for k,v in r['files'].items():assert fp(HERE/k)==v,('Checkpoint changed',k)
    return r
def record(operation,**details):
    scope=os.environ.get('AH_SCOPE','production');p=HERE/f'audit/operations_{scope}_{os.getpid()}.jsonl'
    with p.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX);f.write(json.dumps(clean(dict(operation=operation,scope=scope,utc=utc(),**details)),allow_nan=False)+'\n');f.flush();fcntl.flock(f,fcntl.LOCK_UN)
def forbidden(*a,**k):raise RuntimeError('Unregistered reader/writer, oracle or download path forbidden')
csv_write=compressed=forbidden
def guard(readonly=False):
    def hook(event,args):
        if event in ['socket.connect','socket.getaddrinfo']:raise PermissionError('Network/downloads forbidden')
        paths=[]
        if event=='open':
            p,mode,flags=args
            if (isinstance(mode,str) and any(c in mode for c in 'wax+')) or flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):paths=[p]
        elif event in ['os.mkdir','os.remove','os.rmdir','os.chmod','os.utime','sqlite3.connect']:paths=[args[0]]
        elif event=='os.symlink':paths=[args[1]]
        elif event in ['os.rename','os.link']:paths=args[:2]
        for p in paths:
            if isinstance(p,(str,bytes,os.PathLike)) and os.fsdecode(p)!=os.devnull:
                if readonly:raise PermissionError('Read-only verifier')
                inside(os.fsdecode(p))
    sys.addaudithook(hook)
def snapshot():
    files=[];links={};dirs=[]
    for parent,children,names in os.walk(ROOT,followlinks=False):
        p=Path(parent);children[:]=[n for n in children if n not in ['.git','.venv','node_modules'] and p/n not in [HERE,ROOT/'experiments/turning_point_detection']]
        for n in children.copy():
            q=p/n
            if q.is_symlink():links[str(q.relative_to(ROOT))]=fp(q);children.remove(n)
            else:dirs.append(str(q.relative_to(ROOT)))
        files.extend(p/n for n in names)
    with ThreadPoolExecutor(max_workers=min(12,os.cpu_count() or 1)) as ex:result=dict(zip((str(p.relative_to(ROOT)) for p in files),ex.map(fp,files)))
    result.update(links);return dict(files=result,directories=sorted(dirs))
@lru_cache(None)
def baseline():
    return read(HERE/'audit/source_baseline.json')['files']

def note(p):
    p=Path(p)
    if HERE in p.resolve().parents:return p
    k=str(p.relative_to(ROOT));expected=baseline()[k];s=p.stat()
    assert s.st_size==expected['size'] and s.st_mtime_ns==expected['mtime_ns'],('Input metadata changed',k)
    return p
@contextmanager
def aliases(**mods):
    old={k:sys.modules.get(k) for k in mods};sys.modules.update(mods)
    try:yield
    finally:
        for k,v in old.items():
            if v is None:sys.modules.pop(k,None)
            else:sys.modules[k]=v
def facade(name,**extra):
    m=types.ModuleType(name);m.__dict__.update({k:v for k,v in globals().items() if not k.startswith('_')});m.__dict__.update(OUT=HERE,configure_runtime=configure,POOLS=HERE/'pools',REG=read(V2/'preregistration.json'))
    m.__dict__.update(extra);return m
def load(path,name):
    path=note(path);value=fp(path);assert value==baseline()[str(path.relative_to(ROOT))]
    original_seals=source_seals()
    key=str(path.relative_to(ROOT))
    if key in original_seals:assert value==original_seals[key]['fingerprint'],('Original source differs from baseline frozen record',path)
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);configure();record('unchanged_source_import',path=key,fingerprint=value);return m

@lru_cache(None)
def source_seals():return read(PT/'exp_baseline_v1/source_code_seals.json')

BASELINE=PT/'exp_baseline_v1'

def arrays_hash(es):
    import numpy as np
    h=hashlib.sha256()
    for e in es:
        h.update(json.dumps([e['pair'],e['year'],e['window']]).encode())
        for k in ['bins','z','forward_bps','drift','strong']:
            if k in e:
                a=np.ascontiguousarray(e[k]);h.update(str(a.dtype).encode());h.update(a.tobytes())
    return h.hexdigest()
FAST=PT/'cross_sector_fast_universe_v1'

def registered():
    r=read(HERE/'audit/code_registration.json')
    import importlib.metadata
    assert r['python_version']==sys.version.split()[0], 'Python version changed'
    for name,value in r['package_versions'].items():assert importlib.metadata.version(name)==value,('Package version changed',name)
    for name,value in r['files'].items():assert fp(HERE/name)==value,('Frozen code changed',name)
    prefix=(HERE/'README.md').read_bytes()[:r['README_top_bytes']]
    assert hashlib.sha256(prefix).hexdigest()==r['README_top_sha256']
    initial=read(HERE/'audit/README_before_numeric_checks.json')['fingerprint']
    assert hashlib.sha256(prefix[:initial['size']]).hexdigest()==initial['sha256'],'Initial preregistered README prefix changed'
    current={str(p.relative_to(HERE)) for d in ['scripts','tests'] for p in (HERE/d).rglob('*.py')}|{'verify_anchor_horizon.py'}
    assert current=={p for p in r['files'] if p.endswith('.py')},'Source inventory changed'
    return r

def dump_gzip(path,value):
    import gzip
    path=inside(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        with gzip.open(path,'rb') as f:assert equivalent(pickle.load(f),value)
        return
    tmp=path.with_name(path.name+'.pending_'+str(time.time_ns()))
    with tmp.open('xb') as raw:
        with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0,filename='') as f:pickle.dump(value,f,protocol=5)
    os.link(tmp,path);tmp.unlink()

def load_gzip(path):
    import gzip
    with Path(path).open('rb') as raw:
        with gzip.GzipFile(fileobj=raw,mode='rb') as f:return pickle.load(f)

PROPERTY=PT/'pair_property_persistence_v1'
RWB=PT/'random_walk_benchmark_v1'
UNIV=FAST
PLOTS=HERE/'outputs/plots'
YEARS=list(range(2004,2026))
SEED=20261024
MECHANISM=PT/'spread_mechanism_diagnostic_v1'

def settings():return read(HERE/'addendum_001_user_decisions.json')

_COUNT_FILES={}
def count(operation,n=1):
    """One append-only byte per actual call; preserves counts across interruption."""
    scope=os.environ.get('AH_SCOPE','production');key=(scope,operation,os.getpid())
    if key not in _COUNT_FILES:
        path=HERE/f'audit/count_{scope}__{operation}__{os.getpid()}.bin'
        _COUNT_FILES[key]=path.open('ab',buffering=0)
    _COUNT_FILES[key].write(b'1'*int(n))

# Task-local overrides; original numerical module bodies use this IO facade.
ANCHOR=PT/'anchor_horizon_diagnostic_v1'
SEED=20261026
BASELINE=PT/'exp_baseline_v1'
