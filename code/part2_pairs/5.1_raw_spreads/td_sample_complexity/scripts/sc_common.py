"""Own-directory immutable checkpoints and bounded operation accounting."""
import os,sys,json,hashlib,time,gzip,pickle,tempfile,fcntl
from pathlib import Path
from functools import lru_cache
HERE=Path(__file__).resolve().parents[1]
ROOT=HERE.parents[2];PT=HERE.parent;SOURCE=PT/'trend_anchor_trading_v1'
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
os.environ.update(PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',MPLBACKEND='Agg')
sys.dont_write_bytecode=True;sys.modules['readline']=None
for k,d in [('TMPDIR','tmp'),('MPLCONFIGDIR','mpl_cache'),('XDG_CACHE_HOME','cache')]:
    p=HERE/'runtime'/d;p.mkdir(exist_ok=True);os.environ[k]=str(p)
tempfile.tempdir=str(HERE/'runtime/tmp')
COUNTS={}
_HASH_READ=False
def count(k,n=1):COUNTS[k]=COUNTS.get(k,0)+int(n)
def record(operation,**kw):
    for k,v in kw.items():
        if isinstance(v,int) and not isinstance(v,bool) and k in ['TD_fits','learning_steps','training_episodes','training_validation_evaluations']:count(k,v)
def sha(p):
    global _HASH_READ
    prior=_HASH_READ;_HASH_READ=True
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    _HASH_READ=prior
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def spec():return read(HERE/'preregistration.json')
def stable_seed(*parts):
    import numpy as np
    words=np.frombuffer(hashlib.sha256(json.dumps([20261027,*parts],sort_keys=True).encode()).digest(),dtype='<u4')
    return int(np.random.SeedSequence(words.tolist()).generate_state(1,dtype=np.uint64)[0])
def inside(p):
    p=Path(p).resolve();assert p==HERE or HERE in p.parents,('Write outside experiment',p);return p
def clean(v):
    import numpy as np
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple,np.ndarray)):return [clean(x) for x in v]
    if isinstance(v,(bool,np.bool_)):return bool(v)
    if isinstance(v,(int,np.integer)):return int(v)
    if isinstance(v,(float,np.floating)):return float(v) if np.isfinite(v) else None
    if isinstance(v,Path):return str(v)
    return v
def write(p,v):
    p=inside(p);p.parent.mkdir(parents=True,exist_ok=True);v=clean(v)
    if p.exists():assert read(p)==v,('Existing JSON differs',p);return
    temp=p.with_name(p.name+'.pending_'+str(time.time_ns()))
    with temp.open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
    os.link(temp,p);temp.unlink()
def save(stem,v):
    p=inside(HERE/(stem+'.pkl.gz'));r=p.with_suffix('.ready.json');p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():
        existing=cached(stem)
        assert existing is not None
        return existing
    payload=pickle.dumps(v,protocol=5)
    envelope=dict(payload=payload,payload_sha256=hashlib.sha256(payload).hexdigest())
    tmp=p.with_name(p.name+'.pending_'+str(time.time_ns()))
    with tmp.open('xb') as raw:
        with gzip.GzipFile(fileobj=raw,mode='wb',filename='',mtime=0) as f:pickle.dump(envelope,f,protocol=5)
    os.link(tmp,p);tmp.unlink()
    write(r,dict(file=str(p.relative_to(HERE)),sha256=sha(p)));return v
def cached(stem):
    p=HERE/(stem+'.pkl.gz');r=p.with_suffix('.ready.json')
    if not p.exists():
        assert not r.exists(),('Checkpoint data missing',p)
        return None
    if r.exists():assert sha(p)==read(r)['sha256'],('Changed checkpoint',p)
    with gzip.open(p,'rb') as f:envelope=pickle.load(f)
    assert isinstance(envelope,dict) and 'payload_sha256' in envelope
    assert hashlib.sha256(envelope['payload']).hexdigest()==envelope['payload_sha256']
    if not r.exists():
        # Data is published only after gzip close. A missing receipt is safely
        # reconstructed from the atomically published self-checksummed payload.
        write(r,dict(file=str(p.relative_to(HERE)),sha256=sha(p)))
    return pickle.loads(envelope['payload'])
def csv_new(p,rows):
    import pandas as pd
    p=inside(p);frame=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
    temp=p.with_name(p.name+'.pending_'+str(time.time_ns()));frame.to_csv(temp,index=False)
    if p.exists():
        assert p.read_bytes()==temp.read_bytes(),('Existing CSV differs',p)
    else:os.link(temp,p)
    temp.unlink()
def guard(readonly=False):
    def hook(event,args):
        if event in ['socket.connect','socket.getaddrinfo']:raise PermissionError('No network')
        paths=[]
        if event=='open':
            p,mode,flags=args
            if isinstance(p,(str,bytes,os.PathLike)):
                s=os.fsdecode(p)
                if '/data/raw/' in s and not _HASH_READ and not readonly:
                    raise PermissionError('Raw real prices forbidden; fixture uses sealed cached contexts only')
            if isinstance(mode,str) and any(c in mode for c in 'wax+') or flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):paths=[p]
        elif event in ['os.mkdir','os.remove','os.rmdir','os.chmod','os.utime']:paths=[args[0]]
        elif event in ['os.rename','os.link']:paths=args[:2]
        elif event=='os.symlink':paths=[args[1]]
        for p in paths:
            if isinstance(p,(str,bytes,os.PathLike)) and os.fsdecode(p)!=os.devnull:
                if readonly:raise PermissionError('Read-only verifier')
                inside(p)
    sys.addaudithook(hook)
