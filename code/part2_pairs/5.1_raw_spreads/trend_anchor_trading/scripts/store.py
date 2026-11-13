"""Immutable files and chunk receipts; all writes stay in this task."""
from aio import *
import gzip,csv
import numpy as np,pandas as pd

def equal_streams(a,b):
    while True:
        x=a.read(1048576);y=b.read(1048576)
        if x!=y:return False
        if not x:return True

def pickle_read(path):
    with Path(path).open('rb') as raw:
        if str(path).endswith('.gz'):
            with gzip.GzipFile(fileobj=raw,mode='rb') as f:return pickle.load(f)
        return pickle.load(raw)

def pickle_new(path,value):
    path=inside(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():assert equivalent(pickle_read(path),value);return
    pending=path.with_name(path.name+'.pending_'+str(time.time_ns()))
    with pending.open('xb') as raw:
        if str(path).endswith('.gz'):
            with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0,filename='') as f:pickle.dump(value,f,protocol=5)
        else:pickle.dump(value,raw,protocol=5)
    os.link(pending,path);pending.unlink()

def csv_new(path,frame):
    path=inside(path);path.parent.mkdir(parents=True,exist_ok=True);pending=path.with_name(path.name+'.pending_'+str(time.time_ns()))
    frame=pd.DataFrame(frame)
    with pending.open('xb') as raw:
        if str(path).endswith('.gz'):
            with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0,filename='') as f:f.write(frame.to_csv(index=False).encode())
        else:raw.write(frame.to_csv(index=False).encode())
    if path.exists():
        opener=gzip.open if str(path).endswith('.gz') else open
        with opener(path,'rb') as a,opener(pending,'rb') as b:assert equal_streams(a,b),('Existing CSV differs',path)
    else:os.link(pending,path)
    pending.unlink()

def commit(stem,value,**meta):
    p=HERE/(stem+'.pkl.gz');pickle_new(p,value);receipt(HERE/(stem+'.ready.json'),[p],**meta);return value

def cached(stem):
    marker=HERE/(stem+'.ready.json')
    return pickle_read(HERE/(stem+'.pkl.gz')) if checkpoint(marker) else None

def daily_rows(cell):
    for stem in cell['chunks']:
        chunk=cached(stem)
        assert chunk is not None,('Missing chunk',stem)
        yield from chunk['daily']

def operations():
    from collections import defaultdict
    scopes=defaultdict(lambda:defaultdict(int));events=0;imports={}
    for p in (HERE/'audit').glob('operations_*.jsonl'):
        for line in p.read_text().splitlines():
            r=json.loads(line);events+=1
            if r.get('operation') in ['unchanged_source_import','source_import'] and 'path' in r and 'fingerprint' in r:
                imports[r['path']]=r['fingerprint']
            for k,v in r.items():
                if isinstance(v,(int,float)) and not isinstance(v,bool):scopes[r['scope']][k]+=v
    for p in (HERE/'audit').glob('count_*__*__*.bin'):
        scope,op,pid=p.stem[len('count_'):].split('__');scopes[scope][op]+=p.stat().st_size
    current={k:fp(ROOT/k) for k in imports}
    assert current==imports,'Imported original module changed'
    return dict(events=events,scopes={k:dict(v) for k,v in scopes.items()},downloads=0,package_installs=0,imported_modules_before=imports,imported_modules_after=current,own_sources={str(p.relative_to(ROOT)):fp(p) for p in (HERE/'scripts').glob('*.py')})
