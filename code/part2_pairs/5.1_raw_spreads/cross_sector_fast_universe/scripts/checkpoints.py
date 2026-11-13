"""Append-only payload attempts and immutable per-chunk completion receipts."""
from universe_io import *
import gzip, time, pickle

def restore(value):
    if isinstance(value,dict): return {k:restore(v) for k,v in value.items()}
    if isinstance(value,list): return [restore(v) for v in value]
    if value=='inf': return float('inf')
    if value=='-inf': return -float('inf')
    return value

def ready(path):
    if not path.exists(): return None
    result=read(path)
    for rel,value in result.get('files',{}).items():
        assert fp(HERE/rel)==value,('Checkpoint payload changed',path,rel)
    return result

def receipt(path, files, **metadata):
    write(path,dict(files={str(p.relative_to(HERE)):fp(p) for p in files},**metadata))

def save_rows(stem,rows,**metadata):
    marker=HERE/'runtime/checkpoints'/(stem+'.json')
    if ready(marker): return read(marker)
    payload=HERE/'runtime/checkpoints'/(stem+'_'+str(time.time_ns())+'.json.gz')
    with payload.open('xb') as stream:
        stream.write(gzip.compress(json.dumps(clean(rows),allow_nan=False,separators=(',',':')).encode(),mtime=0))
    receipt(marker,[payload],payload=str(payload.relative_to(HERE)),rows=len(rows),**metadata)
    return read(marker)

def load_rows(marker):
    result=ready(marker);assert result
    with gzip.open(HERE/result['payload'],'rt') as stream:
        return restore(json.load(stream))

def load_context(year):
    marker=ready(HERE/f'runtime/fold_data_{year}.json');assert marker
    with gzip.open(HERE/marker['payload'],'rb') as stream:
        value=pickle.load(stream)
    assert value.year==year and value.cal.max().year<year
    record('fold_cache_load',test_year=year,max_numeric_year=int(value.cal.max().year),test_numeric_rows=0)
    return value

def save_context(context):
    year=context.year;marker=HERE/f'runtime/fold_data_{year}.json'
    if ready(marker):return read(marker)
    payload=HERE/f'runtime/fold_data_{year}_{time.time_ns()}.pkl.gz'
    with payload.open('xb') as stream:
        stream.write(gzip.compress(pickle.dumps(context,protocol=5),mtime=0))
    receipt(marker,[payload],payload=str(payload.relative_to(HERE)),test_year=year,
            max_numeric_year=int(context.cal.max().year),test_numeric_rows=0)
    return read(marker)
