"""Append-only payload attempts and immutable completion receipts (universe pattern)."""
from ppio import *
import gzip, pickle

def restore(value):
    if isinstance(value, dict): return {k: restore(v) for k, v in value.items()}
    if isinstance(value, list): return [restore(v) for v in value]
    if value == 'inf': return float('inf')
    if value == '-inf': return -float('inf')
    return value

def ready(path):
    path = Path(path)
    if not path.exists(): return None
    result = read(path)
    for rel, value in result.get('files', {}).items():
        assert fp(HERE / rel) == value, ('Checkpoint payload changed', path, rel)
    return result

def receipt(path, files, **metadata):
    write(path, dict(files={str(Path(p).relative_to(HERE)): fp(p) for p in files}, **metadata))

def _new(stem, suffix):
    return HERE / 'runtime/checkpoints' / (stem + '_' + str(time.time_ns()) + suffix)

def save_rows(stem, rows, **metadata):
    marker = HERE / 'runtime/checkpoints' / (stem + '.json')
    if ready(marker): return read(marker)
    payload = _new(stem, '.json.gz')
    with payload.open('xb') as stream:
        stream.write(gzip.compress(json.dumps(clean(rows), allow_nan=False, separators=(',', ':')).encode(), mtime=0))
    receipt(marker, [payload], payload=str(payload.relative_to(HERE)), rows=len(rows), **metadata)
    return read(marker)

def load_rows(marker):
    result = ready(marker); assert result, ('Missing checkpoint', marker)
    with gzip.open(HERE / result['payload'], 'rt') as stream:
        return restore(json.load(stream))

def save_object(stem, value, **metadata):
    marker = HERE / 'runtime/checkpoints' / (stem + '.json')
    if ready(marker): return read(marker)
    payload = _new(stem, '.pkl.gz')
    with payload.open('xb') as stream:
        stream.write(gzip.compress(pickle.dumps(value, protocol=5), mtime=0))
    receipt(marker, [payload], payload=str(payload.relative_to(HERE)), **metadata)
    return read(marker)

def load_object(stem):
    result = ready(HERE / 'runtime/checkpoints' / (stem + '.json')); assert result, ('Missing checkpoint', stem)
    with gzip.open(HERE / result['payload'], 'rb') as stream:
        return pickle.load(stream)
