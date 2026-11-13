"""Own-root IO and unchanged-module dependency routing for the new universe."""
import sys
sys.dont_write_bytecode = True
sys.modules['readline'] = None
import os, json, hashlib, datetime, fcntl, types, importlib.util, tempfile, time
from pathlib import Path
from contextlib import contextmanager
from functools import lru_cache

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[2]
PT = HERE.parent
FRESH = PT / 'fresh_holdout_data_v1'
NOISE = PT / 'learner_noise_robustness_v1'
SECTOR = PT / 'universe_sector_expansion_v1'
UNI = PT / 'universe_expansion_v4'
PERSONAL = PT / 'holdout_expanded_personal_gate_v1'
A = PT / 'turning_point_pair_oracle_v1/stage_a'
B = A.parent / 'stage_b'
Q = V2 = PT / 'qlearning_zpos_v2'
HO = PT / 'qlearning_holdout_pre2018_v1'
GATE = PT / 'expanded_convergence_gate_v1'
DIAG = PT / 'zreversion_diagnostic_v2'
TABLES = HERE / 'outputs/tables'

def configure():
    os.environ.update(PYTHONDONTWRITEBYTECODE='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
                      CGATE_READ_ONLY='1', ZDIAG_READ_ONLY='1', MPLBACKEND='Agg')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[key] = '1'
    for key, folder in [('TMPDIR', 'tmp'), ('XDG_CACHE_HOME', 'cache'),
                        ('MPLCONFIGDIR', 'mpl_cache')]:
        os.environ[key] = str(HERE / 'runtime' / folder)
    tempfile.tempdir = str(HERE / 'runtime/tmp')
configure()

def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def fp(path):
    path = Path(path)
    if path.is_symlink():
        return dict(symlink=str(path.readlink()))
    st = path.stat()
    return dict(sha256=sha(path), size=st.st_size, mtime_ns=st.st_mtime_ns)

def clean(value):
    import numpy as np
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if np.isnan(value) else ('inf' if value > 0 else '-inf') if np.isinf(value) else float(value)
    if isinstance(value, (Path, datetime.date)):
        return str(value)
    return value

def inside(path):
    path = Path(path).resolve()
    assert path == HERE or HERE in path.parents, ('External write forbidden', path)
    return path

def write(path, value):
    path = inside(path)
    value = clean(value)
    if path.exists():
        assert read(path) == value, ('Existing checkpoint differs; stop', path)
        return
    pending=path.with_name(path.name+'.pending_'+str(time.time_ns()))
    with pending.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    os.link(pending,path)
    pending.unlink()

def record(operation, **counts):
    scope = os.environ.get('FAST_SCOPE', 'production')
    path = HERE / f'audit/operations_{scope}_{os.getpid()}.jsonl'
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(json.dumps(clean(dict(operation=operation, scope=scope, utc=utc(), **counts)), allow_nan=False) + '\n')
        stream.flush()
        fcntl.flock(stream, fcntl.LOCK_UN)

def forbidden(*args, **kwargs):
    raise RuntimeError('Trading, learner fitting, downloads and unsafe readers forbidden')

_GUARD_MODES=set()
def guard(readonly=False):
    if readonly in _GUARD_MODES:return
    _GUARD_MODES.add(readonly)
    def hook(event, args):
        if event in ('socket.connect', 'socket.getaddrinfo', 'os.system'):
            raise PermissionError('Networking and shell execution forbidden')
        targets = []
        if event == 'open':
            path, mode, flags = args
            if (isinstance(mode, str) and any(c in mode for c in 'wax+')) or flags & (
                    os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
                targets = [path]
        elif event in ('os.mkdir', 'os.remove', 'os.rmdir', 'os.chmod', 'os.utime', 'sqlite3.connect'):
            targets = [args[0]]
        elif event == 'os.symlink':
            targets = [args[1]]
        elif event in ('os.rename', 'os.link'):
            targets = args[:2]
        for path in targets:
            if isinstance(path, (str, bytes, os.PathLike)) and os.fsdecode(path) != os.devnull:
                if readonly:
                    raise PermissionError('Verifier is strictly read-only')
                inside(os.fsdecode(path))
    sys.addaudithook(hook)

@contextmanager
def aliases(**modules):
    previous = {k: sys.modules.get(k) for k in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value

def facade(name, **extra):
    module = types.ModuleType(name)
    module.__dict__.update({k: v for k, v in globals().items() if not k.startswith('_')})
    module.__dict__.update(HERE=HERE, OUT=HERE, write_json=forbidden, json_write=forbidden,
                           csv_write=forbidden, compressed=forbidden, configure=configure)
    module.__dict__.update(extra)
    return module

@lru_cache(None)
def baseline():
    marker = HERE / 'audit/source_baseline.json'
    assert marker.exists(), 'Verified source baseline is required before numerical imports'
    return read(marker)['files']

def note(path):
    path = Path(path)
    key = str(path.relative_to(ROOT))
    assert key in baseline(), ('Unregistered input', key)
    assert fp(path) == baseline()[key], ('Source changed', key)
    return path

def load(path, name):
    path = note(path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    configure()
    record('unchanged_source_import', path=str(path.relative_to(ROOT)), fingerprint=fp(path))
    return module

@lru_cache(None)
def helpers():
    # Only the IO namespace changes. frozen_helpers and every numerical function
    # it loads execute their original source files without source rewriting.
    with aliases(sector_io=facade('sector_io')):
        module = load(SECTOR / 'scripts/frozen_helpers.py', '_fast_unchanged_helpers')
    return module.helpers()

def set_year(year):
    h = helpers()
    h.comp.CAL = h.full_cal[h.full_cal.year < year]
    return h

def era(year):
    eras = read(HERE / 'addendum_001_eras.json')['eras']
    return next(k for k, (lo, hi) in eras.items() if lo <= year <= hi)
