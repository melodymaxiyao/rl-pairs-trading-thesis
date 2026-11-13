"""Read-only input checks, original verifiers, protected-root hashes, source baseline.

Verification scope is the universe run's (addendum_003): historical non-Finder
files exact, sealed input inventories exact, later additions only outside
sealed inputs, and exactly named .DS_Store files excluded from protected
hashing and listed separately. The sealed universe directory is now itself an
input and is checked exactly against its own manifest.
"""
from ppio import *
import subprocess
from concurrent.futures import ThreadPoolExecutor

INPUT_DIRS = ['cross_sector_fast_universe_v1', 'fresh_holdout_data_v1', 'expanded_convergence_gate_v1',
              'universe_sector_expansion_v1', 'learner_noise_robustness_v1']
VERIFIERS = [('cross_sector_fast_universe_v1', 'readonly_entry.py', 'verify_cross_sector_universe.py'),
             ('fresh_holdout_data_v1', 'readonly_entry.py', 'verify_fresh_data.py'),
             ('expanded_convergence_gate_v1', 'readonly_entry.py', 'scripts/verify_gate.py'),
             ('universe_sector_expansion_v1', 'scoped_readonly_entry.py', 'scripts/verify_sector_universe.py'),
             ('learner_noise_robustness_v1', 'scoped_readonly_entry.py', 'scripts/verify_learner_noise.py')]
UNIVERSE_READ = (['scripts/screen.py', 'scripts/events.py', 'scripts/numeric.py', 'scripts/worker.py',
                  'scripts/verification_scope.py', 'scripts/check_prerequisites.py', 'scripts/readonly_entry.py',
                  'scripts/scoped_readonly_entry.py', 'verify_cross_sector_universe.py', 'addendum_001_eras.json',
                  'addendum_003_verification_scope.json', 'preregistration.json',
                  'outputs/tables/pair_fold_inventory.csv.gz', 'outputs/tables/null_simulation.csv.gz']
                 + [f'runtime/null_volatilities_{y}.json' for y in YEARS])
EXTRA_READ = ['experiments/15_pair_trading/expanded_convergence_gate_v1/preregistration.json',
              'experiments/15_pair_trading/fresh_holdout_data_v1/data_quality.csv']
SKIP = ('.git', '.venv', 'node_modules')

def excluded(parent, name):
    return name in SKIP or parent / name in (HERE, ROOT / 'experiments/turning_point_detection')

def protected():
    files, directories = [], []
    for parent, children, names in os.walk(ROOT, followlinks=False):
        parent = Path(parent)
        children[:] = [n for n in children if not excluded(parent, n)]
        for name in children[:]:
            path = parent / name
            if path.is_symlink():
                files.append(path); children.remove(name)
            else:
                directories.append(str(path.relative_to(ROOT)))
        files.extend(parent / name for name in names if name != '.DS_Store')
    files.sort()
    with ThreadPoolExecutor(max_workers=min(12, os.cpu_count() or 1)) as executor:
        values = list(executor.map(fp, files))
    return {'files': dict(zip((str(p.relative_to(ROOT)) for p in files), values)), 'directories': sorted(directories)}

def finder_paths():
    found = {}
    for parent, children, names in os.walk(ROOT, followlinks=False):
        parent = Path(parent)
        children[:] = [n for n in children if not excluded(parent, n)]
        if '.DS_Store' in names:
            path = parent / '.DS_Store'
            found[str(path.relative_to(ROOT))] = fp(path)
    return found

def universe_exact():
    complete = read(UNIV / 'complete.json')
    assert complete['status'] == 'complete' and complete['trading'] is False
    for key, value in complete.items():
        if key.endswith('_sha256'):
            assert sha(UNIV / (key.removesuffix('_sha256') + '.json')) == value, ('Universe seal mismatch', key)
    manifest = read(UNIV / 'manifest.json')['files']
    actual = set()
    for parent, dirs, names in os.walk(UNIV, followlinks=False):
        for name in names:
            actual.add(str((Path(parent) / name).relative_to(UNIV)))
    finder = sorted(p for p in actual if Path(p).name == '.DS_Store' and p not in manifest)
    assert actual - set(finder) == set(manifest) | {'manifest.json', 'complete.json'}, ('Universe inventory changed',
        sorted(actual - set(manifest) - {'manifest.json', 'complete.json'} - set(finder))[:20], sorted(set(manifest) - actual)[:20])
    changed = []
    for rel, value in manifest.items():
        if fp(UNIV / rel) != value:
            if Path(rel).name == '.DS_Store': changed.append(rel)
            else: raise AssertionError(('Universe file changed', rel))
    return dict(input='cross_sector_fast_universe_v1', files=len(manifest), complete_sha256=sha(UNIV / 'complete.json'),
                finder_not_in_manifest=finder, changed_finder=changed)

def complete_markers():
    out = {}
    for name in INPUT_DIRS:
        marker = read(PT / name / 'complete.json')
        assert marker['status'] == 'complete', (name, marker.get('status'))
        out[name] = sha(PT / name / 'complete.json')
    return out

@lru_cache(None)
def universe_scope():
    for name in ['scripts/verification_scope.py', 'scripts/check_prerequisites.py']:
        note(UNIV / name)
    sys.path.insert(0, str(UNIV / 'scripts'))
    try:
        import verification_scope
    finally:
        sys.path.remove(str(UNIV / 'scripts'))
    return verification_scope

def sealed_inputs():
    return dict(complete_markers=complete_markers(), universe=universe_exact(),
                four_sealed_inputs=universe_scope().check_sealed_inputs())

def run_verifiers(python, audit_dir):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', CGATE_READ_ONLY='1',
               ZDIAG_READ_ONLY='1', MPLBACKEND='Agg')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        env[key] = '1'
    results = []
    for name, wrapper, entry in VERIFIERS:
        token = str(time.time_ns())
        log = Path(audit_dir) / f'verifier_{name}_{token}.log'
        command = [python, '-B', str(UNIV / 'scripts' / wrapper), str(PT / name / entry)]
        print('Read-only input verifier:', name, flush=True)
        started = time.time()
        with log.open('x') as stream:
            code = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, text=True).returncode
        tail = log.read_text()[-1500:]
        print(tail, flush=True)
        results.append(dict(input=name, wrapper=wrapper, entry=entry, exit_code=code, seconds=round(time.time() - started, 1),
                            log=str(log.relative_to(HERE))))
        record('input_verifier_completed', input_verifier_calls=1, exit_code=code)
        if code:
            raise RuntimeError('STOP: input verifier failed; original log preserved: ' + str(log))
    return results

def build_source_baseline():
    """Fingerprint every source this task imports or reads; all must match their sealed records."""
    marker = HERE / 'audit/source_baseline.json'
    if marker.exists(): return read(marker)
    universe_baseline = read(UNIV / 'audit/source_baseline.json')['files']
    files = {}
    for rel, value in universe_baseline.items():
        assert fp(ROOT / rel) == value, ('Source changed since universe baseline', rel)
        files[rel] = value
    manifest = read(UNIV / 'manifest.json')['files']
    prefix = str(UNIV.relative_to(ROOT)) + '/'
    for rel in UNIVERSE_READ:
        value = fp(UNIV / rel)
        assert manifest[rel] == value, ('Universe file differs from its manifest', rel)
        files[prefix + rel] = value
    for rel in EXTRA_READ:
        value = fp(ROOT / rel)
        if rel in universe_baseline:
            assert universe_baseline[rel] == value
        files[rel] = value
    price_files = [k for k in files if k.startswith('experiments/15_pair_trading/fresh_holdout_data_v1/data/raw/prices/')]
    write(marker, dict(files=files, universe_baseline_entries=len(universe_baseline), price_files=len(price_files),
                       created_at_utc=utc(), rule='universe source baseline re-verified exactly plus universe files checked against the universe manifest'))
    return read(marker)

def verify_source_baseline():
    files = read(HERE / 'audit/source_baseline.json')['files']
    changed = [rel for rel, value in files.items() if fp(ROOT / rel) != value]
    assert not changed, ('Registered sources changed', changed[:20])
    return len(files)
