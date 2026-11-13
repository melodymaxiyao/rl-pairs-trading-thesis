"""Preparation only: source baseline, light input checks, synthetic tests, code registration.

Never starts the real computation. `baseline` registers sources; `finalize`
runs the synthetic test suite and freezes the code, specification and README top.
"""
from ppio import *
sys.path.insert(0, str(HERE / 'scripts'))
import subprocess, importlib.metadata
import xml.etree.ElementTree as ET
from pp_inputs import build_source_baseline, sealed_inputs, verify_source_baseline

PACKAGES = ['numpy', 'pandas', 'scipy', 'matplotlib', 'exchange-calendars', 'pytest', 'Pillow']

def baseline():
    os.environ['PP_SCOPE'] = 'preparation'
    guard()
    assert not (HERE / 'audit/preflight_ready.json').exists(), 'Production already started'
    result = build_source_baseline()
    print('Registered sources:', len(result['files']), 'including', result['price_files'], 'fresh price files', flush=True)
    seals = sealed_inputs()
    write(HERE / f'audit/preparation_input_seals_{time.time_ns()}.json', seals)
    print('Input seals and inventories exact:', [s['input'] for s in seals['four_sealed_inputs']] + [seals['universe']['input']], flush=True)

def finalize():
    os.environ['PP_SCOPE'] = 'preparation'
    guard()
    assert not (HERE / 'audit/preflight_ready.json').exists() and not (HERE / 'complete.json').exists()
    assert not list(HERE.rglob('*.pyc')) and not list(HERE.rglob('__pycache__'))
    verify_source_baseline()
    token = str(time.time_ns())
    xml = HERE / f'audit/preparation_tests_{token}.xml'; log = HERE / f'audit/preparation_tests_{token}.log'
    env = dict(os.environ, PP_SCOPE='tests_preparation', PP_TEST_PHASE='preparation')
    with log.open('x') as stream:
        code = subprocess.run([sys.executable, '-B', str(HERE / 'scripts/test_entry.py'), '--junitxml', str(xml),
                               '--basetemp', str(HERE / f'runtime/pytest_preparation_{token}'), str(HERE / 'tests'), '-q'],
                              cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, text=True).returncode
    print(log.read_text()[-3000:])
    assert code == 0, 'Preparation tests failed; log preserved: ' + str(log)
    suites = list(ET.parse(xml).getroot().iter('testsuite'))
    tests = sum(int(s.get('tests', 0)) for s in suites); skipped = sum(int(s.get('skipped', 0)) for s in suites)
    files = list((HERE / 'scripts').glob('*.py')) + list((HERE / 'tests').glob('*.py'))
    files += [HERE / name for name in ['verify_property_persistence.py', 'RUN_ALL.zsh', 'preregistration.json', 'request.txt']]
    files += [HERE / 'audit/source_baseline.json'] + sorted(HERE.glob('addendum_*.json'))
    readme = (HERE / 'README.md').read_bytes()
    write(HERE / 'audit/code_registration.json', dict(
        files={str(p.relative_to(HERE)): fp(p) for p in files}, README_top_bytes=len(readme),
        README_top_sha256=hashlib.sha256(readme).hexdigest(), python_version=sys.version.split()[0],
        package_versions={name: importlib.metadata.version(name) for name in PACKAGES}, registered_at_utc=utc()))
    from pp_accounting import operations
    ledger = operations()
    write(HERE / 'audit/operation_ledger_preparation.json', ledger)
    proof = [HERE / 'audit/code_registration.json', xml, log, HERE / 'audit/operation_ledger_preparation.json']
    write(HERE / 'audit/preparation_complete.json', dict(
        status='prepared', production_started=False, synthetic_tests_passed=tests - skipped, real_data_tests_deferred=skipped,
        real_pair_folds_evaluated=0, null_pairs_regenerated=0, downloads=ledger['downloads'], rl_fits=ledger['rl_fits'],
        trading_evaluations=ledger['trading_evaluations'], files={str(p.relative_to(HERE)): fp(p) for p in proof},
        command='zsh ' + str(HERE / 'RUN_ALL.zsh')))
    print('PASS preparation:', tests - skipped, 'tests passed,', skipped, 'real-data tests deferred to the user command.')

if __name__ == '__main__':
    {'baseline': baseline, 'finalize': finalize}[sys.argv[1]]()
