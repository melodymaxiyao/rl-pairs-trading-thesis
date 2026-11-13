"""Register an implementation repair after a preserved production failure.

The original code registration, preparation receipt and failure record are
kept unchanged. The repaired code is re-tested (synthetic suite) and registered
under audit/repairs/<name>/; registered() uses the latest repair registration.
"""
from ppio import *
sys.path.insert(0, str(HERE / 'scripts'))
import subprocess, importlib.metadata
import xml.etree.ElementTree as ET
from pp_prepare import PACKAGES

def main(name, description, failure):
    os.environ['PP_SCOPE'] = 'repair_' + name
    guard()
    folder = HERE / 'audit/repairs' / name
    assert not folder.exists(), 'Repair already registered'
    assert (HERE / 'audit' / failure).is_file(), 'Preserved failure record required'
    # Extraction (no property computation) may be complete; no fold, real or null chunk may exist.
    assert not list((HERE / 'runtime').glob('final_fold_*.json'))
    assert not list((HERE / 'runtime/checkpoints').glob('real_*')) and not list((HERE / 'runtime/checkpoints').glob('null_2*'))
    assert not list(HERE.rglob('*.pyc'))
    folder.mkdir(parents=True)
    token = str(time.time_ns())
    xml = folder / f'tests_{token}.xml'; log = folder / f'tests_{token}.log'
    env = dict(os.environ, PP_SCOPE='tests_repair_' + name, PP_TEST_PHASE='repair')
    with log.open('x') as stream:
        code = subprocess.run([sys.executable, '-B', str(HERE / 'scripts/test_entry.py'), '--junitxml', str(xml),
                               '--basetemp', str(HERE / f'runtime/pytest_repair_{token}'), str(HERE / 'tests'), '-q'],
                              cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, text=True).returncode
    print(log.read_text()[-2500:])
    assert code == 0, 'Repair tests failed; log preserved: ' + str(log)
    suites = list(ET.parse(xml).getroot().iter('testsuite'))
    tests = sum(int(s.get('tests', 0)) for s in suites); skipped = sum(int(s.get('skipped', 0)) for s in suites)
    files = list((HERE / 'scripts').glob('*.py')) + list((HERE / 'tests').glob('*.py'))
    files += [HERE / n for n in ['verify_property_persistence.py', 'RUN_ALL.zsh', 'preregistration.json', 'request.txt']]
    files += [HERE / 'audit/source_baseline.json'] + sorted(HERE.glob('addendum_*.json'))
    readme = (HERE / 'README.md').read_bytes()
    original = read(HERE / 'audit/code_registration.json')
    assert readme[:original['README_top_bytes']] == (HERE / 'README.md').read_bytes()[:original['README_top_bytes']]
    write(folder / 'code_registration.json', dict(
        files={str(p.relative_to(HERE)): fp(p) for p in files}, README_top_bytes=len(readme),
        README_top_sha256=hashlib.sha256(readme).hexdigest(), python_version=sys.version.split()[0],
        package_versions={n: importlib.metadata.version(n) for n in PACKAGES}, registered_at_utc=utc()))
    write(folder / 'repair.json', dict(
        name=name, description=description, preserved_failure=failure, rules_changed=False,
        original_code_registration_sha256=sha(HERE / 'audit/code_registration.json'),
        changed_files=sorted(k for k, v in read(folder / 'code_registration.json')['files'].items() if original['files'].get(k) != v),
        tests_passed=tests - skipped, real_data_tests_deferred=skipped, test_log=str(log.relative_to(HERE)), utc=utc()))
    print('PASS repair', name, ':', tests - skipped, 'tests passed; changed files:', read(folder / 'repair.json')['changed_files'])

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
