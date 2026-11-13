"""Freeze tested source and proof, without launching any matched benchmark job."""
from rw_io import *
from protection import protected
from verification import check_inputs,finder_inventory
from accounting import operations
import xml.etree.ElementTree as ET
import importlib.metadata

def main():
    guard()
    assert checkpoint(HERE/'audit/preflight_ready.json')
    assert checkpoint(HERE/'audit/tests_before_ready.json')
    assert not (HERE/'complete.json').exists() and not (HERE/'runtime/cohorts.pkl.gz').exists()
    assert not list((HERE/'runtime/checkpoints').glob('*_primary_ready.json'))
    proof=read(HERE/'audit/tests_before_ready.json')
    xml=HERE/'audit/release_tests.xml'
    suites=list(ET.parse(xml).getroot().iter('testsuite'))
    assert not any(int(s.get(k,0)) for s in suites for k in ['failures','errors'])
    count=sum(int(s.get('tests',0))-int(s.get('skipped',0)) for s in suites)
    assert read(HERE/'audit/martingale_sanity.json')['PASS']
    assert read(HERE/'audit/sealed_fixture_active.json')['status']=='PASS'
    with (HERE/'README.md').open('a') as stream:
        stream.write(f'\n\n## Prepared implementation\n\n{count} pre-computation tests passed, including 20,000 unfiltered B2 sanity paths, frozen real TD replay, source-row decomposition matching, and continuity train reproduction. Full matched benchmark and 44,000 continuity extensions have not been run.\n\nRun from any directory:\n\n```zsh\nzsh {HERE}/RUN_ALL.zsh\n```\n\nResume with the same command; `--workers` may be set between 1 and 12. All checkpoints stay under this directory. Read-only checks: `.venv/bin/python -B {HERE}/verify_rw_benchmark.py --prepared` before the full run; omit `--prepared` after completion.\n')
    files=list((HERE/'scripts').glob('*.py'))+list((HERE/'tests').glob('*.py'))
    files += [HERE/name for name in ['verify_rw_benchmark.py','RUN_ALL.zsh','preregistration.json','request.txt','runtime/q_kernel.so']]
    files += list(HERE.glob('addendum_*.json'))
    files += [HERE/'audit/source_baseline.json',HERE/'audit/source_baseline_extra.json',HERE/'audit/input_inventories.json']
    files += [HERE/'audit'/name for name in ['verification_scope_report.json','fresh_historical_metadata_check.json','implementation_repairs.json','README_before_numeric_checks.json']]
    top=(HERE/'README.md').read_bytes()
    registration=dict(files={str(p.relative_to(HERE)):fp(p) for p in files},README_top_bytes=len(top),README_top_sha256=hashlib.sha256(top).hexdigest(),
        python_version=sys.version.split()[0],package_versions={name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy','matplotlib','exchange-calendars','pytest','Pillow']})
    write(HERE/'audit/code_registration.json',registration)
    print('Final preparation protected hashing...',flush=True)
    after=protected();assert after==read(HERE/'audit/protected_before.json'),'Protected roots changed during preparation'
    write(HERE/'audit/protected_preparation_after.json',after)
    inputs=check_inputs();finder=finder_inventory();before=read(HERE/'audit/finder_before.json')
    write(HERE/'audit/finder_preparation_after.json',dict(files=finder,changed=[k for k in sorted(set(before)|set(finder)) if before.get(k)!=finder.get(k)]))
    write(HERE/'audit/operation_ledger_preparation.json',operations())
    proof_paths=[HERE/'audit'/name for name in ['code_registration.json','tests_before_ready.json','martingale_sanity.json','sealed_fixture_active.json',
        'preflight_ready.json','protected_preparation_after.json','finder_preparation_after.json','operation_ledger_preparation.json']]
    proof_paths += [HERE/name for name in proof['files']]
    proof_paths += [HERE/'audit/release_tests.xml',HERE/'audit/release_tests.log']
    proof_paths += list((HERE/'audit').glob('input_*_ready.json'))+list((HERE/'audit').glob('input_*.log'))
    proof_paths += [HERE/'audit/verification_scope_report.json',HERE/'audit/fresh_historical_metadata_check.json',HERE/'audit/implementation_repairs.json']
    write(HERE/'audit/preparation_complete.json',dict(status='prepared',full_benchmark_started=False,tests_passed=count,
        inputs=inputs,protected_files=len(after['files']),protected_unchanged=True,
        files={str(p.relative_to(HERE)):fp(p) for p in proof_paths},command='zsh '+str(HERE/'RUN_ALL.zsh')))
    print('PASS prepared code:',count,'tests; full benchmark awaits user command.',flush=True)

if __name__=='__main__':main()
