"""Freeze checked code without starting real reproduction or universe jobs."""
from universe_io import *
from check_prerequisites import protected
from verification_scope import check_sealed_inputs
from accounting import operations
import importlib.metadata
import xml.etree.ElementTree as ET

def main():
    guard()
    assert not (HERE/'complete.json').exists()
    assert not (HERE/'audit/production_protected_before.json').exists()
    assert not list((HERE/'runtime').glob('final_fold_*.json'))
    assert not list(HERE.rglob('*.pyc'))
    suites=list(ET.parse(HERE/'audit/release_tests.xml').getroot().iter('testsuite'))
    count=sum(int(s.get('tests',0)) for s in suites)
    assert count>=32 and not any(int(s.get(k,0)) for s in suites for k in ['errors','failures','skipped'])
    text=(HERE/'README.md').read_text()
    old='Prerequisites are being checked. No research results or completed pipeline are claimed. The user will run the completed pipeline; it has not been started here.'
    new=f'Code preparation is complete: {count} synthetic/implementation tests passed and all four inputs passed read-only verification. The real 2,000-pair reproduction sample and full universe/null computation await the user command below. No research results are claimed.'
    assert old in text
    (HERE/'README.md').write_text(text.replace(old,new))
    files=list((HERE/'scripts').glob('*.py'))+list((HERE/'tests').glob('*.py'))
    files += [HERE/'verify_cross_sector_universe.py',HERE/'RUN_ALL.zsh',HERE/'preregistration.json',HERE/'request.txt']
    files += list(HERE.glob('addendum_*.json'))
    files += [HERE/'audit'/name for name in ['addendum_screen_events.json','addendum_screen_events_sha256.json',
             'addendum_execution_details.json','source_baseline.json','sealed_input_inventories.json','verification_scope_report.json',
             'preregistration_original.json','README_original_top.md']]
    prefix=(HERE/'README.md').read_bytes()
    registration=dict(files={str(p.relative_to(HERE)):fp(p) for p in files},README_top_bytes=len(prefix),
                      README_top_sha256=hashlib.sha256(prefix).hexdigest(),python_version=sys.version.split()[0],
                      package_versions={name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy','matplotlib','exchange-calendars','pytest','Pillow']})
    write(HERE/'audit/code_registration.json',registration)
    seals=check_sealed_inputs()
    print('Final preparation protected-root hash: starting; excludes only task/dependency roots and exactly named Finder metadata.',flush=True)
    after=protected();before=read(HERE/'audit/protected_effective_before.json')
    assert after==before,'Protected non-Finder files or directories changed during preparation'
    write(HERE/'audit/protected_preparation_after.json',after)
    original_inputs=read(next((HERE/'audit').glob('inputs_before_*.json')))
    input_after={name:after['files'].get(name) for name in original_inputs}
    assert input_after==original_inputs
    write(HERE/'audit/inputs_preparation_after.json',input_after)
    finder={}
    for parent,dirs,names in os.walk(ROOT,followlinks=False):
        parent=Path(parent);dirs[:]=[n for n in dirs if n not in ('.git','.venv','node_modules') and parent/n not in (HERE,ROOT/'experiments/turning_point_detection')]
        if '.DS_Store' in names:
            p=parent/'.DS_Store'
            if p.exists():finder[str(p.relative_to(ROOT))]=fp(p)
    write(HERE/'audit/finder_metadata_after.json',finder)
    assert check_sealed_inputs()==seals
    ledger=operations()
    write(HERE/'audit/operation_ledger_preparation.json',ledger)
    audit=dict(status='prepared',production_started=False,synthetic_tests_passed=count,real_cross_sector_reproduction_run=False,
               real_pair_folds_evaluated=0,full_null_simulation_run=False,
               protected_files=len(after['files']),input_files=len(input_after),sealed_input_inventories=seals,
               protected_unchanged=True,inputs_unchanged=True,
               protected_before_sha256=sha(HERE/'audit/protected_effective_before.json'),
               protected_after_sha256=sha(HERE/'audit/protected_preparation_after.json'),
               downloads=0,package_installs=0,rl_fits=0,trading_evaluations=0,
               source_verifiers=dict(invocations=ledger['preparation_input_verifier_invocations'],passes=ledger['preparation_input_verifier_passes'],
                                     original_failures_preserved=ledger['preparation_original_failures_preserved']),
               imported_modules_unchanged=ledger['imported_modules_before']==ledger['imported_modules_after'],
               failure_correction='The retained base-interpreter missing-exchange_calendars test error was resolved with the existing project .venv; no installs.',
               command='zsh '+str(HERE/'RUN_ALL.zsh'))
    write(HERE/'audit/preparation_audit.json',audit)
    proof=[HERE/'audit'/name for name in ['code_registration.json','preparation_audit.json','release_tests.xml',
           'protected_preparation_after.json','inputs_preparation_after.json','finder_metadata_after.json','operation_ledger_preparation.json']]
    proof+=list((HERE/'audit').glob('*_scoped_verifier.log'))
    proof+=list((HERE/'audit').glob('*_verifier_*.log'))
    write(HERE/'audit/preparation_complete.json',dict(status='prepared',production_started=False,
         files={str(p.relative_to(HERE)):fp(p) for p in proof},synthetic_tests_passed=count,real_pair_folds_evaluated=0))
    print('PASS preparation:',count,'tests;',len(after['files']),'protected files unchanged; all four inputs verified; real run not started.',flush=True)

if __name__=='__main__':main()
