"""Finish preparation only after original verifiers and the diagnostic tests pass."""
from mio import *
from preflight import preflight
from verification import check_inputs,finder_inventory
from protection import protected
from ledger import ledger
import subprocess,importlib.metadata

def main():
    guard();preflight();check_inputs()
    log=HERE/f'audit/tests_preparation_{time.time_ns()}.log';env=dict(os.environ,MECH_SCOPE='tests_preparation')
    with log.open('x') as f:
        code=subprocess.run([sys.executable,'-B',str(HERE/'scripts/test_entry.py'),'-q',str(HERE/'tests')],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
        f.write('\nTest process exit code: '+str(code)+'\n')
    assert code==0,('Preparation tests failed; original failure preserved',log,code)
    after=protected();write(HERE/'audit/protected_preparation_after.json',after)
    assert after==read(HERE/'audit/protected_before.json'),'Protected workspace mismatch'
    check_inputs();finder=finder_inventory();write(HERE/'audit/finder_preparation_after.json',finder)
    initial=read(HERE/'audit/finder_before.json');changed=sorted(k for k in set(initial)|set(finder) if initial.get(k)!=finder.get(k))
    write(HERE/'audit/finder_preparation_changes.json',dict(changed_paths=changed))
    text='''
## Run and resume

From any directory:

```zsh
zsh /Users/maxiyao/Desktop/FYP/experiments/15_pair_trading/spread_mechanism_diagnostic_v1/RUN_ALL.zsh
```

Default: four fold workers; use `--workers 2` to reduce memory use. Completed 100-pair chunks and folds are checked and resumed. Any changed checkpoint or frozen input stops the run. Existing failures remain in audit/. The run does not download data, install packages, fit a learner, or evaluate trading.

Read-only preparation check: `.venv/bin/python -B experiments/15_pair_trading/spread_mechanism_diagnostic_v1/verify_mechanism.py --prepared`. After completion, omit `--prepared` to verify the full seal. `complete.json` is created only after all folds, reporting, tests and protected/input checks pass.

P7 uses event-level IPW ratios, recomputed in each bootstrap draw. Pair-fold changes are sensitivity results with undefined-share drops counted. W combines same-sector and same-subindustry; the three original relations remain separate in tables. The 2004 factor formation window begins after 2000-01-31; all other sections retain the full formation year. Primary factor removal retains the formation intercept; alpha removal is descriptive. Nonpositive PCI rho is a <1-session rank category, with numeric half-life NA. Boundary and convergence coverage are reported.

Preparation fixtures read a small train-only real example for schema validation; full research computation has not started. No result or prediction label has been inferred from those fixtures.
'''
    readme=HERE/'README.md'
    if '## Run and resume' not in readme.read_text():
        with readme.open('a') as f:f.write(text)
    files=[p for d in ['scripts','tests'] for p in (HERE/d).rglob('*.py')]+[HERE/'verify_mechanism.py',HERE/'RUN_ALL.zsh',HERE/'request.txt',HERE/'preregistration.json']+list(HERE.glob('addendum_*.json'))
    top=readme.read_bytes();registration=dict(python_version=sys.version.split()[0],package_versions={n:importlib.metadata.version(n) for n in ['numpy','scipy','pandas','statsmodels','matplotlib','pytest','exchange-calendars']},
        files={str(p.relative_to(HERE)):fp(p) for p in sorted(files)},README_top_bytes=len(top),README_top_sha256=hashlib.sha256(top).hexdigest())
    write(HERE/'audit/code_registration_final.json',registration)
    write(HERE/'audit/preparation_ledger_final.json',ledger())
    receipt(HERE/'audit/preparation_ready_final.json',[log,HERE/'audit/preflight_ready.json',HERE/'audit/code_registration_final.json',HERE/'audit/protected_preparation_after.json',HERE/'audit/preparation_ledger_final.json'],
        status='prepared_tests_passed',full_research_started=False,protected_match=True,input_info=check_inputs())
    subprocess.run([sys.executable,'-B',str(HERE/'verify_mechanism.py'),'--prepared'],check=True)
    print('Prepared. User command: zsh '+str(HERE/'RUN_ALL.zsh'),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback
        write(HERE/f'audit/preparation_failure_{time.time_ns()}.json',dict(status='stopped',traceback=traceback.format_exc()));raise
