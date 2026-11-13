"""Preparation seal: input verifiers -> tests -> full protected after hash."""
from aio import *
from preflight import preflight
from verification import check_inputs,finder_inventory
from protection import protected
from run import tests
from ledger import ledger
import importlib.metadata,subprocess

def main():
    guard();preflight();inputs=check_inputs();log=tests('preparation')
    after=protected();write(HERE/'audit/protected_preparation_after.json',after)
    assert after==read(HERE/'audit/protected_before.json'),'Protected root mismatch'
    assert check_inputs()==inputs
    finder=finder_inventory();write(HERE/'audit/finder_preparation_after.json',finder)
    initial=read(HERE/'audit/finder_before.json');write(HERE/'audit/finder_changes.json',dict(changed_paths=sorted(k for k in set(initial)|set(finder) if initial.get(k)!=finder.get(k))))
    readme=HERE/'README.md';text='''
## Run and resume

```zsh
zsh /Users/maxiyao/Desktop/FYP/experiments/15_pair_trading/anchor_horizon_diagnostic_v1/RUN_ALL.zsh
```

Default: four fold workers. Add `--workers 2` to reduce memory use. Every 100 pair-folds is an immutable checkpoint; rerun the same command to resume after interruption. Frozen inputs, code and existing checkpoints are checked before reuse. Original failures remain in audit/. Full research has not run during preparation; only synthetic and small train-only source fixtures were evaluated.

Read-only check: `.venv/bin/python -B experiments/15_pair_trading/anchor_horizon_diagnostic_v1/verify_anchor_horizon.py --prepared`. After full completion, omit `--prepared`. `complete.json` is written only after reporting, tests and input/protected checks pass. Event P&L is counted separately from trading evaluation; no learner, strategy, download or package install is executed.

F_orig appears within-year only. Its N2 z uses synthetic legs from the same daily signs as the fixed-beta N2 spread, solely to obtain annual F parameters; event P&L never uses those annual betas. Q4 primarily compares F_orig within-year KM medians, with A_F as companion.

The driver primary is sum(w*q*delta_S)/sum(w*q*s_entry*delta_d), q=-sign(d_entry); per-event shares, quartiles/IQR, outside-[0,1] frequency, exact price/mean/scale pieces and censored exclusions are also reported. No clipping is applied. Adaptive scale contributions act through terminal overshoot. INFEASIBLE anchors are labelled in every relevant table and plot and excluded from Q2/Q3/Q6.

The sector tables include X, W (same_sector plus same_subindustry) and the original two within-sector relations. Aligned/counter/unknown entry rates use the unconditional pair-year denominator, so their rates add to the total. A relative-to-null advantage and positive absolute net after 10 bps are reported separately.
'''
    if '## Run and resume' not in readme.read_text():
        with readme.open('a') as f:f.write(text)
    files=[p for d in ['scripts','tests'] for p in (HERE/d).rglob('*.py')]+[HERE/'verify_anchor_horizon.py',HERE/'RUN_ALL.zsh',HERE/'preregistration.json',HERE/'request.txt']+list(HERE.glob('addendum_*.json'))
    top=readme.read_bytes();write(HERE/'audit/code_registration.json',dict(python_version=sys.version.split()[0],
        package_versions={k:importlib.metadata.version(k) for k in ['numpy','scipy','pandas','statsmodels','matplotlib','pytest','exchange-calendars']},
        files={str(p.relative_to(HERE)):fp(p) for p in sorted(files)},README_top_bytes=len(top),README_top_sha256=hashlib.sha256(top).hexdigest()))
    write(HERE/'audit/preparation_ledger.json',ledger())
    receipt(HERE/'audit/preparation_ready.json',[log,HERE/'audit/preflight_ready.json',HERE/'audit/code_registration.json',HERE/'audit/preparation_ledger.json',HERE/'audit/protected_preparation_after.json'],
        status='prepared_tests_passed',full_research_started=False,protected_match=True,inputs=inputs)
    # The current full protected SHA check just passed above; avoid hashing it twice.
    subprocess.run([sys.executable,'-B',str(HERE/'verify_anchor_horizon.py'),'--prepared','--no-current-protected'],check=True)
    print('Prepared. User command: zsh '+str(HERE/'RUN_ALL.zsh'),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback
        write(HERE/f'audit/preparation_failure_{time.time_ns()}.json',dict(status='stopped',traceback=traceback.format_exc()));raise
