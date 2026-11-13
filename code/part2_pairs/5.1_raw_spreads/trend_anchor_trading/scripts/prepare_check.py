"""Only cheap serial preparation checks. NEVER calls run.main or heavy preflight."""
from aio import *
import argparse,ast,subprocess
from control import freeze,command
from verification import INPUTS
from store import operations

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=1);args=parser.parse_args();assert args.workers==1
    assert not (HERE/'audit/protected_before.json').exists(),'Main run has already started'
    guard();os.environ['AH_SCOPE']='preparation'
    before={}
    for name in INPUTS:
        c=read(PT/name/'complete.json');assert c['status']=='complete';before[name]=sha(PT/name/'complete.json')
    write(HERE/'audit/prepared_input_seals.json',before)
    for p in list((HERE/'scripts').glob('*.py'))+list((HERE/'tests').glob('*.py'))+[HERE/'verify_trend_anchor.py']:
        compile(p.read_text(),str(p),'exec');ast.parse(p.read_text())
    shell=command(['/bin/zsh','-n',str(HERE/'RUN_ALL.zsh')],'preparation_shell_syntax')
    # Tests have no real input-verifier calls, no protected-root walk and no real
    # saved-data reproduction unless --full-reproduction is explicitly supplied.
    log=command([sys.executable,'-B',str(HERE/'scripts/test_entry.py')],'preparation_synthetic_tests')
    after={name:sha(PT/name/'complete.json') for name in INPUTS};assert after==before
    assert not (HERE/'audit/protected_before.json').exists()
    assert not (HERE/'complete.json').exists()
    ledger=operations();assert not ledger['scopes'].get('production',{}).get('TD_fits',0)
    write(HERE/'audit/preparation_operation_ledger.json',ledger)
    freeze()
    receipt(HERE/'audit/preparation_ready.json',[log,shell,HERE/'audit/code_registration.json',HERE/'audit/prepared_input_seals.json',HERE/'audit/preparation_operation_ledger.json'],status='prepared_not_run',workers=1,nice=10,numerical_threads=1,complete_seals_match=True,real_test_year_computation=False,deferred=['full protected-root SHA hashing','all full read-only input verifiers','saved real/kernel/B2/DSR/PBO reproduction','main rule/TD/null evaluation','real result reporting and plots'])
    # Strict read-only local verification; no heavy current-workspace check.
    args=[sys.executable,'-B',str(HERE/'verify_trend_anchor.py'),'--prepared']
    subprocess.run(args,check=True,cwd=ROOT,env=os.environ.copy())
    print('Prepared; MAIN NOT STARTED. User command: zsh '+str(HERE/'RUN_ALL.zsh')+' --workers 1',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        from control import preserve_failure
        preserve_failure('preparation');raise
