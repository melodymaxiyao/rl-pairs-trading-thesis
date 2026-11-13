"""Check every sealed input before any diagnostic statistics."""
from aio import *
from verification import check_inputs,INPUTS,finder_inventory
import subprocess
from concurrent.futures import ThreadPoolExecutor,as_completed

def one_input(name):
    marker=HERE/f'audit/input_{name}_ready.json'
    if checkpoint(marker):return marker
    log=HERE/f'audit/input_{name}_{time.time_ns()}.log'
    args=[sys.executable,'-B',str(HERE/'scripts/verifier_entry.py'),str(PT/name/INPUTS[name])]
    print('Read-only original verifier:',name,flush=True)
    with log.open('x') as output:code=subprocess.run(args,cwd=ROOT,env=os.environ.copy(),stdout=output,stderr=subprocess.STDOUT).returncode
    record('input_verifier',input_verifier_calls=1,input=name,exit_code=code)
    assert code==0,('Input verifier failed; preserved log',log)
    receipt(marker,[log],input=name,exit_code=code);print('PASS verifier:',name,flush=True)
    return marker

def preflight():
    ready=HERE/'audit/preflight_ready.json'
    if checkpoint(ready):return read(ready)
    assert (HERE/'audit/protected_before.json').exists()
    info=check_inputs(freeze=not (HERE/'audit/input_inventories.json').exists())
    if not (HERE/'audit/finder_before.json').exists():write(HERE/'audit/finder_before.json',finder_inventory())
    # Both workers belong to this task and only verify existing immutable inputs.
    with ThreadPoolExecutor(max_workers=2) as ex:
        futures={ex.submit(one_input,name):name for name in INPUTS}
        logs=[future.result() for future in as_completed(futures)]
    assert check_inputs()==info
    receipt(ready,logs+[HERE/'audit/input_inventories.json'],inputs=info)
    return read(ready)

if __name__=='__main__':
    guard()
    try:preflight()
    except BaseException:
        import traceback
        write(HERE/f'audit/preflight_failure_{time.time_ns()}.json',dict(status='stopped',traceback=traceback.format_exc()));raise
