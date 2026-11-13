"""Input verification before any benchmark arithmetic."""
from rw_io import *
from verification import INPUTS,check_inputs,finder_inventory
import subprocess

def preflight():
    marker=HERE/'audit/preflight_ready.json'
    if checkpoint(marker):return read(marker)
    print('Checking exact sealed inputs...',flush=True)
    info=check_inputs(freeze=not (HERE/'audit/input_inventories.json').exists())
    write(HERE/'audit/finder_before.json',finder_inventory())
    logs=[]
    for name,entry in INPUTS.items():
        done=HERE/f'audit/input_{name}_ready.json'
        if checkpoint(done):logs.append(done);continue
        path=HERE/f'audit/input_{name}_{time.time_ns()}.log'
        print('Read-only verifier:',name,flush=True)
        with path.open('x') as output:
            p=subprocess.Popen([sys.executable,'-B',str(HERE/'scripts/verifier_entry.py'),str(PT/name/entry)],cwd=ROOT,env=os.environ.copy(),stdout=output,stderr=subprocess.STDOUT)
            code=p.wait()
        record('input_verifier',input_verifier_calls=1,input=name,exit_code=code,log=str(path.relative_to(HERE)))
        assert code==0,('Original verifier failed; log preserved',str(path))
        receipt(done,[path],exit_code=code);logs.append(done)
        print('PASS',name,flush=True)
    assert check_inputs()==info
    source={}
    before=read(HERE/'audit/protected_before.json')['files']
    allowed=[FRESH,NOISE/'outputs',TREND/'outputs/tables',SECTOR/'outputs/tables',
             FAST/'runtime/checkpoints',FAST/'runtime/parts',BASELINE/'outputs/models']
    prefixes=[str(p.relative_to(ROOT))+'/' for p in allowed]
    for name,value in before.items():
        p=Path(name)
        keep=(p.suffix=='.py' or p.name=='preregistration.json' or name.startswith('configs/')
              or any(name.startswith(pre) for pre in prefixes)
              or ('/runtime/' in name and (p.name.endswith(('_raw.json','_pair_fold.json')) or p.name.startswith('null_volatilities_') or p.name in ['q_kernel.so','clusters.json'])))
        if keep:source[name]=value
    # Stage-A configuration paths may have arbitrary filenames, but no numeric observations.
    stage=read(A/'preregistration.json');name=stage['source_universe_config'];source[name]=before[name]
    write(HERE/'audit/source_baseline.json',dict(files=source))
    prefix=str((STATE/'outputs/ledgers').relative_to(ROOT))+'/'
    write(HERE/'audit/source_baseline_extra.json',dict(files={k:v for k,v in before.items() if k.startswith(prefix)}))
    receipt(marker,logs+[HERE/'audit/input_inventories.json'],inputs=info)
    return read(marker)

if __name__=='__main__':
    guard()
    try:preflight()
    except BaseException:
        import traceback
        write(HERE/f'audit/preflight_failure_{time.time_ns()}.json',dict(traceback=traceback.format_exc(),status='stopped'))
        raise
