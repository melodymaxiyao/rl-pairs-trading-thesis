"""Preparation freeze, main-only heavy preflight, protected checks and final seal."""
from aio import *
from store import operations
import subprocess,importlib.metadata

CODE_DIRS=['scripts','tests']
def code_paths():return sorted([p for d in CODE_DIRS for p in (HERE/d).rglob('*') if p.suffix in ['.py','.c']]+[HERE/'verify_trend_anchor.py',HERE/'RUN_ALL.zsh',HERE/'preregistration.json',HERE/'request.txt',HERE/'audit/prepared_input_seals.json']+list(HERE.glob('addendum_*')))

def freeze():
    files=code_paths();top=(HERE/'README.md').read_bytes()
    value=dict(status='prepared_not_run',python_version=sys.version.split()[0],package_versions={n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','statsmodels','matplotlib','pytest','exchange-calendars']},files={str(p.relative_to(HERE)):fp(p) for p in files},assets={str(p.relative_to(HERE)):fp(p) for p in [HERE/'runtime/q_kernel.so',HERE/'runtime/fold_kernel.so',HERE/'runtime/adapter_kernel.so']},README_top_bytes=len(top),README_top_sha256=hashlib.sha256(top).hexdigest())
    write(HERE/'audit/code_registration.json',value);return value

def registered():
    r=read(HERE/'audit/code_registration.json');assert r['python_version']==sys.version.split()[0]
    for name,version in r['package_versions'].items():assert importlib.metadata.version(name)==version,('Package version changed',name)
    assert {str(p.relative_to(HERE)) for p in code_paths()}==set(r['files'])
    for k,v in {**r['files'],**r['assets']}.items():assert fp(HERE/k)==v,('Prepared source/asset changed',k)
    top=(HERE/'README.md').read_bytes()[:r['README_top_bytes']];assert hashlib.sha256(top).hexdigest()==r['README_top_sha256']
    spec=read(HERE/'preregistration.json')
    for k,v in spec['addenda'].items():assert sha(HERE/k)==v
    for name,value in read(HERE/'audit/prepared_input_seals.json').items():assert sha(PT/name/'complete.json')==value,('Input completion seal changed after preparation',name)
    return r

def command(args,label):
    path=HERE/f'audit/{label}_{time.time_ns()}.log'
    with path.open('x') as f:r=subprocess.run(args,cwd=ROOT,env=os.environ.copy(),stdout=f,stderr=subprocess.STDOUT)
    if r.returncode:raise RuntimeError('Check failed; original log retained: '+str(path))
    print('PASS',label,str(path),flush=True);return path

def tests(phase,full=False):
    os.environ['AH_SCOPE']='tests_'+phase
    args=[sys.executable,'-B',str(HERE/'scripts/test_entry.py')]
    if full:args+=['--full-reproduction']
    log=command(args,'tests_'+phase);os.environ['AH_SCOPE']='production';return log

def preflight():
    from protection import protected
    from verification import check_inputs,INPUTS,finder_inventory
    registered();before_path=HERE/'audit/protected_before.json'
    print('Full protected SHA hashing (deferred from preparation), one worker.',flush=True)
    current=protected()
    if before_path.exists():assert current==read(before_path),'Protected workspace changed since main run start'
    else:write(before_path,current)
    if not (HERE/'audit/source_baseline.json').exists():write(HERE/'audit/source_baseline.json',dict(files=current['files']))
    if not (HERE/'audit/finder_before.json').exists():write(HERE/'audit/finder_before.json',finder_inventory())
    expected=read(HERE/'audit/prepared_input_seals.json')
    for name,value in expected.items():assert sha(PT/name/'complete.json')==value
    check_inputs(freeze=not (HERE/'audit/input_inventories.json').exists());proofs=[]
    for name,relative in INPUTS.items():
        marker=HERE/f'audit/input_{name}_ready.json'
        if not checkpoint(marker):
            print('Original read-only verifier:',name,flush=True)
            log=command([sys.executable,'-B',str(HERE/'scripts/verifier_entry.py'),str(PT/name/relative)],'input_'+name)
            record('input_verifier',input_verifier_calls=1,input=name);receipt(marker,[log],input=name,exit_code=0)
        proofs.append(marker)
    check_inputs();receipt(HERE/'audit/preflight_ready.json',proofs+[before_path,HERE/'audit/input_inventories.json'],status='PASS')

def preserve_failure(stage):
    import traceback
    write(HERE/f'audit/failure_{stage}_{time.time_ns()}.json',dict(stage=stage,status='stopped',traceback=traceback.format_exc(),utc=utc()))

def finish():
    from protection import protected
    from verification import check_inputs,finder_inventory
    from reporting import verify_part1
    verify_part1();registered();check_inputs();after=protected();assert after==read(HERE/'audit/protected_before.json'),'Protected non-Finder workspace changed'
    write(HERE/'audit/protected_after.json',after)
    finder=finder_inventory();write(HERE/'audit/finder_after.json',finder)
    initial=read(HERE/'audit/finder_before.json');write(HERE/'audit/finder_changes.json',dict(changed_paths=sorted(k for k in set(initial)|set(finder) if initial.get(k)!=finder.get(k))))
    ledger=operations();write(HERE/'operation_ledger.json',ledger)
    assert ledger['downloads']==ledger['package_installs']==0
    write(HERE/'audit.json',dict(status='PASS',development_only=True,protected_match=True,input_checks_passed=True,Part1_sealed_before_Part2=True,full_reproduction_passed=True,preparation_main_run=False,production=ledger['scopes'].get('production',{}),original_input_seals=read(HERE/'audit/prepared_input_seals.json')))
    files={str(p.relative_to(HERE)):fp(p) for p in HERE.rglob('*') if p.is_file() and p.name not in ['manifest.json','complete.json','.DS_Store','pipeline.lock'] and '.pending_' not in p.name}
    write(HERE/'manifest.json',dict(files=files))
    write(HERE/'complete.json',dict(status='complete',development_only=True,manifest_sha256=sha(HERE/'manifest.json'),audit_sha256=sha(HERE/'audit.json'),operation_ledger_sha256=sha(HERE/'operation_ledger.json'),preregistration_sha256=sha(HERE/'preregistration.json'),Part1_seal_sha256=sha(HERE/'audit/part1_sealed.json'),input_seals=read(HERE/'audit/prepared_input_seals.json'),cumulative_trials=read(HERE/'audit/trial_catalogue.json')['cumulative_trials']))
