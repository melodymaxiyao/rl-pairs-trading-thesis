"""Main-only full hashes and unchanged read-only input verifiers."""
from sc_common import *
from collections import Counter
INPUTS={'trend_anchor_trading_v1':'verify_trend_anchor.py','anchor_horizon_diagnostic_v1':'verify_anchor_horizon.py','pair_property_persistence_v1':'verify_property_persistence.py'}

def fingerprint(p):
    p=Path(p)
    if p.is_symlink():return dict(symlink=os.readlink(p))
    s=p.stat();return dict(sha256=sha(p),size=s.st_size,mtime_ns=s.st_mtime_ns)

def snapshot():
    result={};dirs=[];finder={};last=time.monotonic()
    for parent,children,names in os.walk(ROOT,followlinks=False):
        parent=Path(parent);children[:]=sorted(n for n in children if n not in ['.git','.venv','node_modules'] and parent/n not in [HERE,ROOT/'experiments/turning_point_detection'])
        for n in children.copy():
            q=parent/n
            if q.is_symlink():result[str(q.relative_to(ROOT))]=fingerprint(q);children.remove(n)
            else:dirs.append(str(q.relative_to(ROOT)))
        for n in sorted(names):
            q=parent/n;key=str(q.relative_to(ROOT));value=fingerprint(q)
            if n=='.DS_Store':finder[key]=value
            else:result[key]=value
            if time.monotonic()-last>30:print('Protected SHA files',len(result),flush=True);last=time.monotonic()
    return dict(files=result,directories=sorted(dirs)),finder

def historical(directory,current):
    expected=read(directory/'audit/protected_before.json');changed=[]
    for key,value in expected['files'].items():
        if Path(key).name=='.DS_Store':
            if not (ROOT/key).exists() or fingerprint(ROOT/key)!=value:changed.append(key)
        else:assert current['files'].get(key)==value,('Historical non-Finder mismatch',directory.name,key)
    additions=[k for k in current['files'] if k not in expected['files'] and not k.startswith(str(directory.relative_to(ROOT))+'/')]
    print('Historical non-Finder exact:',directory.name,'; changed Finder paths:',changed,'; additions:',dict(Counter(k.split('/')[0] for k in additions)),flush=True)
    return dict(changed_DS_Store_paths=changed,added_by_top_directory=dict(Counter(k.split('/')[0] for k in additions)))

def verify_inputs(current):
    info={}
    for name,verifier in INPUTS.items():
        directory=PT/name;c=read(directory/'complete.json')
        assert c['status']=='complete' and sha(directory/'complete.json')==spec()['input_seals'][name]
        assert sha(directory/'manifest.json')==c['manifest_sha256']
        manifest=read(directory/'manifest.json')['files']
        if isinstance(manifest,list):manifest={r['path']:dict(sha256=r['sha256'],size=r['bytes'],mtime_ns=r['mtime_ns']) for r in manifest}
        for key,value in manifest.items():
            if Path(key).name=='.DS_Store':continue
            actual=current['files'].get(str((directory/key).relative_to(ROOT)))
            assert actual==value,('Sealed artifact mismatch',name,key)
        info[name]=historical(directory,current)
        log=HERE/f'audit/input_{name}_{time.time_ns()}.log'
        import subprocess
        with log.open('x') as f:p=subprocess.run([sys.executable,'-B',str(HERE/'scripts/verifier_entry.py'),name],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        assert p.returncode==0,('Read-only input verifier failed; original log preserved',log)
        print('PASS original input verifier',name,flush=True)
    return info

def registered():
    r=read(HERE/'audit/code_registration.json')
    import importlib.metadata
    assert sys.version.split()[0]==r['python_version']
    for name,value in r['package_versions'].items():assert importlib.metadata.version(name)==value,('Package version changed',name)
    executable={str(p.relative_to(HERE)) for folder in ['scripts','tests'] for p in (HERE/folder).rglob('*') if p.suffix in ['.py','.c']}
    assert executable=={k for k in r['files'] if k.startswith(('scripts/','tests/')) and Path(k).suffix in ['.py','.c']}
    for name,value in spec()['input_seals'].items():assert sha(PT/name/'complete.json')==value,('Input completion seal changed',name)
    for key,value in r['files'].items():assert sha(HERE/key)==value,('Own registered source changed',key)
    prefix=(HERE/'README.md').read_bytes()[:r['README_top_bytes']]
    assert hashlib.sha256(prefix).hexdigest()==r['README_top_sha256']
    for key,value in spec()['addenda'].items():assert sha(HERE/key)==value,('Frozen addendum changed',key)
    for key,value in read(HERE/'audit/source_registration.json').items():assert sha(ROOT/key)==value,('Imported source changed',key)
    return r

def preflight():
    registered();print('Full protected hashing and sealed-input verification, one process.',flush=True)
    current,finder=snapshot();path=HERE/'audit/protected_before.json'
    if path.exists():assert current==read(path),'Protected roots changed since this run started'
    else:write(path,current);write(HERE/'audit/finder_before.json',finder)
    info=verify_inputs(current);write(HERE/'audit/preflight_ready.json',dict(status='PASS',input_seals=spec()['input_seals'],scope=info));return current
