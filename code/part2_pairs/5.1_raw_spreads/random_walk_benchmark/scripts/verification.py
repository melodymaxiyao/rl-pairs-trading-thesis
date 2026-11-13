"""Exact input inventories and the approved historical workspace scope."""
from rw_io import *
from collections import Counter

INPUTS={
    'exp_baseline_v1':'scripts/verify_exp_baseline_resume_v3.py',
    'cross_sector_fast_universe_v1':'verify_cross_sector_universe.py',
    'learner_noise_robustness_v1':'scripts/verify_learner_noise.py',
    'state_ablation_v1':'scripts/verify_state_ablation.py',
    'fresh_holdout_data_v1':'verify_fresh_data.py',
    'trend_exposure_diagnostic_v1':'scripts/verify_trend_resume_v2.py'}

def paths(directory):
    result={}
    for parent,dirs,names in os.walk(directory,followlinks=False):
        parent=Path(parent)
        for name in dirs.copy():
            path=parent/name
            if path.is_symlink():result[str(path.relative_to(directory))]=path;dirs.remove(name)
        result.update({str((parent/name).relative_to(directory)):parent/name for name in names})
    return result

def check_inputs(freeze=False):
    frozen=HERE/'audit/input_inventories.json'
    expected=read(frozen) if frozen.exists() else None
    assert expected is not None or freeze
    # The completed fast-universe protected snapshot predates this task and
    # captures complete inventories of the older sealed inputs, including caches.
    historical=read(FAST/'audit/production_protected_before.json')['files']
    result={}
    for name in INPUTS:
        directory=PT/name;complete=read(directory/'complete.json')
        assert complete['status']=='complete',(name,'incomplete')
        manifest=read(directory/'manifest.json')['files']
        if isinstance(manifest,list):
            manifest={r['path']:dict(sha256=r['sha256'],size=r['bytes'],mtime_ns=r['mtime_ns']) for r in manifest}
        prefix=str(directory.relative_to(ROOT))+'/'
        wanted={k[len(prefix):]:v for k,v in historical.items() if k.startswith(prefix)}
        wanted.update(manifest)
        if name=='cross_sector_fast_universe_v1':
            aliases=read(HERE/'addendum_002_inventory_representation.json')['preexisting_aliases']
            before=read(HERE/'audit/protected_before.json')['files']
            for rel,value in aliases.items():
                assert before[prefix+rel]==value,('Alias absent at task baseline',rel)
                wanted[rel]=value
        for rel in ['manifest.json','complete.json']:wanted[rel]=fp(directory/rel)
        if expected is not None:wanted=expected[name]
        actual=paths(directory)
        assert set(actual)==set(wanted),('Sealed inventory changed',name,'added',sorted(set(actual)-set(wanted))[:20],'missing',sorted(set(wanted)-set(actual))[:20])
        for rel,value in wanted.items():assert fp(actual[rel])==value,('Sealed input changed',name,rel)
        assert sha(directory/'manifest.json')==complete['manifest_sha256']
        hashes={v.get('sha256') for v in wanted.values()}
        for key,value in complete.items():
            if key.endswith('_sha256'):assert value in hashes,('Completion output seal missing',name,key)
        result[name]=wanted
    if freeze:write(frozen,result)
    return {name:dict(files=len(v),complete_sha256=v['complete.json']['sha256']) for name,v in result.items()}

def historical_snapshot(directory):
    expected=read(directory/'audit/protected_before.json');changed=[]
    for rel,value in expected['files'].items():
        path=ROOT/rel
        if path.name=='.DS_Store':
            if not path.exists() or fp(path)!=value:changed.append(rel)
        else:assert (path.exists() or path.is_symlink()) and fp(path)==value,('Historical non-Finder mismatch',directory.name,rel)
    current=read(HERE/'audit/protected_before.json')['files']
    prefix=str(directory.relative_to(ROOT))+'/'
    added=[p for p in current if p not in expected['files'] and not p.startswith(prefix)]
    print('Historical scope:',directory.name,'exact non-Finder files;', 'changed Finder paths',changed,flush=True)
    print('Additions by top directory:',dict(Counter(p.split('/')[0] for p in added)),flush=True)
    print('Additions by experiment:',dict(Counter('/'.join(p.split('/')[:3]) for p in added)),flush=True)
    return expected

def finder_inventory():
    result={}
    for parent,dirs,names in os.walk(ROOT,followlinks=False):
        parent=Path(parent);dirs[:]=[n for n in dirs if n not in ['.git','.venv','node_modules'] and parent/n not in [HERE,ROOT/'experiments/turning_point_detection']]
        if '.DS_Store' in names:
            p=parent/'.DS_Store'
            if p.exists():result[str(p.relative_to(ROOT))]=fp(p)
    return result
