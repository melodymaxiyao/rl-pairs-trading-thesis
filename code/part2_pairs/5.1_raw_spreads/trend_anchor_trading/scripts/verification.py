"""Exact sealed inventories and the approved historical workspace scope."""
from aio import *
from collections import Counter

INPUTS={
 'anchor_horizon_diagnostic_v1':'verify_anchor_horizon.py',
 'exp_baseline_v1':'scripts/verify_exp_baseline_resume_v3.py',
 'random_walk_benchmark_v1':'verify_rw_benchmark.py',
 'pair_property_persistence_v1':'verify_property_persistence.py',
 'fresh_holdout_data_v1':'verify_fresh_data.py',
 'spread_mechanism_diagnostic_v1':'verify_mechanism.py',
 'cross_sector_fast_universe_v1':'verify_cross_sector_universe.py'}

def paths(directory):
    result={}
    for parent,dirs,names in os.walk(directory,followlinks=False):
        parent=Path(parent)
        for name in dirs.copy():
            p=parent/name
            if p.is_symlink():result[str(p.relative_to(directory))]=p;dirs.remove(name)
        result.update({str((parent/name).relative_to(directory)):parent/name for name in names if name!='.DS_Store'})
    return result

def check_inputs(freeze=False):
    saved=HERE/'audit/input_inventories.json'
    expected=read(saved) if saved.exists() else None
    assert expected is not None or freeze
    trusted=read(ANCHOR/'audit/protected_before.json')['files'] if expected is None else None
    result={};info={}
    for name in INPUTS:
        directory=PT/name;complete=read(directory/'complete.json');assert complete['status']=='complete'
        if expected is not None:wanted=expected[name]
        else:
            manifest=read(directory/'manifest.json')['files']
            if isinstance(manifest,list):manifest={r['path']:dict(sha256=r['sha256'],size=r['bytes'],mtime_ns=r['mtime_ns']) for r in manifest}
            prefix=str(directory.relative_to(ROOT))+'/'
            wanted={k[len(prefix):]:v for k,v in trusted.items() if k.startswith(prefix)}
            wanted.update(manifest);wanted.update(complete.get('late_seal_artifacts',complete.get('late_seal_audit_artifacts',{})))
            wanted.update({k:fp(directory/k) for k in ['manifest.json','complete.json']})
        wanted={k:v for k,v in wanted.items() if Path(k).name!='.DS_Store'}
        actual=paths(directory)
        assert set(actual)==set(wanted),('Sealed inventory changed',name,'added',sorted(set(actual)-set(wanted))[:20],'missing',sorted(set(wanted)-set(actual))[:20])
        entries=list(wanted.items())
        with ThreadPoolExecutor(max_workers=1) as executor:
            for offset in range(0,len(entries),4096):
                batch=entries[offset:offset+4096]
                for (key,value),observed in zip(batch,executor.map(fp,[actual[k] for k,_ in batch])):
                    assert observed==value,('Sealed input changed',name,key)
        assert sha(directory/'manifest.json')==complete['manifest_sha256']
        hashes={v.get('sha256') for v in wanted.values()}
        for key,value in complete.items():
            if key.endswith('_sha256'):assert value in hashes,('Missing input completion seal',name,key)
        result[name]=wanted;info[name]=dict(files=len(wanted),complete_sha256=sha(directory/'complete.json'))
    if freeze:write(saved,result)
    return info

def historical_snapshot(directory):
    expected=read(directory/'audit/protected_before.json')
    current=read(HERE/'audit/protected_before.json')['files'];changed=[]
    if 'files' not in expected:
        # The oldest data-only input recorded (size, mtime_ns, mode), not hashes.
        # Preserve that exact check and bridge content to the later sealed RWB
        # SHA snapshot, already verified independently during this preflight.
        anchor=read(RWB/'audit/protected_before.json')['files'];normal={}
        for rel,value in expected.items():
            if rel.startswith('experiments/turning_point_detection/') or any(k in Path(rel).parts for k in ['.git','.venv','node_modules']):continue
            p=ROOT/rel
            if Path(rel).name=='.DS_Store':
                if not p.exists() or [p.stat().st_size,p.stat().st_mtime_ns,p.stat().st_mode]!=value:changed.append(rel)
                continue
            assert p.exists() and [p.stat().st_size,p.stat().st_mtime_ns,p.stat().st_mode]==value,('Historical metadata mismatch',directory.name,rel)
            assert rel in anchor and current.get(rel)==anchor[rel],('Historical SHA anchor mismatch',directory.name,rel)
            normal[rel]=anchor[rel]
        print('Metadata-only historical snapshot checked exactly against its original fields; SHA anchor: sealed RWB snapshot',flush=True)
        expected={'files':normal}
    for rel,value in expected['files'].items():
        if Path(rel).name=='.DS_Store':
            if not (ROOT/rel).exists() or fp(ROOT/rel)!=value:changed.append(rel)
        else:assert rel in current and current[rel]==value,('Historical non-Finder mismatch',directory.name,rel)
    own=str(directory.relative_to(ROOT))+'/'
    added=[k for k in current if k not in expected['files'] and not k.startswith(own)]
    print('Exact historical non-Finder files:',directory.name,len(expected['files']),flush=True)
    print('Changed .DS_Store paths:',changed,flush=True)
    print('Later additions by top directory:',dict(Counter(k.split('/')[0] for k in added)),flush=True)
    print('Later additions by experiment:',dict(Counter('/'.join(k.split('/')[:3]) for k in added)),flush=True)
    return expected

def finder_inventory():
    result={}
    for parent,dirs,names in os.walk(ROOT,followlinks=False):
        parent=Path(parent);dirs[:]=[n for n in dirs if n not in ['.git','.venv','node_modules'] and parent/n not in [HERE,ROOT/'experiments/turning_point_detection']]
        if '.DS_Store' in names:
            q=parent/'.DS_Store'
            if q.exists():result[str(q.relative_to(ROOT))]=fp(q)
    return result
