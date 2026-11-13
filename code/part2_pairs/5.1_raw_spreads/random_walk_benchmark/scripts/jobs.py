"""Resumable pool-fold generation then train-only selection and test execution."""
from rw_io import *
from simulation import market_parameters,matched,continuation
from sources import helpers,fast_modules,limits,original
from cohorts import tables
from execution import execute,select_B3
import numpy as np

def part_ready(pool,year,variant):return HERE/f'runtime/checkpoints/{pool}_{year}_{variant}_ready.json'

def accepted_context(pool,year,rows,K,kind,variant):
    manifests=[];accepted=[];parameters=[]
    for row in rows:
        key=hashlib.sha256((pool+'|'+row['pair']+'|'+str(year)).encode()).hexdigest()[:20]
        param=HERE/f'runtime/parts/parameters_{key}.json'
        param_ready=HERE/f'runtime/checkpoints/parameters_{key}_ready.json'
        if checkpoint(param_ready):value=read(param)
        else:
            value=dict(row,**market_parameters(row));write(param,value);receipt(param_ready,[param])
        parameters.append(value)
        for replicate in range(K):
            token=key+'_'+variant+'_'+str(replicate)
            checkpoint_path=HERE/f'runtime/checkpoints/draw_{token}.json'
            data=HERE/f'runtime/parts/draw_{token}.pkl.gz'
            if checkpoint(checkpoint_path):manifest,es=load_gzip(data)
            else:
                manifest,es=matched(value,pool,replicate,kind,variant)
                dump_gzip(data,(manifest,es));receipt(checkpoint_path,[data],accepted=manifest['accepted'])
            manifests.append(manifest)
            if es is not None:accepted.append((value,manifest,es))
    return manifests,accepted,parameters

def same_stored(actual,stored):
    if isinstance(actual,dict):return isinstance(stored,dict) and set(actual)==set(stored) and all(same_stored(v,stored[k]) for k,v in actual.items())
    if isinstance(actual,(list,tuple,np.ndarray)):return isinstance(stored,(list,tuple)) and len(actual)==len(stored) and all(same_stored(a,b) for a,b in zip(actual,stored))
    if isinstance(actual,(float,np.floating)):
        if np.isnan(actual):return stored is None
        if np.isinf(actual):return stored==('inf' if actual>0 else '-inf') or stored==actual
    return actual==stored

def verify_continuity(year,es,index,stored):
    h=helpers();module=fast_modules().numeric
    for prefix,window,inner in [('',es[:3],False),('inner_',es[:2],True)]:
        actual,_,_=module.episode_statistics(h,window,inner)
        for key,value in actual.items():
            expected=stored.get(prefix+key)
            if prefix+key not in stored:continue
            assert same_stored(value,expected),('Continuity train statistic mismatch',year,index,prefix+key,value,expected)
    value=fast_modules().events.summarize(fast_modules().events.screen_events(es[2:3],h.ho.gate().diag.dec.interval))['KM_med']
    expected=stored['validation_KM_med']
    assert same_stored(value,expected),('Continuity validation KM mismatch',year,index)
    assert es[-1]['window']=='test' and es[-1]['year']==year
    record('continuity_train_reproduced',continuity_train_pair_folds_checked=1)

def run_pool_fold(pool,year,rows,config,variant='primary'):
    if os.environ.get('RW_VALIDATION')!='1':os.environ['RW_SCOPE']='null_'+variant
    guard()
    marker=part_ready(pool,year,variant)
    if checkpoint(marker):return read(marker)
    start=time.monotonic()
    if variant=='continuity':
        manifests=[];accepted=[];parameters=[]
        # Use original verified chunk checkpoint rows, preserving all stored train statistics.
        source=FAST/'runtime/checkpoints'
        stored=[]
        for path in sorted(source.glob(f'null_{year}_*.json')):
            receipt_source=read(note(path));payload=FAST/receipt_source['payload'] if 'payload' in receipt_source else None
            if payload is None:
                files=receipt_source['files'];payload=next(FAST/k for k in files if k.endswith('.gz'))
            import gzip
            if payload.suffix=='.gz':
                with gzip.open(note(payload),'rt') as stream:body=json.load(stream)
                stored.extend(body['rows'] if isinstance(body,dict) else body)
        assert len(stored)==2000,(year,'Missing original 2000 null rows',len(stored))
        for index in range(2000):
            manifest,es=continuation(year,index)
            assert stored[index]['simulation']==index
            assert manifest['sigma_A']==stored[index]['sigma_A'] and manifest['sigma_B']==stored[index]['sigma_B']
            verify_continuity(year,es,index,stored[index])
            manifests.append(manifest);accepted.append((dict(pair=manifest['pair']),manifest,es))
    else:
        K=1 if variant=='price_martingale' else config['K']
        manifests,accepted,parameters=accepted_context(pool,year,rows,K,config['filter'],variant)
    if not accepted:
        payload=HERE/f'runtime/parts/{pool}_{year}_{variant}.pkl.gz'
        dump_gzip(payload,dict(manifest=manifests,results=[],parameters=parameters,risk=None,limits=None,B3={},reason='All requested replicates missing after rejection limits'))
        receipt(marker,[payload],pool=pool,test_year=year,variant=variant,accepted=0,missing=len(manifests),
            attempts=sum(m['attempts'] for m in manifests),executions_reported=0,elapsed_seconds=time.monotonic()-start,payload=str(payload.relative_to(HERE)))
        return read(marker)
    train=[e for row,m,es in accepted for e in es[:3]]
    risk,details=limits(train)
    if variant=='primary':choices={str(c):select_B3(train,risk,c) for c in [0,10,20]}
    else:choices={}
    q=tables(pool,year) if variant=='primary' and config.get('TD_pool') else None
    output=[]
    for row,manifest,es in accepted:
        sign=original().numerical().data.h1_sign(es[:3])[es[0]['pair']]
        cells=['B1','B2','H1s'] if variant=='continuity' else ['B2','H1s'] if variant=='price_martingale' else None
        costs=[10] if variant=='price_martingale' else [0,10,20]
        values=execute(es[-1],risk,choices,sign,q,costs,cells)
        for result in values:
            output.append(dict(pool=pool,pair=manifest['pair'],test_year=year,era=era(year),replicate=manifest['replicate'],
                variant=variant,**result))
    payload=HERE/f'runtime/parts/{pool}_{year}_{variant}.pkl.gz'
    dump_gzip(payload,dict(manifest=manifests,results=output,parameters=parameters,risk=risk,limits=details,B3=choices))
    receipt(marker,[payload],pool=pool,test_year=year,variant=variant,accepted=sum(m['accepted'] for m in manifests),
        missing=sum(not m['accepted'] for m in manifests),attempts=sum(m['attempts'] for m in manifests),
        executions_reported=len(output),elapsed_seconds=time.monotonic()-start,payload=str(payload.relative_to(HERE)))
    print('Completed',pool,year,variant,'accepted',len(accepted),'seconds',round(time.monotonic()-start,1),flush=True)
    return read(marker)
