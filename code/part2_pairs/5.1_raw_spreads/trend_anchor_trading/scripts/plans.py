"""Train-only pooled risk limits, B3 selection and unchanged v3 selection body."""
from aio import *
from store import cached,commit
from market import context
from pools import prepare
from sources import eda,core,baseline_sources,legacy_risk
import native
import numpy as np

WORLD=[('real',0),('N1',0),('N1',1),('N2',0),('N2',1)]
OFF=dict(H=0,stop=0.,lock=False,incremental=False,nbins=9)

def contexts(pool,year,world,rep,variant):
    for row in prepare(year)[pool]:
        value=context(row,year,world,rep,'fresh_beta' if variant=='fresh_beta' else 'primary',False)
        if value.get('unavailable'):
            record('null_train_world_unavailable',null_train_unavailable_members=1,pool=pool,test_year=year,world=world,replicate=rep,pair=row['pair'])
            continue
        yield row,value

def plan(pool,year,world='real',rep=0,variant='primary'):
    stem=f'runtime/plans/{pool}_{year}_{world}_{rep}_{variant}';old=cached(stem)
    if old is not None:return old
    episodes={name:[] for name in ['A_TR','A_R63','A_F','F_orig']};signs={};groups={};sign252={}
    for row,c in contexts(pool,year,world,rep,variant):
        for name in episodes:episodes[name].extend(c['episodes'][name])
        rr=c['r'][c['cal'].year>=year-3];signs[row['pair']]=int(np.sign(np.nansum(rr)))
        tail=rr[-252:];sign252[row['pair']]=int(np.sign(tail.mean()))
        sd=tail.std(ddof=1);groups[row['pair']]=int(abs(tail.mean()*np.sqrt(252)/sd)>=1) if sd>0 else 0
    risks={};details=[]
    for name,es in episodes.items():
        risks[name]={}
        for phase,end in [('inner',year-1),('final',year)]:
            subset=[e for e in es if e['year']<end]
            if not subset:
                risks[name][phase]=dict(OFF);continue
            events=eda().entry_events(subset,1.5);r=eda().derive_limits(events)
            risk=dict(H=int(r['H']),stop=float(r['z_stop']),lock=True,incremental=False,nbins=9)
            if name=='F_orig' and pool=='P_EXP' and world=='real' and phase=='final' and variant=='primary':
                risk=dict(legacy_risk(year));r=dict(r,continuity_uses_original_full_EXP_G_cap_risk=True)
            if variant=='no_forced':risk=dict(OFF)
            risks[name][phase]=risk;details.append(dict(pool=pool,test_year=year,world=world,replicate=rep,variant=variant,anchor=name,phase=phase,test_rows_used=0,**{**r,**risk}))
            count('derive_limits_calls')
    return commit(stem,dict(risks=risks,details=details,signs=signs,sign252=sign252,groups=groups,train_signature=arrays_hash(episodes['A_TR'])))

def b3(pool,year,world='real',rep=0,variant='primary'):
    assert checkpoint(HERE/'audit/part1_sealed.json'),'Part1 must be sealed before Part2 selections'
    stem=f'runtime/choices/{pool}_{year}_{world}_{rep}_{variant}';old=cached(stem)
    if old is not None:return old
    risk=plan(pool,year,world,rep,variant)['risks']['A_TR']['final'];es=[]
    for row,c in contexts(pool,year,world,rep,variant):es.extend(c['episodes']['A_TR'])
    choices={};grid=[]
    for cost in [5,10,20]:
        candidates=[dict(entry=0.,exit=0.,no_trade=True,mean_train_net_bps=0.)]
        for entry in [1.5,2.,2.5]:
            for exit in [0.,.5]:
                values=[]
                for e in es:
                    p,d=native.policy(e,entry=entry,exit=exit,aligned=True,risk=risk)
                    values.append(float(np.sum(p*e['forward_bps']-cost*np.abs(np.diff(p,prepend=0))/2)))
                candidates.append(dict(entry=entry,exit=exit,no_trade=False,mean_train_net_bps=float(np.mean(values)) if values else 0.))
        choices[str(cost)]=max(candidates,key=lambda r:r['mean_train_net_bps']);grid.extend(dict(cost=cost,**r) for r in candidates)
        record('B3_selection',B3_selections=1,pool=pool,test_year=year,world=world,replicate=rep,variant=variant,cost=cost,test_rows_used=0)
    return commit(stem,dict(choices=choices,grid=grid,test_rows_used=0))

def td(pool,year,cell):
    assert checkpoint(HERE/'audit/part1_sealed.json')
    stem=f'outputs/models/{pool}_{year}_{cell}';old=cached(stem)
    if old is not None:return old
    feature=1 if cell=='TD16_align' else 2 if cell=='TD16_strength' else 0
    selected=prepare(year);wanted=set(selected['training_subset']) if pool=='P_RAND' else {r['pair'] for r in selected[pool]}
    es=[]
    for row,c in contexts(pool,year,'real',0,'primary'):
        if row['pair'] in wanted:es.extend(c['episodes']['A_TR'])
    risk=plan(pool,year)['risks']['A_TR'];base=baseline_sources();original_fit=core().fit
    def producer(train,tag):
        if not train or not any(e['year']==year-1 for e in train) or not any(e['year']<year-1 for e in train):return None
        def fit_adapter(a,b,alpha,gamma,seed,label,risk=None,**kw):
            return native.fit(a,b,alpha,gamma,seed,risk,feature,cell!='TD16_align',**kw)
        core().fit=fit_adapter
        try:
            pairs=sorted({e['pair'] for e in train})
            unit=f'{pool}_{year}_{tag}'
            out=base.original_td(train,dict(inner=pairs,final=pairs,inner_fallback=False),risk,source_fold(year),unit,'F','F_all')
            from store import pickle_read
            visits_by_seed={}
            for marker in (HERE/'runtime/fit_cache'/unit).glob('*_ready.json'):
                info=read(marker)
                if info['signature']['validation']==arrays_hash([]):
                    saved=pickle_read(HERE/info['pickle'])
                    visits_by_seed[int(saved['meta']['seed'])]=saved['visits'].sum(axis=1)
            expected=[int(np.random.SeedSequence([20261020,source_fold(year),s,1,0]).generate_state(1,dtype=np.uint64)[0]) for s in range(10)]
            assert set(visits_by_seed)==set(expected),'Missing final state visit counts'
            out['state_visits']=[visits_by_seed[s] for s in expected]
            return out
        finally:core().fit=original_fit
    if cell in ['TD8_group','TD8_shrunk']:
        shared=td(pool,year,'TD8');poolq=shared['ensemble'];groupq=[];vis=[];groupmeta=[]
        for g in [0,1]:
            sub=td_group(pool,year,g,es,risk,producer)
            if sub is None:
                groupq.append(poolq);vis.append(np.zeros(8));groupmeta.append(dict(group=g,fallback='empty inner/validation -> pool'))
                record('empty_group_fallback',empty_group_fallbacks=1,pool=pool,test_year=year,group=g)
            else:
                groupq.append(np.asarray(sub['models']).mean(axis=0));vis.append(np.asarray(sub['state_visits']).mean(axis=0));groupmeta.append(sub['selected'])
        from inference import shrink
        visit=np.asarray(vis);shrunk,weights,n0=shrink(np.asarray(groupq),poolq,visit)
        result=dict(ensemble=np.asarray(groupq) if cell=='TD8_group' else shrunk,group_tables=np.asarray(groupq),shrunk_tables=shrunk,group_visits=visit,n0=n0,weights=weights,selected=groupmeta,feature=0,training_pairs=sorted(wanted))
    else:
        state_visits=[]
        # A wrapper records final visits without changing the original select body.
        raw_fit=native.fit
        def tracked(a,b,alpha,gamma,seed,risk,feature=0,aligned=True,**kw):
            out=raw_fit(a,b,alpha,gamma,seed,risk,feature,aligned,**kw)
            if not b:state_visits.append(out['visits'].sum(axis=1))
            return out
        native.fit=tracked
        try:out=producer(es,cell)
        finally:native.fit=raw_fit
        assert out is not None,('Empty TD pool/inner chain',pool,year,cell)
        models=np.asarray(out['models']);result=dict(models=models,ensemble=models.mean(axis=0),state_visits=out['state_visits'],selected=out['selected'],candidates=out['candidates'],convergence=out['conv'],feature=feature,training_pairs=sorted(wanted))
    return commit(stem,result,training_pairs=len(wanted),test_rows_used=0)

def td_group(pool,year,g,es,risk,producer):
    stem=f'outputs/models/{pool}_{year}_group_{g}';old=cached(stem)
    if old is not None:return old
    state_visits=[];raw=native.fit
    def tracked(a,b,alpha,gamma,seed,risk,feature=0,aligned=True,**kw):
        out=raw(a,b,alpha,gamma,seed,risk,feature,aligned,**kw)
        if not b:state_visits.append(out['visits'].sum(axis=1))
        return out
    native.fit=tracked
    try:out=producer([e for e in es if e['group']==g],f'group_{g}')
    finally:native.fit=raw
    if out is None:return None
    return commit(stem,out,test_rows_used=0)
