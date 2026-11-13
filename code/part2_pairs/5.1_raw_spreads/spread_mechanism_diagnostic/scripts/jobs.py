"""Resumable fold/chunk diagnostics. No selection, trading, or test-price reader."""
from mio import *
from reader import read_sample,partial_sample,metadata,read_train,spy_returns
from sources import set_year,seed_stream,null_generator,universe
from diagnostics import fixed_spread,variance,drift_windows,annual_events,level_events,histogram,DURATION_BINS
from factors import sector_panel,leave_out,fit_residual,cumulative_on_available
from partial_ci import fit as fit_partial
import numpy as np
import pandas as pd

VIEWS=['sample','FAST_cap','FAST_X_cap','FAST_W_cap']
RELATIONS=['cross_sector','same_sector','same_subindustry']

def flags(row):
    # Source sample checkpoints carry original pool membership booleans.
    return {name:bool(row.get('in_'+name,row.get(name,False))) for name in VIEWS[1:]}

def memberships(row,partial=False):
    if row['in_random'] and (not partial or row.get('partial_weight') is not None):
        yield 'sample',float(row['partial_weight'] if partial else row['ipw'])
    if partial and row.get('partial_weight') is None:return
    for name in VIEWS[1:]:
        if row.get(name):yield name,1.

def make_context(year,rows):
    h=set_year(year);cal=h.comp.CAL;classification,membership=metadata()
    legs={r[key] for r in rows for key in ['leg_A','leg_B']}
    candidates=legs|{s for s,c in classification.items() if c.get('classification_namespace')=='GICS sub-industry'
                    and c.get('industry') and c.get('sector') and any(s in m for month,m in membership.items()
                    if f'{year-4}-01'<=month<f'{year}-01')}
    logs={};volumes={};missing=[]
    for ticker in sorted(candidates):
        path=FRESH/'data/raw/prices'/f'{ticker.lower()}_yahoo.csv'
        if not path.exists():missing.append(ticker);continue
        frame=read_train(note(path),year).reindex(cal)
        with np.errstate(invalid='ignore',divide='ignore'):logs[ticker]=np.log(frame['Adj Close'].to_numpy(float))
        volumes[ticker]=frame.Volume.to_numpy(float)
    assert legs<=logs.keys(),('Missing sampled leg',sorted(legs-logs.keys()))
    panel=sector_panel(cal,logs,classification,membership)
    return dict(year=year,cal=cal,logs=logs,volumes=volumes,classification=classification,panel=panel,
                spy=spy_returns(year,cal),missing_proxy_tickers=missing)

def matched_null(cal,a,b,year,pair):
    score=(cal.year>=year-3)&(cal.year<year)
    ra=np.diff(a[score]);rb=np.diff(b[score]);sa=ra.std(ddof=1);sb=rb.std(ddof=1)
    rho=np.corrcoef(ra,rb)[0,1];assert np.isfinite([sa,sb,rho]).all() and sa>0 and sb>0
    rng,digest=seed_stream(pair,year);h=set_year(year)
    for attempt in range(1,201):
        count('null_draws');na,nb=null_generator()(len(cal),sa,sb,rho,rng)
        try:
            spread=fixed_spread(cal,na,nb,year)
            # Eligibility is numerical F construction only; no gate or FAST screen.
            universe().make_episodes(h,cal,na,nb,year,pair,range(year-3,year))
        except (AssertionError,ValueError,FloatingPointError,np.linalg.LinAlgError):continue
        return na,nb,dict(sigma_A=sa,sigma_B=sb,rho=rho,seed_sha256=digest,attempts=attempt,status='accepted')
    return None,None,dict(sigma_A=sa,sigma_B=sb,rho=rho,seed_sha256=digest,attempts=200,status='construction_failed')

def factor_variants(context,a,b,spread):
    cal=context['cal'];year=context['year'];c=context['classification'];p=context['panel']
    sectors=list(dict.fromkeys([c[a]['sector'],c[b]['sector']]))
    factors=[context['spy']]+[leave_out(p,a,b,s)[0] for s in sectors]
    F=np.column_stack(factors);formation=spread['formation'];score=spread['score']
    residual,details,used=fit_residual(spread['r'],F,formation,score)
    result=[]
    for version,r in [('raw',spread['r']),('factor_only',residual),('factor_alpha',residual-details['intercept'])]:
        level=cumulative_on_available(np.where(np.isfinite(F).all(axis=1),r,np.nan))
        events=level_events(cal,level,used,year)
        result.append((dict(version=version,**variance(r[score],[10,60,120]),**details,
                            factor_names=['SPY',*sectors],n_events=len(events),
                            n_reverted=sum(e['reverted'] for e in events),
                            reverted_share=np.mean([e['reverted'] for e in events]) if events else np.nan),events))
    return result

def bank_add(bank,key,year,events,weight):
    if key not in bank:bank[key]=dict(deaths=np.zeros((22,DURATION_BINS)),censors=np.zeros((22,DURATION_BINS)),
                                    pair_mass=np.zeros(22),pairs=np.zeros(22,int))
    target=bank[key];d,c=histogram(events);i=year-2004
    target['deaths'][i]+=weight*d;target['censors'][i]+=weight*c
    target['pair_mass'][i]+=weight;target['pairs'][i]+=1

def process_pair(context,row,partial_weight=None,do_partial=True):
    cal=context['cal'];year=context['year'];a,b=row['leg_A'],row['leg_B']
    meta=dict(pair=row['pair'],leg_A=a,leg_B=b,year=year,era=era(year),relation=row['sector_relation'],
              in_random=bool(row['in_random']),ipw=row['ipw'],partial_weight=partial_weight,**flags(row))
    la=context['logs'][a];lb=context['logs'][b]
    na,nb,parameters=matched_null(cal,la,lb,year,row['pair'])
    out=dict(basic=[],partial=[],events=[],factor=[],trigger_bank={},factor_bank={},null=[dict(meta,**parameters)])
    for dataset,aa,bb in [('real',la,lb),('null',na,nb)]:
        if aa is None:
            if partial_weight is not None and do_partial:
                for window in ['full','A','B']:
                    out['partial'].append(dict(meta,dataset=dataset,window=window,converged=False,
                        reason='Null construction failed after 200 attempts',n_obs=0,starts=[]))
            continue
        m=dict(meta,dataset=dataset);spread=fixed_spread(cal,aa,bb,year)
        out['basic'].append(dict(m,beta=spread['beta'],**variance(spread['r'][spread['score']]),
                                 **drift_windows(cal,spread['r'],year)))
        events=annual_events(cal,aa,bb,year,row['pair'],spread,
                             [context['volumes'][a],context['volumes'][b]] if dataset=='real' else None)
        out['events'].extend(dict(m,**e) for e in events)
        for view,w in memberships(meta):
            for kind in ['volume-spike','jump','gradual']:
                if dataset=='null' and kind=='volume-spike':continue
                part=[e for e in events if e['trigger']==kind]
                bank_add(out['trigger_bank'],(view,meta['relation'],dataset,kind),year,part,w)
        try:
            factors=factor_variants(context,a,b,spread)
            record('factor_formation_window',pair=meta['pair'],test_year=year,dataset=dataset,
                   sessions_used=factors[0][0]['formation_sessions_used'],sessions_lost=factors[0][0]['formation_sessions_lost'])
        except (AssertionError,ValueError,np.linalg.LinAlgError) as error:
            # No complete-case substitution is silent. P7 coverage is reported.
            factors=[]
            for version in ['raw','factor_only','factor_alpha']:
                out['factor'].append(dict(m,version=version,status='not_estimable',reason=str(error)))
        for result,ev in factors:
            out['factor'].append(dict(m,**result,status='ok'))
            for view,w in memberships(meta):bank_add(out['factor_bank'],(view,meta['relation'],dataset,result['version']),year,ev,w)
        if partial_weight is not None and do_partial:
            for window,mask in [('full',spread['score']),('A',(cal.year>=year-3)&(cal.year<=year-2)),('B',cal.year==year-1)]:
                out['partial'].append(dict(m,window=window,**fit_partial(spread['level'][mask])))
    return out

def merge(target,part):
    for key in ['basic','partial','events','factor','null']:target[key].extend(part[key])
    for key in ['trigger_bank','factor_bank']:
        for group,value in part[key].items():
            if group not in target[key]:target[key][group]={k:v.copy() for k,v in value.items()}
            else:
                for k,v in value.items():target[key][group][k]+=v

def run_fold(year,chunk_size=100):
    guard();os.environ['MECH_SCOPE']='production';start=time.monotonic()
    ready=HERE/f'runtime/checkpoints/fold_{year}.json'
    if checkpoint(ready):print('Resumed completed fold',year,flush=True);return
    rows=read_sample(year);selected=partial_sample(rows,year)
    context=None;files=[]
    for offset in range(0,len(rows),chunk_size):
        path=HERE/f'runtime/checkpoints/{year}/chunk_{offset:05d}.pkl.gz';marker=path.with_suffix('.ready.json')
        if checkpoint(marker):files.extend([path,marker]);continue
        if context is None:context=make_context(year,rows)
        out=dict(basic=[],partial=[],events=[],factor=[],null=[],trigger_bank={},factor_bank={})
        for row in rows[offset:offset+chunk_size]:merge(out,process_pair(context,row,selected.get(row['pair'])))
        dump_gzip(path,out);receipt(marker,[path],year=year,offset=offset,rows=min(chunk_size,len(rows)-offset))
        files.extend([path,marker]);print('Checkpoint',year,offset+min(chunk_size,len(rows)-offset),'/',len(rows),flush=True)
    receipt(ready,files,year=year,rows=len(rows),partial_rows=len(selected),seconds=time.monotonic()-start)
    print('Completed mechanism fold',year,'rows',len(rows),'seconds',round(time.monotonic()-start,1),flush=True)
