"""Every path uses the unchanged sealed physical cost ledger and decomposition."""
from aio import *
from store import cached,commit
from pools import prepare
from market import context,spy
from plans import plan,b3,td,OFF
from sources import core,decompose,legacy_episode
import native
import numpy as np,pandas as pd
RULES=['R_F_B2','R_TR_all','R_TR_aln','R_R63_aln','R_F_aln']
BENCH=['B0','H1s','H1s_252','SPY']
LADDER=['B3_TR_aln','TD8','TD16_align','TD16_strength','TD8_group','TD8_shrunk','TD8_z']
ANCHORS=dict(R_F_B2='F_orig',R_TR_all='A_TR',R_TR_aln='A_TR',R_R63_aln='A_R63',R_F_aln='A_F')

def stats(e,paths,details,cost,sign):
    ledgers=[core().ledger(e,p,cost,'trend_anchor_test',details=d) for p,d in zip(paths,details)]
    result=decompose(paths,e['forward_bps'],cost,sign)
    mts=[core().metrics(l,cost)[0] for l in ledgers]
    result.update({k:float(np.mean([v[k] for v in mts])) for k in mts[0] if k not in ['n_trades']})
    result.update(net_return_bps=result['net_bps'],gross_return_bps=result['gross_bps'],cost_drag_bps=result['cost_bps'],pair_sharpe=None,max_drawdown_bps=None,n_seeds=len(paths))
    p=np.atleast_2d(paths);r=e['forward_bps'];gross=(p*r).mean(axis=0);fees=(np.abs(np.diff(p,axis=1,prepend=0))*cost/2).mean(axis=0);net=gross-fees
    cum=np.r_[0.,np.cumsum(net)];sd=float(net.std(ddof=1))
    result.update(pair_sharpe=float(np.sqrt(252)*net.mean()/sd) if sd>0 else np.nan,max_drawdown_bps=float(np.min(cum-np.maximum.accumulate(cum))),seed_sd_net_bps=float(np.std([l.net_bps.sum() for l in ledgers],ddof=1)) if len(ledgers)>1 else 0.)
    assert abs(net.sum()-result['net_bps'])<1e-9
    assert abs(result['gross_bps']-result['drift_bps']-result['timing_bps']-result['other_bps'])<1e-9
    count('ledger_evaluations',len(paths));count('identity_checks',2)
    return result,dict(dates=e['dates'].strftime('%Y-%m-%d').to_numpy(dtype='U10'),net=net,gross=gross,fee=fees,turnover=np.abs(np.diff(p,axis=1,prepend=0)).mean(axis=0)/2,paths=np.asarray(paths,np.int8),forward=np.asarray(r))

def job(pool,year,cell,world='real',rep=0,variant='primary',chunk_size=100):
    stem=f'runtime/cells/{pool}_{year}_{cell}_{world}_{rep}_{variant}';old=cached(stem)
    if old is not None:return old
    training=plan(pool,year,world,rep,variant);anchor=ANCHORS.get(cell,'A_TR');risk=training['risks'][anchor]['final'];scores=[];daily=[];rows=prepare(year)[pool]
    choice=b3(pool,year,world,rep,variant)['choices'] if cell=='B3_TR_aln' else None
    model=td(pool,year,'TD8' if cell=='TD8_z' else cell) if cell.startswith('TD') else None
    for offset in range(0,len(rows),chunk_size):
        chunk=f'{stem}/chunk_{offset:05d}';saved=cached(chunk)
        if saved is not None:scores.extend(saved['scores']);daily.extend(saved['daily']);continue
        ss=[];dd=[]
        for row in rows[offset:offset+chunk_size]:
            c=context(row,year,world,rep,'fresh_beta' if variant=='fresh_beta' else 'primary',True)
            meta=dict(experiment=HERE.name,status='development',period='2004-2025',scenario=variant,pool=pool,test_year=year,fold=year-2003,source_fold=source_fold(year),train_years=','.join(str(y) for y in range(year-3,year)),era=era(year),pair=row['pair'],leg_A=row['leg_A'],leg_B=row['leg_B'],relation=row['sector_relation'],category=row.get('category','all'),origin=row.get('origin','new'),cell=cell,method=cell,method_family='TD' if cell.startswith('TD') else 'benchmark' if cell in BENCH else 'rule',state=cell if cell.startswith('TD') else 'rule',world=world,replicate=rep,variant=variant,anchor=anchor,alignment='any' if cell in ['R_F_B2','R_TR_all','TD16_align'] else 'aligned' if cell in ['R_TR_aln','R_R63_aln','R_F_aln','B3_TR_aln'] or cell.startswith('TD') else 'benchmark',weight=row['weight'],available=not c.get('unavailable'),development_only=True,construction='F' if anchor=='F_orig' else 'fixed_beta',in_G=pool=='P_EXP',in_all_eligible=True,feasible=True,seed=-1,deterministic=cell!='TD8_z',pair_cost_bps=None,source_file=str((HERE/(stem+'.pkl.gz')).relative_to(ROOT)),source_row_key=json.dumps([pool,year,row['pair'],cell,world,rep,variant]))
            if c.get('unavailable'):
                ss.extend(dict(meta,cost=cost,cost_case=str(cost),available=False,unavailable_reason=c['reason'],net_bps=np.nan,gross_bps=np.nan,drift_bps=np.nan,timing_bps=np.nan,other_bps=np.nan,n_trades=np.nan) for cost in [5,10,20]);continue
            es=c['episodes'][anchor];e=next(e for e in es if e['year']==year)
            if cell=='R_F_B2' and pool=='P_EXP' and world=='real' and variant=='primary':e=legacy_episode(row['pair'],year,year)
            if cell=='SPY':
                dates,forward=spy(year);assert np.array_equal(dates,e['dates']);e=dict(e,forward_bps=forward)
            sign=training['signs'][row['pair']]
            fixed_paths=None
            if cell=='R_F_B2':fixed_paths=[core().threshold_targets(e,1.5,0.,risk)];count('rule_executions')
            elif cell in RULES:fixed_paths=[native.policy(e,entry=2.,aligned=cell!='R_TR_all',risk=risk)]
            elif cell in BENCH:
                side=0 if cell=='B0' else 1 if cell=='SPY' else training['sign252'][row['pair']] if cell=='H1s_252' else sign
                fixed_paths=[core().targets(e,constant=side,risk=core().OFF)];count('rule_executions')
            elif cell.startswith('TD'):
                feature=model['feature'];q=model['ensemble']
                if cell in ['TD8_group','TD8_shrunk']:q=q[e['group']]
                tables=model['models'] if cell=='TD8_z' else [q]
                fixed_paths=[native.policy(e,q=q,feature=feature,aligned=cell!='TD16_align',risk=risk) for q in tables]
            for cost in [5,10,20]:
                paths=fixed_paths
                if cell=='B3_TR_aln':
                    v=choice[str(cost)]
                    if v['no_trade']:count('rule_executions')
                    paths=[core().targets(e,constant=0,risk=core().OFF)] if v['no_trade'] else [native.policy(e,entry=v['entry'],exit=v['exit'],aligned=True,risk=risk)]
                p=[v[0] for v in paths];details=[v[1] for v in paths];result,d=stats(e,p,details,cost,sign)
                result.update(trades_per_year=result['n_trades'],mean_holding_sessions=result.get('avg_holding_sessions'),days_in_market=result.get('fraction_in_market'),hit_rate=result.get('trade_hit_rate'))
                ss.append(dict(meta,cost=cost,cost_case=str(cost),applied_cost_bps=cost,**result));dd.append(dict(meta,cost=cost,**d))
        commit(chunk,dict(scores=ss,daily=dd));scores.extend(ss);daily.extend(dd)
    value=dict(scores=scores,chunks=[f'{stem}/chunk_{i:05d}' for i in range(0,len(rows),chunk_size)])
    commit(stem,value,pool=pool,test_year=year,cell=cell,world=world,replicate=rep,variant=variant,pair_folds=len(rows))
    print('Completed',pool,year,cell,world,rep,variant,'pairs',len(rows),flush=True)
    return value
