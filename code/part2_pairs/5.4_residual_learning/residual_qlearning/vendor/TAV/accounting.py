"""Exhaustive trial catalogue; original DSR and CSCV bodies, without training."""
from aio import *
from store import csv_new,cached,commit
from sources import original_inference
from execution import RULES,LADDER
import numpy as np,pandas as pd

def catalogue():
    path=HERE/'audit/trial_catalogue.json'
    if path.exists():return read(path)
    inputs=[];base=0
    for directory in sorted(PT.iterdir()):
        if directory==HERE or not (directory/'complete.json').is_file():continue
        complete=read(directory/'complete.json')
        if complete.get('status')!='complete':continue
        candidates=[(directory/'complete.json',complete.get('cumulative_trials',0))]
        for p in (directory/'outputs/tables').glob('DSR*.json'):
            data=read(p);rows=data if isinstance(data,list) else [data]
            Ns=[r.get('N',0) for r in rows if isinstance(r,dict)]
            candidates.append((p,max(Ns,default=0)))
        for p,n in candidates:
            if n:inputs.append(dict(path=str(p.relative_to(ROOT)),N=int(n),sha256=sha(p)));base=max(base,int(n))
    assert base>=218,'Latest sealed trial accounting not found'
    cells=[dict(pool=pool,cell=cell,cost=cost,variant=variant) for pool in ['P_RAND','P_EXP','P_QUAL'] for cell in RULES+LADDER for cost in [5,10,20] for variant in ['primary','no_forced','fresh_beta']]
    value=dict(prior_cumulative_trials=base,source_accounting=inputs,new_trials=len(cells),cumulative_trials=base+len(cells),cells=cells,convention='Every task pool x strategy cell x cost x variant counts; seeds, folds, pairs, nulls, benchmarks and subgroup displays do not. Repeated reference configurations are conservatively counted again.')
    write(path,value);return value

def run():
    old=cached('runtime/accounting/final')
    if old is not None:
        csv_new(TABLES/'dsr_pbo.csv',old);return
    cat=catalogue();source=original_inference();daily=pd.read_csv(TABLES/'portfolio_daily.csv.gz');daily=daily[(daily.world=='real')&daily.cell.isin(RULES+LADDER)]
    years=pd.read_csv(TABLES/'pair_fold_by_year.csv');years=years[(years.world=='real')&(years.stratum=='pooled')&years.cell.isin(RULES+LADDER)]
    rows=[]
    for pool in ['GLOBAL','P_RAND','P_EXP','P_QUAL']:
        d=daily if pool=='GLOBAL' else daily[daily.pool==pool];a=years if pool=='GLOBAL' else years[years.pool==pool]
        series=[];names=[];annual=[];incomplete=[]
        for key,g in d.groupby(['pool','cell','cost','variant'],sort=True):
            name='|'.join(map(str,key));f=g.sort_values('date');series.append(f.set_index('date').net_bps);names.append(name)
            vals=a[(a.pool==key[0])&(a.cell==key[1])&(a.cost==key[2])&(a.variant==key[3])].set_index('test_year').net_bps.reindex(YEARS)
            annual.append(vals.to_numpy(float))
            if vals.isna().any():incomplete.append(name)
        if series:
            matrix=pd.concat(series,axis=1);matrix.columns=names
            common=matrix.dropna();sd=common.std(ddof=1);sr=common.mean()/sd;valid=np.isfinite(sr)
            if valid.sum()>=2:
                best=sr[valid].idxmax();x=common[best].to_numpy();v=x-x.mean();m2=np.mean(v*v);skew=float(np.mean(v**3)/m2**1.5);kurt=float(np.mean(v**4)/m2**2);variance=float(sr[valid].var(ddof=1))
                for N in [len(names),cat['cumulative_trials']]:
                    out=source.dsr(float(sr[best]),len(x),skew,kurt,variance,N);rows.append(dict(kind='DSR',pool=pool,best_cell_for_accounting_only=best,**out,selects_no_strategy=True))
                    record('DSR',DSR_evaluations=1)
            else:rows.append(dict(kind='DSR',pool=pool,status='NA fewer than two finite Sharpe cells',DSR=None))
        required=sum(1 for r in cat['cells'] if pool=='GLOBAL' or r['pool']==pool)
        if annual and len(annual)==required and not incomplete:
            out,splits=source.pbo(np.asarray(annual));assert out['n_splits']==705432
            commit(f'runtime/accounting/CSCV_{pool}',dict(summary=out,splits=splits,cells=names,annual_matrix=np.asarray(annual)))
            rows.append(dict(kind='PBO',pool=pool,PBO=out['PBO'],n_splits=out['n_splits'],n_cells=len(names),status='computed',years='2004-2025'))
            record('PBO',CSCV_splits=out['n_splits'])
        else:rows.append(dict(kind='PBO',pool=pool,status='NA incomplete annual cell matrix; no imputation',PBO=None,requested_cells=required,available_cells=len(annual),incomplete_cells=json.dumps(incomplete)))
    commit('runtime/accounting/final',rows);csv_new(TABLES/'dsr_pbo.csv',rows)
