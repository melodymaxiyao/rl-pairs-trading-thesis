"""Streaming master export, IPW portfolios, paired inference and frozen Part1 seal."""
from aio import *
from store import pickle_read,csv_new,daily_rows,cached,commit
from execution import RULES,BENCH,LADDER
from inference import estimate,paired,holm
import csv,gzip
import numpy as np,pandas as pd
POOLS_LIST=['P_RAND','P_EXP','P_QUAL']
MASTER=['experiment','status','period','scenario','pool','construction','fold','train_years','test_year','pair','leg_A','leg_B','category','origin','in_G','in_all_eligible','method','method_family','n_seeds','deterministic','cost_bps','pair_cost_bps','net_return_bps','gross_return_bps','cost_drag_bps','n_trades','trades_per_year','mean_holding_sessions','fraction_in_market','trade_hit_rate','pair_sharpe','max_drawdown_bps','forced_stop','forced_cap','drift_bps','timing_bps','seed_sd_net_bps','d_minus_B0','d_minus_B2','d_minus_B3','d_minus_H1s','d_rank_within_pairfold','source_file','source_row_key']
EXTRA=['anchor','alignment','world','replicate','other_bps','cell','cost','cost_case','variant','era','relation','weight','available','unavailable_reason','net_bps','gross_bps','cost_bps_total','days_in_market','hit_rate','source_fold','seed','feasible','development_only','state']

def results(part=None):
    markers=list((HERE/'runtime/cells').glob('*.ready.json'))
    def key(p):
        r=read(p);return tuple(r[k] for k in ['pool','test_year','world','replicate','variant','cell'])
    for marker in sorted(markers,key=key):
        receipt=checkpoint(marker);cell=receipt['cell']
        if part==1 and cell not in RULES+BENCH:continue
        if part==2 and cell not in LADDER:continue
        path=marker.with_name(marker.name.replace('.ready.json','.pkl.gz'))
        yield receipt,pickle_read(path)

@lru_cache(maxsize=1)
def references(pool,year,world,rep,variant,part):
    points=[];maps={};cells=RULES+BENCH+(LADDER if part is None else [])
    for cell in cells:
        value=cached(f'runtime/cells/{pool}_{year}_{cell}_{world}_{rep}_{variant}')
        if value is None:continue
        maps[cell]={(r['pair'],r['cost']):r.get('net_bps',np.nan) for r in value['scores']}
        points.extend(dict(pair=r['pair'],cost=r['cost'],cell=cell,net=r.get('net_bps',np.nan)) for r in value['scores'])
    ranks={}
    if points:
        f=pd.DataFrame(points);f['rank']=f.groupby(['pair','cost']).net.rank(ascending=False,method='min')
        ranks={(r.pair,r.cost,r.cell):r.rank for r in f.itertuples()}
    return maps,ranks

def selected(pool,cells,world='real',part=None):
    out=[]
    for info,value in results(part):
        if info['pool']!=pool or info['cell'] not in cells or info['world']!=world or info['variant']!='primary':continue
        out.extend({k:r.get(k,np.nan) for k in ['pair','test_year','weight','net_bps','cell','replicate','available','relation','era']} for r in value['scores'] if r['cost']==10)
    return pd.DataFrame(out,columns=['pair','test_year','weight','net_bps','cell','replicate','available','relation','era'])

def nullable_average(frame):
    if frame.empty:return frame
    counts=frame.groupby(['pair','test_year','cell']).size();assert (counts==2).all(),'Incomplete two-replicate null cohort'
    out=frame.groupby(['pair','test_year','cell'],as_index=False).agg(net_bps=('net_bps','mean'),weight=('weight','first'),relation=('relation','first'),era=('era','first'),finite_replicates=('net_bps','count'))
    out.loc[out.finite_replicates!=2,'net_bps']=np.nan
    return out

def family():
    rows=[]
    labels=['T1','T2','T3','T4','T5']
    for pool in POOLS_LIST:
        real=selected(pool,['R_TR_aln','R_TR_all','R_F_B2','H1s'],part=1)
        null=nullable_average(selected(pool,['R_TR_aln'],'N2',part=1));a=real[real.cell=='R_TR_aln'];n=null[null.cell=='R_TR_aln']
        contrasts=[a.assign(value=a.net_bps),paired(a,n),paired(a,real[real.cell=='R_TR_all']),paired(a,real[real.cell=='R_F_B2']),paired(a,real[real.cell=='H1s'])]
        for scheme in ['year']+(['component'] if pool=='P_EXP' else []):
            group=[]
            mapping=read(note(BASELINE/'runtime/clusters.json'))['EXP_G_cap']['pair_cluster'] if scheme=='component' else None
            for label,frame in zip(labels,contrasts):
                if mapping is not None:frame=frame.assign(cluster=frame.pair.map(mapping));assert frame.cluster.notna().all()
                group.append(dict(pool=pool,cost=10,variant='primary',id=label,family='primary' if pool=='P_RAND' and scheme=='year' else 'secondary',**estimate(frame,scheme=scheme)))
            adjusted=holm([r['p_one_sided'] for r in group])
            for r,p in zip(group,adjusted):r.update(p_Holm=float(p),supported=bool(r['estimate']>0 and p<=.05))
            rows.extend(group)
    return rows

def stream_export(folder,part):
    folder=inside(folder);folder.mkdir(parents=True,exist_ok=True)
    target=folder/'per_pair_fold_test.csv.gz';pending=target.with_name(target.name+'.pending_'+str(time.time_ns()));aggregates={};strata={};rows_written=0
    with pending.open('xb') as raw:
        with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0,filename='') as stream:
            import io
            text=io.TextIOWrapper(stream,encoding='utf-8',newline='');writer=csv.DictWriter(text,fieldnames=MASTER+EXTRA,extrasaction='ignore');writer.writeheader()
            for info,value in results(part):
                maps,ranks=references(info['pool'],info['test_year'],info['world'],info['replicate'],info['variant'],part) if info else ({},{})
                for r in value['scores']:
                    rr=dict(r,cost_bps_total=r.get('cost_bps'));rr['cost_bps']=r['cost']
                    for name,cell in [('B0','B0'),('B2','R_F_B2'),('B3','B3_TR_aln'),('H1s','H1s')]:rr['d_minus_'+name]=r.get('net_bps',np.nan)-maps.get(cell,{}).get((r['pair'],r['cost']),np.nan)
                    rr['d_rank_within_pairfold']=ranks.get((r['pair'],r['cost'],r['cell']),np.nan)
                    writer.writerow({k:'' if clean(v) is None else clean(v) for k,v in rr.items() if k in MASTER+EXTRA});rows_written+=1
                    if not r['available']:continue
                    for kind,label in [('pooled','ALL'),('era',r['era']),('relation',r['relation']),('relation','X' if r['relation']=='cross_sector' else 'W')]:
                        key=(r['pool'],r['cell'],r['variant'],r['world'],r['replicate'],r['cost'],kind,label,r['test_year'])
                        a=strata.setdefault(key,np.zeros(8));w=r['weight'];a[0]+=w;a[1:]+=w*np.array([r.get(k,0.) for k in ['net_bps','gross_bps','drift_bps','timing_bps','other_bps','n_trades','fraction_in_market']])
                for d in daily_rows(value):
                    key=(d['pool'],d['cell'],d['variant'],d['world'],d['replicate'],d['cost'],d['test_year'])
                    if key not in aggregates:aggregates[key]=dict(dates=d['dates'],den=0.,net=np.zeros(len(d['net'])),gross=np.zeros(len(d['net'])),fee=np.zeros(len(d['net'])),turnover=np.zeros(len(d['net'])))
                    a=aggregates[key];assert np.array_equal(a['dates'],d['dates']);a['den']+=d['weight']
                    for name in ['net','gross','fee','turnover']:a[name]+=d['weight']*d[name]
            text.flush();text.detach()
    if target.exists():
        from store import equal_streams
        with gzip.open(target,'rb') as a,gzip.open(pending,'rb') as b:assert equal_streams(a,b),('Existing master CSV differs; stop without overwrite',target)
    else:os.link(pending,target)
    pending.unlink()
    port=[];annual=[];grouped={}
    for key,a in sorted(aggregates.items()):
        pool,cell,variant,world,rep,cost,year=key;meta=dict(pool=pool,cell=cell,variant=variant,world=world,replicate=rep,cost=cost,test_year=year)
        for name in ['net','gross','fee','turnover']:a[name]/=a['den']
        f=pd.DataFrame(dict(date=a['dates'],net_bps=a['net'],gross_bps=a['gross'],cost_bps=a['fee'],turnover=a['turnover'])).assign(**meta);port.append(f)
        annual.append(dict(meta,net_bps=float(a['net'].sum()),turnover=float(a['turnover'].sum())))
        grouped.setdefault(key[:-1],[]).append((year,a))
    # Portfolio rows are modest: one date per cell/world, never one date per pair.
    csv_new(folder/'portfolio_daily.csv.gz',pd.concat(port,ignore_index=True) if port else [])
    ps=[]
    for key,parts in grouped.items():
        parts=sorted(parts);r=np.concatenate([a['net'] for y,a in parts]);cum=np.r_[0.,np.cumsum(r)];sd=r.std(ddof=1)
        ps.append(dict(zip(['pool','cell','variant','world','replicate','cost'],key),Sharpe=float(np.sqrt(252)*r.mean()/sd) if sd>0 else np.nan,max_drawdown_bps=float(np.min(cum-np.maximum.accumulate(cum))),turnover=float(sum(a['turnover'].sum() for y,a in parts)),net_bps=float(r.sum()),n_days=len(r),n_years=len(parts)))
    csv_new(folder/'portfolio_summary.csv',ps);csv_new(folder/'portfolio_by_year.csv',annual)
    sr=[]
    names=['pool','cell','variant','world','replicate','cost','stratum','label','test_year']
    metrics=['net_bps','gross_bps','drift_bps','timing_bps','other_bps','n_trades','fraction_in_market']
    for key,v in strata.items():sr.append(dict(zip(names,key),weight=v[0],**dict(zip(metrics,v[1:]/v[0]))))
    frame=pd.DataFrame(sr);csv_new(folder/'pair_fold_by_year.csv',frame)
    summaries=[]
    if len(frame):
        for key,g in frame.groupby(names[:-1],sort=True):
            row=dict(zip(names[:-1],key));row.update({m:float(np.average(g[m],weights=g.weight)) for m in metrics})
            if row['variant']=='primary' and row['cost']==10:
                row.update(estimate(g.rename(columns={'net_bps':'value'})))
            summaries.append(row)
    csv_new(folder/'evaluation_summary.csv',summaries)
    record('master_export',exported_pair_fold_rows=rows_written,part=part or 0)
    return dict(pair_rows=rows_written,portfolios=len(ps),strata=len(summaries))

def part1():
    marker=HERE/'audit/part1_sealed.json'
    if marker.exists():verify_part1();return
    folder=TABLES/'part1';stream_export(folder,1);csv_new(folder/'part1_family.csv',family())
    files={str(p.relative_to(HERE)):fp(p) for p in folder.rglob('*') if p.is_file()}
    from store import operations
    assert operations()['scopes'].get('production',{}).get('TD_fits',0)==0
    write(marker,dict(status='sealed_before_part2',files=files,production_TD_fits_before_seal=0))

def verify_part1():
    r=read(HERE/'audit/part1_sealed.json');actual={str(p.relative_to(HERE)) for p in (TABLES/'part1').rglob('*') if p.is_file() and p.name!='.DS_Store'}
    assert actual==set(r['files'])
    for k,v in r['files'].items():assert fp(HERE/k)==v,('Part1 sealed output changed',k)
    return r

def ladder():
    rows=[]
    for pool in POOLS_LIST:
        real=selected(pool,['R_TR_aln']+LADDER);null=nullable_average(selected(pool,LADDER,'N2'))
        comparisons=[('B3_TR_aln','R_TR_aln'),('TD8','B3_TR_aln'),('TD16_align','TD8'),('TD16_strength','TD8'),('TD8_group','TD8'),('TD8_shrunk','TD8'),('TD8','TD8_z')]
        contrasts=[(a+' minus '+b,paired(real[real.cell==a],real[real.cell==b])) for a,b in comparisons]
        contrasts += [(a+' minus N2',paired(real[real.cell==a],null[null.cell==a])) for a in LADDER]
        for comparison,frame in contrasts:
            rows.append(dict(pool=pool,comparison=comparison,**estimate(frame)))
            if pool=='P_EXP':
                mapping=read(note(BASELINE/'runtime/clusters.json'))['EXP_G_cap']['pair_cluster']
                component=frame.assign(cluster=frame.pair.map(mapping));assert component.cluster.notna().all()
                rows.append(dict(pool=pool,comparison=comparison,**estimate(component,scheme='component')))
    return rows

def luck_band():
    f=pd.read_csv(TABLES/'pair_fold_by_year.csv');rows=[]
    null=f[(f.world!='real')&(f.stratum=='pooled')]
    for key,g in null.groupby(['pool','cell','variant','world','cost','test_year']):
        assert len(g)==2
        real=f[(f.world=='real')&(f.stratum=='pooled')]
        for col,val in zip(['pool','cell','variant','cost','test_year'],[key[0],key[1],key[2],key[4],key[5]]):real=real[real[col]==val]
        rows.append(dict(zip(['pool','cell','variant','world','cost','test_year'],key),null_low=float(g.net_bps.quantile(.025)),null_high=float(g.net_bps.quantile(.975)),null_replicates=2,replicate_0=float(g.sort_values('replicate').net_bps.iloc[0]),replicate_1=float(g.sort_values('replicate').net_bps.iloc[1]),real_net=float(real.net_bps.iloc[0]) if len(real) else np.nan,interpretation='empirical range from K=2; not a precise tail estimate'))
    return rows

def policies():
    output={};lines=['# Frozen TD ensemble policies','', 'Bins are intervals; nonmonotone decisions are reported as sets, never forced into a single threshold.','']
    for path in sorted(MODELS.glob('*.ready.json')):
        info=checkpoint(path);payload=pickle_read(path.with_name(path.name.replace('.ready.json','.pkl.gz')))
        if 'ensemble' not in payload:continue
        key=path.stem.replace('.ready','');q=np.asarray(payload['ensemble']);tables=q if q.ndim==3 else q[None,:,:];records=[]
        for group,table in enumerate(tables):
            feature=payload['feature'];rows=[]
            for s in range(len(table)):
                base=s//2 if feature else s;position=base%2;bin_id=base//2;action=int(table[s,1]>table[s,0]);allowed='alignment/side-lock/forced-exit masks still apply'
                label=('aligned' if s%2==0 else 'counter') if feature==1 else ('strong |t252|>=1' if s%2==0 else 'weak |t252|<1') if feature==2 else None
                rows.append(dict(state=s,bin=['<0.5','0.5–1.5','1.5–2.5','>=2.5'][bin_id],position='in-trade' if position else 'flat',feature_value=label,Q_continue=float(table[s,0]),Q_switch=float(table[s,1]),action=('exit' if action else 'hold') if position else ('enter' if action else 'stay'),allowed=allowed))
            entry=[r['bin'] for r in rows if r['position']=='flat' and r['action']=='enter'];exit=[r['bin'] for r in rows if r['position']=='in-trade' and r['action']=='exit']
            records.append(dict(group=group,rows=rows,entry_bins=entry,exit_bins=exit,reference_L0='cross 2.0 / zero crossing',threshold_interpretation='interval/set-valued; feature-conditional when TD16'))
            lines+=['## '+key+' group '+str(group),'','| state | abs(d) | position | feature | action |','|---|---|---|---|---|']+['| '+ ' | '.join(str(r[k]) for k in ['state','bin','position','feature_value','action'])+' |' for r in rows]+['',f'Entry bins: {entry}; exit bins: {exit}. Forced zero/stop/cap/close overrides remain in force.','']
        output[key]=dict(policies=records,selected=payload.get('selected'),n0=payload.get('n0'),group_visits=payload.get('group_visits'),shrink_weights=payload.get('weights'))
    write(TABLES/'policies.json',output)
    p=HERE/'POLICIES.md'
    if not p.exists():
        with p.open('x') as f:f.write('\n'.join(lines))
    return output

def final():
    verify_part1();ready=HERE/'audit/report_ready.json'
    if checkpoint(ready):return
    stream_export(TABLES,None)
    csv_new(TABLES/'part1_family.csv',pd.read_csv(TABLES/'part1/part1_family.csv'))
    csv_new(TABLES/'ladder.csv',ladder());csv_new(TABLES/'luck_band.csv',luck_band());policies()
    limits=[]
    for path in sorted((HERE/'runtime/plans').glob('*.ready.json')):
        payload=pickle_read(path.with_name(path.name.replace('.ready.json','.pkl.gz')));limits.extend(payload['details'])
    for path in sorted((HERE/'runtime/choices').glob('*.ready.json')):
        payload=pickle_read(path.with_name(path.name.replace('.ready.json','.pkl.gz')))
        for cost,choice in payload['choices'].items():limits.append(dict(kind='B3',source=path.stem,cost=int(cost),test_rows_used=0,**choice))
    for path in sorted(MODELS.glob('*.ready.json')):
        payload=pickle_read(path.with_name(path.name.replace('.ready.json','.pkl.gz')))
        if 'selected' in payload:limits.append(dict(kind='TD_hyperparameters',source=path.stem,test_rows_used=0,selection=json.dumps(clean(payload['selected']),sort_keys=True)))
    csv_new(TABLES/'limits_and_choices.csv',limits)
    # Historical null rows are a copied continuity reference. They use their
    # original seeds/construction; never pretend they are these new N1/N2 draws.
    reference=pd.read_csv(note(RWB/'outputs/tables/per_pair_fold_null.csv.gz'))
    reference=reference[reference.pool=='EXP_G_cap'].copy()
    reference['continuity_role']='copied original random-walk null reference; different stream and construction'
    csv_new(TABLES/'continuity_random_walk_reference.csv.gz',reference)
    from accounting import run as accounting
    accounting()
    from figures import make_plots,write_results
    make_plots();write_results()
    files=[p for p in TABLES.iterdir() if p.is_file()]+[HERE/'RESULTS.md',HERE/'POLICIES.md']+list(PLOTS.glob('*.png'))
    receipt(ready,files,status='PASS',plots=5);verify_part1()
