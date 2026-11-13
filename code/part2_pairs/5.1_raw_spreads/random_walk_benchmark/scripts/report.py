"""Copied-real/matched-null inference and standalone scientific figures."""
from rw_io import *
from execution import METRICS
from inference import estimate,family
import numpy as np
import pandas as pd
import csv,gzip

def export_csv(path,rows):
    path=inside(path)
    columns=list(dict.fromkeys(k for r in rows for k in r))
    tmp=path.with_name(path.name+'.pending_'+str(time.time_ns()))
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(tmp,'wt',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for original_row in rows:
            row=clean(original_row)
            writer.writerow({k:json.dumps(v,separators=(',',':')) if isinstance(v,(list,dict)) else v for k,v in row.items()})
    if path.exists():
        # CSV contents, including gzip header metadata, are compared after decoding.
        from itertools import zip_longest
        with opener(path,'rt') as a,opener(tmp,'rt') as b:assert all(x==y for x,y in zip_longest(a,b)),('Existing CSV differs',path)
    else:os.link(tmp,path)
    tmp.unlink()

def available(frame,metric):
    if metric not in frame:return frame.iloc[:0]
    return frame[np.isfinite(pd.to_numeric(frame[metric],errors='coerce'))]

def comparisons(real,null):
    result=[];luck=[]
    # Each real pair is given equal weight; its accepted replicates are averaged first.
    averaged=null.groupby(['pool','variant','pair','test_year','era','cell','cost'],as_index=False)[METRICS].mean()
    pools=sorted(set(real.pool)|set(null.pool))
    for pool in pools:
        variants=sorted(null.loc[null.pool==pool,'variant'].unique()) or ['primary']
        views=['NEW','OLD','ALL_PAIRS'] if pool.startswith('EXP_') else ['OLD'] if pool.startswith('OLD_') else ['NULL_ONLY']
        for variant in variants:
            n=averaged[(averaged.pool==pool)&(averaged.variant==variant)]
            raw_pool=null[(null.pool==pool)&(null.variant==variant)]
            raw_cells={key:frame for key,frame in raw_pool.groupby(['cell','cost'],sort=False)}
            empty_raw=raw_pool.iloc[:0]
            for view in views:
                r=real[real.pool==pool]
                if view not in ['ALL_PAIRS','NULL_ONLY']:r=r[r.view==view]
                if view=='NULL_ONLY':r=r.iloc[:0]
                selected=n if view!='OLD' or pool.startswith('OLD_') else n.iloc[:0]
                for period in ['ALL','E1','E2','E3','E4','E5']:
                    rr=r if period=='ALL' else r[r.era==period]
                    nn=selected if period=='ALL' else selected[selected.era==period]
                    for exclude in ([False,True] if pool.startswith('OLD_') else [False]):
                        if exclude:
                            keep=lambda p:not bool(set(str(p).split('_'))&{'NVDA','TSLA'})
                            rr=rr.loc[rr.pair.map(keep).astype(bool)];nn=nn.loc[nn.pair.map(keep).astype(bool)]
                        cells=sorted(set(rr.cell)|set(nn.cell))
                        for cell in cells:
                            for cost in [0,10,20]:
                                a=rr[(rr.cell==cell)&(rr.cost==cost)]
                                b=nn[(nn.cell==cell)&(nn.cost==cost)]
                                if a.empty and b.empty:continue
                                joined=a[['pair','test_year']+METRICS].merge(b[['pair','test_year']+METRICS],on=['pair','test_year'],suffixes=('_real','_null'),validate='one_to_one')
                                names=[]
                                for metric in METRICS:
                                    joined[metric+'_delta']=joined[metric+'_real']-joined[metric+'_null']
                                    names.extend([metric+'_real',metric+'_null',metric+'_delta'])
                                stats=estimate(joined,names)
                                null_stats=estimate(b,METRICS)
                                real_stats=estimate(a,METRICS)
                                row=dict(pool=pool,view=view,variant=variant,era=period,cell=cell,cost=cost,exclude_NVDA_TSLA=exclude,
                                    real_reference_pair_folds=len(a),null_pair_folds=len(b),paired_pair_folds=len(joined),
                                    matched_cohort='NEW only' if pool.startswith('EXP_') else view)
                                for metric in METRICS:
                                    for side in ['real','null','delta']:
                                        s=stats[metric+'_'+side]
                                        row.update({f'{side}_{metric}_{k}':v for k,v in s.items()})
                                    for side,values in [('real_reference',real_stats),('null_reference',null_stats)]:
                                        row.update({f'{side}_{metric}_{k}':v for k,v in values[metric].items()})
                                for side,frame in [('real',a),('null',b)]:
                                    den=-frame.drift_bps.sum(min_count=1)
                                    row[side+'_mechanical_ratio']=frame.timing_bps.sum(min_count=1)/den if pd.notna(den) and den!=0 else None
                                result.append(row)
                                raw=raw_cells.get((cell,cost),empty_raw)
                                if period!='ALL':raw=raw[raw.era==period]
                                raw=raw.merge(b[['pair','test_year']],on=['pair','test_year'],how='inner')
                                annual=raw.groupby(['test_year','replicate']).net_bps.mean().to_numpy()
                                if len(annual):
                                    real_year=a.groupby('test_year').net_bps.mean()
                                    luck.append(dict(pool=pool,view=view,variant=variant,era=period,cell=cell,cost=cost,exclude_NVDA_TSLA=exclude,
                                        null_pool_years=len(annual),null_mean=float(annual.mean()),q025=float(np.quantile(annual,.025)),q05=float(np.quantile(annual,.05)),
                                        q50=float(np.median(annual)),q95=float(np.quantile(annual,.95)),q975=float(np.quantile(annual,.975)),
                                        share_positive=float(np.mean(annual>0)),real_equal_year_mean=float(real_year.mean()) if len(real_year) else None,
                                        real_empirical_percentile=float(np.mean(annual<=real_year.mean())) if len(real_year) else None))
    return result,luck

def volatility_drag(parameters,real):
    from scipy.stats import spearmanr
    def correlation(x,y):
        return float(spearmanr(x,y).statistic) if len(x)>1 and np.ptp(x)>0 and np.ptp(y)>0 else None
    p=pd.DataFrame(parameters).drop_duplicates(['pair','test_year'])
    p=p[p.get('benchmark_pool',pd.Series('EXP_G_cap',index=p.index))=='EXP_G_cap']
    h=real[(real.pool=='EXP_G_cap')&(real.view=='NEW')&(real.cell=='H1s')&(real.cost==10)]
    p=p.merge(h[['pair','test_year','net_bps','train_drift_sign']],on=['pair','test_year'],validate='one_to_one',suffixes=('_membership',''))
    copied=p.train_drift_sign.notna()
    assert (p.loc[copied,'real_train_drift_sign']==p.loc[copied,'train_drift_sign']).all(),'Copied H1s train sign differs'
    p['known_spread_drift_sign']=np.sign(p.annual_known_log_spread_drift_bps)
    p['sign_agrees']=p.known_spread_drift_sign==p.real_train_drift_sign
    p['implied_H1s_gross_bps']=p.real_train_drift_sign*p.annual_known_log_spread_drift_bps
    rows=p.to_dict('records');summaries=[]
    for era_name in ['ALL','E1','E2','E3','E4','E5']:
        part=p if era_name=='ALL' else p[p.test_year.map(era)==era_name]
        valid=part[(part.known_spread_drift_sign!=0)&(part.real_train_drift_sign!=0)]
        observed=part[np.isfinite(part.net_bps)]
        summaries.append(dict(era=era_name,pair_folds=len(part),nonzero_sign_pairs=len(valid),sign_agreement=float(valid.sign_agrees.mean()) if len(valid) else None,
            correlation_pair_folds=len(observed),
            spearman_known_spread_drift_vs_copied_H1s_net=correlation(observed.annual_known_log_spread_drift_bps,observed.net_bps),
            spearman_implied_H1s_gross_vs_copied_H1s_net=correlation(observed.implied_H1s_gross_bps,observed.net_bps)))
    return rows,summaries

def plots(real,null,comparison,family_rows):
    import matplotlib.pyplot as plt
    out=HERE/'outputs/plots'
    cells=['B1','B2','B3','z9/TD_z','z9/TD_ens']
    r=real[(real.pool=='EXP_G_cap')&(real.view=='NEW')&(real.cost==10)]
    n=null[(null.pool=='EXP_G_cap')&(null.variant=='primary')&(null.cost==10)]
    def save(fig,name):
        fig.tight_layout();path=out/name
        if not path.exists():fig.savefig(path,dpi=150)
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,5));x=np.arange(len(cells))
    for offset,data,label,color in [(-.18,r,'real','#4169a1'),(.18,n,'null','#cc8744')]:
        means=data.groupby('cell')[['drift_bps','timing_bps']].mean().reindex(cells)
        ax.bar(x+offset,means.drift_bps,.34,color=color,label=label+' drift')
        ax.bar(x+offset,means.timing_bps,.34,color=color,alpha=.45,label=label+' timing')
    ax.axhline(0,color='black',lw=.7);ax.set_xticks(x,cells);ax.set_ylabel('bps per pair-year');ax.legend(ncol=2)
    ax.set_title('Drift and timing: signed components; missing real B1 remains blank');save(fig,'01_drift_timing.png')
    fig,axes=plt.subplots(1,len(cells),figsize=(17,4))
    for ax,cell in zip(axes,cells):
        annual=n[n.cell==cell].groupby(['test_year','replicate']).net_bps.mean()
        ax.hist(annual,bins=20,color='#cc8744',alpha=.7)
        real_year=r[r.cell==cell].groupby('test_year').net_bps.mean()
        if len(real_year):ax.axvline(real_year.mean(),color='navy',label='real equal-year mean');ax.legend(fontsize=7)
        ax.set_title(cell);ax.set_xlabel('pool-year mean net bps')
    axes[0].set_ylabel('Null pool-years');save(fig,'02_null_net_luck_band.png')
    fig,ax=plt.subplots(figsize=(10,5));f=pd.DataFrame(comparison)
    for cell in ['B1','B2','B3','z9/TD_z','z9/TD_ens']:
        part=f[(f.pool=='EXP_G_cap')&(f.view=='NEW')&(f.variant=='primary')&(f.cell==cell)&(f.cost==10)&(f.era!='ALL')].sort_values('era')
        ax.plot(part.era,part.delta_gross_bps_estimate,marker='o',label=cell)
    ax.axhline(0,color='black',lw=.7);ax.set_ylabel('Real − matched null gross (bps)');ax.legend();save(fig,'03_excess_gross_by_era.png')
    fig,ax=plt.subplots(figsize=(7,6));part=n[n.cell=='B2']
    ax.hexbin(-part.drift_bps,part.timing_bps,gridsize=55,mincnt=1,bins='log')
    lower=min((-part.drift_bps).min(),part.timing_bps.min());upper=max((-part.drift_bps).max(),part.timing_bps.max())
    ax.plot([lower,upper],[lower,upper],color='red',lw=1);ax.set_xlabel('−drift (bps)');ax.set_ylabel('timing (bps)');save(fig,'04_timing_vs_negative_drift.png')

def headline(rows):
    f={r['id']:r for r in rows if r['scheme']=='year'}
    def fmt(r):
        return 'NA' if r.get('estimate') is None else f"{r['estimate']:.2f} [{r['ci95_low']:.2f}, {r['ci95_high']:.2f}]" if r.get('ci95_low') is not None else f"{r['estimate']:.2f} [interval undefined]"
    no_excess=all(f[k].get('ci95_low') is not None and f[k]['ci95_low']<=0<=f[k]['ci95_high'] for k in ['M2','M3'])
    reading=('H2/H3: the supported sign pattern is compatible with mechanical timing; B2/TD excess timing intervals contain zero. MB excess remains untested.' if no_excess else
             'H2/H3: the sign split alone does not identify skill; measured B2/TD excess is reported above. MB remains unbenchmarked.')
    return [f"M1: null B2 drift {fmt(f['M1']['null_drift'])}; timing {fmt(f['M1']['null_timing'])}; gross {fmt(f['M1'])}; mechanical check {f['M1']['supported']}.",
        f"M2/M3: excess timing B2 {fmt(f['M2'])}; frozen TD_ens {fmt(f['M3'])}; Holm p {f['M2']['p_Holm']:.4g}/{f['M3']['p_Holm']:.4g}.",
        f"M4: excess B2 gross {fmt(f['M4'])}; Holm p {f['M4']['p_Holm']:.4g}.",
        f"M5: excess H1s net {fmt(f['M5'])}; Holm p {f['M5']['p_Holm']:.4g}.",
        reading]

def report_all():
    ready=HERE/'audit/report_ready.json'
    if checkpoint(ready):return read(HERE/'summary.json')
    cohort=load_gzip(HERE/'runtime/cohorts.pkl.gz');real=pd.DataFrame(load_gzip(HERE/cohort['real_path']))
    manifests=[];results=[];parameters=[];acceptance=[]
    for marker in sorted((HERE/'runtime/checkpoints').glob('*_ready.json')):
        value=read(marker)
        if 'pool' not in value or 'payload' not in value:continue
        assert checkpoint(marker)
        part=load_gzip(HERE/value['payload']);manifests.extend(part['manifest']);results.extend(part['results'])
        if value['pool']=='EXP_G_cap' and value['variant']=='primary':parameters.extend(dict(r,benchmark_pool='EXP_G_cap') for r in part['parameters'])
        acceptance.append(dict(pool=value['pool'],test_year=value['test_year'],variant=value['variant'],requested=len(part['manifest']),
            accepted=value['accepted'],missing=value['missing'],attempts=value['attempts'],acceptance_rate=value['accepted']/value['attempts'],
            replicate_completion_rate=value['accepted']/len(part['manifest'])))
    for metric in METRICS:real[metric]=pd.to_numeric(real[metric],errors='raise')
    null=pd.DataFrame(results);comparison,luck=comparisons(real,null);family_rows=family(real,null)
    drag,drag_summary=volatility_drag(parameters,real)
    for name,rows in [('null_pairs_manifest.csv.gz',manifests),('per_pair_fold_null.csv.gz',results),
        ('real_vs_null.csv',comparison),('family_M.csv',family_rows),('luck_band.csv',luck),('acceptance_rates.csv',acceptance),
        ('real_train_volatility_drag.csv',drag),('real_train_volatility_drag_summary.csv',drag_summary)]:export_csv(TABLES/name,rows)
    plots(real,null,comparison,family_rows)
    summary=clean(dict(family=family_rows,requested_null_replicates=len(manifests),accepted=sum(m['accepted'] for m in manifests),
        missing=sum(not m['accepted'] for m in manifests),null_metric_rows=len(results),FAST_cap_excluded=True,volatility_drag=drag_summary,
        H2_H3_reading=headline(family_rows)[-1]))
    write(HERE/'summary.json',summary)
    text='\n\n## Computed benchmark results\n\n'+'\n\n'.join(headline(family_rows))+'\n\nAll intervals above use year blocks. Component estimates and Holm decisions are in family_M.csv. Luck bands use null pool-year means, with replicate indices retained. OLD expansion views are copied references only: their matched nulls were not requested. Old-pool costs without a sealed decomposition retain only the copied available metrics; missing splits remain NA.\n'
    totals=pd.DataFrame(acceptance).groupby(['pool','variant'])[['requested','accepted','missing','attempts']].sum()
    text+='\n| Pool | Variant | Requested | Accepted | Missing | Attempt acceptance |\n|---|---|---:|---:|---:|---:|\n'
    for (pool,variant),r in totals.iterrows():text+=f"| {pool} | {variant} | {int(r.requested)} | {int(r.accepted)} | {int(r.missing)} | {r.accepted/r.attempts:.2%} |\n"
    sens=null[(null.pool=='EXP_G_cap')&(null.variant=='price_martingale')&(null.cost==10)]
    text+='\nPrice-martingale sensitivity (K=1, descriptive only): '
    for cell in ['B2','H1s']:
        part=sens[sens.cell==cell]
        text+=f"{cell} gross {part.gross_bps.mean():.2f}, drift {part.drift_bps.mean():.2f}, timing {part.timing_bps.mean():.2f}, net {part.net_bps.mean():.2f} bps. "
    overall=next(r for r in drag_summary if r['era']=='ALL')
    text+=f"\n\nTrain-implied volatility drag: sign agreement {overall['sign_agreement']}; Spearman of known spread log drift with copied H1s net {overall['spearman_known_spread_drift_vs_copied_H1s_net']}, and of direction-aligned implied H1s gross with copied H1s net {overall['spearman_implied_H1s_gross_vs_copied_H1s_net']}. These are descriptive associations.\n"
    text+='\nEarlier real cells within the null pool-year luck band (10 bps, EXP_G_cap NEW):\n'
    for r in luck:
        if r['pool']=='EXP_G_cap' and r['view']=='NEW' and r['variant']=='primary' and r['era']=='ALL' and r['cost']==10 and r['cell'] in ['B2','B3','H1s','z9/TD_z','z9/TD_ens']:
            text+=f"\n- {r['cell']}: real equal-year mean {r['real_equal_year_mean']:.2f} bps; null central 95% [{r['q025']:.2f}, {r['q975']:.2f}]; empirical percentile {r['real_empirical_percentile']:.1%}.\n"
    byid={r['id']:r for r in family_rows if r['scheme']=='year'}
    if all(byid[k]['ci95_low'] is not None and byid[k]['ci95_low']<=0<=byid[k]['ci95_high'] for k in ['M2','M3']):
        text+='\nFor B2 and frozen TD_ens, the real-minus-null timing intervals contain zero. Their positive raw timing can be explained by the mechanical split. The earlier MB H2/H3 results cannot establish MB excess skill because MB was excluded from this benchmark.\n'
    if text not in (HERE/'README.md').read_text():
        with (HERE/'README.md').open('a') as stream:stream.write(text)
    files=list(TABLES.iterdir())+list((HERE/'outputs/plots').glob('*.png'))+[HERE/'summary.json']
    receipt(ready,files)
    return summary
