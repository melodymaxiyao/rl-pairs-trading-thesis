"""Train-only reports and six required PNGs from completed fold artifacts."""
from universe_io import *
from checkpoints import ready,receipt
from export import csv_new,POOLS
from inference import persistence,interval
import numpy as np
import pandas as pd
from collections import Counter,defaultdict

def diagnostic_data():
    return {y:{k:v for k,v in np.load(HERE/f'runtime/parts/diagnostics_{y}.npz').items()} for y in range(2004,2026)}

def rates_table(funnel,data):
    rows=[]
    for era_name in ['ALL','E1','E2','E3','E4','E5']:
        years=[y for y in data if era_name=='ALL' or era(y)==era_name]
        for relation in ['X','W','same_sector','same_subindustry','null']:
            if relation=='null':
                n=sum(len(data[y]['null_FAST63']) for y in years)
                p=sum(int(data[y]['null_FAST63'].sum()) for y in years)
                p126=sum(int(data[y]['null_FAST126'].sum()) for y in years)
                a15=legacy=None
            else:
                mask=lambda r:r=='cross_sector' if relation=='X' else r!='cross_sector' if relation=='W' else r==relation
                selected=[r for r in funnel if r['test_year'] in years and mask(r['sector_relation'])]
                n=sum(r['eligible'] for r in selected);p=sum(r['FAST63'] for r in selected);p126=sum(r['FAST126'] for r in selected)
                a15=sum(r['FAST63_a15'] for r in selected);legacy=sum(r['FAST63_legacy'] for r in selected)
            rows.append(dict(era=era_name,relation=relation,eligible_pair_folds=n,FAST63=p,FAST126=p126,
                             FAST63_rate=p/n if n else None,FAST126_rate=p126/n if n else None,
                             FAST63_a15=a15,FAST63_legacy=legacy))
    return rows

def persistence_table(data):
    result=[];raw={}
    for era_name in ['ALL','E1','E2','E3','E4','E5']:
        years=[y for y in data if era_name=='ALL' or era(y)==era_name]
        for relation in ['null','cross_sector','same_sector','same_subindustry','W','all']:
            blocks={}
            for year in years:
                d=data[year]
                if relation=='null':
                    blocks[year]=dict(inner=d['null_inner_KM_med'],validation=d['null_validation_KM_med'],fast=d['null_inner_FAST63'])
                else:
                    mask=np.ones(len(d['KM_med']),bool) if relation=='all' else d['sector_relation']!='cross_sector' if relation=='W' else d['sector_relation']==relation
                    blocks[year]=dict(inner=d['inner_KM_med'][mask],validation=d['validation_KM_med'][mask],fast=d['inner_FAST63'][mask])
            point,draws,counts=persistence(blocks,years)
            raw[era_name,relation]=(point,draws)
            row=dict(era=era_name,sector_relation=relation,folds=len(years),bootstrap_draws=2000,seed=20261020,
                     inner_FAST_pair_folds=int(counts[0]),not_inner_FAST_pair_folds=int(counts[2]))
            for index,metric in enumerate(['p_validation_fast_given_inner_FAST','p_validation_fast_given_not_inner_FAST','persistence_difference','spearman']):
                low,high,n=interval(draws[:,index])
                row.update({metric:point[index],metric+'_ci95_low':low,metric+'_ci95_high':high,metric+'_valid_draws':n})
            if relation!='null':
                null_point,null_draws=raw[era_name,'null']
                delta=draws[:,2]-null_draws[:,2]
                low,high,n=interval(delta)
                row.update(persistence_difference_minus_null=point[2]-null_point[2],
                           persistence_difference_minus_null_ci95_low=low,persistence_difference_minus_null_ci95_high=high,
                           persistence_difference_minus_null_valid_draws=n)
            result.append(row)
    return result

def clusters(structures):
    rows=[]
    for pool in [p for p in POOLS if p.endswith('_cap')]:
        edges=[edge for s in structures for edge in s['edges'][pool]]
        parents={}
        def find(a):
            parents.setdefault(a,a)
            while a!=parents[a]:
                parents[a]=parents[parents[a]];a=parents[a]
            return a
        for a,b in edges:parents[find(a)]=find(b)
        counts=Counter(find(a) for a,b in edges);tickers=Counter(find(a) for a in parents)
        biggest=counts.most_common(1)[0][0] if counts else None
        rows.append(dict(pool=pool,components=len(counts),tickers=len(parents),pair_folds=len(edges),
                         largest_component_pair_folds=counts[biggest] if biggest else 0,
                         largest_component_pair_fold_share=counts[biggest]/len(edges) if edges else None,
                         largest_component_tickers=tickers[biggest] if biggest else 0,
                         largest_component_ticker_share=tickers[biggest]/len(parents) if parents else None))
    return rows

def make_tables(data):
    funnel=[r for y in range(2004,2026) for r in read(HERE/f'runtime/funnel_{y}.json')]
    aggregate={}
    for r in funnel:
        key=r['era'],r['sector_relation'];cell=aggregate.setdefault(key,dict(era=key[0],sector_relation=key[1]))
        for field,value in r.items():
            if field not in ['test_year','era','sector_relation']:cell[field]=cell.get(field,0)+value
    structures=[read(HERE/f'runtime/structure_{y}.json') for y in range(2004,2026)]
    rates=rates_table(funnel,data);persistence_rows=persistence_table(data);cluster=clusters(structures)
    tables={'funnel_by_era_relation':list(aggregate.values()),'funnel_by_year_relation':funnel,
            'pass_rates':rates,'inner_persistence':persistence_rows,'cluster_structure':cluster,
            'concentration':[r for s in structures for r in s['concentration']],
            'sector_pair_matrix':[r for s in structures for r in s['matrix']],
            'overlap_with_previous':[r for s in structures for r in s['overlap']]}
    files=[]
    for name,rows in tables.items():
        path=HERE/'outputs/tables'/(name+'.csv');csv_new(path,rows);files.append(path)
    return tables,files

def make_plots(data,tables):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.dpi':130,'axes.spines.top':False,'axes.spines.right':False,'font.size':10})
    folder=HERE/'outputs/plots';files=[]
    def save(fig,name):
        path=folder/name
        pending=path.with_name(path.name+'.pending_'+str(time.time_ns()))
        fig.savefig(pending,format='png',bbox_inches='tight');plt.close(fig)
        if path.exists():assert sha(path)==sha(pending),('Existing plot differs',path)
        else:os.link(pending,path)
        pending.unlink();files.append(path)
    funnel=tables['funnel_by_year_relation'];years=sorted(data)
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),sharey=True)
    for ax,label in zip(axes,['X','W']):
        selected=[r for r in funnel if (r['sector_relation']=='cross_sector')==(label=='X')]
        for field,display in [('candidates','Candidates'),('eligible','Eligible'),('FAST63','FAST63 at 2σ'),('FAST63_a15','FAST63 at 1.5σ'),('FAST63_legacy','Legacy 1.5σ')]:
            values=[sum(r[field] for r in selected if r['test_year']==y) for y in years]
            ax.plot(years,values,label=display,marker='.' if field=='FAST63' else None)
        ax.set(title='Cross-sector X' if label=='X' else 'Same-sector control W',xlabel='Nominal test year',yscale='symlog')
        ax.legend(fontsize=8)
    axes[0].set_ylabel('Pair-fold count (symmetric log scale)');save(fig,'01_funnel_by_year.png')
    fig,ax=plt.subplots(figsize=(9,4.5))
    for relation,color in [('X','#1368aa'),('W','#bd5322'),('null','#676767')]:
        points=[r for r in tables['pass_rates'] if r['relation']==relation and r['era']!='ALL']
        ax.plot([r['era'] for r in points],[100*(r['FAST63_rate'] or 0) for r in points],marker='o',label=relation,color=color)
    ax.set(ylabel='FAST63 at 2σ (% of eligible)',xlabel='Era');ax.legend();save(fig,'02_fast63_rates_by_era.png')
    matrix=Counter()
    for r in tables['sector_pair_matrix']:matrix[r['sector_A'],r['sector_B']]+=r['count']
    sectors=sorted({s for key in matrix for s in key});arr=np.zeros((len(sectors),len(sectors)))
    for (a,b),n in matrix.items():
        ia,ib=sectors.index(a),sectors.index(b);arr[ia,ib]=arr[ib,ia]=n
    fig,ax=plt.subplots(figsize=(10,8))
    if sectors:
        im=ax.imshow(arr,cmap='Blues');fig.colorbar(im,ax=ax,label='FAST_X_cap pair-folds')
        ax.set_xticks(range(len(sectors)),sectors,rotation=70,ha='right');ax.set_yticks(range(len(sectors)),sectors)
    else:ax.text(.5,.5,'No FAST_X_cap pairs',ha='center',transform=ax.transAxes)
    save(fig,'03_sector_pair_heatmap.png')
    fig,ax=plt.subplots(figsize=(10,4.5))
    for relation,color in [('X','#1368aa'),('W','#bd5322')]:
        vals=np.concatenate([d['KM_med'][(d['sector_relation']=='cross_sector')==(relation=='X')] for d in data.values()])
        finite=vals[np.isfinite(vals)]
        ax.hist(finite,bins=np.arange(0,261,5),histtype='step',density=True,label=f'{relation}: {np.isinf(vals).mean():.1%} infinite' if len(vals) else relation,color=color)
    ax.axvline(63,color='black',linestyle='--',label='63');ax.axvline(126,color='gray',linestyle=':',label='126')
    ax.set(xlabel='Annual-event KM median (sessions)',ylabel='Density among finite medians');ax.legend();save(fig,'04_KM_distribution.png')
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,before,after in zip(axes,['FAST','FAST_X_before_cap','FAST_W_before_cap'],['FAST_cap','FAST_X_cap','FAST_W_cap']):
        for pool in [before,after]:
            vals=[r['pairs'] for r in tables['concentration'] if r['pool']==pool]
            if vals:ax.hist(vals,bins=min(35,max(vals)),histtype='step',label=pool)
        ax.set(xlabel='Pairs per active ticker per fold',ylabel='Ticker-fold count',title=after);ax.legend(fontsize=8)
    save(fig,'05_ticker_degrees.png')
    fig,axes=plt.subplots(1,2,figsize=(11,5),sharex=True,sharey=True)
    rng=np.random.default_rng(20261020)
    for ax,prefix,label in zip(axes,['','null_'],['Real eligible pairs','Independent-walk null']):
        x=np.concatenate([d[prefix+'inner_KM_med'] for d in data.values()]);y=np.concatenate([d[prefix+'validation_KM_med'] for d in data.values()])
        if len(x)>30000:
            take=rng.choice(len(x),30000,replace=False);x=x[take];y=y[take]
        ax.hexbin(np.minimum(x,260),np.minimum(y,260),gridsize=45,mincnt=1,bins='log',cmap='viridis')
        ax.axvline(63,color='red',linewidth=.8);ax.axhline(63,color='red',linewidth=.8)
        ax.set(title=label,xlabel='Inner KM median; ∞ shown at 260',xlim=(-2,265),ylim=(-2,265))
    axes[0].set_ylabel('Validation-year KM median; ∞ shown at 260');save(fig,'06_inner_persistence.png')
    return files

def results_summary(tables):
    byera=[]
    for era_name in ['E1','E2','E3','E4','E5']:
        part=[r for r in tables['funnel_by_year_relation'] if r['era']==era_name]
        byera.append(dict(era=era_name,**{k:sum(r[k] for r in part) for k in ['eligible','FAST63','FAST63_a15','FAST63_legacy','FAST_cap','FAST_X_cap','FAST_W_cap','G_pass']}))
    rates={r['relation']:r for r in tables['pass_rates'] if r['era']=='ALL'}
    overlaps=tables['overlap_with_previous'];poolsummary={}
    for pool in POOLS:
        rows=[r for r in overlaps if r['pool']==pool];n=sum(r['pair_folds'] for r in rows)
        poolsummary[pool]=dict(pair_folds=n,NVDA_TSLA_share=sum(r['NVDA_TSLA_pairs'] for r in rows)/n if n else None,
                              previously_traded_pair_folds=sum(r['previously_traded'] for r in rows),
                              previously_traded_share=sum(r['previously_traded'] for r in rows)/n if n else None)
    pers=next(r for r in tables['inner_persistence'] if r['era']=='ALL' and r['sector_relation']=='all')
    sector_counts=Counter()
    for row in tables['sector_pair_matrix']:sector_counts[row['sector_A']+' / '+row['sector_B']]+=row['count']
    legacy=[read(HERE/f'audit/reproduction_{y}.json') for y in range(2004,2026)]
    return dict(by_era=byera,rates=rates,pools=poolsummary,inner_persistence=pers,
                dominant_sector_pairs=sector_counts.most_common(5),clusters=tables['cluster_structure'],
                eligible_pair_folds=sum(r['eligible'] for r in byera),FAST63=sum(r['FAST63'] for r in byera),
                legacy_comparison_rows=sum(r['legacy_rows'] for r in legacy),
                legacy_eligibility_differences=sum(r['legacy_eligibility_differences'] for r in legacy),
                legacy_G_differences=sum(r['legacy_G_differences'] for r in legacy))

def fmt(value,percent=False):
    return 'unavailable' if value is None or not np.isfinite(float(value)) else f'{value:.2%}' if percent else f'{value:.3f}'

def summary_lines(summary):
    p=summary['pools'];rates=summary['rates'];inner=summary['inner_persistence']
    return [f"1. Eligible pair-folds: {summary['eligible_pair_folds']:,}.",
            f"2. FAST63: {summary['FAST63']:,}; X {fmt(rates['X']['FAST63_rate'],True)}, W {fmt(rates['W']['FAST63_rate'],True)}, null {fmt(rates['null']['FAST63_rate'],True)}.",
            '3. Pool sizes: '+', '.join(f'{name} {p[name]["pair_folds"]:,}' for name in ['FAST_cap','FAST_X_cap','FAST_W_cap'])+'.',
            f"4. Inner persistence: probability gap {fmt(inner['persistence_difference'],True)}, versus null gap difference {fmt(inner.get('persistence_difference_minus_null'),True)}; year-block intervals are in inner_persistence.csv.",
            f"5. FAST_cap NVDA/TSLA share {fmt(p['FAST_cap']['NVDA_TSLA_share'],True)}; previously traded share {fmt(p['FAST_cap']['previously_traded_share'],True)}."]

def report_all():
    done=HERE/'audit/report_ready.json'
    if ready(done):return read(done)
    data=diagnostic_data();tables,files=make_tables(data);files+=make_plots(data,tables)
    summary=results_summary(tables)
    anchor=[]
    for relation in ['X','W']:
        for field,label in [('KM_med','primary_2'),('KM_med_a15','nonoverlap_1_5'),('KM_med_legacy','legacy_1_5')]:
            values=np.concatenate([d[field][(d['sector_relation']=='cross_sector')==(relation=='X')] for d in data.values()])
            finite=values[np.isfinite(values)]
            anchor.append(dict(relation=relation,definition=label,pair_folds=len(values),
                               median_of_finite_pair_fold_KM_medians=float(np.median(finite)) if len(finite) else np.nan,
                               infinite_KM_share=float(np.isinf(values).mean()) if len(values) else np.nan))
    path=HERE/'outputs/tables/anchor_comparison.csv';csv_new(path,anchor);files.append(path)
    summary['anchor_comparison']=anchor
    summary['G_comparison']=dict(G_passes=sum(int(d['G_pass'].sum()) for d in data.values()),
                                 FAST_and_G=sum(int((d['FAST63']&d['G_pass']).sum()) for d in data.values()))
    write(HERE/'summary.json',summary)
    text='\n\n## Computed train-only results\n\n'+'\n'.join(summary_lines(summary))+'\n\n'
    text+='| Era | Eligible | FAST63 at 2σ | 1.5σ sensitivity | Legacy 1.5σ | FAST_cap | FAST_X_cap | FAST_W_cap | G |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    for r in summary['by_era']:text+='| '+' | '.join(str(r[k]) for k in ['era','eligible','FAST63','FAST63_a15','FAST63_legacy','FAST_cap','FAST_X_cap','FAST_W_cap','G_pass'])+' |\n'
    text+='\nReal versus null rates by era, including FAST126 counts, are in `outputs/tables/pass_rates.csv`. FAST126 never defines a pool. The 1.5σ sensitivity and legacy event pass counts above quantify the anchor and event-definition effects; their medians and X/W comparisons are in the inventory and funnel tables.\n\n'
    for relation in ['X','W']:
        a=next(r for r in anchor if r['relation']==relation and r['definition']=='primary_2')
        b=next(r for r in anchor if r['relation']==relation and r['definition']=='nonoverlap_1_5')
        rate=summary['rates'][relation]
        text+=f"For {relation}, changing the non-overlapping entry anchor from 2σ to 1.5σ changes FAST63 passes from {rate['FAST63']:,} to {rate['FAST63_a15']:,}. The median of finite pair-fold KM medians changes from {fmt(a['median_of_finite_pair_fold_KM_medians'])} to {fmt(b['median_of_finite_pair_fold_KM_medians'])} sessions; infinite-median shares are {fmt(a['infinite_KM_share'],True)} and {fmt(b['infinite_KM_share'],True)}.\n\n"
    inner=summary['inner_persistence'];lo=inner['persistence_difference_ci95_low'];hi=inner['persistence_difference_ci95_high']
    text+=f"The observed validation-fast probability gap is {fmt(inner['persistence_difference'],True)} (95% year-block interval {fmt(lo,True)} to {fmt(hi,True)}). "
    text+=('The interval supports positive persistence. ' if lo is not None and np.isfinite(lo) and lo>0 else 'The interval does not establish positive persistence. ')
    nlo=inner.get('persistence_difference_minus_null_ci95_low')
    text+=('The gap exceeds the null with a positive lower confidence bound.\n\n' if nlo is not None and np.isfinite(nlo) and nlo>0 else 'The interval does not establish persistence beyond the null.\n\n')
    text+='Dominant sector pairs in FAST_X_cap: '+', '.join(f'{name} ({n:,})' for name,n in summary['dominant_sector_pairs'])+'.\n\n'
    text+=f"Legacy comparison covers {summary['legacy_comparison_rows']:,} recorded pair-folds: {summary['legacy_eligibility_differences']:,} eligibility differences and {summary['legacy_G_differences']:,} G/input differences. Missing new-rule candidates are identified explicitly. Differences never replace fresh recomputation.\n\n"
    text+=f"FAST and G overlap in {summary['G_comparison']['FAST_and_G']:,} eligible pair-folds. Of all FAST pair-folds, {summary['pools']['FAST']['previously_traded_pair_folds']:,} were previously traded. Pool overlap and previously-traded shares are in `overlap_with_previous.csv`. NVDA/TSLA shares use pair-folds containing either name as the numerator.\n\n"
    before_max=max((r['pairs'] for r in tables['concentration'] if r['pool']=='FAST'),default=0)
    text+=f"The maximum ticker degree in FAST is {before_max}; each separately capped pool has degree at most 2.\n\n"
    for name in ['FAST','FAST_cap','FAST_X_cap','FAST_W_cap']:
        p=summary['pools'][name]
        text+=f"{name}: NVDA/TSLA share {fmt(p['NVDA_TSLA_share'],True)}, previously-traded share {fmt(p['previously_traded_share'],True)}.\n\n"
    for r in summary['clusters']:text+=f"{r['pool']}: {r['components']} ticker-sharing components over all years; largest component contains {fmt(r['largest_component_pair_fold_share'],True)} of pair-folds.\n\n"
    text+='Quarterly beta CV uses separate calendar-quarter OLS slopes from the unchanged OLS helper, divided by the absolute mean; daily leg correlation uses log returns. Drift excludes annual F reset jumps. Infinite KM medians remain highest-rank ties in Spearman inference, without dropping censored observations. The bootstrap resamples nominal test-year folds, not pair rows.\n'
    # The frozen top remains byte-for-byte intact; append once after all results.
    marker='## Computed train-only results'
    current=(HERE/'README.md').read_text()
    if marker not in current:
        with (HERE/'README.md').open('a') as stream:stream.write(text)
    else:assert current.endswith(text), 'Existing README results differ'
    files+=[HERE/'summary.json',HERE/'README.md']
    receipt(done,files,PNG_outputs=6,summary_lines=summary_lines(summary));return read(done)
