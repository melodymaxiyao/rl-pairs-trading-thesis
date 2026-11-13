"""Five fixed plots and a plain-language development-only research result."""
from aio import *
from store import csv_new
import numpy as np,pandas as pd

def save(fig,name):
    path=PLOTS/name
    if not path.exists():
        pending=path.with_name(path.stem+'.pending_'+str(time.time_ns())+'.png')
        fig.savefig(pending,dpi=160,bbox_inches='tight');os.link(pending,path);pending.unlink()
    import matplotlib.pyplot as plt
    plt.close(fig)

def make_plots():
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    summary=pd.read_csv(TABLES/'evaluation_summary.csv');f=summary[(summary.variant=='primary')&(summary.cost==10)&(summary.stratum=='pooled')]
    cells=['R_F_B2','R_TR_all','R_TR_aln','R_R63_aln','R_F_aln']
    fig,axes=plt.subplots(1,3,figsize=(16,5),sharey=True)
    for ax,pool in zip(axes,['P_RAND','P_EXP','P_QUAL']):
        for offset,world,color in [(-.22,'real','black'),(0,'N1','tab:blue'),(.22,'N2','tab:orange')]:
            g=f[(f.pool==pool)&(f.world==world)].groupby('cell').net_bps.mean().reindex(cells)
            ax.bar(np.arange(len(cells))+offset,g,width=.2,label=world,color=color,alpha=.8)
            limits=f[(f.pool==pool)&(f.world==world)].groupby('cell')[['ci95_low','ci95_high']].mean().reindex(cells)
            if world=='real':
                ax.vlines(np.arange(len(cells))+offset,limits.ci95_low,limits.ci95_high,color=color)
            else:ax.fill_between(np.arange(len(cells)),limits.ci95_low,limits.ci95_high,color=color,alpha=.12)
        ax.axhline(0,color='gray',lw=.8);ax.set_xticks(range(len(cells)),cells,rotation=50,ha='right');ax.set_title(pool);ax.set_ylabel('IPW net bps / pair-year')
    axes[0].legend();fig.suptitle('Development evidence: real and matched nulls; 10 bps');fig.tight_layout();save(fig,'01_net_null_bands.png')
    daily=pd.read_csv(TABLES/'portfolio_daily.csv.gz');fig,axes=plt.subplots(1,3,figsize=(16,5))
    for ax,pool in zip(axes,['P_RAND','P_EXP','P_QUAL']):
        sub=daily[(daily.pool==pool)&(daily.world=='real')&(daily.variant=='primary')&(daily.cost==10)]
        for cell in ['R_TR_aln','R_TR_all','R_F_B2','H1s','SPY']:
            g=sub[sub.cell==cell].sort_values('date');ax.plot(pd.to_datetime(g.date),g.net_bps.cumsum(),label=cell)
        ax.set_title(pool);ax.set_ylabel('Cumulative additive net bps')
    axes[0].legend(fontsize=8);fig.tight_layout();save(fig,'02_cumulative_net.png')
    fig,ax=plt.subplots(figsize=(10,5));eras=['E1','E2','E3','E4','E5']
    sub=summary[(summary.pool=='P_RAND')&(summary.cell=='R_TR_aln')&(summary.variant=='primary')&(summary.cost==10)&(summary.stratum=='era')]
    for j,world in enumerate(['real','N2']):
        g=sub[sub.world==world].groupby('label').net_bps.mean().reindex(eras);ax.bar(np.arange(5)+(j-.5)*.35,g,width=.35,label=world)
    ax.set_xticks(range(5),['2004–08','2009–12','2013–17','2018–20','2021–25']);ax.axhline(0,color='gray');ax.set_ylabel('IPW net bps / pair-year');ax.legend();ax.set_title('P_RAND trend/aligned rule; 2009–12 highlighted in the results');fig.tight_layout();save(fig,'03_era_real_N2.png')
    fig,ax=plt.subplots(figsize=(12,5));cells=['R_TR_aln','B3_TR_aln','TD8','TD16_align','TD16_strength','TD8_group','TD8_shrunk'];g=f[(f.pool=='P_RAND')&(f.world=='real')].set_index('cell').reindex(cells)
    ax.bar(range(len(cells)),g.net_bps,color='tab:blue');ax.vlines(range(len(cells)),g.ci95_low,g.ci95_high,color='black');ax.set_xticks(range(len(cells)),cells,rotation=30,ha='right');ax.axhline(0,color='gray');ax.set_ylabel('IPW net bps / pair-year');ax.set_title('Q-learning ladder: development, year-block intervals');fig.tight_layout();save(fig,'04_ladder.png')
    pol=read(TABLES/'policies.json');keys=[k for k in pol if k.startswith('P_RAND_2004_') and any(k.endswith(v) for v in ['TD8','TD16_align','TD16_strength','TD8_group','TD8_shrunk'])]
    fig,axes=plt.subplots(1,max(1,len(keys)),figsize=(max(10,len(keys)*3),5));axes=np.atleast_1d(axes)
    for ax,key in zip(axes,keys):
        rec=pol[key]['policies'];mat=np.array([[int(r['action'] in ['enter','exit']) for r in group['rows']] for group in rec]);ax.imshow(mat.T,aspect='auto',vmin=0,vmax=1,cmap=ListedColormap(['#dde7ee','#205fa8']));ax.set_title(key.replace('P_RAND_2004_',''));ax.set_ylabel('State');ax.set_xlabel('Group (pool=0)')
    if not keys:axes[0].text(.5,.5,'No estimable TD policy',ha='center')
    fig.suptitle('Fixed first-fold illustration: 0=stay/hold, 1=enter/exit; masks still apply');fig.tight_layout();save(fig,'05_policies.png')

def write_results():
    family=pd.read_csv(TABLES/'part1_family.csv');f=family[(family.pool=='P_RAND')&(family.scheme=='year')].set_index('id');portfolio=pd.read_csv(TABLES/'portfolio_summary.csv');annual=pd.read_csv(TABLES/'pair_fold_by_year.csv');ladder=pd.read_csv(TABLES/'ladder.csv');accounting=pd.read_csv(TABLES/'dsr_pbo.csv')
    def fmt(v):return 'NA' if not np.isfinite(v) else f'{v:.2f}'
    def desc(k):
        r=f.loc[k];return f"{fmt(r['estimate'])} [{fmt(r.ci95_low)}, {fmt(r.ci95_high)}] bps; Holm p {r.p_Holm:.4g}; {'supported' if r.supported else 'not supported'}"
    era=annual[(annual.pool=='P_RAND')&(annual.cell=='R_TR_aln')&(annual.world=='real')&(annual.variant=='primary')&(annual.cost==10)&(annual.stratum=='pooled')&annual.test_year.between(2009,2012)]
    value=float(np.average(era.net_bps,weights=era.weight)) if len(era) else np.nan
    p=portfolio[(portfolio.pool=='P_RAND')&(portfolio.world=='real')&(portfolio.variant=='primary')&(portfolio.cost==10)].set_index('cell')
    delta=ladder[ladder.pool=='P_RAND']
    ladder_text='; '.join(f"{r.comparison}: {fmt(r.estimate)} [{fmt(r.ci95_low)}, {fmt(r.ci95_high)}]" for r in delta.itertuples() if 'N2' not in r.comparison and 'group' not in r.comparison and 'shrunk' not in r.comparison)
    groups='; '.join(f"{r.comparison}: {fmt(r.estimate)} [{fmt(r.ci95_low)}, {fmt(r.ci95_high)}]" for r in delta.itertuples() if 'minus TD8' in r.comparison and ('group' in r.comparison or 'shrunk' in r.comparison))
    global_rows=accounting[accounting.pool=='GLOBAL'];account_text='; '.join(f"{r.kind}: {fmt(getattr(r,'DSR' if r.kind=='DSR' else 'PBO',np.nan))}" for r in global_rows.itertuples())
    lines=[f'T1/T2 P_RAND/10bps: net {desc("T1")}; excess over N2 {desc("T2")}.',f'T3/T4/T5: {desc("T3")}; {desc("T4")}; {desc("T5")}.',f'2009–12 net {fmt(value)} bps/pair-year; portfolio Sharpe R_TR_aln {fmt(p.loc["R_TR_aln","Sharpe"])} versus SPY {fmt(p.loc["SPY","Sharpe"])}.',f'Ladder differences (bps/pair-year): {ladder_text}. TD8 entry/exit bins, including nonmonotone policies, are in POLICIES.md and policies.json.',f'Grouped comparisons: {groups}. Global accounting {account_text}; all evidence remains development.']
    text='# Trend-anchor trading results\n\n'+'\n\n'.join(lines)+'\n\nT4 compares the old design as a whole with the new design as a whole. A relative-to-null advantage and positive absolute net are separate questions. IPW is retained for P_QUAL; its top-quintile cut is unweighted. Null luck bands use K=2 per pair-fold and have limited tail resolution. All test years were previously inspected; these results do not establish fresh out-of-sample confirmation. Original B2 and native producer continuity checks must pass before these results are sealed.\n\nEvery TD policy and train-selected B3 choice is exported, including counterintuitive or nonmonotone actions; no single threshold is invented for a nonmonotone table. See POLICIES.md for all fold/group/state policies and limits_and_choices.csv for training choices. Costs omit borrow and hedge rebalancing; fixed beta can become stale.\n'
    path=HERE/'RESULTS.md'
    if not path.exists():
        with path.open('x') as out:out.write(text)
    else:assert path.read_text()==text
    write(HERE/'summary.json',dict(lines=lines))
    marker='## Completed test-year evaluation'
    if marker not in (HERE/'README.md').read_text():
        with (HERE/'README.md').open('a') as out:out.write('\n'+marker+'\n\n'+'\n\n'.join(lines)+'\n\nAll policy details: POLICIES.md. Full interpretation and limitations: RESULTS.md.\n')
