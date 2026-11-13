"""Stream full events once, retain compact numeric cells, assemble tables/plots."""
from aio import *
from engine import ANCHORS,INFEASIBLE,ADAPTIVE,HORIZONS
from analysis import Analysis,COLUMNS,difference
import numpy as np
import pandas as pd
import gzip,csv

EVENT_COLUMNS=['pair','leg_A','leg_B','year','era','relation','weight','original_ipw','dataset','anchor','feasibility','regime',
    'fixed_beta','S_entry','S_exit','m_entry','m_exit','s_entry','s_exit',
    'entry_index','exit_index','entry_date','exit_date','duration','direction','d_entry','d_exit','reverted','censored',
    'gross_bps','net_bps','hit','drift_alignment','prior_drift_mean_bps','year_crossing',
    *[name for h in HORIZONS for name in [f'forward_{h}',f'forward_{h}_excluded',f'forward_{h}_exclusion_reason']],
    'price_share','anchor_and_scale_share','driver_numerator','driver_denominator','deviation_change',
    'price_deviation_part','anchor_deviation_part','scale_deviation_part','price_component_share','anchor_component_share','scale_component_share','driver_outside_unit_interval']

def atomic(path,data):
    path=inside(path);tmp=path.with_name(path.name+'.pending_'+str(time.time_ns()));data(tmp)
    if path.exists():assert sha(path)==sha(tmp),('Existing output differs',path);tmp.unlink()
    else:os.rename(tmp,path)

def table(path,frame):atomic(path,lambda p:frame.to_csv(p,index=False,na_rep='NA',float_format='%.12g'))

def compact(e):
    return [e['year'],{'cross_sector':0,'same_sector':1,'same_subindustry':2}[e['relation']],e['drift_alignment'],e['weight'],
        e['duration'],float(e['reverted']),e['gross_bps'],*[e[f'forward_{h}'] for h in HORIZONS],
        *[e.get(k,np.nan) for k in ['driver_numerator','driver_denominator','price_deviation_part','anchor_deviation_part','scale_deviation_part','deviation_change']]]

def gather(years=YEARS):
    bags={};pairs=[];coverage=[];nulls=[]
    path=TABLES/'events.csv.gz';pending=path.with_name(path.name+'.pending_'+str(time.time_ns()))
    with pending.open('xb') as raw:
        with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as zipped:
            import io
            with io.TextIOWrapper(zipped,encoding='utf-8',newline='') as text:
                writer=csv.DictWriter(text,fieldnames=EVENT_COLUMNS);writer.writeheader()
                for year in years:
                    assert checkpoint(HERE/f'runtime/checkpoints/fold_{year}.json')
                    for source in sorted((HERE/f'runtime/checkpoints/{year}').glob('chunk_*.pkl.gz')):
                        assert checkpoint(source.with_suffix('.ready.json'));chunk=load_gzip(source)
                        pairs.extend(chunk['pairs']);coverage.extend(chunk['coverage']);nulls.extend(chunk['null_parameters']);groups={}
                        for event in chunk['events']:
                            writer.writerow({k:event.get(k,'NA') for k in EVENT_COLUMNS})
                            key=tuple(event[k] for k in ['anchor','regime','dataset']);groups.setdefault(key,[]).append(compact(event))
                        for key,values in groups.items():bags.setdefault(key,[]).append(np.asarray(values,float))
    if path.exists():assert sha(path)==sha(pending);pending.unlink()
    else:os.rename(pending,path)
    return bags,pd.DataFrame(pairs),pd.DataFrame(coverage),pd.DataFrame(nulls)

def assemble(bags,pairs,years=YEARS,draws=2000):
    a=Analysis(pairs,years,draws)
    for anchor in ANCHORS:
        for regime in ['within-year','episode-free']:
            if anchor=='F_orig' and regime=='episode-free':continue
            for dataset in ['real','N1','N2']:
                pieces=bags.pop((anchor,regime,dataset),[])
                array=np.concatenate(pieces) if pieces else np.empty((0,len(COLUMNS)))
                a.cell(anchor,regime,dataset,array)
    return a,dict(anchor_summary=pd.DataFrame(a.rows),real_minus_null=pd.DataFrame(a.contrasts()),
                  predictions=pd.DataFrame(a.predictions()),exclusions=pd.DataFrame(a.exclusions),km_curves=pd.DataFrame(a.curves,columns=['anchor','feasibility','regime','dataset','duration','survival','weighted_at_risk']))

def plots(tables,outdir=PLOTS):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    outdir.mkdir(parents=True,exist_ok=True);files=[]
    summary=tables['anchor_summary'];diff=tables['real_minus_null']
    def label(anchor):return anchor+'\nINFEASIBLE' if anchor in INFEASIBLE else anchor
    def finish(fig,name):
        fig.tight_layout();path=outdir/(name+'.png')
        atomic(path,lambda p:fig.savefig(p,format='png',dpi=160,metadata={'Software':'anchor_horizon_diagnostic_v1'}));plt.close(fig);files.append(path)
    def plot_points(ax,frame,xcol,offset=0,label=None,style='o-'):
        if frame.empty:return
        frame=frame.sort_values(xcol);x=frame[xcol].to_numpy(float)+offset;y=frame.estimate.to_numpy(float)
        low=frame.ci95_low.to_numpy(float);hi=frame.ci95_high.to_numpy(float);ok=np.isfinite(x)&np.isfinite(y)
        ax.plot(x[ok],y[ok],style,label=label);good=ok&np.isfinite(low)&np.isfinite(hi)
        ax.vlines(x[good],low[good],hi[good],alpha=.5)
    fig,axes=plt.subplots(1,2,figsize=(15,5))
    for ax,regime in zip(axes,['within-year','episode-free']):
        for dataset,offset,style in [('real',-.13,'o-'),('N1',0,'s--'),('N2',.13,'^:')]:
            f=summary[(summary.metric=='mean_net')&(summary.stratum=='pooled')&(summary.regime==regime)&(summary.dataset==dataset)].copy();f['x']=f.anchor.map(dict(zip(ANCHORS,range(len(ANCHORS)))))
            plot_points(ax,f,'x',offset,dataset,style)
        names=ANCHORS if regime=='within-year' else [a for a in ANCHORS if a!='F_orig']
        ax.axhline(0,color='grey');ax.set(title=regime,ylabel='Mean event net, bps',xticks=range(len(names)),xticklabels=[label(x) for x in names]);ax.tick_params(axis='x',labelsize=8);ax.legend()
    finish(fig,'anchor_net')
    fig,axes=plt.subplots(2,4,figsize=(16,8))
    for ax,anchor in zip(axes.flat,ANCHORS):
        for regime,style in [('within-year','o-'),('episode-free','s--')]:
            f=diff[(diff.anchor==anchor)&(diff.regime==regime)&(diff.dataset=='real-N2')&(diff.stratum=='pooled')&diff.metric.str.startswith('mean_forward_')].copy()
            f['h']=f.metric.str.rsplit('_',n=1).str[-1].astype(int);plot_points(ax,f,'h',label=regime,style=style)
        ax.axhline(0,color='grey');ax.set(title=label(anchor),xlabel='Sessions h',ylabel='Real - N2, bps',xticks=HORIZONS);ax.legend(fontsize=7)
    axes.flat[-1].axis('off');finish(fig,'forward_moves')
    fig,axes=plt.subplots(1,2,figsize=(12,4));curves=tables['km_curves']
    for ax,regime in zip(axes,['within-year','episode-free']):
        for dataset,style in [('real','-'),('N1','--'),('N2',':')]:
            f=curves[(curves.regime==regime)&(curves.dataset==dataset)]
            ax.step(f.duration,f.survival,where='post',label=dataset,ls=style)
        ax.set(title='A_F: '+regime,xlabel='Sessions to reversion',ylabel='IPW survival',ylim=(0,1));ax.legend()
    finish(fig,'KM_A_F')
    fig,axes=plt.subplots(1,2,figsize=(15,5))
    for ax,regime in zip(axes,['within-year','episode-free']):
        for stratum,offset,style in [('drift_aligned',-.08,'o-'),('drift_counter',.08,'s--')]:
            f=diff[(diff.metric=='mean_net')&(diff.regime==regime)&(diff.dataset=='real-N2')&(diff.stratum==stratum)].copy();f['x']=f.anchor.map(dict(zip(ANCHORS,range(len(ANCHORS)))))
            plot_points(ax,f,'x',offset,stratum,style)
        names=ANCHORS if regime=='within-year' else [a for a in ANCHORS if a!='F_orig']
        ax.axhline(0,color='grey');ax.set(title=regime,ylabel='Real - N2 mean net, bps',xticks=range(len(names)),xticklabels=[label(x) for x in names]);ax.tick_params(axis='x',labelsize=8);ax.legend()
    finish(fig,'drift_alignment')
    fig,axes=plt.subplots(1,2,figsize=(12,4));names=sorted(ADAPTIVE)
    for ax,regime in zip(axes,['within-year','episode-free']):
        for dataset,offset,style in [('real',-.08,'o-'),('N2',.08,'s--')]:
            f=summary[(summary.metric=='price_share_ratio')&(summary.regime==regime)&(summary.stratum=='pooled')&(summary.dataset==dataset)].copy();f['x']=f.anchor.map(dict(zip(names,range(3))))
            plot_points(ax,f,'x',offset,dataset,style)
        ax.axhline(0,color='grey');ax.axhline(1,color='grey',ls=':');ax.set(title=regime,ylabel='Price share: IPW ratio of sums',xticks=range(3),xticklabels=names);ax.legend()
    finish(fig,'reversion_drivers');return files

def value(x):return 'NA' if pd.isna(x) else 'infinite' if np.isposinf(x) else '-infinite' if np.isneginf(x) else f'{x:.3g}'

def narrative(tables):
    p=tables['predictions'];primary=p[p.family=='primary'].set_index('prediction');q1=p[(p.prediction=='Q1')&(p.regime=='within-year')]
    best=primary.loc[['Q2-F','Q2-R252','Q2-R63','Q2-TR'],'estimate'].idxmax() if primary.loc[['Q2-F','Q2-R252','Q2-R63','Q2-TR'],'estimate'].notna().any() else 'Q2-F'
    supported=primary.loc[['Q2-F','Q2-R252','Q2-R63','Q2-TR']];supported=supported[supported.label=='supported']
    q4=p[(p.prediction=='Q4')&(p.anchor=='F_orig')];q6=primary.loc['Q6'];q3=primary.loc['Q3'];b=primary.loc[best]
    anchor_names={'Q2-F':'A_F','Q2-R252':'A_R252','Q2-R63':'A_R63','Q2-TR':'A_TR'};s=tables['anchor_summary']
    def real_net(anchor):
        v=s[(s.anchor==anchor)&(s.regime=='within-year')&(s.dataset=='real')&(s.stratum=='pooled')&(s.metric=='mean_net')]
        return float(v.estimate.iloc[0]) if len(v) else np.nan
    viable=[name for name in supported.index if real_net(anchor_names[name])>0]
    next_design='A causal reversion learner merits a separately preregistered test' if viable else 'No causal reversion learner with positive absolute event net is established by this diagnostic'
    if viable and q6.label=='supported':next_design+='; drift alignment is a candidate state'
    lines=[f"Q1 INFEASIBLE real-N2 mean net: "+'; '.join(f'{r.anchor} {value(r.estimate)} bps' for r in q1.itertuples()),
        f"Q2 descriptive best {best}: real-N2 {value(b.estimate)} bps/event; absolute real net {value(real_net(anchor_names[best]))} bps; Holm p {value(b.p_Holm)}, {b.label}.",
        f"Q3 year-end censoring effect: {value(q3.estimate)} bps/event, {q3.label}.",
        'Q4 F_orig: '+('/'.join(q4.label.unique()) if len(q4) else 'NA')+f"; Q6 drift-aligned advantage {value(q6.estimate)} bps/event, {q6.label}.",
        next_design+'.']
    text='\n## Results after the complete run\n\n'
    text+='1. Hindsight anchors: '+lines[0]+'. These anchors are INFEASIBLE; positive hindsight net does not establish executable reversion.\n'
    text+='2. Causal anchors: '+lines[1]+' All four prespecified causal tests retain six-family Holm correction. Choosing the largest estimate is descriptive. Absolute real net and 95% intervals are in anchor_summary.csv.\n'
    text+='3. Annual close: '+lines[2]+' This compares different event sets under the same governing entry rule, not a strategy modification.\n'
    text+='4. Volatility-path comparison: '+lines[3]+' F_orig is within-year only for exact continuity; A_F is the fixed-beta companion. N2 matches absolute returns, including volatility regimes, fat tails and jumps. A change from N1 to N2 cannot identify clustering separately from tails or jumps. Real-minus-N2 comparisons also remove any true drift, so disappearance is compatible with a volatility-path explanation rather than causal proof of clustering alone.\n'
    text+='5. Drift alignment: Q6 uses 252 sessions strictly before entry. Missing warmup is separately counted; aligned and counter-drift absolute net and real-minus-null contrasts are reported by anchor.\n'
    text+='6. Next Q-learning round: '+next_design+'. Only causal-anchor evidence can motivate an executable state; A_TR is the prespecified trend-adjusted option. If no causal hypothesis survives, this task does not establish a reversion-based learner. The earlier H1s result can motivate a separate drift-timing experiment, which was not run here. No learner or strategy was evaluated.\n'
    text+='\nDriver primary ratios use the same direction on numerator and denominator. Exact price/anchor-mean/scale pieces are retained, with medians, quartiles, unbounded shares and censored exclusions. F_orig episode-free is intentionally excluded.\n'
    text+='\n### Five-line summary\n\n'+'\n'.join(lines)+'\n';return text,lines

def report_all():
    ready=HERE/'audit/report_ready.json'
    if checkpoint(ready):return read(ready)['summary']
    bags,pairs,coverage,nulls=gather();assert len(pairs[pairs.dataset=='real'])==66000
    a,tables=assemble(bags,pairs)
    for name,frame in tables.items():table(TABLES/(name+'.csv'),frame)
    table(TABLES/'anchor_coverage.csv',coverage);table(TABLES/'null_parameters.csv',nulls)
    files=plots(tables);body,summary=narrative(tables);path=HERE/'results.md'
    if path.exists():assert path.read_text()==body
    else:path.write_text(body)
    top=read(HERE/'audit/code_registration.json')['README_top_bytes'];readme=HERE/'README.md'
    if readme.stat().st_size==top:
        with readme.open('a') as f:f.write(body)
    else:assert readme.read_bytes()[top:]==body.encode()
    files+=list(TABLES.glob('*'))+[path,readme]
    receipt(ready,files,summary=summary,real_pairs=66000,events=int(sum(r['n_events'] for r in a.exclusions if r['stratum']=='pooled' and r['h']==5)))
    return summary
