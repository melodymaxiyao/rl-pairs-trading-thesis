"""Five-world paired summaries; Tier 2 always explicitly not run."""
from sc_common import *
from sc_kernel import policy_q
from sc_rules import DEFAULT
from schedule import DEFERRED_D_LABEL
import numpy as np,pandas as pd
from scipy.stats import t

def interval(values):
    x=np.asarray(values,float);x=x[np.isfinite(x)]
    mean=float(x.mean()) if len(x) else np.nan
    half=float(t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x))) if len(x)>1 else np.nan
    return dict(mean=mean,ci_low=mean-half,ci_high=mean+half,n_worlds=len(x))

def tier_results(tier):
    from schedule import world_jobs
    from sc_worlds import job_key
    jobs=world_jobs(tier,tier1_verified=(tier==3))
    out=[cached('runtime/worlds/'+job_key(j)) for j in jobs];assert all(x is not None for x in out)
    return out

def seal_tier(tier):
    folder=HERE/f'outputs/tier{tier}';folder.mkdir(exist_ok=True)
    path=HERE/f'audit/tier{tier}_seal.json'
    if path.exists():verify_tier(tier);return
    worlds=tier_results(tier);rows=[];policies={};oracles=[]
    for v in worlds:
        for r in v['rows']:
            row=dict(r);row['selected']=json.dumps(clean(row.get('selected')),sort_keys=True);row['risk']=json.dumps(row['risk'],sort_keys=True);rows.append(row)
        key=json.dumps(v['job'],sort_keys=True);policies[key]={n:dict(q=m['q'],selected=m['selected']) for n,m in v['models'].items()}
        for j,a in enumerate(v['oracle']['selection_metrics']):oracles.append(dict(v['job'],option=j,kind='TD8' if j<256 else 'B3',policy=j if j<256 else j-256,oracle_path_net=a[0],gross=a[1],turnover=a[2],selected=j==v['oracle']['policy_bits'] or j==256+v['oracle']['B3_index']))
    csv_new(folder/'cells.csv',rows);csv_new(folder/'oracle_policies.csv',oracles);write(folder/'policies.json',policies)
    files={str(p.relative_to(HERE)):sha(p) for p in folder.iterdir() if p.is_file()}
    for name in ['preregistration.json','outputs/tables/calibration.json','audit/source_registration.json','audit/code_registration.json']:files[name]=sha(HERE/name)
    from sc_worlds import job_key
    for v in worlds:
        f=HERE/('runtime/worlds/'+job_key(v['job'])+'.pkl.gz');files[str(f.relative_to(HERE))]=sha(f)
    write(path,dict(status='sealed',tier=tier,worlds=len(worlds),files=files,Tier2='deferred, not run'));verify_tier(tier)

def verify_tier(tier):
    from sc_protection import registered
    registered()
    seal=read(HERE/f'audit/tier{tier}_seal.json');assert seal['status']=='sealed' and seal['tier']==tier
    for k,v in seal['files'].items():assert sha(HERE/k)==v,('Tier artifact changed',tier,k)
    return seal

def descriptive_real():
    # Copied hyperparameter choices only. No real result or raw price decoding.
    path=SOURCE/'outputs/tables/limits_and_choices.csv';frame=pd.read_csv(path);rows=[]
    for r in frame.to_dict('records'):
        source=str(r.get('source',''))
        if r.get('kind')!='TD_hyperparameters' or not any(source.endswith('_'+x) for x in ['TD8','TD16_align','TD16_strength']):continue
        v=json.loads(r['selection']);rows.append(dict(source=source,pool=source.split('_')[0]+'_'+source.split('_')[1],cell=source.split('_',3)[-1],alpha=v['alpha'],gamma=v['gamma']))
    assert rows,'No sealed real TD8/TD16 hyperparameter choices'
    csv_new(HERE/'outputs/tables/real_hyperparameter_choices.csv',rows)
    summary=pd.DataFrame(rows).groupby(['pool','cell','alpha','gamma']).size().reset_index(name='folds');csv_new(HERE/'outputs/tables/real_hyperparameter_frequencies.csv',summary)

def report():
    verify_tier(1);verify_tier(3);folder=HERE/'outputs/tables'
    f=pd.concat([pd.read_csv(HERE/f'outputs/tier{tier}/cells.csv') for tier in [1,3]],ignore_index=True)
    csv_new(folder/'cells.csv',f)
    csv_new(folder/'oracle_policies.csv',pd.concat([pd.read_csv(HERE/f'outputs/tier{tier}/oracle_policies.csv') for tier in [1,3]],ignore_index=True))
    write(folder/'policies.json',{str(tier):read(HERE/f'outputs/tier{tier}/policies.json') for tier in [1,3]})
    summaries=[]
    for key,g in f.groupby(['tier','case','k','D','learner'],sort=True):
        row=dict(zip(['tier','case','k','D','learner'],key));row.update(interval(g.net));row['default_retention_share']=float(g.keeps_default.mean()) if row['learner'] in ['B3','B3_shrunk','B3_lcb','L0'] else np.nan
        for col in ['regret','agreement','weighted_agreement','abstention','turnover','gross','oracle_B3_headroom']:
            if col in g and g[col].notna().any():row.update({col+'_'+k:v for k,v in interval(g[col]).items()})
        summaries.append(row)
    summary=pd.DataFrame(summaries);csv_new(folder/'summary.csv',summary)
    edges=[]
    zero=float(f[(f.tier==1)&(f.k==0)&(f.learner=='L0')].net.mean())
    calibration=read(folder/'calibration.json')['chosen']
    for k,g in f[(f.tier==1)&(f.learner=='L0')].groupby('k'):
        edges.append(dict(k=float(k),h=calibration['h'],w=min(float(k)*calibration['w'],.95),L0_net=float(g.net.mean()),L0_k0_net=zero,realised_edge=float(g.net.mean()-zero)))
    csv_new(folder/'realised_edge.csv',edges)
    rules=summary[summary.learner.isin(['L0','B3','B3_shrunk','B3_lcb','oracleB3'])];csv_new(folder/'rule_comparison.csv',rules)
    base=f[(f.tier==1)&(f.k==1)]
    def values(data,name):return data[data.learner==name].set_index('world').net.sort_index()
    def contrast(data,a,b):
        x=values(data,a);y=values(data,b);assert x.index.equals(y.index);return interval(x-y)
    predictions=[]
    def pred(id,label,result=None,**extra):predictions.append(dict(id=id,label=label,**(result or {}),**extra))
    zero=f[(f.tier==1)&(f.k==0)]
    oracles=[w['oracle']['policy_bits'] for w in tier_results(1) if w['job']['k']==0]
    sanity=all((bits&0x55)==0 for bits in oracles) and all(interval(g.net)['ci_low']<=0 for _,g in zero.groupby('learner'))
    pred('S0','consistent' if sanity else 'not reproduced',oracle_never_enter_worlds=sum((b&0x55)==0 for b in oracles))
    means={n:float(values(base,n).mean()) for n in ['L0','B3','TD_v3','TD_tuned']};abst=float(base[base.learner=='TD_v3'].abstention.mean())
    pred('S1','reproduced' if means['TD_v3']<means['B3']<means['L0'] and abst>.5 else 'not reproduced',TD_v3_abstention=abst,**means)
    c=contrast(base,'TD_tuned','TD_v3');pred('S2','positive' if c['ci_low']>0 else 'not established',c)
    c=contrast(base,'TD_tuned','L0');pred('S3','matches by point estimate' if c['mean']>=0 else 'does not match by point estimate',c)
    candidates=[]
    for k,g in f[f.tier==1].groupby('k'):
        if contrast(g,'TD_tuned','L0')['mean']>=0:candidates.append(float(k))
    pred('S4','k part only',smallest_k=min(candidates) if candidates else None,D_part=DEFERRED_D_LABEL,note='k=0 may qualify by abstention; this does not show signal recovery')
    p=f[(f.tier==3)&(f.case=='protocol_ablation')]
    x=p[p.learner=='TD_v3_gamma90'].set_index('world').exit_everywhere;y=p[p.learner=='TD_v3_gamma99'].set_index('world').exit_everywhere
    c=interval(x.sort_index()-y.sort_index());pred('S5','positive' if c['ci_low']>0 else 'not established',c)
    h=f[(f.tier==3)&(f.case=='heterogeneous')];true=contrast(h,'TD_true_grouped','TD_tuned');proxy=contrast(h,'TD_proxy_grouped','TD_tuned')
    z=(values(h,'TD_proxy_grouped')-values(h,'TD_tuned'))-.5*(values(h,'TD_true_grouped')-values(h,'TD_tuned'));half=interval(z)
    pred('S6','supported' if true['ci_low']>0 and half['ci_high']<0 else 'not established',true,proxy_gain=proxy['mean'],proxy_minus_half_true=half['mean'],proxy_minus_half_true_ci_high=half['ci_high'])
    csv_new(folder/'predictions.csv',predictions);descriptive_real();plots(f,summary)
    choices=pd.read_csv(folder/'real_hyperparameter_choices.csv')
    gamma_counts=choices.groupby(['cell','gamma']).size().to_dict()
    cal=read(folder/'calibration.json')['chosen'];edge=contrast_k(f,'L0',1,0)
    text=['# Synthetic sample-complexity results','',f"Calibration h*={cal['h']}, w*={cal['w']}; objective={cal.get('objective')}; fitted excess event net={cal.get('excess_mean_net')}, KM median={cal.get('KM_median')}. L0 edge at k=1: {edge:.3f} bps/pair-year.",'', 'Tier 2: deferred, not run. S4 D: '+DEFERRED_D_LABEL+'.','', 'All evidence is synthetic. The oracle is selected on separate Monte Carlo paths and evaluated on the same fresh test paths as the learners. Negative regret is retained. No real trading trial has been run.','', '## Frozen predictions','']
    text+=['- '+r['id']+': '+r['label'] for r in predictions]
    text+=['','## What the comparisons identify','','TD_tuned versus TD_v3 changes alpha scheduling, gamma and selection together. The fixed-gamma and ensemble cells isolate some components; gamma=1 versus .99 is within tuned alpha. There is no full factorial experiment isolating every interaction. The enumerated oracle is optimal only within its finite class on its selection sample, and is not a convergence guarantee for a partially observed TD8 process.','', 'Grouped tables are trained TD tables with either true-type or train-event proxy labels; true-type grouping is not an oracle policy search. Proxy thresholds are frozen at the training-pair median; undefined proxy scores go to the low group and are counted.','', 'The process is stationary partial cointegration; no legs/common factors, volatility clustering or real-data external validation. The t(4) variant only changes innovation tails. Calibration matches only two target statistics.']
    path=HERE/'RESULTS.md'
    if not path.exists():
        with path.open('x') as stream:stream.write('\n'.join(text))
    readme=HERE/'README.md'
    if '<!-- GENERATED RESULTS -->' not in readme.read_text():
        with readme.open('a') as stream:stream.write('\n\n<!-- GENERATED RESULTS -->\n'+ '\n'.join(text)+'\n')
    lines=[f"Calibration h*={cal['h']}, w*={cal['w']}; realised L0 edge {edge:.2f} bps.",f"S1/S2: TD_v3 {means['TD_v3']:.2f}, B3 {means['B3']:.2f}, L0 {means['L0']:.2f}, TD_tuned {means['TD_tuned']:.2f} bps.",f"S3/S4: {predictions[3]['label']}; smallest k {predictions[4]['smallest_k']}; D {DEFERRED_D_LABEL}.",f"S5: {predictions[5]['label']}; real gamma counts {gamma_counts}.",f"S6: {predictions[6]['label']}; grouping evidence synthetic only."]
    write(HERE/'summary.json',dict(lines=lines));return f

def contrast_k(frame,learner,a,b):
    f=frame[(frame.tier==1)&(frame.learner==learner)]
    return float(f[f.k==a].net.mean()-f[f.k==b].net.mean())

def plots(frame,summary):
    import matplotlib.pyplot as plt
    folder=HERE/'outputs/plots'
    def save(fig,name):
        p=folder/name
        if not p.exists():fig.savefig(p,dpi=140,bbox_inches='tight')
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4))
    for name in ['TD_v3','TD_tuned']:
        vals=[]
        for k,g in frame[frame.tier==1].groupby('k'):vals.append((k,g[g.learner==name].net.mean()-g[g.learner=='L0'].net.mean()))
        ax.plot(*zip(*vals),marker='o',label=name)
    ax.axhline(0,color='grey');ax.set(xlabel='k (D=1 only)',ylabel='TD minus L0 (bps)',title='Phase slice: Tier 2 deferred');ax.legend();save(fig,'phase_diagram.png')
    fig,ax=plt.subplots(figsize=(9,5))
    for name,g in summary[summary.tier==1].groupby('learner'):
        ax.plot(g.k,g['mean'],marker='.',label=name)
    ax.set(xlabel='k',ylabel='Net bps/pair-year');ax.legend(ncol=2);save(fig,'net_vs_k.png')
    fig,ax=plt.subplots(figsize=(9,4));ax.axis('off');ax.text(.5,.5,DEFERRED_D_LABEL,ha='center',va='center',fontsize=17);save(fig,'net_vs_D.png')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for name in ['TD_v3','TD_tuned']:
        g=summary[(summary.tier==1)&(summary.learner==name)]
        axes[0].plot(g.k,g.agreement_mean,label=name);axes[1].plot(g.k,g.abstention_mean,label=name)
    for ax,title in zip(axes,['Oracle agreement','Abstention']):ax.set(xlabel='k',title=title);ax.legend()
    save(fig,'agreement_abstention.png')
    world=next(w for w in tier_results(1) if w['job']['k']==1 and w['job']['world']==0)
    matrix=np.asarray([(policy_q(world['oracle']['policy_bits'])[:,1]>policy_q(world['oracle']['policy_bits'])[:,0]).astype(int)]+[(world['models'][n]['q'][:,1]>world['models'][n]['q'][:,0]).astype(int) for n in ['TD_v3','TD_tuned']])
    fig,ax=plt.subplots(figsize=(9,3));ax.imshow(matrix,vmin=0,vmax=1,aspect='auto');ax.set_yticks(range(3),['oracle256','TD_v3','TD_tuned']);ax.set_xlabel('TD8 state, world 0, k=1');save(fig,'oracle_policy.png')
