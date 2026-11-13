"""Stream checkpoint tables, assemble IPW inference and render five figures."""
from mio import *
from jobs import memberships,RELATIONS
from inference import Analysis
import numpy as np
import pandas as pd
import csv

def csv_atomic(path,frame):
    path=inside(path);tmp=path.with_name(path.name+'.pending_'+str(time.time_ns()))
    frame.to_csv(tmp,index=False,na_rep='NA',float_format='%.12g')
    if path.exists():assert sha(tmp)==sha(path),('Existing report differs',path);tmp.unlink()
    else:os.rename(tmp,path)

def merge_bank(target,bank):
    for key,value in bank.items():
        if key not in target:target[key]={k:v.copy() for k,v in value.items()}
        else:
            for k,v in value.items():target[key][k]+=v

def gather(years=YEARS,root=None):
    root=HERE/'runtime/checkpoints' if root is None else root
    parts={k:[] for k in ['basic','partial','factor','null']};banks={'trigger_bank':{},'factor_bank':{}}
    gross=[];eventpath=TABLES/'event_triggers.csv';pending=eventpath.with_name(eventpath.name+'.pending_'+str(time.time_ns()))
    columns=['pair','leg_A','leg_B','year','era','relation','in_random','ipw','partial_weight','FAST_cap','FAST_X_cap','FAST_W_cap','dataset',
             'entry_date','end_date','duration','reverted','gross_bps','governing_F_gross_bps','trigger','larger_raw_leg',
             'larger_spread_contribution','volume_window_available','jump_window_available','entry_return_bps','trailing_return_sd_bps']
    with pending.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for year in years:
            marker=root/f'fold_{year}.json';assert checkpoint(marker)
            for name in sorted(root.joinpath(str(year)).glob('chunk_*.pkl.gz')):
                assert checkpoint(name.with_suffix('.ready.json'));chunk=load_gzip(name)
                for key in parts:parts[key].append(pd.DataFrame(chunk[key]))
                for key in banks:merge_bank(banks[key],chunk[key])
                for e in chunk['events']:
                    writer.writerow({k:e.get(k) for k in columns})
                    if e['reverted']:
                        for view,w in memberships(e):gross.append(dict(view=view,weight=w,year=e['year'],relation=e['relation'],dataset=e['dataset'],trigger=e['trigger'],gross_bps=e['gross_bps']))
    if eventpath.exists():assert sha(pending)==sha(eventpath);pending.unlink()
    else:os.rename(pending,eventpath)
    frames={k:pd.concat(v,ignore_index=True) if v else pd.DataFrame() for k,v in parts.items()}
    return frames,banks,pd.DataFrame(gross,columns=['view','weight','year','relation','dataset','trigger','gross_bps'])

def plots(tables,partial,outdir=PLOTS):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    outdir.mkdir(parents=True,exist_ok=True);paths=[]
    def finish(fig,name):
        fig.tight_layout();path=outdir/(name+'.png');tmp=path.with_name(name+'.pending_'+str(time.time_ns())+'.png')
        fig.savefig(tmp,dpi=160,metadata={'Software':'spread_mechanism_diagnostic_v1'});plt.close(fig)
        if path.exists():assert sha(tmp)==sha(path);tmp.unlink()
        else:os.rename(tmp,path)
        paths.append(path)
    def base(name):
        f=pd.DataFrame(tables[name]);return f[(f.view=='sample')&(f.era=='all')]
    def points(ax,f,x,offset=0):
        f=f.sort_values(x);xx=pd.to_numeric(f[x],errors='coerce').to_numpy(float)+offset
        y=pd.to_numeric(f.estimate,errors='coerce').to_numpy(float)
        lo=pd.to_numeric(f.ci95_low,errors='coerce').to_numpy(float);hi=pd.to_numeric(f.ci95_high,errors='coerce').to_numpy(float)
        label=str(f.dataset.iloc[0]) if len(f) else None
        ok=np.isfinite(xx)&np.isfinite(y);ax.plot(xx[ok],y[ok],'s--' if label=='null' else 'o-',label=label)
        good=ok&np.isfinite(lo)&np.isfinite(hi);ax.vlines(xx[good],lo[good],hi[good],alpha=.5)
    f=base('vr_term_structure');f=f[f.metric=='median_VR'];fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,rel in zip(axes,RELATIONS):
        for d in ['real','null']:points(ax,f[(f.relation==rel)&(f.dataset==d)],'q')
        ax.axhline(1,color='grey',lw=1);ax.set_xscale('log');ax.set(title=rel,xlabel='Sessions q',ylabel='IPW median VR',xticks=[2,5,10,20,60,120,250],xticklabels=['2','5','10','20','60','120','250']);ax.legend()
    finish(fig,'vr_term_structure')
    fig,axes=plt.subplots(1,3,figsize=(13,4));p=partial[(partial.window=='full')&partial.in_random&partial.converged].copy()
    for ax,col,title in zip(axes,['R2_MR','half_life','swing_bps'],['Transient return share','Positive-rho half-life (sessions)','Transient swing (bps)']):
        for d in ['real','null']:
            s=p[p.dataset==d];v=pd.to_numeric(s[col],errors='coerce').to_numpy(float);w=pd.to_numeric(s.partial_weight,errors='coerce').to_numpy(float)
            ok=np.isfinite(v)&(w>0)
            if ok.any():
                finite=v[ok];positive=finite[finite>0];bins=40
                if col!='R2_MR' and len(positive) and positive.max()>100*positive.min():
                    bins=np.r_[0.,np.geomspace(positive.min(),positive.max()*(1+1e-9),40)]
                    ax.set_xscale('symlog',linthresh=1.)
                ax.hist(finite,bins=bins,weights=w[ok]/w[ok].sum(),histtype='step',linestyle='--' if d=='null' else '-',label=d)
        ax.set(title=title,ylabel='IPW probability mass');ax.legend()
    axes[1].text(.03,.96,'rho <= 0: <1 session; share in table',transform=axes[1].transAxes,va='top',fontsize=8)
    axes[2].axvline(20,color='black',ls='--');finish(fig,'partial_ci_distributions')
    f=base('drift_persistence_horizons');f=f[(f.metric=='sign_agreement_minus_half')&(f.relation=='all')&(f.bucket!='all')]
    fig,axes=plt.subplots(1,4,figsize=(15,4))
    for ax,predictor in zip(axes,['lag2','lag1','last6months','A']):
        for d in ['real','null']:points(ax,f[(f.predictor==predictor)&(f.dataset==d)],'bucket')
        ax.axhline(0,color='grey');ax.set(title=predictor,xticks=range(5),xticklabels=['<.5','.5–1','1–1.5','1.5–2','>=2'],ylabel='Sign agreement - .5');ax.legend()
    finish(fig,'drift_strength')
    f=base('event_trigger_summary');f=f[(f.metric=='KM63')&(f.dataset=='real')];fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,rel in zip(axes,RELATIONS):
        s=f[f.relation==rel].copy();s['x']=s.trigger.map({'volume-spike':0,'jump':1,'gradual':2});points(ax,s,'x')
        ax.set(title=rel,xticks=range(3),xticklabels=['volume','jump','gradual'],ylabel='IPW P(revert <=63), KM',ylim=(0,1))
    finish(fig,'event_triggers')
    f=base('factor_residual');f=f[(f.version=='factor_only')&f.relation.isin(RELATIONS)];fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,metric in zip(axes,['residual_minus_raw_VR_120','residual_minus_raw_reverted_share']):
        for d,off in [('real',-.08),('null',.08)]:
            s=f[(f.metric==metric)&(f.dataset==d)].copy();s['x']=s.relation.map(dict(zip(RELATIONS,range(3))));points(ax,s,'x',off)
        ax.axhline(0,color='grey');ax.set(title='VR(120): residual - raw' if metric.endswith('VR_120') else 'Reverted share: residual - raw',xticks=range(3),xticklabels=['cross-sector','same-sector','same-subindustry']);ax.legend()
    finish(fig,'factor_residual')
    return paths

def readable(value):
    if value is None or isinstance(value,float) and np.isnan(value):return 'NA'
    if np.isneginf(value):return '<1 session'
    if np.isposinf(value):return 'infinite'
    return f'{value:.3g}'

def narrative(analysis,frames):
    p={r['prediction']:r for r in analysis.tables['predictions']};tables=analysis.tables
    def stat(table,metric,**sel):
        matches=[r for r in tables[table] if r['metric']==metric and all(r.get(k)==v for k,v in sel.items())]
        return matches[0]['estimate'] if matches else np.nan
    sel=dict(view='sample',relation='all',era='all',window='full',boundary_policy='all_converged',dataset='real')
    r2=stat('partial_ci_summary','R2_MR',**sel);half=stat('partial_ci_summary','half_life_order',**sel);swing=stat('partial_ci_summary','swing_ge_20',**sel)
    lines=[f"VR real-null: q10 {readable(p['P1']['estimate'])}; q120 {readable(p['P2']['estimate'])}, P2 {p['P2']['label']}.",
           f"Transient: R2 {readable(r2)}, median half-life {readable(half)}, swing>=20 bps share {readable(swing)}; P3 {p['P3']['label']}, P4 {p['P4']['label']}.",
           f"Triggers: volume-minus-gradual KM63 {readable(p['P6']['estimate'])}; P6 {p['P6']['label']}.",
           f"Factors: event-IPW reverted-share DID {readable(p['P7']['estimate'])}; P7 {p['P7']['label']}.",
           f"Drift: lag1-minus-lag2 agreement {readable(p['P5']['estimate'])}; strength/horizon P5 {p['P5']['label']}."]
    text='\n## Results (generated after the complete run)\n\n'
    text+=f"1. Transient size, speed and costs: {lines[1]} All-converged and boundary-excluded estimates, positive-rho coverage and oscillatory fits are in partial_ci_summary.csv.\n"
    text+=f"2. Long-horizon trend: {lines[0]} Cross-sector excess relative to W is {readable(p['P8']['estimate'])} (P8 {p['P8']['label']}).\n"
    text+=f"3. News-proxy divergences: {lines[2]} This is an association with volume, not an identified news effect.\n"
    text+=f"4. Removing factors: {lines[3]} This compares residualisation effects on real and null paths. Paired-share, KM63, alpha-removal, narrower-control and no-2004 sensitivities are reported.\n"
    text+=f"5. Drift persistence: {lines[4]} The horizon/strength/era table reports all five buckets and their uncertainty.\n"
    text+='6. Mechanism interpretation: '+('The primary results jointly support the prespecified directional associations.' if all(p[k]['label']=='supported' for k in ['P2','P3','P6','P7']) else 'The results do not jointly establish the full permanent-news, transient-component and momentum explanation.')
    text+=' A transient component with insufficient amplitude would limit a reversion learner after costs; persistent drift could motivate a drift learner. These diagnostics establish neither tradable out-of-sample edge nor causation, and they run no learner or trading evaluation.\n'
    f=frames['factor'];gap=f[(f.year==2004)&(f.dataset=='real')&(f.version=='factor_only')&(f.status=='ok')]
    if len(gap):text+=f"\n2004 section-5 formation: used {sorted(gap.formation_sessions_used.unique().tolist())} sessions; lost {sorted(gap.formation_sessions_lost.unique().tolist())}. January membership was never back-filled. Other sections use the full formation year.\n"
    text+='\nThe earliest fresh-data historical snapshot stored metadata rather than SHA hashes; its exact metadata check is supplemented by the later sealed RWB SHA snapshot (addendum 005). Failed verifier attempts remain in audit/.\n'
    text+='\n### Five-line summary\n\n'+'\n'.join(lines)+'\n'
    return text,lines

def report_all():
    ready=HERE/'audit/report_ready.json'
    if checkpoint(ready):return read(ready)['summary']
    frames,banks,gross=gather();assert len(frames['basic'][frames['basic'].dataset=='real'])==228765
    a=Analysis();a.vr(frames['basic']);a.partial(frames['partial']);a.drift(frames['basic'])
    a.triggers(banks['trigger_bank'],gross);a.factor(frames['factor'],banks['factor_bank']);a.predictions()
    for name,rows in a.tables.items():csv_atomic(TABLES/(name+'.csv'),pd.DataFrame(rows))
    partial=frames['partial'].copy()
    partial['starts']=partial.starts.map(lambda v:json.dumps(clean(v),separators=(',',':')))
    csv_atomic(TABLES/'partial_ci.csv',partial)
    csv_atomic(TABLES/'null_construction.csv',frames['null'])
    files=plots(a.tables,frames['partial'])
    body,summary=narrative(a,frames);path=HERE/'results.md'
    if path.exists():assert path.read_text()==body
    else:path.write_text(body)
    original=read(HERE/'audit/code_registration_final.json')['README_top_bytes'];readme=HERE/'README.md'
    if readme.stat().st_size==original:
        with readme.open('a') as f:f.write(body)
    else:assert readme.read_bytes()[original:]==body.encode()
    files+=list(TABLES.glob('*.csv'))+[path,readme]
    receipt(ready,files,summary=summary,real_pair_folds=int((frames['basic'].dataset=='real').sum()),
            null_pair_folds=int((frames['basic'].dataset=='null').sum()),partial_fit_rows=len(frames['partial']))
    return summary
