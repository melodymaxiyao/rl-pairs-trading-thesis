"""One synthetic world: freeze train choices, select independent oracles, test."""
from sc_common import *
from sc_simulation import training,test_batches,risk_limits,event_values
from sc_rules import GRID,DEFAULT,choose
from sc_kernel import descriptor,policy_q,batch
from sc_learning import tuned,v3,table_metrics
import numpy as np

def oracle(h,w,sigma,case,k,world,risk):
    stem=f'runtime/oracles/{case}_k{k}_world{world}'
    old=cached(stem)
    if old is not None:return old
    cfg=spec();policies=[descriptor(q=policy_q(i)) for i in range(256)]+GRID;tot=np.zeros((263,3));started=time.monotonic()
    for es in test_batches(h,w,sigma,['oracle',case,k],cfg['oracle_pairs'],case=='fat_tails',case=='heterogeneous'):
        values,_=batch(es,policies,risk);tot+=values.sum(axis=1);count('oracle_policy_evaluations',len(es)*263)
    means=tot/cfg['oracle_pairs'];best=int(np.argmax(means[:256,0]));br=int(np.argmax(means[256:,0]))
    result=dict(policy_bits=best,B3_index=br,selection_metrics=means,oracle_net=float(means[best,0]),oracle_B3_net=float(means[256+br,0]),risk=risk,selection_paths=cfg['oracle_pairs'],seconds=time.monotonic()-started,selection_stream=['oracle',case,k])
    return save(stem,result)

def proxy_scores(es):
    result={}
    for pair in sorted({e['pair'] for e in es}):
        rows=event_values([e for e in es if e['pair']==pair]);result[pair]=float(np.mean([r['net'] for r in rows])) if rows else np.nan
    return result

def grouped_train(es,risk,pooled,proxy=False):
    scores=proxy_scores(es) if proxy else {};finite=[v for v in scores.values() if np.isfinite(v)]
    threshold=float(np.median(finite)) if proxy and finite else None
    if proxy:assert threshold is not None,'No finite train-only proxy scores'
    def group(e):return int(np.isfinite(scores[e['pair']]) and scores[e['pair']]>threshold) if proxy else e['true_type']
    models=[]
    for g in [0,1]:
        sub=[e for e in es if group(e)==g];assert sub,('Empty synthetic training group',g)
        models.append(tuned(sub,risk))
    return dict(q=np.asarray([m['q'] for m in models]),models=np.concatenate([m['models'] for m in models]),selected=dict(grouping='train_event_net' if proxy else 'true_type',threshold=threshold,missing_score_to_low_group=sum(not np.isfinite(v) for v in scores.values()) if proxy else 0),grouped=True,proxy=proxy)

def job_key(job):return f"tier{job['tier']}_{job['case']}_k{job['k']}_D{job['D']}_world{job['world']}"

def execute(job):
    from schedule import authorize
    tier=job['tier']
    assert job['D']==1,'Tier 2 deferred: D>1 is forbidden'
    assert job['world'] in range(5)
    assert job['k'] in spec()['k_grid'] and (tier!=3 or job['k']==1)
    if tier==3:
        from sc_report import verify_tier
        verify_tier(1)
    authorize(tier,tier1_verified=(HERE/'audit/tier1_seal.json').is_file())
    stem='runtime/worlds/'+job_key(job);old=cached(stem)
    if old is not None:return old
    COUNTS.clear();started=time.monotonic();cfg=spec();cal=read(HERE/'outputs/tables/calibration.json')['chosen'];h=cal['h'];w=min(job['k']*cal['w'],.95);sigma=cfg['calibration']['targets']['sigma_bps']
    case=job['case'];dgp='gaussian' if case=='protocol_ablation' else case
    train_key=['world_train',dgp,job['k'],job['world']]
    es=training(h,w,sigma,train_key,cfg['train_pairs'],dgp=='fat_tails',dgp=='heterogeneous');risk=risk_limits(es);risks=dict(inner=risk_limits([e for e in es if e['year']<2003]),final=risk)
    choices=choose(es,risk);timings=dict(train_and_rules=time.monotonic()-started);models={}
    t=time.monotonic();models['TD_v3']=v3(es,risks);timings['TD_v3']=models['TD_v3']['fit_seconds']
    t=time.monotonic();models['TD_tuned']=tuned(es,risk);timings['TD_tuned']=models['TD_tuned']['fit_seconds']
    if case=='protocol_ablation':
        for name,call in [('TD_tuned_50',lambda:tuned(es,risk,50)),('TD_tuned_gamma99',lambda:tuned(es,risk,gamma=.99)),('TD_v3_gamma90',lambda:v3(es,risks,(.90,))),('TD_v3_gamma99',lambda:v3(es,risks,(.99,)))]:models[name]=call()
    if case=='heterogeneous':
        models['TD_true_grouped']=grouped_train(es,risk,models['TD_tuned'])
        models['TD_proxy_grouped']=grouped_train(es,risk,models['TD_tuned'],True)
    # No test path has been generated or inspected in any selection above.
    o=oracle(h,w,sigma,dgp,job['k'],job['world'],risk);timings['oracle']=o['seconds']
    policies={'L0':GRID[DEFAULT],**{name:GRID[choices[name]] for name in ['B3','B3_shrunk','B3_lcb']}}
    policies.update({name:descriptor(q=m['q']) for name,m in models.items() if not m.get('grouped')})
    policies['oracle256']=descriptor(q=policy_q(o['policy_bits']));policies['oracleB3']=GRID[o['B3_index']]
    names=list(policies);totals={name:np.zeros(3) for name in names+list(n for n,m in models.items() if m.get('grouped'))};l0_vis=np.zeros(8);agreements={};N=0;t=time.monotonic()
    oracle_action=(policy_q(o['policy_bits'])[:,1]>policy_q(o['policy_bits'])[:,0]).astype(int)
    for batch_es in test_batches(h,w,sigma,['world_test',dgp,job['k'],job['world']],cfg['test_pairs'],dgp=='fat_tails',dgp=='heterogeneous'):
        values,vis=batch(batch_es,list(policies.values()),risk);N+=len(batch_es);l0_vis+=vis[names.index('L0')]
        for j,name in enumerate(names):totals[name]+=values[j].sum(axis=0)
        for name,m in models.items():
            if not m.get('grouped'):continue
            for g in [0,1]:
                subset=[e for e in batch_es if (int(np.isfinite(e['proxy_score']) and e['proxy_score']>m['selected']['threshold']) if m['proxy'] else e['true_type'])==g]
                if not subset:continue
                vals,v=batch(subset,[GRID[DEFAULT],descriptor(q=m['q'][g])],risk);totals[name]+=vals[1].sum(axis=0)
                same=(m['q'][g,:,1]>m['q'][g,:,0])==oracle_action
                a=agreements.setdefault(name,np.zeros(2));a[0]+=np.dot(v[0],same);a[1]+=v[0].sum()
    timings['test_execution']=time.monotonic()-t
    assert N==cfg['test_pairs'];means={name:v/N for name,v in totals.items()};rows=[]
    for name,v in means.items():
        m=models.get(name);row=dict(job,h=h,w=w,learner=name,net=float(v[0]),gross=float(v[1]),turnover=float(v[2]),regret=float(means['oracle256'][0]-v[0]),oracle_B3_headroom=float(means['oracleB3'][0]-means['L0'][0]),keeps_default=(name=='L0' or name in choices and choices[name]==DEFAULT),risk=risk)
        if m:
            row.update(table_metrics(m));row['selected']=m['selected']
            qs=m['q'] if m.get('grouped') else m['q'][None,:,:]
            same=(qs[:,:,1]>qs[:,:,0])==oracle_action[None,:];row['agreement']=float(same.mean())
            row['weighted_agreement']=float(agreements[name][0]/agreements[name][1]) if m.get('grouped') and agreements[name][1]>0 else float(np.dot(l0_vis,same[0])/l0_vis.sum()) if not m.get('grouped') and l0_vis.sum()>0 else np.nan
        rows.append(row)
    result=dict(job=job,rows=rows,models=models,choices=choices,risk=risk,oracle=o,timings=timings,elapsed_seconds=time.monotonic()-started,counts=dict(COUNTS),test_paths=N)
    save(stem,result);print('Completed',job_key(job),'seconds',round(result['elapsed_seconds'],1),flush=True);return result
