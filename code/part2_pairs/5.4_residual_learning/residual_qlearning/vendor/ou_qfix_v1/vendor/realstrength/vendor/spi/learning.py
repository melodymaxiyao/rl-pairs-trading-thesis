"""Factorial refits and unchanged v3 selection, with per-fit diagnostic receipts."""
from common import *
from sources import setup,seed
from kernel import fit_cached,policy_id
import numpy as np

def ensemble(es,risk,step,gamma,context,early=True,order='original',nseeds=10):
 assert all(e['window']=='train' for e in es)
 stem='runtime/models/'+hashlib.sha256(json.dumps(dict(context=context,step=step,gamma=gamma,early=early,order=order,nseeds=nseeds),sort_keys=True).encode()).hexdigest();old=cached(stem)
 if old is not None:return old
 fits=[fit_cached(es,[],step,gamma,seed(j),risk,early=early,order=order,context=dict(context,seed_index=j,phase='final')) for j in range(nseeds)];tables=np.asarray([f['q'] for f in fits])
 return save(stem,dict(q=tables.mean(axis=0),models=tables,fits=fits,policy_id=policy_id(tables.mean(axis=0)),selected=dict(step=step,gamma=gamma,inner_selection=False)))

def original_v3(es,risks,context):
 stem='runtime/models/v3_'+hashlib.sha256(json.dumps(context,sort_keys=True).encode()).hexdigest();old=cached(stem)
 if old is not None:return old
 fits=[]
 def adapter(train,val,alpha,gamma,seed_value,label,risk=None,**kw):
  step='C01' if alpha==.01 else 'C05';assert alpha in [.01,.05]
  f=fit_cached(train,val,step,gamma,seed_value,risk,context=dict(context,phase='inner' if val else 'final'),**kw);fits.append(f);return f
 out=setup().sources.selection(es,risks,adapter);tables=np.asarray(out['models'])
 return save(stem,dict(q=tables.mean(axis=0),models=tables,fits=fits,policy_id=policy_id(tables.mean(axis=0)),selected=out['selected']))

def fit_rows(model,metadata):
 diag=[];stop=[];traj=[]
 for j,f in enumerate(model['fits']):
  meta=f['meta'];sig=f['signature'];phase=sig['context'].get('phase','final');base=dict(metadata,fit_index=j,phase=phase,seed=meta['seed'],step=meta['alpha'],gamma=meta['gamma'],order=meta['order'],epochs=meta['epochs']);q=f['q'];vis=f['visits'];pid=policy_id(q)
  stop.append(dict(base,stopped_early=meta['converged'] and meta['early'],stop_cause='convergence' if meta['converged'] and meta['early'] else 'epoch_cap',converged=meta['converged'],convergence_epoch=meta['convergence_epoch'],policy_id=pid,never_enter=not bool(pid&0x55),exit_everywhere=(pid&0xAA)==0xAA,greedy_when_stopped_never_enter=(not bool(pid&0x55)) if meta['converged'] and meta['early'] else None))
  for s in range(8):
   for a in range(2):
    W,V,ESS,last,tail,pair,mean,sd,emean,gap=f['diagnostics'][s,a];n=int(vis[s,a]);diag.append(dict(base,state=s,action=a,n=n,Q=q[s,a],W=W,V=V,ESS=ESS,initial_weight=1-W,last_epoch_weight_share=last/W if W else np.nan,last_10pct_episodes_weight_share=tail/W if W else np.nan,last_10pct_unique_pairs_in_final_epoch_weight_share=pair/W if W else np.nan,target_mean=mean,target_sd=sd,predicted_SE=sd*np.sqrt(V),final_epoch_target_mean=emean,recency_gap=gap))
  if metadata['k']==1 and metadata['world']==0 and metadata['cell'] in ['C01_g99','POW_g1'] and phase=='final':
   for epoch,table in enumerate(f['trajectory'],1):
    for s in range(8):
     for a in range(2):traj.append(dict(base,epoch=epoch,state=s,action=a,Q=table[s,a]))
 return diag,stop,traj
