"""One immutable result per cell/world. Streams exclude worker/learner/cost."""
from common import *
from simulation import Model,training,simulate,estimate_ar,model_from_estimate
from dp import grid_model,optimal,oracle8,policy_array,evaluate,physical_equal,exact_q_projection
from evaluation import execute,summary,policy,grid_policy,mc_oracle
from native import train
def design(tier=3):
 cells=[]
 for h in [5,10,20,40]:
  for P in [100,600]:cells.append(dict(tier=1,arm='OU',h=h,P=P,cost=10.,w=0.))
 if tier>=2:
  for family,w in [('W',.1),('W',.3),('T4',0.),('K',0.)]:cells.append(dict(tier=2,arm=family,h=20,P=600,cost=10.,w=w))
 if tier>=3:
  for c in [5.,20.]:cells.append(dict(tier=3,arm='OU',h=20,P=600,cost=c,w=0.))
 return [dict(cell,world=j) for cell in cells for j in range(20)]
def identifier(job):return f"T{job['tier']}_{job['arm']}_w{job['w']:g}_h{job['h']}_P{job['P']}_c{job['cost']:g}_world{job['world']:02d}"
def models(job):
 a=job['arm'];family='OU' if a=='K' else a
 return Model(job['h'],family=family,w=job['w']),Model(2*job['h'] if a=='K' else job['h'],family=family,w=job['w'])
def on_grid(pol,x,stop_sd,sessions):
 if pol['kind']==3:
  own=pol['x'];indices=np.clip(np.floor((x-own[0])/(own[1]-own[0])+.5).astype(int),0,len(own)-1);actions=pol['actions'][:,:,indices].copy()
  for j,p in enumerate([-1,0,1]):
   if p==0:actions[:,j]=np.where(actions[:,j]!=0,-np.sign(x).astype(int),0)
   actions[:,j,abs(x)>=4*stop_sd]=0
  return actions
 return policy_array(x,pol['scale'],stop_sd,bits=pol.get('bits') if pol['kind']==0 else None,rule={1:'AL rule',2:'L0-type rule'}.get(pol['kind']))
def benchmark(job,test_model,points=401,sessions=252,mc_pairs=50000,checkpoint=True):
 key=f"runtime/benchmarks/{job['arm']}_w{job['w']:g}_h{job['h']}_c{job['cost']:g}"
 result=cached(key) if checkpoint else None
 if result is not None:return result
 nominal=Model(test_model.half_life) if test_model.family=='W' else test_model
 x,kernel,initial=grid_model(nominal,points=points);opt=optimal(x,kernel,initial,test_model.sd,job['cost'],sessions)
 if test_model.family=='W':
  paths=simulate(test_model,mc_pairs,['oracle_selection',test_model.family,test_model.half_life,test_model.w],sessions);oracle=mc_oracle(paths,test_model.sd,test_model.sd,job['cost'])
 else:oracle=oracle8(x,kernel,initial,test_model.sd,test_model.sd,job['cost'],sessions)
 result=dict(x=x,kernel=kernel,initial=initial,opt=opt,oracle=oracle)
 if checkpoint:save(key,result);checkpoint_guard()
 return result
def world(job,small=None):
 # small is used exclusively by the synthetic preparation tests, never by run.py.
 os.environ['OU_SCOPE']='test' if small else 'production';s=small or {};T=s.get('sessions',252);points=s.get('points',401);test_n=s.get('test_pairs',5000);mc_n=s.get('mc_pairs',50000);epochs=s.get('epochs',300);P=s.get('pairs',job['P']);is_test=small is not None;name=identifier(job);stem=f'runtime/worlds/{name}'
 old=cached(stem) if not is_test else None
 if old is not None:return old
 started=time.monotonic();tr_model,te_model=models(job);paths=training(tr_model,P,job['world'],T);est=estimate_ar(paths);scale=est['Q_scale'];assert scale>0 and np.isfinite(scale)
 b=benchmark(job,te_model,points,T,mc_n,checkpoint=not is_test);x,k,initial=b['x'],b['kernel'],b['initial'];policies={'OPT-grid':grid_policy(b['opt'],x,te_model.sd),'Oracle-8':policy(b['oracle']['bits'],te_model.sd)}
 if est['valid']:
  phat=model_from_estimate(est);px,pk,pi=grid_model(phat,points=points);ps=est['stationary_sd'];po=oracle8(px,pk,pi,ps,tr_model.sd,job['cost'],T);pg=optimal(px,pk,pi,tr_model.sd,job['cost'],T);policies.update({'PlugIn-8':policy(po['bits'],ps),'PlugIn-grid':grid_policy(pg,px,ps)})
 else:policies.update({'PlugIn-8':policy(0,scale),'PlugIn-grid':policy(0,scale)})
 fits={}
 for learner,step in [('Q_POW','POW'),('Q_C01','C01')]:
  fits[learner]=train(paths,scale,tr_model.sd,step,job['cost'],epochs,checkpoint=None if is_test else f'runtime/seeds/{name}');policies[learner]=policy(fits[learner]['bits'],scale)
 policies.update({'AL rule':dict(kind=1,scale=scale),'L0-type rule':dict(kind=2,scale=scale)})
 test=simulate(te_model,test_n,['test',te_model.family,te_model.half_life,te_model.w,job['P'],job['world']],T)
 truth=simulate(te_model,mc_n,['true_value',te_model.family,te_model.half_life,te_model.w],T) if job['tier']==2 else None
 rows=[];true_values={};paired={}
 for learner in spec()['learners']:
  pol=policies[learner];test_stats=summary(execute(test,pol,job['cost'],te_model.sd));values=execute(truth,pol,job['cost'],te_model.sd) if truth is not None else None
  true_value=float(values[:,0].mean()) if values is not None else float(evaluate(x,k,initial,on_grid(pol,x,te_model.sd,T),job['cost'],T)[0]);true_values[learner]=true_value
  if values is not None:paired[learner]=values[:,0]
  no_entry=not any((pol.get('bits',0)>>j)&1 and lower*pol['scale']<4*te_model.sd for j,lower in enumerate([0,.5,1.5,2.5])) if pol['kind']==0 else None
  row=dict(job,learner=learner,true_value=true_value,true_value_SE=float(values[:,0].std(ddof=1)/np.sqrt(mc_n)) if values is not None else 0.,value_method='independent Monte Carlo' if truth is not None else 'finite-grid policy evaluation',test_net=test_stats.pop('net'),test_net_SE=test_stats.pop('net_SE'),**test_stats,policy_id=pol.get('bits'),normalizer=pol['scale'],no_entry_policy=no_entry,plugin_valid=est['valid'],OPT_grid_label='nominal-OU reference, not an upper bound' if te_model.family=='W' else 'discretized finite-horizon optimum',raw_equals_oracle=pol.get('bits')==b['oracle']['bits'] if pol['kind']==0 else False,physical_equals_oracle=physical_equal(x,pol.get('bits',0),pol['scale'],b['oracle']['bits'],te_model.sd,te_model.sd) if pol['kind']==0 else False)
  rows.append(row)
 for row in rows:
  learner=row['learner'];row.update(regret_oracle=true_values['Oracle-8']-row['true_value'],regret_OPT=np.nan if te_model.family=='W' else true_values['OPT-grid']-row['true_value'],regret_oracle_SE=float((paired['Oracle-8']-paired[learner]).std(ddof=1)/np.sqrt(mc_n)) if paired else 0.)
 validation=[]
 if te_model.family!='W':
  projected,weights=exact_q_projection(x,k,initial,fits['Q_POW']['bits'],scale,te_model.sd,job['cost'],T)
  for state in range(8):
   for action in [0,1]:validation.append(dict(job,type='Q_projection',state=state,action=action,learned_Q=fits['Q_POW']['q'][state,action],exact_Q=projected[state,action],occupancy_weight=weights[state,action]))
 else:validation.append(dict(job,type='Q_projection',status='undefined: observed W sum is not one-dimensional Markov'))
 for learner,f in fits.items():
  for j,fit in enumerate(f['fits']):validation.append(dict(job,type='TD_fit',learner=learner,seed=j,epochs=fit['epochs'],converged=fit['converged'],convergence_epoch=fit['convergence_epoch'],steps=fit['learning_steps']))
 result=dict(job=job,rows=rows,parameters=dict(job,**est,true_mean=tr_model.mean,true_phi=tr_model.phi,true_sigma_e=tr_model.sigma_e,true_half_life=tr_model.half_life),policies=policies,validation=validation,seconds=time.monotonic()-started,benchmark_opt_value=b['opt']['value'],test_mode=is_test)
 # Keep all seed-level trajectories in immutable seed checkpoints, not duplicate them here.
 if not is_test:save(stem,result);count('completed_production_worlds');checkpoint_guard()
 return result
def worker(job):
 from common import guard
 guard();return world(job)
