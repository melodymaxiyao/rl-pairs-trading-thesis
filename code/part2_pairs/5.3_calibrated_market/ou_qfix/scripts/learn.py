"""Exactly one Double ensemble/world/cost; LCB is a frozen deployment rule."""
from common import *
from reproduction import train_data,source_q,source_world
from qkernel import double_fit,nominal_se,policy_bits
def export(result):
 compact=dict(result)
 compact['source_q_fits']=[{k:f[k] for k in ['q','visits','diagnostics','epochs','converged','convergence_epoch','learning_steps']} for f in result['source_q_fits']]
 write(RUN/f"outputs/models/world{result['world']:02d}_c{result['cost']}.json",compact)
def fit_world_cost(args):
 guard();checkpoint();world,cost=args;assert read(RUN/'audit/baseline_reproduction.json')['status']=='PASS_BIT_EXACT';stem=f'runtime/checkpoints/new_world{world:02d}_c{cost}';old=cached(stem)
 if old is not None:export(old);return old
 data,h=train_data(world);assert h==source_world(world,cost)['training_hash'];ret=np.column_stack([data['returns'],np.zeros(len(data['returns']))]);z=np.where(data['valid'],data['s'],np.inf);fits=[]
 for j in range(10):
  key=stem+f'_seed{j:02d}';fit=cached(key)
  if fit is None:
   os.environ['OU_SCOPE']='Double_production';fit=double_fit(z,ret,td_seed(j),cost,epochs=spec()['training']['epochs']);save(key,fit)
  fits.append(fit);print('Double seed committed',world,cost,j,'epochs',fit['epochs'],flush=True);checkpoint()
 doubleq=np.mean([f['q'].mean(axis=0) for f in fits],axis=0);dse=nominal_se(fits,double=True);old=source_q(world,cost);models={}
 for name,q,se,lcb in [('Q_POW',old['q'],old['se'],False),('Q_LCB',old['q'],old['se'],True),('Q_DBL',doubleq,dse,False),('Q_DBL_LCB',doubleq,dse,True)]:
  models[name]=dict(q=q,se=se,bits=policy_bits(q,se if lcb else None),raw_greedy_bits=policy_bits(q),entry_LCB=lcb,z=1.645 if lcb else 0,SE_label=spec()['LCB']['label'])
 result=dict(world=world,cost=cost,models=models,training_hash=h,double_fits=fits,source_q_fits=old['fits'],original_seed_indices=list(range(10)),real_trials=552,new_real_trials=0)
 save(stem,result);export(result);checkpoint();return result
