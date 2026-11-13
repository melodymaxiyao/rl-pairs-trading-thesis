"""Twenty training worlds per point; train-only decisions, fresh shared tests."""
from common import *
from simulation import simulate,estimate_ar
from pipeline import features
from benchmarks import model,observations,fitted_model_selection
from learning import q_pow,band_train,spi_td
from observed import execute,policy,al,summary

def world_job(args):
 guard();os.environ['OU_SCOPE']='training';rank,point,world,oracle=args;stem=f'runtime/worlds/{rank}_{world:02d}';old=cached(stem)
 if old is not None:return old
 m=model(point);p=simulate(m,1500,['ou_realstrength_v1','train',world],sessions=312);train=features(p);estimate=estimate_ar(train['levels']);count('synthetic_pooled_AR_model_fits');plugin=fitted_model_selection(rank,world,estimate);test=observations(m,spec()['test']['fresh_series_years_per_world'],['test',world]);training_hash=hashlib.sha256(np.ascontiguousarray(p).tobytes()).hexdigest();test_hash=hashlib.sha256(test['s'].tobytes()+test['returns'].tobytes()).hexdigest();rows=[];policies={};gates=[]
 for c in spec()['costs']:
  cstem=stem+f'_c{c}';done=cached(cstem)
  if done is None:
   os.environ['OU_SCOPE']='TD_production';q=q_pow(train,rank,world,c);b=band_train(train,c);s=spi_td(train,q['bits'],c);pols={'Oracle-8':policy(oracle['bits'][str(c)]),'AL rule':al(),'Band-train':b['policy'],'PlugIn-8':policy(plugin['bits'][str(c)]),'Q_POW':policy(q['bits']),'SPI_TD':s['policy']};out=[]
   for learner,pol in pols.items():out.append(dict(rank=rank,world=world,cost=c,learner=learner,test=summary(execute(test,pol,c)),policy=pol,plugin_valid=estimate['valid']))
   done=dict(rows=out,policies=pols,SPI=s,band=b,training_hash=training_hash,test_hash=test_hash,estimate=estimate);save(cstem,done);print('World cost committed',rank,world,c,flush=True);checkpoint()
  rows+=done['rows'];policies[str(c)]=done['policies'];gates.append(dict(cost=c,**done['SPI']))
 result=dict(rank=rank,world=world,rows=rows,policies=policies,gates=gates,estimate=estimate,training_hash=training_hash,test_hash=test_hash);save(stem,result);checkpoint();return result
