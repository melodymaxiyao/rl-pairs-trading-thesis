"""Independent MC selection/evaluation with a shared rolling observation map."""
from common import *
from simulation import Model,simulate
from pipeline import features
from observed import execute,policy,band,summary
import gc

def model(point):return Model(point['h0'],point['sigma_total'],family='W',w=point['w'])
def observations(m,n,key):
 p=simulate(m,n,['ou_realstrength_v1',*key],sessions=312);f=features(p)
 return dict(s=f['s'],valid=f['valid'],returns=np.ascontiguousarray(f['returns']))
def policy_name(pol):return json.dumps(pol,sort_keys=True,separators=(',',':'))
def all_policies():return [policy(bits) for bits in range(256)]+[band(a,b) for a in spec()['band_enter'] for b in spec()['band_exit']]
def statistics(out,pol,cost,ref=None,baseline=None):
 G=out[:,1];T=out[:,2];v=G-cost*T;n=len(v);closed=out[:,6].sum();row=dict(policy=pol,net=float(v.mean()),net_SE=float(v.std(ddof=1)/np.sqrt(n)),gross=float(G.mean()),turnover=float(T.mean()),entries=float(out[:,3].mean()),holding_period=float(out[:,5].sum()/closed) if closed else np.nan,gross_exposure=float(out[:,7].mean()),breakeven=float(G.mean()/T.mean()) if T.mean()>0 else np.nan,series_years=n)
 if ref is not None:
  delta=ref-v;row.update(regret_oracle=float(delta.mean()),regret_oracle_SE=float(delta.std(ddof=1)/np.sqrt(n)))
 if baseline is not None:
  delta=v-baseline;row.update(gain_over_AL=float(delta.mean()),gain_over_AL_SE=float(delta.std(ddof=1)/np.sqrt(n)))
 return row

def selection(rank,point):
 stem=f'runtime/benchmarks/selection_{rank}';old=cached(stem)
 if old is not None:return old
 os.environ['OU_SCOPE']='oracle_selection';data=observations(model(point),spec()['oracle']['selection_years'],['oracle_select']);values=np.empty((256,len(spec()['costs'])))
 for bits in range(256):
  out=execute(data,policy(bits),0)
  for j,c in enumerate(spec()['costs']):values[bits,j]=np.mean(out[:,1]-c*out[:,2])
  if bits%32==31:print('Oracle MC selection',rank,bits+1,'/256',flush=True);checkpoint()
 result=dict(rank=rank,bits={str(c):int(np.argmax(values[:,j])) for j,c in enumerate(spec()['costs'])},values=values,stream=['ou_realstrength_v1','oracle_select'],series_years=spec()['oracle']['selection_years']);save(stem,result);del data;gc.collect();return result

def fitted_model_selection(rank,world,estimate):
 stem=f'runtime/plugin/{rank}_world{world:02d}';old=cached(stem)
 if old is not None:return old
 if not estimate['valid']:
  return save(stem,dict(valid=False,bits={str(c):0 for c in spec()['costs']},fallback='pooled AR1 not stationary; deterministic no-trade'))
 os.environ['OU_SCOPE']='plugin_selection';m=Model(estimate['half_life'],estimate['sigma_e'],estimate['mean']);data=observations(m,spec()['plugin_MC_series_years'],['plugin',world]);values=np.empty((256,len(spec()['costs'])))
 for b in range(256):
  out=execute(data,policy(b),0)
  for j,c in enumerate(spec()['costs']):values[b,j]=np.mean(out[:,1]-c*out[:,2])
  if b%32==31:print('PlugIn MC selection',rank,world,b+1,'/256',flush=True);checkpoint()
 result=dict(valid=True,bits={str(c):int(np.argmax(values[:,j])) for j,c in enumerate(spec()['costs'])},stream=['ou_realstrength_v1','plugin',world],series_years=spec()['plugin_MC_series_years'],values=values,method='model-based MC enumeration using rolling features; not1D Bellman approximation');save(stem,result);del data;gc.collect();return result

def truth_table(rank,point,selected,data=None):
 stem=f'runtime/benchmarks/truth_{rank}';old=cached(stem)
 if old is not None:return old
 os.environ['OU_SCOPE']='truth';own=data is None
 if own:data=observations(model(point),spec()['oracle']['truth_years'],['truth'])
 refs={str(c):execute(data,policy(selected['bits'][str(c)]),c)[:,0] for c in spec()['costs']};baseline_out=execute(data,dict(kind=1),0);baselines={str(c):baseline_out[:,1]-c*baseline_out[:,2] for c in spec()['costs']};rows={}
 for j,pol in enumerate(all_policies()):
  out=execute(data,pol,0);rows[policy_name(pol)]={str(c):statistics(out,pol,c,refs[str(c)],baselines[str(c)]) for c in spec()['costs']}
  if j%32==31:print('Independent truth table',rank,j+1,'/268',flush=True);checkpoint()
 result=dict(rows=rows,stream=['ou_realstrength_v1','truth'],independent_of_selection=True,selection_stream=selected['stream'],series_years=spec()['oracle']['truth_years']);save(stem,result)
 if own:del data;gc.collect()
 return result
