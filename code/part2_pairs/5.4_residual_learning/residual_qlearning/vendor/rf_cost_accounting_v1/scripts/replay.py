"""Reconstruct only the sealed real positions; no fits or signal recomputation."""
from common import *
import numpy as np
from sealed_functions import update_positions,holdings_update
from accounting import rf_account,rp_account,exact_arrays

def panel():
 src=source();m=src.cache('runtime/data/calendar_v2/panel');prefix='runtime/data/calendar_v2/'
 for rel,digest in m['files'].items():assert src.expected[rel]==digest;src.verify(rel)
 return dict(meta=m,dates=np.load(src.verify(prefix+'dates.npy'),allow_pickle=False,mmap_mode='r'),returns=np.load(src.verify(prefix+'returns.npy'),allow_pickle=False,mmap_mode='r'),benchmark=np.load(src.verify(prefix+'benchmark.npy'),allow_pickle=False,mmap_mode='r'))
def reference(kind,year=None):
 value=source().cache('runtime/aggregates/'+(f'RP_{year}' if kind=='RP' else kind))
 return {r['date']:r for r in value['daily'] if r['world']=='real' and r['replicate']==0}
def check_row(row,ref):
 for key in ['gross_bps','positions','mean_closed_holding','closed_trades','closed_trades_today','closed_duration_today']:
  exact_arrays([row[key]],[ref[key]],key+' '+row['date'])
 exact_arrays([row['K0']],[ref['turnover']],'turnover '+row['date'])
 for c in [5,10,20]:exact_arrays([row['gross_bps']-c*row['K0']],[ref['gross_bps']-c*ref['turnover']],f'net{c} '+row['date'])
def rf_replay(data):
 src=source();ref=reference('RF');N=data['returns'].shape[1];dates=data['dates'];R=data['returns'];rows=[]
 state=dict(pos=np.zeros(N,np.int8),stock=np.zeros(N),durations=np.zeros(N,int),closed=[],kernel=None,baskets=np.zeros((N,15)))
 def H_from_state(st):
  h=np.zeros((N,N))
  if st['kernel'] is not None:
   k=st['kernel'];h[k['idx']]=k['negative_scaled_beta']@k['W'].T
  return h
 oldH=H_from_state(state)
 jobs=sorted({rel.rsplit('/',1)[-1][:-7] for rel in src.expected if rel.startswith('runtime/jobs/RF_') and rel.endswith('.pkl.gz')})
 for name in jobs:
  stem='runtime/checkpoints/'+name;done=cached(stem)
  if done is not None:
   state=done['state'];rows+=done['rows'];oldH=H_from_state(state);continue
  checkpoint();job=src.cache('runtime/jobs/'+name);output=[];stock_ledger=[];net_ledger=[];pos_ledger=[]
  for day in job['days']:
   t=day['index']
   if int(str(dates[t])[:4])<2004:continue
   idx=day['indices'];score=np.full(N,np.nan);half=np.full(N,np.nan);eligible=np.zeros(N,bool);score[idx]=day['score'][0];half[idx]=day['half'][0];eligible[idx]=day['status']=='ok'
   new=update_positions(state['pos'],score,half,eligible)
   if t==len(dates)-1:new[:]=0
   old_count=len(state['closed']);holdings_update(state['pos'],new,state['durations'],state['closed']);new_closes=state['closed'][old_count:];n=int((new!=0).sum());stock=np.zeros(N);H=np.zeros((N,N));baskets=np.zeros((N,15));kernel=None
   if n:
    local=new[idx];weight=local/day['gross']/n;stock[idx]=weight;negative_scaled_beta=-(weight[:,None]*day['beta']);H[idx]=negative_scaled_beta@day['weights'].T
    q=negative_scaled_beta.sum(axis=0);baskets=day['weights']*q[None,:];kernel=dict(idx=idx.copy(),negative_scaled_beta=negative_scaled_beta.copy(),W=day['weights'].copy())
   held=state['stock']!=0;used=held|np.any(oldH!=0,axis=0)
   gross=float(state['stock'][used]@R[t,used]+oldH[:,used].sum(axis=0)@R[t,used]) if np.isfinite(R[t,used]).all() else np.nan
   metrics=rf_account(stock,state['stock'],H,oldH,baskets,state['baskets'])
   row=dict(definition='RF',date=str(dates[t]),gross_bps=gross,positions=n,entries=int(((state['pos']==0)&(new!=0)).sum()),exits=len(new_closes),mean_closed_holding=float(np.mean(state['closed'])) if state['closed'] else np.nan,closed_trades=len(state['closed']),closed_trades_today=len(new_closes),closed_duration_today=sum(new_closes),**metrics)
   check_row(row,ref[row['date']]);output.append(row);stock_ledger.append(stock.copy());net_ledger.append(stock+H.sum(axis=0));pos_ledger.append(new.copy())
   state=dict(pos=new,stock=stock,durations=state['durations'],closed=state['closed'],kernel=kernel,baskets=baskets);oldH=H
  assert len(output)==sum(str(k)[:7]==name[3:7]+'-'+name[-2:] for k in ref)
  if output:
   import io
   buffer=io.BytesIO();np.savez_compressed(buffer,dates=np.array([r['date'] for r in output]),positions=np.stack(pos_ledger),stock_weights=np.stack(stock_ledger),net_portfolio_weights=np.stack(net_ledger));publish(RUN/('runtime/checkpoints/'+name+'_weights.npz'),buffer.getvalue())
  save(stem,dict(rows=output,state=state))
  rows+=output;print('K0 exact RF',name,len(output),'days',flush=True);checkpoint()
 assert len(rows)==len(ref) and len({r['date'] for r in rows})==len(ref)
 return rows

def stock_index(symbol,meta):
 symbol=str(symbol).upper();symbol=meta['aliases'].get(symbol,symbol)
 return meta['symbols'].index(symbol) if symbol in meta['symbols'] else -1

def rp_replay(data):
 src=source();N=data['returns'].shape[1];dates=data['dates'];rows=[]
 for year in spec()['years']:
  stem=f'runtime/checkpoints/RP_{year}';done=cached(stem)
  if done is not None:rows+=done['rows'];continue
  checkpoint();features=[]
  for part in range(12):features+=src.cache(f'runtime/jobs/RP_{year}_{part:02d}')['series']
  features.sort(key=lambda f:f['series']);ts=features[0]['indices'];assert all(np.array_equal(f['indices'],ts) for f in features)
  count_series=len(features);score=np.stack([f['score'][0] for f in features],axis=-1);half=np.stack([f['half'][0] for f in features],axis=-1);alpha=np.stack([f['alpha'] for f in features],axis=-1);beta=np.stack([f['beta'] for f in features],axis=-1);future=np.stack([f['future'][0,:,0] for f in features],axis=-1)
  pairs={r['pair']:r for r in data['meta']['pairs'][str(year)]};ia=[];ib=[]
  for f in features:
   pair=pairs[f['pair']];ia.append(stock_index(pair['leg_A'],data['meta']));ib.append(stock_index(pair['leg_B'],data['meta']))
  ia=np.array(ia);ib=np.array(ib);pos=np.zeros(count_series,np.int8);oldstock=np.zeros(count_series);oldhedge=np.zeros(count_series);dur=np.zeros(count_series,int);closed=[];ref=reference('RP',year);output=[];pos_ledger=[];stock_ledger=[];hedge_ledger=[]
  for ti,t in enumerate(ts):
   new=update_positions(pos,score[ti],half[ti],np.isfinite(beta[ti]))
   if ti==len(ts)-1:new[:]=0
   old_count=len(closed);holdings_update(pos,new,dur,closed);new_closes=closed[old_count:];active=new!=0;n=int(active.sum());stock=np.zeros(count_series);hedge=stock.copy()
   if n:
    allocation=np.full(n,1/n);stock[active]=new[active]/(1+abs(beta[ti,active]))*allocation;hedge[active]=-stock[active]*beta[ti,active]
   gross=0.
   if ti:
    held=oldstock!=0;one=alpha[ti-1]+future[ti-1];gross=float(oldstock[held]@one[held]) if np.isfinite(one[held]).all() else np.nan
   metrics=rp_account(stock,oldstock,hedge,oldhedge,ia,ib,N)
   row=dict(definition='RP',date=str(dates[t]),gross_bps=gross,positions=n,entries=int(((pos==0)&(new!=0)).sum()),exits=len(new_closes),mean_closed_holding=float(np.mean(closed)) if closed else np.nan,closed_trades=len(closed),closed_trades_today=len(new_closes),closed_duration_today=sum(new_closes),**metrics)
   check_row(row,ref[row['date']]);output.append(row);pos_ledger.append(new.copy());stock_ledger.append(stock.copy());hedge_ledger.append(hedge.copy());pos=new;oldstock=stock;oldhedge=hedge
  assert len(output)==len(ref)
  import io
  buffer=io.BytesIO();np.savez_compressed(buffer,dates=dates[ts],series=np.array([f['series'] for f in features]),positions=np.stack(pos_ledger),stock_weights=np.stack(stock_ledger),hedge_weights=np.stack(hedge_ledger),leg_A_indices=ia,leg_B_indices=ib);publish(RUN/f'runtime/checkpoints/RP_{year}_weights.npz',buffer.getvalue())
  save(stem,dict(rows=output))
  rows+=output;print('K0 exact RP',year,len(output),'days',flush=True);checkpoint()
 return rows
