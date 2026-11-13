"""Fixed capital, entry-frozen physical hedge book and budget-balanced K2 attribution."""
from common import *
from data import entry_vector,panel,scores,M
from kernel import positions
from missing_prices import marked_prices,causal_target,LOG_COLUMNS

def state(s,p):
 if not p:return 0 if abs(s)<.5 else 1 if abs(s)<1.5 else 2 if abs(s)<2.5 else 3
 z=-p*s;return 4+(0 if z>=1.5 else 1 if z>=.5 else 2 if z>=-.5 else 3)

def capital(c):
 if c['kind']=='RP':w=np.array([r['ipw'] for r in c['rows']],float);return w/w.sum()
 u=read(RUN/'audit/RF_potential_members.json');assert sha(RUN/'audit/RF_potential_members.json')==spec()['RF_universe_sha256'];allowed=set(u['symbols']);return np.array([1/u['N'] if name in allowed else 0. for name in c['ids']])

def execution(c,pol,gate,wi=0,weight=None,details=True,cost=0.):
 d=panel();market=marked_prices(d);R=market['returns'];quotes=market['available'];affected=market['affected'];n,T=c['N'],c['T'];S=R.shape[1];canonical=np.argsort(np.asarray(c['ids'],str),kind='stable');rank=np.empty(n,int);rank[canonical]=np.arange(n);z=scores(c,gate,wi);p=np.zeros((n,T),np.int32);pending_exit=np.zeros(n,bool);pending_forced=np.zeros(n,bool);pending_entry={};deferred_since=np.full(n,-1,int);opened=np.zeros(n,int);missing_log=[];w=capital(c) if weight is None else np.asarray(weight,float);book=np.zeros((n,S));stock=np.zeros(n);old=np.zeros(n,np.int32);entry=np.zeros(n,int);dur=np.zeros(n,int)
 raw=np.zeros((n,7));daily=[];bins=[];closed=[[] for _ in range(n)];entry_n=np.zeros((n,4),int);entry_op=np.zeros((n,4),int);exit_n=np.zeros((n,4),int);exit_op=np.zeros((n,4),int);forced=np.zeros(n,int);terminal=np.zeros(n,int);allcloses=np.zeros(n,int);signs=None
 if wi:
  signs=np.stack([M.sign_series(source().prereg['null_seed'],('RP|'+str(c['year'])+':'+name) if c['kind']=='RP' else 'RF|'+name,wi-1,c['dates']) for name in c['ids']])
 perday=[]
 for ti,t in enumerate(c['ts']):
  active=old!=0;gross=np.zeros(n)
  if active.any():
   used=np.any(book[active]!=0,axis=0);assert np.isfinite(R[t,used]).all(),('Unknown held-leg return; no source modification or zero-fill',c['kind'],c['year'],str(d['dates'][t]),[d['meta']['symbols'][j] for j in np.flatnonzero(used&~np.isfinite(R[t]))])
   gross[active]=book[active][:,used]@R[t,used]
   if wi:
    ii=np.flatnonzero(active);scale=c['alpha'][ii,entry[ii]]/c['gross'][ii,entry[ii]];carry=w[ii]*old[ii]*scale;gross[ii]=carry+signs[ii,ti]*(gross[ii]-carry)
  desired=causal_target(z[:,ti],old,pol,terminal=ti==T-1);desired[pending_exit&(old!=0)]=0;new=desired.copy();entry_vectors={};basis_dates={};day_events=[];prior_book=book.copy() if np.any(affected[t]) else None
  for i in canonical:
   if old[i]==0 and i in pending_entry:
    if ti==T-1 or not np.isfinite(z[i,ti]):
     pending_entry.pop(i);deferred_since[i]=-1
    else:desired[i]=pending_entry[i]['side'];new[i]=desired[i]
   if desired[i]==old[i]:continue
   if old[i]:needed=book[i]!=0;action='exit'
   else:
    queued=pending_entry.get(i);v=queued['vector'] if queued else entry_vector(c,i,ti)*desired[i]*w[i];needed=v!=0;action='entry';entry_vectors[i]=v;basis_dates[i]=queued['basis'] if queued else ti
   blocked=np.flatnonzero(needed&~quotes[t])
   if len(blocked):
    new[i]=old[i]
    if deferred_since[i]<0:deferred_since[i]=ti
    if old[i]:pending_exit[i]=True;pending_forced[i]|=not np.isfinite(z[i,ti])
    else:pending_entry[i]=dict(side=int(desired[i]),vector=entry_vectors[i].copy(),basis=basis_dates[i])
    for j in blocked:day_events.append((i,j,'deferred_'+action,float(book[i,j]),action,False))
   elif deferred_since[i]>=0:
    jj=np.flatnonzero(needed&affected[t]);jj=jj if len(jj) else np.flatnonzero(needed)[:1]
    for j in jj:day_events.append((i,j,'executed_deferred_'+action,float(book[i,j]),action,True))
  if prior_book is not None:
   for j in np.flatnonzero(affected[t]):
    for i in canonical:
     if old[i] and prior_book[i,j]!=0:day_events.append((i,j,'carried_valuation' if not quotes[t,j] else 'recovery_valuation',float(prior_book[i,j]),'hold' if desired[i]==old[i] else 'exit',False))
     signal_uses=(i==j) if c['kind']=='RF' else bool((c.get('ia',np.full(n,-1))[i]==j) or (c.get('ib',np.full(n,-1))[i]==j))
     if signal_uses and not np.isfinite(z[i,ti]):day_events.append((i,j,'signal_invalid_unfilled_return',float(prior_book[i,j]),'no_new_entry',False))
  p[:,ti]=new;changed=np.flatnonzero(new!=old);changed=changed[np.argsort(rank[changed],kind='stable')];delta=np.zeros((len(changed),S));own_delta=np.zeros(n);pre_exposure=w*active;raw[:,4]+=pre_exposure;raw[:,5]+=active
  for j,i in enumerate(changed):
   if old[i]:
    delta[j]=-book[i];own_delta[i]=-stock[i];book[i]=0;stock[i]=0;closed[i].append(ti-opened[i]);allcloses[i]+=1
    if not np.isfinite(z[i,ti]) or pending_forced[i]:forced[i]+=1
    if ti==T-1:terminal[i]+=1
   if new[i]:
    v=entry_vectors[i];basis=basis_dates[i];delta[j]+=v;book[i]=v;stock[i]=new[i]*w[i]/c['gross'][i,basis];own_delta[i]+=stock[i];entry[i]=basis;opened[i]=ti
  if len(changed):
   absd=abs(delta);den=absd.sum(axis=0);net=delta.sum(axis=0);netturn=abs(net);share=np.divide(netturn,den,out=np.zeros(S),where=den>0);allocation=absd@share;raw[changed,1]+=allocation;raw[changed,2]+=w[changed]*abs(new[changed]-old[changed])
   np.testing.assert_allclose(allocation.sum(),netturn.sum(),rtol=1e-12,atol=1e-14)
  else:netturn=np.zeros(S);allocation=np.empty(0)
  raw[:,0]+=gross;raw[:,3]+=abs(own_delta)
  override_count=0;valid_count=0
  for i in range(n):
   if np.isfinite(z[i,ti]) and ti<T-1:
    valid_count+=1;st=state(z[i,ti],old[i]);
    if pol.get('kind')==3 and ((pol['bits']>>(8+st))&1):
     baseline_action=abs(z[i,ti])>1.25 if not old[i] else -old[i]*z[i,ti]<.5
     override_count+=int(((pol['bits']>>st)&1)!=baseline_action)
    (entry_op if not old[i] else exit_op)[i,st if not old[i] else st-4]+=1
    if not old[i] and new[i]:entry_n[i,st]+=1
    if old[i] and not new[i]:exit_n[i,st-4]+=1
  row=dict(date=str(d['dates'][t]),test_year=c['year'],gross_bps=float(gross[canonical].sum()),K2=float(netturn.sum()),PP_FULL=float(np.sum((w*abs(new-old))[canonical])),K1=float(abs(own_delta[canonical]).sum()),gross_exposure=float(np.sum((w*(new!=0))[canonical])),net_stock_gross=float(abs(book[canonical].sum(axis=0)).sum()),positions=int(np.count_nonzero(new)),entries=int(np.sum((old==0)&(new!=0))),exits=int(np.sum((old!=0)&(new==0))))
  assert row['K2']<=row['PP_FULL']+1e-12,'Triangle inequality violated'
  row['SPI_override_decisions']=override_count;row['valid_decisions']=valid_count
  daily.append(row)
  if details:
   alloc_full=np.zeros(n);alloc_full[changed]=allocation;pp=np.zeros(n);pp[changed]=w[changed]*abs(new[changed]-old[changed])
   perday.append(np.column_stack([gross,alloc_full,pp,abs(own_delta)]))
  allocation_by_i=dict(zip(changed,allocation));grouped={};fee_recorded=set()
  for i,j,event,physical,requested,executed in day_events:
   item=grouped.setdefault((int(i),int(j)),dict(events=[],physical=physical,requested=requested));item['events'].append(event)
   if requested in ['entry','exit']:item['requested']=requested
  for (i,j),item in grouped.items():
   mult=1. if wi==0 else float(signs[i,ti]);rawret=float(d['returns'][t,j]);markret=float(R[t,j]);physical=item['physical'];record_fee=i not in fee_recorded;fee_recorded.add(i)
   missing_log.append(dict(event=';'.join(sorted(set(item['events']))),date=str(d['dates'][t]),symbol=d['meta']['symbols'][j],series=c['ids'][i],position_before=int(old[i]),position_after=int(new[i]),physical_weight=physical,sealed_return_bps=rawret,valuation_return_bps=markret,recognized_stock_pnl_bps=physical*markret*mult,counterfactual_pnl_bps=np.nan,pnl_effect_interpretation='recognized valuation contribution; sealed daily P&L undefined where its return is absent; N2 residual sign applied',requested_action=item['requested'],requested_date=str(c['dates'][deferred_since[i]]) if deferred_since[i]>=0 else str(c['dates'][ti]),executed_action=('entry' if new[i] else 'exit') if new[i]!=old[i] else 'none',pending_exit=bool(pending_exit[i] and new[i]!=0),K2_fee_bps=float(cost*allocation_by_i.get(i,0.)) if record_fee else 0.,PP_FULL_fee_bps=float(cost*w[i]*abs(new[i]-old[i])) if record_fee else 0.,K1_fee_bps=float(cost*abs(own_delta[i])) if record_fee else 0.,position_fee_recorded_once=record_fee))
  for i in changed:pending_exit[i]=False;pending_forced[i]=False;pending_entry.pop(i,None);deferred_since[i]=-1
  old=new.copy()
 assert not np.any(old),'Unresolved untradeable terminal position: retain checkpoint, do not invent liquidation'
 series=[]
 for i,name in enumerate(c['ids']):
  if w[i]==0:continue
  # Per-series units: attribution is report-only. IPW then reproduces portfolio return.
  ss=dict(series=name,test_year=c['year'],weight=float(w[i]),gross_bps=float(raw[i,0]/w[i]),K2=float(raw[i,1]/w[i]),PP_FULL=float(raw[i,2]/w[i]),K1=float(raw[i,3]/w[i]),gross_exposure=float(raw[i,4]/T/w[i]),active_sessions=int(raw[i,5]),closed_trades=int(allcloses[i]),entries=int(entry_n[i].sum()),exits=int(allcloses[i]),holding_sessions=float(np.mean(closed[i])) if closed[i] else np.nan,forced_exits=int(forced[i]),terminal_exits=int(terminal[i]));series.append(ss)
  for side,op,nn in [('entry',entry_op,entry_n),('exit',exit_op,exit_n)]:
   for b in range(4):bins.append(dict(series=name,test_year=c['year'],side=side,bin=b,opportunities=int(op[i,b]),transitions=int(nn[i,b]),rate=float(nn[i,b]/op[i,b]) if op[i,b] else np.nan))
 return dict(series=series,daily=daily,bins=bins,perday=np.stack(perday) if details else None,positions=p,missing_price_log=missing_log)

def full_training_value(contexts,pol,gate,cost,perday=False):
 """Per-position cost, independent series; no training portfolio netting."""
 from kernel import positions
 values=[];daily=[]
 for c in contexts:
  z=scores(c,gate);p=positions(z,pol);n,T=p.shape;tot=np.zeros(n);day=np.zeros((n,T));d=panel();R=d['returns']
  for i in range(n):
   prev=0;v=None
   for t in range(T):
    pp=int(p[i,t]);fee=cost*abs(pp-prev)
    if pp and not prev:v=entry_vector(c,i,t)
    r=0.
    if pp and t+1<T:
     used=v!=0;rr=R[c['ts'][t+1],used];assert np.isfinite(rr).all(),('Unknown training held-leg return',c['kind'],c['year'],c['ids'][i],t)
     r=pp*float(v[used]@rr)
    day[i,t]=r-fee;prev=pp
  values.extend(day.sum(axis=1));daily.append(day)
 return (float(np.mean(values)),daily) if perday else float(np.mean(values))

def performance(gross,turn,c,exposure):
 gross=np.asarray(gross,float);turn=np.asarray(turn,float);net=gross-c*turn;assert np.isfinite(net).all();sd=net.std(ddof=1);e=float(np.mean(exposure));equity=np.cumprod(1+net/10000);peaks=np.maximum.accumulate(np.r_[1.,equity])[1:];sharpe=float(np.sqrt(252)*net.mean()/sd) if sd>0 else np.nan
 return dict(net_annual_bps=float(252*net.mean()),gross_annual_bps=float(252*gross.mean()),Sharpe=sharpe,employed_net_annual_bps=float(252*net.mean()/e) if e>0 else np.nan,employed_Sharpe=sharpe if e>0 else np.nan,average_gross_exposure=e,max_drawdown=float(np.max(1-equity/peaks)),breakeven_bps=float(gross.sum()/turn.sum()) if turn.sum()>0 else np.nan)
