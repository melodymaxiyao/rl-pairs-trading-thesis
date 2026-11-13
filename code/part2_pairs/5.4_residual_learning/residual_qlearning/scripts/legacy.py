"""Exact legacy daily-refresh |s|-exit replay fixture; never the main strategy."""
from common import *
from data import panel,M
K=load_module('_legacy_sealed_cost','rf_cost_accounting_v1','scripts/accounting.py')

def fixture():
 d=panel();R=d['returns'];N=R.shape[1];checks={}
 # Complete first RP fold: compare every gross, K0 and net daily bit for bit.
 f=[]
 for part in range(12):f+=source().cache(f'runtime/jobs/RP_2004_{part:02d}')['series']
 f.sort(key=lambda r:r['series']);n=len(f);pos=np.zeros(n,np.int8);oldstock=np.zeros(n);rows=[];previous=None
 ref={r['date']:r for r in source().cache('runtime/aggregates/RP_2004')['daily'] if r['world']=='real' and r['replicate']==0}
 oldhedge=np.zeros(n)
 for ti,t in enumerate(f[0]['indices']):
  s=np.array([v['score'][0,ti] for v in f]);h=np.array([v['half'][0,ti] for v in f]);b=np.array([v['beta'][ti] for v in f]);new=M.update_positions(pos,s,h,np.isfinite(b))
  if ti==len(f[0]['indices'])-1:new[:]=0
  count_active=int(np.count_nonzero(new));stock=np.zeros(n);hedge=np.zeros(n)
  if count_active:
   a=new!=0;stock[a]=new[a]/(1+abs(b[a]))*np.full(count_active,1/count_active);hedge[a]=-stock[a]*b[a]
  gross=0.
  if ti:
   held=oldstock!=0;one=np.array([v['alpha'][ti-1]+v['future'][0,ti-1,0] for v in f]);gross=float(oldstock[held]@one[held]) if np.isfinite(one[held]).all() else np.nan
  turn=float(abs(stock-oldstock).sum()+abs(hedge-oldhedge).sum());r=ref[str(d['dates'][t])]
  K.exact_arrays([gross,turn],[r['gross_bps'],r['turnover']],'RP2004 legacy')
  for c in [5,10,20]:K.exact_arrays([gross-c*turn],[r['gross_bps']-c*r['turnover']],f'RP net{c}')
  pos=new;oldstock=stock;oldhedge=hedge
 checks['RP_daily_bit_exact']=len(ref)
 # First complete RF month (source starts trading in2004); no context rebuild.
 ref={r['date']:r for r in source().cache('runtime/aggregates/RF')['daily'] if r['world']=='real' and r['replicate']==0};pos=np.zeros(N,np.int8);stock=np.zeros(N);oldH=np.zeros((N,N));tested=0
 for day in source().cache('runtime/jobs/RF_2004_01')['days']:
  t=day['index'];idx=day['indices'];s=np.full(N,np.nan);h=s.copy();s[idx]=day['score'][0];h[idx]=day['half'][0];eligible=np.zeros(N,bool);eligible[idx]=day['status']=='ok';new=M.update_positions(pos,s,h,eligible);n=int(np.count_nonzero(new));ns=np.zeros(N);H=np.zeros_like(oldH)
  if n:
   weight=new[idx]/day['gross']/n;ns[idx]=weight;H[idx]=-(weight[:,None]*day['beta'])@day['weights'].T
  used=(stock!=0)|np.any(oldH!=0,axis=0);gross=float(stock[used]@R[t,used]+oldH[:,used].sum(axis=0)@R[t,used]) if np.isfinite(R[t,used]).all() else np.nan;turn=float(abs(ns-stock).sum()+abs(H-oldH).sum());r=ref[str(d['dates'][t])]
  K.exact_arrays([gross,turn],[r['gross_bps'],r['turnover']],'RF Jan2004 legacy')
  for c in [5,10,20]:K.exact_arrays([gross-c*turn],[r['gross_bps']-c*r['turnover']],f'RF net{c}')
  pos=new;stock=ns;oldH=H;tested+=1
 checks['RF_daily_bit_exact']=tested;write(RUN/'audit/legacy_AL_fixture.json',dict(status='PASS_BIT_EXACT',real_strategy_fits=0,mode='sealed abs exit and daily active1/n reallocation only',**checks));return checks
