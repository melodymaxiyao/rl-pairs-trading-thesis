"""Pure accounting kernels. No signal, risk, or portfolio-scaling choices."""
import numpy as np

def canonical_order(ids):
 assert len(set(ids))==len(ids),'Unique position identities required'
 return np.argsort(np.asarray(ids,dtype=str),kind='stable')
def l1(x):return float(np.abs(x).sum())
def factors_to_stock(stock,beta,W):return -(stock[:,None]*beta)@W.T

def fixture_costs(ids,stock,prior_stock,hedges,prior_hedges,own_indices,factor_baskets=None,prior_factor_baskets=None):
 order=canonical_order(ids);s=np.asarray(stock)[order];old=np.asarray(prior_stock)[order];H=np.asarray(hedges)[order];previous=np.asarray(prior_hedges)[order];idx=np.asarray(own_indices)[order]
 n=H.shape[1];netstock=np.bincount(idx,weights=s,minlength=n);oldstock=np.bincount(idx,weights=old,minlength=n)
 out=dict(K0=l1(s-old)+l1(H-previous),K1=l1(s-old),K2=l1(netstock+H.sum(axis=0)-oldstock-previous.sum(axis=0)))
 if factor_baskets is not None:out['K3']=l1(s-old)+l1(np.asarray(factor_baskets)-np.asarray(prior_factor_baskets))
 return out

def rf_account(stock,previous_stock,H,previous_H,baskets,previous_baskets):
 leg=l1(stock-previous_stock);hedge=l1(H-previous_H);net_hedge=H.sum(axis=0);previous_net=previous_H.sum(axis=0)
 factor_turn=l1(baskets-previous_baskets)
 return dict(K0=leg+hedge,K1=leg,K2=l1((stock+net_hedge)-(previous_stock+previous_net)),K3=leg+factor_turn,stock_leg_gross_turnover=leg,stock_leg_netted_turnover=leg,hedge_gross_turnover=hedge,hedge_netted_turnover=l1(net_hedge-previous_net),factor_basket_turnover=factor_turn,unnetted_gross_exposure=l1(stock)+l1(H),stock_leg_gross_exposure=l1(stock),hedge_gross_exposure=l1(H),stock_leg_netted_exposure=l1(stock),hedge_netted_exposure=l1(net_hedge),portfolio_netted_gross_exposure=l1(stock+net_hedge))

def rp_account(stock,previous_stock,hedge,previous_hedge,ia,ib,N):
 leg=l1(stock-previous_stock);h=l1(hedge-previous_hedge)
 # Canonical pair order is fixed by sealed feature.series before bincount.
 ok=ia>=0;stock_net=np.bincount(ia[ok],weights=stock[ok],minlength=N);oldstock_net=np.bincount(ia[ok],weights=previous_stock[ok],minlength=N)
 ok=ib>=0;hedge_net=np.bincount(ib[ok],weights=hedge[ok],minlength=N);oldhedge_net=np.bincount(ib[ok],weights=previous_hedge[ok],minlength=N)
 return dict(K0=leg+h,K1=leg,K2=l1((stock_net+hedge_net)-(oldstock_net+oldhedge_net)),stock_leg_gross_turnover=leg,stock_leg_netted_turnover=l1(stock_net-oldstock_net),hedge_gross_turnover=h,hedge_netted_turnover=l1(hedge_net-oldhedge_net),factor_basket_turnover=np.nan,unnetted_gross_exposure=l1(stock)+l1(hedge),stock_leg_gross_exposure=l1(stock),hedge_gross_exposure=l1(hedge),stock_leg_netted_exposure=l1(stock_net),hedge_netted_exposure=l1(hedge_net),portfolio_netted_gross_exposure=l1(stock_net+hedge_net))

def exact_arrays(actual,expected,label):
 a=np.asarray(actual);b=np.asarray(expected)
 assert a.shape==b.shape and np.array_equal(a,b,equal_nan=True),('K0 exact reproduction failed',label,a.shape,b.shape,float(np.nanmax(abs(a-b))) if a.shape==b.shape and np.issubdtype(a.dtype,np.number) else '')

def performance(gross,turnover,cost):
 gross=np.asarray(gross,float);T=np.asarray(turnover,float);net=gross-cost*T;ok=np.isfinite(net);v=net[ok]
 mean=float(v.mean()) if len(v) else np.nan;sd=float(v.std(ddof=1)) if len(v)>1 else np.nan
 path=np.r_[0.,np.cumsum(v)];add_dd=float(np.min(path-np.maximum.accumulate(path))) if len(v) else np.nan
 assert np.all(v>-10000),'Return below-100% requires explicit treatment'
 nav=np.r_[1.,np.cumprod(1+v/10000)];dd=float(np.min(nav/np.maximum.accumulate(nav)-1)) if len(v) else np.nan
 mean_T=float(T[ok].mean()) if ok.any() else np.nan;mean_gross=float(gross[ok].mean()) if ok.any() else np.nan
 break_even=mean_gross/mean_T if mean_T>0 else np.nan
 return dict(net_annual_bps=252*mean,gross_annual_bps=252*mean_gross,net_Sharpe=np.sqrt(252)*mean/sd if sd>0 else np.nan,CAGR=float(nav[-1]**(252/len(v))-1) if len(v) else np.nan,max_drawdown_pct=100*dd,max_additive_drawdown_bps=add_dd,breakeven_bps_per_side=break_even,has_nonnegative_profitable_cost=bool(mean_gross>0),annual_turnover_valid=252*mean_T,annual_turnover_all=252*float(np.mean(T)),days=int(ok.sum()),missing_days=int((~ok).sum()))
