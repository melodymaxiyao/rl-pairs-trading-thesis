"""Authorized valuation-only carry; original return inputs remain untouched."""
from common import *
import pandas as pd
_STATE=None
LOG_COLUMNS=['event','definition','test_year','gate','learner','cost','world','replicate','date','symbol','series','position_before','position_after','physical_weight','sealed_return_bps','valuation_return_bps','recognized_stock_pnl_bps','counterfactual_pnl_bps','pnl_effect_interpretation','requested_action','requested_date','executed_action','pending_exit','K2_fee_bps','PP_FULL_fee_bps','K1_fee_bps','position_fee_recorded_once']

def marked_prices(d):
 global _STATE
 if 'available' in d or 'price_coverage' not in d.get('meta',{}):return dict(returns=d.get('valuation_returns',d['returns']),available=d.get('available',np.isfinite(d['returns'])),affected=d.get('affected',np.zeros_like(d['returns'],bool)),records=[])
 if _STATE is not None:return _STATE
 dates=d['dates'];R=d['returns'];valid=np.zeros_like(R,dtype=bool);marks=np.array(R,copy=True);reg=read(source().verify('audit/input_registration.json'))['selected'];files={Path(rel).name[:-10].upper():(rel,row) for rel,row in reg.items() if rel.startswith('experiments/15_pair_trading/fresh_holdout_data_v1/data/raw/prices/')};symbols=d['meta']['symbols'];records=[];pins={}
 for j,symbol in enumerate(symbols):
  coverage=d['meta']['price_coverage'].get(symbol)
  if not coverage:continue
  first=np.datetime64(coverage['first_date']);last=np.datetime64(coverage['last_date']);inside=(dates>=first)&(dates<=last);valid[:,j]=inside;post=dates>first;anomaly=post&~np.isfinite(R[:,j]);marks[(dates>last),j]=0.
  if not np.any(anomaly&inside):continue
  rel,row=files[symbol];p=INPUT/rel;assert sha(p)==row['sha256'] and p.stat().st_size==int(row['size']);f=pd.read_csv(p,usecols=['Date','Adj Close']);f['Date']=pd.to_datetime(f.Date,utc=True).dt.tz_localize(None).dt.normalize();assert not f.Date.duplicated().any();close=pd.to_numeric(f.set_index('Date')['Adj Close'],errors='coerce').reindex(pd.to_datetime(dates)).to_numpy();close=np.where(np.isfinite(close)&(close>0),close,np.nan);valid[:,j]=np.isfinite(close);carried=pd.Series(close).ffill().to_numpy();rr=np.r_[np.nan,(carried[1:]/carried[:-1]-1)*10000]
  # Do not replace already observed returns with a different numerical path.
  np.testing.assert_array_equal(rr[np.isfinite(R[:,j])],R[np.isfinite(R[:,j]),j]);marks[anomaly,j]=rr[anomaly];pins[rel]=row['sha256']
  for t in np.flatnonzero(anomaly):
   records.append(dict(event='missing_close' if not valid[t,j] else 'quote_recovery',date=str(dates[t]),symbol=symbol,sealed_return_bps=None,valuation_return_bps=float(marks[t,j]),pnl_effect_interpretation='valuation convention only; stock contribution reported per actual position; sealed daily counterfactual undefined',physical_weight=0.,recognized_stock_pnl_bps=0.))
 potential=set(read(RUN/'audit/RF_potential_members.json')['symbols']);train=(dates>=np.datetime64('2001-01-01'))&(dates<np.datetime64('2025-01-01'));training_gaps=[]
 for j,symbol in enumerate(symbols):
  if symbol not in potential:continue
  cov=d['meta']['price_coverage'].get(symbol)
  if not cov:continue
  post=dates>np.datetime64(cov['first_date']);bad=train&post&~valid[:,j]
  if bad.any():training_gaps.extend((symbol,str(dates[t])) for t in np.flatnonzero(bad))
 assert not training_gaps,('Native unchanged training requires quote-complete observed histories',training_gaps)
 _STATE=dict(returns=marks,available=valid,affected=np.isfinite(marks)&~np.isfinite(R),records=records,source_pins=pins,training_quote_gaps=0)
 return _STATE

def causal_target(z,previous,pol,terminal=False):
 z=np.asarray(z,float);previous=np.asarray(previous,np.int32);valid=np.isfinite(z);flat=previous==0;st=np.searchsorted([.5,1.5,2.5],abs(z),side='right');v=np.zeros_like(z);np.multiply(-previous,z,out=v,where=previous!=0);holdbin=np.where(v>=1.5,0,np.where(v>=.5,1,np.where(v>=-.5,2,3)));st=np.where(flat,st,4+holdbin)
 kind=pol.get('kind',0)
 if kind==0:a=((int(pol['bits'])>>st)&1).astype(bool)
 elif kind==2:a=np.where(flat,abs(z)>pol.get('enter',1.25),abs(z)<pol.get('exit',.5))
 else:
  a=np.where(flat,abs(z)>pol.get('enter',1.25),v<pol.get('exit',.5))
  if kind==3:a=np.where((int(pol['bits'])>>(8+st))&1,((int(pol['bits'])>>st)&1).astype(bool),a)
 out=previous.copy();out[a&~flat]=0;enter=a&flat&valid;out[enter]=-np.sign(z[enter]).astype(np.int32);out[~valid]=0
 if terminal:out[:]=0
 return out

def initialise_evidence():
 from data import panel
 d=panel();v=marked_prices(d);write(RUN/'audit/missing_price_revision_v1/valuation_registration.json',dict(source_pins=v['source_pins'],signal_returns='unchanged sealed returns.npy',source_return_sha256=sha(source().verify('runtime/data/calendar_v2/returns.npy')),training_quote_gaps=0,training_years='2001-2024',rule='valuation-only carry; atomic deferred basket; signals use unfilled returns',price_files_modified=0))
 rows=pd.DataFrame(v['records']).reindex(columns=LOG_COLUMNS);publish(RUN/'audit/missing_price_revision_v1/stock_days.csv',rows.to_csv(index=False));publish(RUN/'missing_price_log.csv',rows.to_csv(index=False));return v
