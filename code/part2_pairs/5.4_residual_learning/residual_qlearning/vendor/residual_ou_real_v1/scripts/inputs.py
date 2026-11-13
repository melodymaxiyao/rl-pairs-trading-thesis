"""Verified sealed prices, PIT monthly membership and unchanged P_RAND/IPW rows."""
from common import *
import pandas as pd
from functools import lru_cache
PT=INPUT/'experiments/15_pair_trading';FRESH=PT/'fresh_holdout_data_v1';SOURCE=PT/'trend_anchor_trading_v1'
DATA_STEM='runtime/data/calendar_v2/panel'
DATA_DIR='runtime/data/calendar_v2'

def validate_calendar(calendar,years):
 dates=np.asarray(calendar,dtype='datetime64[D]');assert dates.ndim==1 and len(dates)>0 and np.all(dates[1:]>dates[:-1]), 'Calendar must be nonempty, unique and ordered'
 labels=dates.astype('datetime64[Y]').astype(int)+1970;counts={int(y):int(np.sum(labels==y)) for y in years};missing=[y for y,n in counts.items() if n==0]
 if missing:raise ValueError('Calendar does not cover required test years: '+str(missing)+'; no residual fitting started')
 return counts

def calendar_and_benchmark(files,registration):
 def date_values(path):
  f=pd.read_csv(path,usecols=['Date']);return pd.DatetimeIndex(pd.to_datetime(f.Date,utc=True).dt.tz_localize(None).dt.normalize()).to_numpy(dtype='datetime64[D]')
 plan=registration.get('calendar_and_benchmark')
 if plan:
  warm=date_values(files['SPY']);pieces=[warm[warm<np.datetime64(str(min(spec()['years']))+'-01-01')]];blocks=[]
  for item in plan['ledgers']:
   path=INPUT/item['path'];assert sha(path)==item['sha256']
   with np.load(path,allow_pickle=False) as z:dates=z['dates'].astype('datetime64[D]');forward=z['forward_SPY'].copy()
   assert len(dates)==len(forward)>0 and np.all(dates[1:]>dates[:-1]) and np.all(dates.astype('datetime64[Y]').astype(int)+1970==item['year']) and forward[-1]==0
   pieces.append(dates);blocks.append((dates,forward))
  dates=np.unique(np.concatenate(pieces));benchmark=np.full(len(dates),np.nan)
  for d,fwd in blocks:
   # The producer's fwd[t] belongs to the next realized close. Its terminal
   # year-end zero is a boundary sentinel, not a measured next-day return.
   benchmark[np.searchsorted(dates,d[1:])]=fwd[:-1]
  source=plan['calendar']
 else:
  # Fixtures and standalone panels use the complete sealed date universe,
  # never the length of a possibly truncated benchmark ticker.
  dates=np.unique(np.concatenate([date_values(path) for path in files.values()]));benchmark=None;source='union of sealed price-file dates'
 dates=dates[(dates>=np.datetime64('2000-01-01'))&(dates<=np.datetime64('2025-12-31'))];validate_calendar(dates,spec()['years'])
 return pd.DatetimeIndex(dates),benchmark,source

@lru_cache(1)
def pool(year):
 p=SOURCE/f'runtime/pools/{year}.pkl.gz';reg=json.loads((RUN/'audit/input_registration.json').read_text())['selected'];r=reg[str(p.relative_to(INPUT))];assert sha(p)==r['sha256']
 with gzip.open(p,'rb') as f:value=pickle.load(f)
 rows=value['P_RAND'];assert len(rows)==3000 and len({r['pair'] for r in rows})==3000
 assert all(float(r['ipw'])>0 and int(r['test_year'])==year for r in rows)
 return rows

def build_panel():
 old=cached(DATA_STEM)
 if old is not None:
  for rel,digest in old['files'].items():assert sha(RUN/rel)==digest
  assert old.get('panel_version')==2;validate_calendar(np.load(RUN/(DATA_DIR+'/dates.npy'),mmap_mode='r'),spec()['years'])
  return old
 reg=json.loads((RUN/'audit/input_registration.json').read_text())['selected'];prefix='experiments/15_pair_trading/fresh_holdout_data_v1/data/raw/prices/'
 files={Path(k).name[:-10].upper():INPUT/k for k in sorted(reg) if k.startswith(prefix)}
 assert 'SPY' in files
 membership=pd.read_csv(FRESH/'membership_monthly.csv',dtype=str).fillna('');membership['month_end']=pd.to_datetime(membership.month_end)
 calendar,benchmark,calendar_source=calendar_and_benchmark(files,json.loads((RUN/'audit/input_registration.json').read_text()));symbols=sorted(set(files)|{s.upper() for s in membership.yahoo_symbol if s});prices=np.full((len(calendar),len(symbols)),np.nan);price_coverage={}
 for j,symbol in enumerate(symbols):
  if symbol not in files:continue
  f=pd.read_csv(files[symbol],usecols=['Date','Adj Close']);f['Date']=pd.to_datetime(f.Date,utc=True).dt.tz_localize(None).dt.normalize();assert not f.Date.duplicated().any();v=pd.to_numeric(f.set_index('Date')['Adj Close'],errors='coerce').reindex(calendar).to_numpy();prices[:,j]=np.where(np.isfinite(v)&(v>0),v,np.nan)
  valid=np.flatnonzero(np.isfinite(prices[:,j]));price_coverage[symbol]=dict(first_date=str(calendar[valid[0]].date()),last_date=str(calendar[valid[-1]].date()),observed_sessions=len(valid)) if len(valid) else None
  if j%50==0:checkpoint_guard()
 returns=np.full_like(prices,np.nan);returns[1:]=(prices[1:]/prices[:-1]-1)*10000
 if benchmark is None:benchmark=returns[:,symbols.index('SPY')].copy()
 by_month={d:sorted(s for s in set(g.yahoo_symbol.str.upper()) if s) for d,g in membership.groupby('month_end',sort=True)};months=np.array(sorted(by_month),dtype='datetime64[D]');mask=np.zeros_like(prices,dtype=bool);indices={s:i for i,s in enumerate(symbols)};aliases={}
 for row in membership[['ticker','yahoo_symbol']].drop_duplicates().itertuples(index=False):
  symbol=row.yahoo_symbol.upper()
  if row.ticker not in aliases:aliases[row.ticker]=symbol
  elif aliases[row.ticker]!=symbol:aliases[row.ticker]=None
 for i,date in enumerate(calendar):
  at=np.searchsorted(months,np.datetime64(date,'D'),side='right')-1
  if at>=0:
   for symbol in by_month[pd.Timestamp(months[at])]:
    if symbol in indices:mask[i,indices[symbol]]=True
 classification=pd.read_csv(FRESH/'classification.csv',dtype=str).fillna('');sectors={r.yahoo_symbol.upper():r.sector for r in classification.itertuples(index=False)}
 import io
 written={}
 for name,array in [('returns',returns),('membership',mask),('dates',calendar.to_numpy(dtype='datetime64[D]')),('benchmark',benchmark)]:
  buffer=io.BytesIO();np.save(buffer,array,allow_pickle=False);rel=DATA_DIR+f'/{name}.npy';publish(RUN/rel,buffer.getvalue());written[rel]=sha(RUN/rel)
 all_pairs={}
 for year in spec()['years']:
  rows=pool(year);all_pairs[str(year)]=[dict(pair=r['pair'],leg_A=r['leg_A'],leg_B=r['leg_B'],ipw=float(r['ipw']),test_year=year) for r in rows]
 assert sum(map(len,all_pairs.values()))==66000
 result=dict(panel_version=2,calendar_source=calendar_source,calendar_year_counts=validate_calendar(calendar.to_numpy(dtype='datetime64[D]'),spec()['years']),price_coverage=price_coverage,benchmark_alignment='sealed TAV forward_SPY bps shifted to next realized close; first session each year is unavailable',symbols=symbols,aliases=aliases,sectors=sectors,pairs=all_pairs,files=written,rows=len(calendar),stocks=len(symbols),pair_folds=66000,price_units='bps simple return',membership_convention=spec()['membership'])
 save(DATA_STEM,result);count('input_panels_built');return result
@lru_cache(1)
def panel():
 value=cached(DATA_STEM);assert value is not None, 'Validated calendar_v2 panel required; resume through RUN_ALL.zsh to build it'
 assert value.get('panel_version')==2
 return dict(meta=value,benchmark=np.load(RUN/(DATA_DIR+'/benchmark.npy'),mmap_mode='r',allow_pickle=False),returns=np.load(RUN/(DATA_DIR+'/returns.npy'),mmap_mode='r',allow_pickle=False),membership=np.load(RUN/(DATA_DIR+'/membership.npy'),mmap_mode='r',allow_pickle=False),dates=np.load(RUN/(DATA_DIR+'/dates.npy'),mmap_mode='r',allow_pickle=False),indices={s:i for i,s in enumerate(value['symbols'])})
def index(symbol,data):
 symbol=str(symbol).upper();symbol=data['meta']['aliases'].get(symbol,symbol)
 return data['indices'].get(symbol) if symbol else None
SECTOR_ETF={'Materials':'XLB','Energy':'XLE','Financials':'XLF','Industrials':'XLI','Information Technology':'XLK','Consumer Staples':'XLP','Real Estate':'XLRE','Utilities':'XLU','Health Care':'XLV','Consumer Discretionary':'XLY','Communication Services':'XLC'}
