"""Stream retained sealed CSV outputs; no source context/cell pickle access."""
from common import *
import pandas as pd,numpy as np
PAIR_COLUMNS=['pool','cell','cost','world','replicate','variant','pair','leg_A','leg_B','test_year','weight','available','net_bps','gross_bps','n_trades','relation','era','drift_bps','timing_bps','other_bps']
DAILY_COLUMNS=['pool','cell','cost','world','replicate','variant','date','test_year','net_bps','gross_bps','cost_bps','turnover']
def pairs():
 stem='runtime/references/pairs';old=cached(stem)
 if old is not None:return old
 frames=[]
 for directory,cells in [(SOURCE,['R_TR_aln','B3_TR_aln','TD8']),(TUNED,['TD8_tuned'])]:
  path=note(directory/'outputs/tables/per_pair_fold_test.csv.gz')
  cols=pd.read_csv(path,nrows=0).columns;wanted=[c for c in PAIR_COLUMNS if c in cols]
  for i,f in enumerate(pd.read_csv(path,usecols=wanted,chunksize=200000,float_precision='round_trip')):
   selected=f.loc[f.variant.eq('primary')&f.cell.isin(cells)&f.pool.isin(POOLS)&f.cost.isin(COSTS)].copy()
   if len(selected):frames.append(selected)
   if i%10==0:print('Reference pair rows scanned',directory.name,(i+1)*200000,flush=True)
   checkpoint_guard()
 result=pd.concat(frames,ignore_index=True);keys=['pool','cell','cost','world','replicate','pair','test_year']
 assert not result.duplicated(keys).any();assert set(result.cell)=={'R_TR_aln','B3_TR_aln','TD8','TD8_tuned'}
 return save(stem,result)
def daily():
 stem='runtime/references/daily';old=cached(stem)
 if old is not None:return old
 old_account=source_pickle_at(TUNED/'runtime/accounting/CSCV_GLOBAL.pkl.gz');wanted=set(old_account['cells']);frames=[]
 for directory in [SOURCE,TUNED]:
  path=note(directory/'outputs/tables/portfolio_daily.csv.gz')
  columns=pd.read_csv(path,nrows=0).columns;cols=[c for c in DAILY_COLUMNS if c in columns]
  for f in pd.read_csv(path,usecols=cols,chunksize=200000,float_precision='round_trip'):
   f=f.loc[f.world.eq('real')&f.pool.isin(POOLS)].copy();f['name']=f['pool']+'|'+f.cell+'|'+f.cost.astype(int).astype(str)+'|'+f.variant
   keep=f.name.isin(wanted)|(f.cell.eq('SPY')&f.variant.eq('primary')&f.cost.eq(10))
   if keep.any():frames.append(f.loc[keep])
   checkpoint_guard()
 result=pd.concat(frames,ignore_index=True);assert not result.duplicated(['pool','cell','cost','variant','date']).any()
 return save(stem,result)
def baseline_accounting():return source_pickle_at(TUNED/'runtime/accounting/CSCV_GLOBAL.pkl.gz')
