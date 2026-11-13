"""Copy sealed outcomes with provenance; strict one-to-one joins and no trading."""
from common import *
import numpy as np,pandas as pd
KEYS=['pool','test_year','pair','world','replicate','cell']
def read_outcomes():
 stem='runtime/copied_outcomes';old=cached(stem)
 if old is not None:return old
 out=[]
 for base,cells in [(SOURCE,['R_TR_aln','B3_TR_aln','TD8','H1s']),(TUNED,['TD8_tuned','TD8_tuned_z'])]:
  path=note(base/'outputs/tables/per_pair_fold_test.csv.gz');digest=sha(path);offset=0
  for chunk in pd.read_csv(path,chunksize=100000,float_precision='round_trip',keep_default_na=False):
   selected=chunk.cost.eq(10)&chunk.variant.eq('primary')&chunk.world.isin(['real','N1','N2'])&chunk.cell.isin(cells)
   f=chunk.loc[selected,KEYS+['weight','available','net_bps','source_file','source_row_key']].copy();f['source_csv']=str(path.relative_to(ROOT));f['source_sha256']=digest;f['source_csv_row']=np.flatnonzero(selected)+offset+2
   f['available']=f.available.astype(str).str.lower().eq('true');f['net_bps']=pd.to_numeric(f.net_bps,errors='coerce');out.append(f);offset+=len(chunk)
 f=pd.concat(out,ignore_index=True);assert not f.duplicated(KEYS).any(),'Duplicate sealed outcome keys'
 for (pool,y,cell,world,rep),g in f.groupby(['pool','test_year','cell','world','replicate']):
  from sources import members
  expected={r['pair']:r['weight'] for r in members(y)[pool]};assert set(g.pair)==set(expected),('Outcome membership mismatch',pool,y,cell,world,rep)
  np.testing.assert_array_equal(g.weight.to_numpy(),g.pair.map(expected).to_numpy())
 return save(stem,f.sort_values(KEYS,kind='stable').reset_index(drop=True))
def outcome_matrix(f,null_world='N2'):
 """Keep pair-year missingness explicit; require both N2 replicates."""
 keys=['pool','test_year','pair','cell'];real=f[f.world.eq('real')].copy();null=f[f.world.eq(null_world)].copy()
 real.loc[~real.available,'net_bps']=np.nan;null.loc[~null.available,'net_bps']=np.nan
 n=null.groupby(keys,sort=True).net_bps.agg(['mean','count']).reset_index().rename(columns={'mean':'N2_net','count':'N2_count'})
 z=real.merge(n,on=keys,how='left',validate='one_to_one');assert len(z)==len(real)
 z.loc[z.N2_count.ne(2),'N2_net']=np.nan;z['excess_net']=z.net_bps-z.N2_net;z['null_world']=null_world
 z['null_net']=z.N2_net;z['null_count']=z.N2_count
 return z
