"""Descriptive sealed fixed-beta comparison on the unchanged pair-fold input set."""
from common import *
import pandas as pd
from inputs import panel
from stats import weighted_quantile

def comparison(calibration):
 p=INPUT/'experiments/15_pair_trading/residual_learnability_v1/outputs/tables/residual_stats.csv.gz';data=panel();wanted={(int(y),r['pair']) for y,rows in data['meta']['pairs'].items() for r in rows};frames=[]
 columns=['pool','pair','test_year','world','replicate','window','statistic_name','half_life','weight']
 for f in pd.read_csv(p,usecols=columns,chunksize=200000,float_precision='round_trip'):
  z=f[(f.pool=='P_RAND')&(f.window=='T')&f.world.isin(['real','N2'])&f.statistic_name.str.upper().str.startswith('ECM')].copy()
  if len(z):
   keep=[(int(y),pair) in wanted for y,pair in zip(z.test_year,z.pair)];frames.append(z.loc[keep])
  checkpoint_guard()
 if not frames:raise RuntimeError('Sealed fixed-beta ECM/T baseline rows unavailable; inspect source schema without substituting another statistic')
 old=pd.concat(frames,ignore_index=True);assert not old.duplicated(['pair','test_year','world','replicate']).any()
 rows=[]
 for (world,rep),g in old.groupby(['world','replicate'],sort=True):
  hl=pd.to_numeric(g.half_life,errors='coerce');valid=hl.notna();rows.append(dict(residual='sealed fixed-beta A_TR',world=world,replicate=rep,pair_folds=len(g),expected_pair_folds=66000,missing_pair_folds=66000-len(g),median_half_life_sessions=weighted_quantile(hl,g.weight,.5),half_life_under30_share=float(np.average((hl[valid]>0)&(hl[valid]<30),weights=g.weight[valid])) if valid.any() else np.nan,note='T=three train years; old ECM model and shared-leg N2 seed20261026 differ from rolling60 residual and independent-residual N2; descriptive comparison, not a paired causal effect'))
 for (world,rep),g in calibration[calibration.definition=='RP'].groupby(['world','replicate'],sort=True):rows.append(dict(residual='new rolling60 RP',world=world,replicate=rep,pair_folds=len(g),expected_pair_folds=66000,missing_pair_folds=66000-len(g),median_half_life_sessions=weighted_quantile(g.half_life,g.weight,.5),note='per-pair-fold median valid reverting test-day OU half-life; unavailable/nonreverting windows counted separately'))
 return pd.DataFrame(rows)
