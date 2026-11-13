"""Infeasible ex-post associations, never imported by screen selection."""
from common import *
import numpy as np,pandas as pd
from inference import mean
from report import compact,deciles,weighted_rank_corr
from outcomes import outcome_matrix

def run(train,outcomes):
 f=pd.concat([cached(f'runtime/hindsight/{y}')['frame'] for y in YEARS],ignore_index=True);csv_new(TABLES/'hindsight_stats.csv.gz',f)
 keys=['pool','test_year','pair'];o=outcome_matrix(outcomes);real=f[f.world.eq('real')&f.statistic_name.eq('theta_21')].copy();real['decile']=np.nan;out=[]
 for _,g in real.groupby('test_year'):real.loc[g.index,'decile']=deciles(g.theta,g.pair)
 a=o[o.cell.eq('TD8_tuned')&o.pool.eq('P_RAND')];b=o[o.cell.eq('R_TR_aln')&o.pool.eq('P_RAND')]
 vals={}
 for cell,g in [('TD8_tuned',a),('R_TR_aln',b)]:
  v=real[keys+['weight','decile']].merge(g[keys+['net_bps']],on=keys,how='left',validate='one_to_one');v['value']=v.net_bps;vals[cell]=v
 v=vals['TD8_tuned'].merge(vals['R_TR_aln'][keys+['value']],on=keys,validate='one_to_one',suffixes=('','_l0'));v['value']=v.value-v.value_l0;vals['TD8_tuned_minus_L0']=v
 for cell,v in vals.items():
  for decile,g in v.groupby('decile'):out.append(dict(kind='theta21_decile_vs_real_net',label='hindsight, not a screen',cell=cell,decile=int(decile),**mean(g)))
 t=train[train.pool.eq('P_RAND')&train.window.eq('T')][keys+['world','replicate','theta_21__theta','weight']]
 e=f[f.statistic_name.eq('theta_21')][keys+['world','replicate','theta']];z=t.merge(e,on=keys+['world','replicate'],validate='one_to_one');corr=[]
 for (y,world,rep),g in z.groupby(['test_year','world','replicate']):corr.append(dict(test_year=y,world=world,replicate=rep,weight=g.weight.sum(),value=weighted_rank_corr(g.theta_21__theta.to_numpy(),g.theta.to_numpy(),g.weight.to_numpy())))
 c=pd.DataFrame(corr)
 for world,g in c.groupby('world'):out.append(dict(kind='train_vs_test_theta_within_fold_Spearman',label='hindsight, not a screen',world=world,**mean(g)))
 n=c[c.world.eq('N2')].groupby('test_year').value.mean();r=c[c.world.eq('real')].copy();r['value']=r.value-r.test_year.map(n);out.append(dict(kind='train_test_correlation_real_minus_N2',label='hindsight, not a screen',world='real-N2',**mean(r)))
 return pd.DataFrame(out)
