"""Small synthetic fixtures; expensive sealed real reproduction is main-only."""
from common import *
import numpy as np,pandas as pd
def tiny(pool='P_EXP',year=2004):
 rng=np.random.default_rng(stable_seed('tiny',pool,year));es=[]
 for j in range(3):
  for y in range(year-3,year):
   z=np.array([0.,1.,2.1,2.6,2.,1.,-.2,-1.,-2.1,-2.8,-1.,0.])
   es.append(dict(pair=f'{pool}_{j}',year=y,window='train',fold=source_fold(year),test_year=year,construction='F',dates=pd.bdate_range(str(y)+'-01-02',periods=len(z)),z=z.copy(),drift=np.full(len(z),-1.),strong=np.zeros(len(z),np.int32),forward_bps=rng.normal(0,5,len(z)),bins=np.searchsorted([-2,-1.5,-1,-.5,.5,1,1.5,2],z),beta=np.ones(len(z))))
 return es
