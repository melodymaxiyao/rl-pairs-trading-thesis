"""Exact original AL/Band/Q_best execution before any new scientific work."""
import numpy as np
from io_utils import *
from groups import *
def job(kind,year):
 root=ROOT/f'runtime/reproduction/{kind}_{year}'
 if done(root):return dict(root=str(root),cached=True)
 import bootstrap
 c,d,a=bootstrap.core_modules();ctx=d.rp_context(year,d.panel()['meta']['pairs'][str(year)],verify_stored=True) if kind=='RP' else d.rf_context(year);rows=[]
 for learner in ['AL','Band','Q_best']:
  rel=f'outputs/cells/{kind}_{year}_G0_c5_{learner}/complete.json';mf=J(CORE/'tier_3/manifest.json')['files'];assert H(CORE/rel)==mf[rel];pol=J(CORE/rel)['policy']
  reference=sealed_frame(kind,year,learner,'portfolio_daily.csv.gz')
  for wi in range(3):
   checkpoint();value=a.execution(ctx,pol,'G0',wi,details=False,cost=5);f=reference[(reference.world==('real' if wi==0 else 'N2'))&(reference.replicate==(0 if wi==0 else wi-1))].sort_values('date');actual=value['daily']
   for key in ['gross_bps','K2','PP_FULL','K1','gross_exposure','net_stock_gross','positions','entries','exits']:np.testing.assert_array_equal([r[key] for r in actual],f[key].to_numpy(),err_msg=f'{kind}/{year}/{learner}/{wi}/{key}')
   np.testing.assert_array_equal([r['gross_bps']-5*r['K2'] for r in actual],f.net_bps.to_numpy());rows.append(dict(learner=learner,world=wi,status='EXACT_DAILY_ALL_FIELDS_PASS'))
 return finish(root,{'reproduction.json':rows},phase='R',kind=kind,year=year)
