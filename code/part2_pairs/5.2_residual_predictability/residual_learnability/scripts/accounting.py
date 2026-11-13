"""DSR recomputation from sealed sufficient statistics; no new return series."""
from common import *
import pandas as pd,numpy as np
from sources import setup

def run():
 p=note(TUNED/'outputs/tables/dsr_pbo.csv');f=pd.read_csv(p,float_precision='round_trip');f=f[f.kind.eq('DSR')&f.N.eq(544)];assert len(f)==4
 dsr=setup().sources.original_inference().dsr;out=[]
 for r in f.to_dict('records'):
  args=[r['SR'],int(r['T']),r['skewness'],r['kurtosis'],r['cross_trial_SR_variance']]
  old=dsr(*args,544);assert abs(old['DSR']-r['DSR'])<1e-12,'Sealed DSR sufficient-statistic reproduction failed';count('DSR_evaluations')
  new=dsr(*args,546);count('DSR_evaluations');out.append(dict(pool=r['pool'],best_cell_for_accounting_only=r['best_cell_for_accounting_only'],previous_trials=544,new_screen_cell_trials=2,source=str(p.relative_to(ROOT)),source_sha256=sha(p),interpretation='trial-count adjustment of sealed best-cell DSR; no screened portfolio backtest',**new))
 return pd.DataFrame(out)
