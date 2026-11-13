"""Synthetic statistics only, using the sealed partial-cointegration path body."""
from common import *
import numpy as np,pandas as pd
from sources import setup,synthetic_module
from statistics import adf,kpss,predict
from inference import empirical
NAMES=['ADF_c','ADF_ct','KPSS_c','KPSS_ct','theta_21']
def stats(S):
 m,s=setup().anchor.rolling(S,252,True);d=(S-m)/s;ix=slice(252,1008);x=S[ix];z=d[ix]
 return dict(ADF_c=adf(x,'c'),ADF_ct=adf(x,'ct'),KPSS_c=kpss(x,'c'),KPSS_ct=kpss(x,'ct'),theta_21=predict(x,z,21))
def chunk(job):
 k,w,offset=job;stem=f'runtime/power_chunks/{k}/{offset:05d}';old=cached(stem)
 if old is not None:return dict(k=k,offset=offset)
 calibration=read(note(SYN/'outputs/tables/calibration.json'));h=calibration['chosen']['h'];sigma=calibration['targets']['sigma_bps'];sim=synthetic_module();out=[]
 for i in range(offset,min(20000,offset+100)):
  S,r=sim.path(h,w,sigma,['residual_learnability_power',k],i)
  for name,value in stats(S).items():out.append(dict(k=k,w=w,path=i,world='synthetic',replicate=0,statistic_name=name,**value))
  if spec()['decisions']['power_reference']=='matched_synthetic_N2':
   for rep in [0,1]:
    rng=np.random.default_rng(stable_seed('power_N2',str(k),i,rep));sign=rng.choice(np.array([-1.,1.]),len(S)-1);null=np.r_[S[0],S[0]+np.cumsum(np.diff(S)*sign)];count('null_replicates');count('synthetic_null_paths')
    for name,value in stats(null).items():out.append(dict(k=k,w=w,path=i,world='N2',replicate=rep,statistic_name=name,**value))
 save(stem,pd.DataFrame(out));return dict(k=k,offset=offset)
def grid():
 w=read(note(SYN/'outputs/tables/calibration.json'))['chosen']['w'];out=[(0.,0.),(.5,.5*w),(1.,w),(2.,2*w)]
 if spec()['decisions']['power_k4']=='cap_095':out.append((4.,min(4*w,.95)))
 out.append(('pure_AR',1.));return out

def report(reference=None):
 out=[]
 for k,w in grid():
  f=pd.concat([cached(f'runtime/power_chunks/{k}/{offset:05d}') for offset in range(0,20000,100)],ignore_index=True)
  for name,g in f.groupby('statistic_name'):
   real=g[g.world.eq('synthetic')];tail='upper' if name=='theta_21' else 'lower'
   null=g[g.world.eq('N2')].statistic.to_numpy() if spec()['decisions']['power_reference']=='matched_synthetic_N2' else reference[name]
   p=empirical(real.statistic.to_numpy(),null,tail)
   nominal=(real.statistic>real.critical_5).to_numpy() if name.startswith('KPSS') else real.nominal_p.le(.05).to_numpy()
   for kind,reject in [('nominal_5',nominal),('N2_calibrated_structure_5',p<=.05)]:
    rate=reject.mean();out.append(dict(k=k,w=w,statistic=name,calibration=kind,rejection_rate=rate,mcse=np.sqrt(rate*(1-rate)/len(reject)),paths=len(reject),N2_reference_paths=len(null),null_hypothesis='stationarity (upper-tail rejection)' if kind=='nominal_5' and name.startswith('KPSS') else 'random-walk structure test',strictly_stationary=w==1,reference=spec()['decisions']['power_reference']))
 return pd.DataFrame(out)
