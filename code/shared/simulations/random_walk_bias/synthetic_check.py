"""Only authorized new simulation: Gaussian unit roots; no project imports/data."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ.update(TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache'),PYTHONDONTWRITEBYTECODE='1')
import numpy as np, math,csv,json
assert not str(ROOT.resolve()).startswith(('/Users/maxiyao/Desktop','/Users/maxiyao/Documents'))
seed=20261105; rng=np.random.default_rng(seed); N=100000
def half(b):return -math.log(2)/math.log(b) if 0<b<1 else None
rows=[]
def add(id,location,quantity,value,method):rows.append(dict(id=id,location=location,quantity=quantity,value=value,method=method,source=f'{__file__}:1'))
add('D01','3.2.2 after (3.12)','(1+3*1)/60',4/60,'analytic')
add('D02','3.2.2 after (3.12)','half-life at b=1-0.067',half(1-.067),'analytic')
add('D03','3.2.2 after (3.12)','half-life at b=1-4/60',half(1-4/60),'analytic')
add('D04','3.2.3 after (3.15)','60*ln(2)/20',60*math.log(2)/20,'analytic approximation')
add('D05','3.2.3 after (3.15)','60*(1-2**(-1/20))',60*(1-2**(-1/20)),'analytic exact')
add('D06','5.1','(8.4-8.2)/8.4',(.2)/8.4,'analytic; source numbers rounded')
add('D07','5.1','4/252',4/252,'analytic extrapolation')
add('D08','5.1','half-life at b=1-4/252',half(1-4/252),'analytic extrapolation')
add('D09','5.2 after (5.1)','100*252',100*252,'analytic')
add('D10','5.2 after (5.1)','0.01*sqrt(25200)',.01*math.sqrt(25200),'analytic')
add('D11','5.6 equation (5.2)','rho(1) at W=60',2*59/121,'analytic')
add('D12','5.6 equation (5.2)','AR1-equivalent half-life at W=60',half(2*59/121),'analytic')
add('D13','5.6 equation (5.2)','AR1-equivalent half-life at W=30',half(2*29/61),'analytic')
add('D14','5.6 equation (5.2)','AR1-equivalent half-life at W=120',half(2*119/241),'analytic')
sim=[]
for T in [60,252]:
    bs=[]
    for start in range(0,N,10000):
        # Exactly T observed levels, including X_0=0: T-1 fitted transitions.
        levels=np.column_stack([np.zeros(10000),np.cumsum(rng.standard_normal((10000,T-1)),axis=1)])
        x=levels[:,:-1];y=levels[:,1:];xc=x-x.mean(axis=1,keepdims=True);yc=y-y.mean(axis=1,keepdims=True)
        bs.append(np.sum(xc*yc,axis=1)/np.sum(xc*xc,axis=1))
    b=np.concatenate(bs);m=float(b.mean());median=float(np.median(b));ok=(b>0)&(b<1);h=np.full(N,np.inf);h[ok]=-np.log(2)/np.log(b[ok]);valid=ok&(h<30)
    rec=dict(T_levels=T,n_regression_pairs=T-1,paths=N,mean_b=m,median_b=median,mean_T_b_minus_1=float((T*(b-1)).mean()),half_life_of_mean_b=half(m),half_life_of_median_b=half(median),stationary_fraction=float(ok.mean()),valid_trade_day_condition_fraction=float(valid.mean()),valid_trade_day_condition_count=int(valid.sum()),valid_fraction_MC_SE=float(np.sqrt(valid.mean()*(1-valid.mean())/N)),seed=seed)
    sim.append(rec)
    for k,v in rec.items():
        if k not in ['seed','T_levels']:add(f'S{T}_{k}',f'synthetic T={T}',k,v,'100000 Gaussian random walks, intercept OLS, sequential PCG64 stream, seed 20261105')
with (ROOT/'derived_numbers.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(ROOT/'simulation_results.json').write_text(json.dumps(dict(convention='T observed levels including X0=0; T-1 transitions; intercept OLS; transform mean/median b, not mean/median of individual half-lives; same RNG continued from T60 to T252; batch=10000',results=sim),indent=2))
print(json.dumps(sim,indent=2))
