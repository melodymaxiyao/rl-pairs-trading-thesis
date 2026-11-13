"""Pair clusters, fixed-effects Step0, DSR and deterministic CSCV."""
from s2io import *
import numpy as np,pandas as pd
from scipy.stats import rankdata,spearmanr,norm
from itertools import combinations

def pair_bootstrap(frame,column,alternative='positive',seed=20261015,draws=2000):
    f=frame.dropna(subset=[column]);pairs=sorted(f.pair.unique());groups=[f[f.pair==p][column].to_numpy(float) for p in pairs]
    observed=float(f[column].mean()) if len(f) else np.nan
    rng=np.random.default_rng(seed);sums=np.array([v.sum() for v in groups]);sizes=np.array([len(v) for v in groups]);indices=rng.integers(len(pairs),size=(draws,len(pairs))) if pairs else np.empty((0,0),int)
    bs=sums[indices].sum(axis=1)/sizes[indices].sum(axis=1) if pairs else np.full(draws,np.nan)
    p=float((1+np.sum(bs-observed>=observed))/(draws+1)) if alternative=='positive' else float((1+np.sum(bs-observed<=observed))/(draws+1))
    record('pair_cluster_bootstrap',draws=draws,n_pairs=len(pairs),n_pair_folds=len(f),seed=seed,alternative=alternative)
    return dict(difference=observed,ci95_low=float(np.quantile(bs,.025)),ci95_high=float(np.quantile(bs,.975)),p_one_sided=p,n_pairs=len(pairs),n_pair_folds=len(f),bootstrap_draws=draws)

def partial_rank(frame,outcome):
    f=frame[['pair','category','D',outcome]].dropna().copy();rD=rankdata(f.D,method='average');rY=rankdata(f[outcome],method='average')
    cat=pd.get_dummies(f.category,drop_first=True,dtype=float).to_numpy();X=np.column_stack([np.ones(len(f)),rD,cat]);y=rY
    coef=float(np.linalg.lstsq(X,y,rcond=None)[0][1]);pairs=sorted(f.pair.unique());ixs=[np.flatnonzero(f.pair.to_numpy()==p) for p in pairs]
    rng=np.random.default_rng(20261015);samples=[];invalid=0
    for _ in range(2000):
        ix=np.concatenate([ixs[i] for i in rng.integers(len(pairs),size=len(pairs))])
        covariates=np.delete(X[ix],1,axis=1);res=X[ix,1]-covariates@np.linalg.lstsq(covariates,X[ix,1],rcond=None)[0]
        if res@res<1e-20:invalid+=1;continue
        samples.append(float(np.linalg.lstsq(X[ix],y[ix],rcond=None)[0][1]))
    bs=np.array(samples);p=float((1+invalid+np.sum(bs-coef<=coef))/2001)
    record('Step0_rank_cluster_bootstrap',draws=2000,seed=20261015,outcome=outcome,undefined_draws=invalid)
    return dict(outcome=outcome,coefficient=coef,p_one_sided=p,ci95_low=float(np.quantile(bs,.025)),ci95_high=float(np.quantile(bs,.975)),n_pair_folds=len(f),n_pairs=len(pairs),undefined_draws=invalid,category_levels=sorted(f.category.unique()))
def run_step0():
    assert (HERE/'preregistration.json').exists() and (HERE/'tests/test_s2.py').exists()
    f=pd.read_csv(note(STAGE1/'outputs/tables/part4_go_no_go.csv'),float_precision='round_trip');f=f[f.row_type.eq('pair_fold_outcome')]
    a=partial_rank(f,'Y1');b=partial_rank(f,'Y2');desc=[]
    for cat,g in f.dropna(subset=['D','Y1']).groupby('category'):
        if len(g)>=20:desc.append(dict(category=cat,n_pair_folds=len(g),rho=float(spearmanr(g.D,g.Y1).statistic)))
    power=a['coefficient']<0 and a['p_one_sided']<.10
    out=dict(frozen_at_utc=utc(),Y1=a,Y2_descriptive=b,within_category_spearman=desc,D_within_category_power=bool(power),primary_variant='V1' if power else 'V4',preregistration_sha256=sha(HERE/'preregistration.json'),new_test_trading_evaluations=0)
    write_json(HERE/'step0_decision.json',out);print('Step0 primary',out['primary_variant'],'coefficient',a['coefficient'],'p',a['p_one_sided'],flush=True)

def dsr(sr,T,skew,kurt,var,N):
    threshold=np.sqrt(var)*((1-np.euler_gamma)*norm.ppf(1-1/N)+np.euler_gamma*norm.ppf(1-1/(N*np.e)))
    denominator=np.sqrt(1-skew*sr+(kurt-1)*sr*sr/4)
    return dict(N=N,SR=sr,SR_star=float(threshold),T=T,skewness=skew,kurtosis=kurt,cross_trial_SR_variance=var,DSR=float(norm.cdf((sr-threshold)*np.sqrt(T-1)/denominator)))
def pbo(matrix):
    matrix=np.asarray(matrix,float);n,folds=matrix.shape;assert folds%2==0 and np.isfinite(matrix).all()
    rows=[]
    for selected in combinations(range(folds),folds//2):
        holdout=[i for i in range(folds) if i not in selected];IS=matrix[:,selected].mean(axis=1);OOS=matrix[:,holdout].mean(axis=1);best=int(np.argmax(IS));rank=float(rankdata(OOS,method='average')[best]);omega=rank/(n+1);logit=float(np.log(omega/(1-omega)))
        rows.append(dict(IS_folds=','.join(str(i+1) for i in selected),selected_variant='V'+str(best),IS_mean_bps=float(IS[best]),OOS_mean_bps=float(OOS[best]),OOS_rank=rank,relative_rank=omega,logit=logit,overfit=logit<=0))
    return dict(PBO=float(np.mean([r['overfit'] for r in rows])),n_splits=len(rows),logits=[r['logit'] for r in rows]),rows
