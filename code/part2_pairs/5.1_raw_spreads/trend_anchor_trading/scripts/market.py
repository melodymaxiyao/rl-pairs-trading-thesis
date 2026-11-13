"""Date-bounded readers, identical anchors extended to Y, paired null worlds."""
from aio import *
from store import cached,commit
from originals import helpers,fixed_spread,original_episodes,gaussian
from engine import rolling
import numpy as np,pandas as pd
import csv,io
NA_KEYS={'CPAY_FISV','FIS_FISV','FISV_GPN','FISV_JKHY','FISV_MA','FISV_PYPL','FISV_V'}
ANCHOR_NAMES=['A_TR','A_R63','A_F','F_orig']

@lru_cache(maxsize=1400)
def prices(ticker,year,include_test):
    path=note(FRESH/f'data/raw/prices/{ticker.lower()}_yahoo.csv');rows=[]
    with path.open('rb') as f:
        header=f.readline()
        for line in f:
            if b',' not in line:continue
            y=int(line.lstrip(b'"')[:4])
            if y>year or (y==year and not include_test):break
            if y>=year-4:rows.append(line)
    frame=pd.read_csv(io.BytesIO(header+b''.join(rows)),parse_dates=['Date'],usecols=['Date','Adj Close'],float_precision='round_trip').set_index('Date')
    frame.index=pd.DatetimeIndex(frame.index).tz_localize(None)
    assert frame.index.is_unique and frame.index.is_monotonic_increasing
    assert not len(frame) or frame.index.year.max()<year+int(include_test)
    record('price_read',price_files_read=1,train_numeric_rows=int((frame.index.year<year).sum()),test_numeric_rows=int((frame.index.year==year).sum()),future_numeric_rows=0)
    return frame['Adj Close'].astype(float)

def rng(pair,year,world,rep):
    key=json.dumps([20261026,'TREND_ANCHOR',int(year),pair,world,int(rep)],separators=(',',':'))
    return np.random.default_rng(np.random.SeedSequence(np.frombuffer(hashlib.sha256(key.encode()).digest(),dtype='<u4').tolist()))

def drift_surface(r):
    r=np.asarray(r,float);mu=np.full(len(r),np.nan);t=mu.copy()
    if len(r)>252:
        windows=np.lib.stride_tricks.sliding_window_view(r,252)[:-1]
        means=windows.mean(axis=1);sd=windows.std(axis=1,ddof=1)
        mu[252:]=means
        t[252:]=np.divide(means*np.sqrt(252),sd,out=np.zeros(len(means)),where=sd>0)
    return np.nan_to_num(np.sign(mu)),np.nan_to_num(abs(t)>=1).astype(np.int32),t

def surfaces(cal,level,year):
    out={}
    for name in ['A_TR','A_R63','A_F']:
        if name=='A_F':
            m=np.full(len(cal),np.nan);s=m.copy()
            for y in range(year-3,year+1):
                source=cal.year==y-1;target=cal.year==y
                m[target]=level[source].mean();s[target]=level[source].std(ddof=1)
        else:m,s=rolling(level,63 if name=='A_R63' else 252,name=='A_TR')
        with np.errstate(divide='ignore',invalid='ignore'):d=(level-m)/s
        d[(s<=0)|~np.isfinite(d)]=np.nan;out[name]=dict(m=m,s=s,d=d)
    return out

def assemble(cal,a,b,year,pair,variant='primary'):
    h=helpers();h.comp.CAL=cal
    formation=cal.year==(year-1 if variant=='fresh_beta' else year-4)
    _,beta,_=h.comp.ols(a[formation],b[formation]);r=np.r_[np.nan,np.diff(a-beta*b)*10000/(1+abs(beta))]
    # Match the sealed construction's arithmetic operation order exactly.
    ra=np.r_[np.nan,np.diff(a)*10000];rb=np.r_[np.nan,np.diff(b)*10000]
    r=(ra-beta*rb)/(1+abs(beta));S=np.r_[0.,np.cumsum(r[1:])]
    if variant=='fresh_beta':S-=S[np.flatnonzero(formation)[0]]
    anchors=surfaces(cal,S,year);drift,strong,t=drift_surface(r)
    scored=range(year-3,year+int((cal.year==year).any()))
    from originals import modules
    annual=modules().sources.universe().make_episodes(h,cal,a,b,year,pair,scored)
    for e in annual:
        e.update(window='test' if e['year']==year else 'train',test_year=year)
    fz=np.full(len(cal),np.nan)
    for e in annual:fz[cal.year==e['year']]=e['z']
    anchors['F_orig']=dict(d=fz,m=np.full(len(cal),np.nan),s=np.full(len(cal),np.nan))
    episodes={}
    for name,anchor in anchors.items():
        es=[]
        for y in scored:
            ix=np.flatnonzero(cal.year==y);z=anchor['d'][ix];valid=np.isfinite(z)
            assert valid.all(),('Incomplete anchor scoring year',pair,year,y,name)
            fwd=np.r_[np.diff(S[ix]),0.]
            e=dict(pair=pair,fold=source_fold(year),test_year=year,year=y,window='test' if y==year else 'train',construction='F',dates=cal[ix],z=z,beta=np.full(len(ix),beta),bins=np.searchsorted([-2,-1.5,-1,-.5,.5,1,1.5,2],z,side='right').astype(np.int32),forward_bps=fwd,drift=drift[ix],strong=strong[ix],drift_t=t[ix],previous_z=float(anchor['d'][ix[0]-1]),group=int(abs(t[ix[0]])>=1) if np.isfinite(t[ix[0]]) else 0,category='all',origin='new')
            if name=='F_orig':
                old=next(e for e in annual if e['year']==y);e.update(old)
                e.update(drift=drift[ix],strong=strong[ix],drift_t=t[ix],group=int(abs(t[ix[0]])>=1) if np.isfinite(t[ix[0]]) else 0)
            es.append(e)
        episodes[name]=es
    train=(cal.year>=year-3)&(cal.year<year);returns=np.diff(np.array([a[train],b[train]]),axis=1)
    quarters=[]
    for y in range(year-3,year):
        for q in range(1,5):
            ix=(cal.year==y)&(cal.quarter==q);quarters.append(h.comp.ols(a[ix],b[ix])[1])
    mean=float(np.mean(quarters));quality=dict(leg_return_corr=float(np.corrcoef(returns)[0,1]),spread_sd=float(np.std(r[train],ddof=1)),quarterly_beta_cv=float(np.std(quarters,ddof=1)/abs(mean)) if mean else np.nan)
    return dict(beta=float(beta),episodes=episodes,quality=quality,cal=cal,level=S,r=r,anchors=anchors)

def context(row,year,world='real',rep=0,variant='primary',include_test=False):
    # Train and full readers have separate immutable payloads. Choices are frozen
    # before calling this function with include_test=True.
    stem=f'runtime/contexts/{year}/{row["pair"]}/{row["leg_A"]}--{row["leg_B"]}/{world}_{rep}_{variant}_{"full" if include_test else "train"}'
    old=cached(stem)
    if old is not None:return old
    h=helpers();cal=h.full_cal[(h.full_cal.year>=year-4)&(h.full_cal.year<=year if include_test else h.full_cal.year<year)]
    a,b=[np.log(prices(row[k],year,include_test).reindex(cal).to_numpy(float)) for k in ['leg_A','leg_B']]
    if not np.isfinite([a,b]).all():
        if include_test and year==2025 and row['pair'] in NA_KEYS:
            record('authorised_NA',unavailable_pair_worlds=1,pair=row['pair'],year=year,world=world,replicate=rep)
            return commit(stem,dict(unavailable=True,reason='sealed FISV 2025 numeric gap; no fill or substitution'))
        raise ValueError(('Unexpected incomplete prices; stop without dropping',row['pair'],year))
    if world!='real':
        random=rng(row['pair'],year,world,rep)
        if world=='N2':
            signs=random.choice(np.array([-1.,1.]),size=len(a)-1)
            real_beta=helpers().comp.ols(a[cal.year==year-4],b[cal.year==year-4])[1]
            a=np.r_[a[0],a[0]+np.cumsum(np.diff(a)*signs)];b=np.r_[b[0],b[0]+np.cumsum(np.diff(b)*signs)]
        else:
            train=(cal.year>=year-3)&(cal.year<year);rr=np.diff(np.array([a[train],b[train]]),axis=1)
            sa,sb=rr.std(axis=1,ddof=1);rho=float(np.corrcoef(rr)[0,1])
            if include_test:
                admitted=context(row,year,world,rep,variant,False)
                if admitted.get('unavailable'):return commit(stem,dict(admitted))
                random.bit_generator.state=admitted['N1_state_before_draw']
                a,b=gaussian(len(cal),sa,sb,rho,random)
            else:
                import copy
                count('null_replicates')
                for attempt in range(1,201):
                    state=copy.deepcopy(random.bit_generator.state);a1,b1=gaussian(len(cal),sa,sb,rho,random);count('N1_draws')
                    try:result=assemble(cal,a1,b1,year,row['pair'],variant)
                    except (ValueError,AssertionError,np.linalg.LinAlgError):
                        count('N1_numerical_rejections');continue
                    result.update(unavailable=False,N1_attempt=attempt,N1_state_before_draw=state)
                    return commit(stem,result)
                record('N1_unavailable',N1_failed_worlds=1,pair=row['pair'],test_year=year,replicate=rep)
                return commit(stem,dict(unavailable=True,reason='N1 numerical construction failed after 200 train-only attempts'))
        count('null_replicates' if not include_test else 'null_test_extensions')
    result=assemble(cal,a,b,year,row['pair'],variant)
    if world=='N2':
        # Own annual F betas above are retained; fixed-beta anchors retain the
        # real formation beta, including fresh-beta sensitivity's real beta.
        real_cal=h.full_cal[(h.full_cal.year>=year-4)&(h.full_cal.year<year)]
        aa,bb=[np.log(prices(row[k],year,False).reindex(real_cal).to_numpy(float)) for k in ['leg_A','leg_B']]
        f=real_cal.year==(year-1 if variant=='fresh_beta' else year-4);beta=helpers().comp.ols(aa[f],bb[f])[1]
        # Reuse assemble's identical functions with an explicit fixed-beta path.
        ra=np.r_[np.nan,np.diff(a)*10000];rb=np.r_[np.nan,np.diff(b)*10000];r=(ra-beta*rb)/(1+abs(beta));S=np.r_[0.,np.cumsum(r[1:])]
        if variant=='fresh_beta':S-=S[np.flatnonzero(cal.year==year-1)[0]]
        result['beta']=float(beta);result['r']=r;result['level']=S;sur=surfaces(cal,S,year);g,strong,t=drift_surface(r)
        for name in ['A_TR','A_R63','A_F']:
            result['anchors'][name]=sur[name]
            for e in result['episodes'][name]:
                ix=np.flatnonzero(cal.year==e['year']);e.update(z=sur[name]['d'][ix],forward_bps=np.r_[np.diff(S[ix]),0.],drift=g[ix],strong=strong[ix],drift_t=t[ix],beta=np.full(len(ix),beta),previous_z=float(sur[name]['d'][ix[0]-1]),group=int(abs(t[ix[0]])>=1) if np.isfinite(t[ix[0]]) else 0)
    result['unavailable']=False
    return commit(stem,result)

@lru_cache(maxsize=8)
def spy(year):
    # Copy the sealed baseline's exact routed series, including its historical
    # source-vintage preservation. No fresh/legacy substitution or full-file
    # future-price decoding is introduced by this task.
    path=note(BASELINE/f'outputs/ledgers/production_EXP_G_cap_{year}_bench_F.npz')
    with np.load(path,allow_pickle=False) as z:
        return pd.DatetimeIndex(z['dates']),z['forward_SPY'].copy()
