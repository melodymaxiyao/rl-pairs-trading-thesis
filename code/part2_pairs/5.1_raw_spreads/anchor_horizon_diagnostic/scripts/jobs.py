"""One unique real/N1/N2 path per sampled pair-fold; atomic chunk receipts."""
from aio import *
from originals import calendar,read_prices,read_sample,fixed_spread,gaussian,original_episodes
from engine import calculate
import numpy as np

RELATIONS=['cross_sector','same_sector','same_subindustry']

def sample(year,size=3000):
    rows=[r for r in read_sample(year) if r['in_random']];assert len(rows)==10000
    groups=[sorted([r for r in rows if r['sector_relation']==k],key=lambda r:r['pair']) for k in RELATIONS]
    counts=np.array([len(g) for g in groups]);quotas=size*counts/len(rows);n=np.floor(quotas).astype(int)
    for i in np.argsort(-(quotas-n),kind='stable')[:size-int(n.sum())]:n[i]+=1
    selected=[]
    for k,group in enumerate(groups):
        if not n[k]:continue
        rng=np.random.default_rng(np.random.SeedSequence([SEED,year,k]))
        for i in sorted(rng.choice(len(group),int(n[k]),replace=False)):
            row=group[int(i)];selected.append(dict(row,weight=float(row['ipw'])*len(group)/n[k]))
    assert len(selected)==size;return sorted(selected,key=lambda r:r['pair'])

def stream(pair,year,variant):
    key=json.dumps([SEED,'ANCHOR_HORIZON',int(year),pair,0,variant],separators=(',',':'))
    digest=hashlib.sha256(key.encode()).digest()
    return np.random.default_rng(np.random.SeedSequence(np.frombuffer(digest,dtype='<u4').tolist())),digest.hex()

def sign_flip(r,la,lb,rng):
    signs=rng.choice(np.array([-1.,1.]),size=len(r)-1);nr=np.asarray(r,float).copy();nr[1:]*=signs
    level=np.r_[0.,np.cumsum(nr[1:])]
    a=np.r_[la[0],la[0]+np.cumsum(np.diff(la)*signs)];b=np.r_[lb[0],lb[0]+np.cumsum(np.diff(lb)*signs)]
    return nr,level,a,b,signs

def process_pair(cal,year,row,la,lb):
    meta=dict(pair=row['pair'],leg_A=row['leg_A'],leg_B=row['leg_B'],year=year,era=era(year),
              relation=row['sector_relation'],weight=row['weight'],original_ipw=row['ipw'])
    real=fixed_spread(cal,la,lb,year);score=real['score'];returns=np.diff(np.array([la[score],lb[score]]),axis=1)
    sa,sb=returns.std(axis=1,ddof=1);rho=float(np.corrcoef(returns)[0,1]);assert np.isfinite([sa,sb,rho]).all() and sa>0 and sb>0
    rng,seed=stream(row['pair'],year,'N1');n1=None;failure=None
    for attempt in range(1,201):
        count('N1_draws');a,b=gaussian(len(cal),sa,sb,rho,rng)
        try:
            es=original_episodes(cal,a,b,year,row['pair']);spread=fixed_spread(cal,a,b,year)
        except (ValueError,AssertionError,np.linalg.LinAlgError):continue
        n1=(spread,es);break
    if n1 is None:failure='N1 numerical construction failed after 200 attempts'
    rng2,seed2=stream(row['pair'],year,'N2');r2,s2,a2,b2,signs=sign_flip(real['r'],la,lb,rng2);count('N2_draws')
    es_real=original_episodes(cal,la,lb,year,row['pair']);es2=original_episodes(cal,a2,b2,year,row['pair'])
    paths=[('real',real,es_real),('N2',dict(r=r2,level=s2,beta=real['beta']),es2)]
    if n1:paths.append(('N1',*n1))
    events=[];coverage=[];pairs=[]
    for dataset,spread,episodes in paths:
        ev,cov=calculate(cal,spread['level'],spread['r'],year,episodes)
        events.extend(dict(meta,dataset=dataset,fixed_beta=spread['beta'],**e) for e in ev)
        coverage.extend(dict(meta,dataset=dataset,**c) for c in cov)
        pairs.append(dict(meta,dataset=dataset,beta=spread['beta'],status='ok'))
    if failure:pairs.append(dict(meta,dataset='N1',status='not_estimable',reason=failure))
    return dict(events=events,coverage=coverage,pairs=pairs,null_parameters=[dict(meta,sigma_A=sa,sigma_B=sb,rho=rho,
        N1_seed_sha256=seed,N2_seed_sha256=seed2,N1_attempts=attempt,N1_status='failed' if failure else 'accepted')])

def fold(year,chunk_size=100):
    guard();os.environ['AH_SCOPE']='production';start=time.monotonic()
    ready=HERE/f'runtime/checkpoints/fold_{year}.json'
    if checkpoint(ready):print('Resume completed fold',year,flush=True);return
    rows=sample(year);cal=calendar(year);prices={};files=[]
    for offset in range(0,len(rows),chunk_size):
        path=HERE/f'runtime/checkpoints/{year}/chunk_{offset:04d}.pkl.gz';marker=path.with_suffix('.ready.json')
        if checkpoint(marker):files.extend([path,marker]);continue
        out={k:[] for k in ['events','coverage','pairs','null_parameters']}
        for row in rows[offset:offset+chunk_size]:
            for ticker in [row['leg_A'],row['leg_B']]:
                if ticker not in prices:
                    frame=read_prices(FRESH/f'data/raw/prices/{ticker.lower()}_yahoo.csv',year).reindex(cal)
                    with np.errstate(invalid='ignore',divide='ignore'):prices[ticker]=np.log(frame['Adj Close'].to_numpy(float))
                    assert np.isfinite(prices[ticker]).all()
            result=process_pair(cal,year,row,prices[row['leg_A']],prices[row['leg_B']])
            for k in out:out[k].extend(result[k])
        dump_gzip(path,out);receipt(marker,[path],year=year,offset=offset,pairs=len(out['null_parameters']))
        files.extend([path,marker]);print('Checkpoint',year,offset+len(out['null_parameters']),'/3000',flush=True)
    receipt(ready,files,year=year,real_pairs=len(rows),seconds=time.monotonic()-start)
    print('Completed anchor/horizon',year,'seconds',round(time.monotonic()-start,1),flush=True)
