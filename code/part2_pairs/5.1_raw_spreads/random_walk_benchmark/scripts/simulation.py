"""Matched correlated log random walks, train admission, deterministic rejection."""
from rw_io import *
from sources import helpers,fast_modules,episodes,training_pass
import numpy as np

def rng_for(pool,year,pair,replicate,variant='primary'):
    key=json.dumps([20261022,pool,int(year),pair,int(replicate),variant],separators=(',',':'))
    digest=hashlib.sha256(key.encode()).digest()
    return np.random.default_rng(np.random.SeedSequence(np.frombuffer(digest,dtype='<u4').tolist())),digest.hex()

def draw_logs(length,sa,sb,rho,rng,martingale=False):
    assert sa>0 and sb>0 and abs(rho)<=1+1e-12
    rho=float(np.clip(rho,-1,1));u=rng.standard_normal((length-1,2))
    increments=np.column_stack([sa*u[:,0],sb*(rho*u[:,0]+np.sqrt(max(0,1-rho*rho))*u[:,1])])
    if martingale:increments-=.5*np.array([sa*sa,sb*sb])
    paths=np.vstack([np.full(2,np.log(100.)),np.log(100.)+np.cumsum(increments,axis=0)])
    return paths[:,0],paths[:,1]

@lru_cache(maxsize=1024)
def train_prices(path,year):
    import csv,io,mmap,pandas as pd
    # Admit only formation/train numeric bytes. Legacy adjusted caches use
    # Close; fresh files use Adj Close. market_parameters selects the sealed
    # lineage column, without substituting between those series.
    rows=[]
    with Path(path).open('rb') as stream:
        with mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as data:
            first=data.find(b'\n');assert first>=0
            header=data[:first+1]
            columns=next(csv.reader([header.decode().strip()]))
            assert columns[0]=='Date' and ('Adj Close' in columns or 'Close' in columns)
            position=first+1
            while position<len(data):
                end=data.find(b'\n',position)
                end=len(data) if end<0 else end
                comma=data.find(b',',position,end)
                if comma<0:
                    assert end==position,'Malformed date row'
                else:
                    row_year=int(data[position:comma].strip(b'"\r ')[:4])
                    if year-4<=row_year<year:rows.append(data[position:end]+b'\n')
                position=end+1
    frame=pd.read_csv(io.BytesIO(header+b''.join(rows)),parse_dates=['Date'],float_precision='round_trip').set_index('Date')
    frame.index=pd.DatetimeIndex(frame.index).tz_localize(None)
    assert frame.index.is_unique and ((frame.index.year>=year-4)&(frame.index.year<year)).all()
    return frame.sort_index()

def market_parameters(row):
    year=int(row['test_year']);cal=helpers().full_cal
    dates=cal[(cal.year>=year-3)&(cal.year<year)];values=[];history=[]
    prior=cal[(cal.year>=year-4)&(cal.year<year)]
    for key in ['leg_A_lineage','leg_B_lineage']:
        lineage=row[key]
        if isinstance(lineage,str):lineage=json.loads(lineage)
        path=note(ROOT/lineage['price_file'])
        frame=train_prices(str(path),year)
        column='Close' if lineage['price_column'].startswith('Close') else 'Adj Close'
        p=frame[column].reindex(dates).to_numpy(float)
        assert np.isfinite(p).all() and (p>0).all(),('Incomplete matched train prices',row['pair'],year)
        values.append(np.log(p))
        full=frame[column].reindex(prior).to_numpy(float)
        assert np.isfinite(full).all() and (full>0).all()
        history.append(np.log(full))
        record('train_parameter_reader',admitted_train_rows=len(frame),max_numeric_year=int(frame.index.year.max()),test_year=year,real_test_numeric_rows=0)
    returns=np.diff(np.array(values),axis=1)
    sa,sb=np.std(returns,axis=1,ddof=1);rho=float(np.corrcoef(returns)[0,1])
    assert sa>0 and sb>0 and np.isfinite(rho)
    formation=dates.year==year-1
    _,beta,_=helpers().comp.ols(values[0][formation],values[1][formation])
    n=int((cal.year==year).sum())-1
    known=n*.5*(-sa*sa+beta*sb*sb)/(1+abs(beta))*10000
    train=fast_modules().numeric.make_episodes(helpers(),prior,history[0],history[1],year,row['pair'],range(year-3,year))
    real_sign=helpers().data.h1_sign(train)[row['pair']]
    return dict(sigma_A=float(sa),sigma_B=float(sb),rho=float(np.clip(rho,-1,1)),
                rho_roundoff_clipped=abs(rho)>1,test_F_beta=float(beta),annual_known_log_spread_drift_bps=float(known),
                real_train_drift_sign=int(real_sign),train_numeric_end_year=year-1)

def matched(row,pool,replicate,kind,variant='primary',max_attempts=200):
    year=int(row['test_year']);h=helpers();cal=h.full_cal[(h.full_cal.year>=year-4)&(h.full_cal.year<=year)]
    rng,key=rng_for(pool,year,row['pair'],replicate,variant)
    name=f'{pool}:{year}:{row["pair"]}:{replicate}:{variant}'
    base=dict(pool=pool,pair=row['pair'],test_year=year,replicate=replicate,variant=variant,stream_sha256=key,
              **{k:row[k] for k in ['sigma_A','sigma_B','rho']})
    for attempt in range(1,max_attempts+1):
        la,lb=draw_logs(len(cal),row['sigma_A'],row['sigma_B'],row['rho'],rng,variant=='price_martingale')
        try:es=episodes(cal,la,lb,year,name)
        except ValueError:
            record('construction_rejection',null_attempts=1,construction_rejections=1);continue
        accepted=training_pass(es[:3],kind)
        record('selection_attempt',null_attempts=1,selection_accepts=int(accepted),filter=kind)
        if accepted:return dict(**base,attempts=attempt,accepted=True),es
    return dict(**base,attempts=max_attempts,accepted=False,reason='200_attempt_limit'),None

def continuation(year,index):
    h=helpers();prior=h.full_cal[h.full_cal.year<year];dates=h.full_cal[h.full_cal.year<=year]
    vols=read(note(FAST/f'runtime/null_volatilities_{year}.json'))['volatilities']
    la,lb,sa,sb=fast_modules().worker.simulate_logs(len(prior),year,index,vols)
    rng,key=rng_for('UNIVERSE_CONTINUITY',year,f'NULL_{index:04d}',0,'extension')
    n=len(dates)-len(prior)
    # Never lengthen the original draw: it would change the second leg's train path.
    aa=np.r_[la,la[-1]+np.cumsum(rng.normal(0,sa,n))]
    bb=np.r_[lb,lb[-1]+np.cumsum(rng.normal(0,sb,n))]
    es=episodes(dates,aa,bb,year,f'NULL_{index:04d}')
    return dict(pool='UNIVERSE_CONTINUITY',pair=f'NULL_{index:04d}',test_year=year,replicate=0,variant='continuity',
                stream_sha256=key,sigma_A=sa,sigma_B=sb,rho=0.,attempts=1,accepted=True),es
