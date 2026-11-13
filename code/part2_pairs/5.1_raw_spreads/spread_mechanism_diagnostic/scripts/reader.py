"""Date-first train admission: stop at Y, parse only Y-4 through Y-1."""
from mio import *
import csv,io
import numpy as np
import pandas as pd

def admit(path,year):
    with Path(path).open('rb') as stream:
        header=stream.readline();columns=next(csv.reader([header.decode().strip()]))
        assert columns[0]=='Date' and ('Adj Close' in columns or 'Close' in columns)
        rows=[];previous=None
        for line in stream:
            if b',' not in line:
                assert not line.strip(b'\r\n');continue
            prefix=int(line.lstrip(b'"')[:4])
            if prefix>=year:break
            if prefix<year-4:continue
            date=line[:line.index(b',')].strip(b'"\r ')
            assert previous is None or date>previous,('Unsorted admitted train rows',path)
            previous=date;rows.append(line if line.endswith(b'\n') else line+b'\n')
    return header,rows

def read_train(path,year,columns=('Adj Close','Volume')):
    header,rows=admit(path,year)
    frame=pd.read_csv(io.BytesIO(header+b''.join(rows)),usecols=['Date',*columns],parse_dates=['Date'],float_precision='round_trip').set_index('Date')
    if frame.empty:frame=frame.astype({name:'float64' for name in columns})
    frame.index=pd.DatetimeIndex(frame.index).tz_localize(None)
    assert frame.index.is_unique and ((frame.index.year>=year-4)&(frame.index.year<year)).all()
    record('train_reader',price_files_read=1,train_numeric_rows=len(frame),max_numeric_year=int(frame.index.year.max()) if len(frame) else None,
           test_year=year,test_rows_parsed=0,source=str(path))
    return frame

def read_sample(year):
    import gzip
    marker=read(note(PROPERTY/f'runtime/checkpoints/sample_{year}.json'))
    for key,value in marker['files'].items():assert fp(PROPERTY/key)==value
    with gzip.open(note(PROPERTY/marker['payload']),'rt') as f:rows=json.load(f)
    assert len(rows)==marker['rows'] and sum(r['in_random'] for r in rows)==10000
    assert len({r['pair'] for r in rows})==len(rows)
    return rows

def partial_sample(rows,year,size=1000):
    relations=['cross_sector','same_sector','same_subindustry'];random=[r for r in rows if r['in_random']]
    groups={k:sorted([r for r in random if r['sector_relation']==k],key=lambda r:r['pair']) for k in relations}
    counts=np.array([len(groups[k]) for k in relations]);quota=counts*size/len(random);n=np.floor(quota).astype(int)
    for i in np.argsort(-(quota-n),kind='stable')[:size-int(n.sum())]:n[i]+=1
    result={}
    for index,relation in enumerate(relations):
        part=groups[relation]
        if not n[index]:continue
        rng=np.random.default_rng(np.random.SeedSequence([SEED,year,index]))
        for i in rng.choice(len(part),int(n[index]),replace=False):
            row=part[int(i)];result[row['pair']]=float(row['ipw'])*len(part)/n[index]
    assert len(result)==size
    return result

def metadata():
    def rows(name):
        with note(FRESH/name).open(newline='') as f:return list(csv.DictReader(f))
    classification={r['yahoo_symbol']:r for r in rows('classification.csv')}
    membership={}
    for r in rows('membership_monthly.csv'):membership.setdefault(r['month_end'],set()).add(r['yahoo_symbol'])
    return classification,membership

def spy_returns(year,cal):
    fresh=note(FRESH/'data/raw/prices/spy_yahoo.csv')
    legacy=note(ROOT/'data/raw/15_pair_trading/economic_peer_residual_screen/spy_daily_adjusted_ohlcv.csv')
    result=np.full(len(cal),np.nan)
    for path,column,mask in [(fresh,'Adj Close',cal.year<=2008),(legacy,'Close',cal.year>2008)]:
        if not mask.any():continue
        f=read_train(path,year,(column,));p=f[column].reindex(cal).to_numpy(float)
        with np.errstate(divide='ignore',invalid='ignore'):r=np.r_[np.nan,np.diff(np.log(p))*10000]
        result[mask]=r[mask]
    return result
