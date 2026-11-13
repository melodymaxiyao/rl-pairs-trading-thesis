"""Streaming CSV parts, deterministic gzip and small reporting inventories."""
from universe_io import *
from checkpoints import load_rows,ready,receipt
from screen import assign_pools
from reproduction import compare_sector_fold
import csv,gzip,io,time
import numpy as np
from collections import Counter

POOLS=['FAST','FAST_cap','FAST_X_cap','FAST_W_cap','inner_FAST','inner_FAST_cap','inner_FAST_X_cap','inner_FAST_W_cap']

def scalar(value):
    value=clean(value)
    if value is None:return ''
    if isinstance(value,(dict,list)):return json.dumps(value,separators=(',',':'),allow_nan=False)
    return value

def csv_new(path,rows,columns=None):
    rows=list(rows)
    if columns is None:columns=sorted({k for r in rows for k in r})
    if not columns:columns=['status']
    path=inside(path)
    pending=path.with_name(path.name+'.pending_'+str(time.time_ns()))
    with pending.open('xb') as binary:
        if path.suffix=='.gz':
            gz=gzip.GzipFile(filename='',fileobj=binary,mode='wb',mtime=0)
            stream=io.TextIOWrapper(gz,newline='')
        else:stream=io.TextIOWrapper(binary,newline='')
        writer=csv.DictWriter(stream,fieldnames=columns,extrasaction='raise')
        writer.writeheader()
        for row in rows:writer.writerow({k:scalar(row.get(k)) for k in columns})
        stream.flush()
        if path.suffix=='.gz':stream.close()
        else:stream.detach()
    if path.exists():assert sha(path)==sha(pending),('Existing CSV differs',path)
    else:os.link(pending,path)
    pending.unlink()
    record('CSV_generated',CSV_rows_written=len(rows),CSV_files_written=1)
    return list(columns)

def merge_csv(parts,target):
    headers=set()
    for path in parts:
        with gzip.open(path,'rt',newline='') as stream:headers.update(next(csv.reader(stream)))
    columns=sorted(headers)
    pending=target.with_name(target.name+'.pending_'+str(time.time_ns()))
    with pending.open('xb') as binary:
        gz=gzip.GzipFile(filename='',fileobj=binary,mode='wb',mtime=0) if target.suffix=='.gz' else None
        stream=io.TextIOWrapper(gz if gz else binary,newline='')
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for path in parts:
            with gzip.open(path,'rt',newline='') as source:
                for row in csv.DictReader(source):writer.writerow(row)
        stream.flush()
        if gz:stream.close()
        else:stream.detach()
    if target.exists():assert sha(target)==sha(pending),('Existing merged CSV differs',target)
    else:os.link(pending,target)
    pending.unlink()

def finalize_fold(year,markers,null_markers):
    done=HERE/f'runtime/final_fold_{year}.json'
    if ready(done):return read(done)
    rows=[]
    for marker in markers:rows.extend(load_rows(Path(marker)))
    null=[]
    for marker in null_markers:null.extend(load_rows(Path(marker)))
    assert len(null)==2000 and len({r['simulation'] for r in null})==2000
    pools={**assign_pools(rows),**assign_pools(rows,True)}
    for row in rows:
        row['group_id']=row.get('FAST_cap_group_id','')
        row['inner_group_id']=row.get('inner_FAST_cap_group_id','')
    reproduction,legacy=compare_sector_fold(year,rows)
    write(HERE/f'audit/reproduction_{year}.json',reproduction)
    parts=HERE/'runtime/parts';files=[]
    inventory=parts/f'inventory_{year}.csv.gz';csv_new(inventory,rows);files.append(inventory)
    for name,selected in pools.items():
        membership=[]
        for row in selected:
            entry={k:row[k] for k in ['pair','leg_A','leg_B','test_year','fold','era','sector_A','sector_B',
                                      'sector_relation','in_EXP_ALL','in_EXP_G','in_old68','previously_traded']}
            entry['pool']=name;entry['group_id']=row.get(name+'_group_id','')
            membership.append(entry)
        path=parts/f'{name}_{year}.csv.gz'
        csv_new(path,membership,['pair','leg_A','leg_B','test_year','fold','era','sector_A','sector_B','sector_relation',
                                 'in_EXP_ALL','in_EXP_G','in_old68','previously_traded','pool','group_id'])
        files.append(path)
    for name,values in [('null_simulation',null),('legacy_differences',legacy)]:
        path=parts/f'{name}_{year}.csv.gz';csv_new(path,values);files.append(path)
    funnel=[]
    for relation in ['cross_sector','same_sector','same_subindustry']:
        part=[r for r in rows if r['sector_relation']==relation]
        eligible=[r for r in part if r['eligible']]
        staged=eligible
        entry=dict(test_year=year,era=era(year),sector_relation=relation,candidates=len(part),priced=len(part),
                   train_eligible=sum(r['F_eligible'] for r in part),eligible=len(eligible))
        for field in ['condition_speed','condition_evidence','condition_gross','condition_VR5']:
            staged=[r for r in staged if r[field]];entry[field]=len(staged)
        for field in ['FAST63','FAST126','FAST63_a15','FAST63_legacy','G_pass']+POOLS:
            entry[field]=sum(bool(r.get(field,False)) for r in (part if field.startswith('inner_') else eligible))
        entry['inner_eligible']=sum(bool(r['inner_eligible']) for r in part)
        funnel.append(entry)
    write(HERE/f'runtime/funnel_{year}.json',funnel)
    concentration=[];overlap=[];matrix=Counter();edges={}
    views=dict(pools)
    for prefix in ['', 'inner_']:
        for suffix,relation in [('X','cross_sector'),('W',None)]:
            views[prefix+'FAST_'+suffix+'_before_cap']=[r for r in pools[prefix+'FAST'] if (r['sector_relation']=='cross_sector')==(relation=='cross_sector')]
    for name,part in views.items():
        degrees=Counter(t for r in part for t in (r['leg_A'],r['leg_B']))
        special=sum(r['leg_A'] in ('NVDA','TSLA') or r['leg_B'] in ('NVDA','TSLA') for r in part)
        for ticker,degree in sorted(degrees.items()):
            concentration.append(dict(test_year=year,era=era(year),pool=name,ticker=ticker,pairs=degree,
                                      pool_pair_folds=len(part),NVDA_TSLA_pair_fold_share=special/len(part) if part else None))
        if name in pools:
            overlap.append(dict(test_year=year,era=era(year),pool=name,pair_folds=len(part),NVDA_TSLA_pairs=special,
                                **{f:sum(r[f] for r in part) for f in ['in_EXP_ALL','in_EXP_G','in_old68','previously_traded']}))
        if name.endswith('_cap'):edges[name]=[[r['leg_A'],r['leg_B']] for r in part]
    for r in pools['FAST_X_cap']:
        matrix[tuple(sorted((r['sector_A'],r['sector_B'])))]+=1
    write(HERE/f'runtime/structure_{year}.json',dict(concentration=concentration,overlap=overlap,
        matrix=[dict(sector_A=a,sector_B=b,count=n,test_year=year,era=era(year)) for (a,b),n in sorted(matrix.items())],edges=edges))
    eligible=[r for r in rows if r['eligible']]
    diagnostics=parts/f'diagnostics_{year}.npz'
    arrays={k:np.asarray([r.get(k,np.inf if 'KM_med' in k else False) for r in eligible])
            for k in ['KM_med','KM_med_a15','KM_med_legacy','inner_KM_med','validation_KM_med','inner_FAST63','sector_relation','FAST63','FAST63_a15','FAST63_legacy','G_pass']}
    arrays.update({'null_'+k:np.asarray([r[k] for r in null]) for k in ['KM_med','inner_KM_med','validation_KM_med','inner_FAST63','FAST63','FAST126']})
    pending=diagnostics.with_name(diagnostics.name+'.pending_'+str(time.time_ns()))
    with pending.open('xb') as stream:np.savez_compressed(stream,**arrays)
    if diagnostics.exists():assert sha(diagnostics)==sha(pending),('Diagnostic checkpoint differs',diagnostics)
    else:os.link(pending,diagnostics)
    pending.unlink()
    files.append(diagnostics)
    removals={}
    for label,part,prefix in [('real_final',eligible,''),('real_inner',[r for r in rows if r['inner_eligible']],'inner_'),
                              ('null_final',null,''),('null_inner',null,'inner_')]:
        removals[label]={key:sum(int(r.get(prefix+key,0)) for r in part)
                        for key in ['legacy_events','legacy_removed_overlap','legacy_removed_year_end','legacy_removed_year_start','legacy_year_end_crossings']}
    receipt(done,files,test_year=year,candidate_pair_folds=len(rows),eligible_pair_folds=len(eligible),
            pool_sizes={k:len(v) for k,v in pools.items()},null_pairs=len(null),reproduction=reproduction,unique_event_removals=removals)
    print('Fold',year,'eligible',len(eligible),'FAST63',len(pools['FAST']),
          'caps',*[len(pools[p]) for p in ['FAST_cap','FAST_X_cap','FAST_W_cap']],flush=True)
    return read(done)

def export_all():
    done=HERE/'audit/export_ready.json'
    if ready(done):return read(done)
    parts=HERE/'runtime/parts';tables=HERE/'outputs/tables';files=[]
    for name,stem in [('pair_fold_inventory','inventory'),('null_simulation','null_simulation'),('legacy_differences','legacy_differences')]:
        path=tables/(name+('.csv.gz' if name!='legacy_differences' else '.csv'))
        merge_csv([parts/f'{stem}_{y}.csv.gz' for y in range(2004,2026)],path);files.append(path)
    for pool in POOLS:
        path=HERE/'pools'/(pool+'.csv.gz');merge_csv([parts/f'{pool}_{y}.csv.gz' for y in range(2004,2026)],path);files.append(path)
    path=tables/'candidates_by_fold.csv.gz'
    merge_csv([parts/f'candidates_{y}.csv.gz' for y in range(2004,2026)],path);files.append(path)
    for name in ['missing_members_by_year','yahoo_only_gap']:
        path=tables/(name+'.csv');merge_csv([parts/f'{name}_{y}.csv.gz' for y in range(2004,2026)],path);files.append(path)
    receipt(done,files,tables=len(files));return read(done)
