"""Stream executor output, compare sealed references and report frozen predictions."""
from common import *
import numpy as np,pandas as pd,csv,io
from sources import setup,frozen,members
KEYS=['pair','test_year'];REF=['R_TR_aln','B3_TR_aln','TD8','TD8_z','TD16_align','TD16_strength','TD8_group','TD8_shrunk']
def source_table(name):return pd.read_csv(note(SOURCE/'outputs/tables'/name),float_precision='round_trip')
def copied_absolute(summary,pool,cost,label,cell):
 f=summary[summary.pool.eq(pool)&summary.cell.eq(cell)&summary.cost.eq(cost)&summary.world.eq('real')&summary.variant.eq('primary')&summary.stratum.eq('pooled' if label=='pooled' else 'era')&summary.label.eq('ALL' if label=='pooled' else label)]
 assert len(f)==1,('Ambiguous or missing sealed absolute row',pool,cost,label,cell)
 r=f.iloc[0];p=SOURCE/'outputs/tables/evaluation_summary.csv'
 return dict(estimate=r.net_bps,ci95_low=r.get('ci95_low',np.nan),ci95_high=r.get('ci95_high',np.nan),p_one_sided=r.get('p_one_sided',np.nan),scheme='year',source_file=str(p.relative_to(ROOT)),source_sha256=sha(p),source_csv_row=int(f.index[0])+2,interval_status='copied sealed interval' if np.isfinite(r.get('ci95_low',np.nan)) else 'NA: no interval sealed at this cost')
def stream():
 target=TABLES/'per_pair_fold_test.csv.gz';pending=target.with_name(target.name+'.pending_'+str(time.time_ns()));port=[];frames=[];refs=[];hashes={};count_rows=0
 with gzip.open(note(SOURCE/'outputs/tables/per_pair_fold_test.csv.gz'),'rt') as f:fields=next(csv.reader(f))
 with pending.open('xb') as raw:
  with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0,filename='') as gz:
   text=io.TextIOWrapper(gz,encoding='utf-8',newline='');writer=csv.DictWriter(text,fieldnames=fields,extrasaction='ignore');writer.writeheader()
   for pool in POOLS:
    for y in YEARS:
     fold=cached(f'runtime/folds/{pool}_{y}');assert fold is not None
     acc={}
     for stem in fold['chunks']:
      c=cached(stem);assert c is not None
      for r in c['scores']:
       rr=dict(r,n_seeds=10 if r['cell'].endswith('_z') else 1,deterministic=not r['cell'].endswith('_z'),feasible=True,seed=-1)
       rr['source_row_key']=json.dumps([pool,y,r['pair'],r['cell'],r['world'],r['replicate'],r['cost']])
       writer.writerow(rr);count_rows+=1
      f=pd.DataFrame(c['scores']);keep=f.world.isin(['real','N2']);cols=['pool','cell','cost','world','replicate','pair','test_year','weight','available','net_bps','gross_bps','n_trades','relation','era']
      frames.append(f.loc[keep,cols])
      world,rep=f.iloc[0][['world','replicate']]
      for (cell,cost),a in c['portfolios'].items():
       k=(cell,cost,world,int(rep));v=acc.setdefault(k,dict(dates=a['dates'],weight=0.,**{n:np.zeros(len(a['net'])) for n in ['net','gross','fee','turnover']}));v['weight']+=a['weight']
       for n in ['net','gross','fee','turnover']:v[n]+=a[n]
     for (cell,cost,world,rep),a in sorted(acc.items()):
      assert a['weight']>0
      d=pd.DataFrame(dict(date=a['dates'],**{('net_bps' if n=='net' else 'gross_bps' if n=='gross' else 'cost_bps' if n=='fee' else n):a[n]/a['weight'] for n in ['net','gross','fee','turnover']})).assign(pool=pool,cell=cell,cost=cost,world=world,replicate=rep,variant='primary',test_year=y);port.append(d)
     for cell in REF:
      stem=f'runtime/cells/{pool}_{y}_{cell}_real_0_primary';p=SOURCE/(stem+'.pkl.gz');hashes[str(p.relative_to(ROOT))]=sha(p)
      f=pd.DataFrame(frozen(stem)['scores']);cols=['pool','cell','cost','pair','test_year','weight','available','net_bps','n_trades','relation','era'];refs.append(f[cols])
   text.flush();text.detach()
 if target.exists():assert sha(target)==sha(pending),'Existing master differs; preserved'
 else:os.link(pending,target)
 pending.unlink();count('exported_pair_fold_rows',count_rows)
 write(HERE/'audit/copied_reference_hashes.json',hashes)
 daily=pd.concat(port,ignore_index=True);csv_new(TABLES/'portfolio_daily.csv.gz',daily)
 annual=daily.groupby(['pool','cell','cost','world','replicate','variant','test_year'],sort=True)[['net_bps','turnover']].sum().reset_index();csv_new(TABLES/'portfolio_by_year.csv',annual)
 summary=[]
 for key,g in daily.groupby(['pool','cell','cost','world','replicate','variant'],sort=True):
  r=g.sort_values(['test_year','date']).net_bps.to_numpy();cum=np.r_[0.,r.cumsum()];sd=r.std(ddof=1)
  summary.append(dict(zip(['pool','cell','cost','world','replicate','variant'],key),Sharpe=float(np.sqrt(252)*r.mean()/sd) if sd>0 else np.nan,net_bps=float(r.sum()),net_per_year_bps=float(r.sum()/22),turnover=float(g.turnover.sum()),max_drawdown_bps=float((cum-np.maximum.accumulate(cum)).min()),n_days=len(r),n_years=g.test_year.nunique()))
 csv_new(TABLES/'portfolio_summary.csv',summary)
 return pd.concat(frames,ignore_index=True),pd.concat(refs,ignore_index=True),daily,annual
def comparisons(new,refs):
 inf=setup().inference;pred=[];absolute=[];sealed=source_table('evaluation_summary.csv')
 mapping=read(note(PT/'exp_baseline_v1/runtime/clusters.json'))['EXP_G_cap']['pair_cluster']
 for pool in POOLS:
  for cost in COSTS:
   real=new[new.pool.eq(pool)&new.cost.eq(cost)&new.world.eq('real')];base=refs[refs.pool.eq(pool)&refs.cost.eq(cost)]
   null=new[new.pool.eq(pool)&new.cost.eq(cost)&new.world.eq('N2')&new.cell.eq('TD8_tuned')]
   counts=null.groupby(KEYS).net_bps.count();null=null.groupby(KEYS,as_index=False).agg(net_bps=('net_bps','mean'));null.loc[[counts.get(tuple(k),0)!=2 for k in null[KEYS].itertuples(index=False,name=None)],'net_bps']=np.nan
   a=real[real.cell.eq('TD8_tuned')];choices=[('U1','TD8',base[base.cell.eq('TD8')]),('U3','R_TR_aln',base[base.cell.eq('R_TR_aln')]),('U4','B3_TR_aln',base[base.cell.eq('B3_TR_aln')]),('N2','N2',null),('ensemble','TD8_tuned_z',real[real.cell.eq('TD8_tuned_z')])]
   strata=[('pooled',None)]+[(e,a.era.eq(e)) for e in ['E1','E2','E3','E4','E5']]
   for label,mask in strata:
    aa=a if mask is None else a[mask]
    for scheme in ['year']+(['component'] if pool=='P_EXP' else []):
     for id,name,b in choices:
      joined=aa[KEYS+['net_bps','weight']].merge(b[KEYS+['net_bps']],on=KEYS,how='left',validate='one_to_one',suffixes=('_a','_b'),indicator=True);assert joined['_merge'].eq('both').all(),('Comparator missing',pool,cost,name)
      joined['value']=joined.net_bps_a-joined.net_bps_b
      if scheme=='component':joined['cluster']=joined.pair.map(mapping);assert joined.cluster.notna().all()
      out=inf.estimate(joined,scheme=scheme);primary=id=='U1' and pool=='P_RAND' and cost==10 and label=='pooled' and scheme=='year'
      pred.append(dict(pool=pool,cost=cost,era=label,id=id,comparison='TD8_tuned minus '+name,primary=primary,supported=bool(out['estimate']>0 and out['p_one_sided']<=.05),**out))
    for cell in CELLS+REF:
     f=real[real.cell.eq(cell)] if cell in CELLS else base[base.cell.eq(cell)]
     if mask is not None:f=f[f.era.eq(label)]
     if f.empty:continue
     if cell in CELLS:out=inf.estimate(f.assign(value=f.net_bps))
     else:
      out=copied_absolute(sealed,pool,cost,label,cell)
      observed=float(np.average(f.loc[f.available,'net_bps'],weights=f.loc[f.available,'weight']))
      assert abs(observed-out['estimate'])<=1e-9,('Sealed summary does not match copied pair-folds',pool,cost,label,cell)
     absolute.append(dict(pool=pool,cost=cost,era=label,cell=cell,variant='primary',trades_per_pair_year=float(np.average(f.loc[f.available,'n_trades'],weights=f.loc[f.available,'weight'])),source_kind='new result' if cell in CELLS else 'sealed values copied',**out))
 csv_new(TABLES/'contrasts.csv',pred);csv_new(TABLES/'ladder_with_tuned.csv',absolute)
 return pd.DataFrame(pred),pd.DataFrame(absolute)
def policies():
 output={};rows=[];counts=[];lines=['# Frozen tuned TD8 policies','','Actions are table preferences; alignment, lock and forced-exit masks still apply.']
 for pool in POOLS:
  for y in YEARS:
   m=cached(f'outputs/models/{pool}_{y}_TD8_tuned');old=frozen(f'outputs/models/{pool}_{y}_TD8');key=f'{pool}_{y}';q=m['ensemble'];actions=(q[:,1]>q[:,0]).astype(int);oldactions=(old['ensemble'][:,1]>old['ensemble'][:,0]).astype(int)
   output[key]=dict(pool=pool,test_year=y,ensemble=q,models=m['models'],greedy_actions=actions,selected=m['selected'],development_only=True)
   counts.append(dict(pool=pool,test_year=y,era=era(y),never_enters=not bool(actions[::2].any()),old_never_enters=not bool(oldactions[::2].any())))
   lines.extend(['',f'## {key}','','| bin | flat | in trade |','|---|---|---|'])
   for b,bin in enumerate(['<0.5','0.5–1.5','1.5–2.5','>=2.5']):
    enter=bool(actions[b*2]);hold=not bool(actions[b*2+1]);lines.append(f'| {bin} | {"enter" if enter else "stay"} | {"hold" if hold else "exit"} |')
    rows.append(dict(pool=pool,test_year=y,era=era(y),bin=bin,enter=enter,hold=hold))
 write(HERE/'policies.json',output);publish(HERE/'POLICIES.md','\n'.join(lines)+'\n');csv_new(TABLES/'policies_by_fold.csv',rows);csv_new(TABLES/'abstention_by_fold.csv',counts)
 return pd.DataFrame(rows),pd.DataFrame(counts)
