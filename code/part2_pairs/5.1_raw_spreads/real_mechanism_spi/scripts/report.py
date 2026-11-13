"""Deterministic streamed exports and preregistered development inference."""
from common import *
from real_sources import setup,tuned_model
from references import pairs as reference_pairs,daily as reference_daily
from jobs import CELLS
import numpy as np,pandas as pd,csv
KEYS=['pair','test_year']
SMALL=['pool','cell','cost','world','replicate','pair','leg_A','leg_B','test_year','weight','available','net_bps','gross_bps','n_trades','relation','era']
def stream(tasks=None,output=None,master_fields=None):
 output=TABLES if output is None else Path(output);output.mkdir(parents=True,exist_ok=True)
 tasks=[(p,y,1) for p in POOLS for y in YEARS] if tasks is None else tasks
 if master_fields is None:
  with gzip.open(note(SOURCE/'outputs/tables/per_pair_fold_test.csv.gz'),'rt') as f:master_fields=next(csv.reader(f))
 fields=list(dict.fromkeys([*master_fields,*SMALL,'cost_bps_total','applied_cost_bps']))
 target=output/'per_pair_fold_test.csv.gz';pending=target.with_name(target.name+'.pending_'+str(time.time_ns()));real=[];mapped=[];ports=[];rows=0
 with pending.open('xb') as raw:
  with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0,filename='') as gz:
   text=io.TextIOWrapper(gz,encoding='utf8',newline='');writer=csv.DictWriter(text,fieldnames=fields,extrasaction='ignore');writer.writeheader()
   for pool,y,tier in tasks:
    fold=cached(f'runtime/folds/tier{tier}_{pool}_{y}');assert fold is not None,('Unfinished fold',pool,y)
    acc={}
    for stem in fold['chunks']:
     chunk=cached(stem);assert chunk is not None;frame=pd.DataFrame(chunk['scores']);cell=frame.iloc[0].cell;world=frame.iloc[0].world;rep=int(frame.iloc[0].replicate)
     if cell=='L0_plus_TD':mapped.append(frame.reindex(columns=SMALL));continue
     for row in chunk['scores']:
      row=dict(row,deterministic=True,feasible=True,seed=-1,n_seeds=1,in_G=pool=='P_EXP',in_all_eligible=True)
      row['source_row_key']=json.dumps([pool,y,row['pair'],cell,world,rep,row['cost']],separators=(',',':'))
      writer.writerow(row);rows+=1
     if world=='real':real.append(frame.reindex(columns=SMALL))
     for cost,a in chunk['portfolios'].items():
      key=(cell,int(cost),world,rep);v=acc.setdefault(key,dict(dates=a['dates'],weight=0.,**{n:np.zeros(len(a['net'])) for n in ['net','gross','fee','turnover']}))
      assert np.array_equal(v['dates'],a['dates']);v['weight']+=a['weight']
      for n in ['net','gross','fee','turnover']:v[n]+=a[n]
    for (cell,cost,world,rep),a in sorted(acc.items()):
     assert a['weight']>0
     ports.append(pd.DataFrame(dict(date=a['dates'],pool=pool,test_year=y,cell=cell,cost=cost,world=world,replicate=rep,variant='primary',
       **{('net_bps' if n=='net' else 'gross_bps' if n=='gross' else 'cost_bps' if n=='fee' else n):a[n]/a['weight'] for n in ['net','gross','fee','turnover']})))
    checkpoint_guard()
   text.flush();text.detach()
 if target.exists():assert sha(target)==sha(pending),('Existing master differs',target)
 else:os.link(pending,target)
 pending.unlink();count('exported_pair_fold_rows',rows)
 daily=pd.concat(ports,ignore_index=True);csv_new(output/'portfolio_daily_all_worlds.csv.gz',daily)
 csv_new(output/'portfolio_daily.csv.gz',daily.loc[daily.world.eq('real')])
 annual=daily.groupby(['pool','cell','cost','world','replicate','variant','test_year'],sort=True)[['net_bps','turnover']].sum().reset_index();csv_new(output/'portfolio_by_year.csv',annual)
 summary=[]
 for key,g in daily.groupby(['pool','cell','cost','world','replicate','variant'],sort=True):
  r=g.sort_values('date').net_bps.to_numpy();wealth=np.r_[0.,r.cumsum()];sd=r.std(ddof=1)
  summary.append(dict(zip(['pool','cell','cost','world','replicate','variant'],key),net_bps=r.sum(),net_per_year_bps=r.sum()/g.test_year.nunique(),
    Sharpe=np.sqrt(252)*r.mean()/sd if sd>0 else np.nan,turnover=g.turnover.sum(),max_drawdown_bps=np.min(wealth-np.maximum.accumulate(wealth)),n_years=g.test_year.nunique(),n_days=len(g)))
 csv_new(output/'portfolio_summary.csv',summary)
 return pd.concat(real,ignore_index=True),pd.concat(mapped,ignore_index=True),daily,annual
def spi_tables():
 tests=[];deploy=[]
 for pool in POOLS:
  for y in YEARS:
   for world,rep in WORLDS:
    for learner in ['SPI_all','SPI_TD']:
     g=cached(f'runtime/gates/{pool}_{y}_{learner}_{world}_{rep}');assert g is not None
     tests.extend(dict(r,pool=pool,test_year=y,era=era(y),world=world,replicate=rep,learner=learner) for r in g['tests'])
     for r in g['deployed']:
      names=[setup().policies.SWITCHES[b] for b in range(10) if r['deployed_bits']&(1<<b)]
      deploy.append(dict(r,pool=pool,test_year=y,era=era(y),world=world,replicate=rep,learner=learner,accepted_switches=','.join(names)))
 csv_new(TABLES/'spi_tests.csv',tests);csv_new(TABLES/'spi_deployed.csv',deploy);return pd.DataFrame(tests),pd.DataFrame(deploy)
def policy_tables():
 policies={};diag=[];trajectories=[];lines=['# Frozen Tier 1 policies','','Actions remain subject to the unchanged alignment, lock, stop/cap and annual-close masks. E0 denotes E_0, etc.']
 for pool in POOLS:
  for y in YEARS:
   m=cached(f'outputs/models/{pool}_{y}_TD8_C01g1');assert m is not None
   q=m['ensemble'];policies[f'{pool}_{y}_TD8_C01g1']=dict(pool=pool,test_year=y,cell='TD8_C01g1',TD_ens=q,seed_tables=m['models'],visits=m['visits'],epochs=m['epochs'],selected=m['selected'])
   lines.extend(['',f'## {pool} / {y} — TD8_C01g1','','| bin | flat | in trade |','|---|---|---|'])
   for b in range(4):lines.append(f'| {b} | {"enter" if q[2*b,1]>q[2*b,0] else "stay"} | {"exit" if q[2*b+1,1]>q[2*b+1,0] else "hold"} |')
   for cell,model in [('TD8_C01g1',m),('TD8_tuned',tuned_model(pool,y))]:
    qq=model['ensemble']
    for state in range(8):
     for action in range(2):
      diag.append(dict(pool=pool,test_year=y,era=era(y),cell=cell,seed='ensemble',state=state,bin=state//2,position='in-trade' if state%2 else 'flat',action=action,Q=qq[state,action],delta_Q=qq[state,0]-qq[state,1] if state%2 else qq[state,1]-qq[state,0],hold_below_exit_all_bins=bool(np.all(qq[1::2,0]<qq[1::2,1]))))
   for j in range(10):
    f=cached(f'runtime/fits/{pool}_{y}_TD8_C01g1_{j}');assert f is not None
    for state in range(8):
     for action in range(2):
      a=f['diagnostics'][state,action]
      diag.append(dict(pool=pool,test_year=y,era=era(y),cell='TD8_C01g1',seed=j,state=state,bin=state//2,position='in-trade' if state%2 else 'flat',action=action,Q=f['q'][state,action],visits=int(f['visits'][state,action]),W=a[0],V=a[1],ESS=a[2],epochs=f['meta']['epochs'],converged=f['meta']['converged'],stopping_epoch=f['meta']['convergence_epoch'],last_epoch_weight=a[3],tail_episodes_weight=a[4],tail_pairs_weight=a[5],target_mean=a[6],target_sd=a[7]))
   if y==2004:
    tuned=cached(f'runtime/acceptance/tuned_trajectories/{pool}_{y}');assert tuned is not None
    for cell,fits in [('TD8_C01g1',[cached(f'runtime/fits/{pool}_{y}_TD8_C01g1_{j}') for j in range(10)]),('TD8_tuned',[f for f in tuned['fits']])]:
     for j,f in enumerate(fits):
      for epoch,table in enumerate(f['trajectory'],1):
       for state in range(8):
        for action in range(2):trajectories.append(dict(pool=pool,test_year=y,cell=cell,seed=j,epoch=epoch,state=state,action=action,Q=table[state,action],source='production new fit' if cell=='TD8_C01g1' else 'exact acceptance reproduction; production reuses sealed table'))
 write(HERE/'policies.json',policies);publish(HERE/'POLICIES.md','\n'.join(lines)+'\n');csv_new(TABLES/'q_diagnostics.csv',diag);csv_new(TABLES/'q_trajectories.csv',trajectories)
 return pd.DataFrame(diag),pd.DataFrame(trajectories)
def mapping_check(mapped,refs):
 expected=refs.loc[refs.cell.eq('TD8_tuned')];keys=['pool','cost','world','replicate','pair','test_year']
 joined=mapped.merge(expected,on=keys,suffixes=('_new','_old'),how='outer',indicator=True,validate='one_to_one');assert joined['_merge'].eq('both').all()
 for col in ['net_bps','gross_bps','n_trades']:
  a=joined[col+'_new'].to_numpy(float);b=joined[col+'_old'].to_numpy(float);assert np.array_equal(np.isnan(a),np.isnan(b));assert np.nanmax(np.abs(a-b))<=1e-9
 write(HERE/'audit/full_mapping_check.json',dict(status='PASS',pair_fold_cost_world_rows=len(joined),metrics=['net_bps','gross_bps','n_trades'],tolerance=1e-9,new_trial=False))
def comparisons(new,refs,deploy,diagnostics):
 inf=setup().inference;pred=[];ladder=[];annual_rows=[]
 mapping=read(note(PT/'exp_baseline_v1/runtime/clusters.json'))['EXP_G_cap']['pair_cluster']
 for pool in POOLS:
  for cost in COSTS:
   a=new.loc[new.pool.eq(pool)&new.cost.eq(cost)];old=refs.loc[refs.pool.eq(pool)&refs.cost.eq(cost)&refs.world.eq('real')]
   l0=old[old.cell.eq('R_TR_aln')];tuned=old[old.cell.eq('TD8_tuned')]
   for label in ['pooled','E1','E2','E3','E4','E5']:
    aa=a if label=='pooled' else a[a.era.eq(label)];oo=old if label=='pooled' else old[old.era.eq(label)]
    for cell in ['R_TR_aln','B3_TR_aln','TD8','TD8_tuned',*CELLS]:
     f=aa[aa.cell.eq(cell)] if cell in CELLS else oo[oo.cell.eq(cell)];assert len(f)
     estimate=inf.estimate(f.assign(value=f.net_bps));good=f[np.isfinite(f.net_bps)]
     ladder.append(dict(pool=pool,cost=cost,era=label,cell=cell,variant='primary',source_kind='new cell' if cell in CELLS else 'sealed values with unchanged bootstrap',trades_per_pair_year=np.average(good.n_trades,weights=good.weight),**estimate))
    for id,left,right,margin in [('M1',tuned,aa[aa.cell.eq('TD8_C01g1')],0),('S1',aa[aa.cell.eq('SPI_all')],l0,10),('S2',aa[aa.cell.eq('SPI_all')],l0,0)]:
     if label!='pooled':left=left[left.era.eq(label)];right=right[right.era.eq(label)]
     joined=inf.paired(left,right)
     for scheme in ['year']+(['component'] if pool=='P_EXP' else []):
      f=joined.copy()
      if scheme=='component':f['cluster']=f.pair.map(mapping);assert f.cluster.notna().all()
      out=inf.estimate(f.assign(value=f.value+margin),scheme=scheme)
      for k in ['estimate','ci95_low','ci95_high']:out[k]-=margin
      pred.append(dict(id=id,pool=pool,cost=cost,era=label,margin_bps=margin,primary=id in ['M1','S1'] and pool=='P_RAND' and cost==10 and label=='pooled' and scheme=='year',**out))
   deltas=[]
   for cell in ['SPI_all','SPI_TD','TD8_tuned','TD8_C01g1']:
    f=a[a.cell.eq(cell)] if cell in CELLS else tuned;joined=inf.paired(f,l0)
    for y,g in joined.groupby('test_year',sort=True):deltas.append(dict(pool=pool,cost=cost,test_year=y,era=era(y),cell=cell,delta_vs_L0=np.average(g.loc[np.isfinite(g.value),'value'],weights=g.loc[np.isfinite(g.value),'weight'])))
   annual_rows.extend(deltas)
   checkpoint_guard()
 primary=[r for r in pred if r['primary']];assert len(primary)==2
 _,adjusted=setup().stats.holm([r['p_one_sided'] for r in primary])
 for row,p in zip(primary,adjusted):row['p_Holm']=float(p);row['supported_primary']=bool(p<=.05 and row['estimate']>-row['margin_bps'])
 pred.extend([dict(id='M4',status='NOT_RUN_TIER2_NOT_AUTHORIZED',primary=False)])
 csv_new(TABLES/'predictions.csv',pred);csv_new(TABLES/'ladder_with_new.csv',ladder);csv_new(TABLES/'annual_differences.csv',annual_rows)
 checks=[];annual=pd.DataFrame(annual_rows)
 for pool in POOLS:
  for cost in COSTS:
   f=annual.loc[annual.pool.eq(pool)&annual.cost.eq(cost)].pivot(index='test_year',columns='cell',values='delta_vs_L0')
   d=deploy.loc[deploy.pool.eq(pool)&deploy.cost.eq(cost)&deploy.learner.eq('SPI_all')&deploy.primary_lag]
   q=diagnostics.loc[diagnostics.pool.eq(pool)&diagnostics.cell.eq('TD8_C01g1')&diagnostics.seed.eq('ensemble')].drop_duplicates('test_year')
   a=new.loc[new.pool.eq(pool)&new.cost.eq(cost)];old=refs.loc[refs.pool.eq(pool)&refs.cost.eq(cost)&refs.world.eq('real')&refs.cell.eq('TD8_tuned')]
   checks.append(dict(pool=pool,cost=cost,id='M2_M3_S3_S5',C01_never_entering_folds=int(a.loc[a.cell.eq('TD8_C01g1')].groupby('test_year').n_trades.sum().eq(0).sum()),tuned_never_entering_folds=int(old.groupby('test_year').n_trades.sum().eq(0).sum()),C01_hold_below_exit_all_bins=int(q.hold_below_exit_all_bins.sum()),
    annual_sd_ratio=f.SPI_all.std(ddof=1)/f.TD8_tuned.std(ddof=1) if f.TD8_tuned.std(ddof=1)>0 else np.nan,
    real_deviation_rate=d.loc[d.world.eq('real'),'deployed_bits'].ne(0).mean(),N2_deviation_rate=d.loc[d.world.eq('N2'),'deployed_bits'].ne(0).mean(),real_gate_worlds=int(d.world.eq('real').sum()),N2_gate_worlds=int(d.world.eq('N2').sum())))
 csv_new(TABLES/'descriptive_checks.csv',checks)
 return pd.DataFrame(pred),pd.DataFrame(ladder),annual,pd.DataFrame(checks)
def exposure(daily,old_daily):
 out=[];s=setup();market=old_daily.loc[old_daily.pool.eq('P_RAND')&old_daily.cell.eq('SPY')&old_daily.cost.eq(10)&old_daily.variant.eq('primary')].set_index('date').net_bps
 assert market.index.is_unique
 for pool in POOLS:
  for cost in COSTS:
   f=daily.loc[daily.pool.eq(pool)&daily.cost.eq(cost)&daily.world.eq('real')].pivot(index='date',columns='cell',values='net_bps')
   base=old_daily.loc[old_daily.pool.eq(pool)&old_daily.cost.eq(cost)&old_daily.cell.eq('R_TR_aln')&old_daily.variant.eq('primary')].set_index('date').net_bps
   f['SPI_all_minus_L0']=f.SPI_all-base;f['SPY']=market;f.index=pd.to_datetime(f.index);monthly=f.groupby(f.index.to_period('M')).sum(min_count=1)
   for cell in [*CELLS,'SPI_all_minus_L0']:
    r=s.methods.ols_hac(monthly[cell],monthly.SPY,3,12);count('regressions_run')
    out.append(dict(pool=pool,cost=cost,cell=cell,frequency='monthly',alpha_monthly_bps=r.get('alpha_annual_bps',np.nan)/12,descriptive=True,**r))
 csv_new(TABLES/'exposure.csv',out);return pd.DataFrame(out)
def accounting(new_daily,new_annual,old_daily):
 from references import baseline_accounting
 s=setup();source=s.sources.original_inference();old=baseline_accounting();rows=[];cumulative=549
 for pool in ['GLOBAL',*POOLS]:
  stem=f'runtime/report/accounting_{pool}';prior=cached(stem)
  if prior is not None:rows.extend(prior);continue
  names=[n for n in old['cells'] if pool=='GLOBAL' or n.startswith(pool+'|')];indices=[old['cells'].index(n) for n in names];matrix=[a for a in np.asarray(old['annual_matrix'])[indices]]
  current=new_daily.loc[new_daily.world.eq('real')].copy();current['name']=current.pool+'|'+current.cell+'|'+current.cost.astype(int).astype(str)+'|primary'
  prior_daily=old_daily.loc[old_daily.name.isin(names)];both=pd.concat([prior_daily,current],ignore_index=True)
  if pool!='GLOBAL':both=both[both.pool.eq(pool)]
  series=[];series_names=[]
  for name,g in both.groupby('name',sort=True):assert not g.date.duplicated().any();series.append(g.set_index('date').net_bps);series_names.append(name)
  daily_matrix=pd.concat(series,axis=1);daily_matrix.columns=series_names;complete=daily_matrix.dropna();SR=complete.mean()/complete.std(ddof=1);valid=np.isfinite(SR)
  assert valid.sum()>=2;best=SR[valid].idxmax();x=complete[best].to_numpy();z=x-x.mean();m2=np.mean(z*z)
  result=[]
  for N in [len(series_names),cumulative]:
   r=source.dsr(float(SR[best]),len(x),float(np.mean(z**3)/m2**1.5),float(np.mean(z**4)/m2**2),float(SR[valid].var(ddof=1)),N);count('DSR_evaluations')
   result.append(dict(kind='DSR',pool=pool,best_cell_for_accounting_only=best,cumulative_trials=cumulative,selects_no_strategy=True,**r))
  a=new_annual.loc[new_annual.world.eq('real')];a=a if pool=='GLOBAL' else a[a.pool.eq(pool)]
  for key,g in a.groupby(['pool','cell','cost'],sort=True):
   values=g.set_index('test_year').net_bps.reindex(YEARS);assert np.isfinite(values).all();matrix.append(values.to_numpy());names.append('|'.join(map(str,[*key,'primary'])))
  # The sealed old matrix uses IPW pair-year means, rather than portfolio
  # daily sums. Replace new rows with the same pair-year construction.
  real=cached('runtime/report/real_small');real=real if pool=='GLOBAL' else real[real.pool.eq(pool)]
  for name_index in range(len(indices),len(names)):
   p,cell,cost,_=names[name_index].split('|');f=real.loc[real.pool.eq(p)&real.cell.eq(cell)&real.cost.eq(int(cost))]
   matrix[name_index]=np.array([np.average(g.loc[np.isfinite(g.net_bps),'net_bps'],weights=g.loc[np.isfinite(g.net_bps),'weight']) for _,g in f.groupby('test_year',sort=True)])
  assert len(matrix)==len(series_names);pbo,_=source.pbo(np.asarray(matrix));count('CSCV_splits',pbo['n_splits'])
  result.append(dict(kind='PBO',pool=pool,PBO=pbo['PBO'],n_splits=pbo['n_splits'],performance_columns=len(names),cumulative_trials=cumulative,status='all sealed available annual cells plus new cells; no imputation for unexecuted screens'))
  save(stem,result);rows.extend(result);checkpoint_guard()
 csv_new(TABLES/'dsr_pbo.csv',rows);return pd.DataFrame(rows)
def run():
 refs=reference_pairs();old_daily=reference_daily();new,mapped,daily,annual=stream();save('runtime/report/real_small',new)
 mapping_check(mapped,refs);tests,deploy=spi_tables();diag,traj=policy_tables();pred,ladder,differences,checks=comparisons(new,refs,deploy,diag);exp=exposure(daily,old_daily)
 account=accounting(daily,annual,old_daily)
 write(HERE/'audit/reference_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in [SOURCE/'outputs/tables/per_pair_fold_test.csv.gz',SOURCE/'outputs/tables/portfolio_daily.csv.gz',TUNED/'outputs/tables/per_pair_fold_test.csv.gz',TUNED/'outputs/tables/portfolio_daily.csv.gz',TUNED/'runtime/accounting/CSCV_GLOBAL.pkl.gz']})
 from plots import render
 render(ladder,diag,traj,deploy,daily,old_daily,differences,new,refs)
 primary=pred.loc[pred.primary.eq(True)];m1=primary[primary.id.eq('M1')].iloc[0];s1=primary[primary.id.eq('S1')].iloc[0];s2=pred.loc[pred.id.eq('S2')&pred.pool.eq('P_RAND')&pred.cost.eq(10)&pred.era.eq('pooled')&pred.scheme.eq('year')].iloc[0]
 c=checks.loc[checks.pool.eq('P_RAND')&checks.cost.eq(10)].iloc[0];summary=csv_read(TABLES/'portfolio_summary.csv');strategy=summary.loc[summary.pool.eq('P_RAND')&summary.world.eq('real')&summary.cost.isin([10,20])]
 lines=[f"M1: tuned − C01 {m1.estimate:.2f} [{m1.ci95_low:.2f}, {m1.ci95_high:.2f}] bps/pair-year; p={m1.p_one_sided:.5f}, Holm={m1.p_Holm:.5f}; never-entering folds C01/tuned {int(c.C01_never_entering_folds)}/{int(c.tuned_never_entering_folds)}.",
 'M4: NOT RUN — Tier 2 is not authorized by the selected command.',
 f"S1/S2: SPI_all − L0 {s1.estimate:.2f} [{s1.ci95_low:.2f}, {s1.ci95_high:.2f}]; non-inferiority p={s1.p_one_sided:.5f}, Holm={s1.p_Holm:.5f}; descriptive superiority p={s2.p_one_sided:.5f}.",
 f"S3–S5: annual sd ratio {c.annual_sd_ratio:.3f}; real/N2 deviation {c.real_deviation_rate:.1%}/{c.N2_deviation_rate:.1%}; exact accepted switches in spi_deployed.csv.",
 'SPI_TD/SPI_all at 10 and 20 bps: '+ '; '.join(f"{r.cell} c={int(r.cost)} annual net={r.net_per_year_bps:.2f}, Sharpe={r.Sharpe:.3f}, turnover={r.turnover:.2f}" for r in strategy.itertuples())+'; L0⊕TD reproduces sealed tuned values (audit/full_mapping_check.json).',
 'DSR/PBO: '+ '; '.join(f"{r.pool} {r.kind}={getattr(r,'DSR',np.nan) if r.kind=='DSR' else r.PBO:.5f}" for r in account.loc[(account.kind.eq('PBO'))|(account.kind.eq('DSR')&account.N.eq(549))].itertuples())]
 publish(HERE/'SUMMARY.txt','\n'.join(lines)+'\n')
 text='# Tier 1 results — development evidence\n\n'+'\n\n'.join(f'{i+1}. {line}' for i,line in enumerate(lines))+'\n\n'
 text+='The primary family is M1/S1 with Holm adjustment. All other comparisons, exposures, eras, gate rates and cost sensitivities are descriptive. A confidence interval that includes zero is not evidence of equality. These reused years do not provide a fresh confirmation test.\n\n'
 text+='The gate deploys only primary-lag accepted switches; no test-year value selects policies. The final DSR uses 549 cumulative trials. PBO includes every column of the sealed prior executed-cell matrix plus all new cell/pool/cost columns; the two prior unexecuted screen trials affect DSR multiplicity but have no return series and are not imputed into PBO.\n\n'
 text+='Tuned Q trajectories come from instrumented acceptance reproductions whose seed tables equal the sealed tables exactly. Production uses the original sealed tuned models; these diagnostic reproductions do not introduce a new strategy or tune any setting.\n\n'
 text+='Limitations: reused test years; one forced-exit environment; coarse bins; Newey–West on about 756 training days after cross-sectional averaging; omitted borrow/rebalancing costs; survivorship and current GICS. See exposure.csv for monthly beta and alpha, and accepted switches by fold in spi_deployed.csv.\n'
 publish(HERE/'RESULTS.md',text);return lines
def worker(_):return run()
