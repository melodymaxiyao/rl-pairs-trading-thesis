"""Post-hoc stratification of copied outcomes; all screens fixed from train only."""
from common import *
import numpy as np,pandas as pd
from scipy.stats import rankdata
from inference import *
from outcomes import read_outcomes,outcome_matrix

KEY=['pool','test_year','pair','world','replicate','window']
USE=['ADF_c','ADF_ct','KPSS_c','KPSS_ct','ECM','theta_5','theta_21','theta_63','VR_21','VR_63','EG_c','EG_ct','Johansen','ADF_c_AIC','ADF_ct_AIC']
def compact(f):
 identity=KEY+['weight','era','relation']
 base=f[identity].drop_duplicates();assert not base.duplicated(KEY).any()
 for name in USE:
  g=f[f.statistic_name.eq(name)]
  if g.empty:continue
  cols=[c for c in ['statistic','nominal_p','empirical_p','empirical_q','theta','alpha','alpha_se','half_life','half_life_status','nominal_reject_5','critical_5'] if c in g and g[c].notna().any()]
  g=g[KEY+cols].rename(columns={c:name+'__'+c for c in cols})
  base=base.merge(g,on=KEY,how='left',validate='one_to_one')
 return base

def export_stats():
 """Stream the large long-form table; only compact analysis frames stay in RAM."""
 import io
 dest=TABLES/'residual_stats.csv.gz';tmp=dest.with_name(dest.name+'.pending_'+str(time.time_ns()));frames=[];missing=[];first=True;columns=None
 # Every chunk has a frozen union schema to avoid ragged CSV rows.
 for y in YEARS:
  for pool in POOLS:
   v=cached(f'runtime/pool_folds/{pool}_{y}');f=v['frame'];columns=list(f.columns) if columns is None else list(dict.fromkeys([*columns,*f.columns]));missing.extend(v['unavailable'])
 # Re-open one pool-fold at a time to bound memory.
 if not dest.exists():
  with tmp.open('xb') as raw:
   with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0,filename='') as z:
    with io.TextIOWrapper(z,encoding='utf-8',newline='') as text:
     for y in YEARS:
      for pool in POOLS:
       f=cached(f'runtime/pool_folds/{pool}_{y}')['frame'].reindex(columns=columns)
       f.to_csv(text,index=False,header=first,float_format='%.17g');first=False
  os.link(tmp,dest);tmp.unlink()
 for pool in POOLS:
  parts=[]
  for y in YEARS:parts.append(compact(cached(f'runtime/pool_folds/{pool}_{y}')['frame']))
  frames.append(pd.concat(parts,ignore_index=True))
 csv_new(TABLES/'unavailable.csv',pd.DataFrame(missing,columns=['pool','test_year','pair','world','replicate','reason','weight']))
 return pd.concat(frames,ignore_index=True)

def long_screens(wide):
 out=[]
 for screen,name in [('RC','ADF_ct'),('RL','ADF_c'),('RP','theta_21')]:
  f=wide[KEY+['weight','era','relation',name+'__statistic',name+'__empirical_p',name+'__empirical_q']].rename(columns={name+'__statistic':'statistic',name+'__empirical_p':'empirical_p',name+'__empirical_q':'empirical_q'});f['statistic_name']=name
  primary=select_screen(f[f.pool.eq('P_RAND')],screen)
  fallback_years=set((str(r.world),int(r.replicate),int(r.test_year)) for r in primary[primary.fallback][['world','replicate','test_year']].drop_duplicates().itertuples())
  for pool,g in f.groupby('pool',sort=True):out.append(primary if pool=='P_RAND' else select_screen(g,screen,fallback_years))
 return pd.concat(out,ignore_index=True)

def stationarity_rate(wide,name):
 if name=='Johansen':return wide[name+'__nominal_reject_5'].fillna(False) if name+'__nominal_reject_5' in wide else None
 if name+'__nominal_p' not in wide:return None
 if name!='Johansen' and not np.isfinite(wide[name+'__nominal_p']).any():return None
 if name.startswith('KPSS'):return wide[name+'__statistic']>wide[name+'__critical_5']
 if name=='Johansen':return wide[name+'__nominal_reject_5'].fillna(False)
 return wide[name+'__nominal_p'].le(.05)
def paired_stat(wide,column,null_world='N2'):
 keys=['pool','test_year','pair','window'];r=wide[wide.world.eq('real')][keys+['weight',column]].copy();n=wide[wide.world.eq(null_world)].groupby(keys)[column].agg(['mean','count']).reset_index()
 r=r.merge(n,on=keys,how='left',validate='one_to_one');r['value']=r[column]-r['mean'];r.loc[r['count'].ne(2),'value']=np.nan
 return r

def existence(wide):
 f=wide[wide.window.eq('T')].copy();out=[];pi=[]
 for name in USE:
  if name+'__empirical_p' not in f:continue
  reject=stationarity_rate(f,name)
  if reject is not None:
   key='reject__'+name;f[key]=reject.astype(float);f.loc[~np.isfinite(f[name+'__statistic']),key]=np.nan
   for null_world in ['N2','N1']:
    matched=paired_stat(f,key,null_world)
    for pool,g in matched.groupby('pool'):out.append(dict(pool=pool,statistic=name,null_world=null_world,question='R1' if name=='ADF_ct' and pool=='P_RAND' and null_world=='N2' else 'descriptive',**mean(g)))
  for pool,g in f[f.world.eq('real')].groupby('pool'):
   for label,sub in [('POOLED',g),*[(e,v) for e,v in g.groupby('era')],*[(str(y),v) for y,v in g.groupby('test_year')]]:
    p=sub[name+'__empirical_p'].to_numpy();weights=sub.weight.to_numpy();value=1-pi0(p,weights);sub=sub.copy();sub['value']=(p>.5).astype(float);sub.loc[~np.isfinite(p),'value']=np.nan
    years,W=blocks(sub.test_year);ok=sub[np.isfinite(sub.value)];num=np.array([(ok.loc[ok.test_year.eq(y),'value']*ok.loc[ok.test_year.eq(y),'weight']).sum() for y in years]);den=np.array([ok.loc[ok.test_year.eq(y),'weight'].sum() for y in years]);b=1-np.minimum(1,2*np.divide(W@num,W@den,out=np.full(len(W),np.nan),where=W@den>0));count('bootstrap_draws',len(W))
    ci=describe(value,b,len(years),len(ok));pi.append(dict(pool=pool,statistic=name,group=label,one_minus_pi0=value,unweighted_one_minus_pi0=1-pi0(p),ci95_low=ci['ci95_low'],ci95_high=ci['ci95_high'],folds=len(years),ci_status='year-block' if len(years)>1 else 'NA: one fold cannot identify year-block uncertainty',pair_folds=len(ok)))
 # R1b is a paired difference, with the year ratio recomputed in each draw.
 if 'reject__ADF_ct' in f:
  f['difference']=f['reject__ADF_ct']-f['reject__ADF_c'];g=paired_stat(f,'difference');g=g[g.pool.eq('P_RAND')];out.append(dict(pool='P_RAND',statistic='ADF_ct_minus_ADF_c',question='R1b',**mean(g)))
 return pd.DataFrame(out),pd.DataFrame(pi)

def weighted_rank_corr(a,b,w):
 ok=np.isfinite(a)&np.isfinite(b)&np.isfinite(w)&(w>0)
 if sum(ok)<3:return np.nan
 x=rankdata(np.asarray(a)[ok]);y=rankdata(np.asarray(b)[ok]);w=np.asarray(w)[ok];x=x-np.average(x,weights=w);y=y-np.average(y,weights=w);den=np.sqrt(np.sum(w*x*x)*np.sum(w*y*y))
 return float(np.sum(w*x*y)/den) if den>0 else np.nan

def persistence(wide,screens):
 out=[];keys=['pool','test_year','pair','world','replicate']
 for screen in ['RC','RP']:
  g=screens[screens.screen.eq(screen)];a=g[g.window.eq('A')][keys+['weight','pass']].rename(columns={'pass':'pass_A'});b=g[g.window.eq('B')][keys+['pass']].rename(columns={'pass':'pass_B'});z=a.merge(b,on=keys,validate='one_to_one');z['value']=z.pass_B.astype(float);z['weight']=z.weight/np.where(z.world.ne('real'),2,1)
  for pool,v in z.groupby('pool'):
   r=v.world.eq('real');p=v.pass_A
   for null_world in ['N2','N1']:
    n=v.world.eq(null_world);terms=[(1,r&p,'value'),(-1,r&~p,'value'),(-1,n&p,'value'),(1,n&~p,'value')]
    out.append(dict(kind='pass_rate_DiD',screen=screen,pool=pool,null_world=null_world,question='R2' if screen=='RC' and pool=='P_RAND' and null_world=='N2' else 'descriptive',**contrast(v,terms)))
   for world in ['real','N2','N1']:
    for passing in [True,False]:
     v2=v[v.world.eq(world)&v.pass_A.eq(passing)];out.append(dict(kind='B_pass_given_A',screen=screen,pool=pool,world=world,A_pass=passing,**mean(v2)))
 for variable in ['ECM__statistic','theta_21__theta','theta_21__statistic','ADF_c__statistic','ADF_ct__statistic']:
  a=wide[wide.window.eq('A')][keys+['weight',variable]];b=wide[wide.window.eq('B')][keys+[variable]];z=a.merge(b,on=keys,validate='one_to_one',suffixes=('_A','_B'));corr=[]
  for key,g in z.groupby(['pool','test_year','world','replicate']):
   corr.append(dict(zip(['pool','test_year','world','replicate'],key),weight=g.weight.sum(),value=weighted_rank_corr(g[variable+'_A'].to_numpy(),g[variable+'_B'].to_numpy(),g.weight.to_numpy())))
  c=pd.DataFrame(corr)
  for (pool,world),g in c.groupby(['pool','world']):out.append(dict(kind='IPW_mean_within_fold_Spearman',statistic=variable,pool=pool,world=world,**mean(g)))
  for pool,g in c.groupby('pool'):
   n=g[g.world.eq('N2')].groupby('test_year').value.mean();r=g[g.world.eq('real')].copy();r['value']=r.value-r.test_year.map(n);out.append(dict(kind='rank_persistence_real_minus_N2',statistic=variable,pool=pool,**mean(r)))
 return pd.DataFrame(out)

def joined_outcomes(screens,outcomes,null_world='N2'):
 f=outcome_matrix(outcomes,null_world);s=screens[screens.world.eq('real')&screens.window.eq('T')];keys=['pool','test_year','pair'];s=s[keys+['screen','weight','pass','BH_pass','fallback']].rename(columns={'weight':'screen_weight'})
 z=f.merge(s,on=keys,how='left',validate='many_to_many');assert len(z)==3*len(f) and z.screen.notna().all()
 np.testing.assert_array_equal(z.weight.to_numpy(),z.screen_weight.to_numpy());return z

def did(z):
 out=[]
 for (pool,cell,screen),g in z.groupby(['pool','cell','screen']):
  valid=np.isfinite(g.excess_net);p=g['pass'];r=contrast(g,[(1,p,'excess_net'),(-1,~p,'excess_net')])
  null_world=str(g.null_world.iloc[0]);question='R3' if (pool,cell,screen,null_world)==('P_RAND','R_TR_aln','RC','N2') else 'R4' if (pool,cell,screen,null_world)==('P_RAND','TD8_tuned','RP','N2') else 'R6'
  out.append(dict(question=question,comparison='real_minus_'+null_world+'_pass_minus_fail',null_world=null_world,pool=pool,cell=cell,screen=screen,pass_pair_folds=int(sum(p&valid)),fail_pair_folds=int(sum(~p&valid)),missing_pair_folds=int(sum(~valid)),**r))
 # R5 explicitly uses real-only incremental value; dose response separately
 # uses the real-minus-N2 incremental value.
 if str(z.null_world.iloc[0])!='N2':return pd.DataFrame(out)
 keys=['pool','test_year','pair','screen'];a=z[z.cell.eq('TD8_tuned')][keys+['net_bps','weight','pass']];b=z[z.cell.eq('R_TR_aln')][keys+['net_bps']];v=a.merge(b,on=keys,validate='one_to_one',suffixes=('_td','_l0'));v['value']=v.net_bps_td-v.net_bps_l0
 for (pool,screen),g in v.groupby(['pool','screen']):out.append(dict(question='R5',comparison='real_TD8_tuned_minus_L0_pass_minus_fail',pool=pool,cell='TD8_tuned_minus_L0',screen=screen,**contrast(g,[(1,g['pass'],'value'),(-1,~g['pass'],'value')])))
 return pd.DataFrame(out)

def deciles(series,pairs,higher=True):
 f=pd.DataFrame(dict(value=series,pair=pairs));ok=f.value.notna();ordered=f[ok].sort_values(['value','pair'],ascending=[True,True],kind='stable');out=pd.Series(np.nan,index=f.index)
 if len(ordered):out.loc[ordered.index]=np.minimum(10,1+np.floor(np.arange(len(ordered))*10/len(ordered))).astype(int)
 return out

def dose_response(wide,outcomes):
 f=wide[wide.world.eq('real')&wide.window.eq('T')].copy();metric={'ADF_ct':'ADF_ct__statistic','theta_21_t':'theta_21__statistic','half_life':'ECM__half_life','one_minus_KPSS_p':'KPSS_c__nominal_p'};o=outcome_matrix(outcomes);out=[];keys=['pool','test_year','pair']
 for name,column in metric.items():
  b=f[keys+['weight',column]].copy();b['binning_value']=1-b[column] if name=='one_minus_KPSS_p' else b[column];b['decile']=np.nan
  for _,g in b.groupby(['pool','test_year']):b.loc[g.index,'decile']=deciles(g.binning_value,g.pair)
  for pool,g in b.groupby('pool'):
   cells={}
   for cell in ['R_TR_aln','TD8_tuned']:
    q=o[o.pool.eq(pool)&o.cell.eq(cell)];v=g.merge(q[keys+['excess_net']],on=keys,validate='one_to_one');v['value']=v.excess_net;cells[cell]=v
   a=cells['TD8_tuned'];l=cells['R_TR_aln'];v=a.merge(l[keys+['value']],on=keys,validate='one_to_one',suffixes=('','_l0'));v['value']=v.value-v.value_l0;cells['TD8_tuned_minus_L0']=v
   for cell,v in cells.items():
    for decile,g2 in v.groupby('decile'):out.append(dict(pool=pool,statistic=name,cell=cell,decile=int(decile),bin_direction='ascending statistic within fold; pair alphabetical tie-break',undefined_train_statistic=int(g.decile.isna().sum()),**mean(g2)))
 return pd.DataFrame(out)

def final_report():
 ready=HERE/'audit/report_ready.json'
 if ready.exists():return read(ready)
 wide=export_stats();outcomes=read_outcomes();csv_new(TABLES/'copied_outcomes.csv.gz',outcomes)
 screens=long_screens(wide);csv_new(TABLES/'screens.csv',screens)
 ex,pi=existence(wide);pe=persistence(wide,screens);joined=joined_outcomes(screens,outcomes);di=pd.concat([did(joined),did(joined_outcomes(screens,outcomes,'N1'))],ignore_index=True);dose=dose_response(wide,outcomes)
 family=[]
 for q,table in [('R1',ex),('R2',pe),('R3',di),('R4',di)]:
  rows=table[table.question.eq(q)];assert len(rows)==1,(q,len(rows));family.append(dict(rows.iloc[0],id=q))
 adjusted=holm([r['p_one_sided'] for r in family])
 for r,p in zip(family,adjusted):r.update(holm_p=float(p),supported=bool(p<=.05 and r['estimate']>0),family='R1-R4; 4 hypotheses; alpha=.05')
 for table in [ex,pe,di]:
  table['holm_p']=np.nan;table['supported']=False
  for r in family:
   if 'question' in table:mask=table.question.eq(r['id']);table.loc[mask,'holm_p']=r['holm_p'];table.loc[mask,'supported']=r['supported']
 pred=[*family,*[dict(r,id=r['question'],family='exploratory; not Holm adjusted') for r in di[di.question.isin(['R5','R6'])].to_dict('records')]]
 for name,f in [('existence',ex),('fdr_pi0',pi),('persistence',pe),('did_results',di),('dose_response',dose),('predictions',pd.DataFrame(pred))]:csv_new(TABLES/(name+'.csv'),f)
 candidates=[r for r in family if r['id'] in ['R3','R4'] and r['supported']]
 selected=sorted(candidates,key=lambda r:(-r['estimate'],r['id']))[0] if candidates else None
 transfer=dict(add_P_RES=bool(selected),screen=selected.get('screen') if selected else None,selection_basis='larger supported preregistered R3/R4 DiD after Holm',selected_statistic=selected,preregistration_sha256=sha(HERE/'preregistration.json'),future_task='TD8_tuned and SPI; future predictions require a separate preregistration' if selected else 'No residual-screened pool added',evidence='post-hoc stratification of reused sealed test years; development evidence')
 write(HERE/'pool_transfer.json',transfer);write(TABLES/'pool_transfer.json',transfer)
 import hindsight as hs,power,accounting
 hindsight=hs.run(wide,outcomes);csv_new(TABLES/'hindsight.csv',hindsight)
 reference={n:wide.loc[wide.pool.eq('P_RAND')&wide.window.eq('T')&wide.world.eq('N2'),n+'__statistic'].dropna().to_numpy() for n in power.NAMES}
 p=power.report(reference);csv_new(TABLES/'power.csv',p);dsr=accounting.run();csv_new(TABLES/'dsr.csv',dsr)
 from figures import make
 make(wide,pi,pe,dose,p,hindsight)
 text=readme_results(ex,pi,pe,di,hindsight,p,family,transfer,dsr,screens)
 publish(HERE/'RESULTS.md',text)
 prefix=(HERE/'README.md').read_bytes();reg=read(HERE/'audit/code_registration.json');n=reg['README_top_bytes'];top=prefix[:n];remaining=prefix[n:];addition=('\n\n'+text).encode();assert addition.startswith(remaining),'Unexpected existing README results; preserve and stop'
 if len(remaining)<len(addition):
  with (HERE/'README.md').open('ab') as f:f.write(addition[len(remaining):])
 publish(HERE/'open_questions.md',open_questions(screens));publish(TABLES/'open_questions.md',open_questions(screens))
 value=dict(status='PASS',family=family,transfer=transfer,summary=summary_lines(pi,ex,pe,di,hindsight,p,family,transfer),complete_report=True)
 write(ready,value);return value

def fmt(r):
 return f"{r['estimate']:.4g} [{r['ci95_low']:.4g}, {r['ci95_high']:.4g}]"
def summary_lines(pi,ex,pe,di,hi,power,family,transfer):
 r={x['id']:x for x in family};v=pi[pi.pool.eq('P_RAND')&pi.group.eq('POOLED')];piword=', '.join(f"{n} {v[v.statistic.eq(n)].iloc[0].one_minus_pi0:.3f}" for n in ['ADF_c','ADF_ct','theta_21']);r1b=ex[ex.question.eq('R1b')].iloc[0]
 r5=di[di.question.eq('R5')&di.pool.eq('P_RAND')&di.screen.isin(['RC','RP'])];r5word='; '.join(f'{q.screen} {fmt(q)}' for _,q in r5.iterrows());corr=hi[hi.kind.eq('train_test_correlation_real_minus_N2')].iloc[0];p=power[power.k.astype(str).isin(['1','1.0'])&power.calibration.eq('N2_calibrated_structure_5')]
 return [f"1-pi0 P_RAND: {piword}; R1 {fmt(r['R1'])}, Holm p {r['R1']['holm_p']:.4g}; R1b {fmt(r1b)}.",f"R2 persistence DiD {fmt(r['R2'])}; Holm p {r['R2']['holm_p']:.4g}.",f"R3 L0/RC {fmt(r['R3'])} bps, Holm p {r['R3']['holm_p']:.4g}; R4 tuned/RP {fmt(r['R4'])} bps, Holm p {r['R4']['holm_p']:.4g}.",f"R5 real incremental pass-fail: {r5word} bps; hindsight train/test rank-correlation excess {fmt(corr)} (not a screen).",'Power k=1 (N2 calibrated): '+', '.join(f'{q.statistic} {q.rejection_rate:.3f}' for _,q in p.iterrows())+'; transfer '+('P_RES '+str(transfer['screen']) if transfer['add_P_RES'] else 'no new pool')+'.']
def readme_results(ex,pi,pe,di,hi,power,family,transfer,dsr,screens):
 lines=summary_lines(pi,ex,pe,di,hi,power,family,transfer);r={x['id']:x for x in family}
 return '''# Results (development evidence)\n\n'''+ '\n\n'.join(lines)+f'''\n\n1. The pooled IPW Storey estimates above quantify evidence beyond the matched null, conditional on the empirical-p and mixture assumptions. They are not a census of true cointegrated pairs. Era and pooled CIs resample years; individual-fold points have no year-block CI.\n\n2. R1b compares excess nominal ADF-ct versus ADF-c rejections; its estimate is {fmt(ex[ex.question.eq('R1b')].iloc[0])}. Different power of the two tests prevents reading this difference as proof of trend stationarity.\n\n3. R2's supported flag is {r['R2']['supported']}. The A/B persistence test uses BH tests, with no T-screen fallback. Rank persistence is reported separately.\n\n4. R3/R4 supported flags are {r['R3']['supported']}/{r['R4']['supported']}. These compare real-minus-N2 outcomes between real train-screen passers and failers. R5 uses real-only TD8_tuned minus L0; its intervals appear in did_results.csv.\n\n5. Hindsight bins and train/test correlations are in hindsight.csv. Both the predictor and profit use test-year data here: these are infeasible associations, not prospective selections or causal evidence that Q-learning captured the signal.\n\n6. power.csv reports 20,000 independent T-series per w. At w<1 the DGP contains a random walk, so rejection is detectability of partial structure, not power against strict cointegration. KPSS nominal rejection tests nonstationarity; its calibrated left tail tests structure relative to a random walk.\n\n7. Transfer: {transfer['future_task']}. No future pool or strategy is executed here.\n\nAccounting: 544+2=546 cumulative trials. dsr.csv changes only that input in the sealed sufficient-statistic calculation, keeping the sealed best cell and its returns fixed; it is not a newly optimised screened strategy.\n\nLimitations: actual exchange-session counts vary around 756; formation beta is fixed; ADF lag 5; KPSS nominal p-values are table-bounded; dependent pairs and overlapping train windows; calibration within finite pooled nulls; nulls remove drift as well as predictability; survivorship and current GICS; C/E reuse test years; the two null replicates are averaged only when both are available. Fallback and its scope are frozen in the addenda and visible row by row. Oscillatory ECM fits have undefined logarithmic half-lives and are counted, not coerced to a finite value.\n'''
def open_questions(screens):
 return '''# Interpretation and unresolved research questions\n\n- A fixed formation-beta spread is not the re-estimated textbook EG residual. EG/Johansen are stratified secondary checks.\n- Does an economically stable structure survive selection and transaction costs? No new trading or RL fit is run here.\n- Are the pooled-null exchangeability and Storey mixture assumptions adequate under shared tickers and heteroskedasticity? Reported empirical p-values are fold-conditioned ranks, not exact pair-specific conditional tests.\n- ADF-c versus ADF-ct rejection differences can reflect power as well as deterministic trends.\n- KPSS has the opposite nominal null; a small nominal p rejects stationarity, whereas its lower-tail empirical p tests departure from a random walk toward more stable behaviour. The requested 1-KPSS-p dose axis therefore increases with evidence against stationarity.\n- ECM alpha >=0 yields an infinite half-life; alpha <=-1 has no real-valued logarithmic half-life. Delta intervals are untruncated normal approximations.\n- No future fold enters a screen unless the explicitly approved global post-hoc fallback is selected; that mode is a retrospective design choice, not a prospective backtest.\n- Within-fold Spearman correlations use ordinary ranks with IPW covariance; pooled persistence of ranks is the IPW mean of those fold correlations.\n- Test-year predictability and profit share returns. Hindsight association cannot identify a causal or advance-tradable edge.\n- The transfer decision is a gate for a later preregistered task, not validation on untouched test years.\n'''
