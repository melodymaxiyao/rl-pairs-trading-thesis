import common as C
C.guard()
from pathlib import Path
import io,sys,hashlib,json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Patch
from matplotlib.colors import to_rgb
R=Path('/Users/maxiyao/Desktop/FYP/experiments/single_stock/results');F=Path('/Users/maxiyao/single_stock_runs/inputs_restored_v1');D=R/'spy/tabular_q_12state/horizon_10/cost_0.001'
assert C.js(C.P/'audit/sharpe_verification.json')['passes']
q=[];vis=[];metrics=[];daily=[];seed_detail=[];source_notes=[]
ci=C.rows(R/'summary/bootstrap_intervals.csv');ml=C.rows(R/'summary/metrics_long.csv')
ci=[r for r in ci if r['ticker']=='SPY' and r['state_space']=='tabular_q_12state' and r['horizon']=='10' and r['cost']=='0.001'];assert len(ci)==10
schemas=C.js(C.P/'audit/source_schema.json')
def frame(p):return pd.read_csv(io.BytesIO(C.raw(p)),keep_default_na=False)
for seed in range(10):
 d=D/f'seed_{seed}';tab=C.js(d/'q_table.json');n=np.asarray(tab['visit_counts'],dtype=int);a=np.asarray(tab['q_table'],dtype=float);assert a.shape==n.shape==(12,2)
 vc=C.rows(d/'visit_counts.csv');assert len(vc)==24
 for x in vc:assert n[int(x['state']),int(x['action'])]==int(x['visit_count'])
 gp=np.argmax(a,axis=1);assert all(gp[s]==tab['greedy_policy'][str(s)] for s in range(12))
 cf=C.js(d/'config.json');assert cf['split']=={'train_start':'2008-01-01','train_end':'2021-12-31','test_start':'2022-01-01','test_end':'2025-12-31'}
 assert cf['run']['model']=='tabular_q_12state' and cf['run']['horizon']==10 and cf['run']['cost']==.001 and cf['run']['seed']==seed
 met=C.js(d/'metrics.json');dy=frame(d/'daily_returns.csv');assert len(dy)==1003;assert set(dy['w_new'])<=set([0.,1.]);assert abs(dy['w_old'].mean()-met['exposure'])<1e-12
 q.append(a);vis.append(n);metrics.append(met);daily.append(dy)
 for s in range(12):
  label=tab['state_labels'][str(s)];z=['oversold','neutral','overbought'][s//4];trend='up' if (s//2)%2 else 'down';pos='long' if s%2 else 'cash';assert label==f'{z}/trend={trend}/pos={pos}'
  seed_detail.append(dict(seed=seed,state=s,z_bucket=z,trend=trend,position=pos,q_cash=a[s,0],q_long=a[s,1],advantage_bps=(a[s,1]-a[s,0])*10000,greedy_action='TargetLong' if gp[s] else 'TargetCash',n_cash=int(n[s,0]),n_long=int(n[s,1]),n_total=int(n[s].sum()),source_q=str(d/'q_table.json'),source_q_sha256=C.INPUTS[str(d/'q_table.json')]['sha256']))
q=np.stack(q);vis=np.stack(vis);advantages=(q[:,:,1]-q[:,:,0])*10000;longvotes=np.argmax(q,axis=2).sum(axis=0);meanvis=vis.sum(axis=2).mean(axis=0);minvis=vis.sum(axis=2).min(axis=0)
policy=[]
for s in range(12):
 z=['oversold','neutral','overbought'][s//4];trend='up' if (s//2)%2 else 'down';pos='long' if s%2 else 'cash';k=int(longvotes[s]);maj='TargetLong' if k>5 else 'TargetCash' if k<5 else 'TIE';sma='TargetLong' if trend=='up' else 'TargetCash'
 policy.append(dict(state=s,z_bucket=z,trend=trend,position=pos,n_seeds_long=k,n_seeds=10,advantage_mean_bps=float(advantages[:,s].mean()),advantage_sd_bps=float(advantages[:,s].std(ddof=1)),train_visits_mean=float(meanvis[s]),train_visits_min=int(minvis[s]),test_state_day_share='',test_state_share_status='not_saved: daily logs omit state; no state reconstruction',majority_action=maj,sma200_action=sma,majority_matches_sma200=maj==sma,low_sample_mean_lt30=bool(meanvis[s]<30)))
# Validate source-derived values before plotting; no learner or trading engine called.
assert all(np.isfinite(q.flatten()));assert all(n.sum()>0 for n in vis)
for s,p in enumerate(policy):
 assert p['n_seeds_long']==sum(r['greedy_action']=='TargetLong' for r in seed_detail if r['state']==s)
 assert np.isclose(p['advantage_mean_bps'],np.mean([r['advantage_bps'] for r in seed_detail if r['state']==s]),rtol=0,atol=1e-12)
C.csvout(C.P/'spy_q12_policy.csv',policy);C.csvout(C.P/'spy_q12_seed_detail.csv',seed_detail)
# Saved BH and SMA200 baseline at exactly the same horizon and cost.
bd=R/'spy/buy_hold/horizon_10/cost_0.001/seed_NA';sd=R/'spy/sma200_trend/horizon_10/cost_0.001/seed_NA'
bhm=C.js(bd/'metrics.json');bh=frame(bd/'daily_returns.csv');smam=C.js(sd/'metrics.json');smacf=C.js(sd/'config.json');assert smacf['run']['cost']==.001 and smacf['run']['horizon']==10
for d in [bd,sd]:
 for f in ['metrics.json','config.json','daily_returns.csv']:
  p=d/f;C.raw(p);schemas.append(dict(path=str(p),sha256=C.INPUTS[str(p)]['sha256'],columns=list(C.js(p)) if p.suffix=='.json' else list(frame(p).columns)))
sma_delta=smam['sharpe']-bhm['sharpe'];positive=sum(m['sharpe']-bhm['sharpe']>0 for m in metrics);matches=sum(p['majority_matches_sma200'] for p in policy);years=4
summary=[]
for seed,m in enumerate(metrics):
 x=next(r for r in ci if int(r['seed'])==seed);delta=m['sharpe']-bhm['sharpe'];assert abs(delta-float(x['point_estimate']))<1e-12
 saved=next(r for r in ml if r['ticker']=='SPY' and r['model']=='tabular_q_12state' and r['horizon']=='10' and r['cost']=='0.001' and r['seed']==str(seed));assert m['sharpe']==float(saved['sharpe'])
 summary.append(dict(row_type='seed',seed=seed,strategy_sharpe=m['sharpe'],bh_sharpe=bhm['sharpe'],delta_sharpe=delta,saved_ci_lo=float(x['ci_lower_95']),saved_ci_hi=float(x['ci_upper_95']),positive_delta_seeds=positive,n_seeds=10,long_day_share=float(daily[seed]['w_new'].mean()),saved_exposure_prev_close=m['exposure'],position_changes_total=m['turnover'],position_changes_per_test_year=m['turnover']/years,test_years=years,max_drawdown=m['max_drawdown'],bh_max_drawdown=bhm['max_drawdown'],sma200_delta_sharpe=sma_delta,majority_policy_states_matching_sma200=matches,n_states=12,seed_policy_states_matching_sma200=sum(r['greedy_action']==policy[r['state']]['sma200_action'] for r in seed_detail if r['seed']==seed),ci_status='saved seed-specific 95% moving-block CI; no new inference'))
meanrow={k:float(np.mean([r[k] for r in summary])) for k in ['strategy_sharpe','delta_sharpe','long_day_share','saved_exposure_prev_close','position_changes_total','position_changes_per_test_year','max_drawdown','seed_policy_states_matching_sma200']}
meanrow.update(row_type='seed_mean',seed='',bh_sharpe=bhm['sharpe'],saved_ci_lo='',saved_ci_hi='',positive_delta_seeds=positive,n_seeds=10,test_years=years,bh_max_drawdown=bhm['max_drawdown'],sma200_delta_sharpe=sma_delta,majority_policy_states_matching_sma200=matches,n_states=12,ci_status='seed-mean CI not saved; not recomputed')
summary.append(meanrow)
summar=dict(row_type='SMA200_saved_baseline',seed='',strategy_sharpe=smam['sharpe'],bh_sharpe=bhm['sharpe'],delta_sharpe=sma_delta,saved_ci_lo='',saved_ci_hi='',positive_delta_seeds='',n_seeds='',long_day_share=float(frame(sd/'daily_returns.csv')['w_new'].mean()),saved_exposure_prev_close=smam['exposure'],position_changes_total=smam['turnover'],position_changes_per_test_year=smam['turnover']/years,test_years=years,max_drawdown=smam['max_drawdown'],bh_max_drawdown=bhm['max_drawdown'],sma200_delta_sharpe=sma_delta,majority_policy_states_matching_sma200=matches,n_states=12,seed_policy_states_matching_sma200='',ci_status='no new CI; saved SMA200 horizon10 baseline')
summary.append(summar);C.csvout(C.P/'spy_summary.csv',summary)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300,'axes.unicode_minus':True})
# Figure A: axes label the two state components; no axes title/suptitle.
fig,axs=plt.subplots(2,2,figsize=(9.0,4.9));plotA=[]
for ri,tr in enumerate(['up','down']):
 for cj,pos in enumerate(['cash','long']):
  ax=axs[ri,cj]
  for x,z in enumerate(['oversold','neutral','overbought']):
   p=next(p for p in policy if p['trend']==tr and p['position']==pos and p['z_bucket']==z);k=p['n_seeds_long'];strength=max(k,10-k)/10
   base=np.array(to_rgb('#0072B2' if k>5 else '#555B63'));color=tuple(1-(1-base)*(0.30+0.70*(strength-.5)/.5))
   patch=Rectangle((x-.5,-.5),1,1,facecolor=color,edgecolor='white',linewidth=2,hatch='///' if p['train_visits_mean']<30 else None);ax.add_patch(patch)
   foreground='white' if strength>=.8 else '#17202A';label=f"{k}/10\n{p['advantage_mean_bps']:+.1f} bps";ax.text(x,0,label,ha='center',va='center',fontsize=10,color=foreground,linespacing=1.7)
   plotA.append(dict(state=p['state'],n_seeds_long=k,advantage_mean_bps=p['advantage_mean_bps'],train_visits_mean=p['train_visits_mean'],low_sample_hatch=p['train_visits_mean']<30,label=label))
  ax.set_xlim(-.5,2.5);ax.set_ylim(-.5,.5);ax.set_xticks(range(3),['Oversold','Neutral','Overbought']);ax.set_yticks([]);ax.set_ylabel(f'Trend: {tr}\nPosition: {pos}',rotation=0,ha='right',va='center',labelpad=12)
  for sp in ax.spines.values():sp.set_visible(False)
fig.subplots_adjust(left=.16,right=.99,top=.98,bottom=.18,wspace=.60,hspace=.50)
fig.legend(handles=[Patch(facecolor='#0072B2',label='Majority long'),Patch(facecolor='#555B63',label='Majority cash'),Patch(facecolor='white',edgecolor='#777777',hatch='///',label='Mean training visits < 30')],loc='lower center',bbox_to_anchor=(.56,.015),ncol=3,frameon=False,fontsize=9)
fig.savefig(C.P/'fig_spy_q12_policy.png',dpi=300);plt.close(fig)
# Figure B: use original adjusted-price cache, no replay of signals/trades/returns.
price=F/'data/raw/spy_daily_adjusted_ohlcv.csv';C.raw(price)
# Call the original read-only cache loader; deny its network and source writes.
sys.path.insert(0,str(F));C.raw(F/'fyp_rl/__init__.py');C.raw(F/'fyp_rl/data.py');from fyp_rl import data as data_module
def denied(*a,**k):raise RuntimeError('Download blocked by spy_qtable_figure_v1')
data_module.yf.download=denied
prices=data_module.load_ohlcv('SPY','2006-11-27','2025-12-31',raw_dir=F/'data/raw')
# User expressly requested this plotting overlay; compute only the specified SMA200, not state features.
sma200=prices['Close'].rolling(200).mean();dates=pd.DatetimeIndex(pd.to_datetime(daily[0]['date']))
assert all(list(d['date'])==list(daily[0]['date']) for d in daily) and list(bh['date'])==list(daily[0]['date'])
rets=np.column_stack([d['daily_return'].to_numpy(float) for d in daily]);holds=np.column_stack([d['w_new'].to_numpy(float) for d in daily]);cash_votes=(holds==0).sum(axis=1);netlog=np.log1p(rets).cumsum(axis=0).mean(axis=1);bhlog=np.log1p(bh['daily_return'].to_numpy(float)).cumsum()
for seed,d in enumerate(daily):assert np.allclose(np.log1p(d['daily_return']).cumsum(),np.log(d['V']),atol=1e-12,rtol=0)
plotB=pd.DataFrame({'date':dates.strftime('%Y-%m-%d'),'close':prices.loc[dates,'Close'].to_numpy(),'sma200':sma200.loc[dates].to_numpy(),'cash_seed_count':cash_votes,'majority_cash':cash_votes>5,'seed_mean_cumulative_net_log_return':netlog,'bh_cumulative_net_log_return':bhlog})
plotB.to_csv(C.P/'audit/fig_spy_q12_test_data.csv',index=False)
fig,ax=plt.subplots(2,1,figsize=(10.2,6.0),sharex=True,gridspec_kw={'height_ratios':[1.1,1]})
ax[0].plot(dates,np.log(plotB['close']),color='#20252B',lw=1.15,label='SPY adjusted close');ax[0].plot(dates,np.log(plotB['sma200']),color='#D55E00',lw=1.2,label='200-session SMA')
mask=cash_votes>5;i=0
while i<len(mask):
 if not mask[i]:i+=1;continue
 start=i
 while i+1<len(mask) and mask[i+1]:i+=1
 end=dates[i+1] if i+1<len(dates) else dates[i]+pd.Timedelta(days=1)
 ax[0].axvspan(dates[start],end,color='#D7DADF',alpha=.65,lw=0,zorder=0);i+=1
handles,labels=ax[0].get_legend_handles_labels();handles.append(Patch(facecolor='#D7DADF',label='Cash in at least 6/10 seeds'));ax[0].legend(handles=handles,loc='upper left',frameon=False,ncol=3,fontsize=8.5)
ax[0].set_ylabel('Log adjusted price')
anchor=pd.Timestamp('2021-12-31');td=dates.insert(0,anchor)
ax[1].plot(td,np.r_[0,netlog],color='#0072B2',lw=1.6,label='Strategy: mean of seed log wealth');ax[1].plot(td,np.r_[0,bhlog],color='#555B63',lw=1.3,label='Buy-and-hold');ax[1].set_ylabel('Cumulative net log return');ax[1].legend(loc='upper left',frameon=False,fontsize=9);ax[1].set_xlabel('Test period');ax[1].set_xlim(anchor,dates[-1])
for a in ax:a.grid(axis='y',alpha=.18)
fig.tight_layout();fig.savefig(C.P/'fig_spy_q12_test.png',dpi=300);plt.close(fig)
# Values consumed by the plot must be exactly the exported source-derived values.
assert sorted(p['state'] for p in plotA)==list(range(12))
for pa in plotA:
 p=policy[pa['state']];assert pa['n_seeds_long']==p['n_seeds_long'] and pa['advantage_mean_bps']==p['advantage_mean_bps']
C.dump(C.P/'audit/fig_spy_q12_policy_data.json',plotA)
C.dump(C.P/'audit/additional_source_schema.json',schemas[-6:]+[dict(path=str(price),sha256=C.INPUTS[str(price)]['sha256'],columns=['Date','Open','High','Low','Close','Volume'])]);C.dump(C.P/'audit/build_input_hashes.json',C.INPUTS)
C.dump(C.P/'audit/build_summary.json',{'seed_mean':meanrow,'SMA200':summar,'positive_delta_seeds':positive,'majority_states_matching_sma200':matches,'min_mean_visits':float(meanvis.min()),'std_ddof':1,'daily_state_missing':True,'no_training_or_replay':True,'plot_value_tests_passed':True,'n_daily_records':len(dates)})
print(pd.DataFrame(policy).to_string(index=False));print('SUMMARY',summary)
