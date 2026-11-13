"""Revise saved SPY figures and calculate only the requested saved-return CAGR."""
import os
from pathlib import Path
P=Path(__file__).resolve().parents[1]
os.environ.update(PYTHONDONTWRITEBYTECODE='1',MPLBACKEND='Agg',MPLCONFIGDIR=str(P/'mpl'),TMPDIR=str(P/'tmp'),XDG_CACHE_HOME=str(P/'tmp'))
import common as C
C.guard()
import io,json,math,csv,hashlib,datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch,Rectangle
from PIL import Image
R=Path('/Users/maxiyao/single_stock_runs/spy_qtable_figure_v1');EXP=Path('/Users/maxiyao/Desktop/FYP/experiments/single_stock/results/spy')
ledger=C.js(R/'audit/input_hashes_before.json');checks=[]
def source(p):
 b=C.raw(p);v=C.INPUTS[str(p)]
 if str(p) in ledger:assert v==ledger[str(p)],('Original input ledger mismatch',p)
 checks.append(dict(path=str(p),sha256=v['sha256'],size=v['size'],original_input_ledger_match=str(p)in ledger));return b
def rows(p):return list(csv.DictReader(io.StringIO(source(p).decode('utf-8-sig'))))
def obj(p):return json.loads(source(p))
policy=obj(R/'audit/fig_spy_q12_policy_data.json');plot=rows(R/'audit/fig_spy_q12_test_data.csv');summary=rows(R/'spy_summary.csv');pt=rows(R/'spy_q12_policy.csv');sm={r['row_type']:r for r in summary if r['row_type']!='seed'}
# Generated plotting artifacts were outputs of the prior run, not members of its input ledger.
# Validate their numeric contents against the saved tables and saved returns, never invent a ledger entry.
for a in policy:
 b=next(r for r in pt if int(r['state'])==a['state']);assert int(b['n_seeds_long'])==a['n_seeds_long'];assert float(b['advantage_mean_bps'])==a['advantage_mean_bps'];assert float(b['train_visits_mean'])==a['train_visits_mean']
metricpath=Path('/Users/maxiyao/single_stock_runs/inputs_restored_v1/fyp_rl/metrics.py');metrictext=source(metricpath).decode();(P/'audit/metrics_source.txt').write_text(metrictext)
performance=[];dates0=None;daily_by_key={}
for key in [str(i)for i in range(10)]+['buy_hold','sma200_trend']:
 model='tabular_q_12state'if key.isdigit()else key;seed='seed_'+key if key.isdigit()else'seed_NA';base=EXP/model/'horizon_10/cost_0.001'/seed
 daily=rows(base/'daily_returns.csv');met=obj(base/'metrics.json');ret=np.array([float(r['daily_return'])for r in daily]);dates=[r['date']for r in daily];assert len(ret)==1003 and np.isfinite(ret).all()and (1+ret>0).all()
 if dates0 is None:dates0=dates
 assert dates==dates0
 ann=float(np.prod(1+ret)**(252/len(ret))-1);gate=float(np.sqrt(252)*np.mean(ret)/np.std(ret,ddof=0));assert abs(gate-float(met['sharpe']))<1e-12
 performance.append(dict(seed_or_baseline=key,n_days=len(ret),annual_net_return=ann,saved_sharpe=float(met['sharpe']),verification_sharpe=gate,saved_max_drawdown=float(met['max_drawdown']),saved_cagr=met.get('cagr','')));daily_by_key[key]=ret
seeds=performance[:10];BH=performance[10];SMA=performance[11];mean_ann=float(np.mean([r['annual_net_return']for r in seeds]));mean_gate=float(np.mean([r['verification_sharpe']for r in seeds]));assert f'{mean_gate:.3f}'=='0.930'and f"{BH['verification_sharpe']:.3f}"=='0.667'
assert dates0==[r['date']for r in plot]
np.testing.assert_allclose(np.log1p(daily_by_key['buy_hold']).cumsum(),[float(r['bh_cumulative_net_log_return'])for r in plot],atol=1e-12,rtol=0)
np.testing.assert_allclose(np.mean([np.log1p(daily_by_key[str(i)]).cumsum()for i in range(10)],axis=0),[float(r['seed_mean_cumulative_net_log_return'])for r in plot],atol=1e-12,rtol=0)
# Count labels and curves remain unchanged. Cash shading is based on saved post-execution positions.
for r in plot:assert (int(r['cash_seed_count'])>=6)==(r['majority_cash']=='True')
C.csvout(P/'audit/annual_return_by_seed.csv',performance)
cols=['row','annual_net_return_pct','sharpe','max_drawdown_pct','long_day_share_pct','position_changes_per_year']
def fmt_range(values,scale=1):return f'{min(values)*scale:.10f}–{max(values)*scale:.10f}'
qm=sm['seed_mean'];ss=[r for r in summary if r['row_type']=='seed'];sma_saved=sm['SMA200_saved_baseline']
perfrows=[dict(row='Q-learning 种子均值',annual_net_return_pct=100*mean_ann,sharpe=qm['strategy_sharpe'],max_drawdown_pct=100*float(qm['max_drawdown']),long_day_share_pct=100*float(qm['long_day_share']),position_changes_per_year=qm['position_changes_per_test_year']),dict(row='种子范围',annual_net_return_pct=fmt_range([r['annual_net_return']for r in seeds],100),sharpe=fmt_range([r['saved_sharpe']for r in seeds]),max_drawdown_pct=fmt_range([r['saved_max_drawdown']for r in seeds],100),long_day_share_pct=fmt_range([float(r['long_day_share'])for r in ss],100),position_changes_per_year=fmt_range([float(r['position_changes_per_test_year'])for r in ss])),dict(row='买入并持有',annual_net_return_pct=100*BH['annual_net_return'],sharpe=BH['saved_sharpe'],max_drawdown_pct=100*BH['saved_max_drawdown'],long_day_share_pct='',position_changes_per_year=''),dict(row='SMA200',annual_net_return_pct=100*SMA['annual_net_return'],sharpe=SMA['saved_sharpe'],max_drawdown_pct=100*SMA['saved_max_drawdown'],long_day_share_pct=100*float(sma_saved['long_day_share']),position_changes_per_year=sma_saved['position_changes_per_test_year'])]
C.csvout(P/'performance_summary.csv',perfrows,cols)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300,'pdf.fonttype':42,'ps.fonttype':42,'axes.unicode_minus':True})
# Figure4.1: exact original colors/layout, action -> vote count -> estimated Q advantage.
fig,axs=plt.subplots(2,2,figsize=(10.8,5.4));artists=[]
for ri,tr in enumerate(['up','down']):
 for ci,pos in enumerate(['cash','long']):
  ax=axs[ri,ci]
  for zi in range(3):
   state=4*zi+2*(tr=='up')+(pos=='long');p=next(r for r in policy if r['state']==state);k=p['n_seeds_long'];strength=max(k,10-k)/10
   action='LONG'if k>5 else'CASH'if k<5 else'TIE';base=np.array(to_rgb('#0072B2'if k>5 else'#555B63'));color=tuple(1-(1-base)*(0.30+0.70*(strength-.5)/.5))if k!=5 else'#E5E7EB';fg='white'if strength>=.8 else'#17202A'
   ax.add_patch(Rectangle((zi-.5,-.5),1,1,facecolor=color,edgecolor='white',linewidth=2,hatch='///'if p['train_visits_mean']<30 else None))
   a=ax.text(zi,.19,action,ha='center',va='center',fontsize=18,fontweight='bold',color=fg)
   v=ax.text(zi,-.03,f'{k}/10 long',ha='center',va='center',fontsize=10.5,color=fg)
   q=ax.text(zi,-.25,f"Est. ΔQ {p['advantage_mean_bps']:+.1f} bps",ha='center',va='center',fontsize=8.5,color=fg)
   assert a.get_text()==action and v.get_text()==f'{k}/10 long'and q.get_text()==f"Est. ΔQ {p['advantage_mean_bps']:+.1f} bps";artists.append(dict(state=state,action=action,long_votes=k,estimated_advantage_bps=p['advantage_mean_bps'],font_sizes=[18,10.5,8.5]))
  ax.set(xlim=(-.5,2.5),ylim=(-.5,.5),xticks=range(3),xticklabels=['Oversold','Neutral','Overbought'],yticks=[]);ax.tick_params(axis='x',length=0,pad=8);ax.set_ylabel(f'Trend: {tr}\nPosition: {pos}',rotation=0,ha='right',va='center',labelpad=12)
  for s in ax.spines.values():s.set_visible(False)
fig.subplots_adjust(left=.14,right=.985,top=.97,bottom=.16,wspace=.40,hspace=.38)
fig.legend(handles=[Patch(facecolor='#0072B2',label='Majority long'),Patch(facecolor='#555B63',label='Majority cash'),Patch(facecolor='white',edgecolor='#777777',hatch='///',label='Mean training visits < 30')],loc='lower center',ncol=3,frameon=False,fontsize=9,bbox_to_anchor=(.54,.015))
for ext in ['png','pdf']:fig.savefig(P/f'fig_spy_q12_policy_v2.{ext}',dpi=300)
plt.close(fig)
# Figure4.2: preserve every data point, curve and exact contiguous cash-shading interval.
dates=np.array([np.datetime64(r['date'])for r in plot]);mask=np.array([int(r['cash_seed_count'])>=6 for r in plot]);close=np.array([float(r['close'])for r in plot]);sma=np.array([float(r['sma200'])for r in plot]);netlog=np.array([float(r['seed_mean_cumulative_net_log_return'])for r in plot]);bhlog=np.array([float(r['bh_cumulative_net_log_return'])for r in plot]);anchor=np.datetime64('2021-12-31');td=np.r_[anchor,dates]
fig,ax=plt.subplots(2,1,figsize=(10.2,6.3),sharex=True,gridspec_kw={'height_ratios':[1.1,1]})
linep=ax[0].plot(dates,np.log(close),color='#20252B',lw=1.15,label='SPY adjusted close')[0];linesma=ax[0].plot(dates,np.log(sma),color='#D55E00',lw=1.2,label='200-session SMA')[0]
spans=[];i=0
while i<len(mask):
 if not mask[i]:i+=1;continue
 start=i
 while i+1<len(mask)and mask[i+1]:i+=1
 end=dates[i+1]if i+1<len(dates)else dates[i]+np.timedelta64(1,'D');ax[0].axvspan(dates[start],end,color='#D7DADF',alpha=.65,lw=0,zorder=0);spans.append([str(dates[start]),str(end)]);i+=1
handles,_=ax[0].get_legend_handles_labels();handles.append(Patch(facecolor='#D7DADF',label='Cash (≥ 6 of 10 seeds)'));ax[0].legend(handles=handles,loc='upper left',bbox_to_anchor=(0,.91),frameon=False,ncol=3,fontsize=8.5);ax[0].set_ylabel('Log adjusted price')
lineq=ax[1].plot(td,np.r_[0,netlog],color='#0072B2',lw=1.6,label='Q-learning (mean of 10 seeds)')[0];linebh=ax[1].plot(td,np.r_[0,bhlog],color='#555B63',lw=1.3,label='Buy-and-hold')[0];ax[1].set_ylabel('Cumulative net log return');ax[1].legend(loc='upper left',bbox_to_anchor=(0,.91),frameon=False,ncol=2,fontsize=9);ax[1].set_xlabel('Test period');ax[1].set_xlim(anchor,dates[-1])
for a,label in zip(ax,['(a) Price and positions','(b) Out-of-sample net performance']):a.text(.012,.975,label,transform=a.transAxes,ha='left',va='top',fontsize=10.5,fontweight='medium',bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=1));a.grid(axis='y',alpha=.18)
np.testing.assert_array_equal(linep.get_ydata(),np.log(close));np.testing.assert_array_equal(linesma.get_ydata(),np.log(sma));np.testing.assert_array_equal(lineq.get_ydata(),np.r_[0,netlog]);np.testing.assert_array_equal(linebh.get_ydata(),np.r_[0,bhlog]);assert len(ax)==2 and not any(a.tables for a in ax)
fig.tight_layout(h_pad=1.7)
for ext in ['png','pdf']:fig.savefig(P/f'fig_spy_q12_test_v2.{ext}',dpi=300)
plt.close(fig)
C.dump(P/'audit/plotted_values.json',dict(policy=artists,cash_spans=spans,test_data_source=str(R/'audit/fig_spy_q12_test_data.csv'),n_test_points=len(plot),anchor='2021-12-31',all_curves_exact=True))
C.csvout(P/'audit/source_hash_checks.csv',checks)
C.dump(P/'audit/sharpe_verification.json',dict(seed_mean=mean_gate,buy_hold=BH['verification_sharpe'],passes=True,ddof=0,annualization=252,per_seed=performance))
C.dump(P/'audit/annual_return_summary.json',dict(mean=mean_ann,minimum=min(r['annual_net_return']for r in seeds),maximum=max(r['annual_net_return']for r in seeds),buy_hold=BH['annual_net_return'],sma200=SMA['annual_net_return'],mean_minus_buy_hold=mean_ann-BH['annual_net_return']))
C.dump(P/'input_hashes_before.json',C.INPUTS);after={}
for p,v in list(C.INPUTS.items()):C.raw(p);after[p]=C.INPUTS[p];assert after[p]==v
C.dump(P/'input_hashes_after.json',after)
for name in ['fig_spy_q12_policy_v2.png','fig_spy_q12_test_v2.png']:
 with Image.open(P/name)as im:assert all(abs(d-300)<.01 for d in im.info['dpi']);print('IMAGE',name,im.size,im.info['dpi'])
print('SUMMARY',perfrows);print('DIFFERENCE_PP',100*(mean_ann-BH['annual_net_return']));print('SHARPE',mean_gate,BH['verification_sharpe'])
