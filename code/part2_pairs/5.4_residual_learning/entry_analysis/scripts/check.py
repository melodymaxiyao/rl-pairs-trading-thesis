import common as C
import hashlib,json,csv,math,collections
from pathlib import Path
C.guard();P=C.P;A=P/'audit';ROOT=Path('/Users/maxiyao/FYP_runs/coint_residual_qlearning_v1');CARD=ROOT/'post_tier3_review_v1/policy_card_v1'
prior=C.js(P/'input_hashes_before.json')
for p in A.glob('*hashes.json'):
 for k,v in C.js(p).items():
  if k in prior:assert prior[k]==v
  prior[k]=v
seal=C.js(ROOT/'tier_3/complete.json');mf=C.js(ROOT/'tier_3/manifest.json')['files'];assert C.INPUTS[str(ROOT/'tier_3/manifest.json')]['sha256']==seal['manifest_sha256']
reg=C.js(ROOT/'code_registration.json')['files'];assert C.INPUTS[str(ROOT/'code_registration.json')]['sha256']==seal['code_registration_sha256']
receipt=C.js(CARD/'receipt.json');assert C.INPUTS[str(ROOT/'tier_3/complete.json')]['sha256']==receipt['source_complete_sha256']
used=[]
def sealed(rel):
 p=ROOT/rel;b=C.raw(p);expected=mf.get(rel,reg.get(rel));assert expected is not None and C.INPUTS[str(p)]['sha256']==expected,('sealed mismatch',rel);used.append(rel);return p
for rel in ['scripts/kernel.py','scripts/accounting.py','scripts/missing_prices.py','scripts/data.py','scripts/experiment.py','scripts/learning.py','scripts/policy_kernel.c']:sealed(rel)
for name in ['build.py','fold_policies.csv','aggregate_policies.csv','D6_rates.csv']:
 p=CARD/name;C.raw(p);assert C.INPUTS[str(p)]['sha256']==receipt['artifacts_sha256'][name],name
fold=C.rows(CARD/'fold_policies.csv');aggregate=C.rows(CARD/'aggregate_policies.csv');D6=C.rows(CARD/'D6_rates.csv')
choose=lambda rr:[r for r in rr if r['definition']=='RP'and r['cost']=='10'and r['learner']=='Q_best']
a=next(r for r in choose(aggregate)if r['state']=='0');d=next(r for r in choose(D6)if r['side']=='entry'and r['bin']=='0'and r['world']=='real'and r['replicate']=='0')
rows=[]
for year in range(2004,2026):
 f=next(r for r in choose(fold)if r['test_year']==str(year)and r['state']=='0')
 rel=f'outputs/cells/RP_{year}_G0_c10_Q_best/entries_exits.csv.gz';p=sealed(rel)
 rr=C.rows(p);rr=[r for r in rr if r['definition']=='RP'and r['gate']=='G0'and r['cost']=='10'and r['learner']=='Q_best'and r['side']=='entry'and r['bin']=='0'and r['world']=='real'and r['replicate']=='0'];assert len(rr)==1;r=rr[0]
 cp=sealed(f'outputs/cells/RP_{year}_G0_c10_Q_best/complete.json');cell=C.js(cp);assert cell['policy']['kind']==0;assert str(cell['policy']['bits']&1)==f['action1']
 n,o=int(r['transitions']),int(r['opportunities']);rate=n/o if o else None;assert rate is not None and abs(rate-float(r['rate']))<1e-15
 assert (n==o and f['action']=='enter')or(n==0 and f['action']=='stay'),(year,f,r)
 held=[r for r in choose(fold)if r['test_year']==str(year)and int(r['state'])>=4]
 rows.append(dict(test_year=year,definition='RP',gate='G0',cost_bps_per_side=10,learner='Q_best',state=0,state_label='flat |s|<0.5',deployed_action=f['action'],action1=int(f['action1']),saved_Q_advantage=f['Q_advantage'],saved_nominal_advantage_SE=f['nominal_advantage_SE'],saved_entry_LCB=f['entry_LCB'],actual_entries=n,flat_opportunities=o,entry_rate=rate,entry_rate_percent=100*rate,all_held_state_actions_hold=all(r['action']=='hold'for r in held),policy_source=str(CARD/'fold_policies.csv'),policy_source_key=f'definition=RP;cost=10;learner=Q_best;test_year={year};state=0',policy_source_sha256=C.INPUTS[str(CARD/'fold_policies.csv')]['sha256'],cell_policy_source=str(cp),counts_source=str(p),counts_source_key='side=entry;bin=0;world=real;replicate=0',counts_source_sha256=C.INPUTS[str(p)]['sha256']))
assert sum(r['action1']for r in rows)==int(a['action1_folds'])==13
N=sum(r['actual_entries']for r in rows);O=sum(r['flat_opportunities']for r in rows)
assert N==int(d['transitions'])==14633 and O==int(d['opportunities'])==1794869;assert abs(N/O-float(d['rate']))<1e-15
E=[r for r in rows if r['deployed_action']=='enter'];ST=[r for r in rows if r['deployed_action']=='stay'];opE=sum(r['flat_opportunities']for r in E);opST=sum(r['flat_opportunities']for r in ST);assert opE==N and opST==1780236
assert all(r['all_held_state_actions_hold']for r in E)
tablepath=Path('/Users/maxiyao/FYP_runs/thesis_tables_v3_pairs/outputs/tables/T7_9.csv');tt=C.rows(tablepath)
tr=[r for r in choose(tt)if r['part']=='0'and r['state']=='0'];dr=[r for r in choose(tt)if r['part']=='1'and r['side']=='entry'and r['bin']=='0'and r['world']=='real'and r['replicate']=='0'];assert len(tr)==len(dr)==1
for t,p,r,keys in [(tr[0],CARD/'aggregate_policies.csv',a,['action1_folds','folds']),(dr[0],CARD/'D6_rates.csv',d,['opportunities','transitions','rate'])]:
 assert t['source_sha256']==C.INPUTS[str(p)]['sha256'];assert all(t[k]==r[k]for k in keys)
# Exact plotted row: source table T7_9, part1, RP,c10,entry bin0,real.
figpath=Path('/Users/maxiyao/FYP_runs/thesis_tables_v3_pairs/outputs/figures/figure_data.csv');fig=C.rows(figpath)
match=[r for r in fig if r.get('source_table')=='T7_9'and r.get('table_part')=='1'and r.get('y_column')=='rate'and r.get('panel')=='RP entry rate'and r.get('series')=='10'and r.get('x')=='0']
assert len(match)==1 and float(match[0]['y'])==float(d['rate']);assert match[0]['source_sha256']==C.INPUTS[str(CARD/'D6_rates.csv')]['sha256']
C.dump(A/'figure_rate_rows.json',match)
# Read source for figure, without importing or executing plotting code.
for p in [Path('/Users/maxiyao/FYP_runs/thesis_tables_v3_pairs/scripts/build.py')]:C.raw(p)
C.csvout(P/'fold_comparison.csv',rows)
C.dump(A/'checks.json',dict(sealed_files_verified=used,n_folds=22,enter_policies=13,stay_policies=9,all_enter_folds_rate_one=True,all_stay_folds_rate_zero=True,all_enter_folds_hold_in_all_four_held_states=True,entries=N,opportunities=O,enter_fold_opportunities=opE,stay_fold_opportunities=opST,pooled_rate=N/O,pooled_rate_percent=100*N/O,table_rate_matches=True,new_fits=0,new_backtests=0,new_inference=0))
# Source excerpts with original filenames/line numbers, never import scientific modules.
refs={str(ROOT/'scripts/kernel.py'):[(69,76)],str(ROOT/'scripts/learning.py'):[(16,24)],str(ROOT/'scripts/data.py'):[(93,96)],str(ROOT/'scripts/accounting.py'):[(7,13),(28,44),(67,76),(98,101)],str(ROOT/'scripts/missing_prices.py'):[(33,43)],str(ROOT/'scripts/experiment.py'):[(33,44)],str(CARD/'build.py'):[(87,113),(148,153)],'/Users/maxiyao/FYP_runs/thesis_tables_v3_pairs/scripts/build.py':[(382,404)]}
ex=[]
for p,ranges in refs.items():
 lines=C.text(p).splitlines()
 for lo,hi in ranges:
  ex.append(p+f':{lo}–{hi}\n'+'\n'.join(f'{i}: {lines[i-1]}'for i in range(lo,hi+1)))
(A/'source_excerpts.txt').write_text('\n\n'.join(ex)+'\n')
# Retain first-read digests and check every source after output extraction.
for p,v in C.INPUTS.items():
 if p in prior:assert prior[p]==v,('input changed between reads',p)
 prior[p]=v
C.dump(P/'input_hashes_before.json',prior);after={}
for p,v in prior.items():
 C.raw(p);after[p]=C.INPUTS[p];assert after[p]==v,('input changed',p)
C.dump(P/'input_hashes_after.json',after)
C.dump(A/'dataless_scan.json',{'all_read_inputs_materialized':True,'n_inputs':len(prior),'dataless_skipped':[]})
print('FIGURESCHEMA',list(fig[0]));print('FIGUREEXAMPLE',match[:1]);print('TOTAL',N,O,N/O,'enter_op',opE,'stay_op',opST,'inputs',len(prior))
