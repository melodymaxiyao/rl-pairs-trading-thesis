"""Post-run reporting and integrity checks only; never reruns policy or inference."""
from pathlib import Path
import sys,json,gzip,hashlib,csv,io
R=Path(__file__).resolve().parents[1];B=R.parent
sys.path.insert(0,str(B/'scripts'))
import common as C
C.P=B;C.guard()
from inventory import snapshot
F1=B/'F1_exact_RF_repair_v1';oldF4=B/'F4_next_close_v1'

def sha(p):return C.sha(p)
def table(rows,cols):
 return '\n'.join(['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']+['| '+' | '.join(str(v.get(k,'')) for k in cols)+' |' for v in rows])
def number(v):return f'{float(v):.4f}'
def interval(v,prefix):return number(v[prefix])+' ['+number(v[prefix+'_ci_low'])+', '+number(v[prefix+'_ci_high'])+']'

# Parent and stopped attempts retain their original immutable seals.
for root in [B,F1,oldF4]:
 completion=C.js(root/'complete.json');assert sha(root/'manifest.json')==completion['manifest_sha256']
 for rel,h in C.js(root/'manifest.json')['files'].items():assert sha(root/rel)==h,('sealed earlier attempt modified',root,rel)
auth=C.js(R/'audit/authorization.json');assert sha(R/'scripts/f4.py')==auth['new_code_sha256'];assert sha(R/'preregistration_addendum.md')==auth['addendum_sha256']
for rel,h in auth['unchanged_shared_code'].items():assert sha(B/'scripts'/rel)==h
for rel,h in auth['same_supporting_tests'].items():assert sha(R/rel)==h
code_receipt=C.js(R/'audit/validate_code_identity.json');assert code_receipt['code_sha256']==auth['new_code_sha256']
register=C.js(R/'trial_register.json');count=register['cumulative_real_trials'];assert count==580+len(register['started_cells'])
scientific=(R/'scientific_complete.json').exists();stopped=(R/'audit/STOP.json').exists()
assert scientific or stopped,'Scientific process still running; do not seal'
if scientific:
 validation=C.js(R/'audit/zero_delay_validation.json');assert validation['passes'] and validation['fold_policy_checks']==132
 delayed_code=C.js(R/'audit/delayed_code_identity.json');assert delayed_code['code_sha256']==code_receipt['code_sha256']
 for proof in validation['proof']:
  file=R/f'frozen_decisions/{proof["family"]}_{proof["year"]}_{proof["learner"]}.npz';assert sha(file)==proof['frozen_decisions_sha256']
 assert count==586 and len(register['started_cells'])==6
 rows=C.rows(R/'F4_next_close.csv');h2=C.rows(R/'F4_H2.csv');assert len(rows)==6 and len(h2)==3
 net=[];turn=[]
 for row in rows:
  key={'family':row['family'],'learner':row['learner']}
  net.append(dict(key,same_close=interval(row,'same_close_net_bps'),next_close=interval(row,'next_close_net_bps'),change=interval(row,'change_net_bps')))
  turn.append(dict(key,same_close=interval(row,'same_close_turnover'),next_close=interval(row,'next_close_turnover'),change=interval(row,'change_turnover')))
 h2rows=[{'execution':z['execution'],'estimate':number(z['estimate']),'95% CI':'['+number(z['ci_low'])+', '+number(z['ci_high'])+']'} for z in h2]
 text='''# F4 next-close — 稳健性检验

Complete:6new real trial cells; cumulative580→586. No strategy was retrained; the original frozen policies, scores,hedge coefficients,capital and source valuation rules were read only. No sealed DSR modified.\n\nThe sole permitted one-line net arithmetic repair is audit/f4.diff. Code hashes at zero-delay and delayed launch are identical to the registered repaired-code hash. Five unit tests passed, followed by ALL132zero-delay fold-policy comparisons with exact equality of original-engine/sealed outputs and independent frozen-decision replay. No delayed cell ran before that gate passed.\n\n## Net returns\n\nUnit:bps per committed-capital year;22annual totals equally weighted. Entries show estimate[95%year-blockCI],5000draws,seed20261026;paired years for execution differences. NoHolm;robustness only.\n\n'''+table(net,['family','learner','same_close','next_close','change'])+'\n\n## K2 turnover\n\nAnnual netted stock-leg turnover in original K2 units.\n\n'+table(turn,['family','learner','same_close','next_close','change'])+'\n\n## H2: RF Q_best minus AL\n\n'+table(h2rows,['execution','estimate','95% CI'])+'''\n\n## Execution definition and verification\n\nAt each close, book P&L for the preceding interval is calculated before executing the prior close's frozen target. Every delayed entry retains its original hedge-basis date; positions are not re-hedged daily and returns are not shifted wholesale. Effective G0 target sequences are frozen from the same-close engine rather than recomputed against the delayed state. Entry baskets execute atomically with original quote availability, source valuation carry and terminal-liquidation rules; no final-close new entry. Costs use actual execution-close K2 cross-series netting. These choices were recorded before any delayed result.\n\nAL's actual source policy is kind1 with fixed enter1.25/exit0.5;it is not AlwaysFlat. Earlier parent README wording calling it always-long should not be used as its definition.\n\n## F1 disposition\n\nThe one authorized RF operation-order/layout repair passed all44core/N2/xi1family-year groups. Original-N3 reproduction passed22RPgroups andRF2004–2024,but the unchanged adapter then required a missingRF2025increment receipt. It stopped with no second repair and no new N3_coreIC/CI/MCSE. This is NOT_EVALUATED,not a negative robustness result. See sibling F1_exact_RF_repair_v1/README.md and audit/STOP.json.\n'''
 (R/'README.md').write_text(text)
 question='''# Open questions — F1/F4 continuation

- F1 remains NOT_EVALUATED. The core/xi1/N2 gate passed,but the original-N3 validation adapter requested an unavailableRF2025increment receipt outside the retained formation-history scope. No second F1 repair or new IC calculation was made. This is a validation-scope limitation,not evidence that N3_core is statistically unrobust.
- F4 has no unresolved zero-delay mismatch:all132checks passed exactly after the single explicitly authorized net-expression repair. Interpret the result as an execution robustness check on repeatedly used2004–2025years,not a fresh confirmatory test or proof of deployable profitability.
- F4 freezes the original effective G0 target sequence and original hedge-basis date; it does not simulate a newly trained or execution-feedback-adapted agent. Original cost proxy,borrow omissions,capital convention and price-data limitations are unchanged.
- Previous F2/F3 manuscript table identity and incomplete historical removal-list questions remain in the parent's open_questions.md;those sealed results were not modified.
'''
 (R/'open_questions.md').write_text(question)
else:
 stop=C.js(R/'audit/STOP.json');text='# F4 next-close — 稳健性检验，已停止\n\n'+json.dumps(stop,ensure_ascii=False,indent=2)+'\n\nNo second F4 repair was applied. Record the actual trial count from trial_register.json,not the planned586. F1 remains NOT_EVALUATED; no second F1 repair.\n'
 (R/'README.md').write_text(text);(R/'open_questions.md').write_text(text)
# Actual trial register: exactly six aggregate cells on completion, not132fold trials.
trial_rows=[{'trial_index':581+i,'cell':key,'status':'completed' if scientific else 'started_before_stop','robustness_check':True,'sealed_DSR_modified':False} for i,key in enumerate(register['started_cells'])]
C.csvout(R/'trial_register.csv',trial_rows,['trial_index','cell','status','robustness_check','sealed_DSR_modified'])
# Explicitly authenticate the sealed source DSR/PBO output; never recalculate or rewrite it.
core=Path('/Users/maxiyao/FYP_runs/coint_residual_qlearning_v1');cs=C.js(core/'tier_3/complete.json');assert sha(core/'tier_3/manifest.json')==cs['manifest_sha256'];cm=C.js(core/'tier_3/manifest.json')['files'];dsr_rel='tier_3/dsr_pbo.csv.gz';dh=cm[dsr_rel];dh=dh['sha256'] if isinstance(dh,dict) else dh;assert sha(core/dsr_rel)==dh
C.dump(R/'audit/sealed_DSR_unchanged.json',{'source_file':str(core/dsr_rel),'sealed_sha256':dh,'after_sha256':sha(core/dsr_rel),'matches':True,'recalculated':False,'modified':False})
# Merge first-read source digests; own output/code files are protected by code/seal identity receipts instead.
inputs=C.js(B/'input_hashes_before.json')
for root in [F1,R]:
 for p in (root/'audit').glob('*inputs.json'):
  data=C.js(p)
  for path,v in data.items():
   if not isinstance(v,dict) or 'sha256' not in v or 'size' not in v or Path(path).is_relative_to(B):continue
   if path in inputs:assert inputs[path]==v
   inputs[path]=v
for path,v in list(C.READS.items()):
 if not Path(path).is_relative_to(B):
  if path in inputs:assert inputs[path]==v
  inputs[path]=v
C.dump(R/'audit/input_hashes_before.json',inputs);C.verify(inputs);C.dump(R/'audit/input_hashes_after.json',inputs)
oldnames=json.loads(gzip.decompress(C.raw(B/'audit/FYP_before.json.gz')));newnames=snapshot();added=sorted(set(newnames['files'])-set(oldnames['files']));missing=sorted(set(oldnames['files'])-set(newnames['files']));assert not added and not missing,(added[:20],missing[:20])
C.dump(R/'audit/end_integrity.json',{'status':'PASS','input_files_verified':len(inputs),'input_SHA_differences':0,'FYP_added':added,'FYP_missing':missing,'parent_F2_F3_and_stopped_seals_unchanged':True,'code_identity_zero_and_delayed_equal':scientific,'F1':'NOT_EVALUATED;no_second_repair','new_real_trials':count-580,'cumulative_real_trials':count})
files={str(p.relative_to(R)):sha(p) for p in R.rglob('*') if p.is_file() and p.name not in ['manifest.json','complete.json'] and p.suffix!='.log'}
C.dump(R/'manifest.json',{'files':files,'exclusions':['*.log stdout captures']})
C.dump(R/'complete.json',{'status':'complete' if scientific else 'stopped_no_second_repair','manifest_sha256':sha(R/'manifest.json'),'F1':'NOT_EVALUATED','F4_new_real_trials':count-580,'cumulative_real_trials':count,'old_DSR_modified':False})
print('Final report/integrity complete',R,'trial count',count,flush=True)
