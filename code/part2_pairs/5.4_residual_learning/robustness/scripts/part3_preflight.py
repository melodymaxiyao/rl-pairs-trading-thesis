import support as S
S.guard()
import json,csv,io,hashlib,datetime
from pathlib import Path
R=Path('/Users/maxiyao/FYP_runs/ou_realstrength_v1');O=S.O/'part3'
paths=['preregistration.json','calibration.json','request.md','scripts/calibration.py','scripts/simulation.py','scripts/pipeline.py','scripts/source_stats.py','scripts/residual_models.py','audit/calibration_targets.json','audit/import_registration.json','vendor/residual/outputs/tables/residual_transfer.json','complete.json','manifest.json','code_registration.json']
texts={rel:S.raw(R/rel) for rel in paths}
complete=json.loads(texts['complete.json']);assert hashlib.sha256(texts['manifest.json']).hexdigest()==complete['manifest_sha256']
mf=json.loads(texts['manifest.json'])['files'];cr=json.loads(texts['code_registration.json'])['files']
for rel,b in texts.items():
 if rel in mf:
  val=mf[rel];expected=val['sha256'] if isinstance(val,dict) else val;assert hashlib.sha256(b).hexdigest()==expected,(rel,'manifest')
 elif rel in cr:assert hashlib.sha256(b).hexdigest()==cr[rel],rel
reg=json.loads(texts['preregistration.json']);targets=json.loads(texts['audit/calibration_targets.json']);transfer=json.loads(texts['vendor/residual/outputs/tables/residual_transfer.json'])
assert hashlib.sha256(texts['vendor/residual/outputs/tables/residual_transfer.json']).hexdigest()==targets['source_sha256']
request=texts['request.md'].decode();assert 'RF, real minus N2 where marked' in request
source=Path('/Users/maxiyao/FYP_runs/partial_coint_kalman_v1/tradable_target_summary.csv')
raw=S.raw(source);rows=list(csv.DictReader(io.StringIO(raw.decode())));chosen=[]
for h in [1,5]:
 ref=transfer['IC_h1_h5']['RF'][str(h)];assert ref['real'][0]['era']=='pooled'
 match=[r for r in rows if r['signal']=='core_headline' and r['family']=='RF' and r['scope']=='earlier_chapter_original_scope' and int(r['h'])==h]
 assert len(match)==1
 r=match[0];assert float(r['original_real_minus_N2'])==ref['paired_real_minus_N2']==reg['calibration']['targets'][f'IC_excess_h{h}']
 assert int(r['common_days'])==ref['paired_common_days'];chosen.append(r)
before=dict(S.INPUTS);S.dump(O/'input_hashes_before.json',before)
evidence=[]
for rel,ranges in [('request.md',[(40,50)]),('scripts/calibration.py',[(9,12),(24,24),(40,46)]),('scripts/pipeline.py',[(7,15),(18,29),(37,47)]),('scripts/simulation.py',[(4,27)])]:
 ls=texts[rel].decode().splitlines()
 for lo,hi in ranges:evidence.append({'source':str(R/rel),'lines':[[i,ls[i-1]] for i in range(lo,min(hi,len(ls))+1)]})
S.dump(O/'source_evidence.json',evidence)
assert 'returns=np.diff(paths,axis=1)' in texts['scripts/pipeline.py'].decode()
assert 'future[wi,:,:T-h+1,hi]=sums[:,h:]-sums[:,:-h]' in texts['scripts/pipeline.py'].decode()
assert 'alpha' not in texts['scripts/simulation.py'].decode()
old=json.loads(texts['calibration.json'])['chosen'];records=[]
for label in ['old_sealed','new_not_run']:
 row={'market':label,'h0':old['h0'] if label=='old_sealed' else '', 'w':old['w'] if label=='old_sealed' else '', 'family':'RF','scope':'earlier_chapter_original_scope','status':'historical copied; not rerun' if label=='old_sealed' else 'STOPPED_DEFINITION_GATE'}
 for key,h in [('IC_excess_h1',1),('IC_excess_h5',5),('increment_R2_excess',None),('sigma_epsilon_median',None)]:
  row[key+'_target']=reg['calibration']['targets'][key] if label=='old_sealed' or h is None else chosen[h==5]['tradable_real_minus_N2']
  row[key+'_fitted']=old['fitted'][key] if label=='old_sealed' else ''
  row[key+'_relative_error']=old['relative_errors'][key] if label=='old_sealed' else ''
 row['all_within_25pct']=False if label=='old_sealed' else '';records.append(row)
def csvout(path,rs,cols=None):
 cols=cols or list(dict.fromkeys(k for r in rs for k in r))
 with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rs)
csvout(S.O/'part3_calibration.csv',records)
csvout(S.O/'part3_learners.csv',[],['market','learner','true_net','true_net_SE','breakeven_cost','turnover','holding_days','learned_entry_Q','true_entry_Q','status'])
csvout(S.O/'part3_predictions.csv',[{'id':k,'prediction':p,'supported':'','supporting_numbers':'','status':'NOT_EVALUATED_DEFINITION_GATE'} for k,p in [('R1','Q_POW entryQ bias>100bps'),('R2','Q_DBL lowers entryQ bias by>=40%'),('R3','Q_best true net>Q_POW'),('R4','Band>AL>Q_best')]])
val={'status':'STOPPED_DEFINITION_GATE','definition_gate_passed':False,'original_parameter_reproduction':'NOT_RUN','synthetic_tradable_hand_example':'NOT_RUN; no approved distinct synthetic hedge/intercept mapping','new_calibration_points_run':0,'new_synthetic_markets_run':0,'new_training_runs':0,'new_real_trading_trials':0,'real_trial_count':580,'reason':'Model/simulate yields residual-level OU+RW paths only. features uses diff(path). Future target is sums of those residual increments, without a separate underlying hedge return or rolling alpha representation. Renaming that sum or imposing alpha=0 would reinterpret the missing mechanism, prohibited by Part3 step3.'}
S.dump(S.O/'part3_validation.json',val)
notes=['# Part3 stopped at definition gate','Original calibration is **RF**, not RP/family pool. request.md:43 explicitly says RF. audit/calibration_targets.json authenticates vendor/residual/outputs/tables/residual_transfer.json. Scope is pooled, paired real-minus-equal-mean N2, earlier-chapter original scope. Exact keys IC_h1_h5.RF.1/5.paired_real_minus_N2; common days5512/5420 match core_headline/RF/earlier_chapter_original_scope. No row chosen by closeness.',
'New target excess IC h1='+chosen[0]['tradable_real_minus_N2']+', h5='+chosen[1]['tradable_real_minus_N2']+'. Increment R2 excess remains '+str(reg['calibration']['targets']['increment_R2_excess'])+'. Copied targets, not new synthetic results. Source: '+str(source)+' [signal=core_headline,family=RF,scope=earlier_chapter_original_scope,h=1/5].',
'**STOP: prompt Part3 step3 applies.** simulation.py:4–27 returns only OU+RW residual levels. pipeline.py:9 takes increments; :25–27 takes their h-period sum. No distinct underlying intercept-free hedge returns and rolling alpha are carried. Thus Y_original=sum(hedged_returns-alpha_t) and Y_tr=sum(hedged_returns) cannot be separately constructed without another model/interpretation. residual_models.regress has a generic alpha helper, but calibration_panel does not call pair_rolling/factor_window/regress or carry their coefficients.',
'No calibration, fit, simulation, learner comparison or old-parameter reproduction run. New h0/w blank; R1–R4 NOT_EVALUATED, not false. Historical h20,w.6 in part3_calibration.csv is copied context only. part3_learners.csv has only its schema. This is the requested stop, not a failed replication.',
'Exact source lines in source_evidence.json; input hashes before/after unchanged. Real trading trials remain580. Parts1/2 complete; this stop does not change their decisions.']
(O/'README.md').write_text('\n\n'.join(notes)+'\n')
for p,r in before.items():assert S.sha(p)==r['sha256']
S.dump(O/'input_hashes_after.json',{k:S.INPUTS[k] for k in before})
assert S.js(O/'input_hashes_before.json')==S.js(O/'input_hashes_after.json')
S.dump(O/'complete.json',dict(val,completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),preregistration_sha256=S.sha(O/'preregistration.md')))
print('PART3 STOPPED — exact definition gap; RF original scope verified',flush=True)
