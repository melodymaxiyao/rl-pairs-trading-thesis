import support as S
S.guard()
import ast,csv,io,json,gzip
import numpy as np
import pandas as pd
from pathlib import Path
ROOT=Path('/Users/maxiyao/FYP_runs/coint_residual_qlearning_v1')
YEARS=list(range(2004,2026));COSTS=[2,5,10];LEARNERS=['AL','Band','SPI','PlugIn','Q_POW','Q_best']
BETA=(np.array(COSTS)-np.mean(COSTS))/np.sum((np.array(COSTS)-np.mean(COSTS))**2)
EXPECTED_PREREG='5373a4b2e8b57e98027d67aa02e254125b74ecb7e4ca30a448c3f9b54ff66d10'
def draw_weights():
 ix=np.random.default_rng(20261026).integers(22,size=(5000,22))
 return np.array([np.bincount(r,minlength=22) for r in ix],float)
def slope(v):
 a=np.asarray(v);return (a-a.mean(axis=-1,keepdims=True))@BETA
def interval(point,draws):
 v=np.asarray(draws);finite=np.isfinite(v);lo,hi=np.quantile(v[finite],[.025,.975]) if finite.any() else [None,None]
 return {'estimate':float(point) if np.isfinite(point) else None,'ci_lo':float(lo) if lo is not None else None,'ci_hi':float(hi) if hi is not None else None,'undefined_draws':int((~finite).sum()),'p_one_sided':float(np.mean(v>=0)) if finite.all() else None}
def ratios(matrix,W):
 den=matrix[:,0];num=matrix[:,2];bd=W@den;bn=W@num
 return interval(num.mean()/den.mean() if den.mean()!=0 else np.nan,np.divide(bn,bd,out=np.full(len(W),np.nan),where=bd!=0))
def holm(ps):
 p=np.asarray(ps);order=np.argsort(p,kind='stable');adj=np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]);out=np.empty(len(p));out[order]=np.minimum(adj,1);return out
def checked(p,mf):
 p=Path(p);rel=str(p.relative_to(ROOT));assert rel in mf,('unsealed required source',rel)
 e=mf[rel];e=e['sha256'] if isinstance(e,dict) else e
 b=S.raw(p);assert __import__('hashlib').sha256(b).hexdigest()==e,('sealed mismatch',p)
 return b
def load_annual():
 seal=S.js(ROOT/'tier_3/complete.json')
 assert seal['protocol_complete'] and seal['real_trials']==572
 assert S.sha(ROOT/'tier_3/manifest.json')==seal['manifest_sha256']
 mf=S.js(ROOT/'tier_3/manifest.json')['files']
 code=S.js(ROOT/'code_registration.json');assert S.sha(ROOT/'code_registration.json')==seal['code_registration_sha256']
 p=ROOT/'vendor/TAV/inference.py';assert S.sha(p)==code['files']['vendor/TAV/inference.py']
 ns={'np':np};tree=ast.parse(S.raw(p).decode());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='blocks')
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec'),ns);names,W0=ns['blocks'](YEARS,draws=5000);assert names==YEARS
 np.testing.assert_array_equal(draw_weights(),W0)
 totals={};nulls={};null_replicas={};provenance=[]
 for kind in ['RP','RF']:
  for cost in COSTS:
   for learner in LEARNERS:
    years=[];nrows=[];nullokay=True;ids=None
    for year in YEARS:
     S.resources()
     p=ROOT/f'outputs/cells/{kind}_{year}_G0_c{cost}_{learner}/per_series_year.csv.gz'
     b=checked(p,mf);f=pd.read_csv(io.BytesIO(b),compression='gzip',keep_default_na=False,float_precision='round_trip',usecols=['world','replicate','weight','gross_bps','net_bps','K2'])
     real=f[f.world=='real'];assert len(real)
     weight=real.weight.to_numpy(float);assert abs(weight.sum()-1)<1e-12
     years.append([float(np.sum(real.net_bps.to_numpy(float)*weight)),float(np.sum(real.gross_bps.to_numpy(float)*weight)),float(np.sum(real.K2.to_numpy(float)*weight))])
     nf=f[f.world=='N2'];reps=sorted(nf.replicate.unique().tolist())
     if len(reps)!=2 or (ids is not None and reps!=ids):nullokay=False
     ids=reps
     vals=[]
     for rep in reps:
      q=nf[nf.replicate==rep];w=q.weight.to_numpy(float);assert abs(w.sum()-1)<1e-12
      vals.append([float(np.sum(q.net_bps.to_numpy(float)*w)),float(np.sum(q.gross_bps.to_numpy(float)*w)),float(np.sum(q.K2.to_numpy(float)*w))])
     nrows.append(vals)
     provenance.append({'family':kind,'learner':learner,'cost':cost,'year':year,'source':str(p),'source_sha256':__import__('hashlib').sha256(b).hexdigest()})
    totals[(kind,cost,learner)]=np.array(years)
    if nullokay:nulls[(kind,cost,learner)]=np.array(nrows);null_replicas[(kind,cost,learner)]=ids
 return totals,nulls,provenance
def validate(totals):
 refpath=ROOT/'post_tier3_review_v1/followup_runs/1793580484598758000/results.json'
 tablepath=Path('/Users/maxiyao/FYP_runs/thesis_tables_v3_pairs/outputs/tables/T7_5.csv')
 table=list(csv.DictReader(io.StringIO(S.raw(tablepath).decode())))
 assert {r['source_sha256'] for r in table if r['part']=='1'}=={S.sha(refpath)}
 ref=S.js(refpath);W=draw_weights();out=[]
 for kind in ['RP','RF']:
  a=np.column_stack([totals[(kind,c,'Q_best')][:,0]-totals[(kind,c,'AL')][:,0] for c in COSTS]);r=interval(slope(a.mean(0)),slope(W@a/22))
  old=next(v for v in ref['slopes'] if v['definition']==kind)
  for new,key in [('estimate','estimate'),('ci_lo','ci95_low'),('ci_hi','ci95_high')]:
   np.testing.assert_allclose(r[new],old[key],atol=1e-9,rtol=0)
   tt=next(v for v in table if v['part']=='1' and v['definition']==kind)
   np.testing.assert_allclose(r[new],float(tt[key]),atol=1e-9,rtol=0)
  out.append({'family':kind,'recomputed':r,'sealed':old,'passes':True})
 return {'passes':True,'comparisons':out,'trial_count_current':580,'new_real_trading_trials':0,'source':str(refpath),'tolerance':1e-9}
def csvout(name,records):
 cols=list(dict.fromkeys(k for r in records for k in r))
 with (S.O/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(records)
def main():
 assert S.sha(S.O/'part1/preregistration.md')==EXPECTED_PREREG
 before=S.snapshot();S.dump(S.O/'audit/FYP_inventory_before.json',before);print('FYP_BASELINE',len(before['files']),'skipped',len(before['skipped']),flush=True)
 t,n,p=load_annual();S.dump(S.O/'audit/part1_inputs_before.json',S.INPUTS)
 try:v=validate(t)
 except Exception as ex:
  S.dump(S.O/'part1_validation.json',{'passes':False,'error':str(ex),'later_parts_started':False});raise
 S.dump(S.O/'part1_validation.json',v)
 W=draw_weights();rows=[];slopes={};bs={};ns=[];nullrows=[];annual=[]
 for kind in ['RP','RF']:
  for learner in LEARNERS:
   turn=np.column_stack([t[kind,c,learner][:,2] for c in COSTS]);net=np.column_stack([t[kind,c,learner][:,0] for c in COSTS])
   sl=interval(slope(turn.mean(0)),slope(W@turn/22));ratio=ratios(turn,W);neti=interval(slope(net.mean(0)),slope(W@net/22))
   slopes[kind,learner]=sl;bs[kind,learner]=slope(W@turn/22)
   rows.append({'family':kind,'learner':learner,'T2':turn[:,0].mean(),'T5':turn[:,1].mean(),'T10':turn[:,2].mean(),'S':sl['estimate'],'S_ci_lo':sl['ci_lo'],'S_ci_hi':sl['ci_hi'],'S_p':sl['p_one_sided'],'R':ratio['estimate'],'R_ci_lo':ratio['ci_lo'],'R_ci_hi':ratio['ci_hi'],'R_undefined_draws':ratio['undefined_draws'],'net_slope':neti['estimate'],'net_slope_ci_lo':neti['ci_lo'],'net_slope_ci_hi':neti['ci_hi'],'n_years':22,'new_real_trials':0})
   for j,year in enumerate(YEARS):
    for k,cost in enumerate(COSTS):annual.append({'family':kind,'learner':learner,'year':year,'cost':cost,'T':turn[j,k],'N':net[j,k]})
  for other in ['AL','Band']:
   r=interval(slopes[kind,'Q_best']['estimate']-slopes[kind,other]['estimate'],bs[kind,'Q_best']-bs[kind,other]);ns.append({'family':kind,'comparison':'Q_best minus '+other,**r,'in_Holm':other=='AL'})
  if all((kind,c,'Q_best') in n for c in COSTS):
   # Average matched replica annual totals before slope; replica records also retained.
   nt=np.column_stack([n[kind,c,'Q_best'][:,:,2].mean(axis=1) for c in COSTS])
   real=np.column_stack([t[kind,c,'Q_best'][:,2] for c in COSTS])
   for label,a in [('N2_mean',nt),('real_minus_N2',real-nt)]:
    r=interval(slope(a.mean(0)),slope(W@a/22));rr=ratios(a,W) if label=='N2_mean' else {}
    nullrows.append({'family':kind,'world':label,**r,'R':rr.get('estimate'),'R_ci_lo':rr.get('ci_lo'),'R_ci_hi':rr.get('ci_hi'),'descriptive_only':True})
  else:nullrows.append({'family':kind,'world':'N2','estimate':None,'note':'Required sealed cost/year/replica coverage unavailable; no new N2 run.'})
 tests=[]
 for id,kind,diff in [('C1','RP',False),('C2','RF',False),('C3','RP',True),('C4','RF',True)]:
  r=next(r for r in ns if r['family']==kind and r['comparison']=='Q_best minus AL') if diff else slopes[kind,'Q_best']
  tests.append({'id':id,'family':kind,'statistic':'S_Q_best minus S_AL' if diff else 'S_Q_best',**{k:r[k] for k in ['estimate','ci_lo','ci_hi','p_one_sided']}})
 for r,padj in zip(tests,holm([r['p_one_sided'] for r in tests])):r['holm_p']=float(padj);r['reject']=bool(padj<=.05 and r['estimate']<0)
 csvout('part1_cost_adaptation.csv',rows);csvout('part1_holm.csv',tests);csvout('part1_slope_differences.csv',ns);csvout('part1_N2_descriptive.csv',nullrows);csvout('part1_year_data.csv',annual);csvout('part1_provenance.csv',p)
 after=S.snapshot();S.dump(S.O/'audit/FYP_inventory_after_part1.json',after)
 assert before['files']==after['files'],('FYP inventory changed',set(after['files'])-set(before['files']))
 saved=dict(S.INPUTS)
 for path,rec in saved.items():assert S.sha(path)==rec['sha256']
 S.dump(S.O/'audit/part1_inputs_after.json',S.INPUTS)
 text='# Part 1 cost adaptation\n\nNo new trading or fitting; cumulative real trading trials580. G0/K2 real, annual sealed capital-weighted totals, 22 equally weighted years. Bootstrap5000 seed20261026; p=mean(stat*>=0). Holm family4.\n\nNet-slope reproduction: PASS.\n\n|ID|Estimate|95%CI|p|Holm p|Reject negative null|\n|---|---:|---|---:|---:|---|\n'
 for r in tests:text+='|{id}|{estimate:.8f}|[{ci_lo:.8f},{ci_hi:.8f}]|{p_one_sided:.6f}|{holm_p:.6f}|{reject}|\n'.format(**r)
 text+='\nAnnual units are weighted saved per-series K2 totals, matching the sealed follow-up, rather than day-weighted 252*mean across unequal years. Net slope is bps/committed-capital-year per bps. AL degenerate slopes retained. N2 only existing sealed cells; descriptive. Input hashes unchanged; FYP file inventory unchanged. See part1_provenance.csv and audit/part1_inputs_before/after.json.\n'
 (S.O/'part1/README.md').write_text(text);S.dump(S.O/'part1/complete.json',{'status':'complete','preregistration_sha256':EXPECTED_PREREG,'real_trials':580,'new_real_trading_trials':0,'input_files':len(saved),'tests':tests})
 print(text,flush=True)
if __name__=='__main__':main()
