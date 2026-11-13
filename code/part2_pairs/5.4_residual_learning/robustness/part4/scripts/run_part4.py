import ext_common as C
C.guard()
import sys,json,gzip,hashlib
sys.path.insert(0,str(C.O/'scripts'))
import part1 as P
import numpy as np
def independent_weights():
 rng=np.random.default_rng(20261026)
 ix=rng.integers(11,size=(5000,11));jx=rng.integers(11,size=(5000,11))
 return np.array([np.bincount(r,minlength=11) for r in ix],float),np.array([np.bincount(r,minlength=11) for r in jx],float)
def stat(y,beta=None):
 return y.mean() if beta is None else P.slope(y.mean(0))
def calc(y,W,beta=None):
 return W@y/11 if beta is None else P.slope(W@y/11)
def ci(a):return np.percentile(a,[2.5,97.5])
def load():
 receipt=C.js(C.O/'complete.json');mfraw=C.raw(C.O/'manifest.json')
 assert hashlib.sha256(mfraw).hexdigest()==receipt['manifest_sha256']
 mf=json.loads(mfraw)['files']
 for rel in ['part1_validation.json','part1_year_data.csv','part1_provenance.csv','scripts/part1.py']:
  assert C.sha(C.O/rel)==mf[rel],('Part1 dependency changed',rel)
 assert C.js(C.O/'part1_validation.json')['passes']
 totals,nulls,provenance=P.load_annual()
 assert P.validate(totals)['passes']
 stored=C.rows(C.O/'part1_year_data.csv')
 for r in stored:
  key=(r['family'],int(r['cost']),r['learner']);i=int(r['year'])-2004
  np.testing.assert_allclose([float(r['T']),float(r['N'])],totals[key][i,[2,0]],atol=1e-10,rtol=0)
 return totals,nulls
def main():
 assert C.sha(C.O/'preregistration_addendum.md')==(C.P4/'preregistration_addendum.sha256').read_text().strip()
 t,n=load();base=C.ledger(C.P4/'audit','before',P.S.INPUTS)
 W1,W2=independent_weights();out=[];summary=[]
 specs=[]
 for kind in ['RP','RF']:
  for cost in [2,5,10]:specs.append(('net_Qbest_minus_AL',kind,cost,t[kind,cost,'Q_best'][:,0]-t[kind,cost,'AL'][:,0],False,kind=='RF' and cost==5))
  a=np.column_stack([t[kind,c,'Q_best'][:,2] for c in [2,5,10]])
  b=np.column_stack([t[kind,c,'AL'][:,2] for c in [2,5,10]])
  specs+=[('turnover_S_Qbest',kind,'2/5/10',a,True,False),('turnover_D_Qbest_minus_AL',kind,'2/5/10',a-b,True,False)]
 if ('RP',5,'Q_best') in n:
  y=t['RP',5,'Q_best'][:,0]-n['RP',5,'Q_best'][:,:,0].mean(axis=1)
  specs.append(('net_real_minus_N2_Qbest','RP',5,y,False,False))
 else:(C.P4/'open_questions.md').write_text('N2 RP5 Q_best complete sealed annual results unavailable; no new N2 fit or trading.\n')
 for name,kind,cost,y,sl,H2 in specs:
  full=float(stat(y,sl if sl else None));a=y[:11];b=y[11:];point=[float(stat(z,True if sl else None)) for z in [a,b]]
  draws=[calc(a,W1,True if sl else None),calc(b,W2,True if sl else None)];cis=[ci(d) for d in draws]
  dp=point[1]-point[0];dc=ci(draws[1]-draws[0]);sig=[bool(lo>0 or hi<0) for lo,hi in cis]
  same=[bool(np.sign(z)==np.sign(full)) for z in point];label=['都不同向','一段同向','两段都同向'][sum(same)]
  for period,est,interval,sameflag,significant in [('2004–2014',point[0],cis[0],same[0],sig[0]),('2015–2025',point[1],cis[1],same[1],sig[1]),('late_minus_early',dp,dc,'',bool(dc[0]>0 or dc[1]<0))]:
   out.append({'statistic':name,'family':kind,'cost':cost,'period':period,'estimate':est,'ci_lo':float(interval[0]),'ci_hi':float(interval[1]),'full_sample_estimate':full,'same_sign_as_full':sameflag,'significant_CI_excludes_zero':significant,'late_minus_early':dp,'difference_ci_lo':float(dc[0]),'difference_ci_hi':float(dc[1]),'n_years_per_half':11,'H2_reference':H2,'descriptive_only':True})
  summary.append({'statistic':name,'family':kind,'cost':cost,'full':full,'early':point[0],'early_ci':cis[0].tolist(),'early_significant':sig[0],'late':point[1],'late_ci':cis[1].tolist(),'late_significant':sig[1],'direction':label,'difference':dp,'difference_ci':dc.tolist()})
 C.csvout(C.P4/'part4_split_half.csv',out)
 C.dump(C.P4/'audit/summary.json',summary)
 lines=['# Part4 前后半段一致性（描述性）','','每段仅11年，功效较低；22个年份已被反复用于开发。本检查不构成新的确认性检验，不做Holm。5000次段内year bootstrap，seed20261026；前后段独立抽样。符号与全样本相比，显著仅指95%CI不含0。','','|统计量|残差/成本|前段估计 [95%CI]|显著|后段估计 [95%CI]|显著|方向|','|---|---|---|---|---|---|---|']
 for r in summary:
  lo,hi=r['early_ci'];lo2,hi2=r['late_ci']
  lines.append('|'+r['statistic']+'|'+r['family']+'/'+str(r['cost'])+f"|{r['early']:.5f} [{lo:.5f},{hi:.5f}]|"+('是' if r['early_significant'] else '否')+f"|{r['late']:.5f} [{lo2:.5f},{hi2:.5f}]|"+('是' if r['late_significant'] else '否')+'|'+r['direction']+'|')
 lines+=['','H2对应RF、5bps的net_Qbest_minus_AL行，未重复计作另一项检验。后段减前段的估计/独立抽样CI完整保存在part4_split_half.csv。源为Part1相同封存年度单元，Part1净收益斜率复现及逐年数据核对PASS；见audit/inputs_before/after.json。新增真实trial0，累计580。']
 (C.P4/'part4_summary.md').write_text('\n'.join(lines)+'\n')
 C.verify(base);after=C.ledger(C.P4/'audit','after',P.S.INPUTS);assert base==after
 C.dump(C.P4/'complete.json',{'status':'complete','new_real_trials':0,'real_trials':580,'n_statistics':len(summary),'rows':len(out),'Part1_validation':'PASS','inputs_unchanged':True})
 print('\n'.join(lines),flush=True)
if __name__=='__main__':main()
