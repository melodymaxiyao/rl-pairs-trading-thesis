import sys,json,csv,re,hashlib
from pathlib import Path
sys.path.insert(0,'/Users/maxiyao/FYP_runs/closing_checks_v1/part4/scripts')
import ext_common as C
C.guard();C.resources()
D=C.js(C.P6/'audit/documents.json')
# Each cell consists of verbatim, line-bounded quotations. No experiment statistic is recalculated.
S={}
def add(n,level,summary,criteria=None,rates=None,oos=None,conclusion=None,sample=None,limits=None):
 S[n]={'evidence_level':level,'editorial_scope_summary':summary,'screening_criteria':criteria,'pass_rates':rates,'out_of_sample_comparison':oos,'experiment_conclusion':conclusion,'sample_period_and_pairs':sample,'limitations':limits}
def q(doc,*ranges):return [doc,list(ranges)]
add('economic_peer_residual_screen','开发观察；无交易或样本外检验','全样本残差诊断，不是筛选有效性的检验',q('README.md',(11,23)),None,None,q('README.md',(29,33),(84,92)),q('README.md',(37,43),(94,131)),q('README.md',(3,7)))
add('economic_peer_residual_screen_rolling','开发观察','滚动形成期诊断；保留正面个别结果，不选优',q('README.md',(10,14),(27,32),(50,60)),None,q('README_v2.md',(13,17),(98,102)),q('README_v2.md',(237,239)),q('README.md',(87,87)),q('README_v2.md',(3,9)))
for v in range(1,4):add('universe_expansion_v'+str(v),'只有预备文档；无可用结果','不能把预注册或准备状态当作完成结果',q('README.md',(3,3)),None,None,None,None,q('README.md',(1,5)))
add('universe_expansion_v4','开发观察；train-only','训练期事件有正面改善，未授权交易',q('report.md',(24,24),(55,55)),q('report.md',(3,5)),None,q('report.md',(7,11)),q('report.md',(3,5)),q('report.md',(43,47),(95,101)))
add('universe_sector_expansion_v1','开发观察；无交易检验','扩大行业宇宙，但不是样本外盈利确认',q('README.md',(5,9)),q('README.md',(24,24)),None,q('README.md',(225,225)),q('README.md',(5,5),(24,24)),q('README.md',(3,3),(11,11)))
add('cross_sector_fast_universe_v1','开发观察；内层持久性诊断','FAST63 未显示超过随机游走的持久性；不等于交易策略无效',q('README.md',(3,3),(11,19)),q('README.md',(68,70)),None,q('README.md',(88,88)),q('README.md',(17,17),(68,70)),q('README.md',(13,13),(29,29)))
for v in [1,2]:
 add('zreversion_diagnostic_v'+str(v),'开发观察；'+('v1 完整保护检查未通过，未视为封存完成' if v==1 else 'v2 验证复用 v1 数值，不是第二次独立实验'), '训练期研究方向诊断，无新增交易结果',q('report.md',(3,3),(31,31)),None,None,q('report.md',(7,15)),q('report.md',(3,3),(19,25)),q('report.md',(43,43)) if v==1 else q('README.md',(3,11)))
add('screening_v2','开发观察；无交易检验','四个用户指定配对的分折形成期筛选',q('summary.md',(16,20)),q('summary.md',(24,29)),None,q('summary.md',(29,29)),q('summary.md',(3,14)))
add('rolling_v3','开发观察；非独立确认','12月主设定结果弱；24月个别配对正收益须保留',q('report.md',(30,31),(36,36)),q('report.md',(9,26)),q('report.md',(9,26)),q('report.md',(34,34)),q('report.md',(3,5)),q('report.md',(34,36)))
add('stock_pairs_rolling_v1','开发观察；未运行交易','KO/PEP 和 XOM/CVX 的滚动筛选，通过不代表盈利',q('report.md',(20,20)),q('report.md',(7,8)),None,q('report.md',(21,23)),q('report.md',(7,8),(22,23)),q('report.md',(3,3)))
add('expanded_convergence_gate_v1','开发观察；预注册阈值受此前结果启发','F 的筛选组有正面描述性结果，未构成独立确认',q('REPORT.md',(53,55)),q('REPORT.md',(3,7)),q('REPORT.md',(82,87)),q('REPORT.md',(97,97),(122,122)),q('REPORT.md',(3,7)),q('REPORT.md',(107,107),(122,122)))
add('holdout_expanded_personal_gate_v1','确认性留出学习器比较；筛选增益为描述性','0/3 学习器比较通过，筛选差的区间含零，不能证明筛选无效',q('README.md',(3,3)),q('REPORT.md',(3,3)),q('REPORT.md',(9,15)),q('REPORT.md',(5,5)),q('README.md',(3,3)),q('REPORT.md',(19,19)))
add('refresh_spectrum_stage1_v1','开发观察；预先登记的探索性阶段','C1 通过；保留正面探索性证据，但只授权下一研究阶段',q('REPORT.md',(3,5)),None,q('REPORT.md',(14,14)),q('REPORT.md',(14,14)),q('REPORT.md',(3,5),(14,14)),q('REPORT.md',(3,3)))
add('drift_aligned_reversion_v1','准备完成但未执行','不把计划的检验或 cell 数记为已运行',q('README.md',(25,29)),None,None,q('README.md',(3,3)),None,q('README.md',(3,3),(35,42),(67,67)))
add('pair_property_persistence_v1','开发观察；train-only','速度不持久，部分共变/尺度性质持久；未选新交易池',q('README.md',(17,23),(38,49)),q('summary.json',(3,3)),None,q('summary.json',(4,7)),q('README.md',(17,22)),q('README.md',(3,3)))
add('ggr_large_universe_dev_v1','开发观察','开发期10 bps净收益为正，但不足以确认盈利',q('REPORT.md',(69,73)),None,q('REPORT.md',(7,13)),q('REPORT.md',(13,13)),q('REPORT.md',(3,5),(67,67)),q('REPORT.md',(3,3)))
add('ggr_large_universe_confirm_2004_2008_v1','确认性；一次性留出检验','GGR 冻结策略在10 bps下两项主检验均通过；非距离筛选相对随机筛选的因果比较',q('REPORT.md',(5,5),(56,56)),None,q('REPORT.md',(9,12)),q('REPORT.md',(3,3)),q('REPORT.md',(1,1),(20,20)),q('REPORT.md',(33,35),(54,54)))
add('ggr_confirm_robustness_v1','事后敏感性；只有停止的早期包装器文档','无本版本已完成数值结果',None,None,None,None,None,q('README.md',(1,5)))
for v in [2,3]:
 add('ggr_confirm_robustness_v'+str(v),'事后敏感性；'+('v2 保留完成前审计，最终封存在 v3' if v==2 else 'v3 完成；与 v2 数值重复，不是独立确认'), '支持结果对成本和2008年依赖明显；不改写原确认结果',q('REPORT.md',(10,10)),None,q('REPORT.md',(2,6),(45,50),(58,58),(124,124)),q('REPORT.md',(1,10),(128,128)),q('REPORT.md',(1,1)),q('README.md',(3,7 if v==2 else 9)))
add('meta_filter_dev_v1','开发观察','选定V3；D1改善但D2未改善，保留两个时期',q('REPORT.md',(49,57)),None,q('REPORT.md',(7,13),(24,30),(46,46)),q('REPORT.md',(3,5)),q('REPORT.md',(7,7),(24,24)),q('REPORT.md',(61,61)))
add('meta_filter_confirm_2004_2008_v1','确认性；一次性留出检验','0/3主检验通过；未确认改善，不等同于证明无效',q('REPORT.md',(5,5),(46,48)),None,q('REPORT.md',(9,13)),q('REPORT.md',(3,3)),q('REPORT.md',(1,1),(21,28)),q('REPORT.md',(48,54)))
add('qlearning_zpos_v3','开发观察；Q测试新但基准已看过','冻结门槛后的Q未优于基准，不是独立筛选增益检验',q('README.md',(7,11)),q('summary_lines.json',(8,8)),q('README.md',(23,27)),q('summary_lines.json',(4,7)),q('README.md',(207,207)),q('README.md',(9,11)))
add('qlearning_holdout_pre2018_v1','确认性留出学习器检验','2013–2017 上0/3通过；筛选本身并非随机对照',q('README.md',(7,7),(13,13)),q('README.md',(13,13)),q('README.md',(19,21)),q('README.md',(23,23)),q('README.md',(7,7),(13,13)),q('README.md',(99,99)))
add('mb_frozen_2004_2008_v1','冻结学习器留出检验；市场时期此前已被其他检验使用','两种学习器各0/3；MB_f净收益点估计为正但CI含零',q('README.md',(5,9)),q('README.md',(19,19),(86,86)),q('README.md',(23,25),(90,92)),q('README.md',(19,19),(86,86)),q('README.md',(3,9),(153,153)),q('README.md',(3,3),(165,165)))
add('residual_learnability_v1','开发证据；残差层面诊断','R3/R4 正点估计但均未支持，不新增筛选交易池',q('README.md',(7,19),(38,40)),None,q('RESULTS.md',(7,7),(19,19)),q('RESULTS.md',(25,25)),None,q('RESULTS.md',(1,1),(51,52)))
assert set(S)==set(D) and len(S)==29
fields=['screening_criteria','pass_rates','out_of_sample_comparison','experiment_conclusion','sample_period_and_pairs','limitations']
rows=[];gaps=[];evidence=[]
for name,record in S.items():
 d=D[name];row={'experiment':name,'family':d['family'],'result_location':d['root'],'evidence_level':record['evidence_level'],'editorial_scope_summary':record['editorial_scope_summary']};refs={}
 for field in fields:
  spec=record[field]
  if spec is None:
   row[field]='';gaps.append((name,field));continue
  docname,ranges=spec;doc=d['documents'][docname];lines=doc['text'].splitlines();parts=[];links=[]
  assert not docname.startswith('audit/')
  assert C.sha(doc['path'])==doc['sha256']
  for a,b in ranges:
   assert 1<=a<=b<=len(lines),(name,docname,a,b,len(lines))
   quoted='\n'.join(lines[a-1:b]);parts.append(quoted)
   e={'experiment':name,'field':field,'source_file':doc['path'],'source_sha256':doc['sha256'],'line_start':a,'line_end':b,'quote':quoted}
   evidence.append(e);links.append({k:v for k,v in e.items() if k not in ['experiment','field','quote']})
  row[field]='\n[…]\n'.join(parts);refs[field]=links
 row['source_evidence_json']=json.dumps(refs,ensure_ascii=False);rows.append(row)
C.csvout(C.P6/'part6_screening_summary.csv',rows)
C.dump(C.P6/'audit/excerpts.json',evidence)
# Extraction tests: literal quotations, exact 29 registered names, hashes, null string round-trip, positive findings retained.
for e in evidence:
 doc=next(x for x in D[e['experiment']]['documents'].values() if x['path']==e['source_file'])
 assert e['quote']=='\n'.join(doc['text'].splitlines()[e['line_start']-1:e['line_end']])
assert len({r['experiment'] for r in rows})==29
assert '2/2' in next(r for r in rows if r['experiment']=='ggr_large_universe_confirm_2004_2008_v1')['experiment_conclusion']
assert 'PASS' in next(r for r in rows if r['experiment']=='refresh_spectrum_stage1_v1')['experiment_conclusion']
import io
f=io.StringIO();w=csv.writer(f);w.writerow(['null','']);f.seek(0);assert next(csv.reader(f))==['null','']
# Family labels concern evidence for screening efficacy; failed superiority is not proof of zero efficacy.
def cite(name,doc,a):
 d=D[name]['documents'][doc];return '['+name+' '+doc+':'+str(a)+']('+d['path']+':'+str(a)+')'
family='''# Part 6 — 开发阶段筛选证据汇总

每个实验的原文、行号与 SHA-256 均列于 part6_screening_summary.csv 及 audit/excerpts.json。以下证据级别针对“筛选有效”的主张。预注册不自动使已反复查看时期成为独立确认；未拒绝优越性假设不等于证实无效。版本重复不算独立证据。

| 实验族 | 判定 | 证据及范围 |
| --- | --- | --- |
| 经济同业和行业宇宙 | 仅有开发观察，未确认 | 扩展 v4 的训练期收敛指标改善；行业宇宙和 FAST 筛选没有独立交易留出检验。FAST 的验证期概率差为 0.01 个百分点，95% CI [−2.02, 2.36] 个百分点。v1–v3 只有准备文档，不能当作已完成的失败结果。 SOURCE1 |
| 回归和收敛筛选 | 仅有开发观察，未确认 | F gate 的部分测试组和 refresh-spectrum C1 有正面开发结果；后者 PASS（ρ = −0.324284，p = 0.000500），但两个时期均曾查看。holdout personal gate 的同模型 GP−GU 差 +100.68 [−334.24, 485.02] bps 未确认增益。准备中的 drift-aligned 实验没有结果。 SOURCE2 |
| 距离法和过滤器 | 有确认性证据支持筛选有效 | 证据限于冻结 GGR 距离法策略在 2004–2008、10 bps 下的正月均净收益及正 SPY alpha：2/2 通过，Holm p 均为 0.0317655；这不是距离筛选相对随机筛选的直接实验。事后检查显示成本 13.6 bps 时不再通过，2008 年贡献 66.85%，剔除 2008 年后不通过。V3 meta-filter 的留出改善为 0/3，未确认。 SOURCE3 |
| 筛选后的学习器留出检验 | 仅有开发观察，未确认 | 存在冻结的学习器留出检验，但没有独立确认“筛选比不筛选更有效”的对照。2013–2017 MB_zvs 为 0/3；2004–2008 两个 MB 配方各 0/3。MB_f 净收益 +140.59 [−33.54, 388.55] bps 的正点估计保留。该标签不把这些未通过的学习器检验改称未执行。 SOURCE4 |
| 残差层面的筛选 | 仅有开发观察，未确认 | R3 +27.93 [−51.04, 123.9]、R4 +27.24 [−67.94, 117.8]，Holm p 均 0.7726；报告标为 development evidence，未新增筛选交易池。 SOURCE5 |

## Thesis §5.2 — suggested wording

Development-stage screens covered economic peers, industry universes, convergence measures, distance selection and residual properties; their evidence should not be pooled as independent confirmations. Some development results were positive, including selected F-gate groups and the refresh-spectrum diagnostic, but they did not independently establish a screening improvement. The frozen GGR strategy did pass both registered profitability tests on 2004–2008 at 10 bps, although post-hoc checks showed dependence on transaction costs and on 2008; this did not directly identify an advantage over random pair selection. The meta-filter and screened-learner holdout comparisons did not confirm the proposed improvements, and the residual-screening contrasts also remained inconclusive. Thus, the evidence does not support the blanket statement that every screening method failed, nor does it establish a general, reliable benefit from screening.

只读摘录；无统计重算、无筛选、无回测。新增真实交易 trial = 0；累计仍为 580。
'''
family=family.replace('SOURCE1',cite('universe_expansion_v4','report.md',7)+'；'+cite('cross_sector_fast_universe_v1','README.md',88))
family=family.replace('SOURCE2',cite('expanded_convergence_gate_v1','REPORT.md',82)+'；'+cite('refresh_spectrum_stage1_v1','REPORT.md',14)+'；'+cite('holdout_expanded_personal_gate_v1','REPORT.md',15))
family=family.replace('SOURCE3',cite('ggr_large_universe_confirm_2004_2008_v1','REPORT.md',9)+'；'+cite('ggr_confirm_robustness_v3','REPORT.md',2)+'；'+cite('meta_filter_confirm_2004_2008_v1','REPORT.md',3))
family=family.replace('SOURCE4',cite('qlearning_holdout_pre2018_v1','README.md',19)+'；'+cite('mb_frozen_2004_2008_v1','README.md',90))
family=family.replace('SOURCE5',cite('residual_learnability_v1','RESULTS.md',7))
(C.P6/'part6_family_summary.md').write_text(family)
questions=['# Open questions — Part 6','', '空值是保存的许可来源未给出所需证据，不是零值；形成期通过数不是交易成功率。下列缺项未通过重算、代码推断、外借其他版本结果或近似匹配补全。','']
labels={'screening_criteria':'精确筛选标准/阈值','pass_rates':'真实/零基准筛选通过率','out_of_sample_comparison':'筛选相对对照的样本外差、CI/p/成本','experiment_conclusion':'本版本实际结果结论','sample_period_and_pairs':'完整样本期和 pair 数量','limitations':'局限'}
for n in S:
 missing=[labels[f] for nn,f in gaps if nn==n]
 if missing:questions.append('- '+n+'：未找到 '+ '；'.join(missing)+'。')
questions+=['','附：有数字的摘录仍可能只包含上述字段的一部分，例如收益水平不等于筛选相对对照的增量，报告未列出的 CI/p 不得补算。','v1/v2/v3 重复文档按版本保留，但不能累计为独立证据。zreversion v1 的原保护失败由 v2 README:5 明示；v2 数值复用不修复此前区间。']
(C.P6/'open_questions.md').write_text('\n'.join(questions)+'\n')
C.verify(C.js(C.P6/'audit/inputs_before.json'))
C.dump(C.P6/'audit/inputs_after.json',C.js(C.P6/'audit/inputs_before.json'))
C.dump(C.P6/'audit/extraction_tests.json',{'registered_experiments':29,'verbatim_excerpts':len(evidence),'all_source_hashes_unchanged':True,'null_roundtrip':True,'positive_GGR_and_refresh_retained':True,'new_statistics_computed':0})
C.dump(C.P6/'complete.json',{'status':'complete','rows':29,'new_real_trials':0,'cumulative_real_trials':580,'outputs':{str(p.relative_to(C.P6)):C.sha(p) for p in [C.P6/'part6_screening_summary.csv',C.P6/'part6_family_summary.md',C.P6/'open_questions.md',C.P6/'audit/excerpts.json']}})
print('Part6 COMPLETE',len(rows),'rows',len(evidence),'literal excerpts')
