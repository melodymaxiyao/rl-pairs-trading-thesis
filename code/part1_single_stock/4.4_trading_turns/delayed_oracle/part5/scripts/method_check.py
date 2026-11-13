import sys
sys.path.insert(0,'/Users/maxiyao/FYP_runs/closing_checks_v1/part4/scripts')
import ext_common as C
C.guard()
from pathlib import Path
import json
T=C.REST/'experiments/turning_point_detection/new_stage1';S=C.REST/'experiments/single_stock'
FILES={}
def doc(p):
 p=Path(p)
 if str(p) not in FILES:FILES[str(p)]=C.text(p).splitlines()
 return FILES[str(p)]
def cite(p,a,b):
 ls=doc(p);assert 1<=a<=b<=len(ls)
 return {'path':str(p),'line_start':a,'line_end':b,'sha256':C.READS[str(p)]['sha256'],'excerpt':'\n'.join(ls[a-1:b])}
K=T/'kernel_smoothing.py';A=T/'slope_accuracy.py';RUN=T/'run_slope_accuracy.py';V2=T/'slope_accuracy_diagnostics.py';CFG=T/'configs/slope_accuracy.json'
BC=S/'configs/single_stock_bocpd.json';EC=S/'configs/single_stock_confirmed_turns.json';BF=S/'features_bocpd.py';EF=S/'features_confirmed_turns.py';BR=S/'run_signal_diagnostic_bocpd.py';ER=S/'run_signal_diagnostic_confirmed_turns.py'
AS=C.REST/'thesis_assets/scripts/s8_new_stage1.py';EX=Path('/Users/maxiyao/single_stock_runs/thesis_tables_v3_single/scripts/extract.py')
BReport=C.FYP/'experiments/single_stock/results/bocpd/signal_diagnostic/report.md';EReport=C.FYP/'experiments/single_stock/results/confirmed_turns/signal_diagnostic/report.md'
items=[]
def item(i,quote,status,actual,english,refs):
 items.append({'item':i,'provided_description':quote,'quote_status':'User prompt description/question, NOT a verified verbatim thesis quotation','verdict':status,'actual':actual,'suggested_english':english,'sources':refs})
item(1,'因果斜率：在最近 W = 3h 个对数价格上做加权最小二乘直线拟合，高斯权重为 exp(−j²/2h²)。','一致（补充完整窗口条件）',
'带截距的局部线性WLS，offset为0,-1,...,-(3h-1)，权重exp[-.5(x/h)^2]。输入为log adjusted Close；斜率固定权重满足sum(l)=0、sum(l*x)=1。窗口不完整时为NaN，不是用较短前缀拟合；不要与同文件的local-constant平滑函数的启动方式混写。',
'The causal slope is the weighted least-squares slope of log adjusted close over the most recent 3h trading observations, including the current close, with Gaussian weights exp(-j²/(2h²)); incomplete windows are undefined.',
[cite(K,77,102),cite(K,105,127),cite(RUN,57,67),cite(CFG,18,20)])
item(2,'确认规则：上升阶段之后，斜率连续 d 天低于 −τ 时确认波峰；下降阶段之后，连续 d 天高于 +τ 时确认波谷。死区 τ = k·σ_b，其中 σ_b 是训练期斜率的标准差（ddof 是多少）。事件日期记在确认当天。','一致（补充ddof与初始化）',
'正/负阈值是严格不等式；等于阈值或位于死区为0，因此打断连续d天计数。首个满足d天的稳定方向只初始化状态，不发事件；此后合格稳定方向发生改变时，在第d天发确认事件。负方向=-1映射peak、正方向=1映射trough。σ_b由本折TRAIN有限斜率计算，ddof=1；检测器从train_start开始连续进入TEST，确认事件日期不回填。',
'A peak (trough) is recorded on the confirmation day after d consecutive slopes below -kσ_b (above +kσ_b), relative to the previously established direction. The first stable run initializes direction without an event; σ_b is the training-slope sample standard deviation (ddof=1).',
[cite(A,54,70),cite(A,157,183),cite(RUN,244,249),cite(CFG,20,22)])
item(3,'(h, d, k) 的选择：在三个训练年上，选使对参考转折点的 F1 最大的组合。写出网格范围。','需要修改／补充：明确宏平均选择而非逐标的选择',
'每个fold和每个h_ref分别选一组共同(h,d,k)，目标是在该fold可用标的上的TRAIN combined-F1等权平均，而非每个标的独立选参，也非把三个年度F1简单平均。每个标的TRAIN先按匹配总数算F1，再跨标的宏平均。TRAIN为连续三个完整日历年。原网格h={3,5,10,20},d={1,2,3,5},k={0,.5,1,1.5}；h_ref={5,10,20}。F1同分依次选更小的有符号median delay、h、d、k。k细网格{0,.25,.5,.75,1}是另行事后敏感性，不是主网格。',
'For each fold and reference bandwidth, one common parameter triple is selected by maximizing the equally weighted mean of ticker-level training F1 scores over the available tickers in the three-year training window. Ties are broken by signed median delay, then h, d and k; the primary grids are h∈{3,5,10,20}, d∈{1,2,3,5}, and k∈{0,0.5,1,1.5}.',
[cite(A,42,50),cite(A,117,129),cite(A,147,154),cite(RUN,74,92),cite(CFG,10,19),cite(T/'configs/slope_accuracy_diagnostics.json',9,10)])
item(4,'参考转折点（oracle）：是否就是对称窗口、带宽为 h_ref 的局部线性斜率的零交叉？窗口长度是多少？是否使用全样本价格计算？','需要修改／补充：6h_ref+1与全序列/边界口径',
'是双侧局部线性WLS斜率的离散符号变化，offset=-3h_ref,...,+3h_ref，总长6h_ref+1，不是3h_ref或2h_ref+1。直接在可用全价格序列计算双侧斜率；不是每折先截到TEST末日再平滑。零值保留前一非零符号，NaN重置；日期是首次新非零符号当日，不插值精确零时刻。oracle只评分，不进入因果检测/选参之外的实时信号。TRAIN末尾裁去3h_ref个交易日以防训练参考读到TEST；v1 TEST也年尾裁剪。v2 TEST保留到min(test_end,全价格序列末位-3h_ref)，可使用TEST年结束后的已存价格作参考上下文。封存reference_turns.csv来自v1范围；v2 scoring重新使用相同全序列参考和修订bounds，不能混写两者。',
'Reference turns are sign changes of a retrospective bilateral local-linear slope computed on the full available log-price series, using offsets -3h_ref,…,+3h_ref (6h_ref+1 observations). Training references are trimmed to avoid using test prices; revised test scoring retains all dates with available bilateral context. Reference labels are evaluation-only and use future information.',
[cite(K,82,102),cite(K,130,137),cite(A,73,87),cite(A,157,205),cite(V2,18,42),cite(RUN,240,243)])
item(5,'F1 的匹配规则：确认事件和参考转折点在多少天以内算匹配？是只往后匹配还是双向？类型是否相同？是否一对一？','需要修改／补充：非对称双向窗口和贪心算法',
'设delay=detected_position-reference_position，容许-2<=delay<=2h_ref，即h_ref5/10/20的窗口为[-2,10]/[-2,20]/[-2,40]个交易日；不是只向后，也不是对称±h_ref。要求同类型、每个参考点只能用一次。检测事件按时间先后依次贪心选未使用候选中|delay|最小者；并列选更早参考点。不存在全局最大匹配数优化或额外匹配单调性约束。P=匹配数/检测数，R=匹配数/参考数，F1由两者计算；不同ticker的宏F1不等于合并全标的计数后的microF1。',
'Detected events are matched one-to-one to unused reference turns of the same type within -2≤t_detected-t_reference≤2h_ref trading days. Detections are processed chronologically and assigned greedily to the nearest eligible reference in absolute delay, with ties resolved in favour of the earlier reference.',
[cite(A,90,114),cite(A,117,143),cite(A,198,202),cite(CFG,21,22)])
item(6,'延迟的定义：确认日期减参考日期，单位是交易日；“延迟 / 间距”中的间距如何定义。','需要修改／补充：有符号延迟、年内相邻参考间距及宏汇总',
'延迟是匹配后的位置索引差det-ref，单位交易日且可为负（最早-2），不是自然日或只统计非负值。间距是每个ticker×test-year窗口内相邻参考turn的位置差（combined含peak和trough），不跨年连线；ticker内把各年gaps合并后取均值。S5_1主列先每个ticker用该ticker的median delay/mean spacing，再对7个ticker等权平均。另有LLT对照列采用macro pooled-event median delay/跨ticker event-weighted pooled mean spacing，数值不同，不能当同一定义。两者都不是每个匹配事件delay/其最近reference gap再取平均。',
'Delay is the signed difference between confirmation and reference positions in trading sessions. Reference spacing is the mean gap between consecutive reference turns within ticker-year evaluation windows, pooled within each ticker without linking years. The main macro delay share is the equal-weight mean of the seven ticker-level median-delay/mean-spacing ratios; the pooled-spacing alternative is reported separately.',
[cite(A,124,129),cite(A,133,143),cite(AS,27,42),cite(AS,74,100),cite(EX,93,102),cite(C.REST.parent/'llt_turning_point_v1/preregistration.md',75,75)])
item(7,'BOCPD：模型和实现、hazard rate网格，以及如何从变点得到确认波谷／波峰。','需要补充（作为需要修改条目列出）：run-length概率与方向定义',
'Adams–MacKay BOCPD，经固定sdt-python20.1.4的sdt.changepoint.BayesOnline，engine=python，输入z20，Student-t预测分布/NIG先验(mu0,kappa1,alpha.1,beta.01)。constant hazard的time-scale λ={50,100,200}，实际每步hazard=1/λ。d=3，使用P(r_t=3)>0.5，不用恒为1/λ的P(r_t=0)。c=t-3；比较z20[c:t]（含端点）均值与z20[c-5:c-1]均值，前者更高标trough、更低标peak，相等不发。标签在决策日t发出，不回填c；这不是在变点日直接检出几何极值。',
'BOCPD uses the Adams–MacKay recursion implemented by sdt-python 20.1.4 (BayesOnline, Python engine) on z20 with a Student-t predictive distribution and fixed NIG prior. Constant-hazard time scales are λ∈{50,100,200}; at t, P(r_t=3)>0.5 confirms a source c=t-3, classified as a trough or peak by comparing the observed post-source and preceding-five-session means. Labels are emitted at t, never backdated.',
[cite(BC,4,42),cite(BF,124,135),cite(BF,148,182),cite(BReport,3,7)])
item(8,'EMA规则：用了哪几条EMA、k网格及确认方式。','需要补充（作为需要修改条目列出）：单EMA摆动阈值，不是EMA交叉',
'只用一条z20的causal EMA(span5,adjust=False,min_periods5)，不是5/20或多条EMA交叉。局部波动是原始z20最近20点std(ddof0)，k={1,1.5,2}；在t考察c=t-3处EMA[c-1:c+1]的严格局部极值。候选点与先前反类型anchor的摆幅必须严格大于k*vol[c]（不是vol[t]）。首个极值只初始化anchor，同类型更极端候选可替换anchor但不发事件，反类型且通过幅度阈值才于t确认。',
'The confirmed-turn diagnostic applies a single causal EMA(span=5, adjust=False) to z20. At t it checks the strict local extremum at c=t-3 and emits a label only when its swing from the opposite anchor exceeds k times the 20-session z20 standard deviation at c (ddof=0), for k∈{1,1.5,2}; this is not an EMA-crossover rule.',
[cite(EC,8,25),cite(EF,28,57),cite(EF,60,72),cite(EF,93,135),cite(EF,158,177),cite(EReport,3,5)])
item(9,'两项诊断的42个比较如何构成、统计量是什么、Holm族如何定义。','需要补充（作为需要修改条目列出）：7×2×3，各自独立42项族',
'BOCPD：7tickers×h={5,10}×λ={50,100,200}×固定概率阈值.5=42；EMA：7tickers×h={5,10}×k={1,1.5,2}=42。四个OOS folds拼接后每个组合只产生一个检验，不再乘4；不是h_ref={5,10,20}那个网格。统计量D=mean(future log wealth|trough)-mean(...|peak)，决策收盘t形成标签、t+1开盘做多至t+h收盘，cost=0，端点不能跨fold测试末日。bootstrap按连续20个决策日同时重抽return和state，5000次seed12345；raw p为双侧，Holm-Bonferroni α=.05分别在每个诊断的42项族内施加，不是两者合成84项，也不是按fold分别校正。每类事件不足5时统计量/CI缺省、p设1，仍属于原42项族。两报告均0/42通过Holm，0/14稳健组过gate；不是证明收益差等于0。',
'Each diagnostic has its own 42-test Holm family: seven tickers × two horizons (5 and 10 sessions) × three hazard time scales or swing multipliers. The test statistic is the difference between mean zero-cost next-open-to-horizon-close log returns after confirmed troughs and peaks, stitched across four OOS folds. Paired 20-session moving blocks provide 5,000 bootstrap draws (seed 12345); two-sided p-values are Holm-adjusted within each diagnostic, not within each fold or across a combined 84-test family.',
[cite(BC,45,72),cite(EC,28,55),cite(ER,104,189),cite(ER,233,272),cite(ER,385,397),cite(BR,246,268),cite(BR,302,311),cite(BReport,75,79),cite(EReport,73,77)])
# Verify source-only definitions without imports or executing any numerical function.
cfg=json.loads(C.text(CFG));bc=json.loads(C.text(BC));ec=json.loads(C.text(EC))
assert cfg['std_ddof']==1 and cfg['match_l_pre']==2 and cfg['match_m_multiplier']==2
assert cfg['h_grid']==[3,5,10,20] and cfg['k_grid']==[0,.5,1.,1.5]
assert bc['signal_diagnostic']['parallel_test_family'].endswith('= 42')
assert ec['signal_diagnostic']['parallel_test_family'].endswith('= 42')
assert '[-2' not in C.text(A) or 'delta >= -l_pre' in C.text(A)
# Merge all earlier read-only discovery ledgers for before/after coverage.
before={}
for p in C.P5.glob('source_read_ledger*.json'):
 for k,v in C.js(p).items():
  if not Path(k).is_relative_to(C.P5):
   if k in before:assert before[k]==v
   before[k]=v
for k,v in C.READS.items():
 if not any(Path(k).is_relative_to(r) for r in C.ROOTS):
  if k in before:assert before[k]==v
  before[k]=v
C.dump(C.P5/'audit/inputs_before.json',before)
out=['# Part5 方法核对（只读）','','没有运行或导入检测器、BOCPD、EMA、回测或统计函数；只读取源码、配置、既存报告。','',
'**引文限制**：此前提供的 /Users/maxiyao/Downloads/FYP_thesis_v4 (1).docx 当前不存在；本次也未提供新的论文文件。因此下列“待核对表述”逐字来自本次prompt，7–9是核对问题，不冒充论文逐字引文。对“需要补充”的判定是建议论文应明示的必要细节，不能据此声称已核对了当前论文原句。','',
'**总体**：1、2核心叙述一致；3–9应修改/补充以下细节。真实引用位置未找到的留为open question。']
for r in items:
 out+=['',f"## {r['item']}. {r['verdict']}",'', '待核对表述（prompt）：'+r['provided_description'],'','代码实际情况：'+r['actual'],'','建议英文：'+r['suggested_english'],'','出处：']
 for a in r['sources']:
  out.append('- ['+Path(a['path']).name+':'+str(a['line_start'])+']('+a['path']+':'+str(a['line_start'])+')（至'+str(a['line_end'])+'行；SHA256 '+a['sha256']+'）')
(C.P5/'part5_method_check.md').write_text('\n'.join(out)+'\n')
C.dump(C.P5/'method_items.json',items)
(C.P5/'open_questions.md').write_text('# Open questions\n\n- 原论文 /Users/maxiyao/Downloads/FYP_thesis_v4 (1).docx 当前缺失，未提供新版。已对照本次prompt的九条描述与源码，但不能确认现稿§3.2.1/§4.5是否已经包含建议措辞。需要论文的当前文件/确切原句才能完成逐字引文核对；没有按文件名相似度挑另一版。\n')
C.verify(before)
C.dump(C.P5/'audit/inputs_after.json',{p:C.READS[p] for p in before})
C.dump(C.P5/'complete.json',{'status':'source_method_review_complete_thesis_verbatim_unavailable','items':9,'needs_changes_or_details':[r['item'] for r in items if r['item']>=3],'scientific_functions_executed':0,'real_trials':580,'new_real_trials':0,'input_hashes_unchanged':True})
print('PART5 complete; source review9 items, thesis verbatim gap recorded')
