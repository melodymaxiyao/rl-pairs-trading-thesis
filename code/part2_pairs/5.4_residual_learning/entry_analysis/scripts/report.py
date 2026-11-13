import common as C
C.guard();P=C.P;ROOT='/Users/maxiyao/FYP_runs/coint_residual_qlearning_v1';CARD=ROOT+'/post_tier3_review_v1/policy_card_v1'
rr=C.rows(P/'fold_comparison.csv');ch=C.js(P/'audit/checks.json')
table=['| 测试年 | 部署动作 | 实际入场次数 | 有效空仓机会数 | 入场率 |','| --- | --- | ---: | ---: | ---: |']
for r in rr:table.append(f"| {r['test_year']} | {'入场' if r['deployed_action']=='enter' else '不入场'} | {int(r['actual_entries']):,} | {int(r['flat_opportunities']):,} | {float(r['entry_rate_percent']):.0f}% |")
table.append('| 合计 | 13年入场，9年不入场 | 14,633 | 1,794,869 | 0.8153% |')
body=f'''# Table 5.5 入场计数与实际入场率核对

**结论：(a) 两者口径不同，没有矛盾，也没有发现这两个数据有误。** 13/22 是年度策略的等权动作计数；0.8% 是对所有 pair×交易日的有效空仓机会合并后的实际入场率。选择入场的13年在该状态实际都入场100%，其余9年为0%。本任务只读取封存结果、汇总计数及做一步除法，没有重放交易或进行新推断。

## 1. T7_9 统计什么

范围：RP、G0、Q_best、10 bps/side、空仓状态0，即 `|s|<0.5`，测试年2004–2025。

- 每个年度fold贡献一个已部署策略动作。22个策略中13个动作位为1（enter），9个为0（stay）；不是把种子作多数决，也不是拿跨22年的平均Q表重新决策。
- 每个年度策略先对每个种子的 Double-Q A/B 表取均值，再对10个种子取逐元素均值。空仓状态应用固定的LCB：`Q_enter−Q_stay > 1.645×sqrt(SE_enter²+SE_stay²)`，且margin须有限。
- SE是冻结实现的 **nominal TD-target-noise SE**。每个种子由保存的diagnostics按 `d[...,7]×sqrt(d[...,1])` 得到，先平均A/B再平均10个种子；没有再除以sqrt(10)。它不是种子间Q值的标准差、22fold均值的SE，也不是每个测试日重新计算的置信界。状态表中的 `nominal_advantage_SE` 是上述两个动作SE的平方和开方。
- 因此13/22已经包含LCB；测试时执行保存的policy bits，不会再给这13个年度策略逐日加一道新的LCB。

出处：

- `{ROOT}/scripts/kernel.py:69–76`：Q/SE平均与LCB。
- `{ROOT}/scripts/learning.py:16–24`：10个种子构造策略。
- `{CARD}/build.py:87–100`：复核保存策略，使用 `lcb=True`，生成逐fold动作。
- `{CARD}/build.py:110–113`：按22个年度策略计数。
- `{CARD}/fold_policies.csv`：行键 `definition=RP;cost=10;learner=Q_best;state=0;test_year=2004..2025`。

## 2. Table C.6 / Figure 5.5 的分子、分母

保存的入场率为：

`Σ实际入场 / Σ有效空仓机会 = 14,633 / 1,794,869 = 0.008152684123465278 = 0.8152684%`，显示一位小数即0.8%。

**分母**：每个pair、每个非终止交易日，在决策前仓位为0，且执行用的s-score有效，计一次该状态的机会。有效条件为原始score和half-life均有限、`0<half-life<30`；G0不另要求ADF通过。无效信号日、年度最后一日和已持仓序列不进入entry分母。这里的“机会”**不以该年度策略允许入场为前提**：动作为stay的年份同样计数；同一pair连续空仓的多个交易日会反复计数。机会按pair-day原始次数累计，不用IPW资本权重，也不是先算每年比率再等权平均。

**分子**：在上述机会中，执行后的仓位由0变为非0，才记一次实际入场。实际成交腿报价不全时，atomic basket会延迟执行；该报价条件不作为上述机会数的额外过滤。因此“有效机会”不等于“所有成交条件均满足的可执行订单”。精确 `s=0` 也属于状态0，但目标方向为0，不会因enter动作凭空建立仓位。本次目标行的13个入场fold均为100%，上述特殊情形没有造成该行的低合并比率。

**仓位限制**：代码没有额外的同时持仓数量上限/资金竞争拒单过滤。RP使用固定归一化IPW资本权重；它影响头寸与收益归因，不改变D6机会的原始计数。已有仓位通过 `old!=0` 从entry机会中排除。

**状态一致**：两者使用同一冻结s-score和状态边界。`abs(s)<0.5` 为状态/bin0；精确0.5进入下一桶，之后边界为1.5、2.5。此处不是把持仓时的trade-aligned z与空仓|s|混用。

出处：

- `{ROOT}/scripts/data.py:93–96`：执行score有效条件及G0/G1区别。
- `{ROOT}/scripts/accounting.py:7–13`：状态边界、固定RP权重。
- `{ROOT}/scripts/accounting.py:28–44`：期望动作、缺报价时延迟执行。
- `{ROOT}/scripts/accounting.py:67–76`：机会和实际入场计数的精确定义。
- `{ROOT}/scripts/missing_prices.py:33–42`：同一状态分桶、policy bits与入场方向。
- `{ROOT}/scripts/experiment.py:33–43`：同fold内按side/bin汇总。
- `{CARD}/build.py:148–153`：跨fold合并numerator/denominator。
- `/Users/maxiyao/FYP_runs/thesis_tables_v3_pairs/scripts/build.py:382–404`：图F7_4（论文Figure5.5对应数据）直接读取T7_9 part1的rate。
- `outputs/figures/figure_data.csv` 已核对行键 `figure=F7_4;panel=RP entry rate;series=10;x=0`，保存值与D6完全相同。

## 3. 逐fold对照

{chr(10).join(table)}

每年的计数来自 `{ROOT}/outputs/cells/RP_{{year}}_G0_c10_Q_best/entries_exits.csv.gz`，选定 `side=entry;bin=0;world=real;replicate=0`。动作来自前述fold_policies，并与同cell的complete.json中policy的state0动作位逐年核对。每个文件路径、行键及SHA-256均保存在 `fold_comparison.csv`。

13个入场fold合计14,633次机会、14,633次入场；9个不入场fold合计1,780,236次机会、0次入场。两个组加起来精确复现保存的D6总数；不是13个年度策略的入场意愿在执行中被LCB压到了0.8%。

## 4. 原因与论文建议表述

两种量的权重不同，而且分母受策略本身影响。入场后该pair进入持仓状态，后续持仓日不再贡献空仓机会；stay策略可以在同一桶内连续很多天贡献空仓机会。保存的fold策略还显示：这13个入场年份的四个持仓状态动作全部为hold（无效信号/年度结束仍按原规则处理），进一步解释了它们为何只贡献很少空仓机会。

不能从“13/22年度策略允许入场”推得“所有有效空仓机会中约59%入场”。对本状态，合并后的机会权重几乎全部来自不入场年份。

建议英文：

> At 10 bps, 13 of the 22 annual RP policies permit entry when |s| < 0.5, after the LCB restriction. Each of these 13 policies entered at every saved eligible flat opportunity in that state; the other nine entered at none. The pooled realised entry rate was nevertheless only 0.8% (14,633/1,794,869), because non-entering years generated far more flat-state opportunities. Table 5.5 weights annual policies equally, whereas Table C.6 and Figure 5.5 pool opportunities across years; their denominators are policy-dependent.

可以保留两个数字，修改解释即可。特别是不要再用“还要通过LCB”来解释13/22与0.8%的差距。

## 完整性

使用Tier3 manifest/code_registration及policy-card receipt核对所用封存代码、逐fold计数和策略文件；与T7_9及figure_data的对应行一致。63个来源输入的SHA-256与字节数前后一致；所有实际读取输入均非dataless。没有修改论文或来源文件，没有训练、回测、下载、安装或新增trial；所有写入位于本run目录。
'''
(P/'README.md').write_text(body)
(P/'open_questions.md').write_text('# open_questions\n\n无未解决项。22个fold的实际入场次数和机会数均已保存，逐年与部署动作一致，合计精确复现T7_9及图的保存数据。无需重跑交易、重算SE或做新推断。\n')
H=C.js(P/'input_hashes_before.json');after={}
for p,v in H.items():C.raw(p);after[p]=C.INPUTS[p];assert after[p]==v
C.dump(P/'input_hashes_after.json',after)
C.dump(P/'complete.json',dict(status='complete',conclusion='a_different_weighting_no_contradiction',folds=22,inputs=len(H),input_hashes_unchanged=True,new_trials=0,new_fits=0,new_backtests=0,new_inference=0,entries=14633,opportunities=1794869,pooled_rate=14633/1794869))
print('\n'.join(table));print('README',P/'README.md');print('Verified inputs',len(H))
