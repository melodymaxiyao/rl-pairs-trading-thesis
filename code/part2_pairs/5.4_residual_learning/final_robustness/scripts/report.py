import common as C
C.guard()
import json,hashlib
from pathlib import Path

def md(rows,cols=None):
 cols=cols or list(rows[0]);return '\n'.join(['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']+['| '+' | '.join(str(r.get(c,'')).replace('|','/') for c in cols)+' |' for r in rows])
f1=C.rows(C.P/'F1_N2_vs_N3.csv');f2=C.rows(C.P/'F2_vs_flat.csv');f3=C.rows(C.P/'F3_top5_missing.csv');corr=C.rows(C.P/'F3_missingness_correlations.csv')
questions='''# Open questions — final_checks_v1 / 稳健性检验

1. F1 STOPPED_BEFORE_IC_RESULTS. The RF2007/global index1978/N2r1 score fixture differs: actual−1074.393423094017 vs sealed−1074.3934226986216, abs3.95395318e−7, relative3.68017255e−10, outside original rtol1e−10/atol1e−8. The mismatch is an implementation reproduction gate, not evidence against predictability or a “not robust” statistical outcome. No new F1 IC/excess/CI/MCSE was computed; N3_core columns stay blank. Full xi=1/all-year validation and original sealed N3r0/r1 validation have not completed.
2. A concrete implementation-only repair is proposed in audit/F1_exact_arithmetic_proposed.patch: reuse the original RF matrix subtraction and original rf_signed array layout from coint_residual_qlearning_v1/scripts/data.py:122–127. The first adapter regrouped these operations. User approval to restart after the explicit stop gate is pending. No parameter, seed, scope or tolerance change is proposed.
3. The old core_headline rolling-score N2 results are extracted exactly from sealed tradable_target_summary.csv; a matching historical core-last-increment N2 row is not available. The causal/v1-fixed-mask neg_e row is not substituted. Any later same-definition comparator calculation must remain labelled as newly computed robustness evidence.
4. The current thesis Table5.4 file/row identity was not supplied. F2 reproduces all264 Part1 annual values and matching sealed H1/H2/follow-up annual-K2 contrasts. This does not assert a verbatim check of the current manuscript table. The data pack's T7_3 net_annual_bps is a separate pooled/day-normalized performance quantity and is not substituted for the requested22-year equal-weight statistic.
5. Saved membership is reconstructed monthly membership with source flags for reversal inconsistencies and potentially incomplete change history. Its834 distinct symbols across312months reproduce exactly; year-end counts are the saved data's counts, not an independently verified official historical500-member census. In particular early year-end count515 is retained rather than forced to500.
6. selected_changes.csv is explicitly incomplete. F3 reports known dated removal events and their missing-price counts only; it cannot provide complete annual removal totals. An observed zero in this table means no saved event, not proof that no stock left the index. The completeness flag isFalse for every year.
7. Coverage means at least one positive finite adjusted close during the year on the exact experimental calendar among year-end members. This is not full-year continuous tradability or proof of unbiased survivorship. Separate finite-panel-return counts and per-policy actual entry counts are retained. Missing-price versus annual-return correlations are descriptive and may reflect calendar trends; no causal interpretation.
8. No needed computational source was dataless/unavailable in F2/F3. Actual input and alias checks are recorded in audit ledgers. F4 remains NOT_RUN_AWAITING_EXPLICIT_CONFIRMATION,0new trials,580 cumulative; no frozen DSR is changed.
'''
(C.P/'open_questions.md').write_text(questions)
text='''# final_checks_v1 — 稳健性检验

F2 and F3 complete. F1 is stopped at its numerical reproduction gate, with no new IC outcomes and no statistical decision. F4 NOT_RUN_AWAITING_EXPLICIT_CONFIRMATION. New real trials0; cumulative580. Resource amendment: stop below2GiB free, new output cap1GiB.

## F1 — existing N2 only; N3_core NOT_EVALUATED

'''+md(f1,['signal','family','h','N2_excess_IC','N2_ci95_low','N2_ci95_high','N3_excess_IC','decision'])+'''

The sealed N3 is a causal daily accumulation; N3_core rebuilds the entire60-day residual window at each t, using that t's frozen hedge, date-keyed signs before mean subtraction, and similarly signed future alpha-free targets. These are explicitly separate versions. Original preregistration and user-approved addendum were written before any new F1 outcomes. Gate details and the proposed exact-arithmetic repair are in open_questions.md; tolerances are unchanged and no scientific outcome was computed.

## F2 — vs Always Flat,22-year equal-weight K2

'''+md(f2,['family','cost_bps','learner','mean_net_bps','ci95_low','ci95_high','p_one_sided'])+'''

Unit=bps per committed-capital year, mean of22saved annual totals2004–2025. Bootstrap5000,seed20261026,p=mean(boot<=0),noHolm. A saved p0 means0/5000draws,not exact population probability0. Sources are the same sealed per_series_year files as Part1;264annual values match, and H1/H2 plus cost10follow-up contrast means reproduce. Flat means exactly zero, not AL (which is always-long and may lose money). RFQ_best10bps has a large bootstrap mass at0; its nonnegative CI endpoint does not establish significance.

## F3 — coverage audit

The saved member union834/312months matches. Coverage is measured from original registered adjusted-close files, restricted to the exact residual-panel calendar. The panel has863symbols including benchmarks/other instruments, not863indexmembers. All needed price sources were readable and verified; none was hydrated. Earliest nominal test/trading results are2004,so actual-entry fields before2004are blank.

Worst five years:

'''+md(f3,['year','year_end_members','members_with_price','coverage_fraction','missing_fraction'])+'''

Descriptive Spearman correlations over common22years2004–2025:

'''+md(corr,['family','cost_bps','learner','spearman_rho','n_years'])+'''

The annual CSV additionally reports known removals, missing-price removals, unknown statuses,3000-pair formation-pool size where applicable,and5bpsactualenteredseries/distinctlegs forQ_best/Band/AL,RP/RF. These are labelled rather than conflating the formation pool with traded series. Saved removal history is incomplete,so known removal counts must not be quoted as complete totals. See open_questions.md.

Figure:F3_coverage_robustness.png,300dpi,no internal title,no second axis;plot values equal the coverage table. This is a robustness/data-coverage audit,not a new performance trial.

## Status and evidence

Pre-outcome supporting tests6/6 passed (including small hand IC,weighted ties,exact annual-source comparison,and834/312membership counts). They do not override F1's failed scientific reproduction gate. All originals remain read-only. Final input SHA and FYP inventory receipts are written separately under audit. F4 needs the user's exact execute-F4 instruction and would add6trials580→586; it has not run.
'''
(C.P/'README.md').write_text(text)
(C.P/'figures.md').write_text('# 稳健性检验 — F3 coverage\n\nF3_coverage_robustness.png:300dpi;year-end saved members with at least1positive finite adjusted close on actual experimental-calendar dates in the corresponding year. Source F3_coverage_by_year.csv. No internal title or second axis. Plot/table equality in audit/F3_plot_test.json. Source reconstruction/completeness limits in open_questions.md.\n')
print(C.P/'README.md')
