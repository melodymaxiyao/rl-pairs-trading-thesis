from common import *
def run():
 import pandas as pd
 from report import stream,comparisons,policies
 from accounting import run as accounting
 from figures import plots
 new,refs,daily,annual=stream();pred,ladder=comparisons(new,refs);pol,abstain=policies()
 descriptions=[]
 for pool in POOLS:
  for label in ['pooled','E1','E2','E3','E4','E5']:
   a=abstain[abstain.pool.eq(pool)];p=pol[pol.pool.eq(pool)]
   if label!='pooled':a=a[a.era.eq(label)];p=p[p.era.eq(label)]
   for cost in COSTS:
    applies=pool=='P_RAND' and label=='pooled' and cost==10
    descriptions.append(dict(id='U2',pool=pool,cost=cost,era=label,primary=False,scheme='policy folds',never_entering_folds=int(a.never_enters.sum()),old_never_entering_folds=int(a.old_never_enters.sum()),total_folds=len(a),prediction_applies=applies,prediction_met=bool(a.never_enters.sum()<=5) if applies else None,note='Same real-trained policy at all costs; pooled P_RAND threshold<=5/22 is descriptive.'))
    for bin,g in p.groupby('bin',sort=True):
     for col in ['enter','hold']:descriptions.append(dict(id='U5',pool=pool,cost=cost,era=label,primary=False,scheme='policy folds',state_bin=bin,metric=col+'_share',estimate=float(g[col].mean()),total_folds=len(g),note='Unmasked table preference frequency; alignment/lock/forced-exit masks still apply; policy identical across costs.'))
 csv_new(TABLES/'predictions.csv',pd.concat([pred,pd.DataFrame(descriptions)],ignore_index=True,sort=False))
 acct,prior=accounting(new[new.world.eq('real')],daily);plots(ladder,pol,daily,prior)
 def row(id):return pred[pred.pool.eq('P_RAND')&pred.cost.eq(10)&pred.era.eq('pooled')&pred.scheme.eq('year')&pred.id.eq(id)].iloc[0]
 def number(r):return f"{r['estimate']:.2f} [{r['ci95_low']:.2f}, {r['ci95_high']:.2f}] bps/pair-year"
 u1=row('U1');u3=row('U3');u4=row('U4');a=abstain[abstain.pool.eq('P_RAND')]
 ds=acct[acct.kind.eq('DSR')&acct.pool.eq('GLOBAL')&acct.N.eq(544)].iloc[0];pbo=acct[acct.kind.eq('PBO')&acct.pool.eq('GLOBAL')].iloc[0]
 freqs=pol.groupby(['pool','bin'])[['enter','hold']].mean();other=[]
 for pool in ['P_EXP','P_QUAL']:
  r=pred[pred.pool.eq(pool)&pred.cost.eq(10)&pred.era.eq('pooled')&pred.scheme.eq('year')&pred.id.eq('U1')].iloc[0];other.append(pool+' U1 '+number(r))
 shape=pol[pol.pool.eq('P_RAND')].groupby('bin')[['enter','hold']].mean().reindex(['<0.5','0.5–1.5','1.5–2.5','>=2.5'])
 description='U5 P_RAND, ascending |d| bins: enter shares '+','.join(f'{v:.2f}' for v in shape.enter)+'; hold shares '+','.join(f'{v:.2f}' for v in shape.hold)+' (masks apply); '
 lines=[f"U1 P_RAND/10bps tuned minus TD8: {number(u1)}; p {u1.p_one_sided:.6f}; {'supported' if u1.supported else 'not supported'}.",f"U2 never-entering folds: TD8 {int(a.old_never_enters.sum())}/22, tuned {int(a.never_enters.sum())}/22.",f"U3/U4 tuned minus L0: {number(u3)}; minus B3: {number(u4)}.",description+'; '.join(other)+'.',f"Global cumulative accounting544: DSR {ds.DSR:.4f} (best cell {ds.best_cell_for_accounting_only}); PBO {pbo.PBO:.4f}; development evidence."]
 write(HERE/'summary.json',dict(lines=lines));marker='<!-- RESULTS -->'
 if marker not in (HERE/'README.md').read_text():
  with (HERE/'README.md').open('a') as f:f.write('\n\n'+marker+'\n\n'+'\n\n'.join(lines)+'\n\nU3 is a directional point-estimate prediction, not a claim of significance. The heat map and full per-fold policies show nonmonotone actions and masked-entry qualifications. See predictions.csv for all pools, costs and eras; U1 outside P_RAND10bps/pooled/year is descriptive.\n')
 write(HERE/'audit/report_ready.json',dict(status='PASS',master_sha256=sha(TABLES/'per_pair_fold_test.csv.gz'),summary_sha256=sha(HERE/'summary.json')))
