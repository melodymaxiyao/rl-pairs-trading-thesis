"""One read-only L0 N2 execution; no selections, fits or other executions."""
from common import *
import numpy as np
from sources import setup,frozen,context,MEM

def fixture():
 receipt=HERE/'audit/reproduction_ready.json'
 if receipt.exists():return read(receipt)
 state=setup();y=2004;pool='P_EXP';row=frozen(f'runtime/pools/{y}')[pool][0];pair=row['pair'];rep=0
 # Save train result, then compare its N2 generator directly against the source
 # function and (where present) the original sealed context payload.
 c=context(row,y,'N2',rep);train_level=c['level'].copy()
 os.environ['RESIDUAL_READ_PURPOSE']='reproduction_fixture'
 c=state.market.context(row,y,'N2',rep,'primary',True)
 np.testing.assert_array_equal(c['level'][:len(train_level)],train_level)
 stem=f'runtime/contexts/{y}/{pair}/{row["leg_A"]}--{row["leg_B"]}/N2_0_primary_full';p=SOURCE/(stem+'.pkl.gz')
 if p.exists():
  old=frozen(stem)
  for key in ['level','r']:np.testing.assert_array_equal(c[key],old[key])
  np.testing.assert_array_equal(c['anchors']['A_TR']['d'],old['anchors']['A_TR']['d'])
 # Read-only native kernel points to an already sealed binary. Missing binary
 # is an error rather than permission to compile in an input directory.
 assert (SOURCE/'runtime/adapter_kernel.so').is_file()
 oldhere=state.io.HERE;state.io.HERE=SOURCE
 try:native=load(SOURCE/'scripts/native.py','_residual_fixture_native')
 finally:state.io.HERE=oldhere
 # from aio import * copied SOURCE at module import, so build() is read-only.
 forbidden=lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('No strategy training or selection allowed'))
 plans=types.ModuleType('plans');plans.plan=plans.b3=plans.td=forbidden;plans.OFF={}
 pools=types.ModuleType('pools');pools.prepare=forbidden
 with aliases(sources=state.sources,plans=plans,pools=pools,native=native):exe=load(SOURCE/'scripts/execution.py','_residual_fixture_execution')
 training=frozen(f'runtime/plans/{pool}_{y}_N2_0_primary');risk=training['risks']['A_TR']['final'];e=next(e for e in c['episodes']['A_TR'] if e['year']==y)
 # Exactly one ledger evaluation, at the primary cost.
 paths=native.policy(e,entry=2.,aligned=True,risk=risk);score,_=exe.stats(e,[paths[0]],[paths[1]],10,training['signs'][pair]);count('trading_evaluations')
 expected=next(r for r in frozen(f'runtime/cells/{pool}_{y}_R_TR_aln_N2_0_primary')['scores'] if r['pair']==pair and r['cost']==10)
 for key in ['net_bps','gross_bps','n_trades']:assert score[key]==expected[key],('Exact sealed L0 N2 reproduction failed',pair,key,score[key],expected[key])
 MEM.clear();value=dict(status='PASS',pool=pool,year=y,pair=pair,world='N2',replicate=0,trading_evaluations=1,exact_fields=['net_bps','gross_bps','n_trades'],train_full_prefix_exact=True)
 write(receipt,value);return value
