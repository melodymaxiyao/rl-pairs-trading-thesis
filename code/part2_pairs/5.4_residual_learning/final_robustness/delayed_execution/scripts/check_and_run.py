"""Launch-time identity/resource check; no scientific logic modifications."""
from pathlib import Path
import sys,json,hashlib,runpy
R=Path(__file__).resolve().parents[1];B=R.parent
sys.path[:0]=[str(R/'scripts'),str(B/'scripts')]
import common as C
C.P=R;C.guard()
a=C.js(R/'audit/authorization.json');code=R/'scripts/f4.py';assert C.sha(code)==a['new_code_sha256'];assert C.sha(R/'preregistration_addendum.md')==a['addendum_sha256']
for rel,h in a['unchanged_shared_code'].items():assert C.sha(B/'scripts'/rel)==h
for rel,h in a['same_supporting_tests'].items():assert C.sha(R/rel)==h
stage=sys.argv[sys.argv.index('--stage')+1];assert stage in ['validate','delayed']
C.resources();assert sum(p.stat().st_size for p in B.rglob('*') if p.is_file())<2**30
if stage=='delayed':assert C.js(R/'audit/zero_delay_validation.json')['passes'] and C.js(R/'audit/zero_delay_validation.json')['fold_policy_checks']==132
C.dump(R/f'audit/{stage}_code_identity.json',{'stage':stage,'code_sha256':C.sha(code),'registered_code_sha256':a['new_code_sha256'],'match':True})
sys.argv=[str(code),'--stage',stage]
runpy.run_path(str(code),run_name='__main__')
