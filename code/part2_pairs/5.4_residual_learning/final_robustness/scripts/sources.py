import common as C
import gzip,pickle,json,hashlib
from pathlib import Path
CORE=Path('/Users/maxiyao/FYP_runs/coint_residual_qlearning_v1');RES=Path('/Users/maxiyao/FYP_runs/residual_ou_real_v1');PART=Path('/Users/maxiyao/FYP_runs/partial_coint_kalman_v1');APP=Path('/Users/maxiyao/FYP_runs/thesis_appendix_info_v1')
SEAL=C.js(CORE/'tier_3/complete.json');assert C.sha(CORE/'tier_3/manifest.json')==SEAL['manifest_sha256']
MF=C.js(CORE/'tier_3/manifest.json')['files']
def core(rel):
 assert rel in MF,('not in core seal',rel)
 e=MF[rel];e=e['sha256'] if isinstance(e,dict) else e;b=C.raw(CORE/rel);assert hashlib.sha256(b).hexdigest()==e,rel;return b
anchors=json.loads(core('audit/source_seals.json'));anchor=anchors['residual_ou_real_v1'];assert Path(anchor['root'])==RES
for f,k in [('complete.json','complete_sha256'),('audit.json','audit_sha256'),('code_registration.json','code_registration_sha256'),('preregistration.json','preregistration_sha256')]:assert C.sha(RES/f)==anchor[k],f
EXPECTED=anchor['expected']
def resraw(rel):
 assert rel in EXPECTED,('unsealed residual source',rel)
 b=C.raw(RES/rel);assert hashlib.sha256(b).hexdigest()==EXPECTED[rel],rel;return b
def cache(stem):
 b=resraw(stem+'.pkl.gz');r=json.loads(resraw(stem+'.ready.json'));assert hashlib.sha256(b).hexdigest()==r['sha256'];v=pickle.loads(gzip.decompress(b));assert v['preregistration_sha256']==anchor['preregistration_sha256'];assert hashlib.sha256(v['payload']).hexdigest()==v['sha256'];return pickle.loads(v['payload'])
registry_path=Path('/Users/maxiyao/FYP_runs/fyp_registered_conditionals_v1/REGISTERED_CONDITIONAL_PATHS.json')
assert C.sha(registry_path)=='8bc237d1f1daca88249d305b7388f576842e125c0c431986aeb9c895de0357e8'
registry={r['path']:r for r in C.js(registry_path)['entries']}
wheel=Path('/Users/maxiyao/FYP_runs/rf_cost_accounting_v1/audit/standing_wheel_addendum_20261031/REGISTERED_CONDITIONAL_PATHS_ADDENDUM.json');assert C.sha(wheel)=='a876656bbeceb10f05b5a4edbdcadc1ae946081f482b75c2503c3679b4bac94a'
REG=json.loads(resraw('audit/input_registration.json'))['selected']
_fresh='experiments/15_pair_trading/fresh_holdout_data_v1/'
_b=C.raw(C.FYP/(_fresh+'manifest.json'));assert hashlib.sha256(_b).hexdigest()==REG[_fresh+'manifest.json']['sha256']
for _v in json.loads(_b)['files']:
 REG.setdefault(_fresh+_v['path'],{'sha256':_v['sha256']})
def input_bytes(logical):
 p=C.FYP/logical;e=registry.get(logical);options=[p]+([Path(v) for v in e.get('physical_paths',[]) if Path(v)!=p] if e else [])
 expected=REG.get(logical,{}).get('sha256')
 if e and e.get('sealed_sha256'):
  if expected:assert expected==e['sealed_sha256']
  expected=e['sealed_sha256']
 assert expected,('no exact input seal',logical)
 good=[]
 for q in options:
  if C.check(q,False):
   b=C.raw(q);assert hashlib.sha256(b).hexdigest()==expected,('input mismatch',q)
   if e and e.get('size') is not None:assert len(b)==e['size']
   good.append((q,b))
 if not good:return None,None
 return good[0]
def panel():
 import numpy as np
 meta=cache('runtime/data/calendar_v2/panel');out={'meta':meta}
 for key in ['dates','returns','membership']:
  rel='runtime/data/calendar_v2/'+key+'.npy';assert meta['files'][rel]==EXPECTED[rel];resraw(rel);out[key]=np.load(RES/rel,mmap_mode='r',allow_pickle=False)
 return out
