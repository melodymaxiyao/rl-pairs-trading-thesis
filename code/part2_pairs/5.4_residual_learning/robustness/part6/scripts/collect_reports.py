import sys,re,json,hashlib
from pathlib import Path
sys.path.insert(0,'/Users/maxiyao/FYP_runs/closing_checks_v1/part4/scripts')
import ext_common as C
C.guard()
R4=C.O/'part4'
regpath=Path('/Users/maxiyao/FYP_runs/fyp_registered_conditionals_v1/REGISTERED_CONDITIONAL_PATHS.json')
b=C.raw(regpath);assert hashlib.sha256(b).hexdigest()=='8bc237d1f1daca88249d305b7388f576842e125c0c431986aeb9c895de0357e8'
registry={r['path']:r for r in json.loads(b)['entries']}
wpath=Path('/Users/maxiyao/FYP_runs/rf_cost_accounting_v1/audit/standing_wheel_addendum_20261031/REGISTERED_CONDITIONAL_PATHS_ADDENDUM.json')
assert C.sha(wpath)=='a876656bbeceb10f05b5a4edbdcadc1ae946081f482b75c2503c3679b4bac94a'
rows=C.rows('/Users/maxiyao/FYP_runs/thesis_appendix_info_v1/pairs_register_combined.csv');rr={r['experiment']:r for r in rows}
catalog=C.js('/Users/maxiyao/FYP_runs/thesis_appendix_info_v1/audit/catalog_index.json')
prior=C.js('/Users/maxiyao/FYP_runs/thesis_appendix_info_v1/audit/input_hashes_before.json')
groups={
'economic_and_universe':['economic_peer_residual_screen','economic_peer_residual_screen_rolling']+[f'universe_expansion_v{x}' for x in range(1,5)]+['universe_sector_expansion_v1','cross_sector_fast_universe_v1'],
'regression_and_convergence':['zreversion_diagnostic_v1','zreversion_diagnostic_v2','screening_v2','rolling_v3','stock_pairs_rolling_v1','expanded_convergence_gate_v1','holdout_expanded_personal_gate_v1','refresh_spectrum_stage1_v1','drift_aligned_reversion_v1','pair_property_persistence_v1'],
'distance_and_filters':['ggr_large_universe_dev_v1','ggr_large_universe_confirm_2004_2008_v1']+[f'ggr_confirm_robustness_v{x}' for x in range(1,4)]+['meta_filter_dev_v1','meta_filter_confirm_2004_2008_v1'],
'learner_holdout':['qlearning_zpos_v3','qlearning_holdout_pre2018_v1','mb_frozen_2004_2008_v1'],
'residual_screening':['residual_learnability_v1']}
def allowed(rel):
 p=Path(rel)
 if any(x in p.parts for x in ['vendor','deps','runtime','__pycache__','test','tests','verification','.git']):return False
 n=p.name.lower()
 return p.suffix.lower() in ['.md','.txt','.json'] and (n.startswith(('readme','results','report','summary')) or n=='complete.json')
def resolve(p):
 e=registry.get(str(p.relative_to(C.FYP))) if p.is_relative_to(C.FYP) else None
 options=[p]+([Path(q) for q in e.get('physical_paths',[]) if Path(q)!=p] if e else [])
 good=[]
 for q in options:
  if not C.check(q,False):continue
  if not q.is_file():continue
  raw=C.raw(q)
  if e and e.get('sealed_sha256'):assert hashlib.sha256(raw).hexdigest()==e['sealed_sha256'] and len(raw)==e['size'],('sealed mismatch',q)
  if str(q) in prior:assert hashlib.sha256(raw).hexdigest()==prior[str(q)]['sha256'],('prior read changed',q)
  good.append(q)
 return good[0] if good else None
documents={};miss=[]
for family,names in groups.items():
 for name in names:
  if name not in rr:miss.append({'experiment':name,'reason':'not registered'});continue
  root=Path(rr[name]['result_location']);candidates={}
  for rel,p in catalog.get(name,{}).get('files',{}).items():
   if allowed(rel):candidates[rel]=root/rel
  # Exact registered paths expose materialised quarantined documents without traversing missing trees.
  if root.is_relative_to(C.FYP):
   prefix=str(root.relative_to(C.FYP))+'/'
   for path,e in registry.items():
    if path.startswith(prefix):
     rel=path[len(prefix):]
     if allowed(rel) and len(Path(rel).parts)<=3:candidates[rel]=C.FYP/path
  for rel in ['README.md','RESULTS.md','REPORT.md','summary.json','complete.json']:
   candidates.setdefault(rel,root/rel)
  for folder in ['outputs','reports']:
   q=root/folder
   if C.check(q,False) and q.is_dir():
    for p in q.iterdir():
     if allowed(str(p.relative_to(root))):candidates[str(p.relative_to(root))]=p
  docs={}
  for rel,p in sorted(candidates.items()):
   resolved=resolve(p)
   if resolved is None:continue
   if resolved.stat().st_size>2_000_000:miss.append({'experiment':name,'path':str(resolved),'reason':'summary document >2MB; no automatic truncation/read'});continue
   text=C.text(resolved);docs[rel]={'path':str(resolved),'sha256':C.READS[str(resolved)]['sha256'],'text':text}
  documents[name]={'family':family,'root':str(root),'documents':docs}
  print(name,len(docs),'docs',flush=True)
C.dump(C.P6/'audit/documents.json',documents);C.dump(C.P6/'audit/discovery_gaps.json',miss)
C.dump(C.P6/'audit/inputs_before.json',{k:v for k,v in C.READS.items() if not any(Path(k).is_relative_to(r) for r in C.ROOTS)})
C.dump(C.P6/'audit/unavailable.json',C.UNAVAILABLE)
print('TOTAL',len(documents),sum(len(d['documents']) for d in documents.values()))
