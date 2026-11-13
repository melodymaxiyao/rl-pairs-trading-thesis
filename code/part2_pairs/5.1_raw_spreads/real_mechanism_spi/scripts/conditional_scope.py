"""Authenticated two-path extension of the unchanged storage addendum-2 policy."""
from common import *
EXACT={'.gitignore','experiments/15_pair_trading/economic_peer_residual_screen_rolling/.gitignore'}

def extension():
 authorization=HERE/'addendum_004_gitignore_hydration_authorization.json';a=read(authorization)
 outcome=read(HERE/'addendum_005_gitignore_hydration_outcome.json')
 assert outcome['authorization_sha256']==sha(authorization)
 assert set(a['paths'])==EXACT and set(outcome['fallback_paths'])<=EXACT
 assert sha(HERE/a['baseline_file'])==a['baseline_sha256']==outcome['baseline_sha256']
 b=read(HERE/a['baseline_file']);assert set(b)==EXACT
 return outcome,b

def original_policy():
 sys.path.insert(0,str(COMPACT));import addendum_2_policy as p
 configure()
 for module in [p,p.s]:
  path=Path(module.__file__);assert sha(path)==registration()[str(path.relative_to(ROOT))]
 return p

def known_materialized(outcome):
 seen=set(outcome['materialized_verified_paths'])
 for p in (HERE/'audit/gitignore_hydration').glob('verified_materialized_*.json'):
  r=read(p);assert set(r['paths'])<=EXACT;seen.update(r['paths'])
 return seen

@contextmanager
def install(emit=False):
 """Adapt only the in-memory scope; sealed policy and manifests stay unchanged."""
 outcome,extra=extension();p=original_policy()
 baseline,resolved,snapshot=p.baseline,p.resolved,p.snapshot
 verified=known_materialized(outcome);announced=set()
 def merged_baseline():
  old=baseline();assert not old.keys()&extra.keys();return dict(old,**extra)
 def merged_resolved():return dict(resolved(),**{rel:dict(status='MATERIALIZED_SEALED_HASH_PASS') for rel in verified})
 def scoped_snapshot(verify_materialized=True,persist=False):
  assert not persist,'Sealed addendum-2 state is read-only; publish task-local proofs instead'
  r=snapshot(verify_materialized=verify_materialized,persist=False)
  for row in r['files']:
   if row['path'] not in extra:continue
   row['label']='non-scientific .gitignore, non-live'
   if row['status']=='MATERIALIZED_SEALED_HASH_PASS':
    verified.add(row['path'])
    if emit and row['path'] not in announced:
     print('GITIGNORE_MATERIALIZED_VERIFIED '+json.dumps(row,sort_keys=True),flush=True);announced.add(row['path'])
  r.update(extension_addendum='addendum_005_gitignore_hydration_outcome.json',extension_sha256=sha(HERE/'addendum_005_gitignore_hydration_outcome.json'),extension_paths=sorted(extra),extension_label='non-scientific .gitignore, non-live',original_sealed_files_modified=False)
  return r
 p.baseline,p.resolved,p.snapshot=merged_baseline,merged_resolved,scoped_snapshot
 try:yield p
 finally:p.baseline,p.resolved,p.snapshot=baseline,resolved,snapshot

def state(persist=False):
 with install() as p:r=p.snapshot(verify_materialized=True,persist=False)
 if persist:
  outcome,_=extension();known=known_materialized(outcome)
  new={v['path'] for v in r['files'] if v['path'] in EXACT and v['status']=='MATERIALIZED_SEALED_HASH_PASS'}-known
  if new:write(HERE/f'audit/gitignore_hydration/verified_materialized_{time.time_ns()}.json',dict(paths=sorted(new),proof=r))
 return r

def allowed_paths():
 p=original_policy();return set(p.baseline())|EXACT

def materialization_transition(old,new):
 """Allow only a verified dataless-to-materialized availability transition."""
 rel=old['path']
 if rel!=new['path'] or not int(old['flags'])&DATALESS or int(new['flags'])&DATALESS:return False
 if old['sha256']!='CONDITIONAL_DATALESS_CONTENT_HASH_EXCLUDED':return False
 p=original_policy();expected=dict(p.baseline(),**extension()[1]).get(rel)
 if expected is None:return False
 if any(int(old[k])!=int(new[k]) or int(new[k])!=expected[k] for k in ['size','mode','mtime_ns']):return False
 if int(old['flags'])!=expected['flags'] or int(new['flags'])&~(DATALESS|0x20)!=expected['flags']&~(DATALESS|0x20):return False
 if new['sha256']!=expected['sealed_sha256']:return False
 current=p.inspect_path(rel,expected,{},verify_materialized=True)
 return current['status']=='MATERIALIZED_SEALED_HASH_PASS'

def capture_log(log):
 rows=[json.loads(line.split(' ',1)[1]) for line in Path(log).read_text().splitlines() if line.startswith('GITIGNORE_MATERIALIZED_VERIFIED ')]
 if rows:
  paths={r['path'] for r in rows};assert paths<=EXACT
  for r in rows:assert r['sha256']==extension()[1][r['path']]['sealed_sha256']
  write(HERE/f'audit/gitignore_hydration/verified_materialized_{time.time_ns()}.json',dict(paths=sorted(paths),log=str(Path(log).relative_to(HERE)),log_sha256=sha(log),files=rows))
