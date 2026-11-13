"""Authenticate the exact external OU prune, retaining every old workspace proof."""
from common import *
import prune_auth
REL='experiments/15_pair_trading/PRUNE_REGISTRY.json'
_AUTH=None
def authenticate():
 global _AUTH
 if _AUTH is not None:return _AUTH
 a,rows=prune_auth.authenticate(require_current=False);now=prune_auth.metadata(prune_auth.REGISTRY);old=a['after_metadata'];assert sha(prune_auth.REGISTRY)==a['after_sha256'] and all(now[k]==old[k] for k in ['size','mode','mtime_ns','ino']) and not ((now['flags']^old['flags'])&~0x8040), 'Unapproved registry drift'
 if now['flags']!=old['flags']:write(event('prune_registry_flag_only'),dict(path=str(prune_auth.REGISTRY),before=old,after=now,sha256=a['after_sha256'],action='Keep flags; exact content and other metadata verified'))
 out=RUN/'audit/ou_dp_checkpoint_trim';audit=read(out/'audit.json')
 for f,k in [('retained_seed_models.npz','retained_seed_models_sha256'),('retained_world_bodies.bundle.gz','retained_world_bundle_sha256'),('code_registration.json','code_registration_sha256')]:assert sha(out/f)==audit[k]
 for rel,h in read(out/'code_registration.json')['files'].items():assert sha(RUN/'scripts'/rel)==h
 _AUTH=(a,rows);return _AUTH

def install(a_module,protection):
 a,external=authenticate();old_load=a_module.load;old_meta=a_module.meta
 # Authenticate the immutable A proof against its actual afterimage; the subsequent
 # append/current metadata have been authenticated independently above.
 def load():
  d=read(a_module.DOC_PATH);assert d['after_sha256']==a['before_sha256']
  a_module.meta=lambda p:d['source_after'] if Path(p).absolute()==INPUT/REL else old_meta(p)
  try:return old_load()
  finally:a_module.meta=old_meta
 a_module.load=load
 def approved_registry_hash(pin):
  d=load();assert pin in [d['before_sha256'],d['after_sha256'],a['after_sha256']];return a['after_sha256']
 def expected_registry_record(frozen):
  d=load()
  if frozen is not None:assert all(frozen[k]==d['source_before'][k] for k in ['sha256','size','mode','mtime_ns'])
  m=a['after_metadata'];return dict(frozen or dict(kind='file',target=''),sha256=a['after_sha256'],**{k:m[k] for k in ['size','mode','mtime_ns']})
 a_module.approved_registry_hash=approved_registry_hash;a_module.expected_registry_record=expected_registry_record
 def registry_rows():
  # The external source_root never authorizes a workspace-relative runtime path.
  assert sha(INPUT/REL)==a['after_sha256'];rows={};reg=read(INPUT/REL)
  for e in reg['entries']:
   if e['status']!='active_registered_pruning':continue
   if e.get('scope')=='external_run_checkpoint_bodies':assert e==a['entry'];continue
   for f,k in [('manifest','manifest_sha256'),('recipe','recipe_sha256')]:assert sha(INPUT/e[f])==e[k]
   assert sha(INPUT/e['audit_directory']/'audit.json')==e['audit_sha256']
   with gzip.open(INPUT/e['manifest'],'rt',newline='') as h:rr=list(csv.DictReader(h))
   names=[r['path'] for r in rr];assert len(set(names))==len(names)==e['files'] and hashlib.sha256(('\n'.join(sorted(names))+'\n').encode()).hexdigest()==e['paths_sha256']
   for r in rr:
    p=Path(r['path']);assert not p.is_absolute() and '..' not in p.parts and r['path'] not in rows;rows[r['path']]=r
  return rows
 protection.registry_rows=registry_rows
 return dict(external_checkpoint_absences=len(external),workspace_paths_not_expanded=True,after_sha256=a['after_sha256'])
