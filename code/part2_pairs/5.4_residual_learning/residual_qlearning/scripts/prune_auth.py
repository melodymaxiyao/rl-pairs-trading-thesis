"""One exact authorized external-run checkpoint prune; no directory waivers."""
from pathlib import Path
import json,hashlib,gzip,csv,stat,os
RUN=Path(__file__).resolve().parents[1]
ROOT=Path('/Users/maxiyao/Desktop/FYP');TARGET=Path('/Users/maxiyao/FYP_runs/ou_dp_qlearning_v1');OUT=RUN/'audit/ou_dp_checkpoint_trim';REGISTRY=ROOT/'experiments/15_pair_trading/PRUNE_REGISTRY.json'
DATALESS=0x40000000
def sha(p):
 p=Path(p);s=p.lstat();assert stat.S_ISREG(s.st_mode) and not s.st_flags&DATALESS,('Unavailable/nonregular',str(p))
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def metadata(p):
 p=Path(p);s=p.lstat();return dict(size=s.st_size,mode=stat.S_IMODE(s.st_mode),mtime_ns=s.st_mtime_ns,ino=s.st_ino,flags=s.st_flags)
def rows():
 with gzip.open(OUT/'prune_manifest.csv.gz','rt',newline='') as f:return list(csv.DictReader(f))
def authenticate(require_current=True):
 a=read(OUT/'authorization.json');before=read(OUT/'registry.before.json');after=read(OUT/'registry.approved_after.json')
 assert sha(OUT/'registry.before.json')==a['before_sha256'] and sha(OUT/'registry.approved_after.json')==a['after_sha256']
 reduced=json.loads(json.dumps(after));entry=reduced['entries'].pop();assert entry==a['entry'] and reduced==before
 assert entry['source_root']==str(TARGET) and entry['scope']=='external_run_checkpoint_bodies' and entry['status']=='active_registered_pruning'
 for field,hfield in [('manifest','manifest_sha256'),('recipe','recipe_sha256')]:assert sha(Path(entry[field]))==entry[hfield]
 assert sha(Path(entry['audit_directory'])/'audit.json')==entry['audit_sha256']
 assert sha(OUT/'preregistration.json')==entry['preregistration_sha256']
 assert sha(TARGET/'complete.json')==read(OUT/'source_seals.json')['complete_sha256']
 audit=read(TARGET/'tier_3/audit.json');assert sha(TARGET/'tier_3/audit.json')==read(TARGET/'tier_3/complete.json')['audit_sha256']
 rr=rows();names=[r['path'] for r in rr];assert len(set(names))==len(rr)==entry['files']
 assert hashlib.sha256(('\n'.join(sorted(names))+'\n').encode()).hexdigest()==entry['paths_sha256']
 for row in rr:
  rel=Path(row['path']);p=TARGET/rel
  assert not rel.is_absolute() and '..' not in rel.parts and len(rel.parts)>=3 and rel.parts[:2] in [('runtime','seeds'),('runtime','worlds')] and p.name.endswith('.pkl.gz')
  assert row['physical_path']==str(p) and audit['artifacts'][str(rel)]==row['sha256']
  assert row['ready_path']==str(rel)[:-7]+'.ready.json'
  ready=TARGET/row['ready_path'];assert sha(ready)==row['ready_sha256'] and read(ready)['sha256']==row['sha256']
 if require_current:
  assert sha(REGISTRY)==a['after_sha256'] and metadata(REGISTRY)==a['after_metadata'],'Only exact approved append is allowed'
 return a,{r['physical_path']:r for r in rr}
def checked_source_sha(p,registered,base_sha=sha,accepted=None):
 p=Path(p).absolute();row=registered.get(str(p))
 try:s=p.lstat()
 except FileNotFoundError:
  assert row is not None,('Unregistered missing input',str(p))
  if accepted is not None:accepted.add(str(p))
  # This is the authenticated historical digest, not a newly computed hash.
  return row['sha256']
 if row is not None:
  assert stat.S_ISREG(s.st_mode) and not s.st_flags&DATALESS and s.st_size==int(row['size'])
  assert stat.S_IMODE(s.st_mode)==int(row['mode']) and s.st_mtime_ns==int(row['mtime_ns']),('Registered checkpoint changed before deletion',str(p))
  h=base_sha(p);assert h==row['sha256'];return h
 return base_sha(p)
