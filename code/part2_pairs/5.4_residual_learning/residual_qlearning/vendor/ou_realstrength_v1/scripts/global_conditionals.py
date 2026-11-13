"""Authenticate the seven exact global standing additions and approved policy writes."""
from common import *
ROOT=Path('/Users/maxiyao/FYP_runs/rf_cost_accounting_v1/audit/standing_wheel_addendum_20261031')
REGISTRY=ROOT/'REGISTERED_CONDITIONAL_PATHS_ADDENDUM.json'
REGISTRY_SHA='a876656bbeceb10f05b5a4edbdcadc1ae946081f482b75c2503c3679b4bac94a'
BASE_SHA='8bc237d1f1daca88249d305b7388f576842e125c0c431986aeb9c895de0357e8'
POLICY_SHA='b06ce6ab453783e4d214fb6e8f665349c8e765d6a0cb14ace6e2a575188cf8ba'

def metadata(s):return dict(size=s.st_size,mode=stat.S_IMODE(s.st_mode),mtime_ns=s.st_mtime_ns,flags=s.st_flags,ino=s.st_ino)

def assess_file(path,row):
 """Metadata-only for dataless/absent; original sealed bytes required if readable."""
 path=Path(path)
 try:s=path.lstat()
 except FileNotFoundError:return dict(path=row['path'],status='REGISTERED_UNAVAILABLE',content_read=False)
 observed=metadata(s);changes={k:dict(sealed=v,observed=observed[k]) for k,v in row['sealed_metadata'].items() if k in observed and observed[k]!=v}
 if s.st_flags&DATALESS:return dict(path=row['path'],status='REGISTERED_DATALESS',content_read=False,metadata_changes=changes)
 assert stat.S_ISREG(s.st_mode) and s.st_size==row['size'],('Materialized standing WHEEL kind/size mismatch',row['path'])
 digest=sha(path);assert digest==row['sealed_sha256'],('Materialized standing WHEEL SHA256 mismatch',row['path'])
 after=path.lstat();assert not after.st_flags&DATALESS and after.st_ino==s.st_ino and after.st_size==row['size'],('Unstable standing WHEEL content',row['path'])
 return dict(path=row['path'],status='REGISTERED_MATERIALIZED_SEALED_HASH_PASS',content_read=True,sha256=digest,metadata_changes=changes)

def authenticate(check_members=True):
 assert sha(REGISTRY)==REGISTRY_SHA
 reg=read(REGISTRY);assert reg['scope']=='all present and future FYP tasks' and reg['unique_paths']==7 and reg['combined_unique_paths']==8908
 assert reg['base_registry_sha256']==BASE_SHA and sha(Path(reg['base_registry']))==BASE_SHA
 for file,key in [('authorization_file','authorization_sha256'),('inventory_member_proof','inventory_member_proof_sha256')]:assert sha(Path(reg[file]))==reg[key]
 frozen=read(Path(reg['inventory_member_proof']));rows={e['path']:e for e in reg['entries']};assert len(rows)==7 and set(rows)==set(frozen)
 assert hashlib.sha256(('\n'.join(sorted(rows))+'\n').encode()).hexdigest()==reg['paths_sha256']
 for rel,row in rows.items():
  assert row['physical_paths']==[str(INPUT/rel)] and row['label']=='vendored package metadata of a closed experiment; non-live'
  assert row['sealed_sha256']==frozen[rel]['sha256'] and row['size']==int(frozen[rel]['size'])
 controls=read(ROOT/'policy_changes.json');assert sha(ROOT/'policy_changes.json')==POLICY_SHA and controls['addendum_sha256']==REGISTRY_SHA
 assert controls['authorization_sha256']==reg['authorization_sha256'] and controls['source_policy_writes']==2 and controls['scientific_source_writes']==0
 for rel,row in controls['files'].items():
  assert sha(Path(row['before_snapshot']))==row['before_sha256'] and sha(Path(row['approved_after_snapshot']))==row['after_sha256']
  path=INPUT/rel;assert sha(path)==row['after_sha256'],('Approved policy afterimage changed',rel)
  current=meta(path);old=row['after_metadata'];assert all(current[k]==old[k] for k in ['size','mode','mtime_ns','ino']) and not ((current['flags']^old['flags'])&~0x8040),('Approved policy metadata changed',rel)
 observations=[assess_file(Path(p),row) for row in rows.values() for p in row['physical_paths']] if check_members else []
 return reg,rows,observations

def merge(base):
 reg,extra,_=authenticate(False)
 assert len(base)==8901 and not set(base)&set(extra),'Standing exact registry overlap or unexpected base'
 value=dict(base);value.update(extra);assert len(value)==8908;return value

def install(module):
 original_rows=module.rows;original_persist=module.persist
 def rows(refresh=False):return merge(original_rows(refresh=refresh))
 def persist(label,states):
  result=original_persist(label,states)
  result['standing_addendum']=dict(path=str(REGISTRY),sha256=REGISTRY_SHA,base_registry_sha256=BASE_SHA,global_unique_paths=8908,exact_added_paths=7,policy_changes_sha256=POLICY_SHA)
  return result
 module.rows=rows;module.persist=persist
