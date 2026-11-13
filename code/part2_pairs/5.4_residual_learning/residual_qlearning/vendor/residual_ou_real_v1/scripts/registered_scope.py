"""Authenticated exact standing registry; no source writes or forced hydration."""
from common import *
import csv
CONTROL=Path('/Users/maxiyao/FYP_runs/fyp_registered_conditionals_v1')
REGISTRY=CONTROL/'REGISTERED_CONDITIONAL_PATHS.json'
REGISTRY_SHA256='8bc237d1f1daca88249d305b7388f576842e125c0c431986aeb9c895de0357e8'
_ROWS=None
def rows(refresh=False):
 global _ROWS
 if refresh:_ROWS=None
 if _ROWS is not None:return _ROWS
 assert sha(REGISTRY)==REGISTRY_SHA256,('Standing conditional registry changed',str(REGISTRY))
 r=json.loads(REGISTRY.read_text());assert r['status']=='AUTHORIZED_STANDING_CONDITIONAL_EXACT_PATHS';assert sha(Path(r['authorization_file']))==r['authorization_sha256']
 for source in r['source_registries']:assert sha(Path(source['path']))==source['sha256'],('Registered source list changed',source['path'])
 names=[e['path'] for e in r['entries']];assert len(names)==len(set(names))==r['unique_paths']
 assert hashlib.sha256(('\n'.join(sorted(names))+'\n').encode()).hexdigest()==r['paths_sha256']
 for e in r['entries']:
  rel=Path(e['path']);assert not rel.is_absolute() and '..' not in rel.parts
  assert str(INPUT/rel) in e['physical_paths']
  for p in e['physical_paths']:assert Path(p).is_absolute() and (Path(p).is_relative_to(INPUT) or Path(p).is_relative_to(Path('/Users/maxiyao/Desktop/FYP_dataless')))
 _ROWS={e['path']:e for e in r['entries']};return _ROWS
def meta(s):return dict(size=s.st_size,mode=stat.S_IMODE(s.st_mode),mtime_ns=s.st_mtime_ns,flags=s.st_flags,ino=s.st_ino,dev=s.st_dev,nlink=s.st_nlink,uid=s.st_uid,gid=s.st_gid)
def assess_one(path,row):
 path=Path(path);base=dict(path=row['path'],physical_path=str(path),sealed_sha256=row.get('sealed_sha256'),sealed_size=row.get('size'),content_read=False)
 try:s=path.lstat()
 except FileNotFoundError:return dict(base,status='REGISTERED_UNAVAILABLE',availability='absent',content_hash_excluded=True,metadata_changes={})
 observed=meta(s);frozen=row.get('sealed_metadata',{k:row[k] for k in ['size','mode','mtime_ns','flags','ino'] if k in row});changes={k:dict(sealed=frozen[k],observed=v) for k,v in observed.items() if k in frozen and frozen[k]!=v}
 common=dict(base,metadata=observed,metadata_changes=changes)
 if s.st_flags&DATALESS:return dict(common,status='REGISTERED_DATALESS',availability='dataless',content_hash_excluded=True)
 if row.get('kind','file')=='directory':
  assert stat.S_ISDIR(s.st_mode),('Registered cache namespace is not a directory',row['path'])
  return dict(common,status='REGISTERED_OPAQUE_CACHE_DIRECTORY',availability='directory',content_hash_excluded=False,directory_members_verified=False)
 assert stat.S_ISREG(s.st_mode),('Materialized registered path is not a regular file',row['path'])
 assert row.get('sealed_sha256') and row.get('size') is not None,('No sealed hash/size available for materialized registered file',row['path'])
 assert s.st_size==row['size'],('Materialized registered size mismatch',row['path'],s.st_size,row['size'])
 digest=sha(path);assert digest==row['sealed_sha256'],('Materialized registered sealed hash mismatch',row['path'],digest,row['sealed_sha256'])
 after=path.lstat();assert not after.st_flags&DATALESS and after.st_size==row['size'] and after.st_ino==s.st_ino,('File changed during content verification; no stable proof',row['path'])
 common.update(content_read=True,sha256=digest,metadata=meta(after))
 return dict(common,status='REGISTERED_MATERIALIZED_SEALED_HASH_PASS',availability='materialized',content_hash_excluded=False)
def inspect_row(row):
 # Every materialized alias is checked. A good copy never excuses a bad one.
 physical=[assess_one(Path(p),row) for p in row['physical_paths']];available=[p for p in physical if p['availability']=='materialized']
 return dict(path=row['path'],kind=row['kind'],sealed_sha256=row['sealed_sha256'],size=row['size'],registries=row['registries'],physical=physical,content_hash_excluded=not available if row['kind']=='file' else all(p['content_hash_excluded'] for p in physical),availability='materialized' if available else 'unavailable',verified_aliases=[p['physical_path'] for p in available])
def snapshot(excluded=None):
 result={}
 for rel,row in rows().items():
  if excluded is None or not excluded(INPUT/rel):result[rel]=inspect_row(row)
 return result
def persist(label,states):
 token=str(time.time_ns());path=RUN/f'audit/registered_conditionals_{label}_{token}.json';write(path,dict(status='PASS_WITH_REGISTERED_CONDITIONALS',registry_sha256=REGISTRY_SHA256,logical_paths=len(states),content_hash_excluded=sum(r['content_hash_excluded'] for r in states.values()),materialized_verified=sum(r['availability']=='materialized' for r in states.values()),files=states))
 changes=[]
 for rel,r in states.items():
  for p in r['physical']:
   for field,values in p['metadata_changes'].items():changes.append([rel,p['physical_path'],p['availability'],field,values['sealed'],values['observed'],'LOGGED_NOT_FATAL'])
 log=RUN/f'audit/registered_metadata_changes_{label}_{token}.csv'
 with log.open('x',newline='') as f:
  writer=csv.writer(f);writer.writerow(['path','physical_path','availability','field','sealed','observed','action']);writer.writerows(changes);f.flush();os.fsync(f.fileno())
 return dict(file=str(path.relative_to(RUN)),sha256=sha(path),metadata_log=str(log.relative_to(RUN)),metadata_log_sha256=sha(log),registered_paths=len(states),conditional_paths=sum(r['content_hash_excluded'] for r in states.values()),materialized_verified=sum(r['availability']=='materialized' for r in states.values()),metadata_changes_logged=len(changes))
