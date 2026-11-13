"""Exact 303-path availability policy; read-only verification, optional authorized repair."""
import csv,hashlib,importlib.util,json,os,stat,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
QUARANTINE=Path('/Users/maxiyao/Desktop/FYP_dataless')
CONTROL=Path('/Users/maxiyao/FYP_runs/real_mechanism_spi_v1_finalization')
DATALESS=0x40000000;UF_ALLOWED=0x8040
_WRITABLE={}
def digest(path):
 assert not path.lstat().st_flags&DATALESS,('Dataless content read prohibited',str(path))
 with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def metadata(s):
 return dict(size=s.st_size,mode=stat.S_IMODE(s.st_mode),mtime_ns=s.st_mtime_ns,atime_ns=s.st_atime_ns,ino=s.st_ino,dev=s.st_dev,nlink=s.st_nlink,uid=s.st_uid,gid=s.st_gid,flags=s.st_flags)
def policy():
 auth=json.loads((HERE/'addendum_013_303_dataless_metadata.json').read_text());assert auth['status']=='AUTHORIZED' and auth['path_count']==303
 path=HERE/auth['scope_file'];assert digest(path)==auth['scope_file_sha256']
 rows=json.loads(path.read_text())['conditional_paths'];names=[r['path'] for r in rows]
 assert len(names)==len(set(names))==303
 assert hashlib.sha256(('\n'.join(sorted(names))+'\n').encode()).hexdigest()==auth['paths_sha256']
 for row in rows:assert not Path(row['path']).is_absolute() and '..' not in Path(row['path']).parts
 return auth,rows
def assess(row,current,materialized_sha=None):
 e=row['sealed_metadata'];a=metadata(current)
 assert stat.S_ISREG(current.st_mode),('Conditional path is not a regular file',row['path'])
 assert a['size']==row['size'],('Conditional size differs',row['path'])
 if a['flags']&DATALESS:
  assert e['mode']==row['mode']==0o644
  assert a['mode'] in {e['mode'],0o600},('Unapproved dataless mode',row['path'])
  assert a['mtime_ns'] in {e['mtime_ns'],e['mtime_ns']//10**9*10**9},('Unapproved dataless mtime',row['path'])
  assert a['flags']==e['flags'],('Dataless flags differ',row['path'])
  for field in ['dev','nlink','uid','gid']:assert a[field]==e[field],('Other dataless metadata differs',row['path'],field)
  assert a['ino']>0
  changes={field:dict(sealed=e[field],observed=a[field]) for field in ['mode','mtime_ns','ino'] if a[field]!=e[field]}
  return dict(status='DATALESS_CONDITIONAL_CONTENT_EXCLUDED',metadata_changes=changes,content_read=False,repaired=False)
 assert materialized_sha==row['sha256'],('Materialized sealed hash differs',row['path'])
 assert a['mode']==row['mode'] and a['mtime_ns']==e['mtime_ns'],('Materialized mode/mtime differs; repair requires prior sealed hash proof',row['path'])
 # Same verified availability transition as storage-compaction: DATALESS and
 # compression representation may differ only after an exact content check.
 delta=(a['flags']^e['flags'])&~(DATALESS|0x20)
 assert not delta&~UF_ALLOWED,('Unsupported materialized flag change',row['path'])
 return dict(status='MATERIALIZED_SEALED_HASH_PASS',sha256=materialized_sha,content_read=True,repaired=False,flag_only_changes=delta,observed_inode=a['ino'])
def checked_content(path,row):
 before=path.lstat();assert stat.S_ISREG(before.st_mode) and not before.st_flags&DATALESS
 assert before.st_size==row['size'],('Materialized size differs before hashing',row['path'])
 fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
 try:
  opened=os.fstat(fd);assert (opened.st_dev,opened.st_ino)==(before.st_dev,before.st_ino)
  with os.fdopen(fd,'rb',closefd=False) as stream:value=hashlib.file_digest(stream,'sha256').hexdigest()
 finally:os.close(fd)
 after=path.lstat();a=metadata(before);b=metadata(after)
 for field in ['size','mode','mtime_ns','ino','dev','nlink','uid','gid']:assert a[field]==b[field],('File changed during sealed-hash check',str(path),field)
 assert not after.st_flags&DATALESS and not (before.st_flags^after.st_flags)&~UF_ALLOWED
 assert value==row['sha256'],('Materialized hash mismatch; no repair allowed',row['path'])
 return before,value
def repair_checked(path,row,journal,append,preserve_atime_ns=None):
 before,sha=checked_content(path,row);old=metadata(before);desired=row['sealed_metadata'];preserved_atime=old['atime_ns'] if preserve_atime_ns is None else preserve_atime_ns
 if old['mode']==row['mode'] and old['mtime_ns']==desired['mtime_ns']:return False
 assert not ((before.st_flags^desired['flags'])&~(DATALESS|0x20|UF_ALLOWED)),('Unsupported flags before repair',row['path'])
 append(journal,dict(event='sealed_content_verified_before_metadata_restore',time_ns=time.time_ns(),path=row['path'],physical_path=str(path),sha256=sha,before=old,target_mode=row['mode'],target_mtime_ns=desired['mtime_ns']))
 _WRITABLE[str(path.absolute())]=(before.st_dev,before.st_ino)
 try:
  if old['mode']!=row['mode']:
   append(journal,dict(event='chmod_intent',path=row['path'],before=old,target_mode=row['mode'],sealed_sha256=sha));os.chmod(path,row['mode'],follow_symlinks=False)
  append(journal,dict(event='utime_intent',path=row['path'],preserved_atime_ns=preserved_atime,target_mtime_ns=desired['mtime_ns'],sealed_sha256=sha))
  os.utime(path,ns=(preserved_atime,desired['mtime_ns']),follow_symlinks=False)
  _,after_sha=checked_content(path,row);after=path.lstat()
  assert stat.S_IMODE(after.st_mode)==row['mode'] and after.st_mtime_ns==desired['mtime_ns']
  if after.st_atime_ns!=preserved_atime:os.utime(path,ns=(preserved_atime,desired['mtime_ns']),follow_symlinks=False)
  after=path.lstat();assert after.st_atime_ns==preserved_atime and (after.st_dev,after.st_ino)==(before.st_dev,before.st_ino)
  append(journal,dict(event='restored_and_verified',path=row['path'],physical_path=str(path),before=old,after=metadata(after),sealed_sha256=after_sha,content_writes=0,flags_cleared=False))
 finally:_WRITABLE.pop(str(path.absolute()),None)
 return True
def journal_primitive():
 path=ROOT/'experiments/15_pair_trading/tav1_runtime_prune_v1/prune_engine.py'
 assert digest(path)=='088c6f07f1b76dd1a09b9286b7b8dd7bf6ffdfe8223c5dd76b5217ebdc34b5c9'
 spec=importlib.util.spec_from_file_location('_unchanged_303_journal',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 return module.append
def snapshot(repair=False,journal=None):
 auth,rows=policy();entries=[];absent=[];repairs=0;append=None
 for row in rows:
  found=False
  for base in [ROOT,QUARANTINE]:
   path=base/row['path']
   if not os.path.lexists(path):continue
   found=True;current=path.lstat()
   assert stat.S_ISREG(current.st_mode),('Conditional path type differs',str(path))
   for parent in path.parents:
    if parent==base:break
    assert not parent.is_symlink(),('Conditional path parent substituted by symlink',str(parent))
   value=None;restored=False
   if not current.st_flags&DATALESS:
    hash_before,value=checked_content(path,row)
    if repair and (stat.S_IMODE(current.st_mode)!=row['mode'] or current.st_mtime_ns!=row['sealed_metadata']['mtime_ns']):
     assert journal is not None
     if append is None:append=journal_primitive()
     restored=repair_checked(path,row,Path(journal),append,preserve_atime_ns=hash_before.st_atime_ns);repairs+=int(restored)
    current=path.lstat()
   result=assess(row,current,value);result['repaired']=restored
   entries.append(dict(path=row['path'],physical_path=str(path),sealed_sha256=row['sha256'],sealed_metadata=row['sealed_metadata'],observed_metadata=metadata(current),**result))
  if not found:absent.append(row['path'])
 available={x['path'] for x in entries if x['status']=='MATERIALIZED_SEALED_HASH_PASS'}
 return dict(content_unavailable_paths=303-len(available),status='PASS_AUTHORIZED_303_CONDITIONAL_PROVENANCE',authorized_paths=303,scope_paths_sha256=auth['paths_sha256'],absent_paths=len(absent),dataless_files=sum(x['status']=='DATALESS_CONDITIONAL_CONTENT_EXCLUDED' for x in entries),materialized_files=sum(x['status']=='MATERIALIZED_SEALED_HASH_PASS' for x in entries),accepted_dataless_metadata_changes=sum(bool(x.get('metadata_changes')) for x in entries),metadata_repairs=repairs,downloads=0,content_writes=0,files=entries,still_absent=absent)
def install_guard():
 _,rows=policy();allowed={str((base/r['path']).absolute()) for base in [ROOT,QUARANTINE] for r in rows}
 def hook(event,args):
  if event in ['socket.connect','socket.getaddrinfo']:raise PermissionError('No network or downloads')
  writes=[]
  if event in ['os.scandir','os.listdir'] and isinstance(args[0],(str,bytes,os.PathLike)):
   q=Path(os.fsdecode(args[0])).absolute();assert not q.lstat().st_flags&DATALESS,('No dataless directory expansion',str(q))
  if event=='open':
   p,mode,flags=args
   if isinstance(p,(str,bytes,os.PathLike)):
    q=Path(os.fsdecode(p)).absolute()
    try:s=q.lstat()
    except OSError:s=None
    if s is not None and s.st_flags&DATALESS:raise PermissionError('No dataless content access: '+str(q))
   if isinstance(mode,str) and any(c in mode for c in 'wax+') or isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):writes=[p]
  elif event in ['os.chmod','os.utime']:
   q=Path(args[0]).absolute();key=str(q)
   if key in allowed:
    s=q.lstat();assert key in _WRITABLE and not s.st_flags&DATALESS and (s.st_dev,s.st_ino)==_WRITABLE[key],('Metadata write lacks fresh sealed-hash proof',key)
    return
   writes=[args[0]]
  elif event in ['os.remove','os.rmdir','os.mkdir','os.chflags']:writes=[args[0]]
  elif event in ['os.rename','os.link']:writes=args[:2]
  elif event=='os.symlink':writes=[args[1]]
  for path in writes:
   if isinstance(path,(str,bytes,os.PathLike)):
    q=Path(os.fsdecode(path)).absolute();assert q.is_relative_to(HERE) or q.is_relative_to(CONTROL),('Write outside authorized finalization scope',str(q))
 sys.addaudithook(hook)
