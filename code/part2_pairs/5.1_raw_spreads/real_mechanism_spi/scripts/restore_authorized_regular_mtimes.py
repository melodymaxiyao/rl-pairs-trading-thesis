"""Nine-path regular-file adapter using the unchanged pruning journal primitives."""
from common import *
from protection import frozen_inventory,locks,quick_preflight
import csv,importlib.util,argparse

def metadata(s):
 return dict(atime_ns=s.st_atime_ns,mtime_ns=s.st_mtime_ns,size=s.st_size,mode=stat.S_IMODE(s.st_mode),flags=s.st_flags,ino=s.st_ino,dev=s.st_dev,nlink=s.st_nlink)

def checked(path,row):
 before=path.lstat()
 assert stat.S_ISREG(before.st_mode) and not before.st_flags&DATALESS,('Not a materialized regular file',path)
 assert (before.st_size,stat.S_IMODE(before.st_mode))==(int(row['size']),int(row['mode'])),('Size or mode differs from sealed inventory',path)
 # O_NOFOLLOW prevents a link substituted between lstat and open being read.
 fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
 try:
  opened=os.fstat(fd);assert (opened.st_dev,opened.st_ino)==(before.st_dev,before.st_ino)
  with os.fdopen(fd,'rb',closefd=False) as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
 finally:os.close(fd)
 after=path.lstat()
 for key in ['mtime_ns','size','mode','flags','ino','dev','nlink']:assert metadata(after)[key]==metadata(before)[key],('File changed while hashing',path,key)
 assert digest==row['sha256'],('SHA-256 differs from sealed inventory',path)
 return before,digest

def restore_one(root,row,journal,append,original=None):
 path=Path(root)/row['path'];before=path.lstat()
 original=metadata(before) if original is None else original
 for key in ['size','mode','flags','ino','dev','nlink']:assert metadata(before)[key]==original[key],('Metadata changed since repair intent',path,key)
 # Persist the timestamp snapshot before even reading content for validation.
 append(journal,dict(event='verification_intent',time_ns=time.time_ns(),path=row['path'],before=metadata(before),sealed_sha256=row['sha256']))
 verified,digest=checked(path,row)
 assert metadata(verified)==metadata(before),('Metadata changed before restore',path)
 target=int(row['mtime_ns'])
 intent=dict(event='restore_intent',time_ns=time.time_ns(),path=row['path'],before=metadata(before),
             reversal_before=original,after_timestamps=dict(atime_ns=original['atime_ns'],mtime_ns=target),sealed_sha256=digest)
 append(journal,intent)
 os.utime(path,ns=(original['atime_ns'],target),follow_symlinks=False)
 after,digest=checked(path,row)
 assert after.st_mtime_ns==target
 # A content read may advance atime. Preserve the value captured before repair,
 # then inspect metadata without another content read. Journal this too.
 if path.lstat().st_atime_ns!=original['atime_ns']:
  append(journal,dict(event='preserve_atime_intent',time_ns=time.time_ns(),path=row['path'],before=metadata(path.lstat()),atime_ns=original['atime_ns'],mtime_ns=target))
  os.utime(path,ns=(original['atime_ns'],target),follow_symlinks=False)
 final=path.lstat();expected=dict(original,mtime_ns=target)
 assert metadata(final)==expected,('Post-repair metadata mismatch',path,metadata(final),expected)
 result=dict(path=row['path'],before=original,actual_before=metadata(before),after=metadata(final),sha256=digest)
 append(journal,dict(event='restored_and_verified',time_ns=time.time_ns(),**result))
 return result

def repair_guard(paths):
 allowed={str(ROOT/p) for p in paths}
 def hook(event,args):
  if event in ['socket.connect','socket.getaddrinfo']:raise PermissionError('No network/downloads')
  writes=[]
  if event=='open':
   p,mode,flags=args
   if isinstance(p,(str,bytes,os.PathLike)):
    q=Path(os.fsdecode(p)).absolute()
    if q.exists() and q.lstat().st_flags&DATALESS:raise PermissionError('No dataless content reads')
   if isinstance(mode,str) and any(c in mode for c in 'wax+') or isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):writes=[p]
  elif event=='os.utime':
   q=Path(args[0]).absolute()
   if str(q) in allowed and stat.S_ISREG(q.lstat().st_mode):return
   writes=[args[0]]
  elif event in ['os.mkdir','os.remove','os.rmdir','os.chmod','os.chflags']:writes=[args[0]]
  elif event in ['os.rename','os.link']:writes=args[:2]
  elif event=='os.symlink':writes=[args[1]]
  for p in writes:
   if isinstance(p,(str,bytes,os.PathLike)) and os.fsdecode(p)!=os.devnull:inside(p)
 sys.addaudithook(hook)

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--apply',action='store_true');args=parser.parse_args()
 authorization=read(HERE/'addendum_003_nine_regular_mtimes.json');paths=set(authorization['paths']);assert len(paths)==9
 plan=HERE/authorization['plan_file'];assert sha(plan)==authorization['plan_sha256']
 assert {r['path'] for r in csv.DictReader(plan.open())}==paths
 repair_guard(paths)
 with locks():
  quick_preflight();inventory=frozen_inventory();assert sha(inventory)==authorization['inventory_sha256']
  with gzip.open(inventory,'rt',newline='') as f:rows=[r for r in csv.DictReader(f) if r['path'] in paths]
  assert len(rows)==9 and all(r['kind']=='file' for r in rows)
  plan_rows={r['path']:r for r in csv.DictReader(plan.open())}
  for row in rows:
   p=ROOT/row['path'];assert not any(q.is_symlink() for q in [p,*p.parents] if q.is_relative_to(ROOT))
   for key in ['sha256','size','mode']:assert str(row[key])==str(plan_rows[row['path']][key])
   assert int(row['mtime_ns'])==int(plan_rows[row['path']]['frozen_mtime_ns'])
  tool=PRUNE/'prune_engine.py';assert sha(tool)==registration()[str(tool.relative_to(ROOT))]
  module_spec=importlib.util.spec_from_file_location('_regular_repair_journal',tool);journal_tool=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(journal_tool)
  folder=HERE/'audit/regular_mtime_repair';folder.mkdir(exist_ok=True);journal=folder/'journal.jsonl'
  prior=journal_tool.read_journal(journal,repair=True)
  for record in prior:assert record['path'] in paths
  if not args.apply:
   print(json.dumps(dict(status='READY',authorized_paths=sorted(paths),apply=False)));return
  completed={r['path']:r for r in prior if r['event']=='restored_and_verified'};results=[]
  originals={}
  for record in prior:
   if record['event']=='verification_intent':originals.setdefault(record['path'],record['before'])
  for row in rows:
   rel=row['path']
   if rel in completed:
    original=completed[rel];st,digest=checked(ROOT/rel,row);assert st.st_mtime_ns==int(row['mtime_ns'])
    if (ROOT/rel).lstat().st_atime_ns!=original['before']['atime_ns']:
     journal_tool.append(journal,dict(event='preserve_atime_on_resume',time_ns=time.time_ns(),path=rel,before=metadata((ROOT/rel).lstat()),atime_ns=original['before']['atime_ns'],mtime_ns=int(row['mtime_ns'])))
     os.utime(ROOT/rel,ns=(original['before']['atime_ns'],int(row['mtime_ns'])),follow_symlinks=False)
    results.append({k:original[k] for k in ['path','before','actual_before','after','sha256']});continue
   results.append(restore_one(ROOT,row,journal,journal_tool.append,originals.get(rel)))
  assert sha(tool)==registration()[str(tool.relative_to(ROOT))]
  write(folder/'result.json',dict(status='PASS',authorized_paths=9,restored_and_verified=len(results),follow_symlinks=False,atime_preserved=True,
    source_file_content_writes=0,source_metadata_writes='nine mtimes plus preservation of their captured atimes',
    journal_tool=str(tool.relative_to(ROOT)),journal_tool_sha256=sha(tool),journal_sha256=sha(journal),authorization_sha256=sha(HERE/'addendum_003_nine_regular_mtimes.json'),files=results))
  csv_new(folder/'reversal_plan.csv',[dict(path=r['path'],original_atime_ns=r['before']['atime_ns'],original_mtime_ns=r['before']['mtime_ns'],restored_atime_ns=r['after']['atime_ns'],restored_mtime_ns=r['after']['mtime_ns'],sha256=r['sha256'],size=r['before']['size'],mode=r['before']['mode']) for r in results])
  print(json.dumps(dict(status='PASS',restored_and_verified=9,atime_preserved=True,content_writes=0)),flush=True)

if __name__=='__main__':main()
