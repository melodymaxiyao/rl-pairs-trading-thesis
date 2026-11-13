"""Preserve the nine authorized original atimes after historical hash reads."""
from common import *
from protection import locks
from restore_authorized_regular_mtimes import checked,metadata,repair_guard
import importlib.util

def main():
 authorization=read(HERE/'addendum_003_nine_regular_mtimes.json');paths=set(authorization['paths'])
 repair=read(HERE/'audit/regular_mtime_repair/result.json');assert repair['status']=='PASS'
 assert {r['path'] for r in repair['files']}==paths and len(paths)==9
 repair_guard(paths)
 with locks():
  tool=PRUNE/'prune_engine.py';assert sha(tool)==repair['journal_tool_sha256']
  s=importlib.util.spec_from_file_location('_atime_journal',tool);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
  label=str(time.time_ns());folder=HERE/'audit/regular_mtime_repair';journal=folder/f'atimes_after_verification_{label}.jsonl';results=[]
  for entry in repair['files']:
   p=ROOT/entry['path'];assert not any(q.is_symlink() for q in [p,*p.parents] if q.is_relative_to(ROOT))
   row=dict(path=entry['path'],size=entry['before']['size'],mode=entry['before']['mode'],sha256=entry['sha256'])
   before,digest=checked(p,row);assert before.st_mtime_ns==entry['after']['mtime_ns']
   target=entry['before']['atime_ns']
   m.append(journal,dict(event='preserve_original_atime_intent',time_ns=time.time_ns(),path=entry['path'],before=metadata(before),original_atime_ns=target,unchanged_mtime_ns=before.st_mtime_ns,verified_sha256=digest))
   os.utime(p,ns=(target,before.st_mtime_ns),follow_symlinks=False)
   final=p.lstat();assert metadata(final)==dict(metadata(before),atime_ns=target)
   result=dict(path=entry['path'],before=metadata(before),after=metadata(final),sha256=digest);results.append(result)
   m.append(journal,dict(event='original_atime_preserved',time_ns=time.time_ns(),**result))
  write(folder/f'atimes_after_verification_{label}.json',dict(status='PASS',paths=9,original_atimes_preserved=True,mtimes_unchanged=True,source_content_writes=0,files=results,journal=str(journal.relative_to(HERE)),journal_sha256=sha(journal)))
  print(json.dumps(dict(status='PASS',original_atimes_preserved=9,mtimes_unchanged=True,source_content_writes=0)),flush=True)

if __name__=='__main__':main()
