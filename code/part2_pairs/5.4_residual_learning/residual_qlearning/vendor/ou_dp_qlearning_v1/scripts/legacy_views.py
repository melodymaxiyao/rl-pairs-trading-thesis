"""Read-only legacy views for precisely the standing symlink/30-path rules."""
from common import *
from protection import PT,COMPACT,conditionals,registry_rows,target_check
import csv,importlib
class StatView:
 def __init__(self,value,mtime):self.value=value;self.mtime=mtime
 def __getattr__(self,k):return self.mtime if k=='st_mtime_ns' else self.mtime/1e9 if k=='st_mtime' else getattr(self.value,k)
 def __len__(self):return len(self.value)
 def __getitem__(self,k):return list(self)[k]
 def __iter__(self):
  values=list(self.value);values[8]=self.mtime/1e9;return iter(values)
@contextmanager
def canonical_mtimes():
 with gzip.open(COMPACT/'inventory.csv.gz','rt',newline='') as f:expected={r['path']:r for r in csv.DictReader(f)}
 registered=registry_rows();raw_stat,raw_lstat=os.stat,os.lstat;busy=False;verified=set()
 from provenance import unavailable
 registered.update(unavailable())
 def view(path,value,follow,dir_fd):
  nonlocal busy
  if follow or busy or not stat.S_ISLNK(value.st_mode) or not isinstance(path,(str,bytes,os.PathLike)) or dir_fd is not None:return value
  p=Path(os.path.abspath(os.fsdecode(path)))
  if not p.is_relative_to(INPUT):return value
  rel=str(p.relative_to(INPUT));r=expected.get(rel)
  if r is None or value.st_mtime_ns==int(r['mtime_ns']):return value
  if rel not in verified:
   busy=True
   try:
    assert os.readlink(p)==r['target'] and stat.S_IMODE(value.st_mode)==int(r['mode']);target_check(p,expected,registered)
   finally:busy=False
   print('SYMLINK_TIMESTAMP_ACCEPTED '+json.dumps(dict(path=rel,frozen_mtime_ns=int(r['mtime_ns']),observed_mtime_ns=value.st_mtime_ns)),flush=True);verified.add(rel)
  return StatView(value,int(r['mtime_ns']))
 os.stat=lambda path,*a,**kw:view(path,raw_stat(path,*a,**kw),kw.get('follow_symlinks',True),kw.get('dir_fd'))
 os.lstat=lambda path,*a,**kw:view(path,raw_lstat(path,*a,**kw),False,kw.get('dir_fd'))
 try:yield
 finally:os.stat,os.lstat=raw_stat,raw_lstat
@contextmanager
def conditional_extension():
 # The original 28-path policy checks remain unchanged; its baseline is extended by
 # only the two authenticated non-live .gitignore paths, as previously authorized.
 b=conditionals();sys.path.insert(0,str(COMPACT));p=importlib.import_module('addendum_2_policy');old=p.baseline
 def baseline():
  original=old();assert set(original)<=set(b) and len(original)==28;return b
 p.baseline=baseline
 os.environ['TMPDIR']=str(RUN/'runtime/tmp');os.environ['SQLITE_TMPDIR']=str(RUN/'runtime/tmp')
 try:yield
 finally:p.baseline=old
