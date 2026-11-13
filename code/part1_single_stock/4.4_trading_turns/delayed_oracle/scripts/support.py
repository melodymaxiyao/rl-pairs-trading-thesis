import os,sys,stat,hashlib,json,gzip,io,shutil
from pathlib import Path
O=Path(__file__).resolve().parents[1];FYP=Path('/Users/maxiyao/Desktop/FYP')
INPUTS={};MISSING={}
def readable(p,required=True):
 p=Path(p)
 s=None
 try:s=p.lstat()
 except FileNotFoundError:pass
 if s is None or s.st_flags&0x40000000 or stat.S_ISLNK(s.st_mode):
  reason='missing' if s is None else 'dataless' if s.st_flags&0x40000000 else 'symlink'
  MISSING[str(p)]=reason
  if required:raise RuntimeError((reason,str(p)))
  return False
 return True
def raw(p):
 p=Path(p);readable(p);b=p.read_bytes();h=hashlib.sha256(b).hexdigest()
 if not p.is_relative_to(O):
  rec={'sha256':h,'size':len(b)}
  if str(p) in INPUTS:assert INPUTS[str(p)]==rec,('input changed',p)
  INPUTS[str(p)]=rec
 return b
def js(p):return json.loads(raw(p))
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(raw(p)).hexdigest()
def resources():
 assert shutil.disk_usage(O).free>=4*2**30,'free space below4GiB'
 own=sum(p.stat().st_size for p in O.rglob('*') if p.is_file())
 second=Path('/Users/maxiyao/FYP_runs/closing_checks_v1')
 if second.exists():own+=sum(p.stat().st_size for p in second.rglob('*') if p.is_file())
 assert own<2*2**30,'combined new output budget2GiB reached'
def guard():
 def hook(event,args):
  if event.startswith('socket.'):raise PermissionError('No network')
  if event=='open':
   p,mode,flags=args
   if not isinstance(p,(str,bytes,os.PathLike)):return
   pp=Path(os.fsdecode(p)).absolute()
   writing=(isinstance(mode,str) and any(x in mode for x in 'wax+')) or (isinstance(flags,int) and flags&(os.O_CREAT|os.O_RDWR|os.O_WRONLY|os.O_TRUNC))
   if writing and pp!=Path('/dev/null') and not pp.resolve().is_relative_to(O):raise PermissionError(('external write',str(pp)))
   if not writing:
    try:s=pp.lstat()
    except FileNotFoundError:return
    if s.st_flags&0x40000000:raise PermissionError(('dataless input not read',str(pp)))
  if event in ['os.remove','os.mkdir','os.rmdir']:
   p=Path(args[0]).absolute()
   if event=='os.mkdir' and p.is_dir():return  # existing-directory mkdir is a loader no-op
   if not p.resolve().is_relative_to(O):raise PermissionError(('external mutation',args))
 sys.addaudithook(hook)
def snapshot():
 found=[];skipped=[];todo=[FYP]
 ignored=FYP/'experiments/15_pair_trading/trend_anchor_td_tuned_v1/audit/repairs/protected_diff_v3/finder_snapshot_1792731898004837000 2.json'
 while todo:
  p=todo.pop()
  if p==ignored:skipped.append([str(p),'iCloud conflict copy; non-sealed; ignored; never read']);continue
  s=p.lstat()
  if s.st_flags&0x40000000:skipped.append([str(p),'dataless; not traversed/read']);continue
  if stat.S_ISDIR(s.st_mode):todo.extend(Path(x.path) for x in os.scandir(p))
  else:found.append(str(p.relative_to(FYP)))
 return {'files':sorted(found),'skipped':skipped}
