from pathlib import Path
import os,sys,stat,hashlib,json,gzip,csv,io,shutil
O=Path('/Users/maxiyao/FYP_runs/closing_checks_v1')
P4=O/'part4';P5=Path('/Users/maxiyao/single_stock_runs/lagged_oracle_v1/part5');P6=O/'part6'
ROOTS=[P4,P5,P6];FYP=Path('/Users/maxiyao/Desktop/FYP');REST=Path('/Users/maxiyao/single_stock_runs/inputs_restored_v1')
READS={};UNAVAILABLE={}
def check(p,required=True):
 p=Path(p)
 try:s=p.lstat()
 except FileNotFoundError:s=None
 why='missing' if s is None else 'dataless' if s.st_flags&0x40000000 else 'symlink' if stat.S_ISLNK(s.st_mode) else ''
 if why:
  UNAVAILABLE[str(p)]=why
  if required:raise RuntimeError((str(p),why))
  return False
 return True
def raw(p):
 p=Path(p);check(p);b=p.read_bytes();rec={'sha256':hashlib.sha256(b).hexdigest(),'size':len(b)}
 if str(p) in READS and not any(p.is_relative_to(x) for x in ROOTS):assert READS[str(p)]==rec,('input changed',p)
 READS[str(p)]=rec;return b
def text(p):return raw(p).decode('utf-8-sig')
def js(p):return json.loads(raw(p))
def rows(p):
 b=raw(p);b=gzip.decompress(b) if str(p).endswith('.gz') else b
 return list(csv.DictReader(io.StringIO(b.decode('utf-8-sig'))))
def dump(p,obj):Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvout(p,records,cols=None):
 cols=cols or list(dict.fromkeys(k for r in records for k in r))
 with Path(p).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(records)
def sha(p):return hashlib.sha256(raw(p)).hexdigest()
def ref(p,i):return str(p)+':'+str(i)
def guard():
 def hook(event,args):
  if event.startswith('socket.'):raise PermissionError('No network')
  if event=='open':
   p,mode,flags=args
   if not isinstance(p,(str,bytes,os.PathLike)):return
   p=Path(os.fsdecode(p)).absolute()
   wr=(isinstance(mode,str) and any(x in mode for x in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
   if wr and p!=Path('/dev/null') and not any(p.resolve().is_relative_to(r) for r in ROOTS):raise PermissionError(('external write',p))
   if not wr:
    try:s=p.lstat()
    except FileNotFoundError:return
    if s.st_flags&0x40000000:raise PermissionError(('dataless not read',p))
  if event in ['os.mkdir','os.remove','os.rmdir']:
   p=Path(args[0]).absolute()
   if event=='os.mkdir' and p.is_dir():return
   if not any(p.resolve().is_relative_to(r) for r in ROOTS):raise PermissionError(('external mutation',p))
 sys.addaudithook(hook)
def resources():
 assert shutil.disk_usage(O).free>=4*2**30
 other=Path('/Users/maxiyao/single_stock_runs/lagged_oracle_v1')
 size=sum(p.stat().st_size for root in [O,other] for p in root.rglob('*') if p.is_file())
 assert size<2*2**30
def inventory():
 # Inventory names only. Dataless directory metadata is checked before traversal; no file content opened.
 todo=[FYP];found=[];skipped=[]
 ignored=FYP/'experiments/15_pair_trading/trend_anchor_td_tuned_v1/audit/repairs/protected_diff_v3/finder_snapshot_1792731898004837000 2.json'
 while todo:
  d=todo.pop()
  if d.lstat().st_flags&0x40000000:skipped.append([str(d),'dataless directory; not traversed']);continue
  for e in os.scandir(d):
   p=Path(e.path)
   if p==ignored:skipped.append([str(p),'iCloud conflict copy; non-sealed; ignored; never read']);continue
   if e.is_dir(follow_symlinks=False):todo.append(p)
   else:found.append(str(p.relative_to(FYP)))
 return {'files':sorted(found),'skipped':skipped}
def ledger(directory,phase,extra=None):
 rec={k:v for k,v in READS.items() if not any(Path(k).is_relative_to(r) for r in ROOTS)}
 if extra:
  for k,v in extra.items():
   if not any(Path(k).is_relative_to(r) for r in ROOTS):
    if k in rec:assert rec[k]==v
    rec[k]=v
 dump(Path(directory)/('inputs_'+phase+'.json'),rec)
 return rec
def verify(rec):
 for p,v in rec.items():assert sha(p)==v['sha256'],('changed',p)
