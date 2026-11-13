from pathlib import Path
import os,sys,stat,hashlib,json,csv,io,gzip,shutil
P=Path(__file__).resolve().parents[1];FYP=Path('/Users/maxiyao/Desktop/FYP');READS={};MISSING={}
def check(p,required=True):
 p=Path(p)
 try:s=p.lstat()
 except FileNotFoundError:s=None
 why='missing' if s is None else 'dataless' if s.st_flags&0x40000000 else 'symlink' if stat.S_ISLNK(s.st_mode) else ''
 if why:
  MISSING[str(p)]=why
  if required:raise RuntimeError((why,str(p)))
  return False
 return True
def raw(p):
 p=Path(p);check(p);b=p.read_bytes();r={'sha256':hashlib.sha256(b).hexdigest(),'size':len(b)}
 if not p.is_relative_to(P):
  if str(p) in READS:assert READS[str(p)]==r,('input changed',p)
  READS[str(p)]=r
 return b
def text(p):return raw(p).decode('utf-8-sig')
def js(p):return json.loads(raw(p))
def sha(p):return hashlib.sha256(raw(p)).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def rows(p):
 b=raw(p);b=gzip.decompress(b) if str(p).endswith('.gz') else b
 return list(csv.DictReader(io.StringIO(b.decode('utf-8-sig'))))
def csvout(p,rows,cols=None):
 cols=cols or list(dict.fromkeys(k for r in rows for k in r))
 with Path(p).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
def resources():
 assert shutil.disk_usage(P).free>=2*2**30,'free disk <2GiB'
 assert sum(p.stat().st_size for p in P.rglob('*') if p.is_file())<2**30,'output budget1GiB'
def guard():
 def hook(e,args):
  if e.startswith('socket.'):raise PermissionError('network disabled')
  if e=='open':
   p,m,f=args
   if not isinstance(p,(str,bytes,os.PathLike)):return
   p=Path(os.fsdecode(p)).absolute();wr=(isinstance(m,str) and any(x in m for x in 'wax+'))or(isinstance(f,int)and f&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
   if wr and p!=Path('/dev/null') and not p.resolve().is_relative_to(P):raise PermissionError(('external write',p))
   if not wr:
    try:s=p.lstat()
    except FileNotFoundError:return
    if s.st_flags&0x40000000:raise PermissionError(('dataless',p))
  if e in ['os.mkdir','os.remove','os.rmdir','os.rename']:
   p=Path(args[0]).absolute()
   if e=='os.mkdir' and p.is_dir():return
   if not p.resolve().is_relative_to(P):raise PermissionError(('external mutation',p))
 sys.addaudithook(hook)
def verify(rec):
 for p,r in rec.items():assert sha(p)==r['sha256'],('changed',p)
