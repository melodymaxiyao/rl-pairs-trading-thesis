from pathlib import Path
import os,sys,stat,json,hashlib,csv,io,gzip
P=Path('/Users/maxiyao/single_stock_runs/spy_figure_revision_v1');DOC=P/'unused';INPUTS={};UNAVAILABLE={}
def check(p,required=True):
 p=Path(p)
 try:s=p.lstat()
 except FileNotFoundError:s=None
 why='missing' if s is None else 'dataless' if s.st_flags&0x40000000 else 'symlink' if stat.S_ISLNK(s.st_mode) else ''
 if why:
  UNAVAILABLE[str(p)]=why
  if required:raise RuntimeError((why,str(p)))
  return False
 return True
def raw(p):
 p=Path(p);check(p);b=p.read_bytes();v={'sha256':hashlib.sha256(b).hexdigest(),'size':len(b)}
 if not p.is_relative_to(P) or p.is_relative_to(P/'input'):
  if str(p) in INPUTS:assert INPUTS[str(p)]==v,('input changed',p)
  INPUTS[str(p)]=v
 return b
def text(p):return raw(p).decode('utf-8-sig')
def js(p):return json.loads(raw(p))
def rows(p):
 b=raw(p);b=gzip.decompress(b) if str(p).endswith('.gz') else b
 return list(csv.DictReader(io.StringIO(b.decode('utf-8-sig'))))
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvout(p,rr,cols=None):
 cols=cols or list(dict.fromkeys(k for r in rr for k in r))
 with Path(p).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rr)
def guard():
 def hook(e,a):
  if e.startswith('socket.'):raise PermissionError('No network')
  if e=='open':
   p,m,f=a
   if not isinstance(p,(str,bytes,os.PathLike)):return
   p=Path(os.fsdecode(p)).absolute();wr=(isinstance(m,str) and any(c in m for c in 'wax+'))or(isinstance(f,int)and f&(os.O_CREAT|os.O_WRONLY|os.O_RDWR|os.O_TRUNC))
   if wr and p!=Path('/dev/null') and (not p.resolve().is_relative_to(P) or p.resolve().is_relative_to(P/'input')):raise PermissionError(('readonly source/input',p))
   if not wr:
    try:s=p.lstat()
    except FileNotFoundError:return
    if s.st_flags&0x40000000:raise PermissionError(('dataless never read',p))
  if e in ['os.mkdir','os.remove','os.rmdir','os.rename']:
   p=Path(a[0]).absolute()
   if e=='os.mkdir' and p.is_dir():return
   if not p.resolve().is_relative_to(P) or p.resolve().is_relative_to(P/'input'):raise PermissionError(('external mutation',p))
 sys.addaudithook(hook)
