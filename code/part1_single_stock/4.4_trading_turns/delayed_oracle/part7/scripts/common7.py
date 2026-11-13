import os,sys,stat,hashlib,json,gzip,csv,io,shutil
from pathlib import Path
P=Path(__file__).resolve().parents[1];L=P.parent;FYP=Path('/Users/maxiyao/Desktop/FYP');R=Path('/Users/maxiyao/single_stock_runs/inputs_restored_v1')
INPUTS={}
def check(p):
 p=Path(p);s=p.lstat();assert not s.st_flags&0x40000000,('dataless',p);assert not stat.S_ISLNK(s.st_mode),('symlink',p);return p
def raw(p):
 p=check(p);b=p.read_bytes();v={'sha256':hashlib.sha256(b).hexdigest(),'size':len(b)}
 if not p.is_relative_to(P):
  if str(p) in INPUTS:assert INPUTS[str(p)]==v,('changed',p)
  INPUTS[str(p)]=v
 return b
def js(p):return json.loads(raw(p))
def sha(p):return hashlib.sha256(raw(p)).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def frame(p):
 import pandas as pd
 b=raw(p);b=gzip.decompress(b) if str(p).endswith('.gz') else b
 return pd.read_csv(io.BytesIO(b),keep_default_na=False,float_precision='round_trip')
def guard():
 def hook(event,args):
  if event.startswith('socket.'):raise PermissionError('network blocked')
  if event=='open':
   p,mode,flags=args
   if not isinstance(p,(str,bytes,os.PathLike)):return
   p=Path(os.fsdecode(p)).absolute();wr=(isinstance(mode,str) and any(x in mode for x in 'wax+')) or (isinstance(flags,int) and flags&(os.O_CREAT|os.O_WRONLY|os.O_RDWR|os.O_TRUNC))
   if wr and p!=Path('/dev/null') and not p.resolve().is_relative_to(P):raise PermissionError(('external write',p))
   if not wr:
    try:s=p.lstat()
    except FileNotFoundError:return
    if s.st_flags&0x40000000:raise PermissionError(('dataless read',p))
  if event in ['os.mkdir','os.remove','os.rmdir','os.rename']:
   q=Path(args[0]).absolute()
   if event=='os.mkdir' and q.is_dir():return
   if not q.resolve().is_relative_to(P):raise PermissionError(('external mutation',q))
 sys.addaudithook(hook)
def resources():
 assert shutil.disk_usage(P).free>=4*2**30,'free space <4GiB'
 size=sum(p.stat().st_size for root in [L,Path('/Users/maxiyao/FYP_runs/closing_checks_v1')] for p in root.rglob('*') if p.is_file())
 assert size<2*2**30,'combined output budget 2GiB'
def verify(rec):
 for p,v in rec.items():assert sha(p)==v['sha256'],('source changed',p)
