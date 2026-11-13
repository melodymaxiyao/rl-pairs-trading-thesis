"""Read-only native upstream verifiers with exact authorized bookkeeping chain."""
import sys,json,hashlib,importlib.util,os
from pathlib import Path
OWN=Path(__file__).resolve().parents[1];BASE=OWN.parent;INPUT=Path('/Users/maxiyao/Desktop/FYP')
def H(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def approved_location():
 for name in ['rf_cost_accounting_v1','ou_realstrength_v1','ou_qfix_v1','coint_residual_qlearning_v1']:
  r=BASE/name;after_name='run_locations.after.json' if name=='rf_cost_accounting_v1' else 'run_locations.approved_after.json';a=json.loads((r/'audit/location_authorization.json').read_text());before=json.loads((r/'audit/run_locations.before.json').read_text());after=json.loads((r/'audit'/after_name).read_text());copy=json.loads(json.dumps(after));assert copy['runs'].pop(name)==a['addition'] and copy==before
  assert H(r/'audit/run_locations.before.json')==a['before_sha256'] and H(r/'audit'/after_name)==a['after_sha256']
  if name!='rf_cost_accounting_v1':assert a['before_sha256']==previous
  previous=a['after_sha256']
 assert H(INPUT/'experiments/15_pair_trading/RUN_LOCATIONS.json')==previous

def readonly_guard():
 def hook(event,args):
  if event=='open':
   p,mode,flags=args
   if isinstance(p,(str,bytes,os.PathLike)):
    q=Path(os.fsdecode(p))
    try:assert not q.lstat().st_flags&0x40000000,('No dataless content',str(q))
    except FileNotFoundError:pass
   if isinstance(mode,str) and any(c in mode for c in 'wax+') or isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):raise PermissionError('Native verifier is read-only')
  if event in ['os.rename','os.remove','os.mkdir','os.rmdir','os.chmod','os.chflags','os.utime','os.link','os.symlink','socket.connect','socket.getaddrinfo']:raise PermissionError('Read-only native verifier')
 sys.addaudithook(hook)
def main(name):
 if name=='ou':
  sys.path.insert(0,str(OWN/'scripts'));import verify_trimmed_ou,prune_extension
  authority=prune_extension.authenticate();verify_trimmed_ou.authenticate=lambda:authority
  sys.modules.pop('common',None)
  return verify_trimmed_ou.main(3)
 root=BASE/name;sys.path.insert(0,str(root/'scripts'))
 import common
 if name!='residual_ou_real_v1':
  import integrity
  integrity.location=approved_location
 readonly_guard()
 if name=='residual_ou_real_v1':p=root/'verify_residual_ou.py'
 else:p=root/'scripts/verify.py'
 spec=importlib.util.spec_from_file_location('_native_upstream_verifier',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 if hasattr(m,'location'):m.location=approved_location
 result=m.verify()
 if result is not None:print(json.dumps(result))
 print(json.dumps(dict(upstream=name,approved_bookkeeping_adapter=True,source_writes=0)))
if __name__=='__main__':main(sys.argv[1])
