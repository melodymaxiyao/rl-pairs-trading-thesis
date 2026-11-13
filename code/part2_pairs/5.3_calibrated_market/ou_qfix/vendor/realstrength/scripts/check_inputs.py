"""Read-only native verifiers, including exactly authorized location bookkeeping."""
import sys
from pathlib import Path
OWN=Path(__file__).resolve().parents[1]

def readonly_guard():
 import os
 def hook(event,args):
  if event in ['socket.connect','socket.getaddrinfo']:raise PermissionError('No network')
  if event=='open':
   p,mode,flags=args
   if isinstance(p,(str,bytes,os.PathLike)):
    q=Path(os.fsdecode(p))
    if q.exists():assert not q.lstat().st_flags&0x40000000,('No dataless read',str(q))
   if isinstance(mode,str) and any(x in mode for x in 'wax+') or isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):raise PermissionError('Read-only verifier')
  if event in ['os.rename','os.remove','os.rmdir','os.mkdir','os.utime','os.chmod','os.chflags','os.symlink','os.link']:raise PermissionError('Read-only verifier')
 sys.addaudithook(hook)

def main(which):
 import json,hashlib,importlib.util
 r=Path('/Users/maxiyao/FYP_runs')
 if which in ['ou','residual']:
  import subprocess
  name='ou_dp_qlearning_v1' if which=='ou' else 'residual_ou_real_v1';script='verify_ou_dp.py' if which=='ou' else 'verify_residual_ou.py';p=subprocess.run(['/Users/maxiyao/Desktop/FYP/.venv/bin/python','-I','-B',str(r/name/script)],capture_output=True,text=True);print(p.stdout,end='');print(p.stderr,end='',file=sys.stderr);assert p.returncode==0;return
 root=r/'rf_cost_accounting_v1';sys.path.insert(0,str(root/'scripts'));readonly_guard();import common as source_common,integrity
 def approved_location():
  H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();a=json.loads((OWN/'audit/location_authorization.json').read_text());old=json.loads((root/'audit/location_authorization.json').read_text());assert a['before_sha256']==old['after_sha256'];before=json.loads((OWN/'audit/run_locations.before.json').read_text());after=json.loads((OWN/'audit/run_locations.approved_after.json').read_text());copy=json.loads(json.dumps(after));entry=copy['runs'].pop('ou_realstrength_v1');assert copy==before and entry==a['addition'];assert H(Path('/Users/maxiyao/Desktop/FYP/experiments/15_pair_trading/RUN_LOCATIONS.json'))==a['after_sha256'];assert H(OWN/'audit/run_locations.before.json')==a['before_sha256']
 integrity.location=approved_location
 spec=importlib.util.spec_from_file_location('_readonly_sealed_cost_verifier',root/'scripts/verify.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.verify();print(json.dumps(dict(approved_bookkeeping_adapter=True,numerical_source_changes=0,source_writes=0)))
if __name__=='__main__':main(sys.argv[1])
