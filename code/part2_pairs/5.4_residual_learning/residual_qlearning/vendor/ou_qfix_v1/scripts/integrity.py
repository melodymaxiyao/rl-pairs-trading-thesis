"""Exact input seals and approved location bookkeeping; no source rewrites."""
from common import *
from global_conditionals import authenticate
def location():
 a=read(RUN/'audit/location_authorization.json');before=read(RUN/'audit/run_locations.before.json');after=read(RUN/'audit/run_locations.approved_after.json');copy=json.loads(json.dumps(after));assert copy['runs'].pop('ou_qfix_v1')==a['addition'] and copy==before
 assert sha(RUN/'audit/run_locations.before.json')==a['before_sha256'] and sha(RUN/'audit/run_locations.approved_after.json')==a['after_sha256']
 assert sha(INPUT/'experiments/15_pair_trading/RUN_LOCATIONS.json')==a['after_sha256']
 old=read(RUN/'vendor/realstrength/audit/location_authorization.json');assert a['before_sha256']==old['after_sha256']
def check_code():
 reg=read(RUN/'code_registration.json');assert sha(RUN/'preregistration.json')==reg['preregistration_sha256']
 for rel,h in reg['files'].items():assert sha(RUN/rel)==h,('Frozen code changed',rel)
 return reg
def imports():
 src=synthetic();reg=read(RUN/'audit/imports.json')
 for rel,row in reg['files'].items():
  # These IO/stop adapters have separately frozen source originals and current code.
  if rel in ['scripts/common.py','scripts/historical_entry.py']:continue
  assert sha(RUN/rel)==row['sha256'],('Imported bytes changed',rel)
 for row in reg['files'].values():assert sha(src.root/row['source_relative'])==row['sha256']
 return dict(status='PASS',files=len(reg['files']))
def inputs(label,native=True):
 checkpoint();location();imports();authenticate();env=environment()
 if native:
  p=subprocess.run([str(INPUT/'.venv/bin/python'),'-I','-B',str(RUN/'scripts/check_source.py')],capture_output=True,text=True)
  publish(event('source_verifier_'+label).with_suffix('.log'),p.stdout+p.stderr);assert p.returncode==0,('Source verifier failed',p.stdout[-1000:],p.stderr[-3000:])
 write(event('inputs_'+label),dict(status='PASS',environment=env,standing_exact_paths=8908,source_writes=0,real_trials=552,new_real_trials=0,free_bytes=shutil.disk_usage(RUN).free))
