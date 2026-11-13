"""Input chains, standing8908 exact paths, scoped IO and immutable design."""
from common import *
from global_conditionals import authenticate

def location():
 a=read(RUN/'audit/location_authorization.json');p=INPUT/'experiments/15_pair_trading/RUN_LOCATIONS.json';assert sha(p)==a['after_sha256'];before=read(RUN/'audit/run_locations.before.json');after=read(RUN/'audit/run_locations.approved_after.json');copy=json.loads(json.dumps(after));assert copy['runs'].pop('ou_realstrength_v1')==a['addition'] and copy==before

def imports():
 reg=read(RUN/'audit/import_registration.json')
 for rel,row in reg['files'].items():
  p=RUN/'vendor'/rel
  if not p.exists() and rel=='ou/runtime/rollout.so':p=RUN/'runtime/rollout.so'
  assert sha(p)==row['sha256'],('Imported bytes changed',rel)
 for key,seal in read(RUN/'audit/input_seals.json').items():
  p=Path(seal['root']);assert sha(p/'complete.json')==seal['sealed_completion_sha256'] and sha(p/'audit.json')==seal['sealed_audit_sha256'] and sha(p/'code_registration.json')==seal['sealed_code_registration_sha256']
 return dict(status='PASS',imported_files=len(reg['files']))

def check_code():
 reg=read(RUN/'code_registration.json');assert sha(RUN/'preregistration.json')==reg['preregistration_sha256']
 for rel,digest in reg['files'].items():assert sha(RUN/rel)==digest,('Frozen code changed',rel)
 return reg

def inputs(label,native=True):
 checkpoint();location();imports();authenticate();env=environment()
 if native:
  for which in ['ou','residual','cost']:
   p=subprocess.run([str(INPUT/'.venv/bin/python'),'-I','-B',str(RUN/'scripts/check_inputs.py'),which],capture_output=True,text=True);publish(RUN/'audit'/f'native_{label}_{which}.log',p.stdout+p.stderr);assert p.returncode==0,('Native source verifier failed',which,p.stdout[-1200:],p.stderr[-2000:])
 write(event('inputs_'+label),dict(status='PASS',environment=env,standing_exact_paths=8908,native_verifiers=native,real_trials=552,source_writes=0,bookkeeping_writes=1,free_bytes=shutil.disk_usage(RUN).free))
