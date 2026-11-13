"""Scope, reproducible code freeze, upstream verifiers and fresh historical gates."""
from common import *
from check_inputs import approved_location
from global_conditionals import authenticate as standing

def code():
 reg=read(RUN/'code_registration.json');assert sha(RUN/'preregistration.json')==reg['preregistration_sha256']
 for rel,h in reg['files'].items():assert sha(RUN/rel)==h,('Frozen preparation artifact changed',rel)
 return reg

def input_gate(label,native=True):
 checkpoint();approved_location();standing();env=environment()
 for name in ['residual_ou_real_v1','rf_cost_accounting_v1','ou_realstrength_v1','ou_qfix_v1','ou_dp_qlearning_v1']:seal(name)
 logs=[]
 if native:
  for name in ['ou','residual_ou_real_v1','rf_cost_accounting_v1','ou_realstrength_v1','ou_qfix_v1']:
   p=subprocess.run([str(INPUT/'.venv/bin/python'),'-I','-B',str(RUN/'scripts/check_inputs.py'),name],capture_output=True,text=True);path=event('upstream_'+name+'_'+label).with_suffix('.log');publish(path,p.stdout+p.stderr);logs.append(dict(upstream=name,path=str(path.relative_to(RUN)),sha256=sha(path)));assert p.returncode==0,('Native upstream verification failed',name,p.stdout[-1000:],p.stderr[-3000:])
 write(event('inputs_'+label),dict(status='PASS',environment=env,logs=logs,standing_exact_paths=8908,source_writes=0));return logs

def historical_gate(label):
 command=[str(INPUT/'.venv/bin/python'),'-I','-B',str(RUN/'scripts/historical_entry.py'),label];p=subprocess.run(command);assert p.returncode==0,('Fresh historical gate failed',p.returncode,label);return read(RUN/f'audit/historical_{label}.json')

def freeze():
 assert not list((RUN/'outputs/cells').glob('*/complete.json')) if (RUN/'outputs/cells').exists() else True
 files={}
 for folder in ['scripts','tests','vendor']:
  for p in sorted((RUN/folder).rglob('*')):
   if p.is_file():assert not p.is_symlink();files[str(p.relative_to(RUN))]=sha(p)
 for f in ['runtime/frozen_q.so','runtime/frozen_q.c','runtime/frozen_ou.c','RUN_ALL.zsh','VERIFY.zsh','README.preregistered.md','predictions.csv','audit/source_seals.json','audit/source_copies.json','audit/TAV_code_copies.json','audit/location_authorization.json','audit/user_decisions.json','audit/q_transfer.source.json','audit/request.json','request.md','audit/known_quote_gap.json','audit/episode_padding.json','audit/run_locations.before.json','audit/run_locations.approved_after.json','audit/preregistration_revision_001.json','audit/RF_potential_members.json','audit/kernel_import_proof.json','audit/legacy_AL_fixture.json','audit/residual_reconstruction_fixture.json','audit/RF_reconstruction_exact_fixture.json','audit/tests_preparation_01.log','audit/tests_missing_price_v1.log','audit/missing_price_revision_v1/authorization.json','audit/missing_price_revision_v1/preregistration_change.json','audit/missing_price_revision_v1/valuation_registration.json','audit/missing_price_revision_v1/stock_days.csv','audit/Tier1_launch_authorization.json']:
  files[f]=sha(RUN/f)
 value=dict(status='CODE_AND_DESIGN_FROZEN_BEFORE_REAL_STRATEGY_FITS',preregistration_sha256=sha(RUN/'preregistration.json'),files=files,real_strategy_fits_before_freeze=0,tests_log_sha256=sha(RUN/'audit/tests_missing_price_v1.log'))
 path=RUN/'code_registration.json'
 if path.exists():
  auth=read(RUN/'audit/missing_price_revision_v1/authorization.json');assert sha(path)==auth['before_files']['code_registration.json'] and not (RUN/'audit/real_protocol_started.json').exists()
  value['authorized_prelaunch_revision']=dict(before_sha256=sha(path),authorization_sha256=sha(RUN/'audit/missing_price_revision_v1/authorization.json'));replace(path,value)
 else:write(path,value)
