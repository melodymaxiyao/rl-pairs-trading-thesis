"""Run the sealed native source verifier with only approved location adaptation."""
import sys,json,hashlib,importlib.util
from pathlib import Path
OWN=Path(__file__).resolve().parents[1];SOURCE=OWN.parent/'ou_realstrength_v1'
sys.path.insert(0,str(SOURCE/'scripts'))
import common,integrity
def approved_location():
 H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 a=json.loads((OWN/'audit/location_authorization.json').read_text());old=json.loads((SOURCE/'audit/location_authorization.json').read_text());assert a['before_sha256']==old['after_sha256']
 before=json.loads((OWN/'audit/run_locations.before.json').read_text());after=json.loads((OWN/'audit/run_locations.approved_after.json').read_text());copy=json.loads(json.dumps(after));assert copy['runs'].pop('ou_qfix_v1')==a['addition'] and copy==before
 assert H(OWN/'audit/run_locations.before.json')==a['before_sha256'] and H(OWN/'audit/run_locations.approved_after.json')==a['after_sha256']
 assert H(Path('/Users/maxiyao/Desktop/FYP/experiments/15_pair_trading/RUN_LOCATIONS.json'))==a['after_sha256']
integrity.location=approved_location
spec=importlib.util.spec_from_file_location('_source_readonly_guard',SOURCE/'scripts/check_inputs.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);mod.readonly_guard()
spec=importlib.util.spec_from_file_location('_sealed_synthetic_verifier',SOURCE/'scripts/verify.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);mod.verify()
print(json.dumps(dict(approved_RUN_LOCATIONS_only=True,source_writes=0,numerical_source_changes=0)))
