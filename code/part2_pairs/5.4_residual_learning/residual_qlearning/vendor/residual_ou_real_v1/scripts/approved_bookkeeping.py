"""One authenticated standing-rule RUN_LOCATIONS transition; no snapshot rewrite."""
import json,hashlib
from pathlib import Path
REL='experiments/15_pair_trading/RUN_LOCATIONS.json'
PINS={'authorization.json': '1f0aea654cb405351ceb2cde68994d5c365e53822798d083610c86544207ab59', 'run_locations_before.json': 'f1dcab98e1bc0472428d404a31056f9d2a716087359aa2951aab0f692edf7608', 'run_locations_after.json': '34ff2fbe4946f49d68ed8696f4f2858fc48348b55bea13237521d0cd4c5e535f'}
def compare(old,new,authorization,before,after):
 if old.get('path')!=REL or new.get('path')!=REL:return False
 if old.get('target') is not None or new.get('target') is not None:return False
 if new!=authorization['approved_after_record']:return False
 if old.get('sha256')!=authorization['registry_before_sha256'] or new.get('sha256')!=authorization['registry_after_sha256']:return False
 allowed={'sha256','size','mtime_ns'}
 if {k:v for k,v in old.items() if k not in allowed}!={k:v for k,v in new.items() if k not in allowed}:return False
 reduced=json.loads(json.dumps(after));entry=reduced.get('runs',{}).pop('fyp_prune_v2',None)
 if entry is None or reduced!=before:return False
 return entry['run_directory']=='/Users/maxiyao/FYP_runs/fyp_prune_v2' and entry['input_root']=='/Users/maxiyao/Desktop/FYP' and entry['production_started'] is False
def approve(old,new):
 if old.get('path')!=REL:return None
 directory=Path(__file__).resolve().parents[1]/'audit/approved_bookkeeping_v1';documents={}
 for name,digest in PINS.items():
  raw=(directory/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==digest,('Bookkeeping authorization changed',name);documents[name]=json.loads(raw)
 a=documents['authorization.json']
 assert compare(old,new,a,documents['run_locations_before.json'],documents['run_locations_after.json']),('Unapproved RUN_LOCATIONS transition',old,new)
 return dict(status='PASS_EXACT_APPROVED_STANDING_BOOKKEEPING_WRITE',path=REL,before_sha256=a['registry_before_sha256'],after_sha256=a['registry_after_sha256'],authorization_pins=PINS,accepted_changes=['content:size and SHA256','mtime_ns'],mode_inode_flags_target_unchanged=True,protected_before_snapshot_rewritten=False)
