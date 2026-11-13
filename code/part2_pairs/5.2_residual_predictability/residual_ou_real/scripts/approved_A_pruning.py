"""Exact authenticated A archive registration; never waive another absent/changed path."""
import json,hashlib,gzip,csv,stat
from pathlib import Path
ROOT=Path('/Users/maxiyao/Desktop/FYP')
REL='experiments/15_pair_trading/PRUNE_REGISTRY.json'
DOC_PATH=Path('/Users/maxiyao/FYP_runs/fyp_prune_v2/audit/A_phase2_002/batch_001/resume_authorization.json')
DOC_SHA='dabb9b56bef056279e78b3da11d06e9d5eec64ed5ff12117365be87c6515f51f'
def digest(p):
 p=Path(p);s=p.lstat();assert stat.S_ISREG(s.st_mode) and not s.st_flags&0x40000000
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def meta(p):
 s=p.lstat();return dict(path=REL,size=s.st_size,mode=stat.S_IMODE(s.st_mode),mtime_ns=s.st_mtime_ns,ino=s.st_ino,flags=s.st_flags,target=None,sha256=digest(p))
def compare_control(old,new,doc):
 if new!=doc['source_after']:return False
 a=dict(old);before=dict(doc['source_before'])
 if a.keys()!=before.keys():return False
 delta=a['flags']^before['flags']
 if delta&~0x8040:return False
 a['flags']=before['flags']
 return a==before and a['path']==REL and a['target'] is None

def load():
 assert digest(DOC_PATH)==DOC_SHA,'A prune approval proof changed'
 doc=json.loads(DOC_PATH.read_text());assert doc['group']=='A' and doc['unavailable_members_re_registered']==0
 assert meta(ROOT/REL)==doc['source_after'],'Only the exact approved registry afterimage is authorized'
 before_path=Path(doc['before_copy']);after_path=Path(doc['after_copy'])
 assert digest(before_path)==doc['before_sha256'] and digest(after_path)==doc['after_sha256']
 before=json.loads(before_path.read_text());after=json.loads(after_path.read_text());reduced=json.loads(json.dumps(after));entry=reduced['entries'].pop();assert reduced==before and entry==doc['entry']
 assert digest(Path(entry['manifest']))==entry['manifest_sha256'] and digest(Path(entry['recipe']))==entry['recipe_sha256'] and digest(Path(entry['audit_directory'])/'audit.json')==entry['audit_sha256']
 with gzip.open(entry['manifest'],'rt',newline='') as f:rows=list(csv.DictReader(f))
 names={r['path'] for r in rows};assert len(names)==len(rows)==entry['files']
 assert hashlib.sha256(('\n'.join(sorted(names))+'\n').encode()).hexdigest()==entry['paths_sha256']
 cond=Path('/Users/maxiyao/FYP_runs/fyp_registered_conditionals_v1/REGISTERED_CONDITIONAL_PATHS.json');assert digest(cond)=='8bc237d1f1daca88249d305b7388f576842e125c0c431986aeb9c895de0357e8'
 assert not names&{x['path'] for x in json.loads(cond.read_text())['entries']}
 for receipt in entry['archive_receipts']:
  assert digest(Path(receipt['archive']))==receipt['archive_sha256'] and digest(Path(receipt['manifest']))==receipt['manifest_sha256']
 return doc

def approved_registry_hash(old_pin):
 doc=load();assert old_pin==doc['before_sha256'];return doc['after_sha256']
def expected_registry_record(frozen):
 doc=load();old=doc['source_before']
 if frozen is not None:assert all(frozen[k]==old[k] for k in ['sha256','size','mode','mtime_ns'])
 now=doc['source_after'];return dict(frozen or dict(kind='file',target=''),**{k:now[k] for k in ['sha256','size','mode','mtime_ns']})
def approve_control(old,new):
 doc=load();assert compare_control(old,new,doc),('Unapproved prune-registry transition',old,new)
 return dict(status='PASS_EXACT_APPROVED_ARCHIVE_PRUNE_REGISTRY',group='A',files=doc['entry']['files'],authorization_sha256=DOC_SHA,before_sha256=doc['before_sha256'],after_sha256=doc['after_sha256'],original_protected_snapshot_rewritten=False)
