"""Report-only in-place finalization while relocation is paused; no fitting or result replacement."""
import os,sys,json,hashlib,importlib.util,sqlite3,time,types,subprocess,fcntl,shutil
from pathlib import Path
R=Path(__file__).parent;P=Path('/Users/maxiyao/.codex/fyp_relocation_v2/execution_20261028_171000');RELOCATION=P/'resume_20261028_cache_authorized';ROOT=Path('/Users/maxiyao/Desktop/FYP');HERE=ROOT/'experiments/15_pair_trading/real_mechanism_spi_v1';CACHE=P/'resume_20261028_flags_authorized/report_source_cache';LIMITATION='upstream provenance files unavailable locally; computation inputs were verified at run time; 66 groups and 17,482 checkpoints verified'
def digest(p):
 assert not p.lstat().st_flags&0x40000000,('No dataless access',str(p))
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert json.loads((R/'USER_DECISION.json').read_text())['status']=='AUTHORIZED_IN_PLACE_TIER1_FINALIZATION'
assert not ROOT.is_symlink() and not os.path.lexists('/Users/maxiyao/FYP')
assert sys.executable=='/Users/maxiyao/Desktop/FYP/.venv/bin/python'
os.environ.update(RMS_SCOPE='in_place_finalization',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
sys.dont_write_bytecode=True;sys.path.insert(0,str(HERE/'scripts'))
import common as c
import numpy as np,pandas as pd
assert pd.__version__=='3.0.5' and np.__version__=='2.5.2'
assert Path(sys.prefix).samefile(ROOT/'.venv')
reg=c.registration();reconstructed=json.loads((P/'resume_20261028_flags_authorized/report_source_reconstruction.json').read_text())['found'];dependencies={}
def exact_module(rel,name,aliases):
 key='experiments/15_pair_trading/'+rel;path=CACHE/key;expected=reg[key];assert digest(path)==expected==reconstructed[key]['sha256']
 module=types.ModuleType(name);module.__file__=str(path)
 with c.aliases(**aliases):exec(compile(path.read_bytes(),str(ROOT/key),'exec'),module.__dict__)
 dependencies[key]=dict(sha256=expected,size=path.stat().st_size,source='Resident session-log snapshot; byte-for-byte sealed hash verified; original unavailable code path untouched',snapshot=str(path),log=reconstructed[key]['source_log'],log_line=reconstructed[key]['source_line'])
 return module
def frozen_setup():
 if hasattr(frozen_setup,'value'):return frozen_setup.value
 inf=exact_module('trend_anchor_trading_v1/scripts/inference.py','_final_year_blocks',{'aio':c})
 stats=exact_module('td_mechanism_spi_v1/scripts/inference.py','_final_holm',{'common':c})
 methods=exact_module('thesis_tables_v1/scripts/methods.py','_final_methods',{'common':c})
 accounting=exact_module('stage2_selector_v1/scripts/s2inference.py','_final_accounting',{'s2io':c})
 switches=c.spec()['SPI']['switches'];assert switches==['E0','E1','E2','E3','K2','K3','X0','X1','X2','X3']
 frozen_setup.value=types.SimpleNamespace(inference=inf,stats=stats,methods=methods,policies=types.SimpleNamespace(SWITCHES=switches),sources=types.SimpleNamespace(original_inference=lambda:accounting))
 return frozen_setup.value

def log_flags(rel,expected,current,sha,stage):
 import csv
 p=HERE/'flag_only_changes.csv';exists=p.exists()
 with p.open('a',newline='') as h:
  w=csv.writer(h)
  if not exists:w.writerow(['path','stage','before_flags','current_flags','xor','sha256','size','mode','inode','mtime_ns'])
  w.writerow([rel,stage,expected['flags'],current.st_flags,expected['flags']^current.st_flags,sha,current.st_size,current.st_mode&0o7777,current.st_ino,current.st_mtime_ns]);h.flush();os.fsync(h.fileno())

INPUT_BASELINE=sqlite3.connect('file:'+str(RELOCATION/'phase.sqlite3')+'?mode=ro',uri=True)
original_note=c.note
checked_inputs={}
def checked_note(p):
 p=Path(p)
 value=original_note(p)
 if p.is_relative_to(HERE):return value
 rel=str(p.relative_to(ROOT));row=INPUT_BASELINE.execute('SELECT kind,sha,meta FROM entries WHERE path=?',(rel,)).fetchone();assert row and row[0]=='file',('Reporting input lacks authenticated materialized baseline',rel)
 expected=json.loads(row[2]);s=p.lstat();fingerprint=c.PINNED[rel]
 assert fingerprint['sha256']==row[1],('Reporting input hash differs from baseline',rel)
 assert (s.st_size,s.st_mode&0o7777,s.st_ino,s.st_mtime_ns,s.st_dev,s.st_nlink,s.st_uid,s.st_gid)==(expected['size'],expected['mode'],expected['ino'],expected['mtime_ns'],expected['dev'],expected['nlink'],expected['uid'],expected['gid']),('Reporting input metadata changed',rel)
 delta=expected['flags']^s.st_flags;assert not delta&~0x8040,('Unsupported reporting input flags',rel)
 if delta and rel not in checked_inputs:log_flags(rel,expected,s,row[1],'reporting_input')
 checked_inputs[rel]=row[1]
 return value
c.note=checked_note

def check_opened_reporting_inputs():
 for rel,expected in checked_inputs.items():assert digest(ROOT/rel)==expected,('Reporting input changed during finalization',rel)
 c.write(HERE/'audit/in_place_reporting_inputs.json',dict(status='PASS',files=checked_inputs,baseline='authenticated relocation preflight full hashes; original scientific inputs unchanged'))

def verify_checkpoints():
 db=sqlite3.connect('file:'+str(RELOCATION/'phase.sqlite3')+'?mode=ro',uri=True);count=0
 categories=json.loads((P/'checkpoint_baseline.json').read_text())['verified'];details={}
 for category,expected in categories.items():
  ready_files=list((HERE/category).glob('*.ready.json'));assert len(ready_files)==expected
  for ready in ready_files:
   envelope=json.loads(ready.read_text());payload=HERE/envelope['file']
   for path in [ready,payload]:
    rel=str(path.relative_to(ROOT));row=db.execute('SELECT sha,meta FROM entries WHERE path=?',(rel,)).fetchone();assert row,rel;e=json.loads(row[1]);s=path.lstat()
    assert digest(path)==row[0],('Existing checkpoint content changed',rel)
    assert (s.st_size,s.st_mode&0o7777,s.st_ino,s.st_mtime_ns,s.st_dev,s.st_nlink,s.st_uid,s.st_gid)==(e['size'],e['mode'],e['ino'],e['mtime_ns'],e['dev'],e['nlink'],e['uid'],e['gid']),('Existing checkpoint immutable metadata changed',rel)
    delta=s.st_flags^e['flags'];assert not delta&~0x8040,('Unsupported checkpoint flag difference',rel,e['flags'],s.st_flags)
    if delta:log_flags(rel,e,s,digest(path),'production_checkpoint_verification')
   assert digest(payload)==envelope['sha256'];count+=1
  details[category]=len(ready_files)
 assert count==17482
 return dict(status='PASS',groups=details['runtime/folds'],checkpoint_pairs=count,categories=details,content_and_immutable_metadata_exact=True)
def prior_acceptance():
 values=[]
 for p in sorted((HERE/'audit').glob('tests_acceptance_*.ready.json')):
  v=json.loads(p.read_text());assert v['status']=='PASS' and digest(HERE/v['log'])==v['log_sha256'];values.append(dict(receipt=str(p.relative_to(HERE)),sha256=digest(p),value=v))
 assert values,'Missing original passing real acceptance receipt'
 return values
def main():
 assert not (HERE/'complete.json').exists(),'Already finalized; use the read-only verifier'
 lock=(HERE/'pipeline.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 before=verify_checkpoints();old_ledger=c.ledger();acceptance=prior_acceptance();old_registration=c.read(HERE/'audit/code_registration.json')
 for rel,sha in old_registration['files'].items():assert digest(HERE/rel)==sha,('Prepared code changed',rel)
 readme=HERE/'README.md';prefix=readme.read_bytes();assert hashlib.sha256(prefix).hexdigest()==old_registration['README_sha256']
 unavailable=json.loads((RELOCATION/'unavailable_381_registry.json').read_text());conditional=[x for x in unavailable['entries'] if x['primary_class']=='a'];assert len(conditional)==303
 olddb=sqlite3.connect('file:'+str(P/'baseline.sqlite3')+'?mode=ro',uri=True)
 for row in conditional:
  v=olddb.execute('SELECT meta FROM entries WHERE path=?',(row['path'],)).fetchone();assert v;row['sealed_metadata']=json.loads(v[0])
 # The original report functions and all numerical bodies are retained byte-for-byte.
 import real_sources,report
 real_sources.setup=frozen_setup;report.setup=frozen_setup
 frozen_setup()
 c.write(HERE/'audit/report_source_snapshots.json',dict(status='PASS_EXACT_SEALED_BYTES',files=dependencies,original_locations_untouched=True,scientific_runtime_context_reads=False))
 c.guard()
 print('Report-only finalization: 66 groups / 17482 checkpoints verified; zero new fits.',flush=True)
 lines=report.run()
 for h in c.HANDLES.values():h.flush()
 after=verify_checkpoints();check_opened_reporting_inputs();new_ledger=c.ledger()
 for key in ['TD_fits','learning_steps','executor_calls','null_executions','switch_tests']:
  assert new_ledger['total'].get(key,0)==old_ledger['total'].get(key,0),('Finalization performed an unauthorized numerical production operation',key)
 addon=HERE/'addendum_012_in_place_tier1_finalization.json'
 c.write(addon,dict(user_decision='Finalize Tier 1 in place while relocation is paused; no recovery of the 381, no result changes; exact 303 unavailable upstream provenance files receive conditional content handling',limitation=LIMITATION,conditional_paths=conditional,conditional_content_policy='Protect path/size/mode/mtime/flags where visible; hash excluded only while unavailable or dataless. If materialized, sealed SHA-256 and size/mode must match. No new scientific input or result exemption.',user_decision_sha256=digest(R/'USER_DECISION.json'),relocation_status='PAUSED_BY_USER; workspace remains at Desktop/FYP',runtime_input_evidence='Original registered source hashes, opened-input fingerprints, passing original input verifiers and main acceptance receipts retained; no fresh local content verification is claimed for unavailable upstream artifacts.',report_numerical_sources='audit/report_source_snapshots.json',checkpoint_proof=dict(before=before,after=after),acceptance_reused=acceptance,Tier2='NOT RUN',new_fits=0,result_replacements=0))
 suffix=('\n\n## Tier 1 finalized in place; relocation paused\n\n'+LIMITATION+'.\n\n'+'The exact 303-path conditional provenance scope is recorded in addendum_012_in_place_tier1_finalization.json. Path, size, mode, mtime and flags remain protected wherever visible; a recovered materialized file must match its sealed SHA-256 and size. The 381 cloud-attested items were not recovered. The original source verifiers and runtime input checks are retained as prior evidence; this finalization does not claim a fresh full upstream content check for cloud-unavailable artifacts.\n\n'+'All 66 production groups and 17,482 checkpoint pairs match their frozen checkpoint content hashes. Finalization used existing results and byte-identical, sealed reporting/statistical source snapshots; no fitting, gate recomputation, result replacement, package installation or download occurred. M4/Tier 2 is NOT RUN. These are development results on reused years. See RESULTS.md, SUMMARY.txt and verify_tier1_in_place.py.\n').encode()
 with readme.open('ab') as handle:handle.write(suffix);handle.flush();os.fsync(handle.fileno())
 c.write(HERE/'audit/in_place_finalization_receipt.json',dict(status='PASS_REPORT_ONLY_FINALIZATION',limitation=LIMITATION,README_prefix_bytes=len(prefix),README_prefix_sha256=hashlib.sha256(prefix).hexdigest(),README_append_sha256=hashlib.sha256(suffix).hexdigest(),checkpoint_proof=after,new_fits=0,new_gate_tests=0,existing_results_changed=0,source_dependencies=dependencies,acceptance_receipts=acceptance,original_registration_sha256=digest(HERE/'audit/code_registration.json'),original_verifier_sha256=digest(HERE/'verify_real_mechanism_spi.py'),report_code_sha256=digest(HERE/'scripts/report.py')))
 c.publish(HERE/'verify_tier1_in_place.py',(R/'verify_tier1_in_place.py').read_bytes())
 c.publish(HERE/'finalize_tier1_in_place.py',Path(__file__).read_bytes())
 for h in c.HANDLES.values():h.close()
 c.HANDLES.clear()
 seal=HERE/f'audit/seal_in_place_{time.time_ns()}';seal.mkdir();c.write(seal/'operation_ledger.json',c.ledger())
 c.write(seal/'audit.json',dict(status='PASS_WITH_USER_AUTHORIZED_CONDITIONAL_PROVENANCE',limitation=LIMITATION,conditional_content_paths=303,checkpoint_proof=after,selected_stop_after_tier=1,cumulative_trials=549,Tier2='NOT RUN',report_only=True,prior_acceptance_receipts=acceptance,source_snapshot_receipt_sha256=digest(HERE/'audit/report_source_snapshots.json'),user_decision_sha256=digest(R/'USER_DECISION.json'),relocation='PAUSED_BY_USER'))
 manifest=seal/'manifest.json';files={str(p.relative_to(HERE)):digest(p) for p in c.artifact_paths(manifest)};c.write(manifest,dict(files=files,excluded=['runtime/tmp','runtime/mpl','runtime/cache','pending files','pipeline.lock','complete.json','.DS_Store']))
 complete=dict(status='complete',verifier='verify_tier1_in_place.py',original_verifier='verify_real_mechanism_spi.py',selected_stop_after_tier=1,cumulative_trials=549,evidence='development; reused years; user-authorized conditional upstream provenance',limitation=LIMITATION,manifest_file=str(manifest.relative_to(HERE)),manifest_sha256=digest(manifest),audit_file=str((seal/'audit.json').relative_to(HERE)),audit_sha256=digest(seal/'audit.json'),operation_ledger_file=str((seal/'operation_ledger.json').relative_to(HERE)),operation_ledger_sha256=digest(seal/'operation_ledger.json'),preregistration_sha256=digest(HERE/'preregistration.json'),addenda_sha256={p.name:digest(p) for p in HERE.glob('addendum_*.json')},input_seals=c.spec()['input_seals'],report_source_snapshots_sha256=digest(HERE/'audit/report_source_snapshots.json'),conditional_provenance_paths=303)
 c.write(HERE/'complete.json',complete);print('\n'.join(lines),flush=True)
 fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
if __name__=='__main__':main()
