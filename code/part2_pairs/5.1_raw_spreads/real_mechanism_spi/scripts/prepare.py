"""Preparation never launches production fits; long real acceptance stays main-only."""
from common import *
from protection import quick_preflight,metadata_check,original_verifiers,locks
import subprocess,argparse
def register_sources():
 quick_preflight()
 paths=subprocess.run(['rg','--files','--hidden',str(ROOT),'-g','*.py','-g','*.c','-g','!**/.venv/**','-g','!**/.git/**','-g','!**/node_modules/**','-g','!**/runtime/**','-g','!**/tmp/**','-g','!**/fixtures/**','-g','!**/tests/**','-g','!**/experiments/turning_point_detection/**'],capture_output=True,text=True,check=True).stdout.splitlines()
 reg={}
 for name in paths:
  p=Path(name)
  if p.is_relative_to(HERE) or p.is_symlink() or p.lstat().st_flags&DATALESS:continue
  reg[str(p.relative_to(ROOT))]=sha(p)
 for p in [SOURCE/'runtime/adapter_kernel.so',SOURCE/'runtime/q_kernel.so',SYN/'runtime/tuned_kernel.so']:
  reg[str(p.relative_to(ROOT))]=sha(p)
 c=MECH/'scripts/mechanism_kernel.c';binary=MECH/('runtime/mechanism_'+sha(c)[:16]+'.so')
 assert binary.is_file(),('Sealed instrumented native kernel missing',binary)
 reg[str(binary.relative_to(ROOT))]=sha(binary)
 write(HERE/'audit/source_registration.json',reg)
 for source,target in [(c,HERE/'scripts/mechanism_kernel.c'),(binary,HERE/('runtime/'+binary.name)),(SOURCE/'runtime/q_kernel.so',HERE/'runtime/q_kernel.so')]:
  publish(target,source.read_bytes());assert sha(source)==sha(target)
 write(HERE/'audit/native_copies.json',dict(files=[dict(source=str(c.relative_to(ROOT)),target='scripts/mechanism_kernel.c',sha256=sha(c)),dict(source=str(binary.relative_to(ROOT)),target='runtime/'+binary.name,sha256=sha(binary)),dict(source=str((SOURCE/'runtime/q_kernel.so').relative_to(ROOT)),target='runtime/q_kernel.so',sha256=sha(SOURCE/'runtime/q_kernel.so'))],compiled=False))
 print('Registered unchanged sources',len(reg),flush=True)
def tests(main=False):
 label='acceptance' if main else 'preparation'
 old=passing_tests(label,required=False)
 if old is not None:return old
 code_sha=validation_fingerprint();log=HERE/f'audit/tests_{label}_{time.time_ns()}.log'
 env=dict(os.environ,RMS_SCOPE=label)
 args=[sys.executable,'-B',str(HERE/'scripts/test_entry.py'),'--workers','1','-q',str(HERE/'tests'),'-p','no:cacheprovider','-m','main' if main else 'not main','--basetemp',str(HERE/f'runtime/tmp/pytest_{time.time_ns()}')]
 if main:args.append('-x')
 with log.open('x') as f:
  process=subprocess.Popen(args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=env)
  try:code=process.wait()
  except KeyboardInterrupt:
   flag=Path(os.environ.get('RMS_STOP_FILE',str(HERE/f'runtime/tmp/stop_{time.time_ns()}.json')))
   if not flag.exists():write(flag,dict(reason='interrupt during acceptance'))
   process.wait();raise
 assert code==0,('Tests failed; failure preserved',log)
 assert validation_fingerprint()==code_sha,('Code changed during tests',label)
 write(HERE/f'audit/tests_{label}_{code_sha}.ready.json',dict(status='PASS',log=str(log.relative_to(HERE)),log_sha256=sha(log),workers=1,code_sha256=code_sha))
 print('PASS',label,'tests',flush=True)
 return passing_tests(label)
def finish():
 import importlib.metadata
 check_environment()
 proof=passing_tests('preparation');measured=ledger();production=measured['scopes'].get('production',{})
 assert production.get('TD_fits',0)==production.get('switch_tests',0)==0
 metadata=sorted((HERE/'audit').glob('metadata_preparation_*.json'))[-1]
 inputs=sorted((HERE/'audit').glob('input_verifiers_preparation_*.json'))[-1]
 assert read(metadata)['status']=='PASS' and len(read(inputs))==len(spec()['input_seals'])
 historical=sorted((HERE/'audit').glob('historical_post_regular_repair_*.ready.json'))[-1]
 historical_proof=read(historical);assert historical_proof['status']=='PASS' and sha(HERE/historical_proof['log'])==historical_proof['log_sha256']
 repair=read(HERE/'audit/regular_mtime_repair/result.json')
 assert repair['status']=='PASS' and repair['restored_and_verified']==9 and repair['atime_preserved'] and repair['source_file_content_writes']==0
 assert sha(HERE/'audit/regular_mtime_repair/journal.jsonl')==repair['journal_sha256']
 initial=read(HERE/'audit/README_initial_prefix.json')
 assert hashlib.sha256((HERE/'README.md').read_bytes()[:initial['bytes']]).hexdigest()==initial['sha256']
 names=[p for d in ['scripts','tests'] for p in (HERE/d).rglob('*') if p.is_file() and p.suffix in {'.py','.c'}]
 names+=[HERE/'preregistration.json',HERE/'request.txt',HERE/'PREPARATION.md',HERE/'RUN_ALL.zsh',HERE/'verify_real_mechanism_spi.py',HERE/'pytest.ini',HERE/'audit/source_registration.json',HERE/'audit/native_copies.json',HERE/'audit/implementation_registration.json',HERE/'audit/README_initial_prefix.json',HERE/proof['receipt'],HERE/proof['log'],metadata,inputs]+list(HERE.glob('addendum_*.json'))
 names += [HERE/r['log'] for r in read(inputs)]
 names += [p for p in (HERE/'audit/symlink_repair').rglob('*') if p.is_file() and p.name!='pipeline.lock']
 names += [p for p in (HERE/'audit/regular_mtime_repair').rglob('*') if p.is_file()]
 names += [p for p in (HERE/'audit/gitignore_hydration').rglob('*') if p.is_file()]
 names += [p for p in (HERE/'audit/pandas_restore').rglob('*') if p.is_file()]
 names += [historical,HERE/historical_proof['log'],HERE/'audit/regular_file_mtime_repair_plan.csv']
 names.extend((HERE/'runtime').glob('*.so'))
 write(HERE/'audit/code_registration.json',dict(files={str(p.relative_to(HERE)):sha(p) for p in names},README_sha256=sha(HERE/'README.md'),python_version=sys.version.split()[0],packages={n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','matplotlib','pytest']}))
 write(HERE/'audit/preparation_ledger.json',measured)
 write(HERE/'audit/preparation_ready.json',dict(status='prepared_not_run',production_fits=0,production_switch_tests=0,selected_stop_after_tier=1,short_tests='PASS',short_test_receipt=proof['receipt'],metadata_receipt=str(metadata.relative_to(HERE)),input_verifier_receipt=str(inputs.relative_to(HERE)),post_regular_repair_historical_receipt=str(historical.relative_to(HERE)),regular_mtime_repair='audit/regular_mtime_repair/result.json',gitignore_hydration_outcome='addendum_005_gitignore_hydration_outcome.json',standing_workspace_policy_sha256=sha(ROOT/'AGENTS.md'),
  deferred=['full prune-aware protected historical hashing before and after','real three-pool/fold POW reproduction and L0/null/mapping fixtures','full sealed bootstrap/DSR/PBO reproduction','Tier 1 production'],command='nice -n 10 experiments/15_pair_trading/'+HERE.name+'/RUN_ALL.zsh --stop-after-tier 1'))
 publish(HERE/'audit.json','{"authoritative_file":"complete.json:audit_file","preparation":"audit/preparation_ready.json","status":"see preparation or final seal"}\n')
 publish(HERE/'operation_ledger.json',json.dumps(dict(authoritative_file='complete.json:operation_ledger_file',preparation='audit/preparation_ledger.json',downloads=measured['downloads'],package_installs=measured['package_installs']))+'\n')
 print(HERE/'README.md',flush=True);print('nice -n 10 '+str(HERE.relative_to(ROOT))+'/RUN_ALL.zsh --stop-after-tier 1',flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['register','metadata','inputs','tests','finish','all'],default='all');args=parser.parse_args()
 guard()
 with locks():
  for stage in (['register','metadata','inputs','tests','finish'] if args.stage=='all' else [args.stage]):
   if stage=='register':register_sources()
   elif stage=='metadata':metadata_check('preparation')
   elif stage=='inputs':original_verifiers('preparation')
   elif stage=='tests':tests()
   else:finish()
