"""User-invoked Tier 1 pipeline. Preparation never calls this controller."""
from common import *
from protection import locks,quick_preflight,registered,metadata_check,original_verifiers,historical,protected_snapshot,compare_snapshots
import argparse,subprocess
def output_seal(protected):
 attempt=HERE/f'audit/seal_{time.time_ns()}';attempt.mkdir()
 write(attempt/'operation_ledger.json',ledger())
 write(attempt/'audit.json',dict(status='PASS',evidence='development; reused test years',selected_stop_after_tier=1,cumulative_trials=549,
  source_runtime_contexts_or_cells_for_science=False,protected_scope=protected,input_seals=spec()['input_seals'],
  addenda_sha256={p.name:sha(p) for p in HERE.glob('addendum_*.json')},original_TAV1_result='expected registered absence; full prune-aware remaining-file check PASS',
  main_acceptance=passing_tests('acceptance'),scientific_cells=['TD8_C01g1','SPI_all','SPI_TD'],Tier2='NOT_RUN'))
 manifest=attempt/'manifest.json'
 files={}
 for p in artifact_paths(manifest):
  files[str(p.relative_to(HERE))]=sha(p)
 write(manifest,dict(files=files,excluded=['runtime/tmp','runtime/mpl','runtime/cache','pending files','pipeline.lock','complete.json','.DS_Store']))
 complete=dict(status='complete',verifier='verify_real_mechanism_spi.py',selected_stop_after_tier=1,cumulative_trials=549,evidence='development',
   manifest_file=str(manifest.relative_to(HERE)),manifest_sha256=sha(manifest),audit_file=str((attempt/'audit.json').relative_to(HERE)),audit_sha256=sha(attempt/'audit.json'),
   operation_ledger_file=str((attempt/'operation_ledger.json').relative_to(HERE)),operation_ledger_sha256=sha(attempt/'operation_ledger.json'),preregistration_sha256=sha(HERE/'preregistration.json'),
   addenda_sha256={p.name:sha(p) for p in HERE.glob('addendum_*.json')},input_seals=spec()['input_seals'])
 write(HERE/'complete.json',complete);return complete
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=max(1,(os.cpu_count() or 1)//2));parser.add_argument('--stop-after-tier',type=int,choices=[1],default=1);args=parser.parse_args()
 assert args.workers>=1
 os.environ['RMS_STOP_FILE']=str(HERE/f'runtime/tmp/stop_{time.time_ns()}.json');os.environ['RMS_SCOPE']='production';guard()
 with locks():
  if (HERE/'complete.json').exists():
   subprocess.run([sys.executable,'-B',str(HERE/'verify_real_mechanism_spi.py')],cwd=ROOT,check=True)
   print(HERE/'README.md');print((HERE/'SUMMARY.txt').read_text(),end='');return
  registered();quick_preflight();metadata_check('main_before');original_verifiers('main_before');historical('before');protected_snapshot('before')
  # Explicitly main-only: retained CSV reference extraction and real numerical
  # reproductions happen before production fitting, never during preparation.
  from prepare import tests
  tests(main=True);checkpoint_guard()
  from parallel import run
  from jobs import run_fold
  tasks=[(pool,y,1) for pool in POOLS for y in YEARS]
  def estimate(first):
   seconds=first['seconds'];remaining=max(0,len(tasks)-1);projected=remaining*seconds/args.workers
   value=dict(first_pool=first['pool'],first_year=first['year'],measured_fold_seconds=seconds,workers=args.workers,Tier1_remaining_hours=projected/3600,
      Tier2='NOT RUN; a future one-cell projection would use measured fit/executor components, not a CAP timing claim',limitations='Fold/pool sizes differ; estimate excludes final protected hashing and CSCV reporting.')
   write(HERE/f'audit/runtime_estimate_{time.time_ns()}.json',value)
   print('MEASURED RUNTIME ESTIMATE',json.dumps(value),flush=True)
  run(tasks,run_fold,args.workers,estimate)
  from report import worker as make_report
  run([None],make_report,1);checkpoint_guard();metadata_check('main_after');original_verifiers('main_after');historical('after');protected_snapshot('after')
  protected=compare_snapshots(HERE/'audit/protected_before.csv.gz',HERE/'audit/protected_after.csv.gz');registered();output_seal(protected)
  subprocess.run([sys.executable,'-B',str(HERE/'verify_real_mechanism_spi.py')],cwd=ROOT,check=True)
  print(HERE/'README.md',flush=True);print((HERE/'SUMMARY.txt').read_text(),end='',flush=True)
if __name__=='__main__':
 try:main()
 except (KeyboardInterrupt,CleanPause) as error:
  stop=os.environ.get('RMS_STOP_FILE')
  if stop and not Path(stop).exists():write(Path(stop),dict(reason=str(error) or 'Ctrl-C'))
  print('PAUSED CLEANLY:',str(error) or 'Ctrl-C','Resume the identical command. No source cleanup or repair was performed.',flush=True)
  raise SystemExit(130)
 except BaseException:
  import traceback
  write(HERE/f'audit/failure_{time.time_ns()}.json',dict(status='STOPPED',traceback=traceback.format_exc()))
  raise
