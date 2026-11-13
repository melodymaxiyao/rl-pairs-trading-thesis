"""One-command, resumable, train-only persistence diagnostic. No trading entry point."""
from ppio import *
sys.path.insert(0, str(HERE / 'scripts')); sys.path.insert(0, str(HERE))
import argparse, subprocess, traceback, multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pp_checkpoints import ready, receipt, load_rows, load_object, save_object, save_rows
from pp_inputs import protected, finder_paths, sealed_inputs, run_verifiers, verify_source_baseline
from pp_accounting import operations
from pp_sample import extract
from pp_worker import build_context, real_chunk, null_chunk
from pp_analysis import analyse
from pp_plots import plots
from pp_report import report

REAL_CHUNK, NULL_CHUNK = 500, 100
RUNTIME_DIRS = ['audit', 'runtime/checkpoints', 'runtime/tmp', 'runtime/cache', 'runtime/mpl_cache', 'outputs/tables', 'outputs/plots']

def initialize():
    # Repair checkpoint_dir_v1: checkpoint payloads need runtime/checkpoints to exist.
    for name in RUNTIME_DIRS:
        if not (HERE / name).is_dir(): (HERE / name).mkdir(parents=True)

def command(arguments, label, extra=None):
    env = dict(os.environ, **(extra or {}))
    path = HERE / 'audit' / (label + '_' + str(time.time_ns()) + '.log')
    with path.open('x') as stream:
        with subprocess.Popen(arguments, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1) as process:
            for line in process.stdout:
                stream.write(line); stream.flush(); print(line, end='', flush=True)
            code = process.wait()
    record('subprocess_completed', subprocess_calls=1, label=label, exit_code=code)
    if code: raise RuntimeError('Check failed; preserved log: ' + str(path))
    return path

def registered():
    from verify_property_persistence import registered as check
    return check()

def preflight():
    marker = HERE / 'audit/preflight_ready.json'
    if ready(marker):
        assert sealed_inputs() == read(HERE / 'audit/input_seals_before.json'), 'Input seals changed since preflight'
        return read(marker)
    seals = sealed_inputs()
    sources = verify_source_baseline()
    before = HERE / 'audit/protected_before.json'
    if not before.exists():
        print('Hashing protected workspace (excludes this task, turning_point_detection, .git/.venv/node_modules and .DS_Store)...', flush=True)
        write(before, protected())
    if not (HERE / 'audit/finder_paths_before.json').exists():
        write(HERE / 'audit/finder_paths_before.json', finder_paths())
    results = run_verifiers(sys.executable, HERE / 'audit')
    assert sealed_inputs() == seals, 'Input seals changed during verification'
    write(HERE / 'audit/input_seals_before.json', seals)
    write(HERE / 'audit/input_verifiers.json', results)
    receipt(marker, [HERE / 'audit/input_seals_before.json', HERE / 'audit/input_verifiers.json', before] +
            [HERE / r['log'] for r in results], input_verifiers=len(results), registered_sources=sources)
    return read(marker)

def tests(phase):
    marker = HERE / f'audit/tests_{phase}_ready.json'
    if ready(marker): return read(marker)
    token = str(time.time_ns()); xml = HERE / f'audit/tests_{phase}_{token}.xml'
    arguments = [sys.executable, '-B', str(HERE / 'scripts/test_entry.py'), '--junitxml', str(xml),
                 '--basetemp', str(HERE / f'runtime/pytest_{phase}_{token}'), str(HERE / 'tests'), '-q']
    log = command(arguments, 'tests_' + phase, dict(PP_SCOPE='tests_' + phase, PP_TEST_PHASE=phase))
    import xml.etree.ElementTree as ET
    suites = list(ET.parse(xml).getroot().iter('testsuite'))
    assert not any(int(s.get(k, 0)) for s in suites for k in ['failures', 'errors'])
    receipt(marker, [xml, log], tests=sum(int(s.get('tests', 0)) for s in suites),
            skipped=sum(int(s.get('skipped', 0)) for s in suites), phase=phase)
    return read(marker)

def ensure_context(year):
    if ready(HERE / f'runtime/checkpoints/context_{year}.json'): return
    eligible = load_object(f'eligible_{year}')
    tickers = sorted(set(eligible.leg_A) | set(eligible.leg_B))
    ctx = build_context(year, tickers)
    save_rows(f'ticker_stats_{year}', ctx['ticker_stats'], test_year=year, tickers=len(tickers))
    save_object(f'context_{year}', ctx, test_year=year, max_numeric_year=int(ctx['cal'].max().year), tickers=len(tickers), test_rows_parsed=0)

def finalize_fold(year, real_markers, null_markers):
    done = HERE / f'runtime/final_fold_{year}.json'
    if ready(done): return read(done)
    real = sum(read(m)['reproduced'] for m in real_markers)
    null = sum(read(m)['reproduced'] for m in null_markers)
    sample = read(HERE / f'runtime/checkpoints/sample_{year}.json')
    assert real == sample['sampled_rows'] and null == 2000
    write(HERE / f'audit/reproduction_{year}.json', dict(
        test_year=year, status='PASS', sampled_pair_folds=real,
        window_A_inner_fields_exact=real, window_B_validation_KM_med_exact=real, null_pairs_reproduced_exactly=null,
        null_fields='KM_med, n_events, n_reverted, median_reverted_gross_bps, VR_5, FAST63, G_pass, four G inputs, inner_KM_med, inner_n_reverted, inner_FAST63, validation_KM_med, sigma_A, sigma_B',
        train_volatilities_equal_universe=True, test_rows_parsed=0))
    receipt(done, [Path(m) for m in real_markers + null_markers] + [HERE / f'audit/reproduction_{year}.json'],
            test_year=year, sampled_pair_folds=real, null_pairs=null)
    print('Fold', year, 'complete:', real, 'sampled pair-folds and 2000 null pairs reproduced exactly.', flush=True)
    return read(done)

def compute(workers):
    started = time.perf_counter(); first = None
    plan = {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as executor:
        for year in YEARS:
            if ready(HERE / f'runtime/final_fold_{year}.json'): continue
            ensure_context(year)
            rows = load_rows(HERE / f'runtime/checkpoints/sample_{year}.json')
            real, null, futures = [], [], []
            for number, start in enumerate(range(0, len(rows), REAL_CHUNK)):
                marker = HERE / f'runtime/checkpoints/real_{year}_{number:05d}.json'; real.append(str(marker))
                if not ready(marker): futures.append(executor.submit(real_chunk, year, number, rows[start:start + REAL_CHUNK]))
            for number, start in enumerate(range(0, 2000, NULL_CHUNK)):
                marker = HERE / f'runtime/checkpoints/null_{year}_{number:05d}.json'; null.append(str(marker))
                if not ready(marker): futures.append(executor.submit(null_chunk, year, number, list(range(start, start + NULL_CHUNK))))
            plan[year] = (real, null, futures, len(rows))
        for year in YEARS:
            if year not in plan: continue
            real, null, futures, n = plan[year]
            for future in futures: future.result()
            finalize_fold(year, real, null)
            if first is None and futures:
                first = time.perf_counter() - started
                remaining = sum(plan[y][3] + 2000 for y in plan if y > year)
                done_units = sum(plan[y][3] + 2000 for y in plan if y <= year)
                estimate = dict(first_fold_wall_seconds=first, workers=workers, method='Elapsed wall time to the first finished fold, '
                                'including context building for later folds, scaled by remaining sampled+null pairs; a rough upper bound.',
                                estimated_remaining_seconds=first / max(done_units, 1) * remaining)
                write(HERE / f'audit/runtime_estimate_{time.time_ns()}.json', estimate)
                print('Measured remaining runtime estimate:', round(estimate['estimated_remaining_seconds'] / 60, 1), 'minutes; continuing.', flush=True)
    assert all(ready(HERE / f'runtime/final_fold_{y}.json') for y in YEARS)

def seal():
    verified = HERE / 'audit/unsealed_verification_ready.json'
    if ready(verified): log = HERE / read(verified)['log']
    else:
        log = command([sys.executable, '-B', str(HERE / 'verify_property_persistence.py'), '--unsealed'], 'verify_unsealed')
        receipt(verified, [log], log=str(log.relative_to(HERE)))
    if (HERE / 'operation_ledger.json').exists(): ledger = read(HERE / 'operation_ledger.json')
    else:
        ledger = operations(); write(HERE / 'operation_ledger.json', ledger)
    reproduction = [read(HERE / f'audit/reproduction_{y}.json') for y in YEARS]
    audit = dict(status='PASS', diagnostic_only=True, selects_nothing=True, trading=False, development_only=True,
                 preregistration_sha256=sha(HERE / 'preregistration.json'), request_sha256=sha(HERE / 'request.txt'),
                 input_seals=read(HERE / 'audit/input_seals_before.json'), input_verifiers=read(HERE / 'audit/input_verifiers.json'),
                 protected_before_sha256=sha(HERE / 'audit/protected_before.json'), protected_after_sha256=sha(HERE / 'audit/protected_after.json'),
                 protected_unchanged=True, finder_metadata_excluded_and_listed=True,
                 tests_before=read(HERE / 'audit/tests_before_ready.json'), tests_after=read(HERE / 'audit/tests_after_ready.json'),
                 sampled_pair_folds_reproduced=sum(r['sampled_pair_folds'] for r in reproduction),
                 null_pairs_reproduced=sum(r['null_pairs_reproduced_exactly'] for r in reproduction),
                 operation_ledger_sha256=sha(HERE / 'operation_ledger.json'), independent_verifier_log=str(log.relative_to(HERE)),
                 downloads=ledger['downloads'], rl_fits=ledger['rl_fits'], trading_evaluations=ledger['trading_evaluations'],
                 test_rows_parsed=ledger['test_rows_parsed'], PNGs=4)
    write(HERE / 'audit.json', audit)
    lock = HERE / 'pipeline.lock'
    if not lock.read_text().endswith('sealed\n'):
        with lock.open('a') as stream: stream.write('sealed\n')
    artifacts = {}
    for parent, dirs, names in os.walk(HERE, followlinks=False):
        for name in names:
            p = Path(parent) / name
            if p.parent == HERE and name in ['manifest.json', 'complete.json']: continue
            if name == '.DS_Store': continue
            artifacts[str(p.relative_to(HERE))] = fp(p)
    if not (HERE / 'manifest.json').exists(): write(HERE / 'manifest.json', dict(files=artifacts, created_at_utc=utc()))
    listed = read(HERE / 'manifest.json')['files']
    for name, value in listed.items(): assert fp(HERE / name) == value, ('Seal-resume artifact changed', name)
    late = {name: value for name, value in artifacts.items() if name not in listed}
    write(HERE / 'complete.json', dict(status='complete', diagnostic_only=True, trading=False,
          **{name + '_sha256': sha(HERE / (name + '.json')) for name in ['manifest', 'audit', 'operation_ledger', 'preregistration']},
          sampled_pair_folds=audit['sampled_pair_folds_reproduced'], null_pairs=audit['null_pairs_reproduced'],
          downloads=0, rl_fits=0, trading_evaluations=0, test_rows_parsed=0, late_seal_audit_artifacts=late))

def finish_message():
    print('README:', HERE / 'README.md')
    print('\n'.join(read(HERE / 'summary.json')['lines']))

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--workers', type=int, default=min(12, os.cpu_count() or 1))
    args = parser.parse_args(); workers = max(1, min(args.workers, 12, os.cpu_count() or 1))
    guard(); registered(); initialize()
    with (HERE / 'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (HERE / 'complete.json').exists():
            from verify_property_persistence import verify
            print(verify()); finish_message(); return
        try:
            if (HERE / 'operation_ledger.json').exists():
                sealed_inputs(); assert protected() == read(HERE / 'audit/protected_before.json')
                seal(); finish_message(); return
            record('execution_configuration', workers=workers, BLAS_threads=1, real_chunk_size=REAL_CHUNK, null_chunk_size=NULL_CHUNK)
            preflight()
            print('Extracting eligible pair-folds and the stratified sample from the universe inventory...', flush=True)
            extract()
            tests('before')
            compute(workers)
            print('Computing tables, plots and README results...', flush=True)
            analyse(); plots(); report()
            assert sealed_inputs() == read(HERE / 'audit/input_seals_before.json'), 'Input seals changed'
            verify_source_baseline()
            print('Re-hashing protected workspace...', flush=True)
            after = protected(); write(HERE / 'audit/protected_after.json', after)
            assert after == read(HERE / 'audit/protected_before.json'), 'Protected roots changed'
            write(HERE / f'audit/finder_paths_after_{time.time_ns()}.json', finder_paths())
            write(HERE / 'audit/counter_checkpoint.json', operations())
            tests('after')
            final = protected(); assert final == after, 'Protected roots changed during final tests'
            seal()
        except BaseException as error:
            write(HERE / f'audit/failure_{time.time_ns()}.json', dict(status='stopped', error=repr(error), traceback=traceback.format_exc(), utc=utc()))
            raise
    from verify_property_persistence import verify
    print(verify()); finish_message()

if __name__ == '__main__':
    main()
