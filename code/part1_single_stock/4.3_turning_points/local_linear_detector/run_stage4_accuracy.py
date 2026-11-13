"""Fourth-step accuracy-only experiment; full execution is user-launched."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

# Process-level parallelism, with one numerical thread per worker. Set before
# numpy/pandas imports in both parent and spawn children.
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
sys.dont_write_bytecode = True
REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from experiments.turning_point_detection.new_stage1.run_slope_accuracy import load_accuracy_config, load_prices, repo_path
from experiments.turning_point_detection.new_stage1.run_slope_accuracy_diagnostics import calendar_for_prices, check_existing_hashes, sha
from experiments.turning_point_detection.new_stage1.slope_accuracy import generate_random_walk
from experiments.turning_point_detection.new_stage1.slope_accuracy_diagnostics import null_seed, empirical_null_path
from experiments.turning_point_detection.new_stage1.stage4_accuracy import (
    evaluate_variant, null_comparison, vol_relationship_statistics,
    vol_relationship_comparison, variant_comparison,
)

DEFAULT_CONFIG = HERE / 'configs/stage4_accuracy.json'
TABLE_NAMES = (
    'null_comparison_R500', 'vol_relationship_vs_b1', 'test_accuracy_adaptive',
    'selected_params_adaptive', 'null_comparison_adaptive', 'per_fold_test_vs_vol_adaptive',
    'vol_relationship_adaptive_vs_b1', 'detector_variant_comparison', 'test_accuracy_adaptive_L20',
    'evaluation_samples_stage4', 'null_selected_params_stage4', 'null_generation_stage4',
    'vol_relationship_samples_stage4', 'fold_calendar_stage4', 'runtime_timing_stage4',
    'train_grid_scores_stage4', 'per_fold_test_vs_vol_fixed_R500', 'selected_params_adaptive_L20',
)


def load_config(path=DEFAULT_CONFIG, *, smoke=False):
    ext = json.loads(Path(path).read_text())
    config = load_accuracy_config(repo_path(ext['base_config']))
    previous = json.loads(repo_path(ext['previous_diagnostics_config']).read_text())
    if any(ext['null_seeds'][source] == previous['seed'] for source in ('B1', 'B2')):
        raise ValueError('B1/B2 seeds must differ from the previous round')
    if ext['primary_null'] != 'B1' or ext['secondary_null'] != 'B2' or ext['hypothesis_metric'] != 'f1':
        raise ValueError('Prespecified nulls/F1 hypotheses must be retained')
    if ext['hypothesis_types'] != ['peak', 'trough', 'combined']:
        raise ValueError('All three prespecified turn types must be retained')
    config['output_dir'] = ext['output_dir']
    if smoke:
        config.update({key: ext['smoke'][key] for key in ('tickers', 'h_ref_grid')})
        config['output_dir'] = ext['smoke_output_dir']
        ext['effective_repetitions'] = {source: ext['smoke']['null_repetitions'] for source in ('B0', 'B1', 'B2')}
    else:
        ext['effective_repetitions'] = dict(ext['null_repetitions'])
    return config, ext


def make_null_prices(prices, calendar, config, ext, source, repeat):
    paths, audits = {}, []
    if source == 'B0':
        for fold in calendar.to_dict('records'):
            ticker, fold_id = fold['ticker'], fold['fold_id']
            seed = null_seed(ext['null_seeds'][source], source, repeat, ticker, fold_id)
            paths[(fold_id, ticker)], detail = generate_random_walk(prices[ticker], fold, seed=seed, ddof=config['std_ddof'])
            audits.append(dict(source=source, repeat=repeat, ticker=ticker, fold_id=fold_id, **detail))
    else:
        for ticker, p in prices.items():
            seed = null_seed(ext['null_seeds'][source], source, repeat, ticker)
            paths[ticker], detail = empirical_null_path(p, source, seed)
            audits.append(dict(repeat=repeat, ticker=ticker, fold_id='one_path_all_folds', **detail))
    # Mixed signed/unsigned 64-bit seeds otherwise coerce to float when
    # concatenating sources; stringify BEFORE constructing any DataFrame.
    for row in audits:
        row['seed'] = str(row['seed'])
    return paths, pd.DataFrame(audits)


def relationship_stats(vol, ext, *, source, repeat, variant):
    return vol_relationship_statistics(vol, low=ext['vol_ratio_low'], high=ext['vol_ratio_central_high']).assign(
        source=source, repeat=repeat, variant=variant)


def execute_null_job(task, prices, calendar, config, ext):
    source, repeat = task
    start = time.perf_counter()
    paths, generation = make_null_prices(prices, calendar, config, ext, source, repeat)
    generation_seconds = time.perf_counter()-start
    variants = ['adaptive'] if source == 'B0' else ['fixed', 'adaptive']
    summaries, selections, statistics, timings = [], [], [], []
    for variant in variants:
        begin = time.perf_counter()
        result = evaluate_variant(paths, calendar, config, variant=variant, L=ext['adaptive_L'], source=source, repeat=repeat)
        summaries.append(result['summary']); selections.append(result['selected'])
        if source == 'B1':
            statistics.append(relationship_stats(result['vol'], ext, source=source, repeat=repeat, variant=variant))
        timings.append(dict(source=source, repeat=repeat, variant=variant, evaluation_seconds=time.perf_counter()-begin,
                            generation_seconds=generation_seconds, ticker_fold_reference_units=len(calendar)*len(config['h_ref_grid'])))
    return {'samples': summaries, 'selected': selections, 'statistics': statistics,
            'generation': generation, 'timing': pd.DataFrame(timings)}


_WORKER_STATE = None


def initialize_worker(prices, calendar, config, ext):
    global _WORKER_STATE
    _WORKER_STATE = (prices, calendar, config, ext)


def worker_job(task):
    return execute_null_job(task, *_WORKER_STATE)


def iterate_null_jobs(tasks, prices, calendar, config, ext, workers):
    if workers == 1:
        for task in tasks:
            yield execute_null_job(task, prices, calendar, config, ext)
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=initialize_worker, initargs=(prices, calendar, config, ext)) as pool:
            # Stable task order, independent repeat seeds, no writes in workers.
            yield from pool.map(worker_job, tasks, chunksize=1)


def validate_fixed_baseline(summary, ext):
    saved = pd.read_csv(repo_path(ext['fixed_baseline']))
    keys = ['h_ref', 'ticker', 'type']
    fields = ['n_ref', 'n_det', 'n_match', 'precision', 'recall', 'f1', 'delay_q25', 'delay_median', 'delay_q75', 'n_test_years']
    a, b = summary.set_index(keys).sort_index(), saved.set_index(keys).sort_index()
    if not a.index.equals(b.index):
        raise AssertionError('Saved fixed v2 scope differs; cannot reuse B0')
    np.testing.assert_allclose(a[fields], b[fields], rtol=2e-11, atol=2e-11, equal_nan=True)
    return {'file': ext['fixed_baseline'], 'sha256': sha(repo_path(ext['fixed_baseline'])), 'rows_checked': len(a)}


def reuse_fixed_b0(summary, ext):
    reconciliation = validate_fixed_baseline(summary, ext)
    path = repo_path(ext['fixed_b0_samples'])
    samples = pd.read_csv(path)
    samples = samples.loc[samples.source.eq('B0')].copy()
    keys = ['h_ref', 'ticker', 'type']
    expected = pd.MultiIndex.from_frame(summary[keys])
    actual = pd.MultiIndex.from_frame(samples[keys].drop_duplicates())
    if len(actual) != len(expected) or set(actual) != set(expected):
        raise AssertionError('Historical B0 scope mismatch')
    R = ext['null_repetitions']['B0']
    if not samples.groupby(keys).repeat.nunique().eq(R).all():
        raise AssertionError('Historical B0 repetitions mismatch')
    samples['variant'] = 'fixed'; samples['L'] = ext['adaptive_L']
    return samples, {'fixed_baseline_reconciliation': reconciliation, 'file': str(path), 'sha256': sha(path), 'R': R}


def estimate_full_runtime(timing, full_calendar, config, ext, workers):
    """Measured sequential work projection, not an accuracy prediction."""
    units = len(full_calendar)*len(load_accuracy_config(repo_path(ext['base_config']))['h_ref_grid'])
    rows = []
    for (source, variant), group in timing.loc[timing.source.ne('real')].groupby(['source', 'variant'], sort=True):
        per_unit = float((group.evaluation_seconds/group.ticker_fold_reference_units).mean())
        R = ext['null_repetitions'][source]
        rows.append({'source': source, 'variant': variant, 'R_full': R,
                     'measured_seconds_per_ticker_fold_reference': per_unit,
                     'projected_sequential_evaluation_seconds': per_unit*units*R})
    evaluation = sum(row['projected_sequential_evaluation_seconds'] for row in rows)
    generation = 0.
    for source, group in timing.loc[timing.source.ne('real')].groupby('source', sort=True):
        # Generation is performed once for the paired fixed/adaptive job.
        unique = group.drop_duplicates(['source', 'repeat'])
        scale = len(full_calendar)/len(config['_smoke_calendar']) if source == 'B0' else len(full_calendar.ticker.unique())/len(config['tickers'])
        generation += float(unique.generation_seconds.mean())*scale*ext['null_repetitions'][source]
    sequential = evaluation+generation
    return {'full_ticker_fold_rows': len(full_calendar), 'full_reference_bandwidths': 3,
            'method': 'smoke各源/变体实测每ticker-fold-h_ref耗时乘完整任务量；不含额外真实数据校验、I/O与进程开销；h_ref=10外推到5/20',
            'components': rows, 'projected_sequential_seconds': sequential,
            'parallel_workers': workers, 'ideal_parallel_seconds': sequential/workers,
            'parallel_note': '理想工作量除以进程数，非实测并行承诺；带宽差异、硬件争用和进程开销会改变实际耗时'}


def comparison_header(config, ext, *, smoke, variant):
    family = len(config['h_ref_grid'])*len(ext['hypothesis_types'])
    return [
        f'accuracy-only; variant={variant}; v2: TRAIN末尾截断3h_ref，TEST仅整条数据末尾截断',
        '预先指定：主零基准B1；次要B2；检验指标macro F1；两者分别做Holm，不跨源合并',
        f'完整检验族=3 h_ref x peak/trough/combined=9；本文件实际family_size={family}; mode={"smoke，仅检查流水线" if smoke else "full"}',
        '双侧经验p=min(1,2*min((1+#null<=real)/(R+1),(1+#null>=real)/(R+1)))；包含等值',
        '单标的和B0仅描述分位，无显著性判定；延迟仅描述分位，不另做检验',
        '真实值分位采用midrank；F1为拼接各test年事件后计算，macro按ticker等权',
        f'配置R={json.dumps(ext["effective_repetitions"])}；B1/B2新seed={ext["null_seeds"]["B1"]}/{ext["null_seeds"]["B2"]}',
        '这些零基准衡量单侧/双侧共享数据的机械吻合，不是价格随机游走检验',
    ]


def run(config, ext, *, smoke=False, workers=1):
    wall_start = time.perf_counter()
    output = repo_path(config['output_dir']); tables = output / 'tables'
    targets = [tables / f'{name}.csv' for name in TABLE_NAMES] + [output / 'stage4_accuracy_manifest.json', output / 'stage4_table_inventory.md']
    conflicts = [str(p) for p in targets if p.exists()]
    if conflicts:
        raise FileExistsError('Refusing to overwrite existing outputs: '+', '.join(conflicts))
    output.mkdir(parents=True, exist_ok=True); tables.mkdir(parents=True, exist_ok=True)
    with (output / 'stage4_pipeline.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        protected = {str(p.relative_to(REPO_ROOT)): sha(p) for p in HERE.rglob('*')
                     if p.is_file() and p.name != 'stage4_pipeline.lock' and '__pycache__' not in p.parts}
        for path in [REPO_ROOT/'progress.md', HERE.parent/'progress.md']:
            if path.is_file():
                protected[str(path.relative_to(REPO_ROOT))] = sha(path)
        prices = {}
        for ticker in config['tickers']:
            path = repo_path(config['raw_dir']) / f'{ticker.lower()}_daily_adjusted_ohlcv.csv'
            protected[str(path.relative_to(REPO_ROOT))] = sha(path)
            prices[ticker] = load_prices(ticker, config)
        calendar = calendar_for_prices(prices, config)
        real, times = {}, []
        for variant, L, label in [('fixed', ext['adaptive_L'], 'fixed'), ('adaptive', ext['adaptive_L'], 'adaptive'),
                                  ('adaptive', ext['sensitivity_L'], 'adaptive_L20')]:
            begin = time.perf_counter()
            real[label] = evaluate_variant(prices, calendar, config, variant=variant, L=L, collect_train=True)
            times.append(pd.DataFrame([dict(source='real', repeat=-1, variant=label,
                            evaluation_seconds=time.perf_counter()-begin, generation_seconds=0.,
                            ticker_fold_reference_units=len(calendar)*len(config['h_ref_grid']))]))
            print(f'REAL {label} complete', flush=True)
        samples = [real['fixed']['summary'], real['adaptive']['summary']]
        choices, generations = [], []
        stats = [relationship_stats(real[label]['vol'], ext, source='real', repeat=-1, variant=label)
                 for label in ('fixed', 'adaptive')]
        b0_reuse = {'note': 'smoke不重跑fixed B0；完整运行复用已有R100。当前smoke折范围不同于旧单折smoke，省略fixed B0描述行'}
        if not smoke:
            reused, b0_reuse = reuse_fixed_b0(real['fixed']['summary'], ext)
            samples.append(reused)
        tasks = [(source, repeat) for source, R in ext['effective_repetitions'].items() for repeat in range(R)]
        for i, result in enumerate(iterate_null_jobs(tasks, prices, calendar, config, ext, workers), 1):
            samples.extend(result['samples']); choices.extend(result['selected']); stats.extend(result['statistics'])
            generations.append(result['generation']); times.append(result['timing'])
            print(f'NULL job {i}/{len(tasks)} complete: {tasks[i-1]}', flush=True)
        sample_frame = pd.concat(samples, ignore_index=True)
        stat_frame = pd.concat(stats, ignore_index=True)
        timing = pd.concat(times, ignore_index=True)
        fixed_R = {source: R for source, R in ext['effective_repetitions'].items() if source != 'B0' or not smoke}
        frames = {
            'null_comparison_R500': null_comparison(sample_frame.loc[sample_frame.variant.eq('fixed') & sample_frame.source.ne('real')], real['fixed']['summary'], expected_repetitions=fixed_R),
            'vol_relationship_vs_b1': vol_relationship_comparison(stat_frame.loc[stat_frame.variant.eq('fixed')]),
            'test_accuracy_adaptive': real['adaptive']['summary'].drop(columns=['source', 'repeat', 'variant', 'L']),
            'selected_params_adaptive': real['adaptive']['selected'],
            'null_comparison_adaptive': null_comparison(sample_frame.loc[sample_frame.variant.eq('adaptive') & sample_frame.source.ne('real')], real['adaptive']['summary'], expected_repetitions=ext['effective_repetitions']),
            'per_fold_test_vs_vol_adaptive': real['adaptive']['vol'],
            'vol_relationship_adaptive_vs_b1': vol_relationship_comparison(stat_frame.loc[stat_frame.variant.eq('adaptive')]),
            'detector_variant_comparison': variant_comparison(real['fixed']['summary'], real['adaptive']['summary']),
            'test_accuracy_adaptive_L20': real['adaptive_L20']['summary'].drop(columns=['source', 'repeat']).assign(analysis_label=ext['sensitivity_note']),
            'evaluation_samples_stage4': sample_frame,
            'null_selected_params_stage4': pd.concat(choices, ignore_index=True),
            'null_generation_stage4': pd.concat(generations, ignore_index=True),
            'vol_relationship_samples_stage4': stat_frame,
            'fold_calendar_stage4': calendar,
            'runtime_timing_stage4': timing,
            'train_grid_scores_stage4': pd.concat([r['train_grid'] for r in real.values()], ignore_index=True),
            'per_fold_test_vs_vol_fixed_R500': real['fixed']['vol'],
            'selected_params_adaptive_L20': real['adaptive_L20']['selected'].assign(analysis_label=ext['sensitivity_note']),
        }
        estimate = None
        if smoke:
            full_config = load_accuracy_config(repo_path(ext['base_config']))
            # Calendar-only access to existing CSV dates; no full price loading,
            # detector fitting, references or Monte Carlo on the full universe.
            from experiments.turning_point_detection.new_stage1.slope_accuracy import make_fold_calendar
            full_rows = []
            for ticker in full_config['tickers']:
                path = repo_path(full_config['raw_dir']) / f'{ticker.lower()}_daily_adjusted_ohlcv.csv'
                dates = pd.DatetimeIndex(pd.read_csv(path, usecols=[0]).iloc[:, 0])
                dates = dates[dates <= pd.Timestamp(full_config['as_of'])]
                full_rows.append(make_fold_calendar(ticker, dates, train_years=full_config['train_years'],
                    test_years=full_config['test_years'], step_years=full_config['step_years'], as_of=full_config['as_of'],
                    tsla_min_train_start=full_config['tsla_min_train_start'], market_calendar=full_config['market_calendar']))
            estimate = estimate_full_runtime(timing, pd.concat(full_rows, ignore_index=True),
                          {**config, '_smoke_calendar': calendar}, ext, ext['suggested_parallel_workers'])
        check_existing_hashes(protected)
        inventory = []
        for name, frame in frames.items():
            path = tables / f'{name}.csv'
            with path.open('x') as handle:
                if name in ('null_comparison_R500', 'null_comparison_adaptive'):
                    variant = 'fixed' if name == 'null_comparison_R500' else 'adaptive'
                    for line in comparison_header(config, ext, smoke=smoke, variant=variant):
                        handle.write('# '+line+'\n')
                frame.to_csv(handle, index=False, float_format='%.12g')
            inventory.append(dict(file=f'tables/{name}.csv', rows=len(frame), columns=list(frame.columns), sha256=sha(path)))
        with (output / 'stage4_table_inventory.md').open('x') as handle:
            handle.write('# Stage 4 generated table inventory\n\nCSV头部的#为预设方法说明；读取比较表时使用comment="#"。\n\n')
            for row in inventory:
                handle.write(f'## {row["file"]}\n\nRows: {row["rows"]}\n\nColumns: '+', '.join(row['columns'])+'\n\n')
        manifest = dict(mode='smoke' if smoke else 'full', timestamp=datetime.now().astimezone().isoformat(),
                        config=config, extension=ext, workers=workers, tables=inventory,
                        elapsed_seconds=time.perf_counter()-wall_start, runtime_projection=estimate,
                        fixed_b0_reuse=b0_reuse, protected_reference_hashes=protected,
                        protected_existing_files_unchanged=True,
                        methodology='v2准确度；固定与因果收益波动自适应死区；无交易；B1/B2各源独立macro F1 Holm检验族；未修改progress.md')
        with (output / 'stage4_accuracy_manifest.json').open('x') as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False, allow_nan=False)
        print(json.dumps({'tables': inventory, 'elapsed_seconds': manifest['elapsed_seconds'], 'runtime_projection': estimate}, ensure_ascii=False, indent=2), flush=True)
        return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--output-dir', help='New directory; existing files are never overwritten')
    parser.add_argument('--workers', type=int, help='Independent repeat processes; deterministic seeds')
    args = parser.parse_args()
    config, ext = load_config(args.config, smoke=args.smoke)
    if args.output_dir:
        config['output_dir'] = args.output_dir
    workers = args.workers if args.workers is not None else ext['workers']
    if workers < 1:
        raise ValueError('workers must be positive')
    run(config, ext, smoke=args.smoke, workers=workers)


if __name__ == '__main__':
    main()
