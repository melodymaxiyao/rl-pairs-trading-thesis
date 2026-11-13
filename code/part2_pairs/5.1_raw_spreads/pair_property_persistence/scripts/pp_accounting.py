"""Operation ledger from actual records, including repeated or interrupted attempts."""
from ppio import *
from collections import Counter, defaultdict

SKIP_KEYS = {'operation', 'scope', 'utc', 'test_year', 'max_numeric_year', 'chunk', 'year', 'exit_code', 'workers',
             'chunk_size', 'BLAS_threads', 'path', 'fingerprint', 'source'}
FORBIDDEN = ['downloads', 'rl_fits', 'trading_evaluations', 'test_rows_parsed', 'test_rows_used', 'package_installs']

def operations():
    scopes = defaultdict(Counter); imports = {}; readers = Counter(); events = 0; partial = 0
    for path in sorted((HERE / 'audit').glob('operations_*.jsonl')):
        lines = path.read_text().splitlines()
        for index, line in enumerate(lines):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                assert index == len(lines) - 1, ('Malformed operation ledger', path, index)
                partial += 1; continue
            events += 1; scope = row['scope']; scopes[scope]['operation_records'] += 1
            for key, value in row.items():
                if key in SKIP_KEYS: continue
                if isinstance(value, int) and not isinstance(value, bool):
                    scopes[scope][key] += value
            assert not any(row.get(k, 0) for k in FORBIDDEN), ('Forbidden operation recorded', row)
            if row['operation'] == 'unchanged_source_import':
                assert row['path'] not in imports or imports[row['path']] == row['fingerprint']
                imports[row['path']] = row['fingerprint']
            if row['operation'] in ('train_only_input', 'fold_context_load'):
                assert row.get('max_numeric_year') is None or row['max_numeric_year'] < row['test_year'], ('Test-year numeric row', row)
                readers[(scope, row['operation'])] += 1
    after = {name: fp(ROOT / name) for name in imports}
    total = Counter()
    for counts in scopes.values():
        total.update(counts)
    return dict(downloads=0, package_installs=0, rl_fits=0, trading_evaluations=0, test_rows_parsed=0,
                actual_operation_records=events, partial_log_lines=partial,
                totals=dict(total), scopes={k: dict(v) for k, v in scopes.items()},
                train_only_reader_audit={f'{s}:{o}': n for (s, o), n in readers.items()},
                imported_modules_before=imports, imported_modules_after=after,
                counting_rule='Sums of counts in actual operation records, by scope (production, production_real, production_null, tests_*, preparation). '
                              'Every train_only_input record has max_numeric_year < test_year and test_rows_parsed = 0.')
