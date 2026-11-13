"""Read-only source checks; logs and hashes exclusively in this new task root.

This is a prerequisite check, not the production pipeline. Historical verifier
failures are retained verbatim. No screening or trading is performed here.
"""
import sys
sys.dont_write_bytecode = True
sys.modules['readline'] = None
import os
import json
import hashlib
import subprocess
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[2]
PT = HERE.parent
INPUTS = [
    ('fresh_holdout_data_v1', 'verify_fresh_data.py'),
    ('universe_sector_expansion_v1', 'scripts/verify_sector_universe.py'),
    ('learner_noise_robustness_v1', 'scripts/verify_learner_noise.py'),
    ('expanded_convergence_gate_v1', 'scripts/verify_gate.py'),
]

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def fingerprint(path):
    if path.is_symlink():
        return {'symlink': os.readlink(path)}
    st = path.stat()
    return {'sha256': sha(path), 'size': st.st_size, 'mtime_ns': st.st_mtime_ns}

def write_new(path, obj):
    with path.open('x') as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)
        stream.write('\n')

def protected():
    files, directories = [], []
    for parent, children, names in os.walk(ROOT, followlinks=False):
        parent = Path(parent)
        children[:] = [n for n in children if n not in ('.git', '.venv', 'node_modules')
                       and parent / n not in (HERE, ROOT / 'experiments/turning_point_detection')]
        for name in children[:]:
            path = parent / name
            if path.is_symlink():
                files.append(path)
                children.remove(name)
            else:
                directories.append(str(path.relative_to(ROOT)))
        files.extend(parent / name for name in names if name != '.DS_Store')
    files.sort()
    with ThreadPoolExecutor(max_workers=min(12, os.cpu_count() or 1)) as executor:
        values = list(executor.map(fingerprint, files))
    return {'files': dict(zip((str(p.relative_to(ROOT)) for p in files), values)),
            'directories': sorted(directories)}

def main():
    token = str(time.time_ns())
    audit = HERE / 'audit'
    before = protected()
    write_new(audit / ('protected_before_' + token + '.json'), before)
    input_hashes = {k: v for k, v in before['files'].items()
                    if any(k.startswith('experiments/15_pair_trading/' + name + '/') for name, _ in INPUTS)}
    write_new(audit / ('inputs_before_' + token + '.json'), input_hashes)
    print('Hashed', len(before['files']), 'protected files and', len(input_hashes), 'input files.', flush=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
               CGATE_READ_ONLY='1', ZDIAG_READ_ONLY='1', MPLBACKEND='Agg')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        env[key] = '1'
    results = []
    try:
        for name, entry in INPUTS:
            marker = json.loads((PT / name / 'complete.json').read_text())
            assert marker['status'] == 'complete', (name, marker)
            script = PT / name / entry
            # The child installs an audit hook before importing the original
            # verifier. Its numeric/statistical checks are not replaced.
            cmd = [sys.executable, '-B', str(HERE / 'scripts/readonly_entry.py'), str(script)]
            print('Original read-only verifier:', name, flush=True)
            r = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True)
            log = audit / (name + '_verifier_' + token + '.log')
            with log.open('x') as stream:
                stream.write(r.stdout + r.stderr)
            results.append({'input': name, 'exit_code': r.returncode,
                            'log': str(log.relative_to(HERE)),
                            'complete_sha256': sha(PT / name / 'complete.json')})
            print((r.stdout + r.stderr)[-4500:], flush=True)
            if r.returncode:
                break
    finally:
        after = protected()
        write_new(audit / ('protected_after_' + token + '.json'), after)
        after_inputs = {k: after['files'].get(k) for k in input_hashes}
        write_new(audit / ('inputs_after_' + token + '.json'), after_inputs)
        unchanged = before == after
        write_new(audit / ('prerequisites_' + token + '.json'), {
            'results': results, 'protected_files': len(before['files']),
            'input_files': len(input_hashes), 'protected_unchanged': unchanged,
            'inputs_unchanged': input_hashes == after_inputs,
            'downloads': 0, 'rl_fits': 0, 'trading_evaluations': 0,
            'research_pair_folds_evaluated': 0,
        })
        assert unchanged, 'Protected files changed during prerequisite checking'
    if len(results) != len(INPUTS) or any(r['exit_code'] for r in results):
        raise SystemExit('STOP: prerequisite verification failed; original log retained.')
    print('PASS all four original read-only input verifiers.')

if __name__ == '__main__':
    main()
