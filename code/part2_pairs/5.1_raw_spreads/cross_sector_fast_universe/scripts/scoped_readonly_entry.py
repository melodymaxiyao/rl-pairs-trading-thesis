"""User-approved historical-snapshot adapter with exact sealed inventories.

All source verifier checks execute unchanged. Only their historical workspace
snapshot IO is restricted to pre-existing objects; Finder metadata is historical
scope only. Current whole-workspace protection is independently hashed.
"""
import sys
sys.dont_write_bytecode = True
sys.modules['readline'] = None
import json
import importlib
import runpy
from pathlib import Path
from readonly_entry import guard
from verification_scope import check_sealed_inputs, historical_snapshot

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[2]

def main():
    approval = HERE / 'addendum_003_verification_scope.json'
    assert approval.exists(), 'User-approved verification-scope addendum is required'
    assert json.loads(approval.read_text())['user_approved'] is True
    target = Path(sys.argv[1]).resolve()
    directory = target.parents[1] if target.parent.name == 'scripts' else target.parent
    sys.argv = [str(target), *sys.argv[2:]]
    sys.path.insert(0, str(target.parent))
    sys.addaudithook(guard)
    print('Exact sealed input inventories:', check_sealed_inputs(), flush=True)
    if directory.name == 'universe_sector_expansion_v1':
        module = importlib.import_module('sector_io')
        module.snapshot = lambda: historical_snapshot(directory)
    elif directory.name == 'learner_noise_robustness_v1':
        module = importlib.import_module('s2io')
        module.snapshot = lambda: historical_snapshot(directory)
    runpy.run_path(str(target), run_name='__main__')

if __name__ == '__main__':
    main()
