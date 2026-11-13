"""The user's exact historical and sealed-input verification scope."""
from check_prerequisites import ROOT, HERE, PT, INPUTS, fingerprint, sha, write_new
import os, json, collections

NAMES = [name for name, _ in INPUTS]

def inventory(directory):
    paths = {}
    for parent, dirs, files in os.walk(directory, followlinks=False):
        parent = Path(parent)
        for name in dirs[:]:
            path = parent / name
            if path.is_symlink():
                paths[str(path.relative_to(directory))] = path
                dirs.remove(name)
        for name in files:
            path = parent / name
            paths[str(path.relative_to(directory))] = path
    return paths

from pathlib import Path

def check_sealed_inputs():
    sector = PT / 'universe_sector_expansion_v1'
    historical = json.loads((sector / 'audit/protected_before.json').read_text())['files']
    reports = []
    for name in NAMES:
        directory = PT / name
        complete = json.loads((directory / 'complete.json').read_text())
        assert complete['status'] == 'complete'
        frozen=HERE/'audit/sealed_input_inventories.json'
        if frozen.exists():
            expected_complete=next(r['complete_sha256'] for r in json.loads(frozen.read_text()) if r['input']==name)
            assert sha(directory/'complete.json')==expected_complete,('Input completion seal changed',name)
        for key, value in complete.items():
            if key.endswith('_sha256'):
                path = directory / (key.removesuffix('_sha256') + '.json')
                assert path.is_file() and sha(path) == value, ('Input seal mismatch', name, key)
        if name == 'universe_sector_expansion_v1':
            expected = json.loads((directory / 'manifest.json').read_text())['files'].copy()
        else:
            prefix = str(directory.relative_to(ROOT)) + '/'
            expected = {p[len(prefix):]: v for p, v in historical.items() if p.startswith(prefix)}
        if (directory / 'manifest.json').exists():
            manifest = json.loads((directory / 'manifest.json').read_text())['files']
            if isinstance(manifest, list):
                expected.update({r['path']: dict(sha256=r['sha256'], size=r['bytes'], mtime_ns=r['mtime_ns']) for r in manifest})
            else:
                expected.update(manifest)
            expected.setdefault('manifest.json', fingerprint(directory / 'manifest.json'))
        expected.setdefault('complete.json', fingerprint(directory / 'complete.json'))
        actual = inventory(directory)
        assert set(actual) == set(expected), ('Sealed input file inventory changed', name,
                                              'added', sorted(set(actual) - set(expected)),
                                              'missing', sorted(set(expected) - set(actual)))
        changed_finder = []
        for rel, value in expected.items():
            got = fingerprint(actual[rel])
            if Path(rel).name == '.DS_Store':
                if got != value:
                    changed_finder.append(str(actual[rel].relative_to(ROOT)))
            else:
                assert got == value, ('Sealed input file changed', name, rel)
        reports.append(dict(input=name, files=len(actual), added_files=0,
                            changed_finder_paths=changed_finder,
                            complete_sha256=sha(directory / 'complete.json')))
    return reports

def historical_snapshot(directory):
    expected = json.loads((directory / 'audit/protected_before.json').read_text())
    files = {}
    changed_finder = []
    for rel, previous in expected['files'].items():
        path = ROOT / rel
        if path.name == '.DS_Store':
            if not path.exists() or fingerprint(path) != previous:
                changed_finder.append(rel)
            files[rel] = previous
        else:
            assert path.exists() or path.is_symlink(), ('Missing historical file', rel)
            files[rel] = fingerprint(path)
            assert files[rel] == previous, ('Changed historical file', rel)
    from check_prerequisites import protected
    # Additions are enumerated by paths, without reading or hashing their bytes.
    current_paths = []
    for parent, dirs, names in os.walk(ROOT, followlinks=False):
        parent = Path(parent)
        dirs[:] = [n for n in dirs if n not in ('.git', '.venv', 'node_modules')
                   and parent / n not in (directory, ROOT / 'experiments/turning_point_detection')]
        for n in dirs[:]:
            p = parent / n
            if p.is_symlink():
                current_paths.append(str(p.relative_to(ROOT))); dirs.remove(n)
        current_paths.extend(str((parent / n).relative_to(ROOT)) for n in names)
    added = [p for p in current_paths if p not in expected['files']]
    print('Scoped historical check:', directory.name, len(files), 'exact historical files;',
          'changed Finder paths:', changed_finder, flush=True)
    print('Later additions by workspace top level:', dict(collections.Counter(p.split('/')[0] for p in added)), flush=True)
    print('Later additions by experiment directory:', dict(collections.Counter('/'.join(p.split('/')[:3]) for p in added)), flush=True)
    return {**expected, 'files': files,
            'directories': [p for p in expected['directories'] if (ROOT / p).is_dir()]}
