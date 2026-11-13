from mio import *
fingerprint=fp
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

