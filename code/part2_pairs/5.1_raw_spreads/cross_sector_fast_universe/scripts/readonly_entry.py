"""Execute an unchanged input verifier under a process-wide read-only guard."""
import sys
sys.dont_write_bytecode = True
sys.modules['readline'] = None
import os
import runpy
from pathlib import Path

def guard(event, args):
    if event in ('socket.connect', 'socket.getaddrinfo', 'subprocess.Popen', 'os.system'):
        raise PermissionError('Read-only prerequisite verifier: network and child execution forbidden')
    if event == 'open':
        path, mode, flags = args
        if (isinstance(mode, str) and any(c in mode for c in 'wax+')) or flags & (
                os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            if path != os.devnull:
                raise PermissionError('Read-only prerequisite verifier: write forbidden')
    if event in ('os.mkdir', 'os.remove', 'os.rmdir', 'os.rename', 'os.chmod',
                 'os.utime', 'os.link', 'os.symlink', 'sqlite3.connect'):
        raise PermissionError('Read-only prerequisite verifier: mutation forbidden')

if __name__ == '__main__':
    target = Path(sys.argv[1]).resolve()
    sys.argv = [str(target), *sys.argv[2:]]
    sys.path.insert(0, str(target.parent))
    sys.addaudithook(guard)
    runpy.run_path(str(target), run_name='__main__')
