"""Run the unchanged prune-aware verifier under the explicit symlink addendum."""
from common import *
import runpy
from symlink_scope import canonical_mtimes
from conditional_scope import install
if __name__=='__main__':
 guard(readonly=True);sys.path.insert(0,str(PRUNE));script=PRUNE/'verify_tav1_prune.py';sys.argv=[str(script)]
 with canonical_mtimes(ROOT),install(emit=True):runpy.run_path(str(script),run_name='__main__')
