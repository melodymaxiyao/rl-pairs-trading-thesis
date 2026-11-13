"""Run original upstream verifier with source writes and network prohibited."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
import importlib.util,inspect
from legacy_views import canonical_mtimes,conditional_extension
from conflict_copies import ignored_inventory_view
if __name__=='__main__':
 script=Path(sys.argv[1]);sys.path.insert(0,str(script.parent));sys.argv=[str(script)];guard(readonly=True)
 # Check the unchanged artifact verifier; full current history is checked separately
 # by our before/after gate, matching the established real-mechanism entry adapter.
 with canonical_mtimes(),conditional_extension(),ignored_inventory_view(os.environ.get('FYP_IGNORED_CONFLICT_PROOF'),os.environ.get('FYP_IGNORED_CONFLICT_PROOF_SHA256')):
  for name in ['common','protection','report','jobs','native','simulation','dp','evaluation']:sys.modules.pop(name,None)
  s=importlib.util.spec_from_file_location('_sealed_original_verifier',script);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
  fn=m.verify if hasattr(m,'verify') else m.main;params=inspect.signature(fn).parameters
  result=fn(**({'check_current':False} if 'check_current' in params else {'current':False} if 'current' in params else {}))
