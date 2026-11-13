"""Invoke unchanged read-only input verifiers; recognize only registered TAV1 absence."""
import common as own
from common import *
import inspect,traceback
from symlink_scope import canonical_mtimes
from conditional_scope import install
if __name__=='__main__':
 name,script=sys.argv[1:];run=PT/name;path=run/script;own.guard(readonly=True)
 approved=None
 if name=='trend_anchor_trading_v1':
  from protection import registry_rows
  approved=registry_rows()
 # Never let the new task's generic common module shadow the sealed one.
 sys.modules.pop('common',None);sys.path[:0]=[str(run/'scripts'),str(run),str(path.parent)]
 try:
  with canonical_mtimes(ROOT),install(emit=True):
   s=importlib.util.spec_from_file_location('_rms_input_verifier',path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
   if hasattr(m,'verify'):
    params=inspect.signature(m.verify).parameters
    kw={'check_current':False} if 'check_current' in params else {'current':False} if 'current' in params else {}
    result=m.verify(**kw)
   else:result=m.main()
 except FileNotFoundError as error:
  if approved is None or not error.filename:raise
  relative=str(Path(error.filename).relative_to(ROOT));assert relative in approved,('Original TAV1 failed outside registered absence',relative)
  traceback.print_exc();print('EXPECTED_REGISTERED_ABSENCE',relative,flush=True)
 else:
  if approved is not None:print('TAV1 original passed; all manifest-covered paths were available',flush=True)
 print('INPUT_ARTIFACT_CHECK_COMPLETED',name,flush=True)
