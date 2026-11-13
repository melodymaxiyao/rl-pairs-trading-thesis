"""Run the unchanged historical scope in its own bounded-lifetime process."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
import signal,traceback
from historical import historical
if __name__=='__main__':
 guard()
 def stop(signum,frame):
  path=os.environ.get('OU_REALSTRENGTH_STOP')
  if path:Path(path).touch(exist_ok=True)
 signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
 try:historical(sys.argv[1])
 except Pause as exc:print('Historical guard PAUSED:',str(exc),flush=True);sys.exit(130)
