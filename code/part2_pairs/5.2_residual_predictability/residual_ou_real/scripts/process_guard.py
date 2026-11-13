"""Exact ownership proof for this coordinator's multiprocessing helper."""
import re
def own_resource_tracker(pid,ppid,command,owner_pid,registered_pid,python):
 if registered_pid is None or pid!=registered_pid or ppid!=owner_pid or pid==owner_pid:return False
 prefix=python+' '
 if not command.startswith(prefix):return False
 rest=command[len(prefix):];parts=rest.split('-c ',1)
 if len(parts)!=2:return False
 flags=parts[0].split()
 if any(flag not in {'-B','-I','-E','-s','-S','-u'} for flag in flags):return False
 code=parts[1].strip()
 return re.fullmatch(r'from multiprocessing\.resource_tracker import main;main\([0-9]+\)',code) is not None
def process_rows(text):
 for line in text.splitlines():
  parts=line.split(None,2)
  if len(parts)==3 and parts[0].isdigit() and parts[1].isdigit():yield int(parts[0]),int(parts[1]),parts[2],line
