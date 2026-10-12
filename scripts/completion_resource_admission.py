"""Record a real pre-load admission without importing any neural library."""
from pathlib import Path
from datetime import datetime, timezone
import json
import sys
import psutil

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from completion_kit.common import require_resources

path=ROOT/sys.argv[1]
assert path.is_relative_to(ROOT) and not path.exists()
memory=psutil.virtual_memory()
record={'utc':datetime.now(timezone.utc).isoformat(),'required_ram_gib':8.0,'measured_available_ram_gib':memory.available/2**30,
        'cpu_threads_limit':2,'neural_library_imported':False,'foreign_process_interrupted':False}
try:
    record['admitted_available_ram_gib']=require_resources(8.0)
    record['status']='admitted'
    code=0
except RuntimeError as error:
    if not str(error).startswith('RAM admission refused:'):
        raise
    record['status']='refused_resource_guard'
    record['reason']=str(error)
    code=2
path.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(record))
raise SystemExit(code)
