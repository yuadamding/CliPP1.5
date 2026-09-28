"""Execute the frozen fit unchanged, recording peak memory in its own process."""
from pathlib import Path
import json
import runpy
import sys
import torch

telemetry, helper = Path(sys.argv[1]), sys.argv[2]
sys.argv = sys.argv[2:]
sys.path.insert(0, str(Path(helper).resolve().parent))
try:
    runpy.run_path(helper, run_name='__main__')
except SystemExit as exc:
    if exc.code not in (0, None):
        raise
torch.cuda.synchronize()
properties = torch.cuda.get_device_properties(0)
record = dict(gpu=properties.name, total_bytes=properties.total_memory,
              peak_allocated_bytes=torch.cuda.max_memory_allocated(),
              peak_reserved_bytes=torch.cuda.max_memory_reserved())
with telemetry.open('x') as stream:
    json.dump(record, stream, indent=2)
if record['peak_reserved_bytes'] > .85*record['total_bytes']:
    raise RuntimeError('Observed allocation exceeds 85% physical-memory guard')
