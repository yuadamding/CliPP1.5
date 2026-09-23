"""One bounded receipt observer per accepted job; no scheduler polling."""
from pathlib import Path
import json
import shlex
import subprocess
import sys

sys.path.insert(0, "/storage/CliPP2/scripts")
import stage_sharedcn_lsf20 as s

s.HOP = s.SSH[:2] + ["-F", "/Users/yding4/.ssh/config.d/codex-chain/company-seadragon.conf"] + s.SSH[2:] + ["-o", "HostName=ldragon4", "-o", "HostKeyAlias=seadragon", "seadragon"]
parent = Path(sys.argv[1]).resolve()
plan = json.loads((parent / "PREPARED.json").read_bytes())
launch = json.loads((parent / "launch.json").read_bytes())
accepted, = [json.loads(line.split("=", 1)[1]) for line in launch["stdout"].splitlines() if line.startswith("RECEIPT=")]
code = f'''from pathlib import Path
import json,time
root=Path({plan['remote_root']!r})
a=json.loads((root/'receipts/accepted.json').read_bytes())
assert a['job_id']=={accepted['job_id']!r} and a['job_name']=={plan['job_name']!r}
seen={{}}
deadline=time.monotonic()+{plan['resource_contract']['wall_minutes']*60+1800}
while True:
 for events in sorted((root/'results').glob('*.events.jsonl')):
  lines=events.read_text().splitlines()
  for line in lines[seen.get(events.name,0):]:
   try: value=json.loads(line)
   except json.JSONDecodeError: break
   print('EVENT='+json.dumps(dict(task=events.name,event=value)),flush=True)
   seen[events.name]=seen.get(events.name,0)+1
 terminal=root/'receipts/terminal.json'
 if terminal.is_file():
  print('TERMINAL='+terminal.read_text().replace('\\n',' '),flush=True)
  break
 if time.monotonic()>=deadline:
  print('OBSERVER_TIMEOUT',flush=True)
  break
 time.sleep(5)
'''
command = s.SSH + ["work-mac", shlex.join(s.HOP + ["bash -ls"])]
script = "set -euo pipefail\n" + s.PYTHON + " -u - <<'PYREMOTE'\n" + code + "\nPYREMOTE\n"
with (parent / "observer.log").open("x") as log:
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    proc.stdin.write(script)
    proc.stdin.close()
    for line in proc.stdout:
        log.write(line)
        log.flush()
        if line.startswith("TERMINAL=") or line.startswith("OBSERVER_TIMEOUT"):
            print(line[:3500], end="", flush=True)
        elif line.startswith("EVENT="):
            item = json.loads(line.split("=", 1)[1])
            selected = {k: v for k, v in item["event"].items() if k in (
                "kind", "nodes", "execution", "name", "stage", "status", "search_status", "seconds", "elapsed_seconds", "error_type", "error", "lambda_value", "fixture", "mode", "starts_attempted")}
            print(json.dumps(dict(task=item["task"], **selected)), flush=True)
    raise SystemExit(proc.wait())
