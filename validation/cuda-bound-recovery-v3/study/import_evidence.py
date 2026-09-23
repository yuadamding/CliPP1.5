"""Hash/readback import of one exact terminal qualification's compact evidence."""
from pathlib import Path
import hashlib
import json
import sys

sys.path.insert(0, "/storage/CliPP2/scripts")
import stage_sharedcn_lsf20 as s

s.HOP = s.SSH[:2] + ["-F", "/Users/yding4/.ssh/config.d/codex-chain/company-seadragon.conf"] + s.SSH[2:] + ["-o", "HostName=ldragon4", "-o", "HostKeyAlias=seadragon", "seadragon"]
parent = Path(sys.argv[1]).resolve()
plan = json.loads((parent / "PREPARED.json").read_bytes())
launch = json.loads((parent / "launch.json").read_bytes())
accepted, = [json.loads(line.split("=", 1)[1]) for line in launch["stdout"].splitlines() if line.startswith("RECEIPT=")]
code = f'''from pathlib import Path
import json,hashlib,subprocess
root=Path({plan['remote_root']!r})
a=json.loads((root/'receipts/accepted.json').read_bytes())
assert a['job_id']=={accepted['job_id']!r} and a['job_name']=={plan['job_name']!r}
terminal=json.loads((root/'receipts/terminal.json').read_bytes())
assert terminal['job_id']==a['job_id']
r=subprocess.run(['bjobs','-UF',a['job_id']],capture_output=True,text=True)
assert r.returncode==0 and 'User <yding4>' in r.stdout and a['job_name'] in r.stdout
assert 'Status <DONE>' in r.stdout or 'Status <EXIT>' in r.stdout, r.stdout
paths=[*sorted((root/'receipts').glob('*.json')), *sorted((root/'results').glob('*.json')), *sorted((root/'results').glob('*.jsonl')), *sorted((root/'case-logs').glob('*.err'))]
for folder in sorted((root/'results').glob('*.artifacts')):
 paths.extend(p for p in folder.rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.tsv') and 'trace' not in p.name)
assert len(paths)==len(set(paths))
assert all(p.is_file() and not p.is_symlink() for p in paths)
assert sum(p.stat().st_size for p in paths)<180000000
files={{str(p.relative_to(root)):{{'text':p.read_text(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}} for p in paths}}
print('RECEIPT='+json.dumps(dict(scheduler=r.stdout,files=files,terminal=terminal)))
'''
r = s.call(parent, "terminal-evidence-import", "set -euo pipefail\n" + s.PYTHON + " - <<'PYREMOTE'\n" + code + "\nPYREMOTE\n")
inventory = {}
for name, item in r["files"].items():
    destination = parent / "imported" / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as stream:
        stream.write(item["text"])
    assert hashlib.sha256(destination.read_bytes()).hexdigest() == item["sha256"], name
    inventory[name] = dict(sha256=item["sha256"], bytes=destination.stat().st_size)
with (parent / "IMPORT_MANIFEST.json").open("x") as stream:
    json.dump(dict(files=inventory, scheduler=r["scheduler"], terminal=r["terminal"]), stream, indent=2, sort_keys=True)
    stream.write("\n")
print(json.dumps(dict(job_id=accepted["job_id"], passed=r["terminal"]["passed"],
                     files=len(inventory), bytes=sum(v["bytes"] for v in inventory.values()))))
