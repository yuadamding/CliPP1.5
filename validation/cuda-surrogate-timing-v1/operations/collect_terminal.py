"""Wait on the exact owned terminal receipt, reconcile LSF, import hash-bound evidence."""
from pathlib import Path
import base64
import hashlib
import json
import sys

sys.path.insert(0, '/storage/CliPP2/scripts')
import stage_sharedcn_lsf20 as s

s.HOP=s.SSH[:2]+['-F','/Users/yding4/.ssh/config.d/codex-chain/company-seadragon.conf']+s.SSH[2:]+['-o','HostName=ldragon5','-o','HostKeyAlias=seadragon','seadragon']


def collect(parent):
    plan=json.loads((parent/'PREPARED.json').read_bytes())
    launch=json.loads((parent/'launch.json').read_bytes())
    accepted,=[json.loads(line.split('=',1)[1]) for line in launch['stdout'].splitlines() if line.startswith('RECEIPT=')]
    code=f'''from pathlib import Path
import base64,hashlib,json,subprocess,time
root=Path({plan['remote_root']!r})
a=json.loads((root/'receipts/accepted.json').read_bytes())
assert a['job_id']=={accepted['job_id']!r} and a['job_name']=={plan['job_name']!r}
deadline=time.monotonic()+259200
terminal=root/'receipts/terminal.json'
while not terminal.is_file():
 assert time.monotonic()<deadline,'No terminal receipt within bounded queue/run observation budget'
 time.sleep(30)
t=json.loads(terminal.read_bytes())
assert t['job_id']==a['job_id'] and t['plan_sha256']==a['plan_sha256']
deadline=time.monotonic()+120
while True:
 r=subprocess.run(['bjobs','-UF',a['job_id']],capture_output=True,text=True,timeout=30)
 assert r.returncode==0 and 'User <yding4>' in r.stdout and a['job_name'] in r.stdout
 if 'Status <DONE>' in r.stdout or 'Status <EXIT>' in r.stdout: break
 assert time.monotonic()<deadline,'Terminal application receipt awaits scheduler reconciliation'
 time.sleep(5)
paths=[*sorted((root/'receipts').glob('*.json')),*sorted((root/'results').glob('*.json')),*sorted((root/'case-logs').glob('*.err'))]
for folder in sorted((root/'results').glob('*.trials')):
 paths.extend(p for p in folder.rglob('*') if p.is_file())
assert len(paths)==len(set(paths))
assert all(p.is_file() and not p.is_symlink() and p.suffix in ('.json','.jsonl','.npz','.tsv','.err') for p in paths)
assert sum(p.stat().st_size for p in paths)<180000000,'Evidence import exceeds reviewed bound; preserve remote originals'
files={{str(p.relative_to(root)):dict(data=base64.b64encode(p.read_bytes()).decode(),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths}}
print('RECEIPT='+json.dumps(dict(scheduler=r.stdout,terminal=t,files=files)))
'''
    result=s.call(parent,'terminal-evidence-import','set -euo pipefail\n'+s.PYTHON+" - <<'PYREMOTE'\n"+code+'\nPYREMOTE\n')
    inventory={}
    for name,item in result['files'].items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        destination=parent/'imported'/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        with destination.open('xb') as stream:
            stream.write(base64.b64decode(item['data'],validate=True))
        assert hashlib.sha256(destination.read_bytes()).hexdigest()==item['sha256']
        inventory[name]=dict(sha256=item['sha256'],bytes=destination.stat().st_size)
    with (parent/'IMPORT_MANIFEST.json').open('x') as stream:
        json.dump(dict(files=inventory,scheduler=result['scheduler'],terminal=result['terminal']),stream,indent=2)
        stream.write('\n')
    return result['terminal']


if __name__=='__main__':
    print(json.dumps(collect(Path(sys.argv[1]).resolve())))
