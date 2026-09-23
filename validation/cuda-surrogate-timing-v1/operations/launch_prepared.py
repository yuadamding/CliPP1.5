"""Stage and launch one explicit sealed task; count every uncertain owner as active."""
from pathlib import Path
import json
import fcntl
import hashlib
import os
import shlex
import sys
from datetime import datetime, timezone

sys.path.insert(0, '/storage/CliPP2/scripts')
import stage_sharedcn_lsf20 as s

s.HOP = s.SSH[:2] + ['-F', '/Users/yding4/.ssh/config.d/codex-chain/company-seadragon.conf'] + s.SSH[2:] + ['-o', 'HostName=ldragon5', '-o', 'HostKeyAlias=seadragon', 'seadragon']
p = Path(sys.argv[1]).resolve()
study = Path(__file__).resolve().parent
assert p.parent == study and p.is_dir() and not p.is_symlink()
plan = json.loads((p/'PREPARED.json').read_bytes())
authority = json.loads((study/'STUDY_PLAN.json').read_bytes())
# Serialize study admission until an exact durable owner is recorded. Uncertain
# submissions retain a slot; a bare import filename cannot release one.
lock_stream = (study/'admission.lock').open('a')
fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)

def terminal_imported(parent):
    manifest_path = parent/'IMPORT_MANIFEST.json'
    if not manifest_path.exists():
        return False
    assert manifest_path.is_file() and not manifest_path.is_symlink()
    manifest = json.loads(manifest_path.read_bytes())
    def imported(name):
        path = parent/'imported'/name
        assert path.is_file() and not path.is_symlink()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['files'][name]['sha256']
        return json.loads(path.read_bytes())
    accepted = imported('receipts/accepted.json')
    terminal = imported('receipts/terminal.json')
    bound_plan = json.loads((parent/'PREPARED.json').read_bytes())
    launch = json.loads((parent/'launch.json').read_bytes())
    owner, = [json.loads(line.split('=', 1)[1]) for line in launch['stdout'].splitlines() if line.startswith('RECEIPT=')]
    assert accepted['job_id'] == terminal['job_id'] == owner['job_id']
    assert accepted['job_name'] == bound_plan['job_name']
    sealed_plan = parent/'sealed'/bound_plan['run_id']/'PREPARED.json'
    plan_sha = hashlib.sha256(sealed_plan.read_bytes()).hexdigest()
    assert accepted['plan_sha256'] == terminal['plan_sha256'] == owner['plan_sha256'] == plan_sha
    assert manifest['terminal'] == terminal
    assert ('Status <DONE>' in manifest['scheduler'] or 'Status <EXIT>' in manifest['scheduler'])
    assert f"Job <{accepted['job_id']}>" in manifest['scheduler']
    assert bound_plan['job_name'] in manifest['scheduler'] and 'User <yding4>' in manifest['scheduler']
    return True

active = []
for parent in study.iterdir():
    if not parent.is_dir():
        continue
    requested = (parent/'launch-request.json').exists() or (parent/'launch.json').exists()
    if requested and not terminal_imported(parent):
        active.append(parent.name)
assert len(active) < authority['max_nonterminal_jobs'], active
assert not (p/'launch.json').exists() and not (p/'launch-request.json').exists()
if not (p/'publication.json').exists():
    s.transfer(p)
publication = json.loads((p/'publication.json').read_bytes())
assert publication['returncode'] == 0 and '"published": true' in publication['stdout']
with (p/'launch-request.json').open('x') as stream:
    json.dump(dict(utc=datetime.now(timezone.utc).isoformat(), active_before=active,
                   remote_root=plan['remote_root'], bound_plan='PREPARED.json'), stream, indent=2)
    stream.flush()
    os.fsync(stream.fileno())
root = plan['remote_root']
s.call(p, 'launch', 'set -euo pipefail\numask 077\nexport PYTHONDONTWRITEBYTECODE=1\n' + shlex.join([s.PYTHON, '-B', root+'/launch.py', root]) + '\n')
