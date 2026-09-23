from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import re
import subprocess
import sys
from clipp1d.api import source_provenance
w=Path(__file__).resolve().parent
repo=w.parent.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def files(folder):return {str(p.relative_to(repo)):sha(p) for p in sorted((repo/folder).rglob('*.py'))}
source=source_provenance()
assert source['source_sha256']=='8627daf96b3b72be689a756445dfb60a7798a03d8bdd8d896f837eb1f11184c2'
record=dict(source=source,test_files=files('tests'),benchmark_files=files('benchmarks'),started_utc=datetime.now(timezone.utc).isoformat())
log=w/'pytest-final.log'
with log.open('x') as f:
 subprocess.run([sys.executable,'-m','pytest','-q'],cwd=repo,stdout=f,stderr=subprocess.STDOUT,check=True)
assert source_provenance()['source_sha256']==source['source_sha256']
assert record['test_files']==files('tests') and record['benchmark_files']==files('benchmarks')
m=re.findall(r'(?m)^(\d+) passed(?:, (\d+) warnings?)? in ([0-9.]+)s',log.read_text());assert len(m)==1
count,warnings,seconds=m[0]
record['tests']=dict(passed=int(count),failed=0,skipped=0,warnings=int(warnings or 0),seconds=float(seconds),log_sha256=sha(log),evidence='Complete redirected pytest output; source and test/helper inventories matched before and after execution')
checks={'ruff':[str(Path(sys.executable).with_name('ruff')),'check','src','benchmarks','tests'],'compileall':[sys.executable,'-m','compileall','-q','src','benchmarks','tests'],'diff_check':['git','diff','--check']}
record['static_checks']={}
for name,command in checks.items():
 subprocess.run(command,cwd=repo,check=True)
 record['static_checks'][name]='passed'
record.update(scope='Local CPU unit/reference checks; no allocated-CUDA claim',recorded_utc=datetime.now(timezone.utc).isoformat())
with (w/'LOCAL_VALIDATION.final.json').open('x') as f:json.dump(record,f,indent=2)
print(json.dumps(record['tests']))
