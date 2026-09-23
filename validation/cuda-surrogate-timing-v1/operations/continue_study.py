"""Serial, frozen timing study after six independent CUDA coverage passes."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

import prepare
from collect_terminal import collect

STUDY = Path(__file__).resolve().parent
TEMPLATE = STUDY/'below64-b'
COVERAGE = STUDY.parent/'cuda-surrogate-review-20260923-v1'
STAGES = [('below_one',64,'below64-b','below64-a'),
          ('mixed_support',64,'mixed64-c','mixed64-d'),
          ('below_one',256,'below256-d','below256-b'),
          ('mixed_support',256,'mixed256-e','mixed256-e'),
          ('below_one',512,'below512-f','below512-c'),
          ('mixed_support',512,'mixed512-g','mixed512-f')]


def write_new(path,value):
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())


def progress(**value):
    value.update(utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid())
    temporary=STUDY/'controller-progress.writing'
    with temporary.open('w') as stream:
        json.dump(value,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,STUDY/'controller-progress.json')


def validate_import(parent):
    plan=json.loads((parent/'PREPARED.json').read_bytes())
    manifest=json.loads((parent/'IMPORT_MANIFEST.json').read_bytes())
    for name,entry in manifest['files'].items():
        assert prepare.digest(parent/'imported'/name)==entry['sha256']
    accepted=json.loads((parent/'imported/receipts/accepted.json').read_bytes())
    terminal=manifest['terminal']
    assert accepted['job_id']==terminal['job_id']
    assert accepted['plan_sha256']==terminal['plan_sha256']==prepare.digest(parent/'sealed'/plan['run_id']/'PREPARED.json')
    assert f"Job <{accepted['job_id']}>" in manifest['scheduler']
    assert 'Status <DONE>' in manifest['scheduler'] or 'Status <EXIT>' in manifest['scheduler']
    task,=plan['tasks']
    path=parent/'imported/results'/(task['name']+'.json')
    record=json.loads(path.read_bytes())
    for name,digest in record['artifacts'].items():assert prepare.digest(path.parent/name)==digest
    assert record['source']['source_sha256']==plan['source_sha256']['current']
    return record,terminal,prepare.digest(path)


def next_stage(family,nodes,name,prior):
    subprocess.run([sys.executable,'-B',str(STUDY/'preflight.py'),name],check=True)
    previous=COVERAGE/prior
    plan=json.loads((previous/'PREPARED.json').read_bytes());task,=plan['tasks']
    path=previous/'imported/results'/(task['name']+'.json')
    args=['--fixture',family,'--nodes',str(nodes),'--expected-source-sha256',plan['source_sha256']['current'],
          '--coverage',plan['remote_root']+'/results/'+path.name,'--coverage-sha256',prepare.digest(path),
          '--timeout-seconds','3000']
    task=dict(name=name.split('-')[0],source_role='current',script='time_surrogate_cuda.py',timeout_seconds=3300,args=args)
    parent=STUDY/name
    write_new(parent/'tasks.json',[task]);prepare.prepare(parent,[task],60,source_template=TEMPLATE)
    return parent


def main():
    lock=(STUDY/'controller.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    names=('continue_study.py','collect_terminal.py','prepare.py','preflight.py','launch_prepared.py','worker.py','launch.py')
    helpers={name:prepare.digest(STUDY/name) for name in names}
    proc=Path('/proc')/str(os.getpid())
    write_new(STUDY/'controller-started.json',dict(pid=os.getpid(),start_ticks=(proc/'stat').read_text().rsplit(')',1)[1].split()[19],
        cmdline_sha256=prepare.digest(proc/'cmdline'),utc=datetime.now(timezone.utc).isoformat(),helpers=helpers,
        source_sha256=json.loads((TEMPLATE/'PREPARED.json').read_bytes())['source_sha256']['current'],
        max_nonterminal_research_jobs=1,stages=STAGES))
    completed=[]
    try:
        # The first canary is immutable. Its existing collector owns import;
        # wait for exact terminal reconciliation before admitting replacement.
        deadline=time.monotonic()+21600
        progress(phase='waiting_first_canary_terminal',completed=completed)
        while not (STUDY/'below64-a/IMPORT_MANIFEST.json').exists():
            assert time.monotonic()<deadline,'First canary terminal unresolved'
            time.sleep(10)
        initial,terminal,digest=validate_import(STUDY/'below64-a')
        assert initial['status']=='failed' and initial['error']=='Public timing phases are missing, negative, or nonfinite',initial.get('error')
        write_new(STUDY/'first-canary-disposition.json',dict(attempt='below64-a',job_id=terminal['job_id'],receipt_sha256=digest,
            reason=initial['error'],scope='Timing receipt phase omission; preserved full failure. Corrected helper frozen separately in below64-b; numerical package unchanged.'))
        for family,nodes,name,prior in STAGES:
            assert all(prepare.digest(STUDY/n)==d for n,d in helpers.items()),'Frozen controller helpers changed'
            parent=TEMPLATE if name==TEMPLATE.name else next_stage(family,nodes,name,prior)
            subprocess.run([sys.executable,'-B',str(STUDY/'launch_prepared.py'),str(parent)],check=True)
            progress(phase='waiting_for_exact_terminal',current_attempt=name,completed=completed)
            collect(parent)
            record,terminal,digest=validate_import(parent)
            completed.append(dict(attempt=name,fixture=family,nodes=nodes,status=record['status'],job_id=terminal['job_id'],
                receipt_sha256=digest,timing_summary=record.get('timing_summary'),error=record.get('error')))
            assert record['status']=='passed',record.get('error')
        write_new(STUDY/'controller-complete.json',dict(completed=completed,all_six_timing_stages_passed=True,
            production_default_promoted=False))
        progress(phase='complete',completed=completed)
    except BaseException as error:
        write_new(STUDY/'controller-failed.json',dict(error_type=type(error).__name__,error=str(error),
            traceback=traceback.format_exc(),completed=completed))
        progress(phase='halted',completed=completed,error=str(error))
        raise


if __name__=='__main__':main()
