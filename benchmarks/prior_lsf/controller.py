"""Receipt-bound rolling LSF controller, one study slot, qualification before prespecified stages."""
from collections import Counter
import fcntl
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
from common import command, digest, identity, now, plan, read, verify_files, write

GOOD = {'qualified','validated_complete','validated_incomplete','scientific_failure','resource_failure','resource_timeout','execution_failure','infrastructure_failure'}


def field(pattern, raw):
    values=re.findall(pattern,raw,re.M)
    assert len(values)==1,(pattern,values)
    return values[0]


def verify_admission(raw, a, worker_command, memory, minutes):
    assert field(r'^Job <(\d+)>',raw)==a['job_id']
    assert field(r'Job Name <([^>]+)>',raw)==a['job_name']
    assert field(r'User <([^>]+)>',raw)=='yding4'
    assert field(r'Queue <([^>]+)>',raw)=='egpu'
    assert field(r'Command <([^>]+)>',raw)==worker_command
    assert field(r', (\d+) Task\(s\), Requested Resources',raw)=='2'
    assert field(r'Requested GPU <([^>]+)>',raw)=='num=1:mode=exclusive_process:gmodel=NVIDIAL40'
    assert float(field(r'^ MEMLIMIT\s*\n\s*(\d+(?:\.\d+)?) G\b',raw))==memory
    assert float(field(r'^ RUNLIMIT\s*\n\s*(\d+(?:\.\d+)?) min\b',raw))==minutes
    assert field(r'Requested Resources <([^>]+)>',raw)==f'rusage[mem={memory}] span[hosts=1]'
    state=field(r'Status <([^>]+)>',raw)
    assert state in ('PEND','PSUSP'),'Leave unexpected admission unreleased'
    return state


def submit(root,case,p):
    key=case['key']
    folder=root/'receipts'/key
    folder.mkdir() # An existing intent/unknown/accepted attempt is never repeated.
    name=p['job_prefix']+key
    memory,minutes=case['memory_gb'],case['wall_minutes']
    assert minutes <= p['queue_hard_minutes']
    worker_command=f'{p["python"]} -B {root}/worker.py worker {root} {key}'
    argv=['bsub','-H','-q','egpu','-J',name,'-n','2','-M',str(memory),'-R',f'rusage[mem={memory}] span[hosts=1]',
          '-gpu','num=1:mode=exclusive_process:gmodel=NVIDIAL40','-W',str(minutes),'-cwd',str(root),
          '-oo',str(root/'lsf-logs'/f'{key}.%J.out'),'-eo',str(root/'lsf-logs'/f'{key}.%J.err'),worker_command]
    write(folder/'submit-intent.json',dict(argv=argv,key=key,plan_sha256=digest(root/'PREPARED.json'),utc=now()))
    response=command(argv,120)
    write(folder/'submit-response.json',response)
    ids=re.findall(r'Job <(\d+)> is submitted',response['stdout'])
    if len(ids)==1:
        a=dict(key=key,job_id=ids[0],job_name=name,plan_sha256=digest(root/'PREPARED.json'),initially_held=True)
        write(folder/'accepted.json',a)
    assert response['code']==0 and len(ids)==1,'Ambiguous submission; stop refills, never duplicate'
    observations=[]
    deadline=time.monotonic()+300
    try:
        while True:
            q=command(['bjobs','-UF',a['job_id']],30)
            assert q['code']==0,q
            state=verify_admission(q['stdout'],a,worker_command,memory,minutes)
            observation=dict(job_id=a['job_id'],state=state,raw=q['stdout'],utc=now())
            observations.append(observation)
            if state=='PSUSP':
                write(folder/'admitted.json',observation)
                break
            assert time.monotonic()<deadline,'Leave held; never repeat submission'
            time.sleep(5)
    finally:
        write(folder/'admission-observations.json',observations)
    write(folder/'release-intent.json',dict(job_id=a['job_id'],utc=now()))
    response=command(['bresume',a['job_id']])
    write(folder/'release-response.json',response)
    assert response['code']==0 and a['job_id'] in response['stdout'],'Uncertain release; stop refills'
    return a


def log_proof(root,a):
    path=root/'lsf-logs'/f'{a["key"]}.{a["job_id"]}.out'
    if not path.exists():
        return None
    # LSF appends its report after payload output. Restrict reading to the tail.
    with path.open('rb') as stream:
        stream.seek(max(0,path.stat().st_size-65536))
        raw=stream.read().decode(errors='replace')
    if f'Subject: Job {a["job_id"]}: <{a["job_name"]}>' not in raw or not re.search(r'Job <'+re.escape(a['job_name'])+r'> was submitted .*by user <yding4>',raw):
        return None
    if 'Results reported at ' not in raw or 'Terminated at ' not in raw:
        return None
    state='DONE' if 'Successfully completed.' in raw else 'EXIT' if re.search(r'Exited with exit code|Exited with signal termination|Exited by signal|TERM_(?:MEMLIMIT|RUNLIMIT)',raw) else None
    return dict(state=state,raw=raw,sha256=digest(path)) if state else None


def reconcile(root,active):
    if not active:
        return {},[]
    q=command(['bjobs','-a','-noheader','-o','jobid user stat job_name:200',*active],60)
    states={job:'UNKNOWN' for job in active}
    for line in q['stdout'].splitlines():
        values=line.split()
        if not values:
            continue
        assert len(values)==4,values
        job,user,state,name=values
        assert job in active and user=='yding4' and name==active[job]['job_name'],'Foreign scheduler identity'
        states[job]=state
    done=[]
    for job,a in active.items():
        prior = root/'receipts'/a['key']/'reconciled.json'
        if prior.exists():
            prior_record=read(prior)
            assert prior_record['accepted']==a and prior_record['scheduler_state'] in ('DONE','EXIT')
            states[job]=prior_record['scheduler_state']
            done.append((job,prior_record['outcome']))
            continue
        proof=log_proof(root,a)
        state=states[job]
        if state=='UNKNOWN' and proof:
            state=states[job]=proof['state']
        if state not in ('DONE','EXIT'):
            continue
        folder=root/'receipts'/a['key']
        terminal=folder/'terminal.json'
        outcome=read(terminal) if terminal.exists() else dict(status='infrastructure_failure',error='No application terminal receipt')
        if terminal.exists():
            assert outcome['job_id']==job and outcome['key']==a['key']
        if outcome['status'].startswith('validated_'):
            path=root/'results'/a['key']/'validated.json'
            assert digest(path)==outcome['validated_sha256']
            v=read(path)
            for name,sha in v['output_sha256'].items():
                assert not Path(name).is_absolute() and '..' not in Path(name).parts and digest(path.parent/name)==sha
        elif outcome['status']=='qualified':
            path=root/'results'/a['key']/'QUALIFICATION.json'
            assert digest(path)==outcome['receipt_sha256'] and read(path)['status']=='passed'
        if state=='EXIT':
            reason=next((reason for reason in ('TERM_MEMLIMIT','TERM_RUNLIMIT') if proof and reason in proof['raw']),None)
            outcome=dict(outcome,application_status=outcome['status'],status='resource_failure' if reason else outcome['status'] if outcome['status'] in ('output_validation_failure','setup_failure','scientific_failure','resource_failure','resource_timeout','execution_failure','infrastructure_failure') else 'infrastructure_failure',scheduler_reason=reason)
        record=dict(accepted=a,scheduler_state=state,outcome=outcome,log_proof=proof,
                    query=q,utc=now())
        write(folder/'reconciled.json',record)
        done.append((job,outcome))
    return states,done


def recovered_acceptances(root,active,finished):
    for folder in (root/'receipts').iterdir():
        path=folder/'accepted.json'
        if folder.is_dir() and path.exists():
            a=read(path)
            if a['key'] not in finished:
                active.setdefault(a['job_id'],a)
    return active


def admit_panel(root, canary, status, max_submitted):
    qualification = root/'results/qualification/QUALIFICATION.json'
    terminal = read(root/'receipts/qualification/terminal.json')
    assert terminal['status'] == 'qualified'
    assert digest(qualification) == terminal['receipt_sha256']
    assert read(qualification)['status'] == 'passed'
    assert status.startswith('validated_')
    write(root/'receipts/panel-admitted.json', dict(
        qualification_sha256=digest(qualification), canary_key=canary,
        canary_sha256=digest(root/'results'/canary/'validated.json'),
        canary_search_status=status, max_submitted=max_submitted, utc=now()))


def controller(root):
    p=plan(root)
    lock=(root/'receipts/controller.lock').open('x')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for _ in range(30):
        if (root/'receipts/controller-process.json').exists():
            break
        time.sleep(1)
    assert identity(os.getpid())==read(root/'receipts/controller-process.json')['identity']
    assert digest(__file__)==read(root/'receipts/controller-intent.json')['runner_sha256']
    verify_files(root,full=True)
    units=command(['lsadmin','showconf','lim'])
    assert re.findall(r'^[ \t]*LSF_UNIT_FOR_LIMITS[ \t]*=[ \t]*(\S+)',units['stdout'],re.M)==['GB']
    queue=command(['bqueues','-l','egpu'])
    hard=float(field(r'RUNLIMIT[^\n]*\n\s*(\d+(?:\.\d+)?) min',queue['stdout'].split('MAXIMUM LIMITS:',1)[1]))
    assert hard==p['queue_hard_minutes'] and hard>=360
    params=command(['bparams','-a'])
    assert re.search(r'RESOURCE_RESERVE_PER_TASK\s*=\s*N',params['stdout'])
    write(root/'receipts/resource-contract.json',dict(units=units,queue=queue,reservation_per_task=False,
          requested_memory='16 GB per normal study job; 32 GB for N4000 cost fixtures; hard limit equals reservation',cpu_slots=2,gpu_count=1,hard_minutes=hard,utc=now()))
    write(root/'receipts/controller-ready.json',dict(identity=identity(os.getpid()),utc=now()))
    imported={}
    cases=read(root/'cases.json')
    assert cases==sorted(cases,key=lambda c:c['priority'])
    queue_cases=[c for c in cases if c["key"] not in imported]
    active,finished={},dict(imported)
    phase='qualification'
    halted=None
    states={}
    gate_started=time.monotonic()
    from datetime import datetime,timezone
    remaining=(datetime.fromisoformat(p["deadline_utc"])-datetime.now(timezone.utc)).total_seconds()
    assert remaining>0,"Inherited run deadline exhausted"
    deadline=gate_started+remaining
    qualification=dict(key='qualification',memory_gb=16,wall_minutes=120)
    while True:
        try:
            if time.monotonic()>deadline or (phase in ('qualification','canary') and time.monotonic()-gate_started>72*3600):
                raise TimeoutError('Controller/gate budget exhausted; drain without retries')
            states,done=reconcile(root,active)
            for job,outcome in done:
                a=active.pop(job)
                finished[a['key']]=outcome['status']
                if outcome['status'] not in GOOD:
                    halted=halted or outcome
            if not halted:
                if phase=='qualification':
                    if 'qualification' not in finished and not active:
                        a=submit(root,qualification,p)
                        active[a['job_id']]=a
                    elif finished.get('qualification')=='qualified':
                        phase='canary'
                        gate_started=time.monotonic()
                    elif 'qualification' in finished:
                        raise RuntimeError('Allocated fixture failed; do not launch cohort')
                if phase=='canary':
                    canary=p['canary_key']
                    if canary not in finished and not active:
                        assert queue_cases[0]['key']==canary
                        a=submit(root,queue_cases.pop(0),p)
                        active[a['job_id']]=a
                    elif finished.get(canary,'').startswith('validated_'):
                        admit_panel(root, canary, finished[canary], p['max_submitted'])
                        phase='panel'
                    elif canary in finished:
                        raise RuntimeError('Real-case canary failed; do not launch panel')
                if phase=='panel':
                    # Select only from pilot diagnostics after every development
                    # attempt finishes; accuracy/truth are absent on this host.
                    selection=root/'receipts/factorial-cases.json'
                    dev=[c for c in cases if c['stage']=='development']
                    if not active and all(c['key'] in finished for c in dev) and not selection.exists():
                        assert all(finished[c['key']].startswith('validated_') for c in dev), 'Development publication loss; stop before confirmation'
                        sys.path.insert(0,str(root/'source/src'))
                        sys.path.insert(0,str(root/'source/benchmarks'))
                        from prior_perturbation_diagnostics import select_factorial
                        chosen=select_factorial(root/'results',root/'datasets/development/FIT_MANIFEST.json',root/'receipts/factorial-selection.json')
                        chosen_ids={c['case_id'] for c in chosen}
                        dynamic=[dict(c,key='factorial-'+c['key'],stage='factorial',mode='factorial',
                                      baseline_key=c['key'],priority=[1,c['priority'][1],i])
                                 for i,c in enumerate(dev) if c['case_id'] in chosen_ids]
                        assert len(dynamic)==12
                        write(selection,dict(cases=dynamic,selection_sha256=digest(root/'receipts/factorial-selection.json')))
                        cases.extend(dynamic)
                        queue_cases=sorted(queue_cases+dynamic,key=lambda c:c['priority'])
                    while queue_cases and len(active)<p['max_submitted']:
                        if (root/'receipts/DRAIN.json').exists():
                            halted=dict(reason='explicit drain request')
                            break
                        a=submit(root,queue_cases.pop(0),p)
                        assert a['job_id'] not in active
                        active[a['job_id']]=a
        except BaseException as error:
            if halted is None:
                halted=dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
                write(root/'receipts/controller-halt.json',dict(halted,utc=now()))
            recovered_acceptances(root,active,finished)
        unresolved = [d.name for d in (root/'receipts').iterdir() if d.is_dir() and (d/'submit-intent.json').exists() and not (d/'accepted.json').exists()]
        assert len(active)+len(unresolved)<=p['max_submitted']
        counts=Counter(value for key,value in finished.items() if key!='qualification')
        progress=dict(phase='halted_draining' if halted and active else 'halted' if halted else phase,
            active=active,states=states,finished_counts=dict(counts),planned=len(cases),final_planned=835,
            finished_cases=sum(counts.values()),imported_cases=len(imported),new_finished_cases=sum(counts.values())-len(imported),remaining_to_submit=len(queue_cases),
            max_submitted=p['max_submitted'],unknown_submissions=unresolved,halt_reason=halted,updated_utc=now())
        write(root/'receipts/controller-progress.json',progress,replace=True)
        if halted and not active:
            write(root/'receipts/controller-terminal.json',dict(progress,complete=False))
            return 1
        if phase=='panel' and not active and not queue_cases:
            write(root/'results/summary.json',dict(counts=progress['finished_counts'],planned=len(cases),source_commit=p['source_commit'],source_sha256=p['source_sha256']))
            write(root/'receipts/controller-terminal.json',dict(progress,complete=True))
            return 0
        time.sleep(60)


def launch(root):
    p=plan(root)
    assert sys.executable==p['python']
    verify_files(root,full=True)
    write(root/'receipts/controller-intent.json',dict(runner_sha256=digest(__file__),plan_sha256=digest(root/'PREPARED.json'),utc=now()))
    argv=[sys.executable,'-B',str(root/'controller.py'),'control',str(root)]
    with (root/'receipts/controller.log').open('xb') as log:
        process=subprocess.Popen(argv,cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    owned=identity(process.pid)
    assert owned['pgrp']==owned['session']==process.pid and owned['tty']==0
    write(root/'receipts/controller-process.json',dict(identity=owned,argv=argv,utc=now()))
    deadline=time.monotonic()+180
    while not (root/'receipts/controller-ready.json').exists():
        assert process.poll() is None,'Controller exited; inspect exact log and receipts; never relaunch blindly'
        assert time.monotonic()<deadline,'Controller startup uncertain; reconcile exact PID'
        time.sleep(2)
    assert identity(process.pid)==owned
    print('RECEIPT='+__import__('json').dumps(dict(controller=owned,ready=read(root/'receipts/controller-ready.json'),max_submitted=p['max_submitted'])))


if __name__=='__main__':
    action,root=sys.argv[1:]
    if action=='launch':
        launch(Path(root))
    else:
        sys.exit(controller(Path(root)))
