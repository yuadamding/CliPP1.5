"""Suspended creation and UID-bound, attach-only Kubernetes supervision."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from common import identity, now, plan, read, sha, verify, write


class Lifecycle:
    def __init__(self, root, family, control=None):
        self.root, self.family = root, family
        self.p = plan(root)
        self.control = control
        manifest_root = root/'payload' if control is None else control
        if control is not None:
            if control.resolve() != control or not control.is_relative_to(root/'control'):
                raise ValueError('Unexpected lifecycle generation path')
            generation = read(control/'GENERATION.json')
            if generation['plan_sha256'] != sha(root/'payload/RUN_PLAN.json'):
                raise ValueError('Different generation plan')
            for name, digest in generation['inventory'].items():
                if Path(name).is_absolute() or '..' in Path(name).parts or (control/name).is_symlink() or sha(control/name) != digest:
                    raise ValueError('Changed control generation')
        self.manifest_path = manifest_root/(family+'.manifest.json')
        self.manifest = read(self.manifest_path)
        self.name = self.manifest['metadata']['name']
        self.worker_e = root/'evidence'/family
        self.e = self.worker_e if control is None else root/'evidence'/(family+'-'+control.name)
        self.env = dict(os.environ, KUBECONFIG='/rsrch8/home/bcb/yding4/.kube/config', PYTHONDONTWRITEBYTECODE='1')
        self.k = ['/risapps/noarch/kubectl/1.28.4/bin/kubectl', '--request-timeout=20s',
            '--context', 'yding4_yn-gpu-workload@kubernetes-admin@kubernetes', '-n', 'yn-gpu-workload']

    def kube(self, args, value=None):
        if args[0] == 'create' and sys.argv[2] != 'launch':
            raise ValueError('Supervisor is attach-only')
        result = subprocess.run(self.k+args, input=None if value is None else json.dumps(value),
                                env=self.env, text=True, capture_output=True, timeout=40)
        if result.returncode:
            raise RuntimeError(result.stderr[-3000:])
        return json.loads(result.stdout) if result.stdout.strip() else None

    def job(self):
        return self.kube(['get', 'job', self.name, '--ignore-not-found', '-o', 'json'])

    def pods(self, uid):
        pods = self.kube(['get', 'pods', '-l', f'job-name={self.name},controller-uid={uid}', '-o', 'json'])['items']
        if not all(any(o.get('uid') == uid and o.get('controller') for o in p['metadata'].get('ownerReferences', [])) for p in pods):
            raise ValueError('Foreign Pod owner')
        return pods

    def audit(self, actual):
        m = self.manifest
        if (actual['metadata']['name'] != self.name or
                actual['metadata']['annotations']['seadragon_run_id'] != self.name or
                actual['metadata']['namespace'] != 'yn-gpu-workload'):
            raise ValueError('Wrong admitted Job identity')
        for key in ('completions', 'parallelism', 'completionMode', 'backoffLimit',
                    'activeDeadlineSeconds', 'ttlSecondsAfterFinished', 'suspend'):
            if actual['spec'][key] != m['spec'][key]:
                raise ValueError('Changed admitted '+key)
        expected = m['spec']['template']['spec']
        spec = actual['spec']['template']['spec']
        for key in ('containers', 'volumes', 'securityContext', 'automountServiceAccountToken',
                    'restartPolicy', 'schedulerName', 'priorityClassName', 'nodeSelector', 'affinity'):
            if spec.get(key) != expected.get(key):
                raise ValueError('Changed admitted Pod '+key)
        for key in ('hostNetwork', 'hostPID', 'hostIPC', 'initContainers', 'ephemeralContainers'):
            if spec.get(key):
                raise ValueError('Unexpected '+key)
        if actual['spec']['template']['metadata']['labels'].get('runai/queue') != 'yding4-yn-gpu-workload-queue':
            raise ValueError('Wrong institutional queue')

    def validate(self, current, bound):
        if not current or current['metadata']['uid'] != bound['metadata']['uid'] or current['metadata']['annotations']['seadragon_run_id'] != self.name:
            raise ValueError('Wrong live Job identity')
        for key in ('selector', 'template', 'completions', 'completionMode', 'backoffLimit',
                    'activeDeadlineSeconds', 'ttlSecondsAfterFinished'):
            if current['spec'].get(key) != bound['spec'].get(key):
                raise ValueError('Changed live spec '+key)
        if current['spec']['parallelism'] not in (1, self.p['pools'][self.family]['workers']):
            raise ValueError('Unauthorized concurrency')
        return current

    def patch(self, bound, changes):
        cur = self.validate(self.job(), bound)
        if changes.get('suspend') is False and (not cur['spec']['suspend'] or self.pods(bound['metadata']['uid'])):
            raise ValueError('Activation requires suspended, empty Job')
        fields = [('/metadata/uid', bound['metadata']['uid']),
            ('/metadata/resourceVersion', cur['metadata']['resourceVersion']),
            ('/metadata/annotations/seadragon_run_id', self.name),
            ('/spec/suspend', cur['spec']['suspend']), ('/spec/parallelism', cur['spec']['parallelism']),
            ('/spec/selector', bound['spec']['selector']), ('/spec/template', bound['spec']['template'])]
        ops = [dict(op='test', path=k, value=v) for k, v in fields]
        ops += [dict(op='replace', path='/spec/'+k, value=v) for k, v in changes.items()]
        return self.kube(['patch', 'job', self.name, '--type=json', '-p', json.dumps(ops), '-o', 'json'])

    def capture(self, pod):
        path = self.e/('pod-'+pod['metadata']['uid']+'.terminal.json')
        if path.exists():
            return
        try:
            result = subprocess.run(self.k+['logs', pod['metadata']['name'], '-c', 'main',
                '--tail=100', '--limit-bytes=20000'], env=self.env, capture_output=True, text=True, timeout=40)
            logs = dict(code=result.returncode, stdout=result.stdout, stderr=result.stderr)
        except subprocess.TimeoutExpired:
            logs = dict(error='Bounded log read timed out')
        write(path, dict(pod=pod, logs=logs, utc=now()))

    def cleanup(self, bound):
        cur = self.job()
        uid = bound['metadata']['uid']
        if cur:
            self.validate(cur, bound)
            for pod in self.pods(uid):
                self.capture(pod)
            intent = self.e/'delete-intent.json'
            if not intent.exists():
                options = dict(apiVersion='v1', kind='DeleteOptions', propagationPolicy='Foreground',
                    preconditions=dict(uid=uid, resourceVersion=cur['metadata']['resourceVersion']))
                write(intent, options)
                response = self.kube(['delete', '--raw', '/apis/batch/v1/namespaces/yn-gpu-workload/jobs/'+self.name, '-f', '-'], options)
                write(self.e/'delete-response.json', response)
        end = time.monotonic()+55
        while time.monotonic() < end:
            if self.job() is None and not self.pods(uid):
                write(self.e/'cleanup.json', dict(uid=uid, job_absent=True, pods_absent=True, utc=now()))
                return
            time.sleep(2)
        write(self.e/'cleanup-pending.json', dict(uid=uid, utc=now()))

    def supervise(self):
        lock = (self.e/'supervisor.lock').open('x')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for _ in range(30):
            if (self.e/'process.json').exists():
                break
            time.sleep(1)
        if identity(os.getpid()) != read(self.e/'process.json')['identity']:
            raise ValueError('Unbound supervisor')
        if sha(__file__) != read(self.e/'spawn-intent.json')['source_sha256']:
            raise ValueError('Changed bound supervisor')
        bound = read(self.e/'binding.json')['observations'][-1]
        def interrupted(signum, frame):
            raise InterruptedError('Supervisor stopped')
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, interrupted)
        try:
            write(self.e/'activation-intent.json', dict(uid=bound['metadata']['uid'], utc=now()))
            write(self.e/'activation.json', self.patch(bound, dict(suspend=False)))
            write(self.e/'launch-success.json', dict(name=self.name, uid=bound['metadata']['uid'],
                  initial_parallelism=1, authorized_parallelism=self.p['pools'][self.family]['workers'], utc=now()))
            deadline = time.monotonic()+self.p['deadline_seconds']
            scaled = False
            while time.monotonic() < deadline:
                cur = self.validate(self.job(), bound)
                owned = self.pods(bound['metadata']['uid'])
                for pod in owned:
                    spec = pod['spec']
                    expected = bound['spec']['template']['spec']
                    for k in ('containers', 'volumes', 'securityContext', 'nodeSelector', 'affinity',
                              'automountServiceAccountToken', 'restartPolicy', 'schedulerName'):
                        if spec.get(k) != expected.get(k):
                            raise ValueError('Different admitted Pod '+k)
                    uid = pod['metadata']['uid']
                    path = self.worker_e/('pod-'+uid+'.admitted.json')
                    statuses = pod.get('status', {}).get('containerStatuses', [])
                    if statuses and statuses[0].get('imageID'):
                        if not statuses[0]['imageID'].endswith(self.p['image'].split('@')[1]):
                            raise ValueError('Wrong executed image')
                        if not path.exists():
                            write(path, dict(pod_uid=uid, job_uid=bound['metadata']['uid'], image=self.p['image'],
                                  image_id=statuses[0]['imageID'], pod=pod, utc=now()))
                    if pod.get('status', {}).get('phase') in ('Failed', 'Succeeded'):
                        self.capture(pod)
                suffix = '' if self.control is None else '-'+self.control.name
                qualified = self.root/'qualification'/(self.family+suffix)/'QUALIFIED.json'
                if not scaled and qualified.exists():
                    q = read(qualified)
                    if (q['status'] != 'passed' or q['owner']['job_uid'] != bound['metadata']['uid'] or
                            q['inventory_sha256'] != sha(self.root/'payload/INVENTORY.json') or
                            q['plan_sha256'] != sha(self.root/'payload/RUN_PLAN.json')):
                        raise ValueError('Different qualification identity')
                    write(self.e/'scale-intent.json', dict(qualification_sha256=sha(qualified), utc=now()))
                    response = self.patch(bound, dict(parallelism=self.p['pools'][self.family]['workers']))
                    write(self.e/'scaled.json', dict(job=response, qualification_sha256=sha(qualified), utc=now()))
                    scaled = True
                conditions = [c['type'] for c in cur.get('status', {}).get('conditions', []) if c.get('status') == 'True']
                phases = [p.get('status', {}).get('phase') for p in owned]
                write(self.e/'PROGRESS.json', dict(utc=now(), job_uid=bound['metadata']['uid'],
                    parallelism=cur['spec']['parallelism'], running=phases.count('Running'),
                    pending=phases.count('Pending'), failed=phases.count('Failed'), succeeded=phases.count('Succeeded'), scaled=scaled), replace=True)
                if 'Failed' in conditions or 'Complete' in conditions:
                    write(self.e/'TERMINAL.json', dict(job=cur, pods=owned, utc=now(),
                          pool_completed='Complete' in conditions, whole_study_complete=False))
                    return
                if (self.root/'STOP').exists():
                    raise InterruptedError('Study stop requested')
                time.sleep(10)
            raise TimeoutError('Bounded lifecycle deadline')
        except BaseException as exc:
            write(self.e/'failure.json', dict(error=repr(exc), traceback=traceback.format_exc(), utc=now()))
            raise
        finally:
            self.cleanup(bound)

    def launch(self):
        verify(self.root, full=True)
        if self.job() is not None:
            raise ValueError('Job already exists; reconcile')
        dry = self.kube(['create', '-f', '-', '--dry-run=server', '-o', 'json'], self.manifest)
        write(self.e/'dryrun.json', dry)
        self.audit(dry)
        write(self.e/'arm.json', dict(manifest_sha256=sha(self.manifest_path), utc=now()))
        created = self.kube(['create', '-f', '-', '-o', 'json'], self.manifest)
        write(self.e/'create.json', created)
        child = None
        try:
            self.audit(created)
            if created['spec']['template']['spec'] != dry['spec']['template']['spec']:
                raise ValueError('Admission changed after dry run')
            observations = []
            for _ in range(2):
                cur = self.validate(self.job(), created)
                if not cur['spec']['suspend'] or self.pods(created['metadata']['uid']):
                    raise ValueError('Unstable suspended Job')
                observations.append(cur)
            if observations[0]['spec'] != observations[1]['spec']:
                raise ValueError('Unstable suspended spec')
            write(self.e/'binding.json', dict(observations=observations))
            write(self.e/'spawn-intent.json', dict(source_sha256=sha(__file__), utc=now()))
            with (self.e/'supervisor.log').open('x') as log:
                argv = [sys.executable, '-B', __file__, str(self.root), 'supervise', self.family]
                if self.control is not None:
                    argv.append(str(self.control))
                child = subprocess.Popen(argv,
                    cwd=self.root, env=self.env, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
            time.sleep(.2)
            write(self.e/'process.json', dict(identity=identity(child.pid), utc=now()))
            for _ in range(50):
                if (self.e/'launch-success.json').exists():
                    return read(self.e/'launch-success.json')
                if child.poll() is not None:
                    raise RuntimeError('Supervisor exited; inspect retained evidence')
                time.sleep(1)
            raise TimeoutError('Handshake pending: do not duplicate')
        except BaseException:
            if child is None:
                self.cleanup(created)
            else:
                write(self.e/'attach-required.json', dict(uid=created['metadata']['uid'], pid=child.pid, utc=now()))
            raise


if __name__ == '__main__':
    lifecycle = Lifecycle(Path(sys.argv[1]).resolve(), sys.argv[3], Path(sys.argv[4]).resolve() if len(sys.argv) > 4 else None)
    if sys.argv[2] == 'launch':
        print('RECEIPT='+json.dumps(lifecycle.launch()))
    elif sys.argv[2] == 'supervise':
        lifecycle.supervise()
    else:
        raise ValueError('Unknown lifecycle action')
