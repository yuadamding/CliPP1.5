import csv
import ctypes
import json






import math


import os




import signal


import subprocess




import time




from common import (read, sha)


def check_publication_canary(root, task, output_base=None):
    expected = root/'payload/expected'/task['key']
    contract = read(expected/'EXPECTATION.json')
    name = 'mixture_mutation_clusters.tsv'
    if sha(expected/name) != contract['table_sha256']:
        raise ValueError('Changed publication reference')
    output_base = root/'outputs' if output_base is None else output_base
    target = output_base/task['key']/'guarded/00000'
    def table(path):
        with path.open() as stream:
            rows = list(csv.DictReader(stream, delimiter='\t'))
        result = {r['mutation_id']: r for r in rows}
        if len(result) != len(rows) or not result:
            raise ValueError('Empty or duplicate publication IDs')
        return result
    a, b = table(expected/name), table(target/name)
    if set(a) != set(b):
        raise ValueError('Publication population mismatch')
    deltas = [abs(float(a[n]['mixture_ccf'])-float(b[n]['mixture_ccf'])) for n in a]
    delta = max(deltas)
    if (not all(math.isfinite(d) for d in deltas) or delta > contract['ccf_atol'] or
            any(a[n]['cluster_label'] != b[n]['cluster_label'] or
                a[n]['multiplicity'] != b[n]['multiplicity'] for n in a)):
        raise ValueError('CPU/CUDA publication mismatch')
    receipt = read(target/'EXPERIMENT.json')
    if (receipt['status'] != contract['status'] or
            receipt['original_baseline_preserved'] != contract['original_baseline_preserved'] or
            receipt['structural_decision']['selected_family'] != contract['selected_family']):
        raise ValueError('CPU/CUDA selection identity mismatch')
    return dict(status='passed', max_ccf_delta=delta, mutations=len(a),
                expectation_sha256=sha(expected/'EXPECTATION.json'))


def run_child(argv, env, log, deadline):
    # Reap orphaned compiler helpers in our own child session. Waiting only for
    # the Python leader can leave its descendants charged to an LSF allocation.
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    if libc.prctl(37, ctypes.byref(previous), 0, 0, 0) or libc.prctl(36, 1, 0, 0, 0):
        raise OSError(ctypes.get_errno(), 'Cannot enable child subreaper')
    try:
        return _run_child(argv, env, log, deadline)
    finally:
        if libc.prctl(36, previous.value, 0, 0, 0):
            raise OSError(ctypes.get_errno(), 'Cannot restore child subreaper')


def _finish_group(proc, stream, grace=2.0):
    """Bounded cleanup of the session we created, including an exited leader."""
    started = time.monotonic()
    signals = []

    def send(signum):
        try:
            os.killpg(proc.pid, signum)
            signals.append(signum)
        except ProcessLookupError:
            pass

    def gone():
        # Popen alone owns the leader's exit status; only reap adopted children
        # after Popen has collected it. A live member keeps its group ID bound.
        if proc.poll() is None:
            return False
        while True:
            try:
                pid, _ = os.waitpid(-proc.pid, os.WNOHANG)
            except ChildProcessError:
                break
            if pid == 0:
                break
        try:
            os.killpg(proc.pid, 0)
        except ProcessLookupError:
            return True
        return False

    if not gone():
        send(signal.SIGTERM)
        until = time.monotonic()+grace
        while not gone() and time.monotonic() < until:
            time.sleep(.02)
        if not gone():
            send(signal.SIGKILL)
            until = time.monotonic()+5
            while not gone() and time.monotonic() < until:
                time.sleep(.02)
    absent = gone()
    stream.write(json.dumps(dict(child_group_cleanup=dict(pgid=proc.pid,
        leader_returncode=proc.returncode, signals=signals, group_absent=absent,
        elapsed_seconds=time.monotonic()-started)))+'\n')
    stream.flush()
    if not absent:
        raise RuntimeError('Child process group survived bounded cleanup')


def _run_child(argv, env, log, deadline):
    with log.open('x') as stream:
        proc = subprocess.Popen(argv, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                start_new_session=True)
        def terminate(signum, frame):
            raise InterruptedError(f'Received signal {signum}')
        old = {s: signal.signal(s, terminate) for s in (signal.SIGTERM, signal.SIGINT)}
        try:
            code = proc.wait(timeout=max(1, deadline-time.monotonic()))
        finally:
            try:
                for s in old:
                    signal.signal(s, signal.SIG_IGN)
                _finish_group(proc, stream)
            finally:
                for s, handler in old.items():
                    signal.signal(s, handler)
    if code:
        raise RuntimeError(f'Child exited {code}: {log.name}')
