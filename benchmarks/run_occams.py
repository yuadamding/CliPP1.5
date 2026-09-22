"""Run frozen OCCAMS fits only after the bound local benchmark finishes.

No remote operations, automatic retries, truth labels, or mutable source imports.
The waiting owner and each fit are separate processes with durable receipts.
"""
import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_bytes())


def write(path, value, *, replace=False):
    path = Path(path)
    target = path.with_suffix('.tmp') if replace else path
    with target.open('w' if replace else 'x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    if replace:
        os.replace(target, path)


def owner_alive(owner):
    try:
        # Names inside parentheses may contain spaces; field 22 is starttime.
        fields = Path(f"/proc/{owner['pid']}/stat").read_text().rsplit(')', 1)[1].split()
        return fields[0] != 'Z' and fields[19] == str(owner['proc_start_ticks'])
    except FileNotFoundError:
        return False


def predecessor_state(binding):
    """No elapsed-time fallback: completion, audit, and retired owners are required."""
    root = Path(binding['root'])
    if sha(root/'plan.json') != binding['plan_sha256']:
        raise ValueError('Predecessor plan changed')
    for name in ('HALTED.json', 'finalization-failure.json'):
        if (root/name).exists():
            raise ValueError(f'Predecessor requires attention: {name}')
    if not (root/'COMPLETE.json').exists():
        return 'waiting_for_cases'
    # The predecessor publishes immutable JSON with exclusive writes. Waiting for
    # both owners to exit avoids consuming a receipt while it is still being written.
    if any(owner_alive(owner) for owner in binding['owners']):
        return 'waiting_for_owners_to_exit'
    complete = load(root/'COMPLETE.json')
    expected = binding['cases']
    if complete['cases'] != expected or sum(complete['status'].values()) != expected:
        raise ValueError('Predecessor completion count mismatch')
    if sha(root/'summary-final.json') != complete['summary_sha256']:
        raise ValueError('Predecessor summary hash mismatch')
    if not (root/'final-audit.json').exists():
        return 'waiting_for_audit'
    audit = load(root/'final-audit.json')
    if (audit['status'] != 'passed' or audit['cases'] != expected or
            audit['plan_sha256'] != binding['plan_sha256'] or
            audit['terminal_status'] != complete['status'] or
            audit['summary_sha256'] != complete['summary_sha256'] or
            sha(root/'audited-case-metrics.tsv') != audit['case_metrics_sha256']):
        raise ValueError('Predecessor audit identity mismatch')
    return 'ready'


def verify_files(root, plan):
    if sys.executable != plan['python']:
        raise ValueError('Interpreter differs from frozen plan')
    for name, digest in plan['frozen_files'].items():
        if sha(root/'frozen'/name) != digest:
            raise ValueError(f'Frozen source changed: {name}')


def case_metrics(root, plan, case, directory):
    """Validate public output, and compare final CCFs on identical retained IDs."""
    import numpy as np
    from compare_clipp2 import ari, read_rows
    run = load(directory/'fit/run.json')
    assert run['schema'] == 'clipp1d.run.v4' and run['status'] == 'success'
    assert run['provenance']['source_sha256'] == plan['source']['source_sha256']
    assert run['provenance']['input_sha256'] == case['input_sha256']
    assert set(run['table_sha256']) == {
        'mutation_clusters.tsv', 'cluster_centers.tsv', 'mutation_multiplicity.tsv'}
    for name, digest in run['table_sha256'].items():
        assert sha(directory/'fit'/name) == digest
    selected = run['candidate_provenance']
    assert selected['chain_sha256'] == run['chain_sha256']
    assert selected['model_sha256'] == run['provenance']['model_sha256']
    assert selected['refit_qualified']
    assert selected['partition_sha256'] == hashlib.sha256(
        np.asarray(run['partition_cuts'], dtype=np.int64).tobytes()).hexdigest()
    assert run['selection_score'] <= selected['raw_reference']['refit_score'] + 1e-8
    if selected['candidate_family'] == 'direct_chain_partition':
        assert run['selected_lambda'] is None
        assert selected['selected_raw_certificate'] is None
        assert selected['selected_partition_certified'] is False
    raw = run['raw_diagnostics']
    assert raw['clonal_feasible'] and (raw.get('raw_branch_stationarity_qualified') or
                                     raw.get('separable_scalar_gap_qualified'))
    rows = read_rows(directory/'fit/mutation_clusters.tsv')
    mult = read_rows(directory/'fit/mutation_multiplicity.tsv')
    ids = sorted(rows)
    assert len(ids) == case['retained_mutations'] and set(ids) == set(mult)
    assert hashlib.sha256(json.dumps(ids).encode()).hexdigest() == case['retained_ids_sha256']
    labels = [rows[i]['cluster_label'] for i in ids]
    phi = np.array([float(rows[i]['refitted_ccf']) for i in ids])
    assert np.all(np.isfinite(phi)) and np.all((phi >= 0) & (phi <= 1))
    assert '0' in labels and all(p == 1 for p, label in zip(phi, labels) if label == '0')
    summary = dict(retained_mutations=len(ids), selected_k=len(set(labels)),
                   smf_all_exact_one=float(np.mean(phi != 1)),
                   smf_designated=1-labels.count('0')/len(ids),
                   search_status=run['search_status'],
                   selected_origin=selected['origin'],
                   score_gain=selected['raw_reference']['refit_score']-run['selection_score'],
                   fit_seconds=run['provenance']['elapsed_seconds'],
                   clipp2_agreement=None)
    baseline = case.get('baseline')
    if baseline:
        for name, digest in baseline['files'].items():
            assert sha(Path(baseline['directory'])/name) == digest
        two = read_rows(Path(baseline['directory'])/baseline['clusters_filename'])
        assert set(two) == set(ids), 'Retained populations differ'
        assert {r['tumor_id'] for r in two.values()} == {r['tumor_id'] for r in rows.values()}
        columns = [k for k in next(iter(two.values())) if k.startswith('phi_')]
        assert len(columns) == 1
        other_phi = np.array([float(two[i][columns[0]]) for i in ids])
        assert np.all(np.isfinite(other_phi))
        other_labels = [two[i]['cluster_label'] for i in ids]
        summary['clipp2_agreement'] = dict(
            source_commit=plan['clipp2']['source_commit'], mutations=len(ids),
            partition_ari=ari(labels, other_labels),
            ccf_mean_absolute_difference=float(np.mean(np.abs(phi-other_phi))),
            clipp2_k=len(set(other_labels)), clipp1d_k=len(set(labels)),
            clipp2_smf_all_exact_one=float(np.mean(other_phi != 1)),
            clipp2_smf_designated=1-other_labels.count('0')/len(ids),
            interpretation='Agreement between estimators; OCCAMS has no ground truth here.')
    return summary


def worker(args):
    from benchmark_chain import configure_execution
    controls = configure_execution(cpus=[args.cpu], threads=1)
    root = args.root.resolve()
    assert sha(root/'plan.json') == args.plan_sha256
    plan = load(root/'plan.json')
    case = plan['cases'][args.index]
    directory = root/'runs'/case['tumor_id']
    started = time.monotonic()
    phase = 'setup_failure'
    try:
        verify_files(root, plan)
        assert sha(case['input_path']) == case['input_sha256']
        sys.path.insert(0, str(root/'frozen/src'))
        from clipp1d import fit
        from clipp1d.api import source_provenance
        source = source_provenance()
        for key in ('source_sha256', 'python', 'numpy', 'scipy', 'interpreter'):
            assert source[key] == plan['source'][key], key
        write(directory/'startup.json', dict(pid=os.getpid(), started_utc=now(),
              proc_start_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
              plan_sha256=args.plan_sha256, input_sha256=case['input_sha256'],
              source_commit=plan['source_commit'], source=source, controls=controls))
        phase = 'fit_failure'
        fit(case['input_path'], directory/'fit', max_major_cn=4)
        phase = 'validation_failure'
        metrics = case_metrics(root, plan, case, directory)
        write(directory/'metrics.json', metrics)
        import resource
        write(directory/'terminal.json', dict(status='success', finished_utc=now(),
              elapsed_seconds=time.monotonic()-started,
              max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
              plan_sha256=args.plan_sha256, input_sha256=case['input_sha256'],
              run_sha256=sha(directory/'fit/run.json'),
              metrics_sha256=sha(directory/'metrics.json')))
        return 0
    except Exception as error:
        traceback.print_exc()
        write(directory/'terminal.json', dict(status=phase, finished_utc=now(),
              elapsed_seconds=time.monotonic()-started, error_type=type(error).__name__,
              message=str(error), plan_sha256=args.plan_sha256))
        return 2


def execute(root, plan, plan_hash, index, cpu):
    case = plan['cases'][index]
    directory = root/'runs'/case['tumor_id']
    directory.mkdir(parents=True)
    command = [plan['python'], '-B', str(Path(__file__).resolve()), 'worker',
               '--root', str(root), '--plan-sha256', plan_hash, '--index', str(index),
               '--cpu', str(cpu)]
    with (directory/'worker.log').open('xb') as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        write(directory/'process.json', dict(pid=proc.pid, cpu=cpu, command=command,
              proc_start_ticks=Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')', 1)[1].split()[19],
              started_utc=now(), plan_sha256=plan_hash))
        try:
            code = proc.wait(timeout=plan['case_timeout_seconds'])
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            code = proc.wait()
            if not (directory/'terminal.json').exists():
                write(directory/'terminal.json', dict(status='timeout', finished_utc=now(),
                      timeout_seconds=plan['case_timeout_seconds'], plan_sha256=plan_hash))
    if not (directory/'terminal.json').exists():
        write(directory/'terminal.json', dict(status='process_failure', returncode=code,
              finished_utc=now(), plan_sha256=plan_hash))
    terminal = load(directory/'terminal.json')
    assert terminal['plan_sha256'] == plan_hash
    if terminal['status'] == 'success':
        assert code == 0
        assert sha(directory/'fit/run.json') == terminal['run_sha256']
        assert sha(directory/'metrics.json') == terminal['metrics_sha256']
        for name, digest in load(directory/'fit/run.json')['table_sha256'].items():
            assert sha(directory/'fit'/name) == digest
    return dict(terminal, index=index, tumor_id=case['tumor_id'])


def summarize(root, plan, terminals, *, final=False):
    successful = [t for t in terminals if t['status'] == 'success']
    metrics = [load(root/'runs'/t['tumor_id']/'metrics.json') for t in successful]
    pairs = [m['clipp2_agreement'] for m in metrics if m['clipp2_agreement']]
    result = dict(snapshot_utc=now(), planned=len(plan['cases']), terminal=len(terminals),
                  status=dict(Counter(t['status'] for t in terminals)),
                  clipp2_matched_cases=len(pairs),
                  selected_k_counts=dict(Counter(str(m['selected_k']) for m in metrics)),
                  incomplete_searches=sum(m['search_status'] != 'complete' for m in metrics),
                  median_fit_seconds=statistics.median(m['fit_seconds'] for m in metrics)
                  if metrics else None,
                  mean_partition_agreement_ari=statistics.mean(p['partition_ari'] for p in pairs)
                  if pairs else None,
                  mean_ccf_absolute_difference=statistics.mean(
                      p['ccf_mean_absolute_difference'] for p in pairs) if pairs else None,
                  interpretation='Real tumors: agreement, not accuracy. Timings are local CPU fits.')
    write(root/('summary-final.json' if final else 'summary-current.json'), result,
          replace=not final)
    if final:
        records = []
        for terminal in terminals:
            record = dict(tumor_id=terminal['tumor_id'], status=terminal['status'])
            if terminal['status'] == 'success':
                case = plan['cases'][terminal['index']]
                directory = root/'runs'/terminal['tumor_id']
                assert sha(case['input_path']) == case['input_sha256']
                startup = load(directory/'startup.json')
                assert startup['plan_sha256'] == sha(root/'plan.json')
                assert len(startup['controls']['selected_cpus']) == 1
                assert set(startup['controls']['thread_environment'].values()) == {'1'}
                assert sha(directory/'fit/run.json') == terminal['run_sha256']
                assert sha(directory/'metrics.json') == terminal['metrics_sha256']
                metric = case_metrics(root, plan, case, directory)
                assert metric == load(directory/'metrics.json')
                record.update({k: v for k, v in metric.items() if k != 'clipp2_agreement'})
                record.update(metric['clipp2_agreement'] or {})
            records.append(record)
        with (root/'case-metrics.tsv').open('x') as stream:
            writer = csv.DictWriter(stream, sorted({k for r in records for k in r}),
                                    delimiter='\t', lineterminator='\n')
            writer.writeheader()
            writer.writerows(records)
        stamp = datetime.now(ZoneInfo('America/Chicago')).strftime('%B %d, %Y, %I:%M %p %Z')
        (root/'REPORT.md').write_text(
            f'# OCCAMS CliPP1.5 results\n\nSnapshot: {stamp}.\n\n'
            f"Completed {len(terminals)}/{len(plan['cases'])}; statuses: {result['status']}.\n\n"
            f"Validated CliPP2 comparisons: {len(pairs)}. These measure agreement, not accuracy.\n\n"
            'See [summary](summary-final.json), [case metrics](case-metrics.tsv), '
            'and [source/input plan](plan.json).\n')
    return result


def controller(args):
    root = args.root.resolve()
    assert sha(root/'plan.json') == args.plan_sha256
    plan = load(root/'plan.json')
    verify_files(root, plan)
    (root/'owner.lock').mkdir()  # Persistent claim: never silently restart an uncertain owner.
    write(root/'owner.json', dict(pid=os.getpid(), started_utc=now(),
          proc_start_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
          plan_sha256=args.plan_sha256, command=sys.argv))
    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    prior_state = None
    while not stopped:
        state = predecessor_state(plan['predecessor'])
        if state != prior_state:
            write(root/'status.json', dict(state=state, snapshot_utc=now(),
                  planned=len(plan['cases']), terminal=0, active=[]), replace=True)
            print(now(), state, flush=True)
            prior_state = state
        if state == 'ready':
            break
        time.sleep(30)
    if stopped:
        write(root/'HALTED.json', dict(reason='Stopped before fitting', snapshot_utc=now()))
        return
    verify_files(root, plan)
    assert sha(root/'plan.json') == args.plan_sha256
    assert len(plan['cpus']) == plan['workers'] == len(set(plan['cpus']))
    assert set(plan['cpus']) <= set(os.sched_getaffinity(0))
    for case in plan['cases']:
        assert sha(case['input_path']) == case['input_sha256']
    write(root/'activation.json', dict(started_utc=now(), plan_sha256=args.plan_sha256,
          predecessor_complete_sha256=sha(Path(plan['predecessor']['root'])/'COMPLETE.json'),
          predecessor_audit_sha256=sha(Path(plan['predecessor']['root'])/'final-audit.json'),
          workers=plan['workers'], cpus=plan['cpus'], threads_per_worker=1))
    available = list(plan['cpus'])
    ordered = iter(sorted(range(len(plan['cases'])), key=lambda i: (
        plan['cases'][i]['retained_mutations']**2, plan['cases'][i]['retained_mutations'], i)))
    terminals, active = [], {}
    exhausted = False
    with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
        while active or (not exhausted and not stopped):
            while available and not stopped and not exhausted:
                index = next(ordered, None)
                if index is None:
                    exhausted = True
                    break
                cpu = available.pop(0)
                future = pool.submit(execute, root, plan, args.plan_sha256, index, cpu)
                active[future] = (index, cpu)
            write(root/'status.json', dict(state='draining' if stopped else 'running',
                  snapshot_utc=now(), planned=len(plan['cases']), terminal=len(terminals),
                  status=dict(Counter(t['status'] for t in terminals)),
                  active=[dict(index=i, cpu=cpu) for i, cpu in active.values()]), replace=True)
            done, _ = wait(active, return_when=FIRST_COMPLETED, timeout=30) if active else ([], [])
            for future in done:
                index, cpu = active.pop(future)
                available.append(cpu)
                try:
                    terminal = future.result()
                    terminals.append(terminal)
                    if terminal['status'] in ('setup_failure', 'validation_failure', 'process_failure'):
                        stopped = True
                except Exception:
                    traceback.print_exc()
                    stopped = True
            if done:
                summarize(root, plan, terminals)
    if stopped or len(terminals) != len(plan['cases']):
        write(root/'HALTED.json', dict(snapshot_utc=now(), terminal=len(terminals),
              reason='Admission stopped; inspect case receipts. No automatic retry.'))
        return
    # Apply controls before the final independent public-output audit imports NumPy.
    from benchmark_chain import configure_execution
    configure_execution(cpus=[plan['cpus'][0]], threads=1)
    verify_files(root, plan)
    result = summarize(root, plan, terminals, final=True)
    write(root/'COMPLETE.json', dict(finished_utc=now(), cases=len(terminals),
          status=result['status'], plan_sha256=args.plan_sha256,
          summary_sha256=sha(root/'summary-final.json'),
          metrics_sha256=sha(root/'case-metrics.tsv'), public_outputs_audited=True))
    write(root/'status.json', dict(state='complete', **result), replace=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['controller', 'worker'])
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--index', type=int)
    parser.add_argument('--cpu', type=int)
    args = parser.parse_args()
    if args.mode == 'worker':
        return worker(args)
    try:
        controller(args)
    except Exception as error:
        traceback.print_exc()
        # A duplicate invoker must never write failure state into the live owner's run.
        owner = args.root/'owner.json'
        if owner.exists() and load(owner)['pid'] == os.getpid():
            write(args.root/'controller-failure.json', dict(finished_utc=now(),
                  error_type=type(error).__name__, message=str(error)))
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
