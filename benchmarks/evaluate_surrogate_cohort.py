"""Paired empirical evaluation of the existing outer-surrogate strategies.

One original tumor per GPU job. Each strategy executes its unchanged default
path independently. This is a numerical/accuracy study, not a throughput test.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import sys
from time import perf_counter
import traceback
import zipfile

import numpy as np
import torch

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks import qualify_surrogate_cuda as study
from benchmarks import time_surrogate_cuda as timing
from benchmarks.cohort_staging import stage_case_input, verify_staged_input
from benchmarks.compare_clipp2 import metrics, read_rows
from clipp1d.cuda.policy import QualificationError
from clipp1d.cuda_api import _export, _publish, _validate_result
from clipp1d.model import compile_model
from clipp1d.policy import Policy

common, mixed = study.common, study.mixed
SCHEMA = 'clipp1d.cuda.surrogate_empirical.v1'


def fixed_lambda_comparisons(candidate, control):
    """Compare only qualified raw objectives at the exact same positive lambda."""
    paths = []
    for records in (candidate, control):
        rows = {}
        for row in records:
            lam = row['lambda_value']
            mixed.require(np.isfinite(lam) and lam >= 0 and lam not in rows,
                          'Invalid or duplicate literal lambda')
            rows[lam] = row
        paths.append(rows)
    a, b = paths
    shared = []
    for lam in sorted(a.keys() & b.keys()):
        if lam == 0:
            continue
        x, y = a[lam], b[lam]
        comparable = all(r.get('raw_status') == 'qualified' for r in (x, y))
        row = dict(lambda_value=lam, qualified_objectives_comparable=comparable,
                   raw_objective_difference=x['raw_objective']-y['raw_objective'] if comparable else None)
        for name, r in (('candidate', x), ('control', y)):
            row[name] = {k: r.get(k) for k in ('raw_status', 'refit_status', 'raw_objective',
                'search_complete', 'starts_attempted', 'starts_qualified', 'starts_unresolved',
                'qp_admm_iterations', 'qp_calls', 'seconds', 'score', 'clusters', 'error')}
        shared.append(row)
    return dict(common_positive_literal_lambdas=shared,
                candidate_only_lambdas=sorted(a.keys()-b.keys()),
                control_only_lambdas=sorted(b.keys()-a.keys()),
                zero_penalty_excluded=True,
                scope='Exact same positive penalty and graph; independent continuation trajectories; raw qualification and complete-start coverage remain separate')


def accuracy(output, truth_path, case):
    rows = read_rows(output/'mutation_clusters.tsv')
    calls = read_rows(output/'mutation_multiplicity.tsv')
    truth = read_rows(truth_path)
    mixed.require(common.sha(truth_path) == case['truth_sha256'], 'Truth changed')
    ids = sorted(rows)
    mixed.require(len(ids) == case['retained_mutations'] and set(rows) == set(calls), 'Output population changed')
    import hashlib
    mixed.require(hashlib.sha256(json.dumps(ids).encode()).hexdigest() == case['retained_ids_sha256'],
                  'Output IDs differ from the frozen retained population')
    mixed.require(all(r['tumor_id'] == case['tumor_id'] and r['sample_id'] == case['sample_id']
                      for r in rows.values()), 'Output tumor/sample identity changed')
    scored = {key: value for key, value in rows.items() if key in truth}
    mixed.require(set(rows)-set(truth) == set(case['truth_unmatched_retained']) and
                  len(scored)/len(rows) == case['truth_coverage'], 'Truth coverage changed')
    result = {}
    for mode in ('raw', 'refitted'):
        multiplicity = {key: dict(row, multiplicity_call=row[mode+'_multiplicity_call'])
                        for key, row in calls.items()}
        result[mode] = metrics(scored, truth, multiplicity, mode+'_ccf')
    result.update(true_smf=sum(float(truth[i]['true_ccf']) < 1-1e-12 for i in scored)/len(scored),
                  estimated_smf=sum(row['cluster_label'] != '0' for row in scored.values())/len(scored),
                  truth_matched=len(scored), truth_unmatched=len(rows)-len(scored),
                  smf_definition='Fraction outside closest-to-one designated cluster 0; truth CCF < 1-1e-12',
                  scope='Accuracy on retained truth-matched mutations, separate from numerical qualification')
    return result


@torch.no_grad()
def trial(out, name, data, host, device, kernels, case, truth_path):
    journal = mixed.Journal(out)
    record = dict(strategy=name, status='running', publication_validated=False,
                  diagnostic_observer=False, candidate_tracer=False, artifacts={})
    began = perf_counter()
    calls = study.StartLog()
    try:
        torch.cuda.reset_peak_memory_stats(device)
        model, upload_seconds = common.timed(device, lambda: common.upload(host, device, kernels))
        # Use the original strategy wrapper without observers. Retain the ledger
        # even if the entire path raises before a fit can be returned.
        with study.strategy(name) as calls:
            fitted, seconds = common.timed(device, lambda: timing.selection.fit_tensor_model(model))
        summary = common.fit_summary(fitted)
        summary['pilot_and_graph'].pop('weights', None)
        journal.artifact('fit-summary.json', summary)
        record.update(summary=summary, fit_seconds=seconds)
        phases = dict(input_preparation_seconds=0., device_upload_and_compile_seconds=upload_seconds)
        phases.update({k: fitted.timings[k] for k in ('pilot_seconds', 'graph_build_seconds',
            'path_seconds', 'refit_seconds', 'stage_integrity_seconds')})
        source = dict(study.source_provenance(), backend='cuda', input_sha256=data.input_sha256,
            max_major_cn=4, clonal_constraint=False, clonal_label_rule='nearest_to_one_l2_v1',
            policy=asdict(study.CudaPolicy()), numerical_stages=fitted.timings, phase_seconds=phases,
            input_preparation_scope='Original staged input prepared once before paired trials',
            outer_surrogate_policy=name, research_execution=True,
            production_default_outer_surrogate_policy=study.CONTROL,
            research_driver_sha256=common.sha(Path(__file__)))
        result = _export(fitted, source, wall_started=began)
        _validate_result(result, data, fitted.records)
        output = journal.root/'public-output'
        result = _publish(result, data, output, fitted.records, wall_started=began)
        receipt = mixed.load_json(output/'run.json')
        record['measurements'] = common.validate_public_measurements(result, receipt)
        if result.search_status == 'complete':
            common.validate_public_search(result, receipt, device)
        for path in sorted(output.iterdir()):
            journal.bind(path)
        record.update(status=result.search_status, publication_validated=True,
                      accuracy=accuracy(output, truth_path, case))
    except QualificationError as error:
        record.update(status='numerical_failure', error=str(error), diagnostics=error.diagnostics,
                      traceback=traceback.format_exc())
    except BaseException as error:
        record.update(status='failed', error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
        raise
    finally:
        record.update(elapsed_seconds=perf_counter()-began, artifacts=journal.artifacts,
                      start_attempts=calls.attempts, **calls.counts())
        common.write_json(out, record)
    return record


def compare(candidate, control):
    if not all(r['publication_validated'] for r in (candidate, control)):
        return dict(comparable=False, reason='At least one strategy has no validated selected result')
    a, b = candidate['summary'], control['summary']
    for key in ('pilot_sha256', 'weights_sha256', 'weight_rule', 'gap_floor', 'normalization'):
        mixed.require(a['pilot_and_graph'][key] == b['pilot_and_graph'][key], 'Paired pilots/graph differ')
    same = a['selected_lambda'] == b['selected_lambda']
    return dict(comparable=True, graph_and_pilots_identical=True,
                candidate_search_status=a['search_status'], control_search_status=b['search_status'],
                both_searches_complete=a['search_status'] == b['search_status'] == 'complete',
                candidate_selected_lambda=a['selected_lambda'], control_selected_lambda=b['selected_lambda'],
                selected_positive_penalty=dict(candidate=a['selected_lambda'] > 0, control=b['selected_lambda'] > 0),
                same_selected_lambda=same,
                selected_raw_objective_difference=a['raw_objective']-b['raw_objective'] if same else None,
                exact_labels=a['cluster_labels'] == b['cluster_labels'],
                raw_max_difference=float(np.max(np.abs(np.array(a['raw_ccf'])-np.array(b['raw_ccf'])))),
                refit_max_difference=float(np.max(np.abs(np.array(a['refitted_ccf'])-np.array(b['refitted_ccf'])))),
                score_difference=a['score']-b['score'],
                fixed_lambda_comparisons=fixed_lambda_comparisons(a['path_records'], b['path_records']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--case-sha256', required=True)
    parser.add_argument('--payload', type=Path, required=True)
    parser.add_argument('--payload-sha256', required=True)
    parser.add_argument('--expected-source-sha256', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--timeout-seconds', type=int, default=6600)
    args = parser.parse_args()
    if args.timeout_seconds <= 0 or args.out.exists():
        raise ValueError('Require positive timeout and absent output receipt')
    work = args.out.with_suffix('.trials')
    work.mkdir()
    record = dict(schema=SCHEMA, status='running', source=study.source_provenance(),
                  driver_sha256=common.sha(Path(__file__)), helpers=study.helpers(),
                  policy=asdict(study.CudaPolicy()), started_utc=common.utc_now(), trials=[],
                  production_default_changed=False, command=sys.argv,
                  scope='Operational completion of a paired empirical observation; numerical completeness and accuracy are separate per-strategy fields; no throughput ratio')
    def timeout(signum, frame):
        raise TimeoutError('Paired empirical case wall budget exhausted')
    old_alarm = signal.signal(signal.SIGALRM, timeout)
    old_term = signal.signal(signal.SIGTERM, timeout)
    signal.alarm(args.timeout_seconds)
    try:
        mixed.require(record['source']['source_sha256'] == args.expected_source_sha256, 'Source differs')
        mixed.require(common.sha(args.case) == args.case_sha256 and common.sha(args.payload) == args.payload_sha256,
                      'Frozen case or payload differs')
        case = mixed.load_json(args.case)
        record.update(case=case, case_sha256=args.case_sha256, payload_sha256=args.payload_sha256)
        with zipfile.ZipFile(args.payload) as bundle:
            stage_case_input(work, case, bundle)
            truth_path = work/'truth.tsv'
            with truth_path.open('xb') as stream:
                stream.write(bundle.read(case['truth_member']))
        mixed.require(common.sha(truth_path) == case['truth_sha256'], 'Truth differs')
        data = verify_staged_input(work, case)
        host = compile_model(data, Policy(max_major_cn=4))
        host = host.subset(np.argsort(np.asarray(host.mutation_ids), kind='stable'))
        device = study.require_cuda(args.device)
        free, _ = torch.cuda.mem_get_info(device)
        estimate = 32*8*len(host)**2 + 64*4096*8*4
        mixed.require(estimate <= int(study.CudaPolicy().memory_fraction*free), 'Insufficient GPU workspace')
        record.update(cuda_available=True, gpu=torch.cuda.get_device_name(device),
                      estimated_workspace_bytes=estimate, torch=torch.__version__,
                      cuda_runtime=torch.version.cuda, interpreter=sys.executable, lsf_job_id=os.environ.get('LSB_JOBID'))
        kernels = study.Kernels(device, compiled=True)
        values = {}
        # Alternate which policy is first across the frozen panel; no warm speed claim.
        for name in timing.order(case['empirical_panel_index']):
            result = trial(work/(name+'.json'), name, data, host, device, kernels, case, truth_path)
            values[name] = result
            record['trials'].append(result)
            print(json.dumps(dict(strategy=name, status=result['status'], starts=result['starts_attempted'])), flush=True)
        record.update(comparison=compare(values[study.CANDIDATE], values[study.CONTROL]), status='passed')
    except BaseException as error:
        record.update(status='failed', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_alarm)
        signal.signal(signal.SIGTERM, old_term)
        record.update(finished_utc=common.utc_now(), artifacts={str(p.relative_to(args.out.parent)): common.sha(p)
                      for p in sorted(work.rglob('*')) if p.is_file()})
        common.write_json(args.out, record)
    return int(record['status'] != 'passed')


if __name__ == '__main__':
    raise SystemExit(main())
