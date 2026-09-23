"""Paired CUDA fit timings after qualification, with all diagnostic tracing off.

One cold pair is retained separately; three warm pairs alternate execution order.
Only fit_tensor_model is timed (including its usual audits and refits). Input,
upload, extra comparisons, public export/validation and writes are outside it.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
from statistics import median
import sys
from time import perf_counter
import traceback

import numpy as np
import torch

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks import qualify_surrogate_cuda as study
from clipp1d.cuda import selection
from clipp1d.cuda_api import _export, _publish, _validate_result

common, mixed = study.common, study.mixed
SCHEMA = 'clipp1d.cuda.surrogate_timing.v1'


def coverage(args, source):
    """Verify the original diagnostic evidence; never reattribute its helper source."""
    mixed.require(common.sha(args.coverage) == args.coverage_sha256, 'Coverage receipt changed')
    prior = mixed.load_json(args.coverage)
    mixed.require(prior['schema'] == study.SCHEMA and prior['status'] == 'passed'
                  and prior['cuda_available'] is True and prior['fixture_family'] == args.fixture
                  and prior['nodes'] == args.nodes and prior['source']['source_sha256'] == source['source_sha256']
                  and prior['policy'] == asdict(study.CudaPolicy())
                  and prior['candidate_policy'] == study.CANDIDATE
                  and prior['control_policy'] == study.CONTROL, 'Coverage source/fixture/policy mismatch')
    # Driver bookkeeping may evolve; the recipe and numerical qualification helpers may not.
    for key in ('fixtures', 'common_qualifier', 'driver'):
        mixed.require(prior['helpers'][key] == study.helpers()[key], 'Coverage recipe/helper mismatch')
    candidates = [r for r in prior['trials'] if r['surrogate_policy'] == study.CANDIDATE]
    mixed.require(candidates and all(r['status'] == 'passed' and
                  r['fixture_recipe'] == mixed.fixtures.RECIPE and
                  r['full_path']['search_status'] == 'complete' and
                  r['full_path']['final_export_and_publication_qualified'] for r in candidates),
                  'Coverage lacks complete candidate starts/audits/refits/publication')
    mixed.require(bool(prior['artifacts']), 'Coverage has no bound artifacts')
    for name, sha in prior['artifacts'].items():
        mixed.require(common.sha(mixed.artifact_file(args.coverage.parent, name)) == sha,
                      'Coverage artifact changed')
    return dict(path=str(args.coverage), sha256=args.coverage_sha256,
                original_helpers=prior['helpers'], input_sha256=candidates[0]['input_sha256'])


def order(repeat):
    return (study.CONTROL, study.CANDIDATE) if repeat % 2 == 0 else (study.CANDIDATE, study.CONTROL)


def timed_fit(model, name, device):
    # No journal, observer, candidate tracer, device summaries or file writes here.
    with study.strategy(name) as calls:
        fitted, seconds = common.timed(device, lambda: selection.fit_tensor_model(model))
    return fitted, seconds, calls


def publish(fitted, data, journal, name, began, upload_seconds):
    phases = dict(input_preparation_seconds=0., device_upload_and_compile_seconds=upload_seconds)
    phases.update({key: fitted.timings[key] for key in
                   ('pilot_seconds', 'graph_build_seconds', 'path_seconds', 'refit_seconds',
                    'stage_integrity_seconds')})
    provenance = dict(study.source_provenance(), backend='cuda', input_sha256=data.input_sha256,
                      clonal_constraint=False, numerical_stages=fitted.timings, phase_seconds=phases,
                      input_preparation_scope='Fixture and kernel constructor prepared once outside each trial',
                      policy=asdict(study.CudaPolicy()), outer_surrogate_policy=name,
                      research_execution=True, production_default_outer_surrogate_policy=study.CONTROL,
                      research_driver_sha256=common.sha(Path(__file__)))
    exported = _export(fitted, provenance, wall_started=began)
    _validate_result(exported, data, fitted.records)
    exported = _publish(exported, data, journal.root/'public-output', fitted.records, wall_started=began)
    public = mixed.load_json(journal.root/'public-output/run.json')
    common.validate_public_search(exported, public, fitted.model.device)
    measurements = common.validate_public_measurements(exported, public)
    for path in sorted((journal.root/'public-output').iterdir()):
        journal.bind(path)
    return measurements


@torch.no_grad()
def trial(out, name, repeat, data, host, device, kernels):
    journal = mixed.Journal(out)
    record = dict(surrogate_policy=name, repeat=repeat, warmup=repeat == 0,
                  status='running', diagnostic_observer=False, candidate_tracer=False,
                  artifact_directory=journal.root.name, artifacts={}, publication_qualified=False)
    began = perf_counter()
    calls = study.StartLog()
    try:
        torch.cuda.reset_peak_memory_stats(device)
        model, upload_seconds = common.timed(device, lambda: common.upload(host, device, kernels))
        fitted, seconds, calls = timed_fit(model, name, device)
        # All summaries, graph rechecks and public validation happen after the stop synchronization.
        summary = common.fit_summary(fitted)
        summary['pilot_and_graph'].pop('weights', None)
        journal.artifact('fit-summary.json', summary)
        record.update(fit_seconds=seconds, search_status=fitted.search_status, summary=summary,
                      **calls.counts())
        mixed.require(calls and calls.counts()['starts_raised'] == 0, 'Unexpected start exception')
        if fitted.search_status == 'complete':
            mixed.require(all(r['qualified'] for r in calls), 'Complete fit has unresolved starts')
            record['measurements'] = publish(fitted, data, journal, name, began, upload_seconds)
            record.update(publication_qualified=True, status='passed')
        else:
            mixed.require(fitted.search_status == 'incomplete', 'Unexpected search status')
            record['status'] = 'incomplete'
    except BaseException as error:
        record.update(status='failed', error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
        raise
    finally:
        record.update(elapsed_seconds=perf_counter()-began, artifacts=journal.artifacts)
        common.write_json(out, record)
    return record


def compare(candidate, control):
    a, b = candidate['summary'], control['summary']
    for key in ('pilot_sha256', 'weights_sha256', 'weight_rule', 'gap_floor', 'normalization'):
        mixed.require(a['pilot_and_graph'][key] == b['pilot_and_graph'][key], 'Paired pilots/graph differ')
    same_lambda = a['selected_lambda'] == b['selected_lambda']
    both = candidate['publication_qualified'] and control['publication_qualified']
    return dict(repeat=candidate['repeat'], warmup=candidate['warmup'],
                complete_pair=bool(both), candidate_search_status=a['search_status'],
                control_search_status=b['search_status'], candidate_seconds=candidate['fit_seconds'],
                control_seconds=control['fit_seconds'],
                speedup_control_over_candidate=control['fit_seconds']/candidate['fit_seconds'] if both else None,
                exact_labels=a['cluster_labels'] == b['cluster_labels'],
                refit_max_difference=float(np.max(np.abs(np.array(a['refitted_ccf'])-np.array(b['refitted_ccf'])))),
                score_difference=a['score']-b['score'], same_selected_lambda=same_lambda,
                candidate_selected_lambda=a['selected_lambda'], control_selected_lambda=b['selected_lambda'],
                raw_objective_difference=a['raw_objective']-b['raw_objective'] if same_lambda else None,
                graph_and_pilots_identical=True)


def summarize(pairs):
    eligible = [p for p in pairs if not p['warmup'] and p['complete_pair']]
    return dict(complete_warm_pairs=len(eligible),
                median_paired_speedup=median(p['speedup_control_over_candidate'] for p in eligible) if eligible else None,
                candidate_median_seconds=median(p['candidate_seconds'] for p in eligible) if eligible else None,
                control_median_seconds=median(p['control_seconds'] for p in eligible) if eligible else None,
                scope='Only complete, published warm pairs; incomplete or failed work has no throughput ratio')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', choices=mixed.fixtures.FAMILIES, required=True)
    parser.add_argument('--nodes', type=int, choices=mixed.fixtures.SIZES, required=True)
    parser.add_argument('--expected-source-sha256', required=True)
    parser.add_argument('--coverage', type=Path, required=True)
    parser.add_argument('--coverage-sha256', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--timeout-seconds', type=int, default=3000)
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        raise ValueError('Timing timeout must be positive')
    if args.out.exists():
        raise FileExistsError('Never overwrite a timing receipt')
    work = args.out.with_suffix('.trials')
    work.mkdir()
    record = dict(schema=SCHEMA, status='running', source=study.source_provenance(),
                  helpers=dict(study.helpers(), timing_driver=common.sha(Path(__file__))),
                  policy=asdict(study.CudaPolicy()), fixture_family=args.fixture, nodes=args.nodes,
                  command=sys.argv, started_utc=common.utc_now(), trials=[], comparisons=[],
                  production_default_changed=False, diagnostic_observer=False, candidate_tracer=False,
                  timing_scope='Synchronized fit_tensor_model including normal audits/refit; excludes input/upload, extra graph validation, summaries, export/publication and file writes; warm pairs reuse process/compiler caches')
    def timeout(signum, frame):
        raise TimeoutError('Timing study wall budget exhausted')
    def terminated(signum, frame):
        raise KeyboardInterrupt('Timing study terminated')
    old_alarm = signal.signal(signal.SIGALRM, timeout)
    old_term = signal.signal(signal.SIGTERM, terminated)
    signal.alarm(args.timeout_seconds)
    try:
        mixed.require(record['source']['source_sha256'] == args.expected_source_sha256, 'Source differs')
        record['coverage'] = coverage(args, record['source'])
        device = study.require_cuda(args.device)
        record.update(cuda_available=True, device=str(device), gpu=torch.cuda.get_device_name(device),
                      torch=torch.__version__, cuda_runtime=torch.version.cuda, interpreter=sys.executable,
                      lsf_job_id=os.environ.get('LSB_JOBID'))
        data, host = mixed.fixtures.write_fixture(work/'input.tsv', args.fixture, args.nodes)
        mixed.require(data.input_sha256 == record['coverage']['input_sha256'], 'Input differs from coverage')
        record['input_sha256'] = data.input_sha256
        kernels = study.Kernels(device, compiled=True)
        mixed.require(kernels.compiled, 'Compiled CUDA required')
        for repeat in range(4):
            pair = {}
            for name in order(repeat):
                out = work/f'{repeat}-{name}.json'
                value = trial(out, name, repeat, data, host, device, kernels)
                record['trials'].append({k: v for k, v in value.items() if k != 'summary'})
                pair[name] = value
                print(json.dumps(dict(repeat=repeat, strategy=name, status=value['status'],
                                      fit_seconds=value['fit_seconds'])), flush=True)
                mixed.require(name != study.CANDIDATE or value['status'] == 'passed',
                              'Candidate coverage regressed without instrumentation')
            record['comparisons'].append(compare(pair[study.CANDIDATE], pair[study.CONTROL]))
        record['timing_summary'] = summarize(record['comparisons'])
        record['status'] = 'passed'
    except BaseException as error:
        record.update(status='failed', error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGALRM, old_alarm)
        record.update(finished_utc=common.utc_now(), artifacts={str(p.relative_to(args.out.parent)):common.sha(p)
                      for p in sorted(work.rglob('*')) if p.is_file()})
        common.write_json(args.out, record)
    return int(record['status'] != 'passed')


if __name__ == '__main__':
    raise SystemExit(main())
