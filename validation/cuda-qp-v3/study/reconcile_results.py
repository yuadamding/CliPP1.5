"""Read-only reconciliation of imported CUDA evidence; no fits or GPU calls.

Publishes one new compact receipt only after complete-path accounting, imported
file integrity and every paired attribution sample pass. The historical full
paths establish scientific parity, not matched full-fit throughput.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

STUDY = Path(__file__).resolve().parent
REPO = STUDY.parent.parent
SOURCES = {
    'baseline': 'ea7071788094e0cd3537530fcdc62e4228054fce19ff02ba8f2cf4eacce1dd26',
    'current': '726eaf873550d80cd89d72d7c1253e1c70aa97135011fbad7f9d6c97c9382a88',
}
BENCHMARK_SHA = '39c1a8813417d7ea87c5bf81f84ed57140bad4879d13173e7b39eaf8113307e3'
BASELINE_PATH_SHA = {
    16: '1a5201f6d5fc8eac6536074c33158c1ed13197c9594c388bd379fc9ccda06015',
    64: 'a09977b9b9b60a7fc84c197b69c9442de266fce2ff6c3e8eaac06031021381fd',
    256: 'f7aa4d3baa995447022249658a0229545f58cf87d9d9c56b6e6156e99c623ac3',
}
EXPECTED_ADMM = {16: 6048, 64: 20128, 256: 25904}
EVIDENCE = {}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    require(path.is_file() and not path.is_symlink(), f'Missing regular evidence: {path}')
    EVIDENCE[str(path.relative_to(REPO))] = digest(path)
    return json.loads(path.read_text())


def verified_import(name, *, expect_passed=True):
    root = STUDY / name
    prepared = read(root / 'PREPARED.json')
    require(prepared['source_sha256'] == SOURCES, f'{name}: unexpected source identities')
    sealed = root / 'sealed' / prepared['run_id']
    plan = read(sealed / 'PREPARED.json')
    manifest = read(root / 'IMPORT_MANIFEST.json')
    imported = root / 'imported'
    for relative, expected in manifest['files'].items():
        path = imported / relative
        require(path.resolve().is_relative_to(imported.resolve()), 'Escaping import path')
        require(path.is_file() and not path.is_symlink(), f'Missing import: {relative}')
        require(path.stat().st_size == expected['bytes'] and digest(path) == expected['sha256'],
                f'Import hash/length mismatch: {name}/{relative}')
    terminal = read(imported / 'receipts/terminal.json')
    startup = read(imported / 'receipts/startup.json')
    accepted = read(imported / 'receipts/accepted.json')
    require(terminal == manifest['terminal'] and terminal['passed'] is expect_passed,
            f'{name}: unexpected terminal status')
    require(terminal['job_id'] == startup['job_id'] == accepted['job_id'], 'Job identity mismatch')
    require(accepted['job_name'] == plan['job_name'], 'Accepted job name mismatch')
    require(terminal['plan_sha256'] == accepted['plan_sha256'] == digest(sealed / 'PREPARED.json'),
            'Worker plan identity mismatch')
    require(startup['inventory_sha256'] == prepared['inventory_sha256'], 'Source inventory mismatch')
    require(startup['environment_sha256'] == prepared['environment_sha256'], 'Environment mismatch')
    require(startup['compiler_sha256'] == prepared['compiler']['sha256'], 'Compiler mismatch')
    if expect_passed:
        require(len(terminal['tasks']) == len(plan['tasks']), 'Incomplete worker task coverage')
    else:
        require(0 < len(terminal['tasks']) <= len(plan['tasks']), 'Invalid failed task prefix')
        require(terminal['tasks'][-1]['status'] == 'failed', 'Missing retained failure')
    receipts = {}
    for task, resolved in zip(plan['tasks'], terminal['tasks']):
        role, task_name = task['source_role'], task['name']
        require(resolved['name'] == task_name and resolved['source_role'] == role,
                'Worker task order/role mismatch')
        expected_status = 'failed' if not expect_passed and resolved is terminal['tasks'][-1] else 'passed'
        require(resolved['status'] == expected_status and
                (resolved['returncode'] == 0) == (expected_status == 'passed'), 'Unexpected worker task result')
        path = imported / 'results' / f'{task_name}.json'
        value = read(path)
        require(digest(path) == resolved['receipt_sha256'], 'Worker result hash mismatch')
        require(value['status'] == expected_status and value['lsf_job_id'] == terminal['job_id'],
                'Unqualified result/job identity')
        require(value['source']['source_sha256'] == SOURCES[role], 'Unexpected result source')
        require(startup['runtimes'][role]['source']['source_sha256'] == SOURCES[role],
                'Unexpected startup source')
        script = sealed / 'source/benchmarks' / task['script']
        script_sha = digest(script)
        recorded_sha = value.get('script_sha256', value.get('qualification_script_sha256'))
        require(script_sha == recorded_sha, 'Executed script identity mismatch')
        events = path.with_suffix('.events.jsonl')
        require(digest(events) == value['events_sha256'], 'Result event hash mismatch')
        receipts[task_name] = value
    return imported, receipts, dict(job_id=terminal['job_id'], imported_files=len(manifest['files']),
                                    terminal_sha256=digest(imported / 'receipts/terminal.json'))


def path_work(path, current):
    require(path['search_status'] == 'complete' and path['raw_qualified'], 'Incomplete raw search')
    require(path['clonal_constraint'] is False, 'Clonal fitting constraint restored')
    records = path['path_records']
    require(len(records) == 26 and records[0]['lambda_value'] == 0, 'Default path coverage mismatch')
    require(path['path_position']['coordinates'] == [None] + list(range(-12, 13)), 'Wrong path coordinates')
    require(path['path_position']['recipe_recomputed_exactly'], 'Unverified literal lambda recipe')
    require(path['pilot_and_graph']['recipe_recomputed_exactly'], 'Unverified graph recipe')
    totals = {key: 0 for key in ('qp_admm_iterations', 'qp_calls', 'qp_polish_iterations',
                                'qp_dual_warm_starts', 'qp_dual_warm_resets')}
    for index, record in enumerate(records):
        require(record['raw_status'] == record['refit_status'] == 'qualified' and record['search_complete'],
                f'Unqualified path record {index}')
        starts = record.get('starts', [])
        require(record['starts_attempted'] == len(starts), 'Start coverage mismatch')
        if index == 0:
            require(not starts and record['separable_scalar_gap_qualified'], 'Invalid lambda-zero lane')
            require(record['inner_iterations'] == 0, 'Lambda-zero QP accounting')
        else:
            require(record['starts_qualified'] == len(starts) and record['starts_unresolved'] == 0,
                    'Unresolved path starts')
        admm = sum(start['inner_iterations'] for start in starts)
        for start in starts:
            require(start['qualified'] and start['inner_gap_qualified'] and start['inner_kkt_qualified'],
                    'Unqualified raw start')
            if current:
                require(start['qp_admm_iterations'] == start['inner_iterations'], 'Per-start ADMM mismatch')
        if current:
            require(record['qp_admm_iterations'] == admm, 'Per-record ADMM mismatch')
        totals['qp_admm_iterations'] += admm
        for key in totals.keys() - {'qp_admm_iterations'}:
            subtotal = sum(start[key] for start in starts)
            require(record[key] == subtotal, f'Per-record {key} mismatch')
            totals[key] += subtotal
    times = path['timings']
    require(times['planned_path_complete'] and not times['path_truncated'], 'Incomplete planned search')
    require(times['raw_unresolved_penalties'] == times['refit_unresolved_penalties'] == 0,
            'Unresolved path penalties')
    for key, total in totals.items():
        if current or key != 'qp_admm_iterations':
            require(times[key] == total, f'Path aggregate {key} mismatch')
    return totals


def complete_paths(imported):
    comparisons = []
    for n in (16, 64, 256):
        baseline_path = REPO / 'validation/cuda-review-v3/artifacts' / f'scaling-{n}-path.json'
        current_path = imported / 'results/qualification-current.artifacts' / baseline_path.name
        require(digest(baseline_path) == BASELINE_PATH_SHA[n], 'Historical path identity mismatch')
        a, b = read(baseline_path), read(current_path)
        wa, wb = path_work(a, False), path_work(b, True)
        require(wa['qp_admm_iterations'] == wb['qp_admm_iterations'] == EXPECTED_ADMM[n], 'ADMM parity failed')
        for key in ('pilot_and_graph', 'path_position', 'cluster_labels', 'cluster_centers',
                    'raw_ccf', 'refitted_ccf', 'raw_objective', 'score', 'selected_lambda', 'graph_edges'):
            require(a[key] == b[key], f'N{n}: exact scientific parity failed: {key}')
        require([r['lambda_value'] for r in a['path_records']] ==
                [r['lambda_value'] for r in b['path_records']], 'Literal lambda mismatch')
        qualified = read(current_path.with_name(f'scaling-{n}-qualified.json'))
        require(qualified['qualified'] and qualified['final_device_qualification_and_export_complete'],
                'Final exported fit is not qualified')
        require(qualified['numerical_stages'] == b['timings'], 'Exported timing mismatch')
        for exported, path_key in [('labels', 'cluster_labels'), ('raw_ccf', 'raw_ccf'),
                                   ('refitted_ccf', 'refitted_ccf'), ('score', 'score')]:
            require(qualified[exported] == b[path_key], 'Qualified export/path mismatch')
        comparisons.append(dict(nodes=n, planned_penalties=len(b['path_records']),
                                exact_graph_path_and_selected_fit_parity=True,
                                baseline_work=wa, current_work=wb))
    return dict(status='passed', comparisons=comparisons,
                scope='Historical complete-path scientific parity and all-start work accounting; separate GPU allocations are not matched full-fit latency evidence')


def samples(case):
    return ([(f'warmup-{i}', r) for i, r in enumerate(case['warmups'])] +
            [(f'latency-{i}', r) for i, r in enumerate(case['latency_samples'])] +
            [(key, case[key]) for key in ('dispatch_profile', 'attribution')])


def verify_profile(profile, attribution=False):
    require(profile['numerical_execution'] == 'CUDA float64', 'Non-CUDA profile')
    require(profile['kernel_dispatches'] > 0, 'Empty GPU dispatch evidence')
    require(profile['device_accounting'].startswith('raw Kineto CUDA work only;'), 'Old profiler accounting')
    require(profile['device_events'] == profile['correlated_cuda_work_events'] +
            profile['uncorrelated_cuda_work_events'], 'CUDA work correlation accounting mismatch')
    host = sum(r['host_range_seconds'] for r in profile['stages'].values())
    device = sum(r['device_kernel_seconds'] for r in profile['stages'].values())
    for actual, expected in [(host + profile['unattributed_host_wall_seconds'], profile['instrumented_wall_seconds']),
                             (device + profile['unattributed_device_seconds'], profile['device_kernel_and_copy_seconds'])]:
        require(abs(actual - expected) <= 1e-10 * max(1., abs(expected)), 'Profile accounting overlap/gap')
    if attribution:
        require(all(r['calls'] > 0 and r['device_kernel_seconds'] > 0
                    for r in profile['stages'].values()), 'Missing named stage/device work')
        require(sum(r['calls'] for r in profile['stages'].values()) ==
                sum(profile['stage_operation_counts'].values()), 'CPU stage-call coverage mismatch')
        require(profile['stage_operation_counts']['admm_update'] == profile['certificate']['admm_iterations'],
                'Incomplete attributed ADMM coverage')
    return {key: profile[key] for key in ('instrumented_wall_seconds', 'kernel_dispatches',
            'device_kernel_and_copy_seconds', 'stages', 'stage_operation_counts',
            'unattributed_host_wall_seconds', 'unattributed_device_seconds', 'host_scalar_read_proxies',
            'discarded_cuda_user_annotations', 'correlated_cuda_work_events', 'uncorrelated_cuda_work_events',
            'device_accounting', 'duplicate_cpu_correlation_ids',
            'cuda_work_with_unanimous_duplicate_owners')}


def paired_attribution(imported, receipts):
    script = REPO / 'benchmarks/attribute_qp_cuda.py'
    require(digest(script) == BENCHMARK_SHA, 'Frozen comparison code changed')
    EVIDENCE[str(script.relative_to(REPO))] = BENCHMARK_SHA
    sys.path.insert(0, str(REPO / 'src'))
    spec = importlib.util.spec_from_file_location('frozen_qp_attribution', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    baseline, current = (receipts[f'attribution-{role}'] for role in ('baseline', 'current'))
    require(baseline['script_sha256'] == current['script_sha256'] == BENCHMARK_SHA, 'Benchmark script mismatch')
    require(baseline['lsf_job_id'] == current['lsf_job_id'], 'Pair left its single allocation')
    require(baseline['environment']['device_uuid'] not in ('unavailable', ''), 'Missing device identity')
    comparison = module.compare_receipts(baseline, current)
    by_role = {role: {c['fixture']['name']: c for c in receipt['fixtures']}
               for role, receipt in [('baseline', baseline), ('current', current)]}
    for item in comparison['comparisons']:
        name = item['name']
        common = by_role['baseline'][name]['latency_samples'][0]['certificate']
        checks = []
        for role in ('baseline', 'current'):
            case = by_role[role][name]
            root = imported / f'results/attribution-{role}.artifacts'
            input_path = root / f'{name}.inputs.json'
            literal = read(input_path)
            require(digest(input_path) == case['input_file_sha256'], 'Literal input file mismatch')
            arrays = {key: np.asarray(value, dtype=np.float64) for key, value in literal['arrays'].items()}
            problem = dict(name=name, recipe=case['fixture']['recipe'], arrays=arrays)
            identity = module.fixture_identity(problem)
            require(identity == literal['identity'] == case['fixture'], 'Literal numeric input identity mismatch')
            require(read(root / f'{name}.measurements.json') == case, 'Measurement artifact differs from receipt')
            all_samples = samples(case)
            work = case['work_totals']
            require(work['qp_calls'] == work['independent_original_certificates'] == len(all_samples), 'Certificate count mismatch')
            for metric in ('admm_iterations', 'polish_iterations'):
                require(work[metric] == sum(row['certificate'][metric] for _, row in all_samples), 'QP work total mismatch')
            for label, row in all_samples:
                cert = row['certificate']
                x, reference = np.asarray(cert['x']), np.asarray(common['x'])
                require(x.shape == arrays['h'].shape and cert['minimum_curvature'] == float(arrays['h'].min()),
                        'Curvature/dimension certificate mismatch')
                require(np.all(x >= arrays['lower']) and np.all(x <= arrays['upper']), 'Certificate state violates original box')
                bound = np.sqrt(2 * common['gap'] / common['minimum_curvature']) + np.sqrt(2 * cert['gap'] / cert['minimum_curvature'])
                bound += 256 * np.finfo(float).eps * (1 + np.linalg.norm(reference) + np.linalg.norm(x))
                error = float(np.linalg.norm(x - reference))
                obound = common['gap'] + cert['gap'] + 256 * np.finfo(float).eps * (1 + abs(common['objective']) + abs(cert['objective']))
                oerror = abs(common['objective'] - cert['objective'])
                require(error <= bound and oerror <= obound, f'{role}/{name}/{label}: common-baseline parity failed')
                checks.append(dict(source_role=role, sample=label, solution_l2_error=error,
                                   solution_l2_bound=float(bound), objective_error=oerror, objective_bound=obound))
            verify_profile(case['dispatch_profile'])
            item[f'{role}_attribution'] = verify_profile(case['attribution'], attribution=True)
            item[f'{role}_work_all_eight_calls'] = work
            item[f'{role}_latency_seconds'] = [r['seconds'] for r in case['latency_samples']]
            item[f'{role}_dispatch_device_seconds'] = case['dispatch_profile']['device_kernel_and_copy_seconds']
        item['all_sample_common_baseline_checks'] = checks
        item['all_sample_checks'] = len(checks)
    comparison['stage_boundary_scope'] = ('Baseline dual_flow_repair includes repeated fixed geometry construction; '
        'current prepares reusable geometry in equality_proposal. Individual stage timing shifts include reclassification. '
        'Use disjoint work totals and uninstrumented latency medians for speed claims.')
    comparison['fixture_scope'] = ('resource64/resource256 use matched NumPy literal sin/caps arrays; '
        'they are not bitwise the CUDA-generated resource_probe in qualification-a. mixed64 is a fixed surrogate, not a full fit.')
    comparison['environment'] = baseline['environment']
    comparison['controls'] = baseline['controls']
    comparison['certificate_scope'] = ('Imported original-QP eager CUDA certificates independently emitted after every call; '
        'this local reconciler verifies identities, original boxes, gates, counts and gap-derived pairwise bounds without executing fits')
    return comparison


def publish(path, value):
    require(not path.exists(), f'Refusing to overwrite receipt: {path}')
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    descriptor, temporary = tempfile.mkstemp(prefix='.reconciliation-', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        require(path.read_bytes() == payload, 'Receipt publication readback differs')
    finally:
        os.unlink(temporary)
    return digest(path)


def retained_attempts():
    _, _, original = verified_import('attribution-b')
    invalidation = read(STUDY / 'attribution-b/MEASUREMENT_INVALIDATION.json')
    require(invalidation['status'] == 'measurement_claims_invalidated', 'Missing profiler invalidation')
    for relative, expected in invalidation['affected_receipts'].items():
        require(digest(STUDY / 'attribution-b/imported' / relative) == expected, 'Invalidated receipt changed')
    _, failed, retry = verified_import('attribution-b2', expect_passed=False)
    require(set(failed) == {'attribution-baseline'}, 'Unexpected b2 task coverage')
    failure = failed['attribution-baseline']
    require(failure['error'] == 'Ambiguous CPU profiler correlation ownership', 'Unexpected b2 failure')
    return {
        'attribution-b': dict(**original, original_worker_status='passed',
            disposition='Profiler claims invalidated; original bytes and separate latency/certificates retained',
            invalidation_sha256=digest(STUDY / 'attribution-b/MEASUREMENT_INVALIDATION.json')),
        'attribution-b2': dict(**retry, original_worker_status='failed',
            error_type=failure['error_type'], error=failure['error'],
            disposition='Fail-closed profiler mapping rejection; no completed paired attribution or speed claim'),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), 'Existing reconciliation is immutable')
    aroot, _, ainfo = verified_import('qualification-a')
    paths = complete_paths(aroot)
    broot, breceipts, binfo = verified_import('attribution-b3')
    paired = paired_attribution(broot, breceipts)
    retained = retained_attempts()
    result = dict(schema='clipp1d.qp_study.reconciliation.v1', status='passed',
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  reconciler_sha256=digest(Path(__file__)), source_sha256=SOURCES,
                  imports={'qualification-a': ainfo, 'attribution-b3': binfo},
                  complete_paths=paths, paired_attribution=paired, retained_attempts=retained,
                  evidence_sha256=EVIDENCE)
    sha = publish(args.out, result)
    print(json.dumps(dict(status='passed', receipt=str(args.out), sha256=sha,
                         comparisons=[{k: v for k, v in row.items() if k in (
                             'name', 'baseline_median_seconds', 'current_median_seconds',
                             'baseline_over_current_latency', 'baseline_kernel_dispatches',
                             'current_kernel_dispatches', 'all_sample_checks')}
                             for row in paired['comparisons']]), sort_keys=True))


if __name__ == '__main__':
    main()
