"""CUDA-only replay of exact, source/hash-bound failed surrogate QPs.

Diagnostic execution is not numerical success: every completed receipt carries
all_replays_qualified and unresolved counts. Candidate-only repair experiments
never qualify the original-start solver replay. No pilot/graph is regenerated.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import sys
from time import perf_counter
import traceback

import numpy as np
import torch

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks import capture_failed_qp as capture
from benchmarks import qualify_cuda as common
from clipp1d.api import source_provenance
from clipp1d.cuda import qp
from clipp1d.cuda.kernels import Kernels, differences, adjoint
from clipp1d.cuda.policy import CudaPolicy
from clipp1d.cuda_api import require_cuda


SCHEMA = 'clipp1d.cuda.failed_qp_replay.v1'
CAPTURE_SOURCE = '726eaf873550d80cd89d72d7c1253e1c70aa97135011fbad7f9d6c97c9382a88'
CAPTURE_COMMIT = '430db26cf07466e88e53c6e1a8fbe2be7b7b25e9'
PROBLEM_KEYS = ('h', 'target', 'lower', 'upper', 'caps')
REPAIR_LIMIT = 1024


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def check_source(expected):
    require(re.fullmatch('[0-9a-f]{64}', expected or '') is not None,
            'An explicit expected production source SHA-256 is required')
    source = source_provenance()
    require(source['source_sha256'] == expected, 'Current source differs from requested replay source')
    return source


def certificate(stats, policy):
    require(stats.shape == (3,), 'Invalid original-QP certificate shape')
    values = stats.detach().cpu().tolist()
    finite = all(np.isfinite(value) for value in values)
    gap, scale, kkt = values
    allowed = policy.inner_atol + policy.inner_rtol * scale
    qualified = finite and gap >= 0 and scale >= 0 and 0 <= kkt <= policy.inner_kkt_tol and gap <= allowed
    return dict(gap=gap, scale=scale, kkt=kkt, allowed_gap=allowed,
                independently_qualified=bool(qualified))


def stats_for(x, q, problem, kernels):
    return kernels.gap_kkt(x, q, *(problem[key] for key in PROBLEM_KEYS))


def original_objective(x, problem):
    h, target, lower, upper, caps = (problem[key] for key in PROBLEM_KEYS)
    safe_target = torch.where(lower < upper, target, lower)
    reference = torch.maximum(lower, torch.minimum(upper, safe_target))
    value = qp.quadratic_value(x, h, safe_target, caps, reference)
    require(bool(torch.isfinite(value)), 'Nonfinite original shifted QP objective')
    return float(value)


def check_problem(record, entry=None):
    context = record['context']
    if entry is not None:
        require(context == entry['context'], 'Capture inventory context differs from its record')
    require(context['source_sha256'] == CAPTURE_SOURCE, 'Capture is not the pinned430db26 source')
    require(context['policy'] == asdict(CudaPolicy()), 'Captured QP changed the original numerical policy')
    require(set(record['problem']) == set(PROBLEM_KEYS) | {'start', 'dual'}, 'Captured QP parameter coverage differs')
    require(record['returned']['qualified'] is False, 'Replay requires an actually unresolved captured QP')
    require(type(record['returned']['iterations']) is int and
            0 <= record['returned']['iterations'] <= CudaPolicy().inner_max_iterations,
            'Captured QP exceeded the fixed iteration budget')
    return CudaPolicy(**context['policy'])


def load_problem(receipt, record, device):
    # None preserves the original omitted-start/omitted-dual solver branches.
    return {key: None if descriptor is None else capture.load_tensor(receipt, descriptor, device)
            for key, descriptor in record['problem'].items()}


def save_tensor(path, tensor, receipt_root):
    array = tensor.detach().cpu().contiguous().numpy()
    require(array.dtype == np.float64 and np.isfinite(array).all(), 'Invalid replay state export')
    value = dict(dtype='float64', encoding='json_numbers', shape=list(array.shape), values=array.tolist(),
                 tensor_sha256=hashlib.sha256(array.tobytes()).hexdigest())
    common.write_json(path, value)
    restored = np.asarray(json.loads(path.read_text())['values'], dtype=np.float64)
    require(restored.shape == array.shape and hashlib.sha256(restored.tobytes()).hexdigest() == value['tensor_sha256'],
            'Exported tensor JSON did not roundtrip exactly')
    return dict(path=str(path.relative_to(receipt_root)), sha256=common.sha(path),
                dtype='float64', shape=list(array.shape), tensor_sha256=value['tensor_sha256'])


def saved_state(root, label, x, q, receipt_root):
    return {key: save_tensor(root / f'{label}-{key}.json', value, receipt_root)
            for key, value in [('x', x), ('q', q)]}


def legacy_geometry(x, problem):
    """Literal430db26 geometry; benchmark reference only, never a solver fallback."""
    h, target, lower, upper, caps = (problem[key] for key in PROBLEM_KEYS)
    same = x[:, None] == x[None, :]
    external = caps * differences(x).sign()
    free = lower < upper
    safe_target = torch.where(free, target, x)
    gradient = torch.where(free, h * (x - safe_target), 0.0)
    fixed = ~free
    fixed_count = (same & fixed[None, :]).sum(1)
    at_lower = free & (x == lower)
    at_upper = free & (x == upper)
    lower_count = (same & at_lower[None, :]).sum(1)
    upper_count = (same & at_upper[None, :]).sum(1)
    sizes = same.sum(1)
    return (same, external, gradient, fixed, fixed_count, at_lower, lower_count,
            at_upper, upper_count, sizes, caps)


def legacy_uniform_step(q, same, external, gradient, fixed, fixed_count,
                        at_lower, lower_count, at_upper, upper_count, sizes, caps):
    """Exact archived430db26 tensor expressions; eager CUDA diagnostic only."""
    q = torch.where(same, q, external)
    residual = gradient + adjoint(q)
    group_total = torch.where(same, residual[None, :], 0.0).sum(1)
    normal = torch.where(
        fixed_count > 0,
        torch.where(fixed, group_total / fixed_count.clamp_min(1), 0.0),
        torch.where(
            group_total >= 0,
            torch.where(at_lower, group_total / lower_count.clamp_min(1), 0.0),
            torch.where(at_upper, group_total / upper_count.clamp_min(1), 0.0),
        ),
    )
    correction = residual - normal
    delta = -differences(correction) / sizes[:, None]
    proposal = q + torch.where(same, delta, 0.0)
    return torch.maximum(-caps, torch.minimum(caps, proposal))


def repair_candidate(x, initial_q, problem, kernels, policy, mode, *, limit=REPAIR_LIMIT):
    """Bounded fixed-primal experiment; no solved-QP admission is inherited."""
    require(mode in ('legacy_uniform', 'production_cone') and 1 <= limit <= REPAIR_LIMIT,
            'Invalid fixed-candidate diagnostic controls')
    original_x, q = x.clone(), initial_q.clone()
    before = capture.decompose_gap(x, q, *(problem[key] for key in PROBLEM_KEYS))
    before_certificate = certificate(stats_for(x, q, problem, kernels), policy)
    if mode == 'legacy_uniform':
        geometry = legacy_geometry(x, problem)
        def step(current):
            return legacy_uniform_step(current, *geometry)
    else:
        geometry = qp.prepare_flow_polish(x, *(problem[key] for key in PROBLEM_KEYS))
        def step(current):
            return qp.flow_polish_step(current, geometry, kernels)
    checkpoint, first = q.clone(), None
    stop = 'bounded_limit_unresolved'
    for iteration in range(1, limit + 1):
        q = step(q)
        if iteration == 1 or iteration % policy.check_every == 0 or iteration == limit:
            checked = certificate(stats_for(x, q, problem, kernels), policy)
            if iteration == 1:
                first = checked
            if checked['independently_qualified']:
                stop = 'original_certificate_qualified'
                break
            if torch.equal(q, checkpoint):
                stop = 'stationary_unresolved'
                break
            checkpoint = q.clone()
    require(torch.equal(x, original_x), 'Repair diagnostic changed the candidate primal')
    return q, dict(mode=mode, steps=iteration, stop=stop, before=before,
                   before_certificate=before_certificate, first_step_certificate=first,
                   after=capture.decompose_gap(x, q, *(problem[key] for key in PROBLEM_KEYS)),
                   after_certificate=certificate(stats_for(x, q, problem, kernels), policy),
                   original_shifted_objective=original_objective(x, problem),
                   scope='Fixed captured candidate/primal and original QP; no initialization, grouping, boxes or capacities changed')


def review_counterexample(device, compiled, eager):
    """The user's literal three-node witness; execution backend is explicit."""
    def tensor(value):
        return torch.tensor(value, dtype=torch.float64, device=device)
    x = tensor([.5, .5, .5])
    valid = tensor([[0, 0, -.5], [0, 0, 0], [.5, 0, 0]])
    bad = tensor([[0, -.125, -1], [.125, 0, .125], [1, -.125, 0]])
    problem = dict(h=tensor([128.] * 3), target=tensor([37/64, .5, 127/256]),
                   lower=tensor([1e-6] * 3), upper=tensor([.5, .5, .9]),
                   caps=tensor([[0, .125, 1], [.125, 0, .125], [1, .125, 0]]))
    policy, rows = CudaPolicy(), []
    geometry = legacy_geometry(x, problem)
    require(torch.equal(legacy_uniform_step(valid, *geometry), bad), 'Legacy counterexample transcription differs')
    require(torch.equal(legacy_uniform_step(bad, *geometry), bad), 'Archived uniform bad state is not fixed')
    for label, kernels in [('eager', eager), ('compiled', compiled)]:
        good_stats, bad_stats = stats_for(x, valid, problem, kernels), stats_for(x, bad, problem, kernels)
        require(float(good_stats[0]) == 0 and float(good_stats[2]) == 0, 'Exact valid dual does not certify')
        require(float(bad_stats[0]) == .00054931640625 and float(bad_stats[2]) == .15789473684210525,
                'Uniform fixed-point certificate differs from the literal review')
        context = qp.prepare_flow_polish(x, *(problem[key] for key in PROBLEM_KEYS))
        require(torch.equal(qp.flow_polish_step(valid, context, kernels), valid), 'Cone repair destroyed exact valid dual')
        for start_name, initial in [('zero', torch.zeros_like(valid)), ('legacy_fixed_point', bad)]:
            repaired, detail = repair_candidate(x, initial, problem, kernels, policy, 'production_cone')
            independent = certificate(stats_for(x, repaired, problem, eager), policy)
            require(detail['after_certificate']['independently_qualified'] and independent['independently_qualified'],
                    'Cone repair did not recover the literal witness within its original gates')
            rows.append(dict(backend=label, initialization=start_name, steps=detail['steps'],
                             original_certificate=independent, gap_before=detail['before'], gap_after=detail['after']))
    return dict(status='passed', numerical_execution='CUDA float64' if torch.device(device).type == 'cuda' else 'CPU reference test only',
                exact_valid_dual_preserved=True, legacy_uniform_bad_fixed_point=True,
                legacy_bad_gap=.00054931640625, legacy_bad_kkt=.15789473684210525,
                cone_recoveries=rows, scope='Three-node mechanism proof only; does not attribute any larger captured failure')


def restore_baseline(receipt, record, problem, kernels, policy, device):
    returned = record['returned']
    x, q = (capture.load_tensor(receipt, returned[key], device) for key in ('x', 'q'))
    stored = torch.stack([capture.load_tensor(receipt, returned[key], device) for key in ('gap', 'scale', 'kkt')])
    reproduced = stats_for(x, q, problem, kernels)
    require(torch.equal(reproduced.view(torch.int64), stored.view(torch.int64)),
            f'Captured terminal certificate did not reconstruct exactly: stored={stored.tolist()}, replay={reproduced.tolist()}')
    original = certificate(reproduced, policy)
    require(not original['independently_qualified'], 'Captured failed QP independently passes; investigate before replay')
    terminal = record['terminal_admm']
    tx, tq = (capture.load_tensor(receipt, terminal[key], device) for key in ('x', 'q'))
    states_match = torch.equal(tx, x) and torch.equal(tq, q)
    return dict(returned_certificate=original, certificate_exactly_reconstructed=True,
                original_shifted_objective=original_objective(x, problem),
                returned=capture.decompose_gap(x, q, *(problem[key] for key in PROBLEM_KEYS)),
                terminal_admm=capture.decompose_gap(tx, tq, *(problem[key] for key in PROBLEM_KEYS)),
                terminal_admm_equals_returned=states_match)


@torch.no_grad()
def run(args, receipt, root, record_event):
    device = require_cuda(args.device)
    source = check_source(args.expected_source_sha256)
    require(os.environ.get('CC') and shutil.which(os.environ['CC']), 'CUDA replay requires an available CC compiler')
    captured = capture.load_capture(args.capture, args.capture_sha256)
    require(captured['captures'], 'Capture contains no failed QPs to replay')
    props = torch.cuda.get_device_properties(device)
    receipt.update(source=source, numerical_execution='CUDA float64',
                   environment=dict(gpu=props.name, device_uuid=str(getattr(props, 'uuid', 'unavailable')),
                                    torch=torch.__version__, cuda_runtime=torch.version.cuda),
                   capture=dict(path=str(args.capture), sha256=args.capture_sha256,
                                baseline_source_sha256=CAPTURE_SOURCE, baseline_commit=CAPTURE_COMMIT),
                   policy=asdict(CudaPolicy()), captured_failed_qps=len(captured['captures']))
    kernels, reference = Kernels(device, compiled=True), Kernels(device, compiled=False)
    require(kernels.compiled, 'Missing production compilation admission')
    receipt['review_counterexample'] = review_counterexample(device, kernels, reference)
    record_event('review_counterexample_qualified', receipt['review_counterexample'])
    # Reconstruction of every baseline state precedes any current-source solve.
    baseline_reconstructions = []
    for index, entry in enumerate(captured['captures']):
        stored = capture.load_record(captured, entry)
        policy = check_problem(stored, entry)
        problem = load_problem(captured, stored, device)
        restored = restore_baseline(captured, stored, problem, kernels, policy, device)
        baseline_reconstructions.append(restored)
        record_event('baseline_reconstructed', dict(capture_index=index, certificate=restored['returned_certificate']))
    for index, entry in enumerate(captured['captures']):
        stored = capture.load_record(captured, entry)
        policy = check_problem(stored, entry)
        problem = load_problem(captured, stored, device)
        snapshots = {key: None if value is None else value.clone() for key, value in problem.items()}
        case_root = root / f'capture-{index:04d}'
        case_root.mkdir()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        begin = perf_counter()
        fit = qp.solve_qp(*(problem[key] for key in PROBLEM_KEYS), kernels, policy,
                          start=problem['start'], dual=problem['dual'])
        torch.cuda.synchronize(device)
        seconds = perf_counter() - begin
        peak_allocated, peak_reserved = torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)
        for key, value in problem.items():
            require((value is None and snapshots[key] is None) or torch.equal(value, snapshots[key]),
                    f'Replay mutated original QP tensor {key}')
        checked = certificate(stats_for(fit.x, fit.dual, problem, reference), policy)
        compiled_checked = certificate(stats_for(fit.x, fit.dual, problem, kernels), policy)
        require(fit.iterations <= policy.inner_max_iterations, 'Replay exceeded original QP budget')
        resolved = bool(fit.qualified and checked['independently_qualified'] and compiled_checked['independently_qualified'])
        case = dict(capture_index=index, capture_record_sha256=entry['sha256'], context=stored['context'],
                    original_baseline=baseline_reconstructions[index], qualified=resolved,
                    production_qualified=bool(fit.qualified), eager_original_certificate=checked,
                    original_shifted_objective=original_objective(fit.x, problem),
                    primal_box_feasible=bool(((fit.x >= problem['lower']) & (fit.x <= problem['upper'])).all()),
                    dual_capacity_feasible=bool((fit.dual.abs() <= problem['caps']).all()),
                    dual_exact_skew=bool(torch.equal(fit.dual, -fit.dual.T)),
                    compiled_original_certificate=compiled_checked,
                    seconds=seconds, timing_scope='synchronized original-start production solve only; may include lazy compilation, no repeated latency claim',
                    peak_allocated_bytes=peak_allocated, peak_reserved_bytes=peak_reserved,
                    admm_iterations=fit.iterations, polish_iterations=fit.polish_iterations,
                    result=saved_state(case_root, 'replay', fit.x, fit.dual, args.out.parent),
                    decomposition=capture.decompose_gap(fit.x, fit.dual, *(problem[key] for key in PROBLEM_KEYS)),
                    candidate_repairs=[])
        candidate_seen = {}
        case['candidate_coverage'] = {}
        for origin in ('last_equality_attempt', 'last_refined_state'):
            attempt = stored.get(origin)
            if attempt is None or attempt.get('x') is None or attempt.get('input_q') is None:
                case['candidate_coverage'][origin] = 'no captured primal/input dual; no substitute generated'
                continue
            identity = (attempt['x']['tensor_sha256'], attempt['input_q']['tensor_sha256'])
            if identity in candidate_seen:
                case['candidate_coverage'][origin] = dict(same_literal_problem_as=candidate_seen[identity])
                continue
            candidate_seen[identity] = origin
            case['candidate_coverage'][origin] = 'both repair maps from identical captured input'
            x = capture.load_tensor(captured, attempt['x'], device)
            q = capture.load_tensor(captured, attempt['input_q'], device)
            for mode in ('legacy_uniform', 'production_cone'):
                repaired, detail = repair_candidate(x, q, problem, kernels, policy, mode)
                detail.update(candidate_origin=origin, captured_objective_admitted=attempt.get('objective_admitted'),
                              result=saved_state(case_root, f'{origin}-{mode}', x, repaired, args.out.parent))
                case['candidate_repairs'].append(detail)
        for key, value in problem.items():
            require((value is None and snapshots[key] is None) or torch.equal(value, snapshots[key]),
                    f'Repair diagnostic mutated original QP tensor {key}')
        common.write_json(case_root / 'result.json', case)
        receipt['replays'].append(case)
        record_event('replay_completed', dict(capture_index=index, qualified=resolved,
                     admm_iterations=fit.iterations, polish_iterations=fit.polish_iterations,
                     result_sha256=common.sha(case_root / 'result.json')))
    require(len(receipt['replays']) == receipt['captured_failed_qps'], 'Incomplete captured-QP replay coverage')
    require(check_source(args.expected_source_sha256)['source_sha256'] == source['source_sha256'], 'Replay source changed')
    require(common.sha(__file__) == receipt['script_sha256'] and
            common.sha(capture.__file__) == receipt['capture_helper_sha256'] and
            capture.helper_hashes() == receipt['capture_dependency_hashes'], 'Replay helper changed')
    receipt['compilation_statistics'] = kernels.compilation_diagnostics()
    finish_diagnostics(receipt, args.require_all_qualified)


def finish_diagnostics(receipt, require_all):
    rows = receipt['replays']
    require(rows and len(rows) == receipt['captured_failed_qps'], 'Incomplete captured-QP replay coverage')
    require(all(type(row['qualified']) is bool for row in rows), 'Missing explicit replay qualification')
    passed = all(case['qualified'] for case in rows)
    receipt.update(all_replays_qualified=passed,
                   resolved_qps=sum(case['qualified'] for case in rows),
                   unresolved_qps=sum(not case['qualified'] for case in rows),
                   scientific_status='all_original_start_replays_qualified' if passed else 'unresolved_original_start_replays',
                   diagnostic_status='completed', status='passed')
    if require_all:
        require(passed, 'Promotion requires every original-start replay to qualify')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--capture-sha256', required=True)
    parser.add_argument('--expected-source-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--require-all-qualified', action='store_true')
    parser.add_argument('--timeout-seconds', type=int, default=1800)
    args = parser.parse_args()
    require(args.timeout_seconds > 0, 'Timeout must be positive')
    args.out = args.out.resolve()
    root = args.out.with_suffix('.artifacts')
    events = args.out.with_suffix('.events.jsonl')
    require(not any(path.exists() for path in (args.out, root, events)), 'Replay evidence cannot be overwritten')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir()
    receipt = dict(schema=SCHEMA, status='running', diagnostic_status='incomplete',
                   source=source_provenance(), script_sha256=common.sha(__file__),
                   capture_helper_sha256=common.sha(capture.__file__),
                   capture_dependency_hashes=capture.helper_hashes(), started_utc=common.utc_now(),
                   lsf_job_id=os.environ.get('LSB_JOBID'), command=sys.argv,
                   all_replays_qualified=False, replays=[],
                   qualification_scope='Exact captured surrogate QPs only; no full likelihood search, final refit, or output publication qualification',
                   status_scope='passed denotes validated diagnostic execution; all_replays_qualified is separate and mandatory for promotion')
    began, code = perf_counter(), 0
    with events.open('x', encoding='utf-8') as stream:
        def event(kind, detail):
            stream.write(json.dumps(common.clean(dict(kind=kind, utc=common.utc_now(), **detail)), sort_keys=True, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        def interrupted(signum, frame):
            raise TimeoutError(f'Failed-QP replay interrupted by signal {signum}')
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGALRM, interrupted)
        signal.alarm(args.timeout_seconds)
        try:
            run(args, receipt, root, event)
        except BaseException as error:
            code = 1
            receipt.update(status='failed', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
            event('failure', dict(error_type=type(error).__name__, error=str(error)))
        finally:
            signal.alarm(0)
            receipt.update(finished_utc=common.utc_now(), elapsed_seconds=perf_counter()-began,
                           events_sha256=common.sha(events),
                           artifacts={str(path.relative_to(args.out.parent)): common.sha(path)
                                      for path in sorted(root.rglob('*.json'))})
            common.write_json(args.out, receipt)
    return code


if __name__ == '__main__':
    sys.exit(main())
