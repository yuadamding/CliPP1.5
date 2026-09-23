"""Frozen276 CUDA diagnosis of mixed256's pooled raw state; no fitted-path pass."""
import argparse
from dataclasses import asdict
import inspect
import os
from pathlib import Path
import signal
import sys
import traceback

import torch

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks import capture_failed_qp as capture
from benchmarks import mixed_fixtures as fixtures
from benchmarks import qualify_cuda as common
from benchmarks import qualify_mixed_cuda as mixed
from clipp1d.api import source_provenance
from clipp1d.cuda import audit, qp, solver
from clipp1d.cuda.kernels import Kernels, adjoint, differences
from clipp1d.cuda.policy import CudaPolicy
from clipp1d.cuda_api import require_cuda

SOURCE = '276da27b9a8718f3f06182108ccb0a6b7f04f3937a67f050f9721e031554aea5'
REFERENCE = 'a910183f847f0ce715b7a476b1d244796f6d94736b4ad3975229c91b78a745ca'
INPUT = '6b5cb4f2cfce4cb6fbd81eff8a6ab8dca4e3c810dcfdf4e1ec75fa76fa6b2428'
SCHEMA = 'clipp1d.cuda.pooled_outer_diagnostic.v1'


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def rowwise_lower(r, a, q, allowed, unit):
    """Same conservative Gamma, moved before the monotone negative-part map.

    Gamma has eightfold slack over the N+2 floating-point reductions/operations,
    including coefficient normalization and row subtraction. Each row interval
    bounds its represented linear-plus-adjoint term before min(.,0), so a row
    whose entire interval is positive contributes exactly zero. The final error
    only concerns summation of negative terms and the unchanged constant floor.
    """
    gamma = 8 * (a.numel() + 2) * torch.finfo(a.dtype).eps
    row_error = gamma * (a.abs() + q.abs().sum(-1))
    negative = ((r - row_error).clamp_max(0.) * allowed)
    margin = gamma * (unit + negative.abs().sum())
    return negative.sum() - margin, margin


def rowwise_cut(*args, **kwargs):
    """Copy the frozen iteration exactly, replacing only its lower-bound formula."""
    code = inspect.getsource(audit.directional_cut)
    old = '''margin = (8 * (n + 2) * eps *
                      (unit + ((a.abs() + q.abs().sum(-1)) * allowed).sum()))
            lower = lower - margin'''
    new = 'lower, margin = rowwise_lower(r, a, q, allowed, unit)'
    require(code.count(old) == 1, 'Frozen directional-cut lower-bound source changed')
    namespace = dict(vars(audit), rowwise_lower=rowwise_lower)
    exec(compile(code.replace(old, new), __file__ + '::rowwise_diagnostic', 'exec'), namespace)
    return namespace['directional_cut'](*args, **kwargs)


def cut_problem(model, x, q, caps, sign, terms, policy):
    d = differences(x)
    fused = torch.where(d == 0., caps, 0.)
    external = adjoint(torch.where(d != 0., caps * d.sign(), 0.))
    external_absolute = torch.where(d != 0., caps, 0.).sum(-1)
    initial = torch.where(d == 0., q, 0.)
    derivative = terms[5] if sign > 0 else -terms[4]
    allowed = ((x < model.upper) if sign > 0 else (x > model.lower)).to(x.dtype)
    absolute = derivative.abs() + external_absolute
    linear = torch.where(allowed > 0, derivative + sign * external +
                         policy.stationarity_tol * absolute, 0.)
    moving = (allowed[:, None] > 0) | (allowed[None, :] > 0)
    fc = torch.where(moving, (1 + policy.stationarity_tol) * fused, 0.)
    dual = torch.where(moving, sign * initial, 0.)
    scale = torch.maximum(linear.abs().max(), fc.sum(-1).max()).clamp_min(1.)
    return dict(a=linear/scale, caps=fc/scale, allowed=allowed, initial_dual=dual/scale,
                tolerance=policy.stationarity_tol/scale,
                roundoff_unit=torch.ones_like(scale)/scale, objective_scale=scale,
                unscaled_linear=linear, unscaled_caps=fc)


def run_cut(problem, policy, implementation):
    diagnostics = {}
    ok, direction = implementation(*(problem[key] for key in ('a', 'caps', 'allowed', 'initial_dual')),
                                   policy, diagnostics=diagnostics,
                                   tolerance=problem['tolerance'], roundoff_unit=problem['roundoff_unit'])
    diagnostics['objective_scale'] = float(problem['objective_scale'])
    return dict(qualified=ok, direction=direction, diagnostics=diagnostics)


@torch.no_grad()
def run(args, receipt, journal):
    device = require_cuda(args.device)
    source = source_provenance()
    require(source['source_sha256'] == args.expected_source_sha256 == SOURCE,
            'Exact frozen276 source required; no current overlay allowed')
    require(common.sha(args.reference) == REFERENCE == args.reference_sha256,
            'Archived J full-path authority differs')
    archived = mixed.load_json(args.reference)
    require(archived['search_status'] == 'incomplete', 'Wrong archived path status')
    receipt.update(source=source, cuda_available=True, device=str(device),
                   gpu=torch.cuda.get_device_name(device), torch=torch.__version__,
                   cuda_runtime=torch.version.cuda, policy=asdict(CudaPolicy()),
                   reference_sha256=REFERENCE)
    data, host = fixtures.write_fixture(journal.root/'input.tsv', 'mixed_support', 256)
    require(data.input_sha256 == INPUT, 'Canonical mixed256 input differs')
    journal.bind(journal.root/'input.tsv')
    encoder = capture.Capture(journal, source)
    kernels = Kernels(device, compiled=True)
    model = common.upload(host, device, kernels)
    eager = common.upload(host, device, Kernels(device, compiled=False))
    graph = archived['pilot_and_graph']
    pilot = torch.tensor(graph['pilot_phi'], device=device, dtype=torch.float64)
    weights = torch.tensor(graph['weights'], device=device, dtype=torch.float64)
    require(common.tensor_sha(pilot) == graph['pilot_sha256'] and
            common.tensor_sha(weights) == graph['weights_sha256'], 'Archived literal graph/pilot hashes differ')
    curvature = model.terms(pilot)[2].clamp_min(1.)
    normalized = curvature / curvature.max()
    pooled = (normalized * pilot).sum() / normalized.sum()
    x = pooled.expand_as(pilot).clamp(model.lower, model.upper)
    policy = CudaPolicy()
    states = dict(pilot=pilot, weights=weights, initial_pooled=x,
                  pooled_value=pooled, original_pilot_curvature=curvature,
                  model={key: getattr(model, key) for key in ('alt', 'ref', 'slope', 'log_prior', 'lower', 'upper')})
    terms_by_backend = {}
    for label, current in [('compiled', model), ('eager', eager)]:
        terms = current.terms(x)
        losses, grad, curv, post, left, right = terms
        selected = torch.where(x == current.lower, right, grad)
        selected = torch.where(x == current.upper, left, selected)
        mass = x[:, None] * current.slope
        p = mass.clamp(current.eps, 1-current.eps)
        safe = torch.where(current.slope > 0, current.slope, torch.ones_like(current.slope))
        high = torch.full_like(safe, 1-current.eps)/safe
        low = torch.full_like(safe, current.eps)/safe
        at_high, at_low = (x[:, None] == high), (x[:, None] == low)
        inside = (mass > current.eps) & (mass < 1-current.eps) & ~at_high & ~at_low
        moving_left = torch.where(inside | at_high, current.slope, 0.)
        moving_right = torch.where(inside | at_low, current.slope, 0.)
        raw_second = current.alt[:, None]/p.square() + current.ref[:, None]/(1-p).square()
        left_curvature = (post*moving_left.square()*raw_second).sum(-1)
        right_curvature = (post*moving_right.square()*raw_second).sum(-1)
        rows = []
        ratio = torch.maximum(left_curvature, right_curvature)/curv.clamp_min(1.)
        for i in torch.argsort(ratio, descending=True)[:8].tolist():
            rows.append(dict(node=i, mutation_id=model.mutation_ids[i], x=float(x[i]),
                             upper=float(model.upper[i]), represented_gradient=float(grad[i]),
                             selected_gradient=float(selected[i]), curvature=float(curv[i]),
                             left_curvature=float(left_curvature[i]), right_curvature=float(right_curvature[i]),
                             ratio=float(ratio[i])))
        states[label] = dict(losses=losses, represented_gradient=grad, selected_gradient=selected,
                             curvature=curv, posterior=post, left=left, right=right,
                             left_curvature=left_curvature, right_curvature=right_curvature,
                             missing_left_at_box=(at_high & ~((mass > current.eps) & (mass < 1-current.eps)) &
                                                  (x == current.upper)[:, None]).to(x.dtype),
                             missing_right_at_box=(at_low & ~((mass > current.eps) & (mass < 1-current.eps)) &
                                                   (x == current.lower)[:, None]).to(x.dtype))
        receipt[label] = dict(worst_curvature_ratios=rows)
        terms_by_backend[label] = terms
    journal.artifact('literal-pooled-state.json', encoder._encode(states))
    receipt['pooled_sha256'] = common.tensor_sha(x)
    receipt['pooled_value'] = float(pooled)
    receipt['path7'] = {}
    record = archived['path_records'][7]
    require(record['lambda_value'] == 13852.533445257666 and
            record['starts'][2]['objective_tail'] == [] and record['starts'][2]['backtracks'] == 24,
            'Archived pooled-start identity changed')
    caps = weights * record['lambda_value']
    q = torch.zeros_like(caps)
    original_objective = solver.objective(model, x, caps)
    receipt['path7'].update(original_objective=float(original_objective),
                           archived_objective=record['starts'][2]['objective'],
                           objective_exact_match=float(original_objective) == record['starts'][2]['objective'])
    raw = audit.audit_raw(model, x, q, caps, None, policy)
    receipt['path7']['original_raw_audit'] = dict(qualified=raw.qualified, status=raw.status,
                                                residual=raw.residual, diagnostics=raw.diagnostics)
    all_cuts = {}
    for sign, label in [(1, 'positive'), (-1, 'negative')]:
        problem = cut_problem(model, x, q, caps, sign, terms_by_backend['compiled'], policy)
        all_cuts[label] = dict(problem=problem)
        for name, function in [('original', audit.directional_cut), ('rowwise', rowwise_cut)]:
            result = run_cut(problem, policy, function)
            all_cuts[label][name] = result
            receipt['path7'][label+'_'+name] = {key: value for key,value in result.items() if key != 'direction'}
        journal.record('cut_diagnosed', sign=label, results=receipt['path7'][label+'_original'],
                       rowwise=receipt['path7'][label+'_rowwise'])
    journal.artifact('literal-directional-cuts.json', encoder._encode(all_cuts))
    require(all_cuts['positive']['original']['diagnostics'] == raw.diagnostics['positive'],
            'Reconstructed original positive cut differs from exact raw audit')
    if args.first_qp:
        losses, grad, curv, _, left, right = terms_by_backend['compiled']
        grad = torch.where(x == model.lower, right, grad)
        grad = torch.where(x == model.upper, left, grad)
        h, target = curv.clamp_min(1.), x-grad/curv.clamp_min(1.)
        caps1 = weights*archived['path_records'][1]['lambda_value']
        fit = qp.solve_qp(h, target, model.lower, model.upper, caps1, kernels, policy, start=x, dual=None)
        trial, current = fit.x, solver.objective(model, x, caps1)
        d = trial-x
        trial_loss = model.loss(trial).sum()
        major = losses.sum()+(grad*d+.5*h*d.square()).sum()
        penalty = .5*(caps1*differences(trial).abs()).sum()
        trial_value = trial_loss+penalty
        delta = (grad*d+.5*h*d.square()).sum()+penalty-.5*(caps1*differences(x).abs()).sum()
        margin = 64*torch.finfo(x.dtype).eps*(1+current.abs()+major.abs())
        receipt['first_qp'] = dict(qualified=fit.qualified, iterations=fit.iterations,
            gap=float(fit.gap), kkt=float(fit.kkt), majorization_excess=float(trial_loss-major),
            surrogate_delta=float(delta), objective_change=float(trial_value-current), margin=float(margin))
        journal.artifact('literal-first-surrogate.json', encoder._encode(dict(
            h=h, target=target, start=x, dual=None, caps=caps1, trial=trial, q=fit.dual,
            losses=losses, grad=grad, curvature=curv, major=major, trial_loss=trial_loss,
            trial_value=trial_value, current=current, margin=margin, surrogate_delta=delta)))
    receipt['scientific_fit_qualified'] = False
    receipt['diagnostic_scope'] = ('Frozen-source pooled-state arithmetic, original audit/cut reconstruction, '
                                   'and separate rowwise-cut experiment; negative cut is evaluated diagnostically '
                                   'after the unresolved positive cut, never skipped in a production certificate')
    require(receipt['path7']['objective_exact_match'],
            'Fresh compiled pooled objective does not exactly match archived J; arrays retained, no identity promotion')
    require(source_provenance()['source_sha256'] == SOURCE, 'Source changed during diagnosis')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--expected-source-sha256', required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--reference-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--first-qp', action='store_true')
    parser.add_argument('--timeout-seconds', type=int, default=900)
    args = parser.parse_args()
    require(args.timeout_seconds > 0, 'Positive bounded timeout required')
    journal = mixed.Journal(args.out)
    receipt = dict(schema=SCHEMA, status='running', script_sha256=common.sha(__file__),
                   started_utc=common.utc_now(), source=source_provenance(),
                   lsf_job_id=os.environ.get('LSB_JOBID'), scientific_fit_qualified=False)
    def interrupted(signum, frame):
        raise TimeoutError('Bounded pooled diagnostic interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGALRM, interrupted)
    signal.alarm(args.timeout_seconds)
    code = 0
    try:
        run(args, receipt, journal)
        require(common.sha(__file__) == receipt['script_sha256'], 'Diagnostic helper changed')
        receipt['status'] = 'passed'
    except BaseException as error:
        receipt.update(status='failed', error_type=type(error).__name__, error=str(error),
                       traceback=traceback.format_exc())
        journal.record('diagnostic_failed', error=str(error))
        code = 1
    finally:
        signal.alarm(0)
        receipt.update(finished_utc=common.utc_now(), artifacts=journal.artifacts,
                       events_file=journal.path.name, events_sha256=common.sha(journal.path))
        common.write_json(journal.receipt_path, receipt)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
