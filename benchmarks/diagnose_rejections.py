"""Trace unchanged frozen outer solves at their acceptance decision only.

All six original starts run with the original 150-iteration budget. The tracer
records the final 20 iterations without changing arrays, return values, policy,
or accepted/rejected proposals. Timings include diagnostic instrumentation.
"""
import argparse
from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path
import sys
from time import perf_counter


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    import numpy as np
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def stable_loss_delta(model, before, after):
    """Diagnostic guarded log-ratio delta; not used by production acceptance."""
    import numpy as np
    dtype = np.longdouble
    p = np.clip(model.slope.astype(dtype) * before.astype(dtype)[:, None],
                dtype(model.eps), 1 - dtype(model.eps))
    y = np.clip(model.slope.astype(dtype) * after.astype(dtype)[:, None],
                dtype(model.eps), 1 - dtype(model.eps))
    joint = model.alt[:, None] * np.log(p) + model.ref[:, None] * np.log1p(-p) + model.log_prior
    posterior = np.exp(joint - np.max(joint, axis=1)[:, None])
    posterior /= np.sum(posterior, axis=1)[:, None]
    displacement = y - p
    relative = model.alt[:, None] * np.log1p(displacement / p)
    relative += model.ref[:, None] * np.log1p(-displacement / (1 - p))
    return -np.log1p(np.sum(posterior * np.expm1(relative), axis=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen-package', type=Path, required=True)
    parser.add_argument('--fixture-directory', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--cpu', type=int, default=1)
    args = parser.parse_args()
    from benchmark_chain import configure_execution
    controls = configure_execution(cpus=[args.cpu], threads=1)
    import numpy as np
    sys.path.insert(0, str(args.frozen_package.resolve().parent))
    from clipp1d import api, solver
    from clipp1d.io import read_tumor
    from clipp1d.model import compile_model, evaluate
    from clipp1d.policy import Policy
    from clipp1d.types import FrozenChain
    source = api.source_provenance()
    if (Path(solver.__file__).resolve().parent != args.frozen_package.resolve() or
            source['source_sha256'] != '30a019eba2c99dbfae23b26200164893ab6ac54d82d2ef60a1dec2ce4f8e4055'):
        raise RuntimeError('Expected frozen 72ba3ed package')
    setup = json.loads((args.fixture_directory / 'setup.json').read_text())
    assert sha(setup['input_path']) == setup['input_sha256']
    policy = Policy()
    model = compile_model(read_tumor(setup['input_path'], policy), policy)
    args.outdir.mkdir(parents=True, exist_ok=False)
    lines, first_line = inspect.getsourcelines(solver._solve_outer)
    gate_line = first_line + next(i for i, line in enumerate(lines) if 'if (cache["loss"] <= surrogate' in line)
    records = []
    for fixture in sorted(args.fixture_directory.glob('unresolved-*.npz')):
        tag = fixture.stem.removeprefix('unresolved-')
        receipt = json.loads((args.fixture_directory / ('start-' + tag + '.json')).read_text())
        assert sha(fixture) == receipt['fixture_sha256']
        with np.load(fixture) as data:
            ordered = model.subset(data['chain_order'])
            chain = FrozenChain(data['chain_order'], np.argsort(data['chain_order']), data['chain_weights'], 0., 'offline-fixture')
            events, seen = [], set()
            worst = [None, None]

            def trace_outer(frame, event, arg):
                if event != 'line' or frame.f_lineno != gate_line:
                    return trace_outer
                f = frame.f_locals
                if f['outer'] < 130:
                    return trace_outer
                identity = (f['outer'], f['attempt'], f['profile_calls'])
                if identity in seen:
                    return trace_outer
                seen.add(identity)
                x, trial, terms, h, step = (f[k] for k in ('x', 'trial', 'terms', 'h', 'step'))
                new_losses = evaluate(ordered, trial).loss
                linear = terms.gradient * step
                quadratic = .5 * h * step ** 2
                excess = new_losses - terms.loss - linear - quadratic
                guarded_delta = stable_loss_delta(ordered, x, trial)
                guarded_excess = guarded_delta - linear.astype(np.longdouble) - quadratic.astype(np.longdouble)
                positive = np.maximum(excess, 0)
                guarded_positive = np.maximum(guarded_excess, 0)
                top = np.argsort(-positive, kind='stable')[:10]
                slopes = np.where(ordered.valid, ordered.slope, np.nan)
                crossings = np.zeros(len(ordered), dtype=bool)
                for threshold in (ordered.eps, 1 - ordered.eps):
                    point = threshold / slopes
                    crossings |= np.any(((x[:, None] - point) * (trial[:, None] - point) < 0) |
                                         ((x[:, None] == point) != (trial[:, None] == point)), axis=1)
                gates = dict(majorization=f['cache']['loss'] <= f['surrogate'] + f['margin'],
                    observed_objective=f['cache']['objective'] <= f['current'] + f['margin'],
                    surrogate_descent=f['surrogate'] + f['trial_tv'] <= f['current'] + f['margin'])
                row = dict(outer=f['outer'] + 1, attempt=f['attempt'], curvature_scale=f['trial_scale'],
                    selected_witness=f['selected'], accepted=all(gates.values()), gates=gates,
                    margin=f['margin'], objective_before=f['current'],
                    objective_delta=f['cache']['objective'] - f['current'],
                    majorization_excess=f['cache']['loss'] - f['surrogate'],
                    surrogate_descent_excess=f['surrogate'] + f['trial_tv'] - f['current'],
                    sum_coordinate_excess=float(np.sum(excess)), sum_guarded_coordinate_excess=float(np.sum(guarded_excess)),
                    positive_count=int(np.count_nonzero(excess > 0)), positive_sum=float(np.sum(positive)),
                    guarded_positive_count=int(np.count_nonzero(guarded_excess > 0)),
                    guarded_positive_sum=float(np.sum(guarded_positive)),
                    max_positive=float(np.max(positive)), max_guarded_positive=float(np.max(guarded_positive)),
                    top10_positive_fraction=float(np.sum(positive[top]) / np.sum(positive)) if np.any(positive) else 0.,
                    clipping_crossing_count=int(np.count_nonzero(crossings)),
                    positive_crossing_count=int(np.count_nonzero(crossings & (excess > 0))),
                    max_step=float(np.max(np.abs(step))), max_curvature=float(np.max(h)),
                    coordinate_loss_change_sum=float(np.sum(new_losses - terms.loss)),
                    guarded_loss_change_sum=float(np.sum(guarded_delta)),
                    summed_loss_subtraction_error=float(np.sum(new_losses) - np.sum(terms.loss) - np.sum(guarded_delta)),
                    top_coordinates=[dict(index=int(i), mutation_id=ordered.mutation_ids[i], x=float(x[i]),
                        trial=float(trial[i]), curvature=float(h[i]), gradient=float(terms.gradient[i]),
                        excess=float(excess[i]), guarded_excess=float(guarded_excess[i]),
                        crosses_breakpoint=bool(crossings[i])) for i in top])
                events.append(row)
                if not row['accepted'] and (worst[0] is None or row['max_positive'] > worst[0]['max_positive']):
                    worst[:] = [row, {k: v.copy() for k, v in dict(x=x, trial=trial, h=h, gradient=terms.gradient,
                        old_losses=terms.loss, new_losses=new_losses, coordinate_excess=excess,
                        guarded_coordinate_excess=guarded_excess, caps=f['caps']).items()}]
                return trace_outer

            def trace(frame, event, arg):
                return trace_outer if event == 'call' and frame.f_code is solver._solve_outer.__code__ else None

            started = perf_counter()
            sys.settrace(trace)
            try:
                raw = solver.solve_profiled(ordered, chain, receipt['lambda_value'], data['initial'], policy)
            finally:
                sys.settrace(None)
            elapsed = perf_counter() - started
            result = dict(case=tag, fixture_sha256=sha(fixture), lambda_value=receipt['lambda_value'],
                unchanged_final_x=np.array_equal(raw.x, data['final']), unchanged_final_dual=np.array_equal(raw.dual, data['final_dual']),
                final_x_max_difference=float(np.max(np.abs(raw.x - data['final']))),
                unchanged_status=raw.diagnostics['status'] == receipt['diagnostics']['status'],
                diagnostics=raw.diagnostics, qualification=raw.qualified, wall_seconds_instrumented=elapsed,
                events=events, worst_rejected_event=worst[0])
            if worst[1] is not None:
                np.savez_compressed(args.outdir / (tag + '-worst-rejected.npz'), **worst[1])
            with (args.outdir / (tag + '.json')).open('x') as stream:
                json.dump(clean(result), stream, indent=2, allow_nan=False)
            records.append({k: v for k, v in result.items() if k not in ('events', 'diagnostics', 'worst_rejected_event')})
            print(json.dumps(clean(dict(**records[-1], tail_proposals=len(events),
                rejected=sum(not r['accepted'] for r in events), worst_rejected_event=worst[0]))), flush=True)
    report = dict(schema='clipp1d.outer_rejection_diagnosis.v1', source=source, controls=controls,
        original_fixture_source=setup['source'], input_sha256=setup['input_sha256'], policy=asdict(policy),
        harness_sha256=sha(__file__), records=records,
        scope='Unchanged six-start replay with original 150-iteration budget. Last20 outer iterations traced immediately before actual acceptance decision. Guarded log-ratio arithmetic is diagnostic only and never changes a proposal or gate.',
        source_unchanged=api.source_provenance()['source_sha256'] == source['source_sha256'])
    with (args.outdir / 'rejections.json').open('x') as stream:
        json.dump(clean(report), stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
