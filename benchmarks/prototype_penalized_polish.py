"""Offline bounded penalized-block proposal, followed by original full audits.

This changes the numerical trajectory and is explicitly not a production fix.
Each proposal changes one block from the sweep-start exact partition, includes its two external
TV edges, preserves original boxes and an occupied CCF1, and must decrease the
actual objective. The partition is recomputed every sweep and never frozen.
A merge can therefore be split by a later proposal within the same sweep.
A bounded scalar search supplies proposals, not a global-optimality certificate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen-package', type=Path, required=True)
    parser.add_argument('--fixture-directory', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--sweeps', type=int, default=3)
    args = parser.parse_args()
    from benchmark_chain import configure_execution
    controls = configure_execution(cpus=[1], threads=1)
    import numpy as np
    from scipy.optimize import minimize_scalar
    from diagnose_observed_dual import observed_dual
    from diagnose_rejections import clean
    sys.path.insert(0, str(args.frozen_package.resolve().parent))
    from clipp1d import api, solver
    from clipp1d.io import read_tumor
    from clipp1d.model import compile_model, clipping_breakpoints, evaluate
    from clipp1d.policy import Policy
    from clipp1d.types import FrozenChain
    policy = Policy()
    source = api.source_provenance()
    assert source['source_sha256'] == '30a019eba2c99dbfae23b26200164893ab6ac54d82d2ef60a1dec2ce4f8e4055'
    setup = json.loads((args.fixture_directory / 'setup.json').read_text())
    assert sha(setup['input_path']) == setup['input_sha256']
    model = compile_model(read_tumor(setup['input_path']))
    args.outdir.mkdir(parents=True, exist_ok=False)
    records = []
    for file in sorted(args.fixture_directory.glob('unresolved-*.npz')):
        tag = file.stem.removeprefix('unresolved-')
        receipt = json.loads((args.fixture_directory / ('start-' + tag + '.json')).read_text())
        assert sha(file) == receipt['fixture_sha256']
        started = perf_counter()
        with np.load(file) as saved:
            m = model.subset(saved['chain_order'])
            chain = FrozenChain(saved['chain_order'], np.argsort(saved['chain_order']), saved['chain_weights'], 0., 'offline-fixture')
            caps = receipt['lambda_value'] * chain.weights
            x = saved['final'].copy()
            initial = x.copy()
            q = saved['final_dual'].copy()
            before = solver.objective(m, x, caps)
            sweeps = []
            for sweep in range(args.sweeps):
                cuts = np.r_[0, np.flatnonzero(np.diff(x) != 0) + 1, len(x)]
                proposals = []
                for start, stop in zip(cuts[:-1], cuts[1:]):
                    lo, hi = float(np.max(m.lower[start:stop])), float(np.min(m.upper[start:stop]))
                    outside_clonal = np.any(x[:start] == 1) or np.any(x[stop:] == 1)
                    if np.any(x[start:stop] == 1) and not outside_clonal:
                        lo = hi = 1.
                    if lo == hi:
                        continue
                    block = m.subset(np.arange(start, stop))
                    old_losses = evaluate(block, x[start:stop]).loss
                    current = solver.objective(m, x, caps)
                    local_margin = 128 * np.finfo(float).eps * (1 + float(np.sum(abs(old_losses))))
                    count = [0]

                    def delta(value):
                        count[0] += 1
                        return solver.local_interval_delta(block, x, caps, start, stop, value, old_losses)

                    # Split at actual likelihood clipping and external TV kinks.
                    points = [lo, hi, float(x[start])]
                    points.extend(float(v) for v in clipping_breakpoints(block) if lo <= v <= hi)
                    if start:
                        points.append(float(np.clip(x[start - 1], lo, hi)))
                    if stop < len(x):
                        points.append(float(np.clip(x[stop], lo, hi)))
                    knots = np.unique(points)
                    candidates = [(delta(float(v)), float(v)) for v in knots]
                    for left, right in zip(knots[:-1], knots[1:]):
                        if right > left:
                            candidate = minimize_scalar(delta, bounds=(left, right), method='bounded',
                                options={'xatol': 1e-14, 'maxiter': 100})
                            candidates.append((float(candidate.fun), float(candidate.x)))
                    best_delta, center = min(candidates)
                    trial = x.copy()
                    trial[start:stop] = center
                    value = solver.objective(m, trial, caps)
                    accepted = (best_delta < -local_margin and value < current and
                                np.all(trial >= m.lower) and np.all(trial <= m.upper) and np.any(trial == 1))
                    if accepted:
                        previous = float(x[start])
                        x = trial
                        proposals.append(dict(start=int(start), stop=int(stop), previous=previous, proposed=center,
                            local_objective_delta=best_delta, full_objective_delta=value - current,
                            acceptance_margin=local_margin, scalar_evaluations=count[0]))
                witness = int(np.flatnonzero(x == 1)[0])
                lower, upper = m.lower.copy(), m.upper.copy()
                lower[witness] = upper[witness] = 1.
                context = solver.prepare_audit(m, x)
                retained, feasible = solver.stationarity(m, x, q, lower, upper, caps, context=context)
                q, dual_receipt = observed_dual(context, x, lower, upper, caps, q, retained)
                residual, feasible = solver.stationarity(m, x, q, lower, upper, caps, context=context)
                audits = []
                for anchor in dict.fromkeys((witness, int(np.flatnonzero(x == 1)[-1]))):
                    al, au = m.lower.copy(), m.upper.copy()
                    al[anchor] = au[anchor] = 1.
                    ok, restart = solver._kink_check(m, x, caps, al, au, policy,
                        force_intervals=anchor != witness, context=context)
                    audits.append(dict(anchor=anchor, passed=bool(ok), restart=restart is not None))
                passed = feasible and residual <= policy.stationarity_tol and all(r['passed'] and not r['restart'] for r in audits)
                sweeps.append(dict(sweep=sweep + 1, accepted_proposals=proposals,
                    objective=solver.objective(m, x, caps), observed_dual_stationarity=residual,
                    minimax_dual=dual_receipt, audits=audits, unchanged_first_order_audits_passed=passed))
                if passed or not proposals:
                    break
            # This is a new, explicitly budgeted raw solve. It is never counted as
            # completion within the failed original 150 iterations.
            fresh_started = perf_counter()
            fresh = solver.solve_profiled(m, chain, receipt['lambda_value'], x, policy)
            fresh_seconds = perf_counter() - fresh_started
            result = dict(case=tag, fixture_sha256=sha(file), lambda_value=receipt['lambda_value'],
                original_objective=before, polished_objective=solver.objective(m, x, caps),
                polish_objective_decrease=before - solver.objective(m, x, caps),
                max_polish_displacement=float(np.max(abs(x - initial))), sweep_records=sweeps,
                fresh_full_solver_budget=policy.outer_max_iterations, fresh_full_solver_qualified=fresh.qualified,
                fresh_full_solver_status=fresh.diagnostics['status'], fresh_full_solver_seconds=fresh_seconds,
                fresh_full_solver_diagnostics=fresh.diagnostics, fresh_full_solver_objective=fresh.objective,
                wall_seconds_diagnostic=perf_counter() - started)
            np.savez_compressed(args.outdir / (tag + '.npz'), original=initial, polished=x,
                                polished_dual=q, fresh_final=fresh.x, fresh_final_dual=fresh.dual)
            with (args.outdir / (tag + '.json')).open('x') as stream:
                json.dump(clean(result), stream, indent=2)
            records.append({k: v for k, v in result.items() if k not in ('sweep_records', 'fresh_full_solver_diagnostics')})
            print(json.dumps(clean(dict(**records[-1], sweeps=len(sweeps),
                polished_stationarity=sweeps[-1]['observed_dual_stationarity'],
                polished_audits_passed=sweeps[-1]['unchanged_first_order_audits_passed']))), flush=True)
    with (args.outdir / 'prototype.json').open('x') as stream:
        json.dump(clean(dict(schema='clipp1d.penalized_block_polish_prototype.v1', source=source,
            input_sha256=setup['input_sha256'], controls=controls, wrapper_sha256=sha(__file__), records=records,
            proposal_budget=dict(max_sweeps=args.sweeps, max_scalar_iterations_per_segment=100,
                fresh_full_solver_outer_iterations=policy.outer_max_iterations),
            scope='One offline numerical-trajectory prototype only; no production policy changes. Original failed attempts remain failed. Penalized local block proposals, unchanged audits, then a fresh separately budgeted full raw solve. Scalar proposals are not asserted globally optimal.')), stream, indent=2)


if __name__ == '__main__':
    main()
